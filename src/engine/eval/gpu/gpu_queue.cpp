#include "gpu_queue.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>

#include "core/move/generator/move_generator.hpp"
#include "engine/config/eval.hpp"
#include "engine/eval/pos_eval.hpp"
#include "engine/eval/see.hpp"
#include "engine/eval/virtual_board.hpp"
#include "gpu_backend.hpp"
#include "gpu_encoder.hpp"
#include "gpu_tt.hpp"

#ifdef __APPLE__
#include <pthread.h>
#endif

namespace gpu_eval {

namespace {

// This thread spends most of its time inside the backend's synchronous
// inference dispatch. Profiled over a 10s search, per phase:
//   ANE backend:   infer 7257ms (73%), quietify 2399ms (24%),
//                  load_fen 210ms, nnue-stats 44ms, encode 34ms, rest 25ms
//   Metal backend: infer 7296ms (87%), quietify 1013ms (12%), rest ~160ms
// That dispatch is CPU-side work, so demoting this thread starves the
// accelerator rather than protecting the search: at QOS_CLASS_UTILITY --
// which parks it on the E-cores -- one MPSGraph inference call took 394ms
// against 42ms at QOS_CLASS_USER_INITIATED, a 9.4x throughput difference
// with 10 search threads running. So it asks for the SAME priority band
// as the search threads, not a lower one.
//
// Kept overridable (CHESS26_GPU_QOS=utility|default|user_initiated|
// user_interactive) because this is a scheduling trade-off against the
// search threads on a 4 P-core + 6 E-core machine, and the right answer
// may differ on other hardware.
// ponytail: env var rather than a UCI option -- it is a diagnostic
// knob, promote it if it turns out to need per-match tuning.
void set_gpu_thread_qos() {
#ifdef __APPLE__
    const char *q = std::getenv("CHESS26_GPU_QOS");
    const std::string name = q ? q : "";
    const qos_class_t qos = name == "utility"          ? QOS_CLASS_UTILITY
                            : name == "default"        ? QOS_CLASS_DEFAULT
                            : name == "user_interactive" ? QOS_CLASS_USER_INTERACTIVE
                                                         : QOS_CLASS_USER_INITIATED;
    pthread_set_qos_class_self_np(qos, 0);
#endif
}

// Safety cap on quietify()'s ply loop -- bounds worst-case cost on
// pathological positions (shouldn't realistically be reached; real
// exchange sequences settle in a handful of plies).
constexpr int kQuietifyMaxPlies = 8;

// Self-contained quiescence search used ONLY to settle a "loud" position
// (pending captures/checks) into a quiet one before encoding it for the
// CNN -- deliberately NOT SearchWorker::qsearch (qsearch.cpp), which is
// tightly coupled to the main search's shared transposition table and
// thread-local move-ordering heuristics, and WRITES into the main TT --
// none of which belongs on this thread. This is plain MVV-LVA + SEE
// pruning, no TT, no history tables: simpler and slower per node than
// the real qsearch, which is fine here -- the GPU-prep thread is mostly
// idle (measured submission rate is a small fraction of GPU capacity,
// see gpubench), so it has cycles to spare, unlike the main search
// threads where qsearch's speed is critical.
template <Color Us>
int quietify_qsearch(VBoard &board, int alpha, int beta, int ply, Move &out_best_move) {
    out_best_move = Move(0);
    const bool in_check = board.is_king_attacked<Us>();
    int stand_pat = -engine_constants::eval::Inf;

    if (!in_check) {
        stand_pat = Eval::eval_relative<Us>(board, alpha, beta);
        if (stand_pat >= beta) {
            return beta;
        }
        if (stand_pat > alpha) {
            alpha = stand_pat;
        }
    }

    MoveList list;
    if (in_check) {
        MoveGen::generate_pseudo_legal_moves<Us>(board, list);
    } else {
        MoveGen::generate_pseudo_legal_captures<Us>(board, list);
    }

    for (int i = 0; i < list.count; ++i) {
        const Move &m = list[i];
        list.scores[i] = in_check
            ? 0
            : engine_constants::eval::MvvLvaTable[m.get_flags() == Move::EN_PASSANT_CAP ? PAWN : m.get_to_piece()][m.get_from_piece()];
    }

    int best_score = in_check ? -engine_constants::eval::Inf : stand_pat;
    int moves_searched = 0;

    for (int i = 0; i < list.count; ++i) {
        Move &m = list.pick_best_move(i);

        if (!in_check) {
            const Piece target = (m.get_flags() == Move::EN_PASSANT_CAP) ? PAWN : m.get_to_piece();
            const int victim_val = Eval::get_piece_score(target);
            const int attacker = m.get_from_piece();
            if (Eval::get_piece_score(attacker) > victim_val &&
                compute_see<Us>(board, m.get_to_sq(), static_cast<Piece>(target), static_cast<Piece>(attacker), m.get_from_sq()) < 0) {
                continue;
            }
        }

        if (!board.template is_move_legal<Us>(m)) {
            continue;
        }

        board.template play<Us>(m);
        ++moves_searched;
        Move unused;
        const int score = -quietify_qsearch<!Us>(board, -beta, -alpha, ply + 1, unused);
        board.template unplay<Us>(m);

        if (score >= beta) {
            return beta;
        }
        if (score > best_score) {
            best_score = score;
            if (score > alpha) {
                alpha = score;
                out_best_move = m;
            }
        }
    }

    if (moves_searched == 0 && in_check) {
        return -engine_constants::eval::MateScore + ply;
    }
    return best_score;
}

// Repeatedly plays the best move quietify_qsearch() finds from the
// current position until it reports none (position settled) or
// kQuietifyMaxPlies is reached. Plies played are recorded into
// out_moves (fixed capacity kQuietifyMaxPlies) so the caller can unwind
// them afterwards -- board ends up at the settled position.
void quietify(VBoard &board, Move *out_moves, int &out_num_moves) {
    out_num_moves = 0;
    for (int ply = 0; ply < kQuietifyMaxPlies; ++ply) {
        Move best_move;
        if (board.get_side_to_move() == WHITE) {
            quietify_qsearch<WHITE>(board, -engine_constants::eval::Inf, engine_constants::eval::Inf, 0, best_move);
        } else {
            quietify_qsearch<BLACK>(board, -engine_constants::eval::Inf, engine_constants::eval::Inf, 0, best_move);
        }
        if (best_move.get_value() == 0) {
            break;
        }
        board.play(best_move);
        out_moves[out_num_moves++] = best_move;
    }
}

int eval_relative_dispatch(const VBoard &board, int alpha, int beta) {
    return board.get_side_to_move() == WHITE
        ? Eval::eval_relative<WHITE>(board, alpha, beta)
        : Eval::eval_relative<BLACK>(board, alpha, beta);
}

// NOTE: a bounded minimax (resolve_disagreement()) used to run HERE, on
// this thread, whenever NNUE and the CNN disagreed -- storing its verdict
// instead of the CNN's score. Removed after measurement: it consumed
// ~90% of this thread's wall clock (~1900 nodes per call, on >50% of all
// candidates), capping GPU throughput at ~60 positions/s against a device
// that does 4300/s -- AND, for every position it touched, it replaced the
// CNN score with a TT-less, history-less 3-ply search, i.e. a strictly
// worse version of what the main search already does. The disagreement is
// still MEASURED (see GpuTT::record_cnn_vs_nnue) but no longer acted on.
// See docs/gpu-async-eval/consultative-eval-measurements.md.

} // namespace

bool GpuQueue::push(const GpuTask &task) {
    if (lock_.test_and_set(std::memory_order_acquire)) {
        return false; // contended -- drop rather than block a search worker
    }
    bool pushed = false;
    if (count_ < kQueueCapacity) {
        tasks_[head_] = task;
        head_ = (head_ + 1) & kMask;
        ++count_;
        pushed = true;
    }
    lock_.clear(std::memory_order_release);
    return pushed;
}

namespace {
std::optional<GpuTask> pop_one(GpuTask *tasks, std::atomic_flag &lock, std::size_t &tail, std::size_t &count) {
    if (lock.test_and_set(std::memory_order_acquire)) {
        return std::nullopt;
    }
    std::optional<GpuTask> out;
    if (count > 0) {
        out = tasks[tail];
        tail = (tail + 1) & (kQueueCapacity - 1);
        --count;
    }
    lock.clear(std::memory_order_release);
    return out;
}
} // namespace

bool GpuQueue::start(const std::string &weights_path) {
    if (running_.load(std::memory_order_relaxed)) {
        return true; // already running
    }
    // Join a previous attempt's thread first -- a joinable std::thread
    // left dangling at process exit calls std::terminate() -- fatal in
    // this -fno-exceptions build, not a graceful fallback.
    if (thread_.joinable()) {
        thread_.join();
    }

    // Load synchronously, on the CALLER's thread (the UCI command
    // thread), so this function can report success/failure before
    // returning -- avoids a race where a caller checks GpuBackend::
    // is_ready() right after start() returns, before a background
    // thread would have had a chance to even begin loading.
    if (!GpuBackend::instance().load_weights(weights_path)) {
        std::fprintf(stderr, "gpu_eval: failed to load weights from %s, GPU eval disabled\n", weights_path.c_str());
        return false;
    }

    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread(&GpuQueue::run, this);
    return true;
}

void GpuQueue::stop() {
    running_.store(false, std::memory_order_relaxed);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void GpuQueue::run() {
    set_gpu_thread_qos();

    // VBoard (not plain Board) -- quietify() below needs real static eval
    // (Eval::eval_relative) for its stand-pat check, which requires an
    // up-to-date eval_state/NNUE accumulator as moves are played.
    // MUST be `static`, not a plain stack local: in NNUE builds, VBoard
    // embeds NnueEval's weight tables INLINE (std::array members, not
    // behind a pointer -- see nnue_eval.hpp), making sizeof(VBoard) far
    // larger than a std::thread's default stack size on macOS (512KB,
    // unlike the main thread's 8MB) -- a plain stack-local VBoard here
    // blew the GPU thread's stack immediately on construction (SIGBUS).
    static VBoard scratch;
    Move quiet_moves[kQuietifyMaxPlies];
    int quiet_num_moves = 0;

    // Fixed-size, allocated once for this thread's whole lifetime -- no
    // runtime allocation in the drain/encode/infer loop below. Sized for
    // kMaxDrainTasksPerBatch tasks' worth of candidates (see
    // gpu_config.hpp for the batching rationale).
    static float planes_batch[kMaxBatchPositions][kNumPlanesV3][kPlaneSize];
    static int piece_counts[kMaxBatchPositions];
    static std::uint64_t child_keys[kMaxBatchPositions];
    static std::uint8_t child_depths[kMaxBatchPositions];
    static std::int32_t scores[kMaxBatchPositions];
    // NNUE eval of the settled position, captured at encode time (while
    // `scratch` is still live there) -- compared against the CNN's score
    // after inference purely to MEASURE how far the two models are apart
    // (see GpuTT::record_cnn_vs_nnue); nothing is acted on, so it is only
    // computed under measure_diagnostics(). Profiled at 57ms of this
    // thread's ~9.5s -- small, but it is self-measurement, not work the
    // engine needs.
    static int nnue_cps[kMaxBatchPositions];
    const bool measure = measure_diagnostics();
    // +1 / -1: the CNN scores the SETTLED position, but the score is
    // stored under the UNSETTLED candidate's key (that's the position the
    // search actually probes -- see the store loop below). An odd number
    // of quietify plies flips the side to move between the two, so the
    // relative score has to be negated to stay relative to the key's own
    // side to move.
    static int score_signs[kMaxBatchPositions];

    // Phase 0 (measure_diagnostics() seulement) : de quoi comparer, apres
    // inference, la preference du CNN parmi les candidats d'une tache au coup
    // que la recherche a elle-meme retenu. On garde la tache et la cle de
    // chaque candidat ; apres la boucle de store, TOUS les candidats ont leur
    // score dans le gpu_tt -- ceux qui viennent d'etre calcules parce qu'on
    // les y a mis, ceux qui avaient ete dedupliques parce que c'est
    // precisement pourquoi ils l'ont ete. Une sonde suffit donc, pas besoin
    // de remonter d'un slot de batch vers sa tache.
    static GpuTask drained[kMaxDrainTasksPerBatch];
    static std::uint64_t cand_key[kMaxDrainTasksPerBatch][kNumCandidateMoves];
    static bool cand_ok[kMaxDrainTasksPerBatch][kNumCandidateMoves];
    // Eval NNUE de la position calmee de chaque candidat, ramenee au repere
    // de la cle enfant comme le score CNN -- le controle methodologique
    // decrit dans GpuTT::store_ordering_hint. Calculee pour TOUS les
    // candidats, y compris ceux que la deduplication ecarte de l'inference.
    static int cand_nnue[kMaxDrainTasksPerBatch][kNumCandidateMoves];

    while (running_.load(std::memory_order_relaxed)) {
        // Wall-clock accounting for GpuTT::gpu_thread_busy_percent() --
        // covers this whole iteration (drain + quietify + encode +
        // infer_batch + agree-or-resolve + store), recorded as "busy"
        // below, or as "idle" if the queue was empty and this iteration
        // was just the sleep_for() wait.
        const auto iter_start = std::chrono::steady_clock::now();
        int batch_size = 0;

        // Drain up to kMaxDrainTasksPerBatch queued tasks (or until the
        // queue is empty) into the shared buffers above before firing a
        // single infer_batch() call, instead of one call per task.
        int tasks_drained = 0;
        while (tasks_drained < kMaxDrainTasksPerBatch && batch_size + kNumCandidateMoves <= kMaxBatchPositions) {
            std::optional<GpuTask> task = pop_one(tasks_, lock_, tail_, count_);
            if (!task) {
                break; // queue empty for now
            }
            const int task_index = tasks_drained;
            ++tasks_drained;
            if (measure) {
                drained[task_index] = *task;
                for (int i = 0; i < kNumCandidateMoves; ++i) {
                    cand_ok[task_index][i] = false;
                }
            }

            if (!scratch.load_fen(task->position.to_fen())) {
                continue; // shouldn't happen -- defensively skip a malformed snapshot
            }

            for (int i = 0; i < task->num_candidates; ++i) {
                const Move move = task->candidate_moves[i];
                if (!scratch.is_move_pseudo_legal(move) || !scratch.is_move_legal(move)) {
                    continue; // stale by the time the GPU thread gets to it -- skip
                }
                scratch.play(move);

                // THE key this score gets stored under: the candidate
                // position itself, captured BEFORE quietify() moves the
                // board on. That's the position the main search will
                // actually probe (transp_table.hpp's probe(), negamax's
                // should_qsearch branch -- both look up the node's own
                // hash). Keying on the post-quietify descendant instead,
                // as this used to, meant nothing was ever stored for the
                // position anyone asked about: measured at 18 useful hits
                // across a 13M-node search, i.e. the whole subsystem was
                // invisible to the search. See docs/gpu-async-eval/
                // consultative-eval-measurements.md.
                const std::uint64_t child_key = scratch.get_hash();
                if (measure) {
                    cand_key[task_index][i] = child_key;
                    cand_ok[task_index][i] = true;
                }

                // Settle the position before encoding it -- a static CNN
                // eval on a "loud" position (mid-exchange, in check) is
                // exactly the horizon-effect problem qsearch exists to
                // avoid in the main search. What gets stored is therefore
                // a precomputed qsearch-style value for child_key, with
                // the CNN standing in for the leaf eval.
                quietify(scratch, quiet_moves, quiet_num_moves);
                if (measure) {
                    const int sign = (quiet_num_moves % 2 == 0) ? 1 : -1;
                    cand_nnue[task_index][i] =
                        sign * eval_relative_dispatch(scratch, -engine_constants::eval::Inf, engine_constants::eval::Inf);
                }

                // Skip re-encoding + re-inferring a candidate whose score
                // is already fresh in the GPU TT (same key, current
                // generation) -- measured (via GpuTT::redundant_stores())
                // at ~60-70% of all candidates in a typical search, since
                // neighboring PV-leaf submissions often share the same
                // downstream child positions. Checking here (before
                // encode_planes_v3, the actual compute-heavy step) avoids
                // that wasted work instead of just detecting it after the
                // fact in store(). Checked against child_key, since
                // that's what actually gets stored below.
                std::int16_t existing_score;
                std::uint8_t existing_depth, existing_age;
                bool skip = shared_gpu_tt().probe(child_key, existing_score, existing_depth, existing_age) &&
                            existing_age == shared_gpu_tt().current_age();

                // Also dedupe against candidates already added to THIS
                // batch (from this or an earlier drained task) -- these
                // can't be caught by the probe() above since nothing gets
                // store()'d until after the whole batch's infer_batch()
                // call returns, below.
                for (int j = 0; !skip && j < batch_size; ++j) {
                    if (child_keys[j] == child_key) {
                        skip = true;
                    }
                }

                if (!skip) {
                    const GpuPosition child = GpuPosition::from_board(scratch);
                    encode_planes_v3(child, planes_batch[batch_size]);
                    // Debug aid: set CHESS26_GPU_DEBUG_DUMP_PLANES=/path/to/file to
                    // dump the last-encoded position's raw plane tensor (+ FEN in
                    // a sibling .fen file) for cross-checking against
                    // training/cnn/eval_compare's Python encoder on the exact same
                    // FEN -- e.g. how the plane-encoder/Metal-inference numerical
                    // parity was verified (byte-identical planes, matching score)
                    // during development. No-op (one getenv call) when unset.
                    if (const char *dump_path = std::getenv("CHESS26_GPU_DEBUG_DUMP_PLANES")) {
                        std::ofstream out(dump_path, std::ios::binary);
                        out.write(reinterpret_cast<const char *>(planes_batch[batch_size]), sizeof(planes_batch[batch_size]));
                        std::ofstream fen_out(std::string(dump_path) + ".fen");
                        fen_out << child.to_fen() << " piece_count=" << non_king_piece_count(child);
                    }
                    piece_counts[batch_size] = non_king_piece_count(child);
                    child_keys[batch_size] = child_key;
                    child_depths[batch_size] = task->depth;
                    score_signs[batch_size] = (quiet_num_moves % 2 == 0) ? 1 : -1;
                    // Captured now, while `scratch` is still live at the
                    // settled position -- see nnue_cps' doc above.
                    if (measure) {
                        nnue_cps[batch_size] = eval_relative_dispatch(scratch, -engine_constants::eval::Inf, engine_constants::eval::Inf);
                    }
                    ++batch_size;
                }

                for (int j = quiet_num_moves - 1; j >= 0; --j) {
                    scratch.unplay(quiet_moves[j]);
                }
                scratch.unplay(move);
            }
        }

        if (batch_size == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            const auto idle_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - iter_start).count();
            shared_gpu_tt().record_idle_ns(static_cast<std::uint64_t>(idle_ns));
            continue;
        }

        GpuBackend::instance().infer_batch(&planes_batch[0][0][0], piece_counts, batch_size, scores);

        // Store the CNN score as-is, re-signed for the key's side to move
        // (see score_signs). The NNUE comparison is measurement only now
        // -- see record_cnn_vs_nnue() and the removed resolve_disagreement
        // note at the top of this file.
        for (int i = 0; i < batch_size; ++i) {
            // Offset applied in the SETTLED position's frame (that's where
            // it was measured, and where nnue_cps[i] lives), before the
            // sign flip that moves the score into the stored key's frame.
            const int cnn_cp = scores[i] + kCnnToNnueOffsetCp;
            if (measure) {
                shared_gpu_tt().record_cnn_vs_nnue(cnn_cp, nnue_cps[i]);
            }
            shared_gpu_tt().store(child_keys[i], static_cast<std::int16_t>(score_signs[i] * cnn_cp), child_depths[i]);
        }

        // Phase 0 : on STOCKE le hint, on ne le juge pas ici.
        //
        // Un noeud qui a deja son coup TT classe ce coup premier avec un
        // bonus de 9600 : aucun hint n'y sert a rien, et le comparer au coup
        // TT reviendrait a mesurer la prediction d'une information deja
        // possedee. On ne retient donc que les noeuds soumis SANS coup TT, et
        // la comparaison se fait plus tard, quand la recherche aura conclu
        // sur ce noeud (voir negamax.cpp).
        //
        // La valeur stockee etant relative au trait de l'ENFANT, le parent
        // prefere le candidat au score stocke le PLUS BAS. Apres la boucle de
        // store, tous les candidats ont leur score dans le gpu_tt : ceux
        // qu'on vient de calculer parce qu'on les y a mis, les autres parce
        // que c'est precisement pourquoi ils avaient ete dedupliques.
        if (measure) {
            for (int t = 0; t < tasks_drained; ++t) {
                const GpuTask &task = drained[t];
                if (task.num_candidates < 2 || task.tt_move.get_value() != 0 ||
                    task.blind_best.get_value() == 0) {
                    continue;
                }
                Move cnn_best(0), nnue_best(0);
                int best_score = 0, best_nnue = 0;
                for (int i = 0; i < task.num_candidates; ++i) {
                    if (!cand_ok[t][i]) {
                        continue;
                    }
                    if (nnue_best.get_value() == 0 || cand_nnue[t][i] < best_nnue) {
                        best_nnue = cand_nnue[t][i];
                        nnue_best = task.candidate_moves[i];
                    }
                    std::int16_t sc;
                    std::uint8_t d, a;
                    if (!shared_gpu_tt().probe(cand_key[t][i], sc, d, a) || a != shared_gpu_tt().current_age()) {
                        continue;
                    }
                    if (cnn_best.get_value() == 0 || sc < best_score) {
                        best_score = sc;
                        cnn_best = task.candidate_moves[i];
                    }
                }
                if (cnn_best.get_value() == 0 || nnue_best.get_value() == 0) {
                    continue;
                }
                shared_gpu_tt().store_ordering_hint(task.position.zobrist_key,
                                                    cnn_best.get_from_sq(), cnn_best.get_to_sq(),
                                                    task.blind_best.get_from_sq(), task.blind_best.get_to_sq(),
                                                    nnue_best.get_from_sq(), nnue_best.get_to_sq());
            }
        }

        const auto busy_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - iter_start).count();
        shared_gpu_tt().record_busy_ns(static_cast<std::uint64_t>(busy_ns));
    }
}

GpuQueue &shared_gpu_queue() {
    static GpuQueue instance;
    return instance;
}

} // namespace gpu_eval
