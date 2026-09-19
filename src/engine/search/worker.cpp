#include "engine/search/worker.hpp"

#include <cstdio>
#include "common/logger.hpp"

#include "engine/config/config.hpp"
#include "engine/config/eval.hpp"
#include "engine/engine_manager.hpp"
#include "worker.hpp"

#include <algorithm>
#include <limits>
#include <cstdlib>

template <Color Us>
int SearchWorker::score_move(const Move &move, const Move &tt_move, int ply, const Move &prev_move) const
{
    const uint32_t move_val = move.get_value();
    if (move_val == tt_move.get_value())
        return 9600;

    const Piece from_piece = move.get_from_piece();
    const Piece to_piece = move.get_to_piece();
    const uint32_t flags = move.get_flags();

    // 1. Tactiques (Captures & Promotions)
    if (to_piece != NO_PIECE || flags == Move::Flags::EN_PASSANT_CAP || move.is_promotion())
    {
        const Piece target = (flags == Move::Flags::EN_PASSANT_CAP) ? PAWN : to_piece;
        const int mvv_lva = engine_constants::eval::MvvLvaTable[target][from_piece];

        if (move.is_promotion())
        {
            // Priorise la promotion Dame
            const int promo_piece = move.get_promo_piece();
            if (promo_piece == QUEEN)
                return 9300;
            if (promo_piece == KNIGHT)
                return 8050;
            if (promo_piece == ROOK)
                return 8000;
            if (promo_piece == BISHOP)
                return 8000;
        }

        // On n'appelle SEE que si c'est potentiellement perdant (LVA prend MVV)
        if (Eval::get_piece_score(from_piece) > Eval::get_piece_score(target))
        {
            if (see<Us>(move.get_to_sq(), target, from_piece, move.get_from_sq()) < 0)
                return -1000 + mvv_lva; // Clairement perdant
        }

        return 8600 + mvv_lva; // Captures normales/gagnantes
    }

    // 2. Coups calmes prioritaires
    if (move_val == killer_moves[ply][0].get_value())
        return engine_constants::search::move_ordering::KillerScore;
    if (move_val == killer_moves[ply][1].get_value())
        return engine_constants::search::move_ordering::KillerScore;

    // Counter-move
    if (ply > 0 && prev_move != 0 && move_val == counter_moves[Us][prev_move.get_from_piece()][prev_move.get_to_sq()].get_value())
        return engine_constants::search::move_ordering::CounterScore;

    // Bonus spécial pour les échecs "calmes" (Crucial pour mat en 11)
    // Attention : nécessite que ta MoveGen ou une fonction légère détecte l'échec
    // if (gives_check(move)) return 6000000;

    // 3. History Moves (Score relatif)
    return history_moves[Us][move.get_from_sq()][move.get_to_sq()];
}
std::string SearchWorker::get_pv_line(int depth)
{
    std::string pv_line = "";
    std::vector<Move> moves_to_unplay;
    std::vector<uint64_t> visited_hashes;

    for (int i = 0; i < std::min(depth, 10); i++)
    {
        if (board.is_repetition() || board.get_halfmove_clock() >= 100)
            break;

        Move m = shared_tt.get_move(board.get_hash());

        // 1. Si le coup est 0 (Nœud terminal, Mat, ou pas d'entrée), on arrête.
        if (m.get_value() == 0)
            break;

        // 2. Vérification de pseudo-légalité AVANT de toucher au board
        // Cela évite les asserts ou crashs dans is_move_legal si m est corrompu
        if (!board.is_move_pseudo_legal(m))
            break;

        if (!board.is_move_legal(m))
            break;

        // Détection de cycle
        uint64_t h = board.get_hash();
        bool cycle_detected = false;
        for (uint64_t v : visited_hashes)
            if (v == h)
            {
                cycle_detected = true;
                break;
            }
        if (cycle_detected)
            break;

        visited_hashes.push_back(h);

        pv_line += m.to_uci() + " ";
        board.play(m);
        moves_to_unplay.push_back(m);
    }

    for (int i = (int)moves_to_unplay.size() - 1; i >= 0; i--)
    {
        board.unplay(moves_to_unplay[i]);
    }
    return pv_line;
}

std::string SearchWorker::get_pv_line_with_root(Move root_move, int depth)
{
    if (root_move.get_value() == 0)
        return "";

    if (!board.is_move_pseudo_legal(root_move) || !board.is_move_legal(root_move))
        return "";

    std::string pv_line = root_move.to_uci();
    if (depth <= 1)
        return pv_line;

    pv_line += " ";
    board.play(root_move);
    auto guard = CHESS26_SCOPE_EXIT([&]()
                                    { board.unplay(root_move); });

    std::vector<Move> moves_to_unplay;
    std::vector<uint64_t> visited_hashes;

    for (int i = 0; i < std::min(depth - 1, 10); ++i)
    {
        if (board.is_repetition() || board.get_halfmove_clock() >= 100)
            break;

        Move m = shared_tt.get_move(board.get_hash());
        if (m.get_value() == 0)
            break;
        if (!board.is_move_pseudo_legal(m) || !board.is_move_legal(m))
            break;

        uint64_t h = board.get_hash();
        bool cycle_detected = false;
        for (uint64_t v : visited_hashes)
        {
            if (v == h)
            {
                cycle_detected = true;
                break;
            }
        }
        if (cycle_detected)
            break;

        visited_hashes.push_back(h);
        pv_line += m.to_uci();
        pv_line += " ";
        board.play(m);
        moves_to_unplay.push_back(m);
    }

    for (int i = static_cast<int>(moves_to_unplay.size()) - 1; i >= 0; --i)
        board.unplay(moves_to_unplay[i]);

    return pv_line;
}

int SearchWorker::negamax_with_aspiration(int depth, int last_score)
{
    max_extended_depth = 0;
    int delta = engine_constants::search::aspiration::SmallDelta;
    if (depth >= engine_constants::search::aspiration::HighDepth)
        delta = engine_constants::search::aspiration::HighDelta;
    else if (depth >= engine_constants::search::aspiration::MidDepth)
        delta = engine_constants::search::aspiration::MidDelta;
    int alpha = -engine_constants::eval::Inf;
    int beta = engine_constants::eval::Inf;

    if (depth >= engine_constants::search::aspiration::EnableDepth)
    {
        alpha = last_score - delta;
        beta = last_score + delta;
    }

    int iterations = 0;
    const int max_iterations = engine_constants::search::aspiration::MaxIterations;

    while (true)
    {
        ++iterations;
        int score = negamax(depth, alpha, beta, 0);

        if (abs(score) >= engine_constants::eval::MateScore - depth)
        {
            // Mat trouve a cette profondeur : il faut adopter le coup qui va
            // avec, sinon on annonce la PV de cette iteration et on joue le
            // coup de la precedente (fastchess : "Bestmove does not match
            // beginning of last PV"). Meme condition que le chemin de succes
            // ci-dessous : hors fenetre, out_move est le residu d'un fail-high
            // et ne vaut pas mieux que ce qu'on a deja.
            if (score > alpha && score < beta)
                best_root_move = out_move;
            return score;
        }
        if (abs(score) >= engine_constants::eval::MateScore - engine_constants::search::aspiration::MateWindowMargin)
        {
            alpha = -engine_constants::eval::MateScore;
            beta = engine_constants::eval::MateScore;
            continue;
        }

        // Recherche avortee en cours de route : `score` n'est pas une
        // evaluation, c'est le residu d'un parcours incomplet. Le dernier
        // score digne de confiance est celui de la profondeur precedente.
        if (shared_stop.load(std::memory_order_relaxed))
            return last_score;

        // Succes : score dans la fenetre. A valider AVANT de constater que le
        // temps est ecoule -- sinon une iteration complete etait jetee (et
        // best_root_move laisse sur l'iteration precedente) au seul motif que
        // l'horloge a expire juste apres l'avoir terminee.
        if (score > alpha && score < beta)
        {
            best_root_move = out_move;
            if (manager.should_stop())
                shared_stop.store(true, std::memory_order_relaxed);
            return score;
        }

        // Fail-low ou fail-high, et plus de temps pour la re-recherche qui
        // aurait resolu la fenetre : `score` est une borne (<= alpha ou
        // >= beta), pas une evaluation. La renvoyer faussait le centrage de la
        // fenetre suivante et le score rapporte en UCI, qui alimente la
        // gestion du temps.
        if (manager.should_stop())
        {
            shared_stop.store(true, std::memory_order_relaxed);
            return last_score;
        }

        // Ajustement delta intelligent
        if (score <= alpha)
        {
            // Fail-low
            delta = std::max(delta * 2, engine_constants::search::aspiration::WidenMinDelta);
            alpha = std::max(-engine_constants::eval::Inf, alpha - delta);
            if (thread_id == 0)
                logs::debug << "info string fail low" << std::endl;
        }
        else if (score >= beta)
        {
            // Fail-high
            delta = std::max(delta * 2, engine_constants::search::aspiration::WidenMinDelta);
            beta = std::min(engine_constants::eval::Inf, beta + delta);
            if (thread_id == 0)
                logs::debug << "info string fail high" << std::endl;
        }

        // Sécurité
        if (iterations >= max_iterations || delta > engine_constants::search::aspiration::WidenMaxDelta)
        {
            alpha = -engine_constants::eval::Inf;
            beta = engine_constants::eval::Inf;
        }
    }
}

void SearchWorker::iterative_deepening()
{
    int last_score = 0;
    Move prev_best_root = 0;
    int stable_iterations = 0; // iterations consecutives sans changement du coup racine
    for (int depth = 1; depth < engine_constants::search::MaxDepth; ++depth)
    {
        last_score = negamax_with_aspiration(depth, last_score);

        // "go depth N" : cette profondeur vient d'etre terminee, on s'arrete.
        // Teste ici, dans la MEME branche que l'arret par le temps, pour que
        // la derniere profondeur complete emette son info line comme
        // d'habitude -- et thread 0 leve shared_stop afin que les autres
        // workers se replient au lieu d'entamer la profondeur suivante.
        const int depth_limit = manager.max_depth_limit();
        const bool depth_reached = depth_limit > 0 && depth >= depth_limit;
        if (depth_reached && thread_id == 0)
            shared_stop.store(true, std::memory_order_relaxed);

        // Limite SOUPLE : on vient de terminer une profondeur, faut-il en
        // demarrer une autre ? Decide par thread 0 seulement, qui leve
        // shared_stop pour que tout le monde se replie. L'interet par
        // rapport a la seule limite dure : on rend un resultat COMPLET au
        // lieu d'avorter une iteration a mi-chemin et de jeter son travail.
        //
        // Bonus d'instabilite : si le meilleur coup racine vient de changer,
        // la position n'est pas tranchee et une iteration de plus vaut son
        // prix. S'il est stable, on garde le temps pour plus tard.
        if (thread_id == 0)
        {
            const int soft = manager.soft_limit_ms();
            const Move current_best = best_root_move.get_value() != 0 ? best_root_move : out_move;
            if (prev_best_root.get_value() != 0 && current_best.get_value() != prev_best_root.get_value())
                stable_iterations = 0;
            else if (prev_best_root.get_value() != 0)
                ++stable_iterations;
            prev_best_root = current_best;
            if (soft > 0)
            {
                // Plus le coup racine est stable depuis longtemps, moins une
                // iteration de plus a de chances de le changer -- mesure a
                // l'appui, voir engine_constants::search::time.
                namespace tc = engine_constants::search::time;
                static const double kStability[5] = {tc::StabilityFactor0, tc::StabilityFactor1,
                                                     tc::StabilityFactor2, tc::StabilityFactor3,
                                                     tc::StabilityFactor4};
                const double factor = kStability[std::min(stable_iterations, 4)];
                const double budget = soft * factor * (engine_constants::search::time::SoftStartPercent / 100.0);
                if (manager.elapsed_ms() >= static_cast<long long>(budget))
                    shared_stop.store(true, std::memory_order_relaxed);
            }
        }

        if (depth_reached || shared_stop.load(std::memory_order_relaxed))
        {
            if (thread_id == 0)
            {
                auto elapsed_ms = std::max<long long>(1,
                                                      std::chrono::duration_cast<std::chrono::milliseconds>(
                                                          std::chrono::steady_clock::now() - start_time_ref)
                                                          .count());

                // Vider notre propre reliquat avant de lire le total : sans
                // ca, "nodes" et "nps" sont toujours des multiples de 32768
                // (voir le flush par paquets dans check_stop()), ce qui rend
                // tout comptage de noeuds inexploitable -- y compris pour
                // comparer deux ordonnancements.
                global_nodes.fetch_add(local_nodes, std::memory_order_relaxed);
                local_nodes = 0;
                long long nodes = global_nodes.load(std::memory_order_relaxed);
                long long nps = nodes * 1000 / elapsed_ms;
                const Move pv_root = best_root_move.get_value() != 0 ? best_root_move : out_move;
                const std::string pv_line = get_pv_line_with_root(pv_root, depth);
                logs::uci
                    << "info depth " << depth
                    << " seldepth " << max_extended_depth
                    << " score cp " << last_score
                    << " nodes " << nodes
                    << " nps " << nps
                    << " hashfull " << shared_tt.get_hashfull();
                if (!pv_line.empty())
                    logs::uci << " pv " << pv_line;
                logs::uci
                    << std::endl;
            }
            return;
        }
        // Une ligne "info" par profondeur TERMINEE, comme tout moteur UCI :
        // c'est ce qui permet a un GUI d'afficher la progression, et a un
        // outil d'analyse de voir a quelle profondeur le meilleur coup
        // change. Elle existait mais etait doublement neutralisee -- sous
        // #ifndef NDEBUG *et* via logs::debug, alors que NDEBUG est force
        // pour chess_core (CMakeLists.txt) : aucun build reel ne l'emettait.
        if (thread_id == 0)
        {
            // Meme flush que dans la branche terminale, sinon "nodes" est un
            // multiple de 32768 (voir check_stop()).
            global_nodes.fetch_add(local_nodes, std::memory_order_relaxed);
            local_nodes = 0;
            auto elapsed_ms = std::max<long long>(1,
                                                  std::chrono::duration_cast<std::chrono::milliseconds>(
                                                      std::chrono::steady_clock::now() - start_time_ref)
                                                      .count());

            long long nodes = global_nodes.load(std::memory_order_relaxed);
            long long nps = nodes * 1000 / elapsed_ms;
            logs::uci
                << "info depth " << depth
                << " seldepth " << max_extended_depth
                << " score cp " << last_score
                << " nodes " << nodes
                << " nps " << nps
                << " hashfull " << shared_tt.get_hashfull()
                << " pv " << get_pv_line(depth)
                << std::endl;
        }
    }
    if (thread_id == 0)
    {
        shared_stop.store(true, std::memory_order_relaxed);
        auto elapsed_ms = std::max<long long>(1,
                                              std::chrono::duration_cast<std::chrono::milliseconds>(
                                                  std::chrono::steady_clock::now() - start_time_ref)
                                                  .count());

        long long nodes = global_nodes.load(std::memory_order_relaxed);
        long long nps = nodes * 1000 / elapsed_ms;
        const Move pv_root = best_root_move.get_value() != 0 ? best_root_move : out_move;
        const std::string pv_line = get_pv_line_with_root(pv_root, engine_constants::search::MaxDepth - 1);
        logs::uci
            << "info depth " << engine_constants::search::MaxDepth - 1
            << " seldepth " << max_extended_depth
            << " score cp " << last_score
            << " nodes " << nodes
            << " nps " << nps
            << " hashfull " << shared_tt.get_hashfull();
        if (!pv_line.empty())
            logs::uci << " pv " << pv_line;
        logs::uci
            << std::endl;
    }
}

bool SearchWorker::check_stop()
{
    if ((local_nodes & 32767) == 0)
    {
        global_nodes.fetch_add(local_nodes, std::memory_order_relaxed);
        local_nodes = 0;

        if (thread_id == 0 && manager.should_stop())
        {
            shared_stop.store(true, std::memory_order_relaxed);
        }
        if (shared_stop.load(std::memory_order_relaxed))
            return true;
    }
    ++local_nodes;
    return false;
}

template int SearchWorker::score_move<WHITE>(const Move &move, const Move &tt_move, int ply, const Move &prev_move) const;
template int SearchWorker::score_move<BLACK>(const Move &move, const Move &tt_move, int ply, const Move &prev_move) const;
