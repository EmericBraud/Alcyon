#pragma once
#include "move_generator.hpp"

#define MAX_DEPTH 25
constexpr int INF = 1000000;

struct MoveScorer
{
    Move m;
    int score;
};

using Clock = std::chrono::steady_clock;

const int pawnPhase = 0;
const int knightPhase = 1;
const int bishopPhase = 1;
const int rookPhase = 2;
const int queenPhase = 4;
const int totalPhase = 24; // 16 pions(0) + 4 cavaliers(4) + 4 fous(4) + 4 tours(8) + 2 dames(8)

constexpr int mg_pawn_table[64] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0};

constexpr int mg_knight_table[64] = {
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50};

constexpr int mg_bishop_table[64] = {
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20};

constexpr int mg_rook_table[64] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0};

constexpr int mg_queen_table[64] = {
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20};

constexpr int mg_king_table[64] = {
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 5, 0, 0, 5, 30, 20};

constexpr int eg_king_table[64] = {
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50};

class Computer
{
    Board &board;
    TranspositionTable tt;
    std::array<Move, MAX_DEPTH> move_stack;
    // [Color] [From sq] [To sq]
    int history_moves[2][64][64];
    std::array<int, N_PIECES_TYPE_HALF> pieces_score = {
        100,   // Pawn
        300,   // Knight
        300,   // Bishop
        500,   // Tower
        900,   // Queen
        10000, // King
    };
    long long total_nodes = 0;
    long long tt_cuts = 0;
    long long beta_cutoffs = 0;
    long long q_nodes = 0;

    Move killer_moves[MAX_DEPTH][2];
    Clock::time_point start_time;
    int time_limit_ms = 0;
    bool time_up = false;

    inline int eval() const
    {
        int mg[2] = {0, 0}; // Score Midgame pour [White, Black]
        int eg[2] = {0, 0}; // Score Endgame pour [White, Black]
        int gamePhase = 0;

        for (int color = WHITE; color <= BLACK; ++color)
        {
            for (int piece = PAWN; piece <= KING; ++piece)
            {
                bitboard bb = board.get_piece_bitboard((Color)color, piece);

                while (bb)
                {
                    int sq = std::countr_zero(bb);
                    int mirror_sq = (color == WHITE) ? sq : sq ^ 56;

                    // 1. Matériel et PST de base
                    mg[color] += pieces_score[piece] + mg_pawn_table[mirror_sq]; // Utilise tes tables actuelles
                    eg[color] += pieces_score[piece] + (piece == KING ? eg_king_table[mirror_sq] : mg_pawn_table[mirror_sq]);

                    // 2. Calcul de la phase
                    if (piece != PAWN && piece != KING)
                    {
                        gamePhase += (piece == KNIGHT) ? knightPhase : (piece == BISHOP) ? bishopPhase
                                                                   : (piece == ROOK)     ? rookPhase
                                                                                         : queenPhase;
                    }

                    // 3. Sécurité du Roi (Simplifiée)
                    if (piece == KING)
                    {
                        mg[color] += evaluate_king_safety((Color)color, sq);
                    }

                    bb &= (bb - 1);
                }
            }

            // 4. Structure de pions
            mg[color] += evaluate_pawn_structure((Color)color);
        }

        // Calcul final avec interpolation (Tapered Eval)
        int mgScore = mg[WHITE] - mg[BLACK];
        int egScore = eg[WHITE] - eg[BLACK];

        // On s'assure que gamePhase ne dépasse pas totalPhase
        int phase = std::min(gamePhase, totalPhase);

        return (mgScore * phase + egScore * (totalPhase - phase)) / totalPhase;
    }
    int evaluate_king_safety(Color color, int king_sq) const
    {
        int penalty = 0;
        bitboard pawns = board.get_piece_bitboard(color, PAWN);

        // On définit une zone devant le roi (ex: 3 cases)
        bitboard shield_area = (color == WHITE) ? (1ULL << (king_sq + 7) | 1ULL << (king_sq + 8) | 1ULL << (king_sq + 9)) : (1ULL << (king_sq - 7) | 1ULL << (king_sq - 8) | 1ULL << (king_sq - 9));

        int pawn_count = std::popcount(pawns & shield_area);
        if (pawn_count == 0)
            penalty -= 50; // Roi tout nu !

        return penalty;
    }
    int evaluate_pawn_structure(Color color) const
    {
        int score = 0;
        bitboard pawns = board.get_piece_bitboard(color, PAWN);

        for (int i = 0; i < 8; ++i)
        {
            bitboard file_mask = 0x0101010101010101ULL << i;
            int count = std::popcount(pawns & file_mask);
            if (count > 1)
                score -= 20; // Pénalité pion doublé
        }
        return score;
    }
    int eval_relative(Color side_to_move) const
    {
        int score = eval();
        return (side_to_move == WHITE) ? score : -score;
    }

    int qsearch(int alpha, int beta)
    {
        check_time();
        if (time_up)
            return eval_relative(board.get_side_to_move());
        q_nodes++;
        total_nodes++; // On compte aussi ces noeuds

        // 1. Standing Pat (On suppose qu'on ne fait rien)
        // Si on ne capture pas, quel est le score ?
        // Cela sert de "borne inférieure" (Lower Bound).

        int tt_score;
        Move tt_move;
        bool tt_hit = tt.probe(board.get_hash(), 0, 0, alpha, beta, tt_score, tt_move);
        if (tt_hit)
        {
            return tt_score;
        }

        int alpha_orig = alpha;

        // Coupure Beta "Lazy" : Si la position actuelle est déjà trop forte, on coupe.
        int stand_pat = eval_relative(board.get_side_to_move());
        if (stand_pat >= beta)
            return beta;
        if (stand_pat > alpha)
            alpha = stand_pat;

        const int BIG_DELTA = 975;
        if (stand_pat < alpha - BIG_DELTA)
            return alpha;

        const Color player = board.get_side_to_move();
        std::vector<Move> v = MoveGen::generate_pseudo_legal_captures(board, player);

        std::vector<MoveScorer> scored_moves;
        scored_moves.reserve(v.size());

        for (const Move &m : v)
        {
            int score = score_capture(m);
            // Bonus si c'est le coup suggéré par la TT
            if (tt_move.get_value() != 0 && m.get_value() == tt_move.get_value())
                score += 2000000;
            scored_moves.push_back({m, score});
        }
        std::sort(scored_moves.begin(), scored_moves.end(),
                  [](const MoveScorer &a, const MoveScorer &b)
                  {
                      return a.score > b.score;
                  });
        int best_score = stand_pat;
        Move best_move;
        for (MoveScorer &ele : scored_moves)
        {
            check_time();
            if (time_up)
                break;
            auto &m = ele.m;
            int victim_val = 0;
            if (m.get_flags() == Move::EN_PASSANT_CAP)
            {
                victim_val = pieces_score[PAWN];
            }
            else
            {
                int victim = board.get_piece_on_square(m.get_to_sq()).second;
                if (victim != NO_PIECE)
                    victim_val = pieces_score[victim];
                m.set_to_piece(static_cast<Piece>(victim));
            }
            int attacker_val = pieces_score[m.get_from_piece()];
            if (attacker_val == 0)
                attacker_val = pieces_score[board.get_piece_on_square(m.get_from_sq()).second];

            if (attacker_val > victim_val)
            {
                if (stand_pat + victim_val + 200 < alpha)
                    continue;
            }

            // Delta Pruning (Optimisation optionnelle mais recommandée)
            // Si ma capture ne suffit pas à remonter le score (ex: P x D alors que je perds une Dame), on ignore.
            // (Laissons ça de côté pour l'instant pour la stabilité)

            board.play(m);

            if (MoveGen::is_king_attacked(board, player))
            {
                board.unplay(m);
                continue;
            }

            // Appel récursif à qsearch (pas negamax !)
            int score = -qsearch(-beta, -alpha);

            board.unplay(m);

            if (score >= beta)
            {
                tt.store(board.get_hash(), 0, 0, beta, TT_BETA, m);
                return beta;
            }

            if (score > alpha)
            {
                alpha = score;
                best_score = score;
                best_move = m;
            }
        }

        TTFlag flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
        tt.store(board.get_hash(), 0, 0, best_score, flag, best_move);

        return alpha;
    }
    inline void check_time()
    {
        if (time_up)
            return;

        auto now = Clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

        if (elapsed >= time_limit_ms)
        {
            time_up = true;
        }
    }

public:
    int negamax(int depth, int alpha, int beta, int ply)
    {
        if (ply > 0 && board.is_repetition())
        {
            return 0;
        }
        int tt_score;
        Move tt_move;
        total_nodes++;
        bool tt_hit = tt.probe(board.get_hash(), depth, ply, alpha, beta, tt_score, tt_move);

        if (tt_hit && ply > 0)
        {
            tt_cuts++;
            return tt_score;
        }
        check_time();
        if (time_up)
            return eval_relative(board.get_side_to_move());
        if (depth <= 0)
        {
            return qsearch(alpha, beta);
        }
        if (depth >= 3 && ply > 0 && !MoveGen::is_king_attacked(board, board.get_side_to_move()))
        {
            int stored_ep;
            board.play_null_move(stored_ep);
            int R = 2;
            int score = -negamax(depth - 1 - R, -beta, -beta + 1, ply + 1);

            board.unplay_null_move(stored_ep);

            if (score >= beta)
            {
                if (score >= MATE_SCORE - MAX_DEPTH)
                    return beta;
                return score;
            }
        }

        const Color player = board.get_side_to_move();
        std::vector<Move> v = MoveGen::generate_legal_moves(board);

        std::vector<MoveScorer> scored_moves;
        scored_moves.reserve(v.size());

        for (const Move &m : v)
        {
            scored_moves.push_back({m, score_move(m, board, tt_move, ply)});
        }

        std::sort(scored_moves.begin(), scored_moves.end(),
                  [](const MoveScorer &a, const MoveScorer &b)
                  {
                      return a.score > b.score;
                  });

        int legal_moves_count = 0;
        int best_score = -INF;
        Move best_move_this_node;
        int alpha_orig = alpha;

        int moves_searched = 0;
        for (auto &ele : scored_moves)
        {
            check_time();
            if (time_up)
                break;
            Move &m = ele.m;
            board.play(m);

            legal_moves_count++;
            moves_searched++;

            int score;
            bool needs_full_search = true;
            bool is_tactical = (ele.score >= 900000);
            // LMR (Late Move Reduction)
            // Conditions :
            // 1. Profondeur suffisante (> 2)
            // 2. On a déjà testé les bons coups (moves_searched > 4)
            // 3. Ce n'est pas un coup tactique (pas capture, pas promotion)
            // 4. On n'est pas en échec (trop dangereux de réduire)
            // --- LMR (LATE MOVE REDUCTION) ---
            if (depth >= 3 && moves_searched > 4 &&
                ply > 0 &&
                !is_tactical &&
                !MoveGen::is_king_attacked(board, player))
            {
                int reduction = 1;
                if (depth > 6)
                    reduction = 2;
                score = -negamax(depth - 1 - reduction, -beta, -alpha, ply + 1);
                if (score <= alpha)
                {
                    // Move is bad even tho we searched a bit
                    // It is really a bad move
                    needs_full_search = false;
                }
            }
            // --- RECHERCHE COMPLÈTE (FULL DEPTH) ---
            // On la fait si :
            // A) On n'a pas fait de LMR (premiers coups, captures, échecs...)
            // B) Ou si le LMR a échoué (le coup semblait trop beau pour être vrai)
            if (needs_full_search)
            {

                score = -negamax(depth - 1, -beta, -alpha, ply + 1);
            }

            board.unplay(m);

            if (score >= beta)
            {
                beta_cutoffs++;
                tt.store(board.get_hash(), depth, ply, score, TT_BETA, m);
                if (m.get_to_piece() == NO_PIECE && m.get_flags() != Move::EN_PASSANT_CAP)
                {
                    int bonus = depth * depth;
                    history_moves[board.get_side_to_move()][m.get_from_sq()][m.get_to_sq()] += bonus;
                    if (!(m.get_value() == killer_moves[ply][0].get_value()))
                    {
                        killer_moves[ply][1] = killer_moves[ply][0];
                        killer_moves[ply][0] = m;
                    }
                }
                return score;
            }

            if (score > best_score)
            {
                best_score = score;
                best_move_this_node = m;
                if (score > alpha)
                {
                    alpha = score;
                }
            }
        }

        if (legal_moves_count == 0)
        {
            if (MoveGen::is_king_attacked(board, player)) // Checkmate
            {
                return -MATE_SCORE + ply;
            }
            else
            {
                return 0;
            }
        }
        TTFlag flag;
        if (best_score <= alpha_orig)
            flag = TT_ALPHA;
        else
            flag = TT_EXACT;

        tt.store(board.get_hash(), depth, ply, best_score, flag, best_move_this_node);

        return best_score;
    }

    void play(int time_ms = 5000)
    {
        time_limit_ms = time_ms;
        time_up = false;
        start_time = Clock::now();

        Move best_move;
        total_nodes = 0;
        tt_cuts = 0;
        beta_cutoffs = 0;
        clear_killers();
        age_history();

        for (int current_depth = 1; current_depth <= MAX_DEPTH; ++current_depth)
        {
            if (time_up)
                break;

            int alpha = -INF;
            int beta = INF;

            int score = negamax(current_depth, alpha, beta, 0);

            if (!time_up)
            {
                // On garde le dernier résultat COMPLET
                best_move = tt.get_move(board.get_hash());

                std::cout << "Depth " << current_depth
                          << " | Score " << score
                          << " | Nodes " << total_nodes
                          << std::endl;
            }
        }

        board.play(best_move);
    }

    int score_move(const Move &move, const Board &board, const Move &tt_move, int ply)
    {
        if (tt_move.get_value() != 0 && move.get_value() == tt_move.get_value())
        {
            return 2000000;
        }

        const Piece from_piece{move.get_from_piece()};

        const Piece to_piece{move.get_to_piece()};
        const uint32_t flags{move.get_flags()};

        if (flags == Move::Flags::PROMOTION_MASK)
        {
            return 1000000 + (pieces_score[QUEEN] * 10);
        }

        if (to_piece != NO_PIECE)
        {
            int victim_val = pieces_score[to_piece];
            int attacker_val = pieces_score[from_piece];
            return 1000000 + (victim_val * 10 - attacker_val);
        }
        if (flags == Move::Flags::EN_PASSANT_CAP)
        {
            int victim_val = pieces_score[PAWN];
            int attacker_val = pieces_score[from_piece];
            return 1000000 + (victim_val * 10 - attacker_val);
        }
        if (move.get_value() == killer_moves[ply][0].get_value())
            return 900000;
        if (move.get_value() == killer_moves[ply][1].get_value())
            return 800000;

        if (to_piece == NO_PIECE)
        {
            return history_moves[board.get_side_to_move()][move.get_from_sq()][move.get_to_sq()];
        }

        return 0;
    }
    // Score uniquement pour le Quiescence Search (MVV-LVA)
    int score_capture(const Move &move) const
    {
        int attacker = move.get_from_piece();

        // 2. Identifier la victime (Qui est mangé ?)
        int victim_val = 0;

        if (move.get_flags() == Move::EN_PASSANT_CAP)
        {
            victim_val = pieces_score[PAWN];
        }
        else
        {
            // On regarde sur la case d'arrivée
            const int victim = move.get_to_piece();
            if (victim != NO_PIECE)
            {
                victim_val = pieces_score[victim];
            }
        }

        // 3. Calcul du score
        int attacker_val = pieces_score[attacker];

        // Formule MVV-LVA standard
        int score = 1000000 + (victim_val * 10 - attacker_val);

        // Bonus Promotion
        if (move.get_flags() == Move::PROMOTION_MASK)
        {
            score += pieces_score[QUEEN];
        }

        return score;
    }
    void age_history()
    {
        for (int c = 0; c < 2; ++c)
            for (int f = 0; f < 64; ++f)
                for (int t = 0; t < 64; ++t)
                    history_moves[c][f][t] /= 8;
    }

public:
    Computer(Board &board) : board(board)
    {
        init_zobrist();
        tt.resize(256);
        clear_killers();
        clear_history();
    }

    int eval_position()
    {
        return negamax(MAX_DEPTH, -INF, INF, 0);
    }
    void clear_killers()
    {
        std::memset(killer_moves, 0, sizeof(killer_moves));
    }
    void clear_history()
    {
        std::memset(history_moves, 0, sizeof(history_moves));
    }
};