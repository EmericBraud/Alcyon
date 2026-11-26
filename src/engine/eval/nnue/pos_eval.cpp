#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unistd.h>

#include "engine/eval/pos_eval.hpp"

#ifdef NNUE_EVAL

namespace
{
    // Echantillonnage des positions evaluees pendant la recherche (jeu de donnees "arbres de
    // recherche", docs/experts-et-idees.md) : si ALCYON_DUMP_DIR est defini, une evaluation NNUE
    // complete sur ALCYON_DUMP_RATE (defaut 200) ecrit la FEN de la position dans un fichier par
    // processus et par thread. Sans la variable : une seule comparaison par evaluation.
    struct EvalDumper
    {
        std::FILE *f = nullptr;
        std::uint64_t rng = 0;
        std::uint64_t rate = 0;
        bool init = false;
        ~EvalDumper()
        {
            if (f)
                std::fclose(f);
        }
    };
    thread_local EvalDumper dumper;
    const char *const dump_dir = std::getenv("ALCYON_DUMP_DIR");

    std::string fen_of(const VBoard &b)
    {
        static const char pc[] = {'p', 'n', 'b', 'r', 'q', 'k'};
        std::string s;
        for (int rank = 7; rank >= 0; --rank)
        {
            int empty = 0;
            for (int file = 0; file < 8; ++file)
            {
                const int sq = rank * 8 + file;
                const Piece p = b.get_p(sq);
                if (p == NO_PIECE)
                {
                    ++empty;
                    continue;
                }
                if (empty)
                    s += char('0' + empty), empty = 0;
                const char c = pc[static_cast<int>(p) - static_cast<int>(PAWN)];
                s += b.get_c(sq) == WHITE ? char(c - 'a' + 'A') : c;
            }
            if (empty)
                s += char('0' + empty);
            if (rank)
                s += '/';
        }
        s += b.get_side_to_move() == WHITE ? " w " : " b ";
        const auto cr = b.state.castling_rights;
        std::string c;
        if (cr & WHITE_KINGSIDE) c += 'K';
        if (cr & WHITE_QUEENSIDE) c += 'Q';
        if (cr & BLACK_KINGSIDE) c += 'k';
        if (cr & BLACK_QUEENSIDE) c += 'q';
        s += c.empty() ? "-" : c;
        const int ep = b.state.en_passant_sq;
        if (ep == constants::EnPassantSqNone)
            s += " -";
        else
            s += std::string(" ") + char('a' + ep % 8) + char('1' + ep / 8);
        s += " 0 1\n";
        return s;
    }

    void maybe_dump(const VBoard &b)
    {
        EvalDumper &d = dumper;
        if (!d.init)
        {
            d.init = true;
            const char *r = std::getenv("ALCYON_DUMP_RATE");
            d.rate = r ? std::strtoull(r, nullptr, 10) : 200;
            const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
            d.rng = (static_cast<std::uint64_t>(getpid()) << 32) ^ tid ^ 0x9E3779B97F4A7C15ULL;
            const std::string path = std::string(dump_dir) + "/pos_" + std::to_string(getpid()) + "_" + std::to_string(tid % 100000) + ".fen";
            d.f = std::fopen(path.c_str(), "a");
        }
        d.rng ^= d.rng << 13;
        d.rng ^= d.rng >> 7;
        d.rng ^= d.rng << 17;
        if (d.f && d.rate && d.rng % d.rate == 0)
        {
            const std::string s = fen_of(b);
            std::fwrite(s.data(), 1, s.size(), d.f);
        }
    }
}

namespace Eval
{
    int eval(const VBoard &board, int alpha, int beta)
    {
        // Uses VBoard's incrementally-maintained accumulator (see
        // virtual_board.hpp's play/unplay hooks) -- no per-call recompute.
        // evaluate_abs() returns centipawns relative to the side to move (the
        // network is always evaluated as if "us" = whoever is on move). This
        // function's contract (matching the HCE eval and eval_relative<Us> in
        // pos_eval.hpp) is to return a WHITE-relative absolute score, so flip
        // the sign back for black to move.
        if (dump_dir) [[unlikely]]
            maybe_dump(board);
        const int score = board.get_nnue_eval().evaluate_abs(board) * engine_constants::search::EvalScale / 100;
        return board.get_side_to_move() == WHITE ? score : -score; // @TODO create a templated version of it OR delete this statement (sign is flipped twice)
    }
}

#endif