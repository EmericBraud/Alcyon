#pragma once
#include "interface/gui.hpp"

class UCI
{
    Board b;
    EngineManager e;

    std::expected<int, Move::MoveError> parse_position(Board &board, std::istringstream &is)
    {
        std::string token;
        is >> token;

        if (token == "startpos")
        {
            board.load_fen(FEN_INIT_POS);
        }
        else if (token == "fen")
        {
            std::string fen;
            // Lire les 6 parties du FEN
            for (int i = 0; i < 6; ++i)
            {
                is >> token;
                fen += token + " ";
            }
            board.load_fen(fen);
        }

        // Lire les coups optionnels
        if (is >> token && token == "moves")
        {
            while (is >> token)
            {
                auto expected_move = Board::parse_move_uci(token, board);
                if (expected_move.has_value())
                {
                    board.play(expected_move.value());
                }
                else
                {
                    std::println(std::cerr, "info string Error parsing move {}: error code {}",
                                 token, (int)expected_move.error());
                    return std::unexpected(expected_move.error());
                }
            }
        }
        return 0;
    }

    void parse_go(Board &board, EngineManager &engine, std::istringstream &is)
    {
        std::string token;
        int wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1, depth = -1;

        while (is >> token)
        {
            if (token == "wtime")
                is >> wtime;
            else if (token == "btime")
                is >> btime;
            else if (token == "winc")
                is >> winc;
            else if (token == "binc")
                is >> binc;
            else if (token == "movetime")
                is >> movetime;
            else if (token == "depth")
                is >> depth;
        }

        int time_to_think = 5000; // Valeur par défaut (5s)

        if (movetime != -1)
        {
            time_to_think = movetime - 50; // On retire 50ms de marge de sécurité
        }
        else if (wtime != -1)
        {
            // Logique de gestion du temps (Time Management)
            int my_time = (board.get_side_to_move() == WHITE) ? wtime : btime;
            int my_inc = (board.get_side_to_move() == WHITE) ? winc : binc;

            // Formule simple : (Temps restant / 40) + Incrément
            // On divise par 40 car une partie dure en moyenne 40 coups
            time_to_think = (my_time / 40) + my_inc - 50;
        }

        // Sécurité : ne jamais descendre en dessous de 20ms
        if (time_to_think < 20)
            time_to_think = 20;

        // Lancement de la recherche (dans ton EngineManager)
        engine.start_search(time_to_think);
    }

public:
    UCI() : b(), e(b)
    {
        b.load_fen(FEN_INIT_POS);
    }
    void loop()
    {
        std::string line, token;

        while (std::getline(std::cin, line))
        {
            std::istringstream is(line);
            token.clear();
            is >> token;

            if (token == "uci")
            {
                std::println("id name Chess26");
                std::println("id author Emeric");
                std::println("option name Hash type spin default 64 min 1 max 2048");
                std::println("option name Move Overhead type spin default 100 min 0 max 1000");
                std::println("option name Ponder type check default false");
                std::println("option name Threads type spin default 16 min 1 max 16");
                std::println("uciok");
            }
            else if (token == "isready")
            {
                std::cout << "readyok" << std::endl;
            }
            else if (token == "ucinewgame")
            {
                b.load_fen(FEN_INIT_POS);
                e.clear();
            }
            else if (token == "position")
            {
                parse_position(b, is);
            }
            else if (token == "go")
            {
                parse_go(b, e, is);
            }
            else if (token == "quit")
            {
                break;
            }
            else if (token == "gui")
            {
                GUI g{b};
                g.run();
            }
        }
    }
};