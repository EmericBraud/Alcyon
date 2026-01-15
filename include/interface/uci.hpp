#pragma once

#include <expected>
#include <iostream>

#include "core/utils/logger.hpp"
#include "core/board.hpp"
#include "core/move/move_generator.hpp"

#include "engine/zobrist.hpp"
#include "engine/book.hpp"
#include "engine/engine_manager.hpp"

#include "interface/gui.hpp"

class UCI
{
    Board b;
    EngineManager e;
    bool ponder_enabled = true;

    std::expected<int, Move::MoveError> parse_position(Board &board, std::istringstream &is)
    {
        std::string token;
        is >> token;

        if (token == "startpos")
        {
            board.load_fen(core::constants::FenInitPos);
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
                    std::cerr << std::flush;
                    return std::unexpected(expected_move.error());
                }
            }
        }
        return 0;
    }

    void parse_go(Board &board, EngineManager &engine, std::istringstream &is)
    {
        std::string token;
        bool is_ponder = false;
        bool is_infinite = false;
        int wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1, depth = -1;

        // Lecture des options UCI
        while (is >> token)
        {
            if (token == "ponder")
                is_ponder = true;
            else if (token == "wtime")
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
            else if (token == "infinite")
                is_infinite = true;
        }

        core::logs::debug << "info string DEBUG: Checking Book..." << std::endl;
        core::logs::debug << "info string DEBUG: My Hash is " << std::hex << board.polyglot_key() << std::dec << std::endl;
        if (!is_infinite && !is_ponder)
        {
            Move book_move = Book::probe(board);

            if (book_move.get_value() != 0)
            {
                core::logs::debug << "info string DEBUG: Book Move Found!" << std::endl;
                if (board.is_move_pseudo_legal(book_move) && board.is_move_legal(book_move))
                {
                    core::logs::uci << "bestmove " << book_move.to_uci() << std::endl;
                    return;
                }
            }
            else
            {
                core::logs::debug << "info string DEBUG: No move found in book." << std::endl;
            }
        }

        // Time Management
        int time_to_think = 5000;

        if (movetime != -1)
        {
            time_to_think = movetime - 50;
        }
        else if (wtime != -1)
        {
            int my_time = (board.get_side_to_move() == WHITE) ? wtime : btime;
            int my_inc = (board.get_side_to_move() == WHITE) ? winc : binc;

            time_to_think = (my_time / 30) + (my_inc / 2);
        }

        if (time_to_think < 20)
            time_to_think = 20;

        if (is_infinite)
        {
            engine.start_search(0, false, true);
            return;
        }
        engine.start_search(time_to_think, is_ponder && ponder_enabled, is_infinite, ponder_enabled);
    }

    void set_option(std::istringstream &is)
    {
        std::string word, name, value, option;

        while (is >> word)
        {
            if (word == "name")
            {
                name.clear();
                while (is >> word && word != "value")
                    name += word + " ";
            }
            if (word == "value")
            {
                value.clear();
                while (is >> word)
                    value += word + " ";
            }
        }

        if (name == "Ponder ")
            ponder_enabled = (value == "true ");
        else if (name == "Hash ")
        {
            int size = std::stoi(value);
            e.get_tt().resize(size);
            core::logs::debug << "info string Hash table resized" << std::endl;
        }
        else if (name == "Clear Hash ")
        {
            e.get_tt().clear();
            core::logs::debug << "info string Hash table cleared" << std::endl;
        }
    }

public:
    UCI() : b(), e(b)
    {
        MoveGen::initialize_bitboard_tables();
        init_zobrist();
        b.load_fen(core::constants::FenInitPos);
        Book::init(DATA_PATH "komodo.bin");
    }
    void loop()
    {
        std::string line, token;

        while (std::getline(std::cin, line))
        {
            core::logs::debug << "info string << " << line << std::endl;

            std::istringstream is(line);
            token.clear();
            is >> token;

            if (token == "uci")
            {
                core::logs::uci << "id name Chess26" << std::endl;
                core::logs::uci << "id author Emeric" << std::endl;
                core::logs::uci << "option name Hash type spin default 512 min 1 max 2048" << std::endl;
                core::logs::uci << "option name Move Overhead type spin default 100 min 0 max 1000" << std::endl;
                core::logs::uci << "option name Ponder type check default false" << std::endl;
                core::logs::uci << "option name Threads type spin default 16 min 1 max 16" << std::endl;
                core::logs::uci << "uciok" << std::endl;
            }
            else if (token == "setoption")
            {
                set_option(is);
            }
            else if (token == "ponderhit")
            {
                e.convert_ponder_to_real();
            }
            else if (token == "stop")
            {
                e.stop();
            }
            else if (token == "isready")
            {
                core::logs::uci << "readyok" << std::endl;
            }
            else if (token == "ucinewgame")
            {
                e.clear();
                b.load_fen(core::constants::FenInitPos);
            }
            else if (token == "position")
            {
                e.stop();
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