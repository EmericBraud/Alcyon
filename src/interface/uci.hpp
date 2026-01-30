#pragma once

#include <expected>
#include <iostream>
#include <vector>

#include "common/logger.hpp"
#include "core/board/board.hpp"
#include "core/move/generator/move_generator.hpp"

#include "core/board/zobrist.hpp"
#include "engine/eval/book.hpp"
#include "engine/engine_manager.hpp"
#include "engine/config/config.hpp"

#include "interface/gui.hpp"
#include "interface/uci_option.hpp"

class UCI
{
    VBoard b;
    EngineManager e;
#ifdef SPSA_TUNING
    std::vector<UCIOption<int>> int_options;
    std::vector<UCIOption<double>> double_options;
#endif
    bool ponder_enabled = true;

    std::expected<int, Move::MoveError> parse_position(VBoard &board, std::istringstream &is)
    {
        std::string token;
        is >> token;

        if (token == "startpos")
        {
            board.load_fen(constants::FenInitPos);
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

    void parse_go(VBoard &board, EngineManager &engine, std::istringstream &is)
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

        logs::debug << "info string DEBUG: Checking Book..." << std::endl;
        logs::debug << "info string DEBUG: My Hash is " << std::hex << board.polyglot_key() << std::dec << std::endl;
        if (!is_infinite && !is_ponder)
        {
            Move book_move = Book::probe(board);

            if (book_move.get_value() != 0)
            {
                logs::debug << "info string DEBUG: Book Move Found!" << std::endl;
                if (board.is_move_pseudo_legal(book_move) && board.is_move_legal(book_move))
                {
                    logs::uci << "bestmove " << book_move.to_uci() << std::endl;
                    return;
                }
            }
            else
            {
                logs::debug << "info string DEBUG: No move found in book." << std::endl;
            }
        }
        if (!is_infinite && !is_ponder && std::popcount(board.get_occupancy<NO_COLOR>()) <= engine_constants::eval::SyzygyMaxPieces)
        {
            logs::debug << "info string DEBUG: Checking TB..." << std::endl;
            TableBase::RootResult r = e.get_tb().probe_root(board);
            if (r.move != 0)
            {
                logs::uci << "bestmove " << r.move.to_uci() << std::endl;
                return;
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

            time_to_think = (my_time / 28) + (my_inc / 2);
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
        std::string content = is.str();

        for (UCIOption<int> &int_option : int_options)
        {
            std::istringstream content_copy(content);
            if (int_option.parse_input(content_copy))
            {
                logs::debug << "Input parsed : " << int_option.get_name() << std::endl;
                return;
            }
        }
        for (UCIOption<double> &double_option : double_options)
        {
            std::istringstream content_copy(content);
            if (double_option.parse_input(content_copy))
            {
                logs::debug << "Input parsed : " << double_option.get_name() << std::endl;
                return;
            }
        }

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

        if (name == "Threads ")
        {
            try
            {
                int threads = std::stoi(value);
                e.set_threads(threads);
            }
            catch (...)
            {
                logs::debug << "info string Error parsing thread value" << std::endl;
            }
        }
        else if (name == "Ponder ")
            ponder_enabled = (value == "true ");
        else if (name == "Hash ")
        {
            int size = std::stoi(value);
            e.get_tt().resize(size);
            logs::debug << "info string Hash table resized" << std::endl;
        }
        else if (name == "Clear Hash ")
        {
            e.get_tt().clear();
            logs::debug << "info string Hash table cleared" << std::endl;
        }
    }

public:
    UCI() : b(), e(b)
    {
        MoveGen::initialize_bitboard_tables();
        init_zobrist();
        b.load_fen(constants::FenInitPos);
        Book::init(DATA_PATH "komodo.bin");
#ifdef SPSA_TUNING
        int_options = {
            UCIOption<int>(&engine_constants::search::razoring::MaxDepth, "razoring_max_depth"),
            UCIOption<int>(&engine_constants::search::razoring::MarginDepthFactor, "razoring_depth_factor"),
            UCIOption<int>(&engine_constants::search::razoring::MarginConst, "razoring_margin_const"),

            UCIOption<int>(&engine_constants::search::reverse_futility_pruning::MaxDepth, "rfp_max_depth"),
            UCIOption<int>(&engine_constants::search::reverse_futility_pruning::MarginDepthFactor, "rfp_marg_d_fact"),
            UCIOption<int>(&engine_constants::search::reverse_futility_pruning::MarginDepthFactor, "rfp_marg_const"),

            UCIOption<int>(&engine_constants::search::iterative_deepening::MaxDepth, "itd_max_depth"),
            UCIOption<int>(&engine_constants::search::iterative_deepening::NewDepthIncr, "itd_new_depth_inc"),

            UCIOption<int>(&engine_constants::search::null_move_pruning::MinDepth, "nmp_min_depth"),
            UCIOption<int>(&engine_constants::search::null_move_pruning::RConst, "nmp_r_const"),
            UCIOption<int>(&engine_constants::search::null_move_pruning::RDiv, "nmp_r_div"),

            UCIOption<int>(&engine_constants::search::futility_pruning::MaxDepth, "fp_max_depth"),
            UCIOption<int>(&engine_constants::search::futility_pruning::MarginConst, "fp_margin_const"),
            UCIOption<int>(&engine_constants::search::futility_pruning::MarginDepthFactor, "fp_margin_d_fact"),

            UCIOption<int>(&engine_constants::search::singular::MinDepth, "singulat_min_depth"),

            UCIOption<int>(&engine_constants::search::null_move_reduction::MaxDepth, "nmr_max_depth"),
            UCIOption<int>(&engine_constants::search::null_move_reduction::MaxMovesConst, "nmr_max_moves_const"),

            UCIOption<int>(&engine_constants::search::late_move_reduction::MinDepth, "lmr_min_depth"),
            UCIOption<int>(&engine_constants::search::late_move_reduction::MinMovesSearched, "lmr_min_moves_searched"),
            UCIOption<int>(&engine_constants::search::late_move_reduction::MaxDepthReduction, "lmr_max_depth_reduction"),

            UCIOption<int>(&engine_constants::search::see_pruning::MaxDepth, "see_pruning_max_depth"),
            UCIOption<int>(&engine_constants::search::see_pruning::ThresholdDepthFactor, "threshold_depth_factor"),

        };

        double_options = {
            UCIOption<double>(&engine_constants::search::late_move_reduction::TableInitConst, "lmr_table_init_const"),
            UCIOption<double>(&engine_constants::search::late_move_reduction::TableInitDiv, "lmr_table_init_div"),
        };
#endif
    }
    void loop()
    {
        std::string line, token;

        while (std::getline(std::cin, line))
        {
            logs::debug << "info string << " << line << std::endl;

            std::istringstream is(line);
            token.clear();
            is >> token;

            if (token == "uci")
            {
                int default_threads = std::thread::hardware_concurrency();

                logs::uci << "id name Chess26" << std::endl;
                logs::uci << "id author Emeric" << std::endl;
                logs::uci << "option name Threads type spin default " << default_threads << " min 1 max 128" << std::endl;
                logs::uci << "option name Hash type spin default 512 min 1 max 2048" << std::endl;
                logs::uci << "option name Move Overhead type spin default 100 min 0 max 1000" << std::endl;
                logs::uci << "option name Ponder type check default false" << std::endl;
                logs::uci << "option name Threads type spin default 16 min 1 max 16" << std::endl;
                for (auto int_option : int_options)
                {
                    int_option.init_print();
                }
                for (auto double_option : double_options)
                {
                    double_option.init_print();
                }
                logs::uci << "uciok" << std::endl;
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
                logs::uci << "readyok" << std::endl;
            }
            else if (token == "ucinewgame")
            {
                e.clear();
                b.load_fen(constants::FenInitPos);
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