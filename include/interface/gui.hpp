#pragma once

#include <array>

#include <SFML/Graphics.hpp>

#include "core/utils/file.hpp"
#include "core/utils/constants.hpp"
#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"
#include "core/move/history.hpp"
#include "core/board.hpp"

#include "engine/engine_manager.hpp"

#define W_HEIGHT 600
#define W_WIDTH 600
#define SQ_PX_SIZE 75
#define BG_COLOR sf::Color(100, 100, 100)
#define SQ_COLOR sf::Color(70, 100, 200)
#define SQ_COLOR_2 sf::Color(255, 255, 255)

#define SQ_SELECTED_COLOR sf::Color(170, 180, 225)

#define SQ_ATTACK_COLOR sf::Color(180, 80, 80)
#define SQ_ATTACK_COLOR_2 sf::Color(180, 100, 100)

class GUI
{
public:
    GUI(Board &initial_board, bool auto_play = false, Color computer_side = BLACK) : board(initial_board), computer(EngineManager(initial_board)), window(sf::RenderWindow(sf::VideoMode(W_WIDTH, W_HEIGHT), "Chess 26", sf::Style::Titlebar | sf::Style::Close)), is_sq_selected(false), selected_sq(0), last_piece(0), auto_play(auto_play), computer_side(computer_side)
    {
        history.clear();
        window.setFramerateLimit(60);
        for (auto color : {WHITE, BLACK})
        {
            for (int piece{PAWN}; piece <= KING; ++piece)
            {
                const std::string fullPath = core::file::get_data_path(core::file::PieceImagePath[color][piece]);

                if (!m_piece_textures[color][piece].loadFromFile(fullPath))
                {
                    throw std::runtime_error("Impossible to open piece's image file: " + fullPath);
                }
                m_piece_textures[color][piece].setSmooth(false);
            }
        }
        board.attach_history(&history);
    }

    void run();

private:
    Board &board;
    History history;
    EngineManager computer;
    sf::RenderWindow window;
    std::array<std::array<sf::Texture, core::constants::PieceTypeCount>, 2> m_piece_textures;
    bool is_sq_selected;
    int selected_sq;
    int last_piece;
    bool auto_play;
    Color computer_side;
    void draw_board();
    void draw_pieces();
    void draw_sq(int sq, sf::Color color);
    int sq_to_board_sq(const int sq);
};