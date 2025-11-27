#pragma once

#include <SFML/Graphics.hpp>

#include "move_generator.hpp"

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
    GUI(Board &inital_board) : board(inital_board), window(sf::RenderWindow(sf::VideoMode(W_WIDTH, W_HEIGHT), "Chess 26", sf::Style::Titlebar | sf::Style::Close)), is_sq_selected(false), selected_sq(0), last_piece(0)
    {
        window.setFramerateLimit(60);
        const std::array<std::array<const char *, N_PIECES_TYPE_HALF>, 2> PIECE_IMAGE_PATHS = {
            {// WHITE
             {
                 "pieces/white-pawn.png",   // PAWN
                 "pieces/white-knight.png", // KNIGHT
                 "pieces/white-bishop.png", // BISHOP
                 "pieces/white-rook.png",   // ROOK
                 "pieces/white-queen.png",  // QUEEN
                 "pieces/white-king.png"    // KING
             },
             // BLACK
             {
                 "pieces/black-pawn.png",   // PAWN
                 "pieces/black-knight.png", // KNIGHT
                 "pieces/black-bishop.png", // BISHOP
                 "pieces/black-rook.png",   // ROOK
                 "pieces/black-queen.png",  // QUEEN
                 "pieces/black-king.png"    // KING
             }}};
        for (auto color : {WHITE, BLACK})
        {
            for (int piece{PAWN}; piece <= KING; ++piece)
            {
                const std::string fullPath = std::string(DATA_PATH) + PIECE_IMAGE_PATHS[color][piece];

                if (!m_piece_textures[color][piece].loadFromFile(fullPath))
                {
                    throw std::runtime_error("Impossible to open piece's image file: " + fullPath);
                }
                m_piece_textures[color][piece].setSmooth(false);
            }
        }
    }

    void run();

private:
    Board &board;
    sf::RenderWindow window;
    std::array<std::array<sf::Texture, N_PIECES_TYPE_HALF>, 2> m_piece_textures;
    bool is_sq_selected;
    int selected_sq;
    int last_piece;
    void draw_board();
    void draw_pieces();
    void draw_sq(int sq, sf::Color color);
    int sq_to_board_sq(const int sq);
};