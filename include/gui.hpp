#pragma once

#include <SFML/Graphics.hpp>

#include "move_generator.hpp"

#define W_HEIGHT 600
#define W_WIDTH 600
#define SQ_PX_SIZE 75
#define BG_COLOR sf::Color(100, 100, 100)
#define SQ_COLOR sf::Color(50, 70, 200)

class GUI
{
public:
    GUI(Board &inital_board) : board(inital_board), window(sf::RenderWindow(sf::VideoMode(W_WIDTH, W_HEIGHT), "Chess 26"))
    {
        window.setFramerateLimit(60);
    }

    void run();

private:
    Board &board;
    sf::RenderWindow window;
    void draw_board();
};