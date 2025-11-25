#include "gui.hpp"

void GUI::run()
{
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }
        }
        window.clear(BG_COLOR);
        draw_board();
        window.display();
    }
}

void GUI::draw_board()
{
    for (int row{0}; row < 8; ++row)
    {
        for (int col{0}; col < 8; col++)
        {
            sf::RectangleShape rect;
            rect.setSize(sf::Vector2f(SQ_PX_SIZE, SQ_PX_SIZE));
            rect.setPosition(SQ_PX_SIZE * col, SQ_PX_SIZE * row);
            rect.setFillColor(((row + col) % 2 == 0) ? SQ_COLOR : sf::Color::White);

            window.draw(rect);
        }
    }
}
