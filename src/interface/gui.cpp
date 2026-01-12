#include "interface/gui.hpp"

void GUI::run()
{
    while (window.isOpen())
    {
        sf::Event event;
        // computer.play();
        if (auto_play && board.get_side_to_move() == computer_side)
        {
            logs::debug << "Playing position with depth " << MAX_DEPTH << " ..." << std::endl;
            computer.start_search();
        }
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                window.close();
            }
            else if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left)
            {
                const int clickX{event.mouseButton.x}, clickY{event.mouseButton.y};
                const int sq_col{clickX / SQ_PX_SIZE}, sq_row{7 - clickY / SQ_PX_SIZE};
                selected_sq = sq_col + 8 * sq_row;
                if ((board.get_occupancy(board.get_side_to_move()) & (1ULL << selected_sq)))
                {
                    is_sq_selected = true;
                    last_piece = selected_sq;
                }
                else
                {
                    if (is_sq_selected && MoveGen::get_legal_moves_mask(board, last_piece) & (1ULL << selected_sq))
                    {
                        PieceInfo info = board.get_piece_on_square(last_piece);
                        Move move = 0;
                        if (info.second == KING && abs(last_piece - selected_sq) == 2)
                        {                                 // Castle
                            if (last_piece < selected_sq) // King side
                            {
                                move = Move(last_piece, selected_sq, info.second, Move::Flags::KING_CASTLE);
                            }
                            else
                            { // Queen side
                                move = Move(last_piece, selected_sq, info.second, Move::Flags::QUEEN_CASTLE);
                            }
                        }
                        else
                        {
                            move = Move(last_piece, selected_sq, info.second);
                            MoveGen::init_move_flags(board, move);
                        }
                        board.play(move);
                    }
                    is_sq_selected = false;
                }
            }
            else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::E)
            {
                logs::debug << "Evaluating position with depth " << MAX_DEPTH << " ..." << std::endl;
                const int score{computer.evaluate_position(5000)};
                logs::debug << "Position score : " << score << std::endl;
            }
            else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::P)
            {
                logs::debug << "Playing position with depth " << MAX_DEPTH << " ..." << std::endl;
                computer.start_search(20000, false, false, true);
            }
            else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::U)
            {
                board.undo_last_move();
                logs::debug << "Move undone" << std::endl;
            }
        }
        window.clear(BG_COLOR);
        draw_board();
        draw_pieces();
        window.display();
    }
}

void GUI::draw_sq(int sq, sf::Color color)
{
    const int col{sq % 8}, row{7 - sq / 8};
    sf::RectangleShape rect;
    rect.setSize(sf::Vector2f(SQ_PX_SIZE, SQ_PX_SIZE));
    rect.setPosition(SQ_PX_SIZE * col, SQ_PX_SIZE * row);
    rect.setFillColor(color);

    window.draw(rect);
}

int GUI::sq_to_board_sq(const int sq)
{
    const int col{sq % 8};
    const int row{7 - sq / 8};
    return col + 8 * row;
}

void GUI::draw_board()
{
    for (int row{0}; row < 8; ++row)
    {
        for (int col{0}; col < 8; col++)
        {
            if (is_sq_selected && last_piece == 8 * row + col)
            {
                draw_sq(8 * row + col, SQ_SELECTED_COLOR);
            }
            else if (is_sq_selected && (MoveGen::get_legal_moves_mask(board, last_piece) & (1ULL << (row * 8 + col))))
            {
                draw_sq(8 * row + col, ((row + col) % 2 == 0) ? SQ_ATTACK_COLOR : SQ_ATTACK_COLOR_2);
            }
            else
            {
                draw_sq(8 * row + col, ((row + col) % 2 == 0) ? SQ_COLOR : SQ_COLOR_2);
            }
        }
    }
}

void GUI::draw_pieces()
{

    for (auto color : {WHITE, BLACK})
    {
        for (int piece{PAWN}; piece <= KING; ++piece)
        {
            const bitboard b = board.get_piece_bitboard(color, piece);
            bitboard temp_bb = b;
            while (temp_bb != 0)
            {
                const int sq = pop_lsb(temp_bb);
                const sf::Texture &texture = m_piece_textures[color][piece];
                sf::Vector2u textureSize = texture.getSize();
                const double scale = static_cast<double>(SQ_PX_SIZE) / static_cast<double>(textureSize.x);
                sf::Sprite sprite;
                sprite.setTexture(texture);
                sprite.setPosition((sq % 8) * SQ_PX_SIZE, (7 - sq / 8) * SQ_PX_SIZE);
                sprite.setScale(scale, scale);
                window.draw(sprite);
            }
        }
    }
}
