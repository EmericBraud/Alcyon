#include "move_generator.hpp"
#include "gtest/gtest.h"

class ZorbistTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};

TEST_F(ZorbistTest, ZobristKeyUnchangec)
{
    Board b;
    b.load_fen("4k3/2P5/8/8/8/8/8/4K3 w - - 0 1");
    Move move(Square::c7, Square::c8, PAWN);
    MoveGen::init_move_flags(b, move);

    const U64 key = b.get_hash();

    b.play(move);
    ASSERT_NE(key, b.get_hash());
    b.unplay(move);
    ASSERT_EQ(key, b.get_hash());
}

TEST_F(ZorbistTest, ZobristTransposition)
{
    Board b1, b2;
    // Position de départ
    std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    b1.load_fen(fen);
    b2.load_fen(fen);

    // Séquence 1 : 1. e3 e6 2. Nf3 Nc6
    Move e2e3(Square::e2, Square::e3, PAWN);
    MoveGen::init_move_flags(b1, e2e3);
    Move e7e6(Square::e7, Square::e6, PAWN);
    MoveGen::init_move_flags(b1, e7e6);
    Move g1f3(Square::g1, Square::f3, KNIGHT);
    MoveGen::init_move_flags(b1, g1f3);
    Move b8c6(Square::b8, Square::c6, KNIGHT);
    MoveGen::init_move_flags(b1, b8c6);

    b1.play(e2e3);
    b1.play(e7e6);
    b1.play(g1f3);
    b1.play(b8c6);

    // Séquence 2 (Ordre différent) : 1. Nf3 Nc6 2. e3 e6
    Move Nf3_2(Square::g1, Square::f3, KNIGHT);
    MoveGen::init_move_flags(b2, Nf3_2);
    Move Nc6_2(Square::b8, Square::c6, KNIGHT);
    MoveGen::init_move_flags(b2, Nc6_2);
    Move e3_2(Square::e2, Square::e3, PAWN);
    MoveGen::init_move_flags(b2, e3_2);
    Move e6_2(Square::e7, Square::e6, PAWN);
    MoveGen::init_move_flags(b2, e6_2);

    b2.play(Nf3_2);
    b2.play(Nc6_2);
    b2.play(e3_2);
    b2.play(e6_2);

    ASSERT_EQ(b1.get_hash(), b2.get_hash()) << "Les positions identiques par transposition doivent avoir le même hash.";
}

TEST_F(ZorbistTest, ZobristIntegrityDeep)
{
    Board b;
    b.load_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    const U64 original_hash = b.get_hash();

    std::vector<Move> history;
    // On joue 5 coups légaux au hasard
    for (int i = 0; i < 5; ++i)
    {
        auto moves = MoveGen::generate_legal_moves(b);
        if (moves.empty())
            break;
        Move m = moves[0];
        b.play(m);
        history.push_back(m);
    }

    // On annule tout
    for (auto it = history.rbegin(); it != history.rend(); ++it)
    {
        b.unplay(*it);
    }

    ASSERT_EQ(original_hash, b.get_hash()) << "Le hash après play/unplay doit revenir à la valeur initiale.";
}

TEST_F(ZorbistTest, ZobristComponents)
{
    Board b;
    b.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    U64 hash1 = b.get_hash();

    // 1. Test du trait (Side to move)
    // On simule un changement de trait sans bouger de pièce
    b.switch_trait();
    U64 hash2 = b.get_hash();
    ASSERT_NE(hash1, hash2);

    // 2. Test des droits de roque
    // Si une tour est capturée, le hash doit changer à cause des droits de roque,
    // en plus de la disparition de la tour.
    b.load_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    U64 hash_start = b.get_hash();
    Move capture_rook(Square::a1, Square::a8, ROOK);
    MoveGen::init_move_flags(b, capture_rook);
    b.play(capture_rook); // Perd le roque blanc et noir côté dame
    ASSERT_NE(hash_start, b.get_hash());
}