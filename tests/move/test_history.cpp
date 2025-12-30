#include "gtest/gtest.h"
#include "core/board.hpp"

class HistoryTest : public ::testing::Test
{
protected:
    // Helper pour accéder aux membres privés si nécessaire ou vérifier les états
    bool IsOwned(const Board &b)
    {
        return reinterpret_cast<uintptr_t>(b.history_tagged) & 1ULL;
    }
};

// 1. Test de la propriété par défaut (Heap)
TEST_F(HistoryTest, DefaultConstructor_OwnsHeap)
{
    Board b;
    EXPECT_TRUE(IsOwned(b)) << "Par défaut, le Board doit posséder son historique sur le tas.";
    EXPECT_NE(b.get_history(), nullptr);

    // Vérifie que l'historique est bien initialisé (count = 0)
    EXPECT_EQ(b.get_history()->size(), 0);
}

// 2. Test du comportement sur la PILE (Stack)
TEST_F(HistoryTest, StackHistory_NoDoubleFree)
{
    History stack_history;
    {
        Board b;
        // Simule le passage sur la pile
        if (IsOwned(b))
            delete b.get_history();
        b.history_tagged = reinterpret_cast<History *>(&stack_history); // Bit 0 est 0 car aligné

        EXPECT_FALSE(IsOwned(b)) << "Le Board ne doit pas posséder l'historique de la pile.";

        b.get_history()->push_back({12345, 0, 0, Move()});
        EXPECT_EQ(stack_history.size(), 1);
    }
    // Si le destructeur de 'b' tente de delete stack_history, le test crashera ici.
    SUCCEED();
}

// 3. Test de la Copie Profonde (Crucial pour Negamax)
TEST_F(HistoryTest, DeepCopy_IndependentHistory)
{
    Board b1;
    b1.get_history()->push_back({0xAAAA, 10, 0, Move()});

    // Copie de b1 vers b2
    Board b2 = b1;

    EXPECT_TRUE(IsOwned(b2));
    EXPECT_NE(b1.get_history(), b2.get_history()) << "Les adresses d'historique doivent être différentes.";
    EXPECT_EQ(b2.get_history()->size(), 1);
    EXPECT_EQ(b2.get_history()->back().zobrist_key, 0xAAAA);

    // Modifie b2 et vérifie que b1 ne change pas
    b2.get_history()->push_back({0xBBBB, 11, 0, Move()});
    EXPECT_EQ(b1.get_history()->size(), 1);
    EXPECT_EQ(b2.get_history()->size(), 2);
}

// 4. Test du Déplacement (Move)
TEST_F(HistoryTest, MoveConstructor_TransfersOwnership)
{
    Board b1;
    History *original_ptr = b1.get_history();
    b1.get_history()->push_back({0x123, 1, 1, Move()});

    Board b2(std::move(b1));

    EXPECT_EQ(b2.get_history(), original_ptr);
    EXPECT_TRUE(IsOwned(b2));
    // b1 doit être "neutre" (ici nullptr selon votre implémentation)
    // EXPECT_EQ(b1.history_tagged, nullptr);
}

// 5. Test d'intégrité après Play/Unplay
TEST_F(HistoryTest, PlayUnplay_HistoryIntegrity)
{
    Board b;
    b.load_fen(STARTING_POS_FEN);
    U64 initial_key = b.get_hash();

    Move m1(Square::e2, Square::e4, PAWN, Move::Flags::DOUBLE_PUSH);
    b.play(m1);
    EXPECT_EQ(b.get_history()->size(), 1);
    EXPECT_EQ(b.get_history()->back().zobrist_key, initial_key);

    b.unplay(m1);
    EXPECT_EQ(b.get_history()->size(), 0);
    EXPECT_EQ(b.get_hash(), initial_key) << "Zobrist Key corrompue après Unplay.";
}