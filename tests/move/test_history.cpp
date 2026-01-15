#include "gtest/gtest.h"
#include "core/move/generator/move_generator.hpp"
#include <random>

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

        b.get_history()->push_back({12345, 0, 0, Move(), b.state.en_passant_sq, b.state.castling_rights});
        EXPECT_EQ(stack_history.size(), 1);
    }
    // Si le destructeur de 'b' tente de delete stack_history, le test crashera ici.
    SUCCEED();
}

// 3. Test de la Copie Profonde (Crucial pour Negamax)
TEST_F(HistoryTest, DeepCopy_IndependentHistory)
{
    Board b1;
    b1.get_history()->push_back({0xAAAA, 10, 0, Move(), b1.state.en_passant_sq, b1.state.castling_rights});

    // Copie de b1 vers b2
    Board b2 = b1;

    EXPECT_TRUE(IsOwned(b2));
    EXPECT_NE(b1.get_history(), b2.get_history()) << "Les adresses d'historique doivent être différentes.";
    EXPECT_EQ(b2.get_history()->size(), 1);
    EXPECT_EQ(b2.get_history()->back().zobrist_key, 0xAAAA);

    // Modifie b2 et vérifie que b1 ne change pas
    b2.get_history()->push_back({0xBBBB, 11, 0, Move(), b2.state.en_passant_sq, b2.state.castling_rights});
    EXPECT_EQ(b1.get_history()->size(), 1);
    EXPECT_EQ(b2.get_history()->size(), 2);
}

// 4. Test du Déplacement (Move)
TEST_F(HistoryTest, MoveConstructor_TransfersOwnership)
{
    Board b1;
    History *original_ptr = b1.get_history();
    b1.get_history()->push_back({0x123, 1, 1, Move(), b1.state.en_passant_sq, b1.state.castling_rights});

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
    b.load_fen(constants::FenInitPos);
    U64 initial_key = b.get_hash();

    Move m1(Square::e2, Square::e4, PAWN, Move::Flags::DOUBLE_PUSH);
    b.play(m1);
    EXPECT_EQ(b.get_history()->size(), 1);
    EXPECT_EQ(b.get_history()->back().zobrist_key, initial_key);

    b.unplay(m1);
    EXPECT_EQ(b.get_history()->size(), 0);
    EXPECT_EQ(b.get_hash(), initial_key) << "Zobrist Key corrompue après Unplay.";
}

TEST_F(HistoryTest, ZobristConsistency_DeepRandomSearch)
{
    Board b;
    b.load_fen(constants::FenInitPos);

    const int MAX_PLY = 50;     // Profondeur de la simulation
    const int ITERATIONS = 100; // Nombre de parcours complets

    std::mt19937 rng(42); // Seed fixe pour la reproductibilité

    for (int i = 0; i < ITERATIONS; ++i)
    {
        U64 keys_at_ply[MAX_PLY + 1];
        std::vector<Move> moves_played;

        keys_at_ply[0] = b.get_hash();

        // --- PHASE 1 : JOUER DES COUPS ---
        int actual_plies = 0;
        for (int p = 0; p < MAX_PLY; ++p)
        {
            MoveList list;
            // Utilise ton wrapper dynamique pour gérer le template <Us>
            if (b.get_side_to_move() == WHITE)
                MoveGen::generate_legal_moves<WHITE>(b, list);
            else
                MoveGen::generate_legal_moves<BLACK>(b, list);

            if (list.count == 0)
                break; // Mat ou Pat

            // Sélectionne un coup aléatoire
            std::uniform_int_distribution<int> dist(0, list.count - 1);
            Move m = list.moves[dist(rng)];

            b.play(m);
            moves_played.push_back(m);
            actual_plies++;
            keys_at_ply[actual_plies] = b.get_hash();

            // Vérification optionnelle : la clé après play ne doit pas être
            // identique à la clé avant play (sauf exception rarissime)
            EXPECT_NE(keys_at_ply[actual_plies], keys_at_ply[actual_plies - 1])
                << "Coup joué: " << m.to_uci() << " n'a pas modifié la Zobrist Key.";
        }

        // --- PHASE 2 : DÉFAIRE LES COUPS ---
        for (int p = actual_plies - 1; p >= 0; --p)
        {
            Move m = moves_played[p];
            b.unplay(m);

            EXPECT_EQ(b.get_hash(), keys_at_ply[p])
                << "Échec à la profondeur " << p
                << " après avoir défait le coup " << m.to_uci();

            // Vérification de l'intégrité structurelle (Mailbox vs Bitboards)
            // Si la Zobrist est fausse, c'est souvent ici que ça casse
            for (int sq = 0; sq < 64; ++sq)
            {
                Piece p_mailbox = b.get_p(sq); // supposant que ceci renvoie la pièce
                if (p_mailbox != NO_PIECE)
                {
                    Color c = b.get_c(sq);
                    // Vérifie que le bitboard de la pièce possède bien ce bit
                    EXPECT_TRUE(b.get_piece_bitboard(c, p_mailbox) & (1ULL << sq))
                        << "Incohérence Mailbox/Bitboard pour la pièce " << p_mailbox
                        << " sur la case " << sq;
                }
            }
        }

        // --- PHASE FINALE : RETOUR AU DÉPART ---
        EXPECT_EQ(b.get_hash(), keys_at_ply[0])
            << "La clé finale après avoir tout annulé ne correspond pas à la clé initiale.";
    }
}