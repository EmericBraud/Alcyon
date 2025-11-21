#include "move_generator.hpp"

namespace MoveGen
{
    std::array<U64, BOARD_SIZE> KnightAttacks;
    std::array<U64, BOARD_SIZE> KingAttacks;
    std::array<U64, BOARD_SIZE> RookMasks;
    std::array<U64, BOARD_SIZE> BishopMasks;
    std::array<U64, BOARD_SIZE> PawnAttacksWhite;
    std::array<U64, BOARD_SIZE> PawnAttacksBlack;
    std::array<U64, BOARD_SIZE> PawnPushWhite;
    std::array<U64, BOARD_SIZE> PawnPushBlack;
    std::array<U64, BOARD_SIZE> PawnPush2White;
    std::array<U64, BOARD_SIZE> PawnPush2Black;
}
/**
 * Génère le Bitboard d'attaque pour une pièce non-glissante donnée
 * sur une case donnée.
 * * @param sq La case de départ (0-63).
 * @param shifts Les déplacements possibles (tableau de {dx, dy}).
 * @return Le Bitboard d'attaque.
 */
static U64 generate_attacks(const int sq, const std::array<int, 16> &shifts)
{
    U64 attacks = 0ULL;
    const int rank = sq / 8; // Rang (ligne) 0-7
    const int file = sq % 8; // Colonne (file) 0-7

    // Boucle à travers tous les déplacements possibles
    for (int i = 0; i < static_cast<int>(shifts.size() / 2); ++i)
    {
        int target_rank = rank + shifts[i * 2];
        int target_file = file + shifts[i * 2 + 1];

        // Vérification des bords de l'échiquier
        if (target_rank >= 0 && target_rank <= 7 &&
            target_file >= 0 && target_file <= 7)
        {
            int target_sq = target_rank * 8 + target_file;

            attacks |= sq_mask(target_sq);
        }
    }
    return attacks;
}

void MoveGen::initialize_rook_masks()
{
    for (int sq{0}; sq < BOARD_SIZE; ++sq)
    {
        const int sq_rank = sq / 8;
        const int sq_file = sq % 8;
        U64 mask = 0ULL;

        for (int file = sq_file + 1; file < 7; ++file)
            mask |= sq_mask(sq_rank * 8 + file);

        for (int file = sq_file - 1; file > 0; --file)
            mask |= sq_mask(sq_rank * 8 + file);

        for (int rank = sq_rank + 1; rank < 7; ++rank)
            mask |= sq_mask(rank * 8 + sq_file);

        for (int rank = sq_rank - 1; rank > 0; --rank)
            mask |= sq_mask(rank * 8 + sq_file);

        RookMasks[sq] = mask;
    }
}

void MoveGen::initialize_bishop_masks()
{
    for (int sq{0}; sq < BOARD_SIZE; ++sq)
    {
        const int sq_rank = sq / 8;
        const int sq_file = sq % 8;
        U64 mask = 0ULL;

        for (int rank = sq_rank + 1, file = sq_file + 1; rank < 7 && file < 7; ++rank, ++file)
            mask |= sq_mask(rank * 8 + file);

        for (int rank = sq_rank - 1, file = sq_file + 1; rank > 0 && file < 7; --rank, ++file)
            mask |= sq_mask(rank * 8 + file);

        for (int rank = sq_rank - 1, file = sq_file - 1; rank > 0 && file > 0; --rank, --file)
            mask |= sq_mask(rank * 8 + file);

        for (int rank = sq_rank + 1, file = sq_file - 1; rank < 7 && file > 0; ++rank, --file)
            mask |= sq_mask(rank * 8 + file);

        BishopMasks[sq] = mask;
    }
}
void MoveGen::initialize_pawn_masks()
{
    for (int sq{0}; sq < BOARD_SIZE; ++sq)
    {
        const int sq_col = sq % 8;
        const int sq_row = sq / 8;

        // White side
        U64 white_attack_mask{0ULL};
        U64 white_push_mask{0ULL};
        U64 white_push2_mask{0ULL};
        if (sq_row < 7)
        {
            if (sq_col > 0)
            {
                white_attack_mask |= sq_mask(sq_col - 1, sq_row + 1);
            }
            if (sq_col < 7)
            {
                white_attack_mask |= sq_mask(sq_col + 1, sq_row + 1);
            }
            white_push_mask |= sq_mask(sq_col, sq_row + 1);
            if (sq_row == 1)
                white_push2_mask |= sq_mask(sq_col, sq_row + 2);
        }

        // Black side
        U64 black_attack_mask{0ULL};
        U64 black_push_mask{0ULL};
        U64 black_push2_mask{0ULL};
        if (sq_row > 0)
        {
            if (sq_col > 0)
            {
                black_attack_mask |= sq_mask(sq_col - 1, sq_row - 1);
            }
            if (sq_col < 7)
            {
                black_attack_mask |= sq_mask(sq_col + 1, sq_row - 1);
            }
            black_push_mask |= sq_mask(sq_col, sq_row - 1);
            if (sq_row == 6)
                black_push2_mask |= sq_mask(sq_col, sq_row - 2);
        }
        PawnAttacksWhite[sq] = white_attack_mask;
        PawnPushWhite[sq] = white_push_mask;
        PawnPush2White[sq] = white_push2_mask;

        PawnAttacksBlack[sq] = black_attack_mask;
        PawnPushBlack[sq] = black_push_mask;
        PawnPush2Black[sq] = black_push2_mask;
    }
}

void MoveGen::initialize_bitboard_tables()
{
    // Déplacements du Cavalier (8 mouvements : {dr, df, dr, df, ...})
    const std::array<int, 16> KNIGHT_SHIFTS = {
        -2, -1, -2, 1, // Haut
        -1, -2, -1, 2, // Côtés
        1, -2, 1, 2,   // Côtés
        2, -1, 2, 1    // Bas
    };

    // Déplacements du Roi (8 mouvements : {dr, df, dr, df, ...})
    const std::array<int, 16> KING_SHIFTS = {
        -1, 0, -1, 1,
        0, 1, 1, 1,
        1, 0, 1, -1,
        0, -1, -1, -1};

    for (int sq = 0; sq < BOARD_SIZE; ++sq)
    {
        KnightAttacks[sq] = generate_attacks(sq, KNIGHT_SHIFTS);

        KingAttacks[sq] = generate_attacks(sq, KING_SHIFTS);
    }

    MoveGen::initialize_rook_masks();
    MoveGen::initialize_bishop_masks();
    MoveGen::initialize_pawn_masks();

    std::cout << "Bitboard tables initialized." << std::endl;
}
std::vector<Move> MoveGen::generate_pseudo_legal_moves(const Board &board, const Color color)
{
    std::vector<Move> moves;

    // 1. Boucle sur les 6 types de pièces (Pion à Roi)
    for (int piece_type = PAWN; piece_type <= KING; ++piece_type)
    {

        // 2. Obtention du Bitboard de cette pièce et couleur
        U64 piece_bitboard = board.get_piece_bitboard(color, piece_type);

        // 3. Boucle sur chaque pièce de ce type (pop_and_scan)
        while (piece_bitboard != 0)
        {
            int from_sq = pop_lsb(piece_bitboard); // Obtient et retire l'indice du bit LSB

            // 4. Dispatcher la logique (structure Switch/Case rapide)
            U64 target_mask;
            switch (piece_type)
            {
            case KNIGHT:
                target_mask = KnightAttacks[from_sq];
                break;
            case ROOK:
                target_mask = generate_rook_moves(from_sq, board); // Fonction plus complexe
                break;
            // ... autres pièces
            default:
                // log error
                break;
            }

            // 5. Filtrage et Extraction des Coups (pop_and_scan sur target_mask)
            // ... (Convertit le target_mask en une liste de structures Move et les ajoute à 'moves')
        }
    }
    return moves;
}

U64 MoveGen::generate_rook_moves(int from_sq, const Board &board)
{
    return U64();
}
