#pragma once

#pragma once

#include <cassert>

#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"

namespace feature_encoder
{
    template <Color color>
        requires(color != NO_COLOR)
    int get_feature_index(int king_sq, Color piece_color, Piece piece_type, int piece_sq)
    {
        assert(piece_type != KING);
        // 1. Mise à jour de la perspective (Vertical Flip pour les Noirs)
        // Pour les blancs, on ne change rien. Pour les noirs, le roi en e8 devient e1.
        int final_ksq = color == WHITE ? king_sq : (king_sq ^ 56);
        int final_psq = color == WHITE ? piece_sq : (piece_sq ^ 56);

        // 2. Miroir horizontal (hm^) : On ramène toujours le roi sur les colonnes a-d
        if ((final_ksq % 8) > 3)
        {
            final_ksq ^= 7;
            final_psq ^= 7;
        }

        // 3. Calcul de l'index de la pièce (0 à 11)
        // Typiquement : PionAllié=0...RoiAllié=5, PionEnnemi=6...RoiEnnemi=11
        // /!\ Vérifie que ton enum Piece correspond (Pion=0, Cavalier=1, etc.)
        const int p_idx = static_cast<int>(piece_type) + (piece_color == color ? 0 : 6);

        // 4. Mapping du Roi sur 32 cases (Rangée * 4 colonnes + Colonne)
        int king_mapped_idx = (final_ksq / 8) * 4 + (final_ksq % 8);

        // 5. Index final : 32 cases roi * 12 pièces * 64 cases
        // 32 * 12 * 64 = 24576 total
        return (king_mapped_idx * 12 * 64) + (p_idx * 64) + final_psq;
    }
}