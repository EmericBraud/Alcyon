#pragma once

#include <cstdint>
#include <memory>

#include "common/huge_pages.hpp"

namespace Eval
{
    // Cache zobrist -> éval statique NNUE. Contrairement au score de la TT,
    // une éval statique ne dépend que de la position : elle ne périme jamais
    // (pas de profondeur, pas de borne), donc pas de politique de
    // remplacement ni de vidage entre les coups -- une entrée écrasée coûte
    // juste un recalcul plus tard. L'iterative deepening revisite quasi tout
    // l'arbre de l'itération précédente : chaque hit économise un forward
    // pass NNUE complet (catch-up lazy de l'accumulateur + couches denses),
    // le premier poste du profil (~45% du temps de recherche).
    //
    // Une entrée = un seul mot de 64 bits : tag = les 48 bits hauts de la clé
    // zobrist, éval dans les 16 bits bas. Lecture/écriture en un seul mot
    // aligné -- pas de déchirement possible en pratique sur x86, même
    // convention lockless que la TT (dont les lectures sont déjà
    // non-atomiques). Un faux match de tag demande une collision sur 48 bits
    // à index égal : négligeable.
    class EvalCache
    {
        // 2^21 entrées x 8B = 16MB : assez grand pour couvrir plusieurs
        // itérations d'un search, assez petit pour ne pas concurrencer la TT.
        static constexpr std::size_t NumEntries = std::size_t(1) << 21;
        static constexpr std::uint64_t TagMask = ~std::uint64_t(0) << 16;

        std::unique_ptr<std::uint64_t[]> entries;

    public:
        EvalCache() : entries(std::make_unique<std::uint64_t[]>(NumEntries))
        {
            cpu::advise_huge_pages(entries.get(), NumEntries * sizeof(std::uint64_t));
        }

        bool probe(std::uint64_t key, int &eval_out) const
        {
            const std::uint64_t e = entries[key & (NumEntries - 1)];
            if ((e & TagMask) != (key & TagMask))
                return false;
            eval_out = static_cast<std::int16_t>(e & 0xFFFF);
            return true;
        }

        void store(std::uint64_t key, int eval)
        {
            entries[key & (NumEntries - 1)] =
                (key & TagMask) | static_cast<std::uint16_t>(static_cast<std::int16_t>(eval));
        }
    };
}
