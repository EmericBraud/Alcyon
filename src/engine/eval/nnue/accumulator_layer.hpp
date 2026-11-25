#pragma once

#include <type_traits>

#include <cstdlib>
#include <array>
#include <cstdint>
#include <utility>
#include <memory>
#include <experimental/simd>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "core/piece/color.hpp"
#include "common/aligned_array.hpp"
#include "common/cpu.hpp"
#include "common/simd.hpp"

// NThreatFeatures: rows [0, NThreatFeatures) are stored as int8 (Full_Threats
// segment of the combined feature set, see nnue_eval.hpp) instead of int16 --
// those rows are ~half the file's weight bytes (170MB -> ~108MB: 60,720 rows
// at 1024 * 1 byte instead of 1024 * 2 bytes) and, per profiling, are nearly
// all of the threat-diff apply pass's DRAM traffic (the halfka segment updates via a
// different, much colder path -- HalfKA piece toggles, one/two features per
// non-king move). Widening int8 -> int16 on the fly (sign-extend, like
// vpmovsxbw) costs a few cycles of ALU work that's fully hidden behind the
// cache-miss latency the load itself already pays, so this halves bytes
// fetched from DRAM per update with no change to the arithmetic result
// (weights are quantized losslessly into int8 by the same trainer/export
// pipeline that already stores Full_Threats as int8 in the file -- see
// nnue_eval.hpp's read_segment/weight_is_int8).
template <int NFeatures, int NNeurons, int NThreatFeatures>
class AccumulatorLayer
{
    static_assert(NThreatFeatures >= 0 && NThreatFeatures <= NFeatures,
                  "NThreatFeatures must be within [0, NFeatures]");

public:
    // Public so callers can snapshot/restore the full per-perspective state
    // (see raw_state/restore_raw_state below) without depending on its
    // internal layout beyond "one opaque, copyable blob".
    using AccTable = AlignedArray<std::array<std::int16_t, NNeurons>, 2>;

private:
    using BiasTable = AlignedArray<std::int16_t, NNeurons>;
    // Feature transformer weights: Full_Threats rows are stored int8 (see
    // class comment above), HalfKAv2_hm^ rows stay int16 (scale = 127) --
    // matching the file's own per-segment encoding, so no widening happens
    // at load time (nnue_eval.hpp's read_segment reads each segment directly
    // into its native width).
    using Int8WeightTable = AlignedArray<std::array<std::int8_t, NNeurons>, NThreatFeatures>;
    using Int16WeightTable = AlignedArray<std::array<std::int16_t, NNeurons>, NFeatures - NThreatFeatures>;

    std::unique_ptr<AccTable> accumulators;
    std::shared_ptr<const BiasTable> biases;
    std::shared_ptr<const Int8WeightTable> threat_weights;
    std::shared_ptr<const Int16WeightTable> halfka_weights;

    // HalfKAv2_hm^ rows are already int16 -- pas de conversion a faire, mais
    // la boucle etait laissee scalaire et confiee a l'auto-vectorisation, la
    // ou le chemin int8 est explicitement vectorise. Le profil (xctrace,
    // accumulator_layer.hpp:62 et :64) la donne a 12,4 % du temps du moteur,
    // donc elle merite le meme traitement : SIMD explicite, deroule par deux.
    template <bool activate>
    static void apply_row(std::array<std::int16_t, NNeurons> &acc, const std::array<std::int16_t, NNeurons> &w_row)
    {
        constexpr std::size_t W = simd::SimdSize16;

        std::size_t j = 0;
        for (; j + 2 * W <= static_cast<std::size_t>(NNeurons); j += 2 * W)
        {
            simd::int16_v w0, w1, a0, a1;
            w0.copy_from(w_row.data() + j, stdx::element_aligned);
            w1.copy_from(w_row.data() + j + W, stdx::element_aligned);
            a0.copy_from(acc.data() + j, stdx::element_aligned);
            a1.copy_from(acc.data() + j + W, stdx::element_aligned);
            if constexpr (activate)
            {
                a0 += w0;
                a1 += w1;
            }
            else
            {
                a0 -= w0;
                a1 -= w1;
            }
            a0.copy_to(acc.data() + j, stdx::element_aligned);
            a1.copy_to(acc.data() + j + W, stdx::element_aligned);
        }
        for (; j < static_cast<std::size_t>(NNeurons); ++j)
        {
            if constexpr (activate)
                acc[j] += w_row[j];
            else
                acc[j] -= w_row[j];
        }
    }

    // Full_Threats rows are int8 in memory (half the bytes fetched from DRAM
    // per update, see class comment above): each chunk of int16_v::size()
    // int8 weights is loaded and widened (sign-extended, like vpmovsxbw) to
    // int16 lanes before adding/subtracting -- the widen is pure ALU work
    // that overlaps with the load's cache-miss latency, so it costs nothing
    // extra versus a native int16 row on the memory-bound hot path this
    // serves (NnueEval::apply_list/initialize_perspective).
    template <bool activate>
    static void apply_row(std::array<std::int16_t, NNeurons> &acc, const std::array<std::int8_t, NNeurons> &w_row)
    {
        using int8_n = stdx::fixed_size_simd<std::int8_t, simd::SimdSize16>;
        constexpr std::size_t W = simd::SimdSize16;

        std::size_t j = 0;

        // Deroulage par deux : cette boucle ne chargeait que
        // SimdSize16 = 8 octets de poids par iteration, soit la moitie d'un
        // registre NEON (native_simd<int8_t> en tient 16), parce que sa
        // largeur est calee sur le vecteur int16 de sortie. Deux chargements
        // contigus par iteration laissent le compilateur les fusionner en un
        // seul acces 16 octets, et divisent par deux le nombre d'iterations
        // de la boucle la plus chaude du moteur (27 % du temps au profil).
        for (; j + 2 * W <= static_cast<std::size_t>(NNeurons); j += 2 * W)
        {
            int8_n r0, r1;
            r0.copy_from(w_row.data() + j, stdx::element_aligned);
            r1.copy_from(w_row.data() + j + W, stdx::element_aligned);
            const simd::int16_v w0 = stdx::simd_cast<simd::int16_v>(r0);
            const simd::int16_v w1 = stdx::simd_cast<simd::int16_v>(r1);

            simd::int16_v a0, a1;
            a0.copy_from(acc.data() + j, stdx::element_aligned);
            a1.copy_from(acc.data() + j + W, stdx::element_aligned);
            if constexpr (activate)
            {
                a0 += w0;
                a1 += w1;
            }
            else
            {
                a0 -= w0;
                a1 -= w1;
            }
            a0.copy_to(acc.data() + j, stdx::element_aligned);
            a1.copy_to(acc.data() + j + W, stdx::element_aligned);
        }

        for (; j + simd::SimdSize16 <= static_cast<std::size_t>(NNeurons); j += simd::SimdSize16)
        {
            int8_n raw;
            raw.copy_from(w_row.data() + j, stdx::element_aligned);
            const simd::int16_v widened = stdx::simd_cast<simd::int16_v>(raw);

            simd::int16_v acc_v;
            acc_v.copy_from(acc.data() + j, stdx::element_aligned);
            if constexpr (activate)
                acc_v += widened;
            else
                acc_v -= widened;
            acc_v.copy_to(acc.data() + j, stdx::element_aligned);
        }
        for (; j < static_cast<std::size_t>(NNeurons); ++j)
        {
            if constexpr (activate)
                acc[j] += w_row[j];
            else
                acc[j] -= w_row[j];
        }
    }

public:
    // Chaque table arrive soit comme un tableau a copier dans le tas (chemin
    // historique), soit comme un shared_ptr deja pret -- typiquement une vue
    // sur un fichier mappe en lecture seule, partagee entre processus (voir
    // common/mapped_file.hpp). Le second cas evite N copies physiques des
    // 111 Mo de poids quand N instances du moteur tournent en parallele.
    template <typename T, typename A>
    static std::shared_ptr<const T> adopt_or_copy(A &&a)
    {
        if constexpr (std::is_same_v<std::decay_t<A>, std::shared_ptr<const T>>)
            return std::forward<A>(a);
        else
            return std::make_shared<const T>(std::forward<A>(a));
    }

    template <typename B, typename W8, typename W16>
    AccumulatorLayer(B &&b, W8 &&w8, W16 &&w16)
    {
        biases = adopt_or_copy<BiasTable>(std::forward<B>(b));
        threat_weights = adopt_or_copy<Int8WeightTable>(std::forward<W8>(w8));
        halfka_weights = adopt_or_copy<Int16WeightTable>(std::forward<W16>(w16));
        accumulators = std::make_unique<AccTable>();
        reset();
    }
    AccumulatorLayer(const AccumulatorLayer &other)
    {
        accumulators = std::make_unique<AccTable>(*other.accumulators);
        biases = other.biases;
        threat_weights = other.threat_weights;
        halfka_weights = other.halfka_weights;
    }

    AccumulatorLayer &operator=(const AccumulatorLayer &other)
    {
        if (this != &other)
        {
            *accumulators = *other.accumulators;
            biases = other.biases;
            threat_weights = other.threat_weights;
            halfka_weights = other.halfka_weights;
        }
        return *this;
    }

    // Declared explicitly: a user-declared copy-ctor/assignment suppresses
    // the implicitly-generated move-ctor/assignment, which would otherwise
    // make std::move(AccumulatorLayer) silently fall back to the copy-ctor
    // above (an rvalue still binds to a const& parameter) instead of a real
    // move -- harmless now that copying is cheap (shared_ptr + a small
    // per-instance accumulator), but declaring these keeps that true even if
    // a future member is added that isn't cheap to copy.
    AccumulatorLayer(AccumulatorLayer &&) noexcept = default;
    AccumulatorLayer &operator=(AccumulatorLayer &&) noexcept = default;

    void reset()
    {
        // On copie les biais dans les deux accumulateurs (perspective Blanche et Noire)
        (*accumulators)[WHITE] = *biases;
        (*accumulators)[BLACK] = *biases;
    }

    // Resets only one perspective's accumulator -- used when a king move
    // only invalidates that perspective's features (see
    // NnueEval::initialize_perspective()), leaving the other perspective's
    // accumulator untouched.
    template <Color perspective>
    void reset_perspective()
    {
        (*accumulators)[perspective] = *biases;
    }

    // Hints the weight row for `feature` into cache before it's actually
    // needed. Meant to be called several iterations ahead of the matching
    // update_feature() call in a caller's loop -- see initialize_perspective/
    // NnueEval::apply_list in nnue_eval.hpp for why "several" (a single
    // iteration of lookahead doesn't give the memory subsystem enough time
    // to complete the fetch before the row is actually read).
    void prefetch(int feature) const
    {
        if (feature < NThreatFeatures)
            cpu::prefetch<std::array<std::int8_t, NNeurons>, false, 0>(&(*threat_weights)[feature]);
        else
            cpu::prefetch<std::array<std::int16_t, NNeurons>, false, 0>(&(*halfka_weights)[feature - NThreatFeatures]);
    }

    template <bool activate, Color perspective>
    void update_feature(int feature)
    {
        // On déréférence d'abord le pointeur (*accumulators)
        auto &acc = (*accumulators)[perspective];

        if (feature < NThreatFeatures)
            apply_row<activate>(acc, (*threat_weights)[feature]);
        else
            apply_row<activate>(acc, (*halfka_weights)[feature - NThreatFeatures]);
    }

    // Mise a jour fusionnee : toutes les lignes retirees puis ajoutees d'un
    // coup sont appliquees bloc par bloc, l'accumulateur restant dans les
    // registres le temps du bloc. Avec update_feature, chaque ligne relisait
    // et reecrivait les NNeurons valeurs de l'accumulateur ; ici il n'est lu
    // et ecrit qu'une fois par coup. Meme arithmetique (sommes int16 dans le
    // meme ordre par lane), donc resultat identique bit a bit.
    // `reset` : on part des biais (rafraichissement apres un coup de roi).
    void swap_table(std::unique_ptr<AccTable> &other)
    {
        accumulators.swap(other);
    }

#if defined(__aarch64__)
    // Variante NEON : les lignes sont d'abord
    // converties en pointeurs et triees par largeur (int8 menaces / int16
    // HalfKA), puis chaque bloc de l'accumulateur applique les quatre listes
    // a la suite. L'ordre des additions change, mais l'arithmetique int16 est
    // modulo 2^16 (associative et commutative) : resultat identique bit a bit.
    static constexpr int MaxSplitRows = 128;

    void fused_split(const std::int16_t *src, std::int16_t *dst, const int *rem, int nrem, const int *add, int nadd) const
    {
        const std::int8_t *r8[MaxSplitRows], *a8[MaxSplitRows];
        const std::int16_t *r16[MaxSplitRows], *a16[MaxSplitRows];
        int nr8 = 0, nr16 = 0, na8 = 0, na16 = 0;
        const std::int8_t *t8 = threat_weights->data()->data();
        const std::int16_t *t16 = halfka_weights->data()->data();
        for (int i = 0; i < nrem; ++i)
        {
            const int f = rem[i];
            if (f < NThreatFeatures)
                r8[nr8++] = t8 + std::size_t(f) * NNeurons;
            else
                r16[nr16++] = t16 + std::size_t(f - NThreatFeatures) * NNeurons;
        }
        for (int i = 0; i < nadd; ++i)
        {
            const int f = add[i];
            if (f < NThreatFeatures)
                a8[na8++] = t8 + std::size_t(f) * NNeurons;
            else
                a16[na16++] = t16 + std::size_t(f - NThreatFeatures) * NNeurons;
        }
        // NEON : 32 registres, donc 16 accumulateurs de 8 lanes (bloc de 128
        // valeurs) ; les lignes int8 se chargent par 16 octets et s'elargissent
        // avec saddw/saddw2 (ssubw/ssubw2) -- deux fois moins de chargements
        // que la version generique (ldr d + saddw par 8 octets).
        constexpr int U = 16;
        constexpr int Block = 8 * U;
        for (int base = 0; base < NNeurons; base += Block)
        {
            int16x8_t a[U];
            for (int u = 0; u < U; ++u)
                a[u] = vld1q_s16(src + base + 8 * u);
            for (int i = 0; i < nr8; ++i)
                for (int u = 0; u < U; u += 2)
                {
                    const int8x16_t w = vld1q_s8(r8[i] + base + 8 * u);
                    a[u] = vsubw_s8(a[u], vget_low_s8(w));
                    a[u + 1] = vsubw_high_s8(a[u + 1], w);
                }
            for (int i = 0; i < nr16; ++i)
                for (int u = 0; u < U; ++u)
                    a[u] = vsubq_s16(a[u], vld1q_s16(r16[i] + base + 8 * u));
            for (int i = 0; i < na8; ++i)
                for (int u = 0; u < U; u += 2)
                {
                    const int8x16_t w = vld1q_s8(a8[i] + base + 8 * u);
                    a[u] = vaddw_s8(a[u], vget_low_s8(w));
                    a[u + 1] = vaddw_high_s8(a[u + 1], w);
                }
            for (int i = 0; i < na16; ++i)
                for (int u = 0; u < U; ++u)
                    a[u] = vaddq_s16(a[u], vld1q_s16(a16[i] + base + 8 * u));
            for (int u = 0; u < U; ++u)
                vst1q_s16(dst + base + 8 * u, a[u]);
        }
    }
#endif

    template <Color perspective>
    void update_features_fused(const int *rem, int nrem, const int *add, int nadd, bool reset)
    {
        update_features_fused_from<perspective>(*accumulators, rem, nrem, add, nadd, reset);
    }

    // Variante lue depuis un autre tableau (l'ancien etat, reste dans la pile
    // de tampons) et ecrite dans le tampon courant.
    template <Color perspective>
    void update_features_fused_from(const AccTable &src_table, const int *rem, int nrem, const int *add, int nadd, bool reset)
    {
        const std::int16_t *src = reset ? biases->data() : src_table[perspective].data();
        apply_rows(src, (*accumulators)[perspective].data(), rem, nrem, add, nadd);
    }

    // Biais, et accumulateur courant d'une perspective en ecriture (pour le
    // cache de rafraichissement de NnueEval).
    const std::int16_t *bias_data() const { return biases->data(); }
    template <Color perspective>
    std::int16_t *acc_data() { return (*accumulators)[perspective].data(); }

    // dst = src - lignes rem + lignes add (dst peut etre src).
    void apply_rows(const std::int16_t *src, std::int16_t *dst, const int *rem, int nrem, const int *add, int nadd) const
    {
        constexpr std::size_t W = simd::SimdSize16;
        constexpr std::size_t U = 8; // vecteurs par bloc
        constexpr std::size_t Block = W * U;
        static_assert(NNeurons % Block == 0, "bloc doit diviser NNeurons");
        using int8_n = stdx::fixed_size_simd<std::int8_t, W>;

#if defined(__aarch64__)
        if constexpr (NNeurons % 128 == 0)
        {
            if (nrem <= MaxSplitRows && nadd <= MaxSplitRows)
            {
                fused_split(src, dst, rem, nrem, add, nadd);
                return;
            }
        }
#endif
        for (std::size_t base = 0; base < static_cast<std::size_t>(NNeurons); base += Block)
        {
            simd::int16_v a[U];
            for (std::size_t u = 0; u < U; ++u)
                a[u].copy_from(src + base + u * W, stdx::element_aligned);
            auto apply = [&](int feature, bool activate)
            {
                if (feature < NThreatFeatures)
                {
                    const std::int8_t *row = (*threat_weights)[feature].data() + base;
                    for (std::size_t u = 0; u < U; ++u)
                    {
                        int8_n r;
                        r.copy_from(row + u * W, stdx::element_aligned);
                        const simd::int16_v w = stdx::simd_cast<simd::int16_v>(r);
                        if (activate)
                            a[u] += w;
                        else
                            a[u] -= w;
                    }
                }
                else
                {
                    const std::int16_t *row = (*halfka_weights)[feature - NThreatFeatures].data() + base;
                    for (std::size_t u = 0; u < U; ++u)
                    {
                        simd::int16_v w;
                        w.copy_from(row + u * W, stdx::element_aligned);
                        if (activate)
                            a[u] += w;
                        else
                            a[u] -= w;
                    }
                }
            };
            for (int i = 0; i < nrem; ++i)
                apply(rem[i], false);
            for (int i = 0; i < nadd; ++i)
                apply(add[i], true);
            for (std::size_t u = 0; u < U; ++u)
                a[u].copy_to(dst + base + u * W, stdx::element_aligned);
        }
    }

    const auto &get_accumulator(int perspective) const
    {
        return (*accumulators)[perspective];
    }

    template <Color perspective>
    const auto &get_accumulator() const
    {
        return (*accumulators)[perspective];
    }

    // Raw access to both perspectives' state, for saving/restoring a full
    // snapshot (see NnueEval::push_state/pop_state) -- much cheaper than
    // replaying the incremental feature updates when unwinding a move.
    const AccTable &raw_state() const
    {
        return *accumulators;
    }
    void restore_raw_state(const AccTable &saved)
    {
        *accumulators = saved;
    }
};