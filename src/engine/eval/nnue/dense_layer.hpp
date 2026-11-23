// dense_layer.hpp
#pragma once

#include <array>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif
#include <bit>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <utility>

#include "common/aligned_array.hpp"
#include "dot_product.hpp"

// WeightScaleBits_: hidden-layer weights are quantized with scale 2^WeightScaleBits_
// (64 by default, matching nnue-pytorch/Stockfish quantization). Set to 0 for
// unquantized/identity test weights.
//
// Inputs are uint8_t activations in [0, 127] (every activation feeding a
// dense layer -- pairwise-square, squared-CReLU, ClippedReLU -- clamps to
// that range); weights are int8_t. That contract is what lets the inner
// loops run on nnue::dot::dot_u8_i8's unsigned x signed fast paths (see
// dot_product.hpp).
template <int NInputs_, int NNeurons_, int WeightScaleBits_ = 6>
class DenseLayer
{
public:
    static constexpr int NInputs = NInputs_;
    static constexpr int NNeurons = NNeurons_;

private:
    // weights/biases are read-only after construction and identical across
    // every thread's copy of this layer (one per layer-stack bucket, shared
    // by every VBoard/SearchWorker) -- shared_ptr<const T> means copying a
    // DenseLayer (e.g. via VBoard's copy ctor, done once per search thread)
    // just bumps a refcount instead of duplicating the underlying arrays.
    using WeightTable = AlignedArray<std::array<std::int8_t, NInputs_>, NNeurons_>;
    using BiasTable = AlignedArray<std::int32_t, NNeurons_>;

    std::shared_ptr<const WeightTable> weights;
    std::shared_ptr<const BiasTable> biases;
    // Poids transposes [entree][neurone], pour la version creuse de get_raw
    // (on ne parcourt que les entrees non nulles).
    // Poids regroupes pour get_raw_sparse : [entree/4][neurone][4].
    using GroupedTable = AlignedArray<std::int8_t, NInputs_ * NNeurons_>;
    std::shared_ptr<const GroupedTable> weights_g;

    std::uint8_t relu(std::int32_t acc) const;

public:
    void process(
        const std::array<std::uint8_t, NInputs_> &inputs,
        std::array<std::uint8_t, NNeurons_> &output) const;

    std::int32_t get_result(const std::array<std::uint8_t, NInputs_> &inputs) const
        requires(NNeurons_ == 1);

    template <std::size_t NInputsA, std::size_t NInputsB>
    std::int32_t get_result_split(
        const std::array<std::uint8_t, NInputsA> &inputs_a,
        const std::array<std::uint8_t, NInputsB> &inputs_b) const
        requires(NNeurons_ == 1 && NInputsA + NInputsB == static_cast<std::size_t>(NInputs_))
    {
        const std::int8_t *row = (*weights)[0].data();
        return (*biases)[0] +
               nnue::dot::dot_u8_i8<static_cast<int>(NInputsA)>(inputs_a.data(), row) +
               nnue::dot::dot_u8_i8<static_cast<int>(NInputsB)>(inputs_b.data(), row + NInputsA);
    }

    // Raw (unshifted, unclamped) bias + dot-product per neuron. Used by layers
    // whose activation isn't a plain ClippedReLU (e.g. the layer-stack L1,
    // which feeds a squared-CReLU and has an extra skip-connection output).
    std::array<std::int32_t, NNeurons_> get_raw(
        const std::array<std::uint8_t, NInputs_> &inputs) const;

    // Meme resultat que get_raw (sommes entieres : l'ordre ne change rien),
    // mais ne traite que les entrees non nulles -- l'entree de la premiere
    // couche, issue du produit par paires d'activations bornees, est
    // majoritairement nulle.
    std::array<std::int32_t, NNeurons_> get_raw_sparse(
        const std::array<std::uint8_t, NInputs_> &inputs) const
    {
#if defined(__ARM_FEATURE_DOTPROD) && defined(__aarch64__)
        if constexpr (NNeurons_ % 4 == 0 && NInputs_ % 4 == 0)
        {
            // Entrees par groupes de 4 octets : un groupe non nul se traite
            // en NNeurons_/4 instructions sdot (4 entrees x 4 neurones). Les
            // entrees sont <= 127, donc lisibles en int8 signe.
            constexpr int G = NNeurons_ / 4;
            int32x4_t acc[G];
            for (int g = 0; g < G; ++g)
                acc[g] = vld1q_s32(biases->data() + 4 * g);
            const std::int8_t *w4 = weights_g->data();
            // 1) liste des groupes non nuls, sans branchement (ecriture
            //    inconditionnelle, avance conditionnelle de l'indice) ;
            // 2) parcours de la liste : plus aucun saut imprevisible.
            constexpr int NG = NInputs_ / 4;
            std::uint32_t grp[NG];
            std::memcpy(grp, inputs.data(), NInputs_);
            std::uint16_t nz[NG];
            int n = 0;
            for (int j = 0; j < NG; ++j)
            {
                nz[n] = static_cast<std::uint16_t>(j);
                n += grp[j] != 0;
            }
            for (int t = 0; t < n; ++t)
            {
                const int j = nz[t];
                const int8x16_t x = vreinterpretq_s8_u32(vdupq_n_u32(grp[j]));
                const std::int8_t *row = w4 + std::size_t(j) * NNeurons_ * 4;
                for (int g = 0; g < G; ++g)
                    acc[g] = vdotq_s32(acc[g], vld1q_s8(row + 16 * g), x);
            }
            std::array<std::int32_t, NNeurons_> out;
            for (int g = 0; g < G; ++g)
                vst1q_s32(out.data() + 4 * g, acc[g]);
            return out;
        }
#elif defined(__AVX2__)
        if constexpr (NNeurons_ % 8 == 0 && NInputs_ % 4 == 0)
        {
            // Meme principe en AVX2 : un groupe non nul de 4 entrees, diffuse
            // dans un registre, multiplie (maddubs : u8 x i8 -> paires int16,
            // sans saturation car entrees <= 127) puis somme par 4 (madd avec
            // des 1) donne 8 sorties int32 par registre.
            constexpr int G = NNeurons_ / 8;
            __m256i acc[G];
            for (int g = 0; g < G; ++g)
                acc[g] = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(biases->data() + 8 * g));
            const std::int8_t *w4 = weights_g->data();
            const __m256i ones = _mm256_set1_epi16(1);
            constexpr int NG = NInputs_ / 4;
            std::uint32_t grp[NG];
            std::memcpy(grp, inputs.data(), NInputs_);
            std::uint16_t nz[NG];
            int n = 0;
            for (int j = 0; j < NG; ++j)
            {
                nz[n] = static_cast<std::uint16_t>(j);
                n += grp[j] != 0;
            }
            for (int t = 0; t < n; ++t)
            {
                const int j = nz[t];
                const __m256i x = _mm256_set1_epi32(static_cast<int>(grp[j]));
                const std::int8_t *row = w4 + std::size_t(j) * NNeurons_ * 4;
                for (int g = 0; g < G; ++g)
                {
                    const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row + 32 * g));
                    acc[g] = _mm256_add_epi32(acc[g], _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), ones));
                }
            }
            std::array<std::int32_t, NNeurons_> out;
            for (int g = 0; g < G; ++g)
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(out.data() + 8 * g), acc[g]);
            return out;
        }
#endif
        return get_raw(inputs);
    }

    template <typename T, typename U>
    DenseLayer(T &&_weights, U &&_biases)
        : weights(std::make_shared<const WeightTable>(std::forward<T>(_weights))),
          biases(std::make_shared<const BiasTable>(std::forward<U>(_biases)))
    {
        auto t = std::make_shared<GroupedTable>();
        for (int o = 0; o < NNeurons_; ++o)
            for (int i = 0; i < NInputs_; ++i)
                (*t)[(std::size_t(i / 4) * NNeurons_ + o) * 4 + (i % 4)] = (*weights)[o][i];
        weights_g = std::move(t);
    }
};

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline std::uint8_t DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::relu(std::int32_t acc) const
{
    return static_cast<std::uint8_t>(std::clamp(acc >> WeightScaleBits_, 0, 127));
}

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline void DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::process(
    const std::array<std::uint8_t, NInputs_> &inputs,
    std::array<std::uint8_t, NNeurons_> &output) const
{
    for (int j = 0; j < NNeurons_; ++j)
    {
        const std::int32_t acc =
            (*biases)[j] + nnue::dot::dot_u8_i8<NInputs_>(inputs.data(), (*weights)[j].data());
        output[j] = relu(acc);
    }
}

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline std::int32_t DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::get_result(
    const std::array<std::uint8_t, NInputs_> &inputs) const
    requires(NNeurons_ == 1)
{
    return (*biases)[0] + nnue::dot::dot_u8_i8<NInputs_>(inputs.data(), (*weights)[0].data());
}

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline std::array<std::int32_t, NNeurons_> DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::get_raw(
    const std::array<std::uint8_t, NInputs_> &inputs) const
{
    std::array<std::int32_t, NNeurons_> output{};

    for (int j = 0; j < NNeurons_; ++j)
        output[j] = (*biases)[j] + nnue::dot::dot_u8_i8<NInputs_>(inputs.data(), (*weights)[j].data());

    return output;
}
