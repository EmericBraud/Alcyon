#pragma once

#ifdef TEXEL_TUNING

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "common/file.hpp"
#include "common/logger.hpp"
#include "engine/config/eval.hpp"
#include "engine/eval/tuning/eval_features.hpp"
#include "engine/eval/virtual_board.hpp"

struct TexelSample
{
    std::string fen;
    double result = 0.5;
};

struct TexelTrainingSample
{
    double result = 0.5;
    double base_mg = 0.0;
    double base_eg = 0.0;
    int phase = 0;
    EvalFeatures features;
};

template <std::size_t N>
void dump_array(std::ofstream &out, const char *name, const TunableParam (&arr)[N])
{
    out << name << " = { ";
    for (std::size_t i = 0; i < N; ++i)
    {
        if (i)
            out << ", ";
        out << arr[i].value;
    }
    out << " }\n";
}
struct TexelGradients
{
    double doubledFilesMgMalus = 0.0;
    double doubledFilesEgMalus = 0.0;

    double isolatedFilesMgMalus = 0.0;
    double isolatedFilesEgMalus = 0.0;

    double openFileMalus = 0.0;
    double semiOpenFileMalus = 0.0;

    double heavyEnemiesOpenFileMalus = 0.0;
    double heavyEnemiesSemiOpenFileMalus = 0.0;

    double bishopPairMgBonus = 0.0;
    double bishopPairEgBonus = 0.0;

    double kingDistFromCenterBonus = 0.0;
    double closeKingBonus = 0.0;

    std::array<double, 8> passed_bonus_mg{};
    std::array<double, 8> passed_bonus_eg{};

    std::array<double, 9> knight_mob{};
    std::array<double, 14> bishop_mob{};
    std::array<double, 15> rook_mob{};
    std::array<double, 28> queen_mob{};

    std::array<double, constants::PieceTypeCount> pieces_score{};
};

static std::optional<TexelSample> parse_quiet_labeled_line(const std::string &line)
{
    std::istringstream iss(line);

    std::string board;
    std::string stm;
    std::string castling;
    std::string ep;
    std::string tag;
    std::string result_token;

    if (!(iss >> board >> stm >> castling >> ep >> tag >> result_token))
        return std::nullopt;

    if (tag != "c9")
        return std::nullopt;

    if (!result_token.empty() && result_token.back() == ';')
        result_token.pop_back();

    double result = 0.5;

    if (result_token == "\"1-0\"")
        result = 1.0;
    else if (result_token == "\"1/2-1/2\"")
        result = 0.5;
    else if (result_token == "\"0-1\"")
        result = 0.0;
    else
        return std::nullopt;

    std::string fen = board + " " + stm + " " + castling + " " + ep + " 0 1";
    return TexelSample{fen, result};
}

class TexelTuner
{
    static constexpr int lines_nb = 1428000;
    static constexpr int percent = lines_nb / 100;
    static constexpr double texelK = 1.2755;

    static constexpr int epochs = 1000;
    static constexpr std::size_t batch_size = 4096;

    double learning_rate = 10000.0;

    double sigmoid(double s) const
    {
        return 1.0 / (1.0 + std::pow(10.0, -s / 400.0));
    }

    double sigmoid_derivative_from_p(double p) const
    {
        return (std::log(10.0) * texelK / 400.0) * p * (1.0 - p);
    }

    double score_sample(const TexelTrainingSample &s) const
    {
        using namespace engine_constants::eval;

        const EvalFeatures &f = s.features;

        double mg = s.base_mg;
        double eg = s.base_eg;

        for (int i = PAWN; i <= QUEEN; ++i)
        {
            mg += f.material[i] * pieces_score[i];
            eg += f.material[i] * pieces_score[i];
        }

        mg += f.doubled_files * doubledFilesMgMalus;
        eg += f.doubled_files * doubledFilesEgMalus;

        mg += f.isolated_files * isolatedFilesMgMalus;
        eg += f.isolated_files * isolatedFilesEgMalus;

        mg += f.open_files_near_king * openFileMalus;
        mg += f.semi_open_files_near_king * semiOpenFileMalus;
        mg += f.heavy_on_open * heavyEnemiesOpenFileMalus;
        mg += f.heavy_on_semi_open * heavyEnemiesSemiOpenFileMalus;

        mg += f.bishop_pair_mg * bishopPairMgBonus;
        eg += f.bishop_pair_eg * bishopPairEgBonus;

        eg += f.king_dist_center * kingDistFromCenterBonus;
        eg += f.king_closeness * closeKingBonus;

        for (int i = 0; i < 8; ++i)
        {
            mg += f.passed_mg[i] * passed_bonus_mg[i];
            eg += f.passed_eg[i] * passed_bonus_eg[i];
        }

        for (int i = 0; i < 9; ++i)
        {
            mg += f.knight_mob[i] * knight_mob[i];
            eg += f.knight_mob[i] * knight_mob[i];
        }

        for (int i = 0; i < 14; ++i)
        {
            mg += f.bishop_mob[i] * bishop_mob[i];
            eg += f.bishop_mob[i] * bishop_mob[i];
        }

        for (int i = 0; i < 15; ++i)
        {
            mg += f.rook_mob[i] * rook_mob[i];
            eg += f.rook_mob[i] * rook_mob[i];
        }

        for (int i = 0; i < 28; ++i)
        {
            mg += f.queen_mob[i] * queen_mob[i];
            eg += f.queen_mob[i] * queen_mob[i];
        }

        return (mg * s.phase + eg * (totalPhase - s.phase)) / totalPhase;
    }

    TexelTrainingSample make_training_sample(const TexelSample &sample) const
    {
        VBoard board;
        board.load_fen(sample.fen);

        const EvalState &state = board.get_eval_state();

        const double base_mg =
            state.mg_pst[WHITE] - state.mg_pst[BLACK];

        const double base_eg =
            state.eg_pst[WHITE] - state.eg_pst[BLACK];

        EvalFeatures features = Eval::extract_eval_features(board);

        return TexelTrainingSample{
            sample.result,
            base_mg,
            base_eg,
            state.phase,
            features};
    }

    void save_params(const std::string &path, int epoch, double train_loss, double valid_loss)
    {
        const std::filesystem::path output_path(path);
        const auto parent = output_path.parent_path();

        if (!parent.empty())
            std::filesystem::create_directories(parent);

        std::ofstream out(output_path);

        if (!out.is_open())
            throw std::runtime_error("Cannot open params output file: " + path);

        using namespace engine_constants::eval;

        out << "// Texel tuning checkpoint\n";
        out << "// epoch = " << epoch << "\n";
        out << "// train_loss = " << train_loss << "\n";
        out << "// valid_loss = " << valid_loss << "\n\n";

        out << "openFileMalus = " << openFileMalus.value << "\n";
        out << "semiOpenFileMalus = " << semiOpenFileMalus.value << "\n";
        out << "heavyEnemiesOpenFileMalus = " << heavyEnemiesOpenFileMalus.value << "\n";
        out << "heavyEnemiesSemiOpenFileMalus = " << heavyEnemiesSemiOpenFileMalus.value << "\n";

        out << "bishopPairMgBonus = " << bishopPairMgBonus.value << "\n";
        out << "bishopPairEgBonus = " << bishopPairEgBonus.value << "\n";

        out << "kingDistFromCenterBonus = " << kingDistFromCenterBonus.value << "\n";
        out << "closeKingBonus = " << closeKingBonus.value << "\n";

        out << "doubledFilesMgMalus = " << doubledFilesMgMalus.value << "\n";
        out << "doubledFilesEgMalus = " << doubledFilesEgMalus.value << "\n";
        out << "isolatedFilesMgMalus = " << isolatedFilesMgMalus.value << "\n";
        out << "isolatedFilesEgMalus = " << isolatedFilesEgMalus.value << "\n\n";

        dump_array(out, "passed_bonus_mg", passed_bonus_mg);
        dump_array(out, "passed_bonus_eg", passed_bonus_eg);

        dump_array(out, "knight_mob", knight_mob);
        dump_array(out, "bishop_mob", bishop_mob);
        dump_array(out, "rook_mob", rook_mob);
        dump_array(out, "queen_mob", queen_mob);

        out << "pieces_score = { ";
        for (int i = 0; i < constants::PieceTypeCount; ++i)
        {
            if (i)
                out << ", ";
            out << pieces_score[i].value;
        }
        out << " }\n";
    }

    std::vector<TexelSample> load_raw_samples()
    {
        std::ifstream epd_file(file::get_data_path("tuning_epd/quiet-labeled.v7.epd"));

        if (!epd_file.is_open())
            throw std::runtime_error("Cannot open tuning file");

        std::vector<TexelSample> raw;
        raw.reserve(lines_nb);

        std::string line;
        int parsed = 0;

        while (std::getline(epd_file, line))
        {
            auto sample = parse_quiet_labeled_line(line);

            if (!sample.has_value())
            {
                logs::error << "[ERROR] couldn't parse line" << std::endl;
                continue;
            }

            raw.push_back(std::move(*sample));
            ++parsed;

            if (percent > 0 && parsed % percent == 0)
            {
                logs::uci << "[INFO] Loaded raw "
                          << parsed / percent
                          << "% of dataset"
                          << std::endl;
            }
        }

        return raw;
    }

    std::vector<TexelTrainingSample> build_training_samples_parallel(const std::vector<TexelSample> &raw)
    {
        std::vector<TexelTrainingSample> samples(raw.size());

        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const unsigned thread_count = std::min<unsigned>(hw, 8);

        std::atomic<std::size_t> next{0};
        std::atomic<int> done{0};

        auto worker = [&]()
        {
            while (true)
            {
                const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);

                if (i >= raw.size())
                    break;

                samples[i] = make_training_sample(raw[i]);

                const int current_done = ++done;
                if (percent > 0 && current_done % percent == 0)
                {
                    logs::uci << "[INFO] Processed features "
                              << current_done / percent
                              << "% of dataset"
                              << std::endl;
                }
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(thread_count);

        for (unsigned i = 0; i < thread_count; ++i)
            threads.emplace_back(worker);

        for (auto &t : threads)
            t.join();

        return samples;
    }

    static void add_gradients(TexelGradients &a, const TexelGradients &b)
    {
        a.doubledFilesMgMalus += b.doubledFilesMgMalus;
        a.doubledFilesEgMalus += b.doubledFilesEgMalus;

        a.isolatedFilesMgMalus += b.isolatedFilesMgMalus;
        a.isolatedFilesEgMalus += b.isolatedFilesEgMalus;

        a.openFileMalus += b.openFileMalus;
        a.semiOpenFileMalus += b.semiOpenFileMalus;

        a.heavyEnemiesOpenFileMalus += b.heavyEnemiesOpenFileMalus;
        a.heavyEnemiesSemiOpenFileMalus += b.heavyEnemiesSemiOpenFileMalus;

        a.bishopPairMgBonus += b.bishopPairMgBonus;
        a.bishopPairEgBonus += b.bishopPairEgBonus;

        a.kingDistFromCenterBonus += b.kingDistFromCenterBonus;
        a.closeKingBonus += b.closeKingBonus;

        for (int i = 0; i < 8; ++i)
        {
            a.passed_bonus_mg[i] += b.passed_bonus_mg[i];
            a.passed_bonus_eg[i] += b.passed_bonus_eg[i];
        }

        for (int i = 0; i < 9; ++i)
            a.knight_mob[i] += b.knight_mob[i];

        for (int i = 0; i < 14; ++i)
            a.bishop_mob[i] += b.bishop_mob[i];

        for (int i = 0; i < 15; ++i)
            a.rook_mob[i] += b.rook_mob[i];

        for (int i = 0; i < 28; ++i)
            a.queen_mob[i] += b.queen_mob[i];

        for (int i = PAWN; i <= QUEEN; ++i)
        {
            a.pieces_score[i] += b.pieces_score[i];
        }
    }

    void accumulate_gradient_for_sample(const TexelTrainingSample &s, TexelGradients &g, double &loss) const
    {
        const double eval = score_sample(s);
        const double p = sigmoid(texelK * eval);
        const double error = p - s.result;

        loss += error * error;

        const double common = 2.0 * error * sigmoid_derivative_from_p(p);

        const double mg_scale = static_cast<double>(s.phase) / engine_constants::eval::totalPhase;
        const double eg_scale = static_cast<double>(engine_constants::eval::totalPhase - s.phase) / engine_constants::eval::totalPhase;

        const EvalFeatures &f = s.features;

        g.doubledFilesMgMalus += common * f.doubled_files * mg_scale;
        g.doubledFilesEgMalus += common * f.doubled_files * eg_scale;

        g.isolatedFilesMgMalus += common * f.isolated_files * mg_scale;
        g.isolatedFilesEgMalus += common * f.isolated_files * eg_scale;

        g.openFileMalus += common * f.open_files_near_king * mg_scale;
        g.semiOpenFileMalus += common * f.semi_open_files_near_king * mg_scale;

        g.heavyEnemiesOpenFileMalus += common * f.heavy_on_open * mg_scale;
        g.heavyEnemiesSemiOpenFileMalus += common * f.heavy_on_semi_open * mg_scale;

        g.bishopPairMgBonus += common * f.bishop_pair_mg * mg_scale;
        g.bishopPairEgBonus += common * f.bishop_pair_eg * eg_scale;

        g.kingDistFromCenterBonus += common * f.king_dist_center * eg_scale;
        g.closeKingBonus += common * f.king_closeness * eg_scale;

        for (int i = 0; i < 8; ++i)
        {
            g.passed_bonus_mg[i] += common * f.passed_mg[i] * mg_scale;
            g.passed_bonus_eg[i] += common * f.passed_eg[i] * eg_scale;
        }

        const double mob_scale = mg_scale + eg_scale;

        for (int i = 0; i < 9; ++i)
            g.knight_mob[i] += common * f.knight_mob[i] * mob_scale;

        for (int i = 0; i < 14; ++i)
            g.bishop_mob[i] += common * f.bishop_mob[i] * mob_scale;

        for (int i = 0; i < 15; ++i)
            g.rook_mob[i] += common * f.rook_mob[i] * mob_scale;

        for (int i = 0; i < 28; ++i)
            g.queen_mob[i] += common * f.queen_mob[i] * mob_scale;

        for (int i = PAWN; i <= QUEEN; ++i)
        {
            g.pieces_score[i] += common * f.material[i]; // mg_scale + eg_scale = 1
        }
    }

    void apply_gradients(const TexelGradients &g, double scale)
    {
        using namespace engine_constants::eval;

        doubledFilesMgMalus.value -= scale * g.doubledFilesMgMalus;
        doubledFilesEgMalus.value -= scale * g.doubledFilesEgMalus;

        isolatedFilesMgMalus.value -= scale * g.isolatedFilesMgMalus;
        isolatedFilesEgMalus.value -= scale * g.isolatedFilesEgMalus;

        openFileMalus.value -= scale * g.openFileMalus;
        semiOpenFileMalus.value -= scale * g.semiOpenFileMalus;

        heavyEnemiesOpenFileMalus.value -= scale * g.heavyEnemiesOpenFileMalus;
        heavyEnemiesSemiOpenFileMalus.value -= scale * g.heavyEnemiesSemiOpenFileMalus;

        bishopPairMgBonus.value -= scale * g.bishopPairMgBonus;
        bishopPairEgBonus.value -= scale * g.bishopPairEgBonus;

        kingDistFromCenterBonus.value -= scale * g.kingDistFromCenterBonus;
        closeKingBonus.value -= scale * g.closeKingBonus;

        for (int i = 0; i < 8; ++i)
        {
            passed_bonus_mg[i].value -= scale * g.passed_bonus_mg[i];
            passed_bonus_eg[i].value -= scale * g.passed_bonus_eg[i];
        }

        for (int i = 0; i < 9; ++i)
            knight_mob[i].value -= scale * g.knight_mob[i];

        for (int i = 0; i < 14; ++i)
            bishop_mob[i].value -= scale * g.bishop_mob[i];

        for (int i = 0; i < 15; ++i)
            rook_mob[i].value -= scale * g.rook_mob[i];

        for (int i = 0; i < 28; ++i)
            queen_mob[i].value -= scale * g.queen_mob[i];

        for (int i = PAWN; i <= QUEEN; ++i)
        {
            pieces_score[i].value -= scale * g.pieces_score[i];
        }
    }

    double compute_loss_parallel(const std::vector<TexelTrainingSample> &samples, std::size_t begin, std::size_t end) const
    {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const unsigned thread_count = std::min<unsigned>(hw, 8);

        std::vector<double> partial(thread_count, 0.0);
        std::vector<std::thread> threads;

        const std::size_t n = end - begin;
        const std::size_t chunk = (n + thread_count - 1) / thread_count;

        for (unsigned t = 0; t < thread_count; ++t)
        {
            const std::size_t b = begin + t * chunk;
            const std::size_t e = std::min(end, b + chunk);

            threads.emplace_back([&, t, b, e]()
                                 {
                double local = 0.0;

                for (std::size_t i = b; i < e; ++i)
                {
                    const double eval = score_sample(samples[i]);
                    const double p = sigmoid(texelK * eval);
                    const double d = samples[i].result - p;
                    local += d * d;
                }

                partial[t] = local; });
        }

        for (auto &th : threads)
            th.join();

        const double sum = std::accumulate(partial.begin(), partial.end(), 0.0);
        return sum / static_cast<double>(n);
    }

    TexelGradients compute_batch_gradient_parallel(
        const std::vector<TexelTrainingSample> &samples,
        std::size_t begin,
        std::size_t end,
        double &batch_loss) const
    {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const unsigned thread_count = std::min<unsigned>(hw, 8);

        std::vector<TexelGradients> partial_gradients(thread_count);
        std::vector<double> partial_losses(thread_count, 0.0);
        std::vector<std::thread> threads;

        const std::size_t n = end - begin;
        const std::size_t chunk = (n + thread_count - 1) / thread_count;

        for (unsigned t = 0; t < thread_count; ++t)
        {
            const std::size_t b = begin + t * chunk;
            const std::size_t e = std::min(end, b + chunk);

            threads.emplace_back([&, t, b, e]()
                                 {
                TexelGradients local_g{};
                double local_loss = 0.0;

                for (std::size_t i = b; i < e; ++i)
                    accumulate_gradient_for_sample(samples[i], local_g, local_loss);

                partial_gradients[t] = local_g;
                partial_losses[t] = local_loss; });
        }

        for (auto &th : threads)
            th.join();

        TexelGradients total{};

        for (unsigned t = 0; t < thread_count; ++t)
            add_gradients(total, partial_gradients[t]);

        batch_loss = std::accumulate(partial_losses.begin(), partial_losses.end(), 0.0);
        return total;
    }

    void train()
    {
        logs::uci << "[INFO] Loading raw samples..." << std::endl;
        std::vector<TexelSample> raw = load_raw_samples();

        logs::uci << "[INFO] Building compact training samples..." << std::endl;
        std::vector<TexelTrainingSample> samples = build_training_samples_parallel(raw);

        raw.clear();
        raw.shrink_to_fit();

        if (samples.empty())
            throw std::runtime_error("No training samples loaded");

        std::mt19937 rng(42);
        std::shuffle(samples.begin(), samples.end(), rng);

        const std::size_t train_size = static_cast<std::size_t>(samples.size() * 0.8);
        const std::size_t valid_begin = train_size;
        const std::size_t valid_end = samples.size();

        logs::uci << "[INFO] Training samples: "
                  << train_size
                  << " | Validation samples: "
                  << valid_end - valid_begin
                  << std::endl;

        const double initial_train_loss = compute_loss_parallel(samples, 0, train_size);
        const double initial_valid_loss = compute_loss_parallel(samples, valid_begin, valid_end);

        logs::uci << "[INFO] Initial train loss = "
                  << initial_train_loss
                  << " | Initial valid loss = "
                  << initial_valid_loss
                  << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            std::shuffle(samples.begin(), samples.begin() + train_size, rng);

            double epoch_loss_sum = 0.0;
            std::size_t seen = 0;

            for (std::size_t start = 0; start < train_size; start += batch_size)
            {
                const std::size_t end = std::min(start + batch_size, train_size);

                double batch_loss = 0.0;
                TexelGradients g = compute_batch_gradient_parallel(samples, start, end, batch_loss);

                const double scale = learning_rate / static_cast<double>(end - start);
                apply_gradients(g, scale);

                epoch_loss_sum += batch_loss;
                seen += end - start;
            }

            const double train_loss = epoch_loss_sum / static_cast<double>(seen);
            const double valid_loss = compute_loss_parallel(samples, valid_begin, valid_end);

            logs::uci << "[INFO] Epoch "
                      << epoch + 1
                      << " | train loss = "
                      << train_loss
                      << " | valid loss = "
                      << valid_loss
                      << std::endl;

            save_params(
                file::get_data_path("tuning_output/texel_params_epoch_" + std::to_string(epoch + 1) + ".txt"),
                epoch + 1,
                train_loss,
                valid_loss);
        }
    }

public:
    void start_tuning()
    {
        train();
        logs::uci << "[INFO] Task done !" << std::endl;
    }
};

#endif