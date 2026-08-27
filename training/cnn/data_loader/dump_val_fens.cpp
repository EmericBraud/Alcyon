// One-off utility: dumps FEN + game result for a sample of the
// validation split (same hash-based split used by training), so
// NNUE (which only speaks FEN via its UCI "eval" command) can be
// compared against real game outcomes the same way the CNN already
// is via PlaneBatchDataset's tensors directly.
//
// Usage: dump_val_fens <binpack> <val_percent> <n_positions> <out.tsv>
#include <fstream>
#include <iostream>
#include <string>

#include "nnue_training_data_stream.h"
#include "plane_batch.h"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: dump_val_fens <binpack> <val_percent> <n_positions> <out.tsv>\n";
        return 1;
    }
    const std::string binpack = argv[1];
    const int val_percent = std::stoi(argv[2]);
    const int n_positions = std::stoi(argv[3]);
    const std::string out_path = argv[4];

    auto predicate = [val_percent](const binpack::TrainingDataEntry& e) {
        const bool in_validation_split =
            (chess26::cnn::hash_position(e.pos) % 100) < static_cast<std::uint64_t>(val_percent);
        return !in_validation_split;  // skip unless it's in the validation split
    };

    auto stream = training_data::open_sfen_input_file_parallel(
        /*concurrency=*/4, std::vector<std::string>{binpack}, /*cyclic=*/false, predicate,
        /*rank=*/0, /*world_size=*/1);

    std::ofstream out(out_path);
    int count = 0;
    while (count < n_positions) {
        auto entry = stream->next();
        if (!entry.has_value()) break;
        out << entry->pos.fen() << '\t' << entry->result << '\t' << entry->score << '\n';
        ++count;
    }
    std::cerr << "wrote " << count << " positions to " << out_path << '\n';
    return 0;
}
