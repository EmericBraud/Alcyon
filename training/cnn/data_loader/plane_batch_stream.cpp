#include "plane_batch_stream.h"

#include <functional>

#include "plane_batch.h"

namespace chess26::cnn {

namespace {

std::function<bool(const binpack::TrainingDataEntry&)> make_split_predicate(int val_percent,
                                                                             bool is_validation) {
    if (val_percent <= 0) {
        return [](const binpack::TrainingDataEntry&) { return false; };  // no split, keep everything
    }
    return [val_percent, is_validation](const binpack::TrainingDataEntry& e) {
        const bool in_validation_split = (hash_position(e.pos) % 100) < static_cast<std::uint64_t>(val_percent);
        // Skip the entry unless it belongs to the split this stream was built for.
        return in_validation_split != is_validation;
    };
}

}  // namespace

PlaneBatchStream::PlaneBatchStream(int concurrency,
                                    const std::vector<std::string>& filenames,
                                    int batch_size,
                                    bool cyclic,
                                    int val_percent,
                                    bool is_validation)
    : m_batch_size(batch_size),
      m_stream(training_data::open_sfen_input_file_parallel(
          concurrency,
          filenames,
          cyclic,
          make_split_predicate(val_percent, is_validation),
          /*rank=*/0,
          /*world_size=*/1)) {}

PlaneBatch* PlaneBatchStream::next() {
    std::vector<binpack::TrainingDataEntry> entries;
    entries.reserve(m_batch_size);

    for (int i = 0; i < m_batch_size; ++i) {
        auto entry = m_stream->next();
        if (!entry.has_value()) break;
        entries.push_back(*entry);
    }

    if (entries.empty()) return nullptr;
    return new PlaneBatch(entries);
}

}  // namespace chess26::cnn
