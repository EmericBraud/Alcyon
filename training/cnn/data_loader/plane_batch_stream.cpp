#include "plane_batch_stream.h"

namespace chess26::cnn {

PlaneBatchStream::PlaneBatchStream(int concurrency,
                                    const std::vector<std::string>& filenames,
                                    int batch_size,
                                    bool cyclic)
    : m_batch_size(batch_size),
      m_stream(training_data::open_sfen_input_file_parallel(
          concurrency,
          filenames,
          cyclic,
          [](const binpack::TrainingDataEntry&) { return false; },  // no skip
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
