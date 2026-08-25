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

PlaneBatchStreamImpl::PlaneBatchStreamImpl(int concurrency,
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

PlaneBatch* PlaneBatchStreamImpl::next() {
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

PlaneBatchStream::PlaneBatchStream(int concurrency,
                                    const std::vector<std::string>& filenames,
                                    int batch_size,
                                    bool cyclic,
                                    int val_percent,
                                    bool is_validation,
                                    int queue_capacity)
    : m_impl(concurrency, filenames, batch_size, cyclic, val_percent, is_validation),
      m_capacity(static_cast<std::size_t>(queue_capacity)) {
    m_worker = std::thread(&PlaneBatchStream::worker_loop, this);
}

PlaneBatchStream::~PlaneBatchStream() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_not_full.notify_all();   // wake the worker if it's blocked pushing
    m_not_empty.notify_all();  // wake next() if a caller is still waiting
    if (m_worker.joinable()) m_worker.join();

    for (PlaneBatch* batch : m_queue) delete batch;
}

void PlaneBatchStream::worker_loop() {
    while (true) {
        PlaneBatch* batch = m_impl.next();  // blocking I/O + plane encoding, off the consumer thread

        std::unique_lock<std::mutex> lock(m_mutex);
        m_not_full.wait(lock, [&] { return m_stop || m_queue.size() < m_capacity; });
        if (m_stop) {
            delete batch;  // no-op if batch == nullptr
            return;
        }

        if (batch == nullptr) {
            m_exhausted = true;
            lock.unlock();
            m_not_empty.notify_all();
            return;
        }

        m_queue.push_back(batch);
        lock.unlock();
        m_not_empty.notify_one();
    }
}

PlaneBatch* PlaneBatchStream::next() {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_not_empty.wait(lock, [&] { return m_stop || !m_queue.empty() || m_exhausted; });

    if (!m_queue.empty()) {
        PlaneBatch* batch = m_queue.front();
        m_queue.pop_front();
        lock.unlock();
        m_not_full.notify_one();
        return batch;
    }

    return nullptr;  // m_stop or m_exhausted with an empty queue
}

}  // namespace chess26::cnn
