#pragma once

// Plain C ABI, loaded via ctypes on the Python side — same pattern
// nnue-pytorch uses for SparseBatch/FenBatch (see
// training/nnue-pytorch/data_loader/cpp/training_data_loader_abi.h),
// deliberately not pybind11 so the Python wrapper here can stay a
// thin ctypes.Structure mirror with no extra build dependency.

#if defined(_WIN32) || defined(_WIN64)
#define PLANE_API extern "C" __declspec(dllexport)
#else
#define PLANE_API extern "C" __attribute__((visibility("default")))
#endif

extern "C" {

struct PlaneBatchCStream;

struct PlaneBatchCView {
    int size;
    float* planes;
    float* score;
    float* result;
    int* piece_count;
    float* nnue_score;
    void* handle;  // opaque PlaneBatch*, owned until destroy_plane_batch()
};

// val_percent/is_validation: see PlaneBatchStream's constructor doc in
// plane_batch_stream.h — lets two streams built from the SAME files
// yield disjoint, reproducible train/validation splits without
// physically cutting the binpack file.
//
// nnue_path: path to the .nnue weight file used to compute nnue_score
// per position (see plane_batch.h's PlaneBatch::nnue_score) — chess26's
// own NNUE, evaluated once per position, non-incrementally.
PLANE_API PlaneBatchCStream* create_plane_batch_stream(int concurrency,
                                                         int num_files,
                                                         const char* const* filenames,
                                                         int batch_size,
                                                         bool cyclic,
                                                         int val_percent,
                                                         bool is_validation,
                                                         const char* nnue_path);

PLANE_API void destroy_plane_batch_stream(PlaneBatchCStream* stream);

// Returns a view with size == 0 (planes/score/result == nullptr) once
// the stream is exhausted and not cyclic. Caller must call
// destroy_plane_batch(view) exactly once per non-empty view returned.
PLANE_API PlaneBatchCView fetch_next_plane_batch(PlaneBatchCStream* stream);

PLANE_API void destroy_plane_batch(PlaneBatchCView view);

}  // extern "C"
