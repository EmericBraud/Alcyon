import ctypes
import glob
import os

import numpy as np
import torch

# Kept in sync by hand with plane_batch.h / model.py — not imported
# to avoid a fragile cross-package import path.
NUM_PLANES = 19
BOARD_SIZE = 8


def _pin_and_move(t: torch.Tensor, device, use_pinned_memory=False) -> torch.Tensor:
    # Must copy off PlaneBatch-backed memory before destroy_plane_batch() frees it.
    if torch.cuda.is_available() and use_pinned_memory:
        out = torch.empty(t.shape, dtype=t.dtype, layout=t.layout, device="cpu", pin_memory=True)
        out.copy_(t)
        if device == "cpu" or (isinstance(device, torch.device) and device.type == "cpu"):
            return out
        return out.to(device=device, non_blocking=True)

    out = torch.empty(t.shape, dtype=t.dtype, layout=t.layout, device="cpu")
    out.copy_(t)
    if device == "cpu" or (isinstance(device, torch.device) and device.type == "cpu"):
        return out
    return out.to(device=device)


class PlaneBatchCView(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("planes", ctypes.POINTER(ctypes.c_float)),
        ("score", ctypes.POINTER(ctypes.c_float)),
        ("result", ctypes.POINTER(ctypes.c_float)),
        ("piece_count", ctypes.POINTER(ctypes.c_int)),
        ("handle", ctypes.c_void_p),
    ]

    def get_tensors(self, device, use_pinned_memory=False):
        size = self.size

        plane_size = BOARD_SIZE * BOARD_SIZE
        planes_cpu = torch.from_numpy(
            np.ctypeslib.as_array(self.planes, shape=(size * NUM_PLANES * plane_size,))
        )
        score_cpu = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(size,)))
        result_cpu = torch.from_numpy(np.ctypeslib.as_array(self.result, shape=(size,)))
        piece_count_cpu = torch.from_numpy(
            np.ctypeslib.as_array(self.piece_count, shape=(size,))
        )

        planes = _pin_and_move(planes_cpu, device, use_pinned_memory).view(
            size, NUM_PLANES, BOARD_SIZE, BOARD_SIZE
        )
        score = _pin_and_move(score_cpu, device, use_pinned_memory).view(size, 1)
        result = _pin_and_move(result_cpu, device, use_pinned_memory).view(size, 1)
        piece_count = _pin_and_move(
            piece_count_cpu.to(dtype=torch.int64), device, use_pinned_memory
        ).view(size)

        return planes, score, result, piece_count


class CPlaneDataLoaderAPI:
    def __init__(self):
        self.dll = self._load_library()
        self._define_prototypes()

    def _load_library(self):
        candidates = []
        for pattern in ("./build/**/*plane_batch_loader.*", "./build/*plane_batch_loader.*"):
            for lib in glob.glob(pattern, recursive=True):
                if lib.endswith((".so", ".dll", ".dylib")):
                    candidates.append(os.path.abspath(lib))

        last_error: OSError | None = None
        for lib in sorted(set(candidates), key=os.path.getmtime, reverse=True):
            try:
                return ctypes.cdll.LoadLibrary(lib)
            except OSError as e:
                last_error = e

        if last_error is not None:
            raise OSError("Found plane_batch_loader shared libraries but failed to load any of them.") from last_error
        raise FileNotFoundError(
            "Cannot find plane_batch_loader shared library. Build it first: "
            "cmake -S training/cnn/data_loader -B training/cnn/data_loader/build && "
            "cmake --build training/cnn/data_loader/build -j"
        )

    def _define_prototypes(self):
        # PlaneBatchCStream* create_plane_batch_stream(
        #     int concurrency, int num_files, const char* const* filenames,
        #     int batch_size, bool cyclic)
        self.dll.create_plane_batch_stream.restype = ctypes.c_void_p
        self.dll.create_plane_batch_stream.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_bool,
        ]

        # void destroy_plane_batch_stream(PlaneBatchCStream* stream)
        self.dll.destroy_plane_batch_stream.argtypes = [ctypes.c_void_p]

        # PlaneBatchCView fetch_next_plane_batch(PlaneBatchCStream* stream)
        self.dll.fetch_next_plane_batch.restype = PlaneBatchCView
        self.dll.fetch_next_plane_batch.argtypes = [ctypes.c_void_p]

        # void destroy_plane_batch(PlaneBatchCView view)
        self.dll.destroy_plane_batch.argtypes = [PlaneBatchCView]


try:
    c_lib = CPlaneDataLoaderAPI()
except FileNotFoundError as e:
    raise ImportError(f"Failed to initialize CPlaneDataLoaderAPI: {e}.")
