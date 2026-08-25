import ctypes

from ._native import PlaneBatchCView, c_lib


def _to_c_str_array(str_list):
    c_str_array = (ctypes.c_char_p * len(str_list))()
    c_str_array[:] = [s.encode("utf-8") for s in str_list]
    return c_str_array


def create_plane_batch_stream(
    concurrency: int,
    filenames: list[str],
    batch_size: int,
    cyclic: bool,
    val_percent: int = 0,
    is_validation: bool = False,
) -> ctypes.c_void_p:
    return c_lib.dll.create_plane_batch_stream(
        concurrency,
        len(filenames),
        _to_c_str_array(filenames),
        batch_size,
        cyclic,
        val_percent,
        is_validation,
    )


def destroy_plane_batch_stream(stream: ctypes.c_void_p):
    c_lib.dll.destroy_plane_batch_stream(stream)


def fetch_next_plane_batch(stream: ctypes.c_void_p) -> PlaneBatchCView:
    return c_lib.dll.fetch_next_plane_batch(stream)


def destroy_plane_batch(view: PlaneBatchCView):
    c_lib.dll.destroy_plane_batch(view)
