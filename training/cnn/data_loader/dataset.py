import torch

from . import stream


class PlaneBatchProvider:
    """Iterator pulling PlaneBatch views off the C++ stream.

    v1: no background prefetch thread — each __next__ blocks on the
    synchronous PlaneBatchStream::next() (see plane_batch_stream.h).
    nnue-pytorch's dataset.py has a more elaborate prefetch/pinned-
    memory pipeline (FixedNumBatchesDataset) worth porting here if
    this turns out to be a training-throughput bottleneck.
    """

    def __init__(
        self,
        filenames: list[str],
        batch_size: int,
        cyclic: bool = True,
        num_workers: int = 1,
        device: str = "cpu",
        use_pinned_memory: bool = False,
    ):
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.device = device
        self.use_pinned_memory = use_pinned_memory
        self._stream = stream.create_plane_batch_stream(
            num_workers, filenames, batch_size, cyclic
        )

    def __iter__(self):
        return self

    def __next__(self):
        view = stream.fetch_next_plane_batch(self._stream)
        if view.size == 0:
            raise StopIteration
        tensors = view.get_tensors(self.device, use_pinned_memory=self.use_pinned_memory)
        stream.destroy_plane_batch(view)
        return tensors

    def __del__(self):
        if hasattr(self, "_stream"):
            stream.destroy_plane_batch_stream(self._stream)


class PlaneBatchDataset(torch.utils.data.IterableDataset):
    """Wraps PlaneBatchProvider so torch.utils.data.DataLoader can consume it.

    Each yielded item is already a full batch (planes, score, result) —
    use DataLoader(dataset, batch_size=None) on the training side, the
    batching itself happens in C++, not in torch's collate step.
    """

    def __init__(
        self,
        filenames: list[str],
        batch_size: int,
        cyclic: bool = True,
        num_workers: int = 1,
        use_pinned_memory: bool = False,
    ):
        super().__init__()
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.use_pinned_memory = use_pinned_memory
        self.device = "cpu"

    def __iter__(self):
        return PlaneBatchProvider(
            self.filenames,
            self.batch_size,
            cyclic=self.cyclic,
            num_workers=self.num_workers,
            device=self.device,
            use_pinned_memory=self.use_pinned_memory,
        )
