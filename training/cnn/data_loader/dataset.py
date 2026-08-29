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
        val_percent: int = 0,
        is_validation: bool = False,
        nnue_path: str = "",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.device = device
        self.use_pinned_memory = use_pinned_memory
        self._stream = stream.create_plane_batch_stream(
            num_workers, filenames, batch_size, cyclic, val_percent, is_validation, nnue_path,
            rank, world_size,
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

    val_percent/is_validation: split filenames into disjoint, position
    -hash-based train/validation subsets without physically cutting the
    binpack file — see plane_batch_stream.h. Two datasets built from the
    same filenames with is_validation flipped never see the same
    position. Leave val_percent=0 (default) to use every position, e.g.
    when filenames already points at a dedicated held-out file.

    nnue_path: path to a .nnue weight file. When non-empty, each batch
    also yields chess26's own NNUE static evaluation per position (see
    plane_batch.h's PlaneBatch::nnue_score) — used by v5's residual-
    correction training. Leave empty ("") to skip NNUE evaluation
    entirely (nnue_score is filled with 0.0 in that case).

    rank/world_size: for multi-GPU (DDP) training, shards the binpack
    across processes so each rank trains on a disjoint chunk sequence
    — see plane_batch_stream.h. Defaults (0, 1) disable sharding
    entirely, e.g. for single-GPU training or a validation stream every
    rank reads independently.
    """

    def __init__(
        self,
        filenames: list[str],
        batch_size: int,
        cyclic: bool = True,
        num_workers: int = 1,
        use_pinned_memory: bool = False,
        val_percent: int = 0,
        is_validation: bool = False,
        nnue_path: str = "",
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.use_pinned_memory = use_pinned_memory
        self.val_percent = val_percent
        self.is_validation = is_validation
        self.nnue_path = nnue_path
        self.rank = rank
        self.world_size = world_size
        self.device = "cpu"

    def __iter__(self):
        return PlaneBatchProvider(
            self.filenames,
            self.batch_size,
            cyclic=self.cyclic,
            num_workers=self.num_workers,
            device=self.device,
            use_pinned_memory=self.use_pinned_memory,
            val_percent=self.val_percent,
            is_validation=self.is_validation,
            nnue_path=self.nnue_path,
            rank=self.rank,
            world_size=self.world_size,
        )
