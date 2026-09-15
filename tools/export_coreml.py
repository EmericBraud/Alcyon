#!/usr/bin/env python3
"""Export data/gpu/v3_weights.bin to a CoreML .mlpackage so inference can run
on the Apple Neural Engine, for a throughput comparison against the MPSGraph
(GPU) backend.

There is no PyTorch definition of this model in this repo -- the architecture
is reconstructed here from src/engine/eval/gpu/metal_backend.mm, which is the
only complete description of both the graph and the weight-file layout. Keep
the two in sync.

Difference from the MPSGraph graph, deliberately: the phase-bucket selection
is NOT part of this model. MPSGraph takes a [N, 4] one-hot second input and
does the dot product inside the graph; here the model just outputs all four
buckets ([N, 4]) and the C++ side picks the right column. That keeps the
model single-input, which matters because the ANE deals poorly with extra
small side inputs, and the selection is a single array index on the CPU.

Usage:  python3 tools/export_coreml.py [--weights PATH] [--out PATH]
"""

import argparse
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import coremltools as ct

SE_REDUCTION = 4
PLANE_H = PLANE_W = 8
PSQT_PLANES = 12

# The model is exported at ONE FIXED batch size, and the backend chunks any
# larger batch into pieces of it (see coreml_backend.mm).
#
# This is not a simplification, it is the whole reason the ANE works at all.
# Measured with ct.EnumeratedShapes covering [1,8,32,64,128,256], CPU_AND_NE
# and CPU_ONLY were indistinguishable (1060 vs 998 positions/s) -- the ANE was
# silently refusing the model entirely, with an "E5RT encountered an STL
# exception. msg = std::bad_cast" from the ANE runtime. The same network at a
# fixed shape runs on the ANE at 15614 positions/s.
#
# 32 is the measured optimum, from a fixed-shape sweep on CPU_AND_NE:
#   8 -> 12456, 16 -> 14575, 32 -> 15614, 48 -> 14203,
#   64 -> 14282, 96 -> 13293, 128 -> 13705 positions/s
# Note this is the opposite of the GPU/MPSGraph backend, which needs the
# largest batch it can get (4127 positions/s at 256, 201 at 1) because its
# cost is dominated by per-call dispatch.
BATCH_SIZE = 32


class WeightReader:
    """Sequential float32 reader matching metal_backend.mm's load() order."""

    def __init__(self, path):
        with open(path, "rb") as f:
            self.channels, self.num_blocks, self.num_planes, self.num_buckets = struct.unpack("<4i", f.read(16))
            self.buf = np.frombuffer(f.read(), dtype="<f4")
        self.off = 0

    def take(self, count, shape=None):
        chunk = self.buf[self.off : self.off + count]
        if len(chunk) != count:
            raise ValueError(f"weights file truncated at offset {self.off} (wanted {count} floats)")
        self.off += count
        return chunk.reshape(shape) if shape else chunk

    def conv3x3(self, out_ch, in_ch):
        w = self.take(out_ch * in_ch * 9, (out_ch, in_ch, 3, 3))
        b = self.take(out_ch)
        return w, b

    def linear(self, out_features, in_features):
        # Stored [out, in] row-major -- exactly nn.Linear's layout, so unlike
        # metal_backend.mm's read_linear() no transpose is needed here.
        w = self.take(out_features * in_features, (out_features, in_features))
        b = self.take(out_features)
        return w, b

    def assert_fully_consumed(self):
        if self.off != len(self.buf):
            raise ValueError(f"{len(self.buf) - self.off} float(s) left unread -- layout mismatch")


def _load_conv(module, w, b):
    module.weight.data = torch.from_numpy(w.copy())
    module.bias.data = torch.from_numpy(b.copy())


def _load_linear(module, w, b):
    module.weight.data = torch.from_numpy(w.copy())
    module.bias.data = torch.from_numpy(b.copy())


class SEResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduced = channels // SE_REDUCTION
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.se_fc1 = nn.Linear(channels, reduced)
        self.se_fc2 = nn.Linear(reduced, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        pooled = out.mean(dim=(2, 3))
        gate = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(pooled))))
        out = out * gate.view(-1, self.channels, 1, 1)
        return F.relu(out + residual)


class V3Net(nn.Module):
    """Outputs [N, num_buckets] -- all bucket logits, trunk + PSQT. The caller
    selects the column for the position's phase bucket."""

    def __init__(self, channels, num_blocks, num_planes, num_buckets):
        super().__init__()
        self.num_blocks = num_blocks
        self.mid_index = num_blocks // 2 - 1  # where the mid-trunk pool is taken
        self.stem = nn.Conv2d(num_planes, channels, 3, padding=1)
        self.blocks = nn.ModuleList(SEResidualBlock(channels) for _ in range(num_blocks))
        self.head_fc1 = nn.ModuleList(nn.Linear(channels * 2, channels) for _ in range(num_buckets))
        self.head_fc2 = nn.ModuleList(nn.Linear(channels, 1) for _ in range(num_buckets))
        self.psqt = nn.Conv2d(PSQT_PLANES, num_buckets, 1, bias=False)

    def forward(self, planes):
        x = F.relu(self.stem(planes))
        pooled_mid = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == self.mid_index:
                pooled_mid = x.mean(dim=(2, 3))
        pooled_final = x.mean(dim=(2, 3))
        head_input = torch.cat([pooled_mid, pooled_final], dim=1)

        trunk = torch.cat(
            [fc2(F.relu(fc1(head_input))) for fc1, fc2 in zip(self.head_fc1, self.head_fc2)],
            dim=1,
        )
        # Spatial SUM, not mean -- matches legacy_model.py's PSQTHead and
        # metal_backend.mm's mean-times-64.
        psqt = self.psqt(planes[:, :PSQT_PLANES]).sum(dim=(2, 3))
        return trunk + psqt


def build_net(reader):
    net = V3Net(reader.channels, reader.num_blocks, reader.num_planes, reader.num_buckets)

    _load_conv(net.stem, *reader.conv3x3(reader.channels, reader.num_planes))
    reduced = reader.channels // SE_REDUCTION
    for block in net.blocks:
        _load_conv(block.conv1, *reader.conv3x3(reader.channels, reader.channels))
        _load_conv(block.conv2, *reader.conv3x3(reader.channels, reader.channels))
        _load_linear(block.se_fc1, *reader.linear(reduced, reader.channels))
        _load_linear(block.se_fc2, *reader.linear(reader.channels, reduced))
    for fc1, fc2 in zip(net.head_fc1, net.head_fc2):
        _load_linear(fc1, *reader.linear(reader.channels, reader.channels * 2))
        _load_linear(fc2, *reader.linear(1, reader.channels))
    net.psqt.weight.data = torch.from_numpy(
        reader.take(reader.num_buckets * PSQT_PLANES, (reader.num_buckets, PSQT_PLANES, 1, 1)).copy()
    )

    reader.assert_fully_consumed()
    return net.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="data/gpu/v3_weights.bin")
    parser.add_argument("--out", default="data/gpu/v3_model.mlpackage")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help=f"fixed batch size to export at (default {BATCH_SIZE}); must match "
                             "kAneBatch in coreml_backend.mm")
    parser.add_argument("--fp32", action="store_true",
                        help="export float32 instead of float16; the ANE only runs fp16, so this "
                             "is for numerical-parity checking against the MPSGraph backend")
    args = parser.parse_args()

    reader = WeightReader(args.weights)
    print(f"channels={reader.channels} blocks={reader.num_blocks} "
          f"planes={reader.num_planes} buckets={reader.num_buckets}")
    net = build_net(reader)

    example = torch.zeros(args.batch, reader.num_planes, PLANE_H, PLANE_W)
    traced = torch.jit.trace(net, example)

    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="planes",
                              shape=(args.batch, reader.num_planes, PLANE_H, PLANE_W),
                              dtype=np.float32)],
        outputs=[ct.TensorType(name="bucket_logits", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )
    model.save(args.out)
    print(f"saved {args.out} ({'fp32' if args.fp32 else 'fp16'}, fixed batch {args.batch})")

    # Reference values for the parity check against the C++ backends: a fixed
    # pseudo-random input, so both sides can be compared on identical planes.
    rng = np.random.default_rng(0)
    probe = rng.random((1, reader.num_planes, PLANE_H, PLANE_W), dtype=np.float32)
    with torch.no_grad():
        torch_out = net(torch.from_numpy(probe)).numpy()
    coreml_out = model.predict({"planes": np.repeat(probe, args.batch, axis=0)})["bucket_logits"][0]
    print("torch  bucket_logits:", np.array2string(torch_out[0], precision=4))
    print("coreml bucket_logits:", np.array2string(coreml_out, precision=4))
    print("max abs diff:", float(np.abs(torch_out[0] - coreml_out).max()))


if __name__ == "__main__":
    main()
