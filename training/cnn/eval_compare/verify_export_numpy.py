"""Sanity check for export_weights_for_metal.py's BN-folding math: replay
the exported binary's forward pass in pure numpy and compare against
PyTorch's own model output on random inputs. This is also the exact
reference algorithm src/engine/eval/gpu/metal_backend.mm must replicate --
debugging the math here in numpy is much cheaper than debugging it in
Objective-C++/MPSGraph.
"""

import argparse
import struct
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model_loader import load_model  # noqa: E402


def read_tensor(f, shape):
    n = 1
    for d in shape:
        n *= d
    return np.frombuffer(f.read(n * 4), dtype="<f4").reshape(shape).astype(np.float64)


def conv2d(x, w, b):
    # x: (N,C,8,8), w: (O,C,3,3), b: (O,), padding=1, stride=1, no dilation.
    out = F.conv2d(torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b), padding=1)
    return out.numpy().astype(np.float64)


def relu(x):
    return np.maximum(x, 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def linear(x, w, b):
    return x @ w.T + b


def load_exported(path):
    with open(path, "rb") as f:
        channels, num_blocks, num_planes, num_buckets = struct.unpack("<iiii", f.read(16))
        stem_w = read_tensor(f, (channels, num_planes, 3, 3))
        stem_b = read_tensor(f, (channels,))
        blocks = []
        for _ in range(num_blocks):
            w1 = read_tensor(f, (channels, channels, 3, 3))
            b1 = read_tensor(f, (channels,))
            w2 = read_tensor(f, (channels, channels, 3, 3))
            b2 = read_tensor(f, (channels,))
            reduced = channels // 4
            se_w1 = read_tensor(f, (reduced, channels))
            se_b1 = read_tensor(f, (reduced,))
            se_w2 = read_tensor(f, (channels, reduced))
            se_b2 = read_tensor(f, (channels,))
            blocks.append((w1, b1, w2, b2, se_w1, se_b1, se_w2, se_b2))
        heads = []
        for _ in range(num_buckets):
            hw1 = read_tensor(f, (channels, channels * 2))
            hb1 = read_tensor(f, (channels,))
            hw2 = read_tensor(f, (1, channels))
            hb2 = read_tensor(f, (1,))
            heads.append((hw1, hb1, hw2, hb2))
        psqt_w = read_tensor(f, (num_buckets, 12, 1, 1))
    return dict(channels=channels, num_blocks=num_blocks, num_planes=num_planes,
                num_buckets=num_buckets, stem_w=stem_w, stem_b=stem_b, blocks=blocks,
                heads=heads, psqt_w=psqt_w)


def forward_numpy(exported, planes, piece_count):
    x = relu(conv2d(planes, exported["stem_w"], exported["stem_b"]))
    mid_point = exported["num_blocks"] // 2
    pooled_mid = None
    for i, (w1, b1, w2, b2, se_w1, se_b1, se_w2, se_b2) in enumerate(exported["blocks"]):
        residual = x
        out = relu(conv2d(x, w1, b1))
        out = conv2d(out, w2, b2)
        pooled = out.mean(axis=(2, 3))
        gate = sigmoid(linear(relu(linear(pooled, se_w1, se_b1)), se_w2, se_b2))
        out = out * gate[:, :, None, None]
        x = relu(out + residual)
        if i == mid_point - 1:
            pooled_mid = x.mean(axis=(2, 3))
    pooled_final = x.mean(axis=(2, 3))

    boundaries = (24, 16, 8)
    bucket = np.zeros_like(piece_count)
    for bnd in boundaries:
        bucket += (piece_count < bnd).astype(bucket.dtype)

    n = planes.shape[0]
    trunk_logits = np.zeros(n)
    for b_idx, (hw1, hb1, hw2, hb2) in enumerate(exported["heads"]):
        mask = bucket == b_idx
        if not mask.any():
            continue
        h = relu(linear(np.concatenate([pooled_mid[mask], pooled_final[mask]], axis=-1), hw1, hb1))
        trunk_logits[mask] = linear(h, hw2, hb2).squeeze(-1)

    # PSQT: conv1x1 (12 -> num_buckets) then spatial sum, then gather bucket.
    psqt_conv_out = np.einsum("nchw,bc->nbhw", planes[:, :12], exported["psqt_w"][:, :, 0, 0])
    psqt_per_bucket = psqt_conv_out.sum(axis=(2, 3))  # (n, num_buckets)
    psqt_logits = psqt_per_bucket[np.arange(n), bucket]

    return trunk_logits + psqt_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exported", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, is_v5, _ = load_model(args.checkpoint, device)
    assert not is_v5

    exported = load_exported(args.exported)
    n, c, planes_h, planes_w = args.num_samples, exported["num_planes"], 8, 8
    rng = np.random.default_rng(0)
    planes = rng.random((n, c, planes_h, planes_w)).astype(np.float64)
    piece_count = rng.integers(2, 31, size=n)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(planes).float(), torch.from_numpy(piece_count).long()).numpy().astype(np.float64)

    numpy_out = forward_numpy(exported, planes, piece_count)

    max_abs_diff = np.abs(torch_out - numpy_out).max()
    print(f"torch:  {torch_out}")
    print(f"numpy:  {numpy_out}")
    print(f"max abs diff: {max_abs_diff:.8f}")
    if max_abs_diff < 1e-3:
        print("PASS -- export/BN-folding math matches PyTorch.")
    else:
        print("FAIL -- mismatch, do not trust the Metal backend built on this export logic yet.")


if __name__ == "__main__":
    main()
