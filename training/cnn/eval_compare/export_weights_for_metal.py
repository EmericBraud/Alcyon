"""Export a legacy (v1/v3/v4) checkpoint's weights to a flat raw binary
file the engine's native Metal backend (src/engine/eval/gpu/metal_backend.mm)
loads directly -- no ONNX/CoreML, just float32 arrays in a documented
order, since MPSGraph is being driven by hand from raw tensors.

Every Conv2d+BatchNorm2d pair is folded into a single conv weight + bias
at export time (standard inference-time BN folding), so the Metal side
never implements BatchNorm at all -- just conv2d-with-bias:

    folded_weight = conv.weight * (bn.weight / sqrt(bn.running_var + eps)).reshape(-1,1,1,1)
    folded_bias   = bn.bias - bn.running_mean * bn.weight / sqrt(bn.running_var + eps)

File format (all little-endian, matches the host's native float32/int32 --
this is a same-machine dev tool, not a portable interchange format):

    header:
        int32   channels
        int32   num_blocks
        int32   num_planes
        int32   num_phase_buckets
    stem:
        float32[channels, num_planes, 3, 3]   stem conv weight (BN-folded)
        float32[channels]                     stem conv bias   (BN-folded)
    for each of num_blocks residual blocks (in order):
        float32[channels, channels, 3, 3]     conv1 weight (BN-folded)
        float32[channels]                     conv1 bias   (BN-folded)
        float32[channels, channels, 3, 3]     conv2 weight (BN-folded)
        float32[channels]                     conv2 bias   (BN-folded)
        float32[channels/4, channels]         se.fc1 weight   (reduction=4, see model.py)
        float32[channels/4]                   se.fc1 bias
        float32[channels, channels/4]         se.fc2 weight
        float32[channels]                     se.fc2 bias
    for each of num_phase_buckets heads (in order):
        float32[channels, channels*2]         head.fc1 weight
        float32[channels]                     head.fc1 bias
        float32[1, channels]                  head.fc2 weight
        float32[1]                            head.fc2 bias
    psqt:
        float32[num_phase_buckets, 12, 1, 1]  psqt conv weight (no bias, no BN)
"""

import argparse
import struct
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import legacy_model  # noqa: E402


def fold_conv_bn(conv, bn, eps=1e-5):
    with torch.no_grad():
        std = torch.sqrt(bn.running_var + eps)
        scale = bn.weight / std
        folded_weight = conv.weight * scale.view(-1, 1, 1, 1)
        folded_bias = bn.bias - bn.running_mean * scale
    return folded_weight.contiguous(), folded_bias.contiguous()


def write_tensor(f, tensor):
    f.write(tensor.numpy().astype("float32").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    channels = sd["stem_conv.weight"].shape[0]
    num_planes = sd["stem_conv.weight"].shape[1]
    num_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    num_phase_buckets = sd["psqt.conv.weight"].shape[0]
    has_se = any(k.startswith("blocks.0.se.") for k in sd)
    if not has_se:
        raise SystemExit("this exporter assumes SE blocks are present (v3/v4, not v1)")

    print(f"channels={channels} num_blocks={num_blocks} num_planes={num_planes} "
          f"num_phase_buckets={num_phase_buckets}")

    with open(args.out, "wb") as f:
        f.write(struct.pack("<iiii", channels, num_blocks, num_planes, num_phase_buckets))

        # Stem: build throwaway nn.Module-free BN objects from the state dict
        # to reuse fold_conv_bn's logic without instantiating the full model.
        class _Conv:
            def __init__(self, w):
                self.weight = w

        class _BN:
            def __init__(self, prefix):
                self.weight = sd[f"{prefix}.weight"]
                self.bias = sd[f"{prefix}.bias"]
                self.running_mean = sd[f"{prefix}.running_mean"]
                self.running_var = sd[f"{prefix}.running_var"]

        w, b = fold_conv_bn(_Conv(sd["stem_conv.weight"]), _BN("stem_bn"))
        write_tensor(f, w)
        write_tensor(f, b)

        for i in range(num_blocks):
            prefix = f"blocks.{i}"
            w1, b1 = fold_conv_bn(_Conv(sd[f"{prefix}.conv1.weight"]), _BN(f"{prefix}.bn1"))
            w2, b2 = fold_conv_bn(_Conv(sd[f"{prefix}.conv2.weight"]), _BN(f"{prefix}.bn2"))
            write_tensor(f, w1)
            write_tensor(f, b1)
            write_tensor(f, w2)
            write_tensor(f, b2)
            write_tensor(f, sd[f"{prefix}.se.fc1.weight"])
            write_tensor(f, sd[f"{prefix}.se.fc1.bias"])
            write_tensor(f, sd[f"{prefix}.se.fc2.weight"])
            write_tensor(f, sd[f"{prefix}.se.fc2.bias"])

        for i in range(num_phase_buckets):
            prefix = f"heads.{i}"
            write_tensor(f, sd[f"{prefix}.fc1.weight"])
            write_tensor(f, sd[f"{prefix}.fc1.bias"])
            write_tensor(f, sd[f"{prefix}.fc2.weight"])
            write_tensor(f, sd[f"{prefix}.fc2.bias"])

        write_tensor(f, sd["psqt.conv.weight"])

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
