#!/usr/bin/env python3
"""Produce a patched GGUF by applying weight-XOR + scale-bit flips to the raw bytes.

Usage:
    python apply_patches_gguf.py \\
        --input  Bonsai-8B.gguf \\
        --output Bonsai-8B-patched.gguf \\
        --patches v2_patches/ v3_patches/

Q1_0_g128 group layout:
    [scale_fp16 (2 bytes)][packed_bits (16 bytes)]   — total 18 bytes / group

Patch records:
    {"layer", "proj", "group", "type": "group" | "scale", ["bit": 4..9]}

Weight-XOR: flip all 128 sign bits → XOR packed_bits (16 bytes) with 0xFF.
Scale flip: flip one mantissa bit (4..9) of the FP16 scale. Bit b maps to
           byte b//8, bit position b%8.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys


GROUP_BYTES = 18  # 2 bytes scale + 16 bytes packed bits


def layer_tensor_name(layer: int, proj_suffix: str) -> str:
    """Match xor_search.BonsaiEngine.layer_name()."""
    return f"blk.{layer}.{proj_suffix}"


def load_patches(dirs):
    patches_by_file = {}
    for d in dirs:
        for pf in sorted(glob.glob(os.path.join(d, "patch_*.json"))):
            data = json.load(open(pf))
            patches_by_file[pf] = data.get("flips", [])
    return patches_by_file


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="base Bonsai-8B.gguf")
    ap.add_argument("--output", required=True, help="output patched.gguf")
    ap.add_argument("--patches", nargs="+", required=True,
                    help="patch directories, e.g. v2_patches v3_patches")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from gguf import GGUFReader  # type: ignore

    # Register Q1_0 quant type so GGUFReader doesn't throw on type 41.
    # Mirrors xor_search.patch_gguf_q1_0().
    from gguf.constants import GGMLQuantizationType, GGML_QUANT_SIZES
    try:
        GGMLQuantizationType(41)
    except ValueError:
        obj = int.__new__(GGMLQuantizationType, 41)
        obj._name_ = "Q1_0"; obj._value_ = 41
        GGMLQuantizationType._member_map_["Q1_0"] = obj
        GGMLQuantizationType._value2member_map_[41] = obj
    if GGMLQuantizationType(41) not in GGML_QUANT_SIZES:
        GGML_QUANT_SIZES[GGMLQuantizationType(41)] = (128, 18)

    # Read tensor offsets from base
    reader = GGUFReader(args.input)
    # tensor.data_offset is ALREADY ABSOLUTE (gguf library computes
    # start_offs + offset_tensor[0] in _build_tensors, where start_offs
    # is reader.data_offset). Do NOT add reader.data_offset again.
    tmap = {}
    for t in reader.tensors:
        tmap[t.name] = {
            "offset": int(t.data_offset),
            "shape": tuple(int(s) for s in t.shape),
            "type": str(t.tensor_type).rsplit(".", 1)[-1],
        }
    print(f"loaded {len(tmap)} tensor metadata from {args.input}")
    # Spot-check an FFN tensor
    sample = next((n for n in tmap if "ffn_gate" in n), None)
    if sample:
        print(f"  sample: {sample} shape={tmap[sample]['shape']} type={tmap[sample]['type']} "
              f"offset=0x{tmap[sample]['offset']:x}")

    # Collect patches, group by tensor
    patches_by_file = load_patches(args.patches)
    total_flips = sum(len(v) for v in patches_by_file.values())
    print(f"loaded {total_flips} flips across {len(patches_by_file)} patch files")

    # Copy the GGUF
    if not args.dry_run:
        shutil.copyfile(args.input, args.output)
        print(f"copied {args.input} -> {args.output}")

    # Apply in place
    grouped = {}  # tensor_name -> list of (group, type, bit)
    for pf, flips in patches_by_file.items():
        for f in flips:
            name = layer_tensor_name(f["layer"], f["proj"])
            if name not in tmap:
                print(f"WARN: tensor {name} not found (skipping flip from {pf})")
                continue
            grouped.setdefault(name, []).append(f)

    n_w = n_s = 0
    if args.dry_run:
        for name, flips in grouped.items():
            nw = sum(1 for f in flips if f.get("type") != "scale")
            ns = sum(1 for f in flips if f.get("type") == "scale")
            print(f"  {name}: {nw} weight + {ns} scale flips")
            n_w += nw
            n_s += ns
        print(f"dry run: would apply {n_w} weight + {n_s} scale flips")
        return

    with open(args.output, "rb+") as f:
        for name, flips in grouped.items():
            base_offset = tmap[name]["offset"]
            for flip in flips:
                group = int(flip["group"])
                group_offset = base_offset + group * GROUP_BYTES
                if flip.get("type") == "scale":
                    bit = int(flip["bit"])
                    if not 0 <= bit <= 15:
                        raise ValueError(f"scale bit {bit} out of range")
                    byte_idx = bit // 8
                    bit_in_byte = bit % 8
                    f.seek(group_offset + byte_idx)
                    orig = f.read(1)[0]
                    new = orig ^ (1 << bit_in_byte)
                    f.seek(group_offset + byte_idx)
                    f.write(bytes([new]))
                    n_s += 1
                else:
                    # Weight XOR: flip all 128 sign bits (bytes 2..17 of group)
                    f.seek(group_offset + 2)
                    orig = f.read(16)
                    new = bytes(b ^ 0xFF for b in orig)
                    f.seek(group_offset + 2)
                    f.write(new)
                    n_w += 1
    print(f"applied {n_w} weight + {n_s} scale flips to {args.output}")


if __name__ == "__main__":
    main()
