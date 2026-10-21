"""DeepSeek-native quantized weights (DeepSeek V4 / V4.1 checkpoints).

- FP8: E4M3 codes in square blocks (32x32 in V4.1) with one power-of-two scale per block, stored as E8M0
  in a companion `<name>.scale` tensor of shape [ceil(out/32), ceil(in/32)].
- MXFP4 (routed experts): E2M1 codes packed two per byte (low nibble = even input index) as int8
  [out, in/2], with one E8M0 scale per (row, 32 input values) in `<name>.scale` [out, in/32].

probec chooses every scale, so a designed weight is written only if the format holds it bit for bit;
anything else raises. The decoders give the values an engine dequantizes, for the CPU reference.
"""
import torch

from .quant import E2M1, quant_fp4, quant_fp8


def fp8_block(name, w, block=32):
    """(codes float8_e4m3fn [o, i], scales float8_e8m0fnu [ceil(o/b), ceil(i/b)]) for an exact weight."""
    w = torch.as_tensor(w, dtype=torch.float32)
    q, s, exact = quant_fp8(w, block)
    if not exact:
        raise ValueError(f"{name}: not exactly representable as FP8 E4M3 with {block}x{block} power-of-two scales")
    return q, s.to(torch.float8_e8m0fnu)


def mxfp4(name, w):
    """(packed int8 [o, i/2], scales float8_e8m0fnu [o, i/32]) for an exact weight."""
    w = torch.as_tensor(w, dtype=torch.float32)
    codes, s, exact = quant_fp4(w, 32, "e8m0")
    if not exact:
        raise ValueError(f"{name}: not exactly representable as MXFP4 (E2M1 with a power-of-two scale per 32)")
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8).view(torch.int8)
    return packed, s.to(torch.float8_e8m0fnu)


def add_fp8(writer, name, w, block=32):
    """Write `<name>.weight` (E4M3) and `<name>.scale` (E8M0)."""
    q, s = fp8_block(name, w, block)
    writer.add(name + ".weight", q)
    writer.add(name + ".scale", s)


def add_mxfp4(writer, name, w):
    """Write `<name>.weight` (packed E2M1, int8) and `<name>.scale` (E8M0)."""
    p, s = mxfp4(name, w)
    writer.add(name + ".weight", p)
    writer.add(name + ".scale", s)


def dequant_fp8(q, s, block=32):
    """float32 [o, i] from E4M3 codes and per-block E8M0 scales."""
    o, i = q.shape
    full = s.float().repeat_interleave(block, 0).repeat_interleave(block, 1)[:o, :i]
    return q.float() * full


def dequant_mxfp4(packed, s):
    """float32 [o, 2 * packed columns] from packed E2M1 (low nibble first) and per-32 E8M0 scales."""
    b = packed.view(torch.uint8).long()
    codes = torch.stack((b & 15, b >> 4), dim=-1).reshape(b.shape[0], -1)
    mag = E2M1[codes & 7]
    val = torch.where((codes & 8) > 0, -mag, mag)
    return val * s.float().repeat_interleave(32, 1)
