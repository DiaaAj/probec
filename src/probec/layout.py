"""Lane layouts: named fields of one vector (a residual, a latent, a query), handed out in order.

Engines quantize activations in groups of lanes (FP8 per 32 with power-of-two scales, FP4 per 16 or 32),
so a designed value survives only if its group's maximum maps it onto the grid. probec's values are
powers of two (±1 codes, +1 fills, ballast 16, scaled copies), which share any group exactly; the two
things that break are a group with no nonzero lane (a 0/0 scale in kernels without an amax floor) and a
group whose maximum is not a power of two. `Layout` places fields, `pin_energy` adds +1 fill and ballast
lanes so the vector's energy is exactly rms^2 * width (RMSNorm then divides by exactly rms), and
`check_groups` verifies the group rules on a concrete vector.
"""
from dataclasses import dataclass

import numpy as np

from .codes import ballast


@dataclass(frozen=True)
class Field:
    start: int
    size: int

    @property
    def stop(self):
        return self.start + self.size

    @property
    def lanes(self):
        return slice(self.start, self.stop)


class Layout:
    def __init__(self, width, name="vector"):
        self.width, self.name = width, name
        self.fields: dict[str, Field] = {}
        self.next = 0

    def add(self, name, size, align=1):
        """Allocate `size` lanes at the next offset that is a multiple of `align`."""
        if name in self.fields:
            raise ValueError(f"{self.name}: field {name!r} allocated twice")
        start = -(-self.next // align) * align
        if start + size > self.width:
            raise SystemExit(f"probec: {self.name} width {self.width} too small for field {name!r} "
                             f"({size} lanes at {start})")
        self.fields[name] = f = Field(start, size)
        self.next = start + size
        return f

    def __getitem__(self, name):
        return self.fields[name]

    def pin_energy(self, n_unit, rms=8.0, bval=16.0, align=1):
        """After the fields: k lanes of +1 and nb ballast lanes at `bval`, so that n_unit unit-magnitude
        lanes plus these give energy rms^2 * width. Returns (ones field, ballast field)."""
        k, nb = ballast(n_unit, self.width, bval, rms)
        ones = self.add("ones", k, align) if k else Field(self.next, 0)
        bal = self.add("ballast", nb)
        return ones, bal

    def summary(self):
        return {n: [f.start, f.size] for n, f in self.fields.items()}


GROUPS = {"fp8": 32, "fp4-e8m0": 32, "fp4-e4m3": 16}


def roundtrip(vec, fmt):
    """`vec` (float32) after the engines' quantize-dequantize: 'fp8' (E4M3 per 32 lanes, power-of-two
    scales: activations, window KV), 'fp4-e8m0' (E2M1 per 32, power-of-two scales: index keys and
    queries) or 'fp4-e4m3' (E2M1 per 16, E4M3 scales: the compressed KV cache)."""
    import torch

    from .dsv41.kernels import act_quant, fp4_act_quant
    x = torch.tensor(np.asarray(vec, dtype=np.float32)).reshape(1, -1)
    if fmt == "fp8":
        return act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True).numpy()[0]
    scale = torch.float8_e8m0fnu if fmt == "fp4-e8m0" else torch.float8_e4m3fn
    return fp4_act_quant(x.clone(), GROUPS[fmt], inplace=True, scale_dtype=scale).numpy()[0]


def check_groups(vec, fmt, where="vector", allow_zero_groups=False):
    """`vec` survives quantization format `fmt` bit for bit, and (unless allowed) no lane group is all
    zero (kernels without an amax floor would divide 0 by 0)."""
    v = np.asarray(vec, dtype=np.float32)
    g = GROUPS[fmt]
    bad = []
    if not allow_zero_groups:
        bad += [f"lanes {i}-{i + g - 1} all zero" for i in range(0, len(v) - len(v) % g, g)
                if not np.any(v[i:i + g])]
    lost = np.nonzero(roundtrip(v, fmt) != v)[0]
    if len(lost):
        bad.append(f"{len(lost)} lanes change under {fmt} (first {lost[0]}: {v[lost[0]]})")
    if bad:
        raise ValueError(f"{where}: " + "; ".join(bad[:6]))
