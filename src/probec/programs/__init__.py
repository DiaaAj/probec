"""Probe programs: each builds exact weights for a target spec and returns (meta, codes)."""
from . import hybrid, mla, moe, posrecall, recall

PROGRAMS = {"recall": recall, "posrecall": posrecall, "moe": moe, "mla": mla, "hybrid": hybrid}
