"""Sharded bf16 safetensors writer that refuses any value bf16 cannot hold exactly."""
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file


class Writer:
    def __init__(self, out_dir, shard_bytes=2 << 30):
        self.out = out_dir
        self.shard_bytes = shard_bytes
        self.cur, self.cur_bytes = {}, 0
        self.files, self.index = [], {}
        self.inexact = []
        os.makedirs(out_dir, exist_ok=True)

    @staticmethod
    def to_bf16(name, arr):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        t = torch.from_numpy(a).to(torch.bfloat16)
        if not torch.equal(t.float(), torch.from_numpy(a)):
            raise ValueError(f"{name}: value not exactly representable in bf16")
        return t

    def add(self, name, arr, exact=True):
        """arr: float32 numpy (checked), or a bf16 torch tensor built from checked values.

        exact=False is for tensors that only shape attention scores (which need margin, not exactness):
        they are rounded to bf16 deliberately and listed in `self.inexact`."""
        if isinstance(arr, torch.Tensor):
            t = arr
        elif exact:
            t = self.to_bf16(name, arr)
        else:
            t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(torch.bfloat16)
            self.inexact.append(name)
        n = t.numel() * t.element_size()
        if self.cur and self.cur_bytes + n > self.shard_bytes:
            self.flush()
        self.cur[name] = t.contiguous()
        self.cur_bytes += n

    def flush(self):
        if not self.cur:
            return
        fn = f"model-{len(self.files) + 1:05d}.safetensors"
        save_file(self.cur, os.path.join(self.out, fn), metadata={"format": "pt"})
        for k, t in self.cur.items():
            self.index[k] = (fn, t.numel() * t.element_size())
        self.files.append(fn)
        self.cur, self.cur_bytes = {}, 0

    def close(self):
        self.flush()
        if len(self.files) == 1:
            os.replace(os.path.join(self.out, self.files[0]), os.path.join(self.out, "model.safetensors"))
            return
        total = len(self.files)
        names = {}
        for i, fn in enumerate(self.files):
            new = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
            os.replace(os.path.join(self.out, fn), os.path.join(self.out, new))
            names[fn] = new
        idx = {"metadata": {"total_size": sum(b for _, b in self.index.values())},
               "weight_map": {k: names[fn] for k, (fn, _) in sorted(self.index.items())}}
        with open(os.path.join(self.out, "model.safetensors.index.json"), "w") as f:
            json.dump(idx, f, indent=1)
