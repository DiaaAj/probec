"""CPU kernels for DeepSeek-V4.1's reference model: quantizers, GEMMs, sparse attention, Sinkhorn."""
import torch

from probec.dsv41 import kernels as K


def test_act_quant_power_of_two_scales_keep_dyadic_values():
    x = torch.tensor([[1.0, -0.125, 16.0, 2.0] * 8], dtype=torch.bfloat16)
    y = K.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
    assert torch.equal(y, x)
    q, s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    assert float(s.float()) == 2.0 ** -4             # 16 / 448 -> 2^ceil(log2 0.0357) = 2^-4


def test_fp4_kv_scales_turn_unit_codes_into_1_03125():
    x = torch.ones(1, 16, dtype=torch.bfloat16)
    y = K.fp4_act_quant(x.clone(), 16, inplace=True, scale_dtype=torch.float8_e4m3fn)
    assert float(y[0, 0]) == 1.03125                 # e4m3(1/6) = 0.171875, 6 * 0.171875
    x[0, 0] = 1.5                                    # one lane at 1.5 pins the scale at 0.25
    y = K.fp4_act_quant(x.clone(), 16, inplace=True, scale_dtype=torch.float8_e4m3fn)
    assert torch.equal(y, x)


def test_e2m1_rounds_ties_to_even():
    q = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75])
    assert K.e2m1_round(q).tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0]


def test_fp8_and_fp4_gemm_match_dequantized_matmul():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(3, 64, generator=g)
    w = torch.randn(40, 64, generator=g)
    aq, as_ = K.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
    wq, ws = K.act_quant(w, 32, "ue8m0", torch.float8_e8m0fnu)
    wblk = ws.float()                                # per row and K block; expand to per 32-row block layout
    b_s = wblk[::32]                                 # rows 0 and 32 stand for their 32-row blocks
    deq_a = aq.float() * as_.float().repeat_interleave(32, -1)
    deq_w = (wq.float().reshape(40, 2, 32) * b_s[torch.arange(40) // 32][:, :, None]).reshape(40, 64)
    with torch.no_grad():
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            c = K.fp8_gemm(aq, as_, wq, b_s.contiguous(), torch.float8_e8m0fnu, 32)
        finally:
            torch.set_default_dtype(prev)
    assert torch.allclose(c, deq_a @ deq_w.T, rtol=1e-5, atol=1e-5)
    p, ps = K.fp4_act_quant(w, 32)                   # FP4 weights with E8M0 scales per (row, 32)
    deq_p = K._unpack_fp4(p) * ps.float().repeat_interleave(32, -1)
    torch.set_default_dtype(torch.float32)
    try:
        c4 = K.fp4_gemm(aq, as_, p, ps, torch.float8_e8m0fnu, 32)
    finally:
        torch.set_default_dtype(prev)
    assert torch.allclose(c4, deq_a @ deq_p.T, rtol=1e-5, atol=1e-5)


def test_sparse_attn_with_sink_and_dead_slots():
    g = torch.Generator().manual_seed(1)
    b, m, h, d, n = 1, 2, 4, 64, 150
    q = torch.randn(b, m, h, d, generator=g).to(torch.bfloat16)
    kv = torch.randn(b, n, d, generator=g).to(torch.bfloat16)
    sink = torch.randn(h, generator=g)
    idx = torch.randint(0, n, (b, m, 130), generator=g).int()
    idx[0, 0, 5] = -1
    o = K.sparse_attn(q, kv, sink, idx, 0.125)
    for mi in range(m):                              # dense reference (fp32 P): close, not bitwise
        sel = idx[0, mi].long()
        ok = sel != -1
        rows = kv[0, sel[ok]].float()
        s = (q[0, mi].float() @ rows.T) * 0.125
        mx = s.amax(-1, keepdim=True)
        e = torch.exp(s - mx)
        ref = (e @ rows) / (e.sum(-1, keepdim=True) + torch.exp(sink[:, None] - mx))
        assert torch.allclose(o[0, mi].float(), ref, rtol=2e-2, atol=2e-2)


def test_sinkhorn_pins_to_identity_with_large_bases():
    base = torch.zeros(24)
    base[:4] = torch.tensor([32.0, -32.0, -32.0, -32.0])     # pre selects copy 0
    base[4:8] = torch.tensor([0.0, -32.0, -32.0, -32.0])     # post = 2 * sigmoid(0) = 1 on copy 0
    base[8:] = (torch.eye(4) * 64.0 - 32.0).flatten()        # comb ~ identity
    pre, post, comb = K.hc_split_sinkhorn(torch.zeros(1, 1, 24), torch.ones(3), base)
    assert float(pre[0, 0, 0]) == 1.0 + 1e-6 or abs(float(pre[0, 0, 0]) - 1.0) < 2e-6
    assert float(post[0, 0, 0]) == 1.0
    assert torch.allclose(comb[0, 0], torch.eye(4), atol=1e-5)
