# Error Log — Phase 3: Custom Model Architectures

> **Project:** `Custom_Model_Architectures`
> **Test Command:** `pytest tests/ -x -v`
> **Total Tests:** 47 | **Final Status:** ✅ 47 Passed  ❌ 0 Failed | **Date:** 2026-07-17

---

## 1. Test Failure Log

| # | Status | Test | File | Error Type | Pytest Message |
|---|:---:|---|---|---|---|
| 1 | ❌ → ✅ | `test_causal_mask_first_token_unchanged` | `tests/test_attention.py` | `AssertionError` | `First-token outputs differ under causal and no-mask — the mask is incorrectly applied.` |
| 2 | ❌ → ✅ | `test_output_is_contiguous` | `tests/test_attention.py` | `AssertionError` | `Output tensor is not contiguous.` |
| 3 | ❌ → ✅ | `test_aggr_mode[lstm]` | `tests/test_graph.py` | `ValueError` (PyG) | `Can not perform aggregation since the 'index' tensor is not sorted.` |

---

## 2. Root Cause & Fix Details

| # | Module | Root Cause | Fix Applied | File Changed |
|---|---|---|---|---|
| 1 | `SelfAttention` | Causal mask forces row-0 to attend **only to itself** (all other positions → `-inf`). All-zeros mask lets row-0 attend to **all tokens**. These yield different softmax distributions over `V`, so position-0 outputs are never equal — the test premise was mathematically wrong. | Replaced with a **position-0 isolation test**: two inputs share position-0 but differ at positions `1..T-1`. Under causal mask, pos-0 output must be identical for both inputs regardless of future tokens. | `tests/test_attention.py` |
| 2 | `SpatialAttention` | `torch.transpose()` returns a **strided view** — it reorders axes without copying data, leaving the tensor non-contiguous in memory. Calling `.view()` on a non-contiguous tensor is undefined behavior and was producing a non-contiguous output. | Added `.contiguous()` between `transpose` and `view` in `SpatialAttention.forward` to force a memory-contiguous copy before reshaping. | `fusion/models/custom/attention/spatial_attention.py` |
| 3 | `GraphSAGEEncoder` | PyG's `LSTMAggregation` (`aggr="lstm"`) processes node messages sequentially with an LSTM and **requires `edge_index` sorted by destination node** (`edge_index[1]`). The `_make_batch()` helper generates random unsorted edges — valid for `mean`/`max` (order-invariant) but explicitly rejected by `lstm`. | Added `edge_index = edge_index[:, edge_index[1].argsort()]` pre-sort inside the `lstm` branch of the parametrized test. Production module code was **not changed**. | `tests/test_graph.py` |

---

## 3. Code Diff Summary

| # | Location | Before (Buggy) | After (Fixed) |
|---|---|---|---|
| 1 | `tests/test_attention.py` | `assert torch.allclose(out_causal[:, 0, :], out_nomask[:, 0, :], atol=1e-6)` | `assert torch.allclose(out_a[:, 0, :], out_b[:, 0, :], atol=1e-6)` where `x_b[:, 1:, :] = torch.randn(1, T-1, embed_dim)` |
| 2 | `spatial_attention.py` L53 | `x_out = x_attended.transpose(1, 2)` then `x_out.view(B, C, H, W)` | `x_out = x_attended.transpose(1, 2).contiguous()` then `x_out.view(B, C, H, W)` |
| 3 | `tests/test_graph.py` | `out = model(x, edge_index, batch)` with unsorted edges | `sort_idx = edge_index[1].argsort(); edge_index = edge_index[:, sort_idx]` before forward call |

---

## 4. Non-Failure Warnings

| # | Warning | Source | Frequency | Impact | Recommendation |
|---|---|---|:---:|---|---|
| W1 | `asyncio_default_fixture_loop_scope` unset | `pytest-asyncio 0.25.0` | 1× | None | Add `asyncio_mode = auto` in `pytest.ini` |
| W2 | `torch.jit.script` is deprecated | `torch 2.13.0` | 2× | None | Internal PyG usage — not user code |
| W3 | `typing._eval_type` `type_params` missing | `torch_geometric 2.8.0` + Python 3.13 | 328× | None | PyG upstream compatibility issue — awaiting fix in future PyG release |

---

## 5. Final Run Results

| Metric | Value |
|---|:---:|
| ✅ Passed | **47** |
| ❌ Failed | **0** |
| ⏭️ Skipped | **0** |
| ⚠️ Warnings | **330** |
| ⏱️ Duration | **22.20s** |
| 🐍 Python | **3.13.7** |
| 🔬 PyTest | **8.3.4** |
| 🔥 PyTorch | **2.13.0** |
| 📡 PyG | **2.8.0** |
