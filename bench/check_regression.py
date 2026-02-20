#!/usr/bin/env python3
import json
import sys

if len(sys.argv) != 3:
    print("usage: check_regression.py <baseline.json> <current.json>")
    sys.exit(2)

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    baseline = json.load(f)["binary_2048"]
with open(sys.argv[2], 'r', encoding='utf-8') as f:
    current = json.load(f)["binary_2048"]

latency_keys = ["scalar_ms", "avx2_ms", "avx512_naive_ms", "avx512_blocked_ms", "dispatch_st_ms", "dispatch_mt_ms"]
failed = False
for key in latency_keys:
    b = float(baseline.get(key, 0.0))
    c = float(current.get(key, 0.0))
    if b <= 0.0:
        continue
    if c > b * 1.05:
        print(f"regression: {key} baseline={b} current={c}")
        failed = True

# Cross-check dominance constraints when measurements exist.
if current["avx2_ms"] > current["scalar_ms"]:
    print("dominance failure: avx2 slower than scalar")
    failed = True
if current["dispatch_mt_ms"] > current["dispatch_st_ms"] * 1.15:
    print("dominance failure: multi-thread dispatch unexpectedly slower")
    failed = True
if current.get("avx512_blocked_ms", current["avx2_ms"]) > current.get("avx512_naive_ms", current["avx2_ms"]) * 1.10:
    print("dominance failure: blocked avx512 slower than naive avx512")
    failed = True

sys.exit(1 if failed else 0)
