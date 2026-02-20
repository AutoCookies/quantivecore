#!/usr/bin/env python3
import json
import sys

if len(sys.argv) != 3:
    print("usage: check_regression.py <baseline.json> <current.json>")
    sys.exit(2)

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    baseline = json.load(f)
with open(sys.argv[2], 'r', encoding='utf-8') as f:
    current = json.load(f)

keys = ["scalar_ms", "avx2_ms", "dispatch_st_ms", "dispatch_mt_ms"]
failed = False
for key in keys:
    b = baseline["binary_2048"][key]
    c = current["binary_2048"][key]
    if b <= 0:
        continue
    if c > b * 1.05:
        print(f"regression: {key} baseline={b} current={c}")
        failed = True

sys.exit(1 if failed else 0)
