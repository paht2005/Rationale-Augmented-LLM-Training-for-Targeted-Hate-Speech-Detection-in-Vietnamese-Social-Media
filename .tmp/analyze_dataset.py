#!/usr/bin/env python3
"""Exact statistics from dataset_rationale.json for paper revision."""
import json
from collections import Counter

import os
DATA = os.path.join(os.path.dirname(__file__), "..", "dataset", "processed", "dataset_rationale.json")

with open(DATA, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"{'='*60}")
print(f"DATASET_RATIONALE.JSON — EXACT STATISTICS")
print(f"{'='*60}\n")

# 1. Total
print(f"1. TOTAL ENTRIES: {len(data)}\n")

# 2. Split by dataset field
splits = Counter(e.get("dataset", "MISSING") for e in data)
print("2. SPLIT BY 'dataset' FIELD:")
for k, v in splits.most_common():
    print(f"   {k}: {v}")
print()

# 3. Implied statement presence
train = [e for e in data if e.get("dataset") == "train"]
test  = [e for e in data if e.get("dataset") == "test"]

def has_implied(e):
    v = e.get("implied_statement")
    return v is not None and isinstance(v, str) and len(v.strip()) > 0

def has_rationale(e):
    v = e.get("rationale")
    return v is not None and isinstance(v, list) and len(v) > 0

train_with_impl   = [e for e in train if has_implied(e)]
train_no_impl     = [e for e in train if not has_implied(e)]
test_with_impl    = [e for e in test  if has_implied(e)]
test_no_impl      = [e for e in test  if not has_implied(e)]

print("3. TRAIN — IMPLIED_STATEMENT:")
print(f"   With implied_statement:    {len(train_with_impl)}")
print(f"   Without implied_statement: {len(train_no_impl)}")
print(f"   Total train:               {len(train)}\n")

print("4. TEST — IMPLIED_STATEMENT:")
print(f"   With implied_statement:    {len(test_with_impl)}")
print(f"   Without implied_statement: {len(test_no_impl)}")
print(f"   Total test:                {len(test)}\n")

# 5. Fully usable for Stage 2 (train + content + implied + rationale)
usable = [e for e in train if has_implied(e) and has_rationale(e)
          and e.get("content", "").strip()]
print("5. STAGE-2 USABLE (train + content + implied + rationale):")
print(f"   Count: {len(usable)}")
print(f"   % of train: {len(usable)/len(train)*100:.1f}%\n")

# 6. Label distribution — ALL entries
all_labels = []
for e in data:
    all_labels.extend(e.get("labels", []))
print("6. LABEL DISTRIBUTION — ALL ENTRIES:")
for label, count in Counter(all_labels).most_common():
    print(f"   {label}: {count}")
total_label_instances = sum(Counter(all_labels).values())
print(f"   --- Total label instances: {total_label_instances}")
multi = sum(1 for e in data if len(e.get("labels", [])) > 1)
print(f"   --- Multi-label entries: {multi} / {len(data)}\n")

# 7. Label distribution — USABLE TRAIN entries only
usable_labels = []
for e in usable:
    usable_labels.extend(e.get("labels", []))
print("7. LABEL DISTRIBUTION — STAGE-2 USABLE TRAIN:")
dist = Counter(usable_labels)
for label, count in dist.most_common():
    pct = count / len(usable) * 100
    print(f"   {label}: {count} ({pct:.1f}%)")
total_usable_labels = sum(dist.values())
hate_total = sum(v for k, v in dist.items() if k.endswith("#hate"))
off_total  = sum(v for k, v in dist.items() if k.endswith("#offensive"))
print(f"   --- Total label instances: {total_usable_labels}")
print(f"   --- Hate-level total: {hate_total} ({hate_total/total_usable_labels*100:.1f}%)")
print(f"   --- Offensive-level total: {off_total} ({off_total/total_usable_labels*100:.1f}%)")
usable_multi = sum(1 for e in usable if len(e.get("labels", [])) > 1)
print(f"   --- Multi-label entries: {usable_multi} / {len(usable)}\n")

# 8. Label distribution — TEST entries
test_labels = []
for e in test:
    test_labels.extend(e.get("labels", []))
print("8. LABEL DISTRIBUTION — TEST:")
for label, count in Counter(test_labels).most_common():
    print(f"   {label}: {count}")
print(f"   --- Total test label instances: {sum(Counter(test_labels).values())}\n")

# 9. Compare with paper's claim of 1,744
print("9. PAPER COMPARISON:")
print(f"   Paper claims: 1,744 tuples")
print(f"   Actual Stage-2 usable train: {len(usable)}")
print(f"   Actual total in file: {len(data)}")
print(f"   Train entries (all): {len(train)}")
print(f"   Discrepancy: paper says 1,744 vs actual usable = {len(usable)}")
print(f"   NOTE: The file also contains {len(test)} test entries")
print(f"         and {len(train_no_impl)} train entries without implied_statement")
print()
print(f"{'='*60}")
print("DONE")
