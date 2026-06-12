#!/usr/bin/env python3
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def norm(vec):
    if vec is None:
        return None
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(float(x))]
    return None if not xs else sum(xs) / len(xs)


def quantile(xs, q):
    xs = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    if not xs:
        return None
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def bucket(value, edges):
    if value is None:
        return "missing"
    for edge in edges:
        if value <= edge:
            return f"<= {edge:g}"
    return f"> {edges[-1]:g}"


def summarize(records):
    rows = []
    for rec in records:
        rec = dict(rec)
        rec["init_xyz_norm"] = norm(rec.get("init_raw_peg_in_hole_xyz") or rec.get("init_peg_in_hole_xyz"))
        rec["final_xyz_norm"] = norm(rec.get("final_raw_peg_in_hole_xyz") or rec.get("final_peg_in_hole_xyz"))
        rec["min_xyz"] = rec.get("min_peg_in_hole_xyz")
        rows.append(rec)

    def block(name, subset):
        n = len(subset)
        if n == 0:
            return {"name": name, "n": 0}
        succ = [1.0 if r.get("success") else 0.0 for r in subset if r.get("success") is not None]
        return {
            "name": name,
            "n": n,
            "success": mean(succ),
            "abnormal": mean([1.0 if r.get("abnormal_robot") else 0.0 for r in subset if r.get("abnormal_robot") is not None]),
            "timeout": mean([1.0 if r.get("time_out") else 0.0 for r in subset if r.get("time_out") is not None]),
            "init_xyz_mean": mean([r.get("init_xyz_norm") for r in subset]),
            "init_xyz_p90": quantile([r.get("init_xyz_norm") for r in subset], 0.9),
            "min_xyz_mean": mean([r.get("min_xyz") for r in subset]),
            "final_xyz_mean": mean([r.get("final_xyz_norm") for r in subset]),
            "gripper_init_mean": mean([r.get("init_gripper_mean") for r in subset]),
            "teacher_policy_l2_mean": mean([r.get("teacher_policy_l2") for r in subset]),
            "teacher_policy_raw_l2_mean": mean([r.get("teacher_policy_raw_l2") for r in subset]),
            "episode_return_mean": mean([r.get("episode_return") for r in subset]),
            "episode_length_mean": mean([r.get("episode_length") for r in subset]),
        }

    out = {"overall": [], "by_init_xyz": [], "by_label_and_init_xyz": [], "failure_vs_success": []}
    labels = sorted(set(str(r.get("label", "unknown")) for r in rows))
    for label in labels:
        subset = [r for r in rows if str(r.get("label", "unknown")) == label]
        out["overall"].append(block(label, subset))
        out["failure_vs_success"].append(block(f"{label}:success", [r for r in subset if r.get("success") is True]))
        out["failure_vs_success"].append(block(f"{label}:fail", [r for r in subset if r.get("success") is False]))

    edges = [0.03, 0.06, 0.10, 0.16, 0.24, 0.35]
    buckets = defaultdict(list)
    for rec in rows:
        buckets[bucket(rec.get("init_xyz_norm"), edges)].append(rec)
        buckets[(str(rec.get("label", "unknown")), bucket(rec.get("init_xyz_norm"), edges))].append(rec)
    for key in sorted(k for k in buckets if isinstance(k, str)):
        out["by_init_xyz"].append(block(key, buckets[key]))
    for key in sorted(k for k in buckets if isinstance(k, tuple)):
        label, b = key
        out["by_label_and_init_xyz"].append(block(f"{label}:{b}", buckets[key]))
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: summarize_failure_strat.py RECORDS.jsonl [OUT.json]")
    path = Path(sys.argv[1])
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize(records)
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
