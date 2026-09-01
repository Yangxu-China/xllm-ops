#!/usr/bin/env python3
# coding: utf-8
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Merge profiling op_summary CSV with test stdout shape info.

The profiling CSV has per-op timing but no shape info (all N/A).  The test
stdout has shape info (Hq, Hkv, beam, prompt, maxDs, dstep, ...) but no
timing.  This script merges them by row order: the Nth OP-MATRIX-V2 line
in the log corresponds to the Nth data row in the CSV.

Usage:
  python3 x_attention_v2_jit_merge_prof.py --log <run.log> --csv <op_summary.csv> -o <output.csv>

The output CSV is written to the profiling directory (same as --csv dir by default,
or override with -o).

Output columns:
  Hq,Hkv,grp,batch,beam,prompt,maxDs,dstep,dtype,diff,
  Task_Duration_us,aicore_time_us,aic_mte2_ratio,aic_mac_ratio,
  aic_scalar_ratio,aic_mte1_ratio,aic_fixpipe_ratio,cube_utilization,
  aiv_vec_ratio,aiv_scalar_ratio,aiv_mte2_ratio,aiv_mte3_ratio
"""

import argparse
import csv
import os
import re
import sys


LOG_PATTERN = re.compile(
    r"OP-MATRIX-V2\s+"
    r"Hq=(\d+)\s+"
    r"Hkv=(\d+)\s+"
    r"d=\d+\s+"
    r"b=(\d+)\s+"
    r"beam=(\d+)\s+"
    r"prompt=(\d+)\s+"
    r"maxDs=(\d+)\s+"
    r"dstep=(\d+)\s+"
    r"(?:S=(\d+)\s+U=(\d+)\s+)?"          # optional: newer logs omit S/U
    r"dtype=(\S+):\s+"
    r"max\|dev-gold\|=([\d.e+-]+)"
)

LOG_FIELDS = ["Hq", "Hkv", "grp", "batch", "beam", "prompt", "maxDs", "dstep",
              "shared_cores", "unshared_cores", "dtype", "diff"]

OP_NAME_FILTER = "x_attention"

CSV_FIELDS = [
    ("Task Duration(us)", "Task_Duration_us"),
    ("aicore_time(us)", "aicore_time_us"),
    ("Block Num", "block_num"),
    ("aic_mte2_ratio", "aic_mte2_ratio"),
    ("aic_mac_ratio", "aic_mac_ratio"),
    ("aic_scalar_ratio", "aic_scalar_ratio"),
    ("aic_mte1_ratio", "aic_mte1_ratio"),
    ("aic_fixpipe_ratio", "aic_fixpipe_ratio"),
    ("cube_utilization(%)", "cube_utilization"),
    ("aiv_vec_ratio", "aiv_vec_ratio"),
    ("aiv_scalar_ratio", "aiv_scalar_ratio"),
    ("aiv_mte2_ratio", "aiv_mte2_ratio"),
    ("aiv_mte3_ratio", "aiv_mte3_ratio"),
]


def parse_log(log_path):
    """Extract shape info from OP-MATRIX-V2 lines in the test stdout log."""
    entries = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if m:
                hq, hkv, batch, beam, prompt, maxDs, dstep, S, U, dtype, diff = m.groups()
                hq = int(hq)
                hkv = int(hkv)
                grp = hq // hkv
                entries.append({
                    "Hq": int(hq),
                    "Hkv": int(hkv),
                    "grp": int(grp),
                    "batch": int(batch),
                    "beam": int(beam),
                    "prompt": int(prompt),
                    "maxDs": int(maxDs),
                    "dstep": int(dstep),
                    "shared_cores": int(S) if S is not None else "",
                    "unshared_cores": int(U) if U is not None else "",
                    "dtype": dtype,
                    "diff": float(diff),
                })
    return entries


def parse_csv(csv_path):
    """Extract timing fields from the profiling op_summary CSV."""
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    filtered = [r for r in rows if OP_NAME_FILTER in r.get("Op Name", "")]
    if not filtered:
        print(f"WARNING: no rows matching '{OP_NAME_FILTER}' in Op Name; using all rows",
              file=sys.stderr)
        filtered = rows
    out = []
    for r in filtered:
        row = {}
        for raw_key, out_key in CSV_FIELDS:
            val = r.get(raw_key, "")
            val = val.strip() if val else ""
            try:
                row[out_key] = float(val)
            except (ValueError, TypeError):
                row[out_key] = val
        out.append(row)
    return out


def merge(log_entries, csv_rows):
    """Merge by row order: log_entries[i] <-> csv_rows[i]."""
    if len(log_entries) != len(csv_rows):
        print(f"WARNING: row count mismatch: log={len(log_entries)} csv={len(csv_rows)}", file=sys.stderr)
        print("  Merging by min(log, csv) rows. Extra rows are dropped.", file=sys.stderr)
    n = min(len(log_entries), len(csv_rows))
    merged = []
    for i in range(n):
        row = {}
        row.update(log_entries[i])
        row.update(csv_rows[i])
        merged.append(row)
    return merged


def write_csv(merged, out_path):
    out_fields = LOG_FIELDS + [out_key for _, out_key in CSV_FIELDS]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in merged:
            writer.writerow({k: row.get(k, "") for k in out_fields})
    print(f"Merged {len(merged)} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge profiling CSV with test shape info")
    parser.add_argument("--log", required=True, help="Path to test stdout log file")
    parser.add_argument("--csv", required=True, help="Path to op_summary_*.csv")
    parser.add_argument("-o", "--output", default=None,
                        help="Output CSV path (default: same dir as --csv, named merged_summary.csv)")
    args = parser.parse_args()

    log_entries = parse_log(args.log)
    if not log_entries:
        print(f"ERROR: no OP-MATRIX-V2 lines found in {args.log}", file=sys.stderr)
        sys.exit(1)
    print(f"Parsed {len(log_entries)} log entries from {args.log}")

    csv_rows = parse_csv(args.csv)
    if not csv_rows:
        print(f"ERROR: no data rows in {args.csv}", file=sys.stderr)
        sys.exit(1)
    print(f"Parsed {len(csv_rows)} CSV rows from {args.csv}")

    merged = merge(log_entries, csv_rows)

    if args.output:
        out_path = args.output
    else:
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        out_path = os.path.join(csv_dir, "merged_summary.csv")

    write_csv(merged, out_path)


if __name__ == "__main__":
    main()
