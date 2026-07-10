#!/usr/bin/env python

import argparse
import os
import re
import json
import csv

def parse_txt_to_csv(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []
    current_topk = None
    current_method = None

    pattern_topk = re.compile(r'Processing with TOP_K:\s*(\d+)')
    pattern_method = re.compile(r'Evaluate pose_estimation methods:\s*(\S+)')

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        topk_match = pattern_topk.match(line)
        if topk_match:
            current_topk = int(topk_match.group(1))
            i += 1
            continue

        method_match = pattern_method.match(line)
        if method_match:
            current_method = method_match.group(1)
            i += 1
            # Parse the JSON block starting on the next line
            json_lines = []
            while i < len(lines) and not lines[i].startswith('Evaluate') and not lines[i].startswith('Processing'):
                json_lines.append(lines[i])
                i += 1
            try:
                metrics = json.loads(''.join(json_lines))
                for metric in metrics.keys():
                    if 'AUC' in metric or 'Precision' in metric:
                        metrics[metric] = metrics[metric] * 100

                # Flatten row
                row = {
                    "TOP_K": current_topk,
                    "Method": current_method
                }
                row.update({k: f"{v:.2f}" if isinstance(v, float) else v for k, v in metrics.items()})
                results.append(row)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON near line {i}")
        else:
            i += 1

    if results:
        # Output CSV path
        base, _ = os.path.splitext(txt_file)
        csv_file = base + ".csv"
        keys = results[0].keys()
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV file written to: {csv_file}")
    else:
        print("No data found to write to CSV.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert evaluation .txt file to .csv format.")
    parser.add_argument("--txt_path", help="Path to the input .txt file.")
    args = parser.parse_args()
    parse_txt_to_csv(args.txt_path)
