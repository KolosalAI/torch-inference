#!/usr/bin/env python3
"""Print a one-line coverage summary from an lcov.info file."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "coverage/lcov.info"
total = hit = 0
with open(path) as f:
    for line in f:
        if line.startswith("DA:"):
            parts = line[3:].split(",")
            if len(parts) >= 2:
                total += 1
                if int(parts[1].strip()) > 0:
                    hit += 1

if total == 0:
    print("No coverage data found.")
    sys.exit(1)

pct = hit / total * 100
print(f"Coverage: {pct:.2f}% ({hit}/{total} lines)")
if pct < 95:
    print(f"FAIL: below 95% target (gap: {95 - pct:.2f}%)")
    sys.exit(1)
else:
    print("PASS: ≥95% target met")
