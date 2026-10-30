"""Rewrite doc table references to match paper.aux. Companion to check_crossrefs.py."""
import os
import re
import sys

from check_crossrefs import DOCS, EXPECTED, numbers


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    nums = numbers("paper.aux")
    total = 0
    for doc in DOCS:
        if not os.path.exists(doc):
            continue
        lines = open(doc).read().splitlines(keepends=True)
        changed = 0
        for i, line in enumerate(lines):
            for heading, (first, last) in EXPECTED.items():
                if heading not in line:
                    continue
                want_first = nums.get(first)
                want_last = nums.get(last) if last else None
                if not want_first:
                    continue
                repl = ("Tables {}–{}".format(want_first, want_last)
                        if want_last else "Table {}".format(want_first))
                new = re.sub(r"Tables?\s+[IVXL]+(?:\s*[-–]+\s*[IVXL]+)?",
                             repl, line, count=1)
                if new != line:
                    lines[i] = new
                    changed += 1
        if changed:
            open(doc, "w").writelines(lines)
        print("{}: {} updated".format(os.path.basename(doc), changed))
        total += changed
    print("{} reference(s) rewritten".format(total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
