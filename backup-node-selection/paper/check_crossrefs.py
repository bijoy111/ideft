"""
Check that the docs' claims about paper table and section numbers are true.

Table and section numbers are not stable: inserting one table renumbers every
later one, and inserting a subsection shifts the letters after it. The docs
(EXPERIMENTS.md, README.md, docs/reproduce.html) quote those numbers so a reader
can find the table a command produces, and every insertion silently falsifies
some of them. This reads the real numbers out of paper.aux and reports any claim
that no longer holds, so a stale pointer fails loudly instead of misdirecting
someone.

    python3 check_crossrefs.py          # from paper/
"""

import os
import re
import sys

AUX = "paper.aux"
DOCS = [
    "../EXPERIMENTS.md",
    "../README.md",
    "../MEASUREMENT_FIXES.md",
    "../../docs/reproduce.html",
]

# Which label each document reference is about, keyed by the heading text that
# introduces it. A reference we cannot attribute is reported separately rather
# than assumed correct.
EXPECTED = {
    "Selection-time overhead": ("tab:overhead", None),
    "Throughput and failure rate": ("tab:throughput16", None),
    "Heuristic weight sensitivity": ("tab:terms", None),
    "Hash baselines": ("tab:hrw", None),
    "Cross-algorithm check": ("tab:ppodqn", None),
    "Bare-metal validation": ("tab:cloudlab", "tab:loadcurve"),
    "Online selection under churn": ("tab:online-churn", "tab:online-alpha"),
    "Migration cost of real": ("tab:occupancy", "tab:crossover"),
}


def numbers(path):
    """label -> printed number, from the aux file."""
    out = {}
    with open(path) as handle:
        for line in handle:
            m = re.match(r"\\newlabel\{([^}]+)\}\{\{(?:\\mbox\s*)?\{?([IVXL]+(?:-[A-Z])?)\}?", line)
            if m:
                out.setdefault(m.group(1), m.group(2))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    if not os.path.exists(AUX):
        sys.stderr.write("{} not found -- build the paper first\n".format(AUX))
        return 2
    nums = numbers(AUX)

    problems = []
    for doc in DOCS:
        if not os.path.exists(doc):
            continue
        text = open(doc).read()
        for line in text.splitlines():
            for heading, (first, last) in EXPECTED.items():
                if heading not in line:
                    continue
                claimed = re.findall(r"Tables?\s+([IVXL]+)(?:\s*[-\u2013]+\s*([IVXL]+))?", line)
                if not claimed:
                    continue
                got_first, got_last = claimed[0]
                want_first = nums.get(first)
                want_last = nums.get(last) if last else None
                if want_first and got_first != want_first:
                    problems.append("{}: claims Table {} for {}, actual {}".format(
                        os.path.basename(doc), got_first, first, want_first))
                if want_last and got_last and got_last != want_last:
                    problems.append("{}: claims range ending {} for {}, actual {}".format(
                        os.path.basename(doc), got_last, last, want_last))

    if problems:
        print("STALE CROSS-REFERENCES ({})".format(len(problems)))
        for p in problems:
            print("  " + p)
        return 1
    print("cross-references ok ({} labels in {})".format(len(nums), AUX))
    return 0


if __name__ == "__main__":
    sys.exit(main())
