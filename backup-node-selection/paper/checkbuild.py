"""Report LaTeX build problems from paper.log.

Written because `grep -c` on this machine prints nothing and exits 1 even when a
pattern is absent, so a build check of the form `grep -c '^!' paper.log` reports
an empty string rather than a count and reads as success whatever happened. Two
real defects -- a tabular with four declared columns and five actual ones, and
seven overfull boxes -- survived several such checks before this script existed.
"""
import re
import sys

log = open(sys.argv[1] if len(sys.argv) > 1 else "paper.log",
           encoding="utf-8", errors="replace").read().splitlines()

errors = [l for l in log if l.startswith("!")]
overfull = [l for l in log if "Overfull" in l]
undefined = [l for l in log if re.search(r"[Uu]ndefined (reference|citation|control)", l)]
multiply = [l for l in log if "Multiply" in l or "multiply defined" in l]
pages = re.search(r"Output written on \S+ \((\d+) pages", "\n".join(log))

print("pages:            {}".format(pages.group(1) if pages else "NO PDF PRODUCED"))
print("errors:           {}".format(len(errors)))
print("undefined refs:   {}".format(len(undefined)))
print("multiply-defined: {}".format(len(multiply)))
print("overfull boxes:   {}".format(len(overfull)))
for label, items in (("ERROR", errors), ("UNDEFINED", undefined),
                     ("MULTIPLY", multiply), ("OVERFULL", overfull)):
    for line in items[:12]:
        print("  [{}] {}".format(label, line[:120]))
sys.exit(1 if (errors or undefined or multiply) else 0)
