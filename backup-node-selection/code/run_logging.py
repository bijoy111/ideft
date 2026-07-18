"""Save experiment output to the findings/ folder as well as the terminal.

Each measurement script calls capture_to_findings("<label>") once after parsing its
arguments. Everything printed from then on goes to both the terminal and
../findings/<label>.txt, opened in write mode so re-running a scenario replaces its
previous results instead of appending.

findings/ is resolved relative to this file, not the working directory, so it lands in
the repo wherever the script is launched from. The torch CUDA warning goes to stderr, so
it stays out of the captured files.
"""
import atexit
import datetime
import os
import sys

FINDINGS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "findings")
)


class _Tee:
    """Writes to several streams at once, flushing each write so the captured file is
    complete even if the run is interrupted."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def findings_path(label):
    """Absolute path of the findings file for a given scenario label."""
    return os.path.join(FINDINGS_DIR, "{}.txt".format(label))


def capture_to_findings(label):
    """Mirror stdout to findings/<label>.txt (overwrite), with a header recording the
    command and timestamp. Returns the file path and restores stdout at exit."""
    os.makedirs(FINDINGS_DIR, exist_ok=True)
    path = findings_path(label)
    fh = open(path, "w")

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    fh.write("# iDEFT findings: {}\n".format(label))
    fh.write("# command: {}\n".format(" ".join(sys.argv)))
    fh.write("# generated: {}\n".format(stamp))
    fh.write("# (reproducing this scenario overwrites this file)\n\n")
    fh.flush()

    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, fh)

    def _restore():
        sys.stdout = real_stdout
        fh.close()

    atexit.register(_restore)
    return path


def write_findings(label, text):
    """Write a block of text straight to findings/<label>.txt (overwrite), for callers
    that want to record one result without capturing a whole run."""
    os.makedirs(FINDINGS_DIR, exist_ok=True)
    path = findings_path(label)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(path, "w") as fh:
        fh.write("# iDEFT findings: {}\n".format(label))
        fh.write("# command: {}\n".format(" ".join(sys.argv)))
        fh.write("# generated: {}\n\n".format(stamp))
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
    return path
