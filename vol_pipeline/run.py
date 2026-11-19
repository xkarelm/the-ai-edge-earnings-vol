"""Build the post-earnings volatility dataset.

    python run.py                 # build everything (reuses the SUE table if present)
    python run.py --rebuild-sue   # also recompute SUE from I/B/E/S (about 2 minutes)

Run setup_inputs.py once first. Output goes to data/processed/, and the full
printed log to data/processed/build_log.txt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C  # noqa: E402


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main() -> None:
    missing = [str(p) for p in [C.EVENT_FILE, C.CALENDAR_FILE] + [C.CACHE / n for n in C.CACHE_ITEMS]
               if not p.exists()]
    if missing:
        sys.exit("missing inputs (run `python setup_inputs.py` first):\n  " + "\n  ".join(missing))
    from pipeline import build
    C.OUT.mkdir(parents=True, exist_ok=True)
    with open(C.OUT_LOG, "w") as log:
        out = sys.stdout
        sys.stdout = Tee(out, log)
        try:
            build.run(rebuild_sue="--rebuild-sue" in sys.argv)
        finally:
            sys.stdout = out
    print(f"\nlog: {C.OUT_LOG}")


if __name__ == "__main__":
    main()
