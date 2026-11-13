"""Stage 1: fit the solar disk and flat-field every frame, once.

    python scripts/01_preprocess.py

Writes ``<work>/cache/<subset>/<stem>.npy`` and ``<work>/cache/<subset>_disks.json``.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filaseg import preprocess  # noqa: E402

if __name__ == "__main__":
    for subset in ("train", "test"):
        started = time.time()
        disks = preprocess.build_cache(subset)
        print(f"{subset}: {len(disks)} frames cached in {time.time() - started:.0f} s")
