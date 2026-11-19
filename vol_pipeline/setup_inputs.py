"""Copy the pipeline's inputs into data/raw/ (run once).

    python setup_inputs.py                 # copy from the team's data checkout around this folder
    python setup_inputs.py --from-cache /path/to/drive/cache   # WRDS cache downloaded elsewhere

On macOS (APFS) the copies are clones: they take no extra disk space.
Existing files in data/raw/ are left alone unless --force is given.
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":   # APFS clone, falls back to a normal copy on other volumes
        subprocess.run(["cp", "-cR" if src.is_dir() else "-c", str(src), str(dst)], check=True)
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = sys.argv[1:]
    sources = dict(C.SETUP_SOURCES)
    if "--from-cache" in args:
        cache = Path(args[args.index("--from-cache") + 1]).expanduser().resolve()
        sources.update({C.CACHE / n: cache / n for n in C.CACHE_ITEMS})
    force = "--force" in args

    missing = []
    for dst, src in sources.items():
        if dst.exists() and not force:
            print(f"  exists   {dst.relative_to(C.ROOT)}")
            continue
        if not src.exists():
            missing.append(str(src))
            continue
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        copy(src, dst)
        print(f"  copied   {dst.relative_to(C.ROOT)}  <-  {src}")
    if missing:
        sys.exit("not found (copy the WRDS cache from the data folder: 3_4_surprise_returns/cache):\n  "
                 + "\n  ".join(missing))
    print("inputs ready in", C.RAW)


if __name__ == "__main__":
    main()
