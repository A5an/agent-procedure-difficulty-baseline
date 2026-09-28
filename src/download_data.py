"""Fetch the raw benchmark releases into external_data/ (about 6 GB on disk) and verify them.

Everything comes from the original public sources listed in data_manifest.tsv:
  SOPBench        github.com/Leezekun/SOPBench, commit d2622008 (task data and the released runs)
  tau2-bench      github.com/sierra-research/tau2-bench, commit b7ea9074 (task files) and the public
                  leaderboard bucket sierra-tau-bench-public (one results file per submission)

Every file is checked against the manifest: git blob hash for files from GitHub, SHA-256 for the
leaderboard files. Downloads resume after an interruption; rerunning skips verified files.

  python src/download_data.py [sopbench] [tau2] [banking]      default: all three
  python src/download_data.py --verify                         check what is on disk, download nothing
"""

import hashlib
import shutil
import sys
import tarfile
import time
import urllib.request

from common import EXTERNAL, ROOT

GROUPS = {"sopbench": "SOPBench/", "tau2": "tau2-domains/", "banking": "tau2-banking/"}


def manifest():
    rows = []
    for line in (ROOT / "data_manifest.tsv").read_text().splitlines()[1:]:
        kind, path, url, size, h = line.split("\t")
        rows.append({"kind": kind, "path": path, "url": url, "size": int(size) if size else None, "hash": h})
    return rows


def digest(path, kind):
    if kind == "gitsha1":
        b = path.read_bytes()
        return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def ok(row):
    p = EXTERNAL / row["path"]
    if not p.exists() or (row["size"] is not None and p.stat().st_size != row["size"]):
        return False
    kind, want = row["hash"].split(":", 1)
    return digest(p, kind) == want


def fetch(url, dest, size=None):
    """Download with resume (HTTP Range) and retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(6):
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have == size:
            break
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(part, mode) as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            if size is None:
                break
        except OSError as e:
            print(f"  retry {attempt + 1} after error: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    part.rename(dest)


def main(args):
    verify_only = "--verify" in args
    groups = [a for a in args if a in GROUPS] or list(GROUPS)
    rows = [r for r in manifest() if any(r["path"].startswith(GROUPS[g]) for g in groups)]
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    for r in rows:
        if r["kind"] != "archive" or verify_only:
            continue
        target = EXTERNAL / r["path"]
        checks = [c for c in rows if c["kind"] == "check" and c["path"].startswith(r["path"] + "/")]
        if target.exists() and all(ok(c) for c in checks):
            print(f"{r['path']}: present and verified")
            continue
        tgz = EXTERNAL / "_downloads" / (r["path"] + ".tar.gz")
        if not tgz.exists():
            print(f"{r['path']}: downloading {r['url']}", flush=True)
            fetch(r["url"], tgz)
        print(f"{r['path']}: extracting", flush=True)
        tmp = EXTERNAL / "_downloads" / (r["path"] + "_extract")
        shutil.rmtree(tmp, ignore_errors=True)
        with tarfile.open(tgz) as t:
            t.extractall(tmp)
        (top,) = list(tmp.iterdir())
        shutil.rmtree(target, ignore_errors=True)
        top.rename(target)
        shutil.rmtree(tmp)
    bad = []
    files = [r for r in rows if r["kind"] == "file"]
    for i, r in enumerate(files, 1):
        if ok(r):
            continue
        if verify_only:
            bad.append(r["path"])
            continue
        print(f"[{i}/{len(files)}] {r['path']} ({r['size'] / 1e6:.0f} MB)", flush=True)
        fetch(r["url"], EXTERNAL / r["path"], r["size"])
        if not ok(r):
            bad.append(r["path"])
    bad += [r["path"] for r in rows if r["kind"] == "check" and not ok(r)]
    n = sum(r["kind"] != "archive" for r in rows)
    print(f"verified {n - len(bad)} of {n} files" + (f"; FAILED: {bad[:10]}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
