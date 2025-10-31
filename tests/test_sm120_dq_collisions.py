import pathlib
import re
import subprocess
import sys

RE_COLLISION = re.compile(
    r"warp=(\d+)/lane=(\d+)/elem=(\d+)/dst=(\d+)"
)


def run_probe(binary: pathlib.Path):
    result = subprocess.run(
        [str(binary)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout


def collect_hits(output: str):
    hits = {}
    for match in RE_COLLISION.finditer(output):
        warp, lane, elem, dst = map(int, match.groups())
        hits.setdefault(dst, set()).add((warp, lane))
    return hits


def main():
    repo_root = pathlib.Path(__file__).resolve().parents[1] / "external" / "FlashMLA"
    binary = repo_root / "build" / "sm120_copy_index_test"
    if not binary.exists():
        print(f"probe binary {binary} missing; build it first.", file=sys.stderr)
        return 1

    output = run_probe(binary)
    hits = collect_hits(output)

    collisions = {
        column: sorted(list(visitors))
        for column, visitors in hits.items()
        if len(visitors) > 1
    }

    total_hits = sum(len(v) for v in hits.values())
    unique_columns = len(hits)

    print(f"Observed {unique_columns} TMEM columns across {total_hits} thread hits.")

    if not collisions:
        print("No collisions detected.")
        return 0

    print(f"Detected {len(collisions)} colliding columns:")
    for column, visitors in sorted(collisions.items()):
        formatted = ", ".join(f"(warp {w}, lane {l})" for w, l in visitors)
        print(f"  column {column}: {formatted}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
