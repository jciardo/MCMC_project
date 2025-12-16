from pathlib import Path

def _normalize_axis(values, N: int, axis_name: str):
    mn, mx = min(values), max(values)

    # Case: already 0-based [0..N-1]
    if 0 <= mn and mx <= N - 1:
        return list(values)

    # Case: 1-based [1..N] -> convert to 0-based
    if 1 <= mn and mx <= N:
        return [v - 1 for v in values]

    # Anything else is suspicious
    raise ValueError(
        f"Axis {axis_name} out of expected range for N={N}. "
        f"min={mn}, max={mx}, expected either [0..{N-1}] or [1..{N}]."
    )

def normalize_positions(positions, N: int):
    # positions is e.g. history["best_positions"] = list(state.iter_queens())
    pos = [(int(x), int(y), int(z)) for (x, y, z) in positions]

    if len(pos) != N * N:
        raise ValueError(f"Expected {N*N} queens, got {len(pos)}")

    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    zs = [p[2] for p in pos]

    xs = _normalize_axis(xs, N, "x")
    ys = _normalize_axis(ys, N, "y")
    zs = _normalize_axis(zs, N, "z")

    norm = list(zip(xs, ys, zs))

    # Final sanity check
    for (x, y, z) in norm:
        if not (0 <= x < N and 0 <= y < N and 0 <= z < N):
            raise ValueError(
                f"Still out-of-range after normalization: {(x,y,z)} for N={N}"
            )

    return norm

def write_queens_xyz(path: str | Path, positions, N: int) -> None:
    norm = normalize_positions(positions, N)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x, y, z in norm:
            f.write(f"{x},{y},{z}\n")
