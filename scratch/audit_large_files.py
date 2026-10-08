import pathlib

root = pathlib.Path('.').resolve()
files = []
for p in root.rglob('*'):
    if p.is_file():
        # Exclude .git and .venv
        parts = p.parts
        if '.git' in parts or '.venv' in parts:
            continue
        try:
            files.append((p.relative_to(root), p.stat().st_size))
        except Exception:
            pass

files.sort(key=lambda x: x[1], reverse=True)

print(f"{'Relative Path':<70} | {'Size (MB)':>10} | {'Size (Bytes)':>12}")
print("-" * 98)
for rel, size in files[:40]:
    size_mb = size / (1024 * 1024)
    print(f"{str(rel):<70} | {size_mb:>10.2f} | {size:>12,}")
