"""
Dataset audit script for GlacierGuard-AI.
Computes SHA-256 hashes, file sizes, formats, and detects duplicate files.
"""
import hashlib
from pathlib import Path
import geopandas as gpd
import xarray as xr
import pandas as pd

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    root = Path('.')
    print("=" * 80)
    print("AUDITING LOCAL DATASETS IN GlacierGuard-AI")
    print("=" * 80)

    raw_dir = root / 'raw'
    data_dir = root / 'data'
    
    all_files = []
    for d in [raw_dir, data_dir]:
        if d.exists():
            for f in sorted(d.rglob('*')):
                if f.is_file():
                    all_files.append(f)
                    
    print(f"Total files found in raw/ and data/: {len(all_files)}")
    
    seen_hashes = {}
    duplicates = []
    
    for f in all_files:
        size = f.stat().st_size
        sha = compute_sha256(f)
        rel = f.relative_to(root).as_posix()
        print(f"{size:>10} bytes | {sha[:16]}... | {rel}")
        if sha in seen_hashes:
            print(f"  [!] DUPLICATE: Exactly identical to {seen_hashes[sha]}")
            duplicates.append((rel, seen_hashes[sha], sha))
        else:
            seen_hashes[sha] = rel

    print("\n" + "=" * 80)
    print("DUPLICATES SUMMARY:")
    for d, orig, sha in duplicates:
        print(f"  - {d} is duplicate of {orig} (hash {sha})")
        
    print("=" * 80)

if __name__ == '__main__':
    main()
