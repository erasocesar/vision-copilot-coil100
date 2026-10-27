"""Descarga (si hace falta), verifica y extrae COIL-100.

Uso: python data/download.py [--zip RUTA_LOCAL] [--out data/raw]
"""
import argparse
import hashlib
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
SHA256 = "fcb4b8b01c0778f416d8d7a09f45082c4f7075cc74743dcddd5c44307bd4afe7"
NAME_RE = re.compile(r"^coil-100/obj(\d{1,3})__(\d{1,3})\.png$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=Path("data/coil-100.zip"))
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    if not args.zip.exists():
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando {URL}")
        urllib.request.urlretrieve(URL, args.zip)

    digest = sha256_file(args.zip)
    if digest != SHA256:
        print(f"SHA-256 no coincide: {digest}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(args.zip) as zf:
        for info in zf.infolist():
            # Solo las imágenes con el patrón esperado; nada de rutas arbitrarias.
            if not NAME_RE.match(info.filename):
                continue
            target = args.out / Path(info.filename).name
            target.write_bytes(zf.read(info))
            n += 1
    print(f"{n} imágenes extraídas en {args.out} (SHA-256 {digest})")
    return 0 if n == 7200 else 1


if __name__ == "__main__":
    sys.exit(main())
