"""Fetch the numerical members of the public SND specimen ZIPs via HTTP Range.

The specimen archives also contain large DIC images and point clouds. Only
DOFS, heat-pinching and loading records are read from those archives. The small
MaterialData archive is read in full to retain the original test metadata.
ZIP CRC checking is performed by zipfile when a member is read in full.
"""

import argparse
import io
import json
from pathlib import Path
import zipfile

import requests


BASE = "https://chalmers.open.care.snd.se/downloads/public/2023-234/1/"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "jansson2024"


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.position = 0
        response = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=90)
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f"Server did not honor byte range for {url}")
        self.size = int(response.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        bases = {0: 0, 1: self.position, 2: self.size}
        position = bases[whence] + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.position + size, self.size)
        if self.position >= end:
            return b""
        requested = f"bytes={self.position}-{end - 1}"
        response = self.session.get(self.url, headers={"Range": requested}, timeout=180)
        response.raise_for_status()
        expected = f"bytes {self.position}-{end - 1}/{self.size}"
        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
            raise RuntimeError(f"Unexpected range response for {requested}")
        if len(response.content) != end - self.position:
            raise RuntimeError("Truncated byte range")
        self.position = end
        return response.content

    def close(self):
        self.session.close()
        super().close()


def fetch(specimen, list_only=False):
    url = BASE + f"data/{specimen}.zip"
    with RangeReader(url) as remote, zipfile.ZipFile(remote) as archive:
        members = [
            info for info in archive.infolist()
            if not info.is_dir()
            and (specimen == "MaterialData"
                 or "distributedoptical" in info.filename.lower()
                 or "loaddisplacement" in info.filename.lower())
        ]
        rows = []
        for info in members:
            print(f"{info.filename}: {info.file_size:,} bytes; compressed {info.compress_size:,}", flush=True)
            row = {"member": info.filename, "bytes": info.file_size, "crc32": f"{info.CRC:08x}"}
            if not list_only:
                target = (DEST / info.filename).resolve()
                if not target.is_relative_to(DEST.resolve()):
                    raise ValueError(f"Unsafe ZIP member: {info.filename}")
                if not target.exists():
                    payload = archive.read(info)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload)
                elif target.stat().st_size != info.file_size:
                    raise RuntimeError(f"Existing file has a different size: {target}")
                row["local_path"] = target.relative_to(ROOT).as_posix()
            rows.append(row)
        if not list_only:
            DEST.mkdir(parents=True, exist_ok=True)
            manifest = {"dataset_doi": "10.5878/dvcn-bg03", "version": 1,
                        "license": "CC0-1.0", "archive_url": url,
                        "archive_bytes": remote.size, "members": rows}
            (DEST / f"{specimen}_source.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specimens", nargs="+")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    for specimen in args.specimens:
        fetch(specimen, args.list_only)
