"""Create a public XLSX copy without personal or local-path metadata."""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PUBLIC_AUTHOR = "Hybrid Fund Allocation Research"
MAC_USER_PREFIX = "/" + "Users/"
BANNED_LOCAL_FRAGMENTS = (MAC_USER_PREFIX, "file://", "Desktop/")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _sanitize_metadata_xml(member: str, data: bytes) -> bytes:
    metadata_member = (
        member == "docProps/core.xml"
        or member.startswith("xl/comments")
        or member.startswith("xl/persons/")
    )
    if not metadata_member:
        return data
    root = ET.fromstring(data)
    for element in root.iter():
        if _local_name(element.tag) in {"creator", "lastModifiedBy", "author"}:
            element.text = PUBLIC_AUTHOR
        if "displayName" in element.attrib:
            element.set("displayName", PUBLIC_AUTHOR)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def sanitize_workbook(source: Path, destination: Path) -> None:
    """Write a sanitized workbook copy while preserving non-metadata members."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError("source and destination must be different paths")
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary = tempfile.NamedTemporaryFile(
        prefix=destination.stem + "-",
        suffix=".xlsx",
        dir=destination.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with ZipFile(source) as input_archive, ZipFile(
            temporary_path,
            "w",
            ZIP_DEFLATED,
        ) as output_archive:
            for info in input_archive.infolist():
                if info.filename.startswith("xl/externalLinks/"):
                    raise ValueError("workbook contains an external local path")
                data = input_archive.read(info.filename)
                if info.filename.endswith((".xml", ".rels")):
                    text = data.decode("utf-8", errors="replace")
                    if any(fragment in text for fragment in BANNED_LOCAL_FRAGMENTS):
                        raise ValueError("workbook contains an external local path")
                    data = _sanitize_metadata_xml(info.filename, data)
                output_archive.writestr(info, data)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    sanitize_workbook(args.source, args.destination)
    print(f"Sanitized workbook written to {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
