from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest


CORE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties
 xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/">
 <dc:creator>private-user</dc:creator>
 <cp:lastModifiedBy>private-user</cp:lastModifiedBy>
</cp:coreProperties>"""
PERSON_XML = """<?xml version="1.0" encoding="UTF-8"?>
<xltc:personList xmlns:xltc="http://schemas.microsoft.com/office/spreadsheetml/2018/threadedcomments">
 <xltc:person displayName="private-user" id="{PERSON-ID}" />
</xltc:personList>"""
COMMENT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<x:comments xmlns:x="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
 <x:authors><x:author>private-user</x:author></x:authors>
</x:comments>"""


def _write_fixture(path: Path, relationship: str | None = None) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("docProps/core.xml", CORE_XML)
        archive.writestr("xl/persons/person.xml", PERSON_XML)
        archive.writestr("xl/comments1.xml", COMMENT_XML)
        archive.writestr("xl/workbook.xml", "<workbook>keep-me</workbook>")
        if relationship is not None:
            archive.writestr("xl/_rels/workbook.xml.rels", relationship)


def test_sanitizer_replaces_personal_metadata_and_preserves_other_xml(tmp_path: Path):
    from scripts.sanitize_workbook import PUBLIC_AUTHOR, sanitize_workbook

    source = tmp_path / "source.xlsx"
    destination = tmp_path / "public.xlsx"
    _write_fixture(source)

    sanitize_workbook(source, destination)

    with ZipFile(destination) as archive:
        combined = b"\n".join(archive.read(name) for name in archive.namelist())
        assert b"private-user" not in combined
        assert PUBLIC_AUTHOR.encode() in combined
        assert archive.read("xl/workbook.xml") == b"<workbook>keep-me</workbook>"


def test_sanitizer_rejects_local_file_relationship(tmp_path: Path):
    from scripts.sanitize_workbook import sanitize_workbook

    source = tmp_path / "source.xlsx"
    destination = tmp_path / "public.xlsx"
    local_target = "/" + "Users/example/Desktop/input.xlsx"
    relationship = f'<Relationship Target="{local_target}" />'
    _write_fixture(source, relationship)

    with pytest.raises(ValueError, match="external local path"):
        sanitize_workbook(source, destination)

    assert not destination.exists()


def test_sanitizer_never_overwrites_the_source(tmp_path: Path):
    from scripts.sanitize_workbook import sanitize_workbook

    source = tmp_path / "source.xlsx"
    _write_fixture(source)

    with pytest.raises(ValueError, match="different paths"):
        sanitize_workbook(source, source)
