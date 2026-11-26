"""Repair the OPF encoding only when it matches the native parser error."""

import re
import zipfile
from xml.etree import ElementTree as ET


def fix_opf(original, target, error):
    if "parse opf" not in error.lower() or "charsetreader is nil" not in error.lower():
        raise ValueError("The reported failure is not an OPF encoding error.")
    with zipfile.ZipFile(original) as src:
        container = ET.fromstring(src.read("META-INF/container.xml"))
        member = container.find(".//{*}rootfile").attrib["full-path"]
        raw = src.read(member)
        decl = re.match(
            rb"""<\?xml\s+[^?]*?encoding\s*=\s*["']([a-zA-Z0-9._-]+)["']""", raw
        )
        if not decl or decl[1].decode().lower() not in error.lower():
            raise ValueError(
                "The local XML declaration differs from the parser diagnostic."
            )
        encoding = decl[1].decode()
        updated = raw[: decl.start(1)] + b"UTF-8" + raw[decl.end(1) :]
        updated = updated.decode(encoding).encode("utf-8")
        with zipfile.ZipFile(target, "x") as dst:
            for info in src.infolist():
                dst.writestr(
                    info, updated if info.filename == member else src.read(info)
                )
    return member, encoding
