"""Stable fingerprint shared by notebook execution and website validation."""

import hashlib
import json


def source_hash(notebook):
    cells = [(cell.cell_type, cell.source) for cell in notebook.cells]
    return hashlib.sha256(json.dumps(cells, ensure_ascii=False).encode()).hexdigest()
