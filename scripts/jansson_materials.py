"""Read the cube-test machine records without loading the optical-data stack."""

import csv
from pathlib import Path

import numpy as np


def read_cube(path):
    """Read the machine schema, including the specimen-specific loaded area."""
    with Path(path).open(encoding="cp1252", newline="") as stream:
        rows = list(csv.reader(stream, delimiter=";"))
    start = next(i for i, row in enumerate(rows) if row[0] == "Piston stroke")
    metadata = {row[0]: row[1:] for row in rows[:start]}
    columns, units = rows[start:start + 2]
    values = np.asarray([[float(cell.replace(",", ".")) for cell in row]
                         for row in rows[start + 2:] if row], dtype=float)
    if columns != ["Piston stroke", "Standard force", "Test time"] or units != ["mm", "kN", "s"]:
        raise ValueError(f"Unsupported cube schema: {path}")
    dimensions = [float(metadata[key][0].replace(",", "."))
                  for key in ("Specimen thickness a", "Specimen width b")]
    return metadata, columns, units, values, float(np.prod(dimensions))


def read_wedge(path):
    """Read the actual five-channel WST acquisition schema, retaining force names."""
    with Path(path).open(encoding="cp1252") as stream:
        header = [stream.readline().rstrip("\r\n") for _ in range(5)]
        data = np.loadtxt((line.replace(",", ".") for line in stream), delimiter="\t")
    columns, units = header[3].split("\t"), header[4].split("\t")
    expected = ["Ch 1 CMOD", "Ch 1 Displacement", "Split_force", "Time", "Ch 1 Force"]
    if columns != expected or units != ["mm", "mm", "kN", "s", "kN"]:
        raise ValueError(f"Unsupported wedge-test schema: {path}")
    if data.shape[1] != len(columns) or not np.isfinite(data).all():
        raise ValueError(f"Invalid wedge-test numerical records: {path}")
    return columns, units, data
