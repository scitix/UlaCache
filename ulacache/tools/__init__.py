"""Optional offline and benchmark utilities; device modules load on demand."""

from .snapshot.format import (
    SnapshotLayout,
    SnapshotRequest,
    SnapshotReader,
    SnapshotWriter,
    finalize_snapshot,
)

__all__ = [
    "SnapshotLayout",
    "SnapshotRequest",
    "SnapshotReader",
    "SnapshotWriter",
    "finalize_snapshot",
]
