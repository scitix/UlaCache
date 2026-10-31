"""Borrow eager forward metadata while retaining captured device lifetimes."""

from dataclasses import dataclass, field
from weakref import ref


@dataclass(frozen=True, init=False)
class CPScorePlan:
    bucket: object
    lengths: object
    schedule: object
    _epoch_reference: object = field(repr=False)

    def __init__(self, bucket, lengths, schedule, epoch, *, retain_epoch=True):
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "schedule", schedule)
        reference = None
        if not retain_epoch:
            try:
                reference = ref(epoch)
            except TypeError:
                # Preserve opaque non-weak-referenceable epoch tags. Native
                # metadata that owns its plan must support weak references.
                pass
        if reference is None:

            def reference():
                return epoch

        object.__setattr__(self, "_epoch_reference", reference)

    @property
    def epoch(self):
        return self._epoch_reference()
