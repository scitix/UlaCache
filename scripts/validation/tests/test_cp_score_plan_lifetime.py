"""Prevent eager metadata cycles without releasing captured device owners."""
import gc
import weakref

import pytest

from ulacache.indexer.plan import CPScorePlan


class NativeMetadata:
    pass


def test_eager_metadata_and_payload_retire_without_cyclic_gc():
    class Payload:
        pass
    was_enabled = gc.isenabled()
    gc.disable()
    references = []
    try:
        for _ in range(512):
            metadata = NativeMetadata()
            metadata.page_table = Payload()
            references.append(weakref.ref(metadata.page_table))
            metadata.cp_index_plan = CPScorePlan(None, None, None, metadata, retain_epoch=False)
            assert metadata.cp_index_plan.epoch is metadata
            del metadata
        assert all(reference() is None for reference in references)
    finally:
        if was_enabled:
            gc.enable()


def test_expired_eager_plan_cannot_match_a_new_native_epoch():
    metadata = NativeMetadata()
    plan = CPScorePlan(None, None, None, metadata, retain_epoch=False)
    metadata.cp_index_plan = plan
    del metadata
    assert plan.epoch is None
    assert plan.epoch is not NativeMetadata()


def test_captured_plan_keeps_native_device_owner_alive_until_release():
    metadata = NativeMetadata()
    reference = weakref.ref(metadata)
    plan = CPScorePlan(None, None, None, metadata, retain_epoch=True)
    del metadata
    assert plan.epoch is reference() and reference() is not None
    del plan
    assert reference() is None


@pytest.mark.parametrize('retain', [False, True])
def test_opaque_epoch_tags_remain_compatible_and_plan_is_frozen(retain):
    epoch = object()
    plan = CPScorePlan('bucket', 'lengths', 'schedule', epoch, retain_epoch=retain)
    assert plan.epoch is epoch
    assert (plan.bucket, plan.lengths, plan.schedule) == ('bucket', 'lengths', 'schedule')
    with pytest.raises(AttributeError):
        plan.bucket = 'replacement'
