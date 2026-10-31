"""Memory accounting against the recorded eight-H200 source_v6 component."""
from ulacache.storage.capacity import position_residency_bytes


def test_position_inventory_includes_mapping_buffers_and_shared_workspace_once():
    # Actual source_v6: two layers, one Index anchor, G4096/H4096, Q2/K2048.
    # The backend ring descriptor/bitmap measurement was178256 bytes.
    actual_local_bytes, mapping_bytes = 6060468, 32784
    # Consolidated main adds one null deferred-route pointer per group to
    # the source_v6 frame ABI. This is a prediction delta, not new GPU evidence.
    consolidated_frame_bytes = 8
    result=position_residency_bytes(4096,4096,layers=2,groups=1,
        queries=2,topk=2048,row_bytes=656,maximum_write_records=8192,
        presence_bytes=lambda _:178256)
    assert result == dict(gpu_bytes=actual_local_bytes+mapping_bytes+consolidated_frame_bytes,
        cpu_bytes=5373952,mapping_bytes=mapping_bytes)
