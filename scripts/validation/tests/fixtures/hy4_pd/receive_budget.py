"""Source-derived PD buffer arithmetic; no allocation or launch geometry choice."""
from __future__ import annotations

PAGE=64


def require_full_cold_capacity(staging_records, full_history_records):
    if type(staging_records) is not int or staging_records < PAGE or staging_records % PAGE:
        raise ValueError('Native receive staging must contain complete page64 records')
    if type(full_history_records) is not int or full_history_records < 0:
        raise ValueError('Cold history must be a nonnegative complete record count')
    required=(full_history_records+PAGE-1)//PAGE*PAGE
    if staging_records < required:
        raise ValueError('Unsegmented native receive cannot admit this full cold history; enlarge bounded staging before launch')
    return dict(staging_records=staging_records,full_history_records=full_history_records,
        required_page_aligned_records=required,native_segmented_receive_bound=False,
        permanent_window_refusal_avoided=True)


def cpu_components(*, topology, host_rows, staging_records, target_layers,
                   target_index_layers, draft_layers=1,row_bytes=656,index_bytes=132,
                   copy_chunk_records=4096,pending_records=0):
    if topology not in ('tp','dpa'):raise ValueError('Select native TP or DPA')
    for name,value in locals().copy().items():
        if name not in ('topology',) and (type(value) is not int or value < 0):
            raise ValueError('Invalid byte-geometry integer: '+name)
    if host_rows%PAGE or staging_records%PAGE or not copy_chunk_records or pending_records>staging_records:
        raise ValueError('Page-aligned storage and bounded pending records are required')
    result=dict(target_host_KV=target_layers*host_rows*row_bytes,
        target_KV_receive=target_layers*staging_records*row_bytes,
        replicated_draft_KV_receive=draft_layers*staging_records*row_bytes,
        target_Index_receive=(target_index_layers*staging_records*index_bytes if topology=='tp' else 0),
        independent_draft_Index_receive=(draft_layers*staging_records*index_bytes if topology=='tp' else 0),
        persistent_copy_scratch=(1+draft_layers)*copy_chunk_records*row_bytes,
        pending_source_rows_int64=8*pending_records)
    result['persistent_tensor_total']=sum(result.values())
    # Publication copies only selected directory columns, independently per
    # domain; unlike the staging payload this temporary is chunk bounded.
    columns=8*target_index_layers*min(copy_chunk_records,staging_records)
    reservation=8*staging_records+8*(staging_records//PAGE)+8*PAGE
    index_peak=0
    if topology=='tp':
        k=min(copy_chunk_records,staging_records)
        # 4 int64 Cx132 intermediates, four int64 C vectors, byte staging,
        # plus the column/index-selection masks. Conservative tensor bound.
        index_peak=(4*8*index_bytes+4*8+index_bytes)*k+9*index_bytes
    result.update(publication_selected_columns_tensor_bound=columns,
        reservation_transient_tensor_bound=reservation,
        Index_publication_transient_tensor_bound=index_peak,
        known_tensor_peak_bound=result['persistent_tensor_total']+columns+reservation+index_peak,
        canonical_draft_KV_and_Index_are_GPU=True,
        Python_transport_allocator_overheads_not_claimed_exact=True)
    return result


def host_row_upper_bound(*, topology,staging_records,target_layers,target_index_layers,
                        cpu_budget_bytes=128_000_000_000,non_tensor_reserve_bytes=0,**kwargs):
    base=cpu_components(topology=topology,host_rows=0,staging_records=staging_records,
        target_layers=target_layers,target_index_layers=target_index_layers,**kwargs)
    available=cpu_budget_bytes-base['known_tensor_peak_bound']-non_tensor_reserve_bytes
    if available<0:raise ValueError('Bounded receive buffers exceed the CPU budget before target KV')
    rows=(available//(target_layers*kwargs.get('row_bytes',656)))//PAGE*PAGE
    return dict(host_rows_upper_bound_from_known_terms=rows,cpu_budget_bytes=cpu_budget_bytes,
        non_tensor_reserve_bytes=non_tensor_reserve_bytes,known_fixed_terms=base,
        launch_geometry_selected=False,actual_CPU_budget_passed=False,
        non_tensor_and_native_registration_overheads_pending=True)


def coupled_GPU_KV_and_Index_terms(*,topology,gpu_rows,host_rows,target_layers,
                                  target_index_layers,draft_layers=1,row_bytes=656,index_bytes=132):
    degree=8 if topology=='tp' else 1
    native_rows=degree*(gpu_rows+host_rows)
    return dict(target_KV_GPU=target_layers*gpu_rows*row_bytes,
        target_Index_GPU=target_index_layers*(gpu_rows+host_rows)*index_bytes,
        replicated_draft_KV_GPU=draft_layers*native_rows*row_bytes,
        independent_draft_Index_GPU=draft_layers*native_rows*index_bytes,
        draft_native_rows_including_reserved_page=native_rows,
        physical_target_rows_per_rank=gpu_rows,
        weights_localbuffer_Graph_workspace_and_other_metadata_pending=True,
        actual_GPU_budget_passed=False)
