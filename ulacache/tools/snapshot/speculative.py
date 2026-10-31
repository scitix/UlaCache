"""Serialize native linear-MTP continuation without framework imports."""

import hashlib
from pathlib import Path

TENSOR_FIELDS = ("topk_p", "topk_index", "hidden_states", "bonus_tokens", "draft_probs")


def save_continuation(path, draft_input, request_index, native_rows):
    """Save exact native tensors; normalize cached DSA IDs to sequence positions."""
    import torch

    tensors = {}
    for field in TENSOR_FIELDS:
        value = getattr(draft_input, field, None)
        tensors[field] = (
            value[request_index : request_index + 1].detach().cpu()
            if value is not None
            else None
        )
    topk = getattr(draft_input, "dsa_topk_indices", None)
    if topk is not None:
        selected = topk[request_index : request_index + 1].detach().cpu().long()
        rows = native_rows.detach().cpu().long()
        sorted_rows, order = torch.sort(rows)
        where = torch.searchsorted(sorted_rows, selected.clamp(min=0))
        valid = selected > 0
        clipped = where.clamp(max=len(rows) - 1)
        if torch.any(
            valid & ((where >= len(rows)) | (sorted_rows[clipped] != selected))
        ):
            raise ValueError(
                "Native draft seed DSA IDs exceed the saved causal history"
            )
        tensors["dsa_topk_positions"] = torch.where(valid, order[clipped], -1)
        tensors["dsa_topk_sentinels"] = torch.where(valid, -1, selected)
        tensors["dsa_topk_dtype"] = str(topk.dtype).split(".")[-1]
    else:
        tensors["dsa_topk_positions"] = None
    tensors["num_tokens_per_req"] = draft_input.num_tokens_per_req
    tensors["num_tokens_for_logprob_per_req"] = (
        draft_input.num_tokens_for_logprob_per_req
    )
    path = Path(path)
    with path.open("xb") as stream:
        torch.save(tensors, stream)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_continuation(path, expected_sha256, native_rows, device):
    """Return constructor fields for the unchanged native EagleDraftInput."""
    import torch

    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Native speculative continuation checksum differs")
    data = torch.load(path, map_location="cpu", weights_only=True)
    fields = {
        name: data[name].to(device) if data[name] is not None else None
        for name in TENSOR_FIELDS
    }
    positions = data["dsa_topk_positions"]
    if positions is not None:
        rows = native_rows.detach().cpu().long()
        if torch.any(positions >= len(rows)):
            raise ValueError(
                "Saved draft seed position exceeds restored causal history"
            )
        indices = torch.where(
            positions >= 0, rows[positions.clamp(min=0)], data["dsa_topk_sentinels"]
        )
        fields["dsa_topk_indices"] = indices.to(
            device, dtype=getattr(torch, data["dsa_topk_dtype"])
        )
    else:
        fields["dsa_topk_indices"] = None
    for name in ["num_tokens_per_req", "num_tokens_for_logprob_per_req"]:
        fields[name] = data[name]
    return fields
