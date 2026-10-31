"""Fixed owner-local recent/cold metadata for exclusive FlowKV residency.

Allocate once before IPC publication and Graph capture. Register, writers and
native permanent release maintain membership at their existing CUDA boundaries;
this factory never reconstructs it by scanning the physical pool.
"""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass


class BucketView(ct.Structure):
    _fields_ = [
        ("bits", ct.c_void_p),
        ("summary", ct.c_void_p),
        ("offsets", ct.c_int32 * 8),
        ("lengths", ct.c_int32 * 8),
        ("levels", ct.c_int32),
        ("capacity", ct.c_int32),
    ]


class Work(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in ("expire_slots", "victims", "incoming")]
    _fields_ += [("capacity", ct.c_int32)]


class Group(ct.Structure):
    _fields_ = [
        (name, ct.c_void_p)
        for name in (
            "buckets",
            "bucket_epochs",
            "epoch",
            "gpu_bucket",
            "gpu_tags",
            "access_time",
        )
    ]
    _fields_ += [
        ("recent_count", ct.c_int32),
        ("logical_capacity", ct.c_int32),
        ("work", Work),
    ]


class NaiveGroup(ct.Structure):
    _fields_ = Group._fields_ + [("naive_histogram", ct.c_void_p)]


assert ct.sizeof(BucketView) == 88 and ct.sizeof(Work) == 32
assert ct.sizeof(Group) == 88 and Group.work.offset == 56
assert ct.sizeof(NaiveGroup) == 96 and NaiveGroup.work.offset == 56


def presence_shape(slots):
    if not 0 < slots <= 2**31 - 3:
        raise ValueError(
            "Presence slots must fit the core positive int32 capacity bound"
        )
    leaf_words = (slots + 31) // 32
    lengths = [(leaf_words + 31) // 32]
    while lengths[-1] > 1:
        lengths.append((lengths[-1] + 31) // 32)
    offsets, size = [], 0
    for length in lengths:
        offsets.append(size)
        size += length
    assert len(lengths) <= 8
    return offsets, lengths, size


def presence_metadata_bytes(slots, total_buckets=256):
    if not 2 <= total_buckets <= 2**31 - 2:
        raise ValueError("Total buckets must fit the recent-ID encoding")
    _, lengths, size = presence_shape(slots)
    recent = total_buckets - 1
    return (
        4 * total_buckets * ((slots + 31) // 32 + size + 1)
        + 8 * recent
        + 4 * slots
        + ct.sizeof(BucketView) * total_buckets
        + ct.sizeof(Group)
    )


@dataclass
class PresenceRing:
    bits: object  # uint32 [R+1, ceil(x/32)]
    summary: object  # uint32 [R+1, sum presence lengths+1 total]
    bucket_epochs: object  # int64 [R], fixed recent-ID age labels
    gpu_bucket: object  # int32 [x], FREE/COLD/recent IDs
    bucket_views: object  # int64 [R+1,11], POD BucketView88
    view: object  # int64 [1,11], POD Group88
    expire_slots: object  # alias native leased_slots[:U]
    victims: object  # alias native candidate_slots[:U]
    incoming: object  # alias native plan.flat[:U]
    capacity: int  # U=min(native W,x)
    recent_count: int
    offsets: tuple
    lengths: tuple
    naive_histogram: object = None

    @property
    def metadata_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.bits,
                self.summary,
                self.bucket_epochs,
                self.gpu_bucket,
                self.bucket_views,
                self.view,
                self.naive_histogram,
            )
            if t is not None
        )

    def clear(self):
        """After native retirement only; preserve the canonical native epoch/time."""
        self.bits.zero_()
        self.summary.zero_()
        self.bucket_epochs.fill_(-1)
        self.gpu_bucket.fill_(-1)
        if self.naive_histogram is not None:
            self.naive_histogram.zero_()


def create_presence_ring(state, workspace, *, total_buckets=256):
    """Allocate before capture, alias the original State9 and three Work arrays.

    Every bucket owns leaf bits, hierarchical nonempty presence, and ONE
    appended total at summary[size]. No per-leaf/subtree population counts.

    Expiry scratch is retired before victim selection. Victims survive until
    original assign completes. Incoming scratch is retired after metadata fill,
    before native assign overwrites the original five-row SoA plan. No presence-ring
    operation owns or resets the original eight native workspace counters.
    """
    import torch
    from ulacache.compiler import recency_policy

    x, logical = state.gpu_tags.numel(), state.directory.numel()
    expected = presence_metadata_bytes(x, total_buckets)
    recent = total_buckets - 1
    offsets, lengths, size = presence_shape(x)
    W = workspace.new_rows.numel()
    U = min(W, x)
    if (
        U <= 0
        or workspace.leased_slots.numel() < U
        or workspace.candidate_slots.numel() < U
    ):
        raise ValueError("Original workspace must supply the bounded ring scratch")
    if not workspace.plan.is_contiguous() or workspace.plan.numel() < U:
        raise ValueError("Native plan must expose the contiguous incoming scratch")
    if state.epoch.numel() != 1 or state.access_time.numel() != logical:
        raise ValueError("Ring metadata requires the canonical logical access times")
    device = state.gpu_tags.device
    if recency_policy() == "naive":
        # Retain exactly the supplied native pools/workspaces. Do not expand
        # resident KV capacity when the PB membership storage is absent.
        empty = torch.empty(0, dtype=torch.uint32, device=device)
        labels = torch.empty(0, dtype=torch.int64, device=device)
        inverse = torch.empty(0, dtype=torch.int32, device=device)
        bins = torch.zeros(256, dtype=torch.int32, device=device)
        descriptor = BucketView(0, 0, (ct.c_int32 * 8)(), (ct.c_int32 * 8)(), 0, x)
        bucket_views = (
            torch.tensor(list(bytes(descriptor)), dtype=torch.uint8, device=device)
            .view(torch.int64)
            .reshape(1, 11)
        )
        expire_slots = workspace.leased_slots.reshape(-1)[:U]
        victims = workspace.candidate_slots.reshape(-1)[:U]
        incoming = workspace.plan.reshape(-1)[:U]
        group = NaiveGroup(
            bucket_views.data_ptr(),
            0,
            state.epoch.data_ptr(),
            0,
            state.gpu_tags.data_ptr(),
            state.access_time.data_ptr(),
            0,
            logical,
            Work(expire_slots.data_ptr(), victims.data_ptr(), incoming.data_ptr(), U),
            bins.data_ptr(),
        )
        view = (
            torch.tensor(list(bytes(group)), dtype=torch.uint8, device=device)
            .view(torch.int64)
            .reshape(1, 12)
        )
        result = PresenceRing(
            empty,
            empty,
            labels,
            inverse,
            bucket_views,
            view,
            expire_slots,
            victims,
            incoming,
            U,
            0,
            (),
            (),
            bins,
        )
        assert result.metadata_bytes == 256 * 4 + 88 + 96
        return result
    leaf_words = (x + 31) // 32
    bits = torch.zeros((total_buckets, leaf_words), dtype=torch.uint32, device=device)
    summary = torch.zeros((total_buckets, size + 1), dtype=torch.uint32, device=device)
    bucket_epochs = torch.full((recent,), -1, dtype=torch.int64, device=device)
    gpu_bucket = torch.full((x,), -1, dtype=torch.int32, device=device)
    descriptors = [
        BucketView(
            bits.data_ptr() + bid * leaf_words * 4,
            summary.data_ptr() + bid * (size + 1) * 4,
            (ct.c_int32 * 8)(*offsets),
            (ct.c_int32 * 8)(*lengths),
            len(lengths),
            x,
        )
        for bid in range(total_buckets)
    ]
    bucket_views = (
        torch.tensor(
            list(b"".join(bytes(v) for v in descriptors)),
            dtype=torch.uint8,
            device=device,
        )
        .view(torch.int64)
        .reshape(total_buckets, 11)
    )
    expire_slots = workspace.leased_slots.reshape(-1)[:U]
    victims = workspace.candidate_slots.reshape(-1)[:U]
    incoming = workspace.plan.reshape(-1)[:U]
    work = Work(expire_slots.data_ptr(), victims.data_ptr(), incoming.data_ptr(), U)
    group = Group(
        bucket_views.data_ptr(),
        bucket_epochs.data_ptr(),
        state.epoch.data_ptr(),
        gpu_bucket.data_ptr(),
        state.gpu_tags.data_ptr(),
        state.access_time.data_ptr(),
        recent,
        logical,
        work,
    )
    view = (
        torch.tensor(list(bytes(group)), dtype=torch.uint8, device=device)
        .view(torch.int64)
        .reshape(1, 11)
    )
    result = PresenceRing(
        bits,
        summary,
        bucket_epochs,
        gpu_bucket,
        bucket_views,
        view,
        expire_slots,
        victims,
        incoming,
        U,
        recent,
        tuple(offsets),
        tuple(lengths),
    )
    assert result.metadata_bytes == expected
    return result
