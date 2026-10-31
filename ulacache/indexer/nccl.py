"""Private native NCCL Index gather for devices without multicast mappings."""

from contextlib import contextmanager


class NCCLIndexOwner:
    def __init__(
        self,
        serving_group,
        *,
        max_queries,
        max_page_width,
        device,
        communicator_factory,
    ):
        import torch
        import torch.distributed as dist

        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            raise ValueError("Construct the CUDA Index owner before Graph capture")
        if any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1
            for n in (max_queries, max_page_width)
        ):
            raise ValueError("Index communication bounds must be positive integers")
        self.world_size = serving_group.world_size
        self.rank_in_group = serving_group.rank_in_group
        self.cpu_group = serving_group.cpu_group
        if (
            self.world_size < 2
            or self.world_size & (self.world_size - 1)
            or dist.get_world_size(self.cpu_group) != self.world_size
            or dist.get_rank(self.cpu_group) != self.rank_in_group
        ):
            raise ValueError(
                "Index gather requires the serving group's power-of-two rank order"
            )
        self.max_queries, self.max_page_width = max_queries, max_page_width
        columns = (
            ((max_page_width + self.world_size - 1) // self.world_size + 3) // 4 * 256
        )
        self.maximum_gather_bytes = self.world_size * max_queries * columns * 4
        self.gathered_workspace = torch.empty(
            self.maximum_gather_bytes // 4, dtype=torch.float32, device=self.device
        )
        self.communicator = communicator_factory(self.cpu_group, self.device)
        if not self.communicator.available:
            raise RuntimeError("The native NCCL Index communicator is unavailable")
        self.compact_scores = False
        self._closed = False

    def validate_bounds(self, max_queries, max_page_width):
        if self._closed or (max_queries, max_page_width) != (
            self.max_queries,
            self.max_page_width,
        ):
            raise ValueError(
                "Shared Index owner requires identical native Graph bounds"
            )

    @contextmanager
    def graph_capture(self):
        if self._closed:
            raise RuntimeError("Index owner is closed")
        with self.communicator.change_state(enable=True):
            yield

    def all_gather_into_tensor(self, output, local):
        import torch

        if (
            self._closed
            or local.ndim != 2
            or output.ndim != 2
            or local.dtype != torch.float32
            or output.dtype != torch.float32
            or local.device != self.device
            or output.device != self.device
            or not local.is_contiguous()
            or not output.is_contiguous()
            or output.shape != (self.world_size * local.shape[0], local.shape[1])
            or local.numel() == 0
            or output.numel() * 4 > self.maximum_gather_bytes
            or output.numel() * 4 % (16 * self.world_size)
            or local.data_ptr() % 16
            or output.data_ptr() % 16
            or output.data_ptr() != self.gathered_workspace.data_ptr()
        ):
            raise ValueError(
                "Index NCCL requires its aligned contiguous FP32 active prefix"
            )
        with self.communicator.change_state(enable=True):
            self.communicator.all_gather(output, local)

    def close(self):
        import torch
        import torch.distributed as dist

        if self._closed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot close an Index owner during Graph capture")
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu_group)
        self.gathered_workspace = None
        self.communicator = None
        self._closed = True
