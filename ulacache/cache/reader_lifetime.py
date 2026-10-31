"""Device publication and reader retirement for serial residency selections."""

from ulacache.compiler import cache_once, load_jit


@cache_once
def _compile_residency_publication():
    return load_jit(
        "ulacache_residency_publication",
        cuda_files=["csrc/cache/reader_lifetime.cuh"],
        cuda_wrappers=[
            (name, "ulacache::residency_publication_" + name)
            for name in (
                "begin_selection",
                "publish_residency",
                "finish_readers",
                "finish_all_readers",
                "wait_for_readers",
                "wait_for_all_readers",
                "validate_peer",
            )
        ],
    )


class DistributedResidencyPublication:
    """Three device phases per selection, independently counted by domain.

    All owners execute the same domain sequence on their current CUDA streams,
    including idle queries. Before beginning the next selection, native
    selection tensors have been copied to stable exported storage. That copy
    may replace the previous selection after this rank passed its previous
    ``publish_residency``: every owner has then finished reading the old IDs.

    ``begin_selection`` waits for every previous reader before returning to
    owner promotion/writes and publishes the new consumer selections.
    ``publish_residency`` waits for every owner's promoted records and current
    writes. ``finish_readers`` publishes local reader completion. None of
    these phases synchronizes the CPU or changes logical record ownership.
    """

    def __init__(
        self, residency_domain_count, device, cpu_group, *, ipc_manager_factory
    ):
        import torch
        from ulacache.transfer.ipc import FlowKVPeerShards

        if residency_domain_count < 1:
            raise ValueError("Residency publication needs at least one domain")
        self.cpu_group = cpu_group
        self.epochs = torch.zeros(
            residency_domain_count, dtype=torch.int64, device=device
        )
        self.flags = torch.zeros(
            (residency_domain_count, 3), dtype=torch.int64, device=device
        )
        self._device_module = _compile_residency_publication()
        self.peer_flags = FlowKVPeerShards(
            self.flags, cpu_group, ipc_manager_factory=ipc_manager_factory
        )
        for pointer in self.peer_flags.ptrs.cpu().tolist():
            self._device_module.validate_peer(self.flags, int(pointer))
        self._closed = False

    def begin_selection(self, residency_domain_index):
        self._device_module.begin_selection(
            self.epochs, self.flags, self.peer_flags.ptrs, residency_domain_index
        )

    def publish_residency(self, residency_domain_index):
        self._device_module.publish_residency(
            self.epochs, self.flags, self.peer_flags.ptrs, residency_domain_index
        )

    def finish_readers(self, residency_domain_index):
        self._device_module.finish_readers(
            self.epochs, self.flags, residency_domain_index
        )

    def finish_all_readers(self):
        """Publish completed independent domains in one stream-ordered kernel."""
        self._device_module.finish_all_readers(self.epochs, self.flags)

    def wait_for_readers(self, residency_domain_index):
        """Retire readers before native permanent reclamation or relocation."""
        self._device_module.wait_for_readers(
            self.epochs, self.peer_flags.ptrs, residency_domain_index
        )

    def wait_for_all_readers(self):
        self._device_module.wait_for_all_readers(self.epochs, self.peer_flags.ptrs)

    def get_device_storage_inventory(self):
        tensors = (self.epochs, self.flags, self.peer_flags.ptrs)
        return {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for tensor in tensors
        }

    def close(self):
        """Collectively unmap flags after the caller retires Graphs and readers."""
        import torch
        import torch.distributed as distributed

        if self._closed:
            return
        self.wait_for_all_readers()
        torch.cuda.synchronize(self.epochs.device)
        distributed.barrier(group=self.cpu_group)
        self.peer_flags.close()
        distributed.barrier(group=self.cpu_group)
        self._closed = True
