import ctypes
import json
import struct
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError as error:
    raise unittest.SkipTest('NumPy is required for direct-address component tests') from error

from ulacache.transfer.direct_receive import (
    DirectReceivePlan, RecordBank, RecordRegion, coalesce_writes, submit_direct_batch,
    order_unpublished_cpu_locations,
)
from ulacache.transfer.direct_receiver import send_record_chunk
import ulacache.transfer.direct_receive as direct_receive


class DirectReceiveTests(unittest.TestCase):
    def test_unpublished_cpu_order_preserves_slot_sets_gpu_prefix_and_tail(self):
        locations=np.array([[90,-12,7,-4,-9,8,-3,91],
                            [92,4,-22,-6,5,-15,6,93]],dtype=np.int64)
        original=locations.copy()
        self.assertTrue(order_unpublished_cpu_locations(locations[:,1:-1]))
        for before,after in zip(original,locations):
            np.testing.assert_array_equal(before[before>=0],after[before>=0])
            np.testing.assert_array_equal(np.sort(before[before<=-2]),np.sort(after[after<=-2]))
            self.assertTrue(bool((np.diff(-after[after<=-2]-2)>0).all()))
        np.testing.assert_array_equal(locations[:,[0,-1]],original[:,[0,-1]])
        ordered=locations.copy()
        self.assertFalse(order_unpublished_cpu_locations(locations))
        np.testing.assert_array_equal(locations,ordered)

    def test_unpublished_order_rejects_invalid_metadata_without_mutation(self):
        for locations in [np.ones(4,dtype=np.int64),np.ones((2,4),dtype=np.int32),
                          np.ones((2,4),dtype=np.float64)]:
            before=locations.copy()
            with self.assertRaises(ValueError):order_unpublished_cpu_locations(locations)
            np.testing.assert_array_equal(locations,before)
        readonly=np.array([[-9,-2]],dtype=np.int64);readonly.flags.writeable=False
        with self.assertRaises(ValueError):order_unpublished_cpu_locations(readonly)

    def region(self, rows=32, width=4, fill=0xCD):
        data=(ctypes.c_ubyte*(rows*width))(*([fill]*(rows*width)))
        self.keep.append(data)
        return data,RecordRegion(ctypes.addressof(data),len(data),width)

    def setUp(self):
        self.keep=[]

    def test_mixed_target_and_logical_draft_write_real_bytes(self):
        source,s=self.region(fill=0)
        for i in range(len(source)):source[i]=(i*7+3)%251
        hot,g=self.region();cold,h=self.region();draft,d=self.region()
        plan=DirectReceivePlan([9,10,11,12,13,14],[[2,3,-7,-8,8,9]],
            [RecordBank(0,0,g,h),RecordBank(1,None,d)],page_size=4)
        batch=plan.transfer_batch({0:s,1:s},[1,3],destination_start=0,record_count=6)
        calls=[]
        class Engine:
            def batch_transfer_sync(self,session,src,dst,lengths):
                calls.append((session,len(src)))
                for a,b,n in zip(src,dst,lengths):ctypes.memmove(b,a,n)
                return 0
        self.assertEqual(submit_direct_batch(Engine(),'peer',batch),0)
        self.assertEqual(len(calls),1)
        source_rows=[4,5,6,7,12,13]
        for i,(tier,slot) in enumerate([(hot,2),(hot,3),(cold,5),(cold,6),(hot,8),(hot,9)]):
            self.assertEqual(bytes(tier[slot*4:slot*4+4]),bytes(source[source_rows[i]*4:source_rows[i]*4+4]))
            self.assertEqual(bytes(draft[(9+i)*4:(10+i)*4]),bytes(source[source_rows[i]*4:source_rows[i]*4+4]))
        for data,touched in [(hot,{2,3,8,9}),(cold,{5,6}),(draft,set(range(9,15)))]:
            for row in set(range(32))-touched:self.assertEqual(bytes(data[row*4:row*4+4]),b'\xcd'*4)
        self.assertEqual(int(batch[2].sum()),6*4*2)

    def test_partial_suffix_does_not_touch_cached_prefix_or_tail(self):
        src,s=self.region(fill=0x31);dst,d=self.region()
        plan=DirectReceivePlan([1,2,3,4,5,6],[[1,2,3,4,5,6]],[RecordBank(0,0,d)],page_size=4)
        batch=plan.transfer_batch({0:s},[2],destination_start=3,record_count=2)
        for a,b,n in zip(*batch):ctypes.memmove(int(b),int(a),int(n))
        self.assertEqual(bytes(dst[16:24]),b'\x31'*8)
        self.assertEqual(bytes(dst[:16]),b'\xcd'*16)
        self.assertEqual(bytes(dst[24:]),b'\xcd'*(len(dst)-24))

    def test_coalesce_requires_contiguity_at_both_ends(self):
        src,dst,size=coalesce_writes([100,104,108,112],[200,204,300,296],4)
        np.testing.assert_array_equal(src,[100,108,112])
        np.testing.assert_array_equal(dst,[200,300,296])
        np.testing.assert_array_equal(size,[8,4,4])

    def test_group_specific_mapping_is_preserved(self):
        _,g=self.region();_,h=self.region();_,other=self.region()
        plan=DirectReceivePlan([1,2],[[4,-5],[-7,9]],
            [RecordBank(0,0,g,h),RecordBank(1,1,other,h)],page_size=4)
        np.testing.assert_array_equal(plan.destination_addresses(0,0,2),[g.address+16,h.address+12])
        np.testing.assert_array_equal(plan.destination_addresses(1,0,2),[h.address+20,other.address+36])

    def test_wire_round_trip_dense_and_both_run_directions(self):
        _,g=self.region(rows=512);_,h=self.region(rows=512)
        locations=np.stack([np.arange(100,228),-np.arange(32,160)-2,
            np.tile([0,-2],64)+np.repeat(np.arange(64)*3,2)])
        # Third group deliberately fragmented; choose bounded distinct locations.
        locations[2]=np.arange(128)*2
        plan=DirectReceivePlan(np.arange(1,129),locations,
            [RecordBank(0,0,g,h),RecordBank(1,1,g,h),RecordBank(2,2,g,h)],page_size=64)
        payload=plan.to_bytes();loaded=DirectReceivePlan.from_bytes(payload,record_limit=128)
        np.testing.assert_array_equal(loaded.locations,plan.locations)
        np.testing.assert_array_equal(loaded.logical_rows,plan.logical_rows)
        self.assertEqual(loaded.banks,plan.banks)
        self.assertLess(len(payload),plan.locations.nbytes+plan.logical_rows.nbytes)
        for bank in range(3):np.testing.assert_array_equal(loaded.destination_addresses(bank,7,80),plan.destination_addresses(bank,7,80))

    def test_invalid_or_unbound_allocations_are_rejected(self):
        _,g=self.region(rows=4);_,h=self.region(rows=4)
        for rows,locations in [([1,1],[[0,1]]),([1],[[-1]]),([1],[[4]]),([1],[[-6]])]:
            with self.assertRaises(ValueError):DirectReceivePlan(rows,locations,[RecordBank(0,0,g,h)],page_size=4)

    def test_wire_bounds_and_request_limit_are_enforced(self):
        _,g=self.region();p=DirectReceivePlan([1],[[2]],[RecordBank(0,0,g)],page_size=4);wire=p.to_bytes()
        for bad in [wire[:-1],wire+b'x',b'wrong',wire[:8]+b'\xff'*4]:
            with self.assertRaises(ValueError):DirectReceivePlan.from_bytes(bad,record_limit=1)
        with self.assertRaises(ValueError):DirectReceivePlan.from_bytes(wire,record_limit=0)

    def test_shared_group_still_checks_each_bank_capacity_and_signed_cold_bound(self):
        _,large=self.region(rows=16);_,small=self.region(rows=2)
        for banks,locations in [
                ([RecordBank(0,0,large),RecordBank(1,0,small)],[[5]]),
                ([RecordBank(0,0,large,large),RecordBank(1,0,large,small)],[[-7]]),
                ([RecordBank(0,0,large,large)],[[np.iinfo(np.int64).min]])]:
            with self.assertRaises(ValueError):
                DirectReceivePlan([1],locations,banks,page_size=4)

    def test_vector_run_decoder_rejects_gaps_overlap_steps_and_overflow(self):
        _,region=self.region(rows=64)
        plan=DirectReceivePlan(np.arange(1,5),[[-2,-3,-4,-5]],
            [RecordBank(0,0,region,region)],page_size=4)
        wire=plan.to_bytes();size=struct.unpack_from('<I',wire,8)[0]
        original=json.loads(wire[12:12+size])
        cases=[[[0,2,-2,-1],[3,1,-5,-1]],[[0,3,-2,-1],[2,2,-5,-1]],
               [[0,4,-2,2]],[[0,4,np.iinfo(np.int64).max,1]],
               [[0,4,np.iinfo(np.int64).min,-1]],[]]
        for bad in cases:
            header=dict(original,groups=[dict(encoding='runs',count=1),
                                        dict(encoding='runs',count=len(bad))])
            payload=json.dumps(header,separators=(',',':')).encode()
            encoded=(plan.MAGIC+struct.pack('<I',len(payload))+payload+
                     np.array([[0,4,1,1]],dtype='<i8').tobytes()+
                     np.asarray(bad,dtype='<i8').reshape(-1,4).tobytes())
            with self.assertRaises(ValueError):
                DirectReceivePlan.from_bytes(encoded,record_limit=4)

    def test_many_compressed_mixed_runs_reconstruct_exact_record_addresses(self):
        _,region=self.region(rows=16384)
        rng=np.random.default_rng(27);n=4096
        pages=rng.choice(200,64,replace=False)
        cold=-(pages[:,None]*64+np.arange(64)).reshape(-1)-2
        cold[[17,333,1801,2999]]=[37,38,39,40]
        plan=DirectReceivePlan(np.arange(64,64+n),np.stack((cold,cold[::-1])),
            [RecordBank(0,0,region,region),RecordBank(1,1,region,region)],page_size=64)
        loaded=DirectReceivePlan.from_bytes(plan.to_bytes(),record_limit=n)
        np.testing.assert_array_equal(loaded.locations,plan.locations)
        for bank in range(2):
            np.testing.assert_array_equal(loaded.destination_addresses(bank,0,n),
                                          plan.destination_addresses(bank,0,n))

    def test_failed_batch_does_not_claim_later_writes_completed(self):
        calls=[]
        class Engine:
            def batch_transfer_sync(self,session,src,dst,lengths):
                calls.append((src,dst,lengths));return -1
        batch=tuple(np.array(x,dtype=np.int64) for x in [[100,200],[300,400],[4,4]])
        self.assertEqual(submit_direct_batch(Engine(),'peer',batch,max_segments=1),-1)
        self.assertEqual(len(calls),1)

    def test_empty_suffix_is_valid_without_payload_writes(self):
        _,g=self.region();p=DirectReceivePlan([],np.empty((1,0),dtype=np.int64),[RecordBank(0,0,g)],page_size=4)
        restored=DirectReceivePlan.from_bytes(p.to_bytes(),record_limit=0)
        self.assertEqual(restored.record_count,0)
        batch=restored.transfer_batch({0:g},[],destination_start=0,record_count=0)
        self.assertTrue(all(len(v)==0 for v in batch))

    def test_bounded_fragmented_writes_preserve_all_banks_and_prefix(self):
        rng=np.random.default_rng(17)
        src,s=self.region(rows=128,fill=0)
        for i in range(len(src)):src[i]=(i*7+3)%251
        banks=[];buffers=[]
        locations=[]
        for group in range(3):
            slots=rng.choice(128,43,replace=False).astype(np.int64)
            locations.append(np.where(np.arange(43)%3,slots,-slots-2))
        for layer in range(7):
            hot,g=self.region(rows=128);cold,h=self.region(rows=128)
            banks.append(RecordBank(layer,layer%3,g,h));buffers.append((hot,cold))
        draft,d=self.region(rows=128);banks.append(RecordBank(7,None,d))
        plan=DirectReceivePlan(np.arange(40,83),locations,banks,page_size=4)
        pages=np.array([7,2,11,1,17,5,21,3,25])
        total=0
        for batch in plan.iter_transfer_batches({i:s for i in range(8)},pages,
                destination_start=5,record_count=35,max_segments=7):
            self.assertLessEqual(len(batch[0]),7)
            for a,b,n in zip(*batch):ctypes.memmove(int(b),int(a),int(n))
            total+=int(batch[2].sum())
        self.assertEqual(total,35*8*4)
        source_rows=(pages[:,None]*4+np.arange(4)).reshape(-1)[:35]
        for layer,(hot,cold) in enumerate(buffers):
            touched=[set(),set()]
            for i in range(35):
                location=locations[layer%3][5+i];tier=0 if location>=0 else 1
                slot=int(location if tier==0 else -location-2);touched[tier].add(slot)
                self.assertEqual(bytes((hot,cold)[tier][slot*4:slot*4+4]),
                                 bytes(src[source_rows[i]*4:source_rows[i]*4+4]))
            for tier,data in enumerate((hot,cold)):
                for slot in set(range(128))-touched[tier]:
                    self.assertEqual(bytes(data[slot*4:slot*4+4]),b'\xcd'*4)
        for slot in range(128):
            expected=(bytes(src[source_rows[slot-45]*4:source_rows[slot-45]*4+4])
                      if 45<=slot<80 else b'\xcd'*4)
            self.assertEqual(bytes(draft[slot*4:slot*4+4]),expected)

    def test_stream_validation_precedes_first_write_and_failure_stops_send(self):
        _,source=self.region(rows=8);_,short=self.region(rows=2)
        _,dest=self.region(rows=8)
        plan=DirectReceivePlan([1,2,3,4],[[4,1,5,2]],
            [RecordBank(0,0,dest),RecordBank(1,0,dest)],page_size=4)
        invalid=plan.iter_transfer_batches({0:source,1:short},[1],
            destination_start=0,record_count=4,max_segments=1)
        with self.assertRaises(ValueError):next(invalid)
        calls=[];audits=[]
        class Engine:
            def batch_transfer_sync(self,session,src,dst,lengths):
                calls.append((src,dst,lengths));return -1
        status=send_record_chunk(Engine(),'peer',plan,{0:source,1:source},[1],
            page_offset=0,record_count=4,record_limit=4,audit=audits.append)
        self.assertEqual(status,-1)
        self.assertEqual(len(calls),1)
        self.assertEqual(audits[0]['batches'],1)
        self.assertEqual(audits[0]['bytes'],32)

    def test_parallel_failure_joins_every_submitted_native_call(self):
        n=70000
        _,source=self.region(rows=n)
        banks=[RecordBank(i,0,self.region(rows=2*n)[1]) for i in range(3)]
        plan=DirectReceivePlan(np.arange(1,n+1),[np.arange(n)*2],banks,page_size=4)
        pages=np.arange(n//4)
        second_started=threading.Event();first_failed=threading.Event()
        release_second=threading.Event();returned=threading.Event()
        calls=[];lock=threading.Lock();results=[];errors=[];audits=[]
        class Engine:
            def batch_transfer_sync(self,session,src,dst,lengths):
                with lock:
                    ordinal=len(calls);calls.append(len(src))
                if ordinal==0:
                    assert second_started.wait(5)
                    first_failed.set()
                    return -1
                second_started.set()
                assert release_second.wait(5)
                return 0
        with ThreadPoolExecutor(max_workers=2) as executor:
            def transfer():
                try:
                    results.append(send_record_chunk(Engine(),'peer',plan,
                        {i:source for i in range(3)},pages,page_offset=0,
                        record_count=n,record_limit=n,executors=[executor],
                        max_inflight=2,max_segments=65536,audit=audits.append))
                except BaseException as error:errors.append(error)
                finally:returned.set()
            worker=threading.Thread(target=transfer)
            worker.start()
            try:
                self.assertTrue(first_failed.wait(5))
                self.assertFalse(returned.wait(.05))
            finally:
                release_second.set();worker.join(5)
            self.assertFalse(worker.is_alive())
        self.assertFalse(errors,errors)
        self.assertEqual(results,[-1])
        self.assertEqual(calls,[65536,65536])
        self.assertEqual(audits[0]['peak_inflight_batches'],2)

    def test_shared_geometry_keeps_bank_bases_widths_and_suffix_bytes(self):
        locations=np.array([[1,2,-5,8,-10,-11,20,21,-29,3,4,-6],
                            [10,-12,11,-13,12,-14,13,-15,14,15,16,17]])
        layouts=[(0,4),(0,8),(1,4),(0,4),(0,8),(None,12)]
        banks=[];sources={};buffers=[]
        for layer,(group,width) in enumerate(layouts):
            source,s=self.region(rows=128,width=width,fill=0)
            for i in range(len(source)):source[i]=(layer*71+i*7)%251
            hot,g=self.region(rows=128,width=width);cold,h=self.region(rows=128,width=width)
            banks.append(RecordBank(layer,group,g,h if group is not None else None))
            sources[layer]=s;buffers.append((source,hot,cold))
        plan=DirectReceivePlan(np.arange(40,52),locations,banks,page_size=4)
        # Seven record tiles exceed the two-entry shared-layout cache. Reused
        # layouts must still be computed at most once per tile, with real bytes.
        with patch.object(direct_receive,'_write_geometry',wraps=direct_receive._write_geometry) as geometry:
            for batch in plan.iter_transfer_batches(sources,[5,1],destination_start=3,
                    record_count=7,max_segments=1):
                for a,b,n in zip(*batch):ctypes.memmove(int(b),int(a),int(n))
            self.assertLessEqual(geometry.call_count,4*7)
        source_rows=[20,21,22,23,4,5,6]
        for layer,(group,width) in enumerate(layouts):
            source,hot,cold=buffers[layer];touched=[set(),set()]
            mapping=plan.logical_rows if group is None else locations[group]
            for i,source_row in enumerate(source_rows):
                location=int(mapping[3+i]);tier=int(location<0);slot=location if not tier else -location-2
                touched[tier].add(slot)
                self.assertEqual(bytes((hot,cold)[tier][slot*width:(slot+1)*width]),
                                 bytes(source[source_row*width:(source_row+1)*width]))
            for tier,data in enumerate((hot,cold)):
                for slot in set(range(128))-touched[tier]:
                    self.assertEqual(bytes(data[slot*width:(slot+1)*width]),b'\xcd'*width)

    def test_tuned_batches_preserve_real_bytes_and_join_pending_writes(self):
        rng=np.random.default_rng(53);n=180
        mapping=np.asarray([rng.choice(512,n,replace=False) for _ in range(2)])
        locations=np.where(np.arange(n)%3,mapping,-mapping-2)
        buffers=[];banks=[];sources={}
        for bank,group in enumerate([0,0,1,None]):
            source,s=self.region(rows=512,fill=0)
            for i in range(len(source)):source[i]=(bank*39+i*7)%251
            hot,g=self.region(rows=512);cold,h=self.region(rows=512)
            banks.append(RecordBank(bank,group,g,h if group is not None else None))
            sources[bank]=s;buffers.append((source,hot,cold))
        plan=DirectReceivePlan(np.arange(1,n+1),locations,banks,page_size=4)
        pages=rng.choice(128,45,replace=False)
        rows=(pages[:,None]*4+np.arange(4)).reshape(-1)
        for limit in [13,64,512]:
            for _,hot,cold in buffers:
                ctypes.memset(ctypes.addressof(hot),0xCD,len(hot))
                ctypes.memset(ctypes.addressof(cold),0xCD,len(cold))
            calls=[];lock=threading.Lock()
            class Writer:
                def batch_transfer_sync(self,session,src,dst,lengths):
                    with lock:calls.append(len(src))
                    for a,b,size in zip(src,dst,lengths):ctypes.memmove(b,a,size)
                    return 0
            with ThreadPoolExecutor(max_workers=4) as executor:
                status=send_record_chunk(Writer(),'peer',plan,sources,pages,
                    page_offset=0,record_count=n,record_limit=n,
                    executors=[executor],max_inflight=4,max_segments=limit)
            self.assertEqual(status,0);self.assertLessEqual(max(calls),limit)
            for bank,group in enumerate([0,0,1,None]):
                source,hot,cold=buffers[bank];touched=[set(),set()]
                for i in range(n):
                    location=int(plan.logical_rows[i] if group is None else locations[group,i])
                    tier=int(location<0);slot=location if not tier else -location-2
                    touched[tier].add(slot)
                    self.assertEqual(bytes((hot,cold)[tier][slot*4:(slot+1)*4]),
                                     bytes(source[rows[i]*4:(rows[i]+1)*4]))
                for tier,data in enumerate((hot,cold)):
                    for slot in set(range(512))-touched[tier]:
                        self.assertEqual(bytes(data[slot*4:(slot+1)*4]),b'\xcd'*4)


if __name__=='__main__':
    unittest.main(verbosity=2)


class NativePDLaunchTests(unittest.TestCase):
    def setUp(self):
        import importlib.util
        import os
        from pathlib import Path
        root = os.environ.get('HISPARSE_NATIVE_SOURCE_ROOT')
        if not root:
            self.skipTest('Set HISPARSE_NATIVE_SOURCE_ROOT to the patched framework')
        path = Path(root) / 'python/sglang/srt/ulacache/pd_record_adapter.py'
        spec = importlib.util.spec_from_file_location('_pd_launch_contract', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.calls = []
        self.module.install = lambda **kw: self.calls.append(kw)

    def test_regular_serving_and_other_native_transports_remain_unchanged(self):
        env = {'unrelated': 'value'}
        self.module.configure_native_launch(None, [], env)
        self.module.configure_native_launch(None, ['--disaggregation-mode', 'prefill',
            '--disaggregation-transfer-backend', 'nixl'], env)
        self.assertEqual(env, {'unrelated': 'value'})
        self.assertEqual(self.calls, [])

    def test_native_prefill_and_dpa_decode_select_the_same_protocol(self):
        from types import SimpleNamespace
        env = {}
        self.module.configure_native_launch(None,
            ['--disaggregation-mode=prefill', '--attn-dp-size', '8'], env)
        self.assertEqual(env['ULACACHE_PD_CACHE_KIND'], 'native')
        self.assertEqual(env['ULACACHE_PD_ATTENTION_MODE'], 'dpa')
        self.assertEqual(env['SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE'], '1')
        self.assertIsNone(self.calls.pop()['target_cache_installer'])
        env = {}
        self.module.configure_native_launch(SimpleNamespace(adapter='native_dpa'),
            ['--disaggregation-mode', 'decode', '--attn-dp-size', '8'], env)
        self.assertEqual(env['ULACACHE_PD_CACHE_KIND'], 'ulacache')
        self.assertIsNone(self.calls.pop()['target_cache_installer']())

    def test_unknown_ownership_or_unsafe_retirement_fails_before_installation(self):
        from types import SimpleNamespace
        config = SimpleNamespace(adapter='native_dpa')
        cases = [([], {}), (['--attn-dp-size', '8'],
            {'SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE': '0'}),
            (['--attn-dp-size', '8', '--disaggregation-transfer-backend', 'nixl'], {})]
        for args, env in cases:
            before = dict(env)
            with self.assertRaises((ValueError, NotImplementedError)):
                self.module.configure_native_launch(config,
                    ['--disaggregation-mode', 'decode'] + args, env)
            self.assertEqual(env, before)
        self.assertEqual(self.calls, [])

    def test_native_prefix_mtp_validation_requires_an_explicit_true_capability(self):
        import __future__
        import ast
        import logging
        import os
        from pathlib import Path
        from types import SimpleNamespace

        path = Path(os.environ['HISPARSE_NATIVE_SOURCE_ROOT']) / 'python/sglang/srt/arg_groups/pd_disaggregation_hook.py'
        names = {'register_decode_radix_speculation_capability',
                 '_ulacache_pd_radix_supported', 'handle_pd_disaggregation'}
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
        namespace = dict(_decode_radix_speculation_capabilities={},
            resolving_view=lambda value: value, resolved_view=lambda value: value,
            attn_dp_enabled_of=lambda _: True, logger=logging.getLogger('test'),
            envs=SimpleNamespace(SGLANG_DISAGG_STAGING_BUFFER=SimpleNamespace(get=lambda: False)),
            declare_resolution=lambda value, _reason, **kw: value.__dict__.update(kw))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                     flags=__future__.annotations.compiler_flag), namespace)
        config = SimpleNamespace(disaggregation_mode='decode',
            disaggregation_transfer_backend='mooncake',
            disaggregation_decode_allocation_policy='default', dcp_size=1,
            disaggregation_decode_host_receive_threshold=1, enable_hisparse=False,
            disaggregation_decode_enable_radix_cache=True, speculative_algorithm='EAGLE',
            disaggregation_decode_extra_slots=0, enable_pd_role_switch=False)
        validate = namespace['handle_pd_disaggregation']
        with self.assertRaisesRegex(ValueError, 'incompatible with speculative'):
            validate(config)
        register = namespace['register_decode_radix_speculation_capability']
        register('not-a-boolean', lambda _: 1)
        with self.assertRaisesRegex(ValueError, 'incompatible with speculative'):
            validate(config)
        register('validated-backend', lambda value: value is config)
        validate(config)
        self.assertFalse(config.disable_radix_cache)
        with self.assertRaises(ValueError):
            register('validated-backend', lambda _: True)


def test_pd_checker_requires_incremental_bytes_and_native_retirement():
    import importlib.util
    from pathlib import Path
    import pytest
    path = Path(__file__).parents[1] / 'check_pd.py'
    spec = importlib.util.spec_from_file_location('_pd_check_test', path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    records = [dict(room=17, decode_rank=1, turn=1, input_tokens=1024)]
    receive = dict(kind='direct-receive', room=17, rank=1,
        transport_quiesced=True, payload_staging_bytes=0,
        prefix_records=960, received_records=64)
    send = dict(kind='direct-send', room=17, rank=1, records=64, bytes=64*51824, status=0)
    assert checker.validate_audits(records, [receive, send], 51824)[0]['received_tokens'] == 64
    for change in [dict(transport_quiesced=False), dict(payload_staging_bytes=1),
                   dict(prefix_records=0), dict(received_records=63)]:
        with pytest.raises(ValueError):
            checker.validate_audits(records, [dict(receive, **change), send], 51824)
    with pytest.raises(ValueError):
        checker.validate_audits(records, [receive, dict(send, bytes=1)], 51824)
    with pytest.raises(LookupError):
        checker.validate_audits(records, [send], 51824)


def test_native_pd_health_check_does_not_advertise_direct_destinations():
    import __future__
    import ast
    import logging
    import os
    from pathlib import Path
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    import pytest

    root = os.environ.get('HISPARSE_NATIVE_SOURCE_ROOT')
    if not root:
        pytest.skip('Set HISPARSE_NATIVE_SOURCE_ROOT to the patched framework')
    base = Path(root) / 'python/sglang/srt/disaggregation'
    native_fake = next(n for n in ast.parse((base / 'utils.py').read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name == '_is_fake_transfer')
    fake_class = next(n for n in ast.parse((base / 'fake/conn.py').read_text()).body
                      if isinstance(n, ast.ClassDef) and n.name == 'FakeKVReceiver')
    fake_send = next(n for n in fake_class.body
                     if isinstance(n, ast.FunctionDef) and n.name == 'send_metadata')
    adapter = base.parent / 'ulacache/pd_direct_adapter.py'
    install = next(n for n in ast.parse(adapter.read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'install_receiver')
    hooks = [n for n in install.body if isinstance(n, ast.FunctionDef)
             and n.name in {'allocate', 'metadata'}]
    allocator = NS(_udcache_receiving_records=False)
    fake = NS()
    receiver = Mock()
    receiver.page_size = 64
    receiver.reserve.return_value.to_bytes.return_value = b'direct-addresses'
    queue = NS(token_to_kv_pool_allocator=allocator,
               token_to_kv_pool=NS(pd_direct_receiver=receiver))
    calls = []
    ns = dict(FAKE_BOOTSTRAP_HOST='FAKE',
              get_disagg=lambda: NS(disaggregation_transfer_backend='mooncake'),
              logger=logging.getLogger('native-fake-test'), time=__import__('time'))
    def allocated(self, req):
        calls.append(allocator._udcache_receiving_records)
    def sent(self, request, rows, page, state=None, **kwargs):
        if request.req.bootstrap_host == 'FAKE':
            ns['send_metadata'](fake, rows, state_indices=state, **kwargs)
        return 'native-result'
    ns.update(original_allocate=allocated, original_metadata=sent)
    exec(compile(ast.Module(body=[native_fake, fake_send, *hooks], type_ignores=[]),
                 str(adapter), 'exec', flags=__future__.annotations.compiler_flag), ns)
    for host in ['FAKE', 'prefill.example']:
        req = NS(bootstrap_host=host, bootstrap_room=7, rid='check')
        ns['allocate'](queue, req)
        assert allocator._udcache_receiving_records is False
        assert ns['metadata'](queue, NS(req=req), [64], 64,
                              decode_prefix_len=0) == 'native-result'
        if host == 'FAKE':
            receiver.reserve.assert_not_called()
            receiver.metadata_sent.assert_not_called()
            assert fake.has_sent_metadata
    assert calls == [False, True]
    receiver.reserve.assert_called_once()
    receiver.metadata_sent.assert_called_once()
