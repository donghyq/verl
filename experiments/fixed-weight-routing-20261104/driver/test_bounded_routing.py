import asyncio,copy,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
import screen
from observed_directory import ObservedDirectory
from test_screen import fixtures

def router(root,urls):
    calibration={'service_seconds':1,'advisor':{'source_receipt_sha256':'a'*64,
        'prompt_tokens_min':1,'prompt_tokens_max':9000,'prefill_s_per_token':.01,
        'host_read_s_per_token':0,'host_read_fixed_s':0}}
    return screen.JointRouter(root,screen.Wire(root,time.time()+30),urls,'unused',
        dict(zip(urls,['opaque-A','opaque-B'])),calibration,'cpu-fixture-model','bounded-cpu-only',fixed_weights=True)

class BoundedRoutingTests(unittest.TestCase):
    def test_one_real_native_query_and_original_advisor_decision(self):
        with fixtures() as (urls,counts),tempfile.TemporaryDirectory() as root:
            async def run():
                r=router(root,urls)
                try:
                    h,row=await r.headers(screen.request_body(list(range(100)),'salt',4))
                    self.assertEqual(h['x-smg-target-worker-id'],'opaque-A')
                    self.assertEqual(row['status'],'shadow_estimate')
                    self.assertEqual(counts['native_probes'],2) # One fanout to two real native endpoints.
                    self.assertEqual(counts['generate'],0)
                finally:r.close()
            asyncio.run(run())

    def test_lock_wait_and_slow_producer_fall_back_with_no_sample_failure(self):
        with fixtures() as (urls,counts),tempfile.TemporaryDirectory() as root:
            async def run():
                r=router(root,urls);release=threading.Event();actual=r.client.query
                def slow(q):release.wait(.5);return actual(q)
                r.client.query=slow
                try:
                    rows=await asyncio.gather(*(r.headers(screen.request_body(list(range(100)),'salt',4)) for _ in range(16)))
                    self.assertTrue(all(h=={} and v['status']=='fallback' for h,v in rows))
                    self.assertLess(max(v['route_seconds'] for _,v in rows),.25)
                    h,row=await r.headers(screen.request_body(list(range(100)),'salt',4))
                    self.assertEqual(row['reason'],'query_in_flight')
                    release.set();await asyncio.shield(r.producer_task);r.client.query=actual
                    h,row=await r.headers(screen.request_body(list(range(100)),'salt',4))
                    self.assertTrue(h);self.assertEqual(counts['generate'],0)
                finally:release.set();r.close()
            asyncio.run(run())

    def test_advice_timeout_does_not_rebind_an_inflight_query(self):
        with fixtures() as (urls,counts),tempfile.TemporaryDirectory() as root:
            async def run():
                r=router(root,urls);release=threading.Event();started=threading.Event()
                class SlowObservation(ObservedDirectory):
                    def query(self,q):started.set();release.wait(.5);return super().query(q)
                try:
                    with patch.object(screen,'ObservedDirectory',SlowObservation):
                        h,row=await r.headers(screen.request_body(list(range(100)),'salt',4))
                        self.assertFalse(h);self.assertTrue(started.is_set())
                        await asyncio.sleep(.03) # Advice ticket has expired, underlying query is still alive.
                        self.assertFalse(r.advisor.pending.done())
                        probes=counts['native_probes']
                        h,row=await r.headers(screen.request_body(list(range(100)),'salt',4))
                        self.assertEqual(row['reason'],'query_in_flight');self.assertEqual(counts['native_probes'],probes)
                        release.set();await asyncio.shield(r.advisor.pending)
                        if r.advice_ticket is not None:await asyncio.shield(r.advice_ticket)
                    h,row=await r.headers(screen.request_body(list(range(100)),'salt',4));self.assertTrue(h)
                finally:release.set();r.close()
            asyncio.run(run())

    def test_closed_native_client_does_not_abort_a_sample(self):
        with fixtures() as (urls,counts),tempfile.TemporaryDirectory() as root:
            async def run():
                r=router(root,urls);r.client.close()
                try:
                    h,row=await r.headers(screen.request_body(list(range(100)),'salt',4))
                    self.assertFalse(h);self.assertEqual(row['status'],'fallback');self.assertEqual(counts['generate'],0)
                finally:r.close()
            asyncio.run(run())

    def test_observation_is_bound_to_request_version_namespace_and_generation(self):
        with fixtures() as (urls,counts),tempfile.TemporaryDirectory() as root:
            r=router(root,urls)
            try:
                query={'model':r.model,'tenant_id':'component-screen','lora_name':'','block_size':1,
                    'token_ids':list(range(100)),'cache_salt':'salt','expected_weight_version':'1',
                    'native_allowed_instances':r.client.instances}
                observed=r.client.query(query)
                bound={**query,'native_expected_rank_generations':{k:v['native_query']['rank_generations'] for k,v in observed.items()}}
                snapshot=ObservedDirectory(observed,query);self.assertEqual(snapshot.query(bound),observed)
                for key,value in [('token_ids',[5]),('cache_salt','other'),('expected_weight_version','2'),
                    ('native_allowed_instances',{'foreign':'url'}),('native_expected_rank_generations',{}),('lora_name','other')]:
                    bad=copy.deepcopy(bound);bad[key]=value
                    with self.assertRaises(ValueError):snapshot.query(bad)
                expired=ObservedDirectory(observed,query,clock=lambda:time.time()+3)
                with self.assertRaises(ValueError):expired.query(bound)
                snapshot.close()
                with self.assertRaises(TimeoutError):snapshot.query(bound)
            finally:r.close()

if __name__=='__main__':unittest.main()
