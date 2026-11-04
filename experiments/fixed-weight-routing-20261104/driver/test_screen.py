"""CPU HTTP protocol tests. Synthetic timing is never performance evidence."""
import asyncio
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from screen import JointRouter, Wire, complete_groups, digest, request_body, run_arm, sample
from screen_window import matches, validate


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


@contextlib.contextmanager
def fixtures(ports=None):
    counts = {"generate": 0, "targets": [], "model_version": "1", "busy_a": False}
    servers, urls = [], []
    class Engine(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def send(self, status, body):
            raw = json.dumps(body).encode(); self.send_response(status)
            self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw)))
            self.end_headers(); self.wfile.write(raw)
        def do_GET(self):
            delay=counts.get('registration_ready_at',0)-time.time()
            if delay>0:time.sleep(delay)
            if self.path=='/model_info':time.sleep(counts.get('model_info_delay',0))
            if self.path in ("/health", "/health_generate"): self.send(200, {"ok": True})
            elif self.path in ("/model_info", "/server_info"):
                self.send(200, {"model_path": str(Path.home()/"models/Qwen3.5-35B-A3B-BF16"),
                     "weight_version": counts["model_version"], "architectures": ["Qwen3_5MoeForCausalLM"], "tp_size": 1})
            else: self.send(404, {})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            idx = self.server.index
            if self.path == "/token_kv_snapshot":
                counts['native_probes']=counts.get('native_probes',0)+1
                query = body["cache_query"];tokens = query["token_ids"];n = len(tokens)-1 if idx == 0 else 0
                self.send(200, {"status": "PASS", "ranks": [{"tp_rank": 0,"status":"PASS",
                    "query_digest": query["query_digest"],"prompt_tokens":len(tokens),"cache_salt":query["cache_salt"],
                    "actual_weight_version":counts["model_version"],"coverage_source":"native_read_only_full_mamba_tree",
                    "host_state_advertised":False,"generation_admission_authorized":False,
                    "engine_instance_id":f"cpu-fixture-{idx}","cache_generation":str(idx)*64,
                    "rank_match":dict.fromkeys(("device","cpu_local","cpu_share","disk"),n),
                    "native_complete_device_prefix":n,"waiting_requests":100 if idx==0 and counts["busy_a"] else 0,
                    "running_requests":4 if idx==0 and counts["busy_a"] else 0,"observed_at_unix":time.time()}]})
            elif self.path == "/generate":
                counts["generate"] += 1;counts["targets"].append(dict(self.headers))
                budget=body["sampling_params"]["max_new_tokens"];ids=[50+i for i in range(budget)]
                self.send(200, {"text":"CPU_FIXTURE_ONLY","output_ids":ids,"seen_salt":body.get("cache_salt"),
                    "seen_session":self.headers.get("x-smg-routing-key"),"engine":idx,
                    "meta_info":{"finish_reason":{"type":"length"},"weight_version":"1","completion_tokens":budget,
                       "output_token_logprobs":[[-.1,i,None] for i in ids],"cached_tokens":0,"e2e_latency":.001}})
            else:self.send(200,{})
    try:
        for idx in (0,1):
            server=Server(("127.0.0.1",0 if ports is None else ports[idx]),Engine);server.index=idx
            threading.Thread(target=server.serve_forever,daemon=True).start();servers.append(server)
            urls.append(f"http://127.0.0.1:{server.server_port}")
        yield urls, counts
    finally:
        for server in servers:server.shutdown();server.server_close()


class ScreenTests(unittest.TestCase):
    def test_window_expiry_scope_uuid_and_process_identity(self):
        work=Path('/tmp/fixed-screen-test')
        window={"status":"APPROVED","approval_source":"user_message","approval_turn_id":"previous-user-reply",
            "codelab_id":"617728841","work_dir":str(work),"gpu_count":2,
            "scope":"FIXED_WEIGHT_ROUTING_COMPONENT_SCREEN","keep_instance_after_window":True,
            "actual_device_name":"NVIDIA A800-SXM4-80GB","started_at":1000,"absolute_deadline":3400,
            "gpu_uuids":["GPU-a","GPU-b"]}
        self.assertEqual(validate(window,1001,work),3400)
        for change in [{"absolute_deadline":3401},{"absolute_deadline":float('nan')},{"gpu_count":4},
                       {"scope":"TRAIN"},{"gpu_uuids":["GPU-a","GPU-a"]},{"keep_instance_after_window":False}]:
            with self.assertRaises(ValueError):validate({**window,**change},1001,work)
        with self.assertRaises(ValueError):validate(window,3400,work)
        row={"pgid":7,"start_ticks":"10","port":30513,"identity_argument":"/work/engine_entry.py"}
        cmd=b'python\0/work/engine_entry.py\0--port\0'+b'30513\0'
        self.assertTrue(matches(row,cmd,7,"10"));self.assertFalse(matches(row,cmd,7,"11"))

    def test_real_shadow_advisor_and_real_http_directory_pick_complete_prefix(self):
        with fixtures() as (urls, counts), tempfile.TemporaryDirectory() as root:
            async def test():
                calibration={"service_seconds":1,"advisor":{"source_receipt_sha256":"a"*64,
                    "prompt_tokens_min":1,"prompt_tokens_max":9000,"prefill_s_per_token":.01,
                    "host_read_s_per_token":0,"host_read_fixed_s":0}}
                router=JointRouter(root,Wire(root,time.time()+30),urls,"unused",dict(zip(urls,["opaque-A","opaque-B"])),
                     calibration,"cpu-fixture-model","cpu-protocol-only",fixed_weights=True)
                try:
                    headers,row=await router.headers(request_body(list(range(100)),"salt",4))
                    self.assertEqual(row["status"],"shadow_estimate")
                    self.assertEqual(headers["x-smg-target-worker-id"],"opaque-A")
                    self.assertEqual(row["candidates"]["0"]["declared_device_prefix"],99)
                    self.assertEqual(row["production_advisor_mode"],"shadow")
                    counts["busy_a"]=True
                    headers,row=await router.headers(request_body(list(range(100)),"salt",4))
                    self.assertEqual(headers["x-smg-target-worker-id"],"opaque-B")
                    counts["model_version"]="2"
                    headers,row=await router.headers(request_body(list(range(100)),"salt",4))
                    self.assertFalse(headers)
                finally:router.close()
            asyncio.run(test())

    def test_dynamic_weights_refused(self):
        with self.assertRaises(ValueError):
            JointRouter(None,None,None,None,None,None,None,None,fixed_weights=False)

    def test_complete_group_rejects_missing_duplicate_or_boolean_member(self):
        rows=[{"group":0,"member":m,"turn":t,"finished":1} for m in range(8) for t in range(2)]
        self.assertEqual(len(complete_groups(rows,8,2)),1)
        with self.assertRaises(ValueError):complete_groups(rows[:-1],8,2)
        with self.assertRaises(ValueError):complete_groups(rows+rows[:1],8,2)
        rows[0]["member"]=False
        with self.assertRaises(ValueError):complete_groups(rows,8,2)

    def test_http200_abort_cannot_count_as_completed_sample(self):
        with self.assertRaises(ValueError):
            sample({"meta_info":{"finish_reason":{"type":"abort"},"completion_tokens":0}},4)

    def test_unknown_http409_is_one_attempt(self):
        hits=[]
        class Reject(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length",0)))
                raw=b'{"error":{"code":"native_target_rejected_before_admission"}}'
                hits.append(1);self.send_response(409);self.send_header("Content-Length",str(len(raw)));self.end_headers()
                self.wfile.write(raw)
        server=Server(("127.0.0.1",0),Reject);threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as root:
                code,body,_=Wire(root,time.time()+5).sync("reject",f"http://127.0.0.1:{server.server_port}","/generate",{})
                self.assertEqual(code,409);self.assertEqual(len(hits),1)
        finally:server.shutdown();server.server_close()


if __name__=="__main__":unittest.main()
