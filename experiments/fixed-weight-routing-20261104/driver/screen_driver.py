"""Three native calibrations then three predeclared paired component screens."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from commands import checked_command
from screen import Wire, compare_samples, digest, request_body, run_arm, sample, save


WORK=Path(__file__).resolve().parent
URLS=["http://127.0.0.1:30513","http://127.0.0.1:30514"]
GATEWAY="http://127.0.0.1:30515"
BINARY=Path.home()/"project-b-gateway-cache-salt-20261104-r1/binary/smg"
MODEL=Path.home()/"models/Qwen3.5-35B-A3B-BF16"


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)


async def main():
    from transformers import AutoTokenizer
    window=json.loads((WORK/"registered-window.json").read_text())
    gate=json.loads((WORK/"cpu-cli-gate.json").read_text())
    wire=Wire(WORK,window["absolute_deadline"]-60)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    filler=tokenizer("The warehouse robot scans shelves, checks item counts, and records verified locations. ",
                      add_special_tokens=False)["input_ids"]
    def prompt(index):
        head=tokenizer(f"Inventory task {index}. Record the next observation and continue the inspection.\n",
                       add_special_tokens=False)["input_ids"]
        return (head+filler*(8192//len(filler)+1))[:8192]
    prompts=[prompt(i) for i in range(4)]
    suffix=tokenizer("\nTool observation: count verified; inspect the next shelf.\n",add_special_tokens=False)["input_ids"]
    save(WORK/"workload-materialized.json",{"prompts":prompts,"suffix":suffix,"digest":digest([prompts,suffix]),
        "limits":"Synthetic fixed-weight inventory/tool-observation load; no reward model or optimizer. Calibration uses prompt0, confirmation prompts2/3."})
    reused=WORK/"reused-calibration.json"
    if reused.exists():
        if hashlib.sha256(reused.read_bytes()).hexdigest()!=window.get("reused_calibration_sha256"):
            raise ValueError("predeclared prior calibration digest changed")
        calibration=json.loads(reused.read_text())
        save(WORK/"calibration.json",calibration)
    else:
        slopes,services,pairs=[],[],[]
        for idx,url in enumerate(URLS):
            for repeat in range(3):
                salt=f"component-calib-{idx}-{repeat}"
                code,seed,_=await wire.call("calib-seed",url,"/generate",request_body(prompts[1],salt,65))
                assert code==200;ids,_=sample(seed,65)
                history=prompts[1]+ids+suffix
                payloads=[("warm",request_body(history,salt,4)),("cold",request_body(history,salt+"-cold",4))]
                if repeat%2:payloads.reverse()
                samples={}
                for name,body in payloads:
                    code,response,row=await wire.call(f"calib-{idx}-{repeat}-{name}",url,"/generate",body)
                    assert code==200;sample(response,4);samples[name]=(response,row)
                warm,wr=samples["warm"];cold,cr=samples["cold"]
                equal=sample(warm,4)==sample(cold,4)
                prefix=warm["meta_info"]["cached_tokens"]
                assert cold["meta_info"]["cached_tokens"]==0
                pair={"engine":idx,"repeat":repeat,"warm_seconds":wr["wall_seconds"],"cold_seconds":cr["wall_seconds"],
                      "warm_cached_tokens":prefix,"exact_ids_and_logprobs":equal}
                pairs.append(pair)
                save(WORK/"calibration-pairs.json",pairs)
                if not equal or prefix!=8192:
                    raise ValueError("8K calibration counterpart not equivalent or did not reuse complete mixed prefix; preserve receipts, do not expand")
                slopes.append(max(0,cr["wall_seconds"]-wr["wall_seconds"])/prefix)
            code,response,row=await wire.call("calib-service-32",url,"/generate",
                 request_body(prompts[1],f"calib-service-{idx}",32))
            assert code==200;sample(response,32);services.append(row["wall_seconds"])
        calibration={"pairs":pairs,"service_seconds":statistics.median(services),
          "advisor":{"source_receipt_sha256":hashlib.sha256((WORK/"calibration-pairs.json").read_bytes()).hexdigest(),
             "prompt_tokens_min":8192,"prompt_tokens_max":9000,"prefill_s_per_token":statistics.median(slopes),
             "host_read_s_per_token":0,"host_read_fixed_s":0},
          "limits":"A800 local calibration; negative warm/cold samples retained. Queue estimates are approximate, not measured scheduler seconds."}
        save(WORK/"calibration.json",calibration)
    trials=[]

    async def trial(native_variant,arm,label,inputs):
        at=time.time()
        for url in URLS:
            code,_,_=await wire.call(label+"-flush",url,"/flush_cache",{})
            assert code==200
        command=checked_command(gate,native_variant,BINARY,MODEL)
        with (WORK/(label+"-gateway.log")).open("ab") as log:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                 start_new_session=True,env=dict(os.environ,SGLANG_ENABLE_STABLE_TARGET_ROUTING="1"))
        ticks=Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")",1)[1].split()[19]
        owned={"pid":proc.pid,"pgid":os.getpgid(proc.pid),"start_ticks":ticks,"port":int(GATEWAY.rsplit(":",1)[1]),
               "identity_argument":str(BINARY),"command":command,"started":time.time()}
        save(WORK/"gateway-owned.json",owned);save(WORK/(label+"-gateway-owned.json"),owned)
        try:
            while time.time()<window["absolute_deadline"]-90:
                if proc.poll() is not None:raise RuntimeError("Gateway exited; diagnose original log")
                try:
                    code,_,_=await wire.call(label+"-health",GATEWAY,"/health")
                    if code==200:break
                except OSError:pass
                await asyncio.sleep(.1)
            else:raise TimeoutError("Gateway health deadline")
            value=await run_arm(WORK,wire,GATEWAY,URLS,arm,inputs,suffix,calibration,str(MODEL),label,groups=len(inputs))
            value.update(native_variant=native_variant,full_trial_started=at)
        finally:stop(proc)
        value["full_trial_finished"]=time.time()
        value["full_trial_seconds"]=value["full_trial_finished"]-at
        trials.append(value);save(WORK/(label+"-result.json"),value)
        save(WORK/"completed-trials.json",trials)
        return value

    native=[]
    for name in ("native_default","native_tuned","native_session"):
        native.append(await trial(name,name,"calibration-"+name,[prompts[0]]))
    winner=min(native,key=lambda x:x["full_trial_seconds"])["native_variant"]
    save(WORK/"native-baseline-selection.json",{"winner":winner,"metric":"full_trial_seconds",
         "values":{x["native_variant"]:x["full_trial_seconds"] for x in native},
         "selected_before_first_joint_gpu_trial":True,"confirmation_prompt_ids":[2,3]})
    paired=[]
    for repeat,order in enumerate((("native","joint"),("joint","native"),("native","joint"))):
        result={}
        for arm in order:
            if window["absolute_deadline"]-time.time()<180:
                raise TimeoutError("insufficient original window for next paired block; completed receipts retained")
            result[arm]=await trial(winner,"joint" if arm=="joint" else winner,
                                  f"confirm-{repeat}-{arm}",prompts[2:4])
        quality=compare_samples(result["native"],result["joint"])
        paired.append({"pair":repeat,"order":order,"quality":quality,
          "relative_component_throughput_gain":result["native"]["full_trial_seconds"]/result["joint"]["full_trial_seconds"]-1,
          "native_seconds":result["native"]["full_trial_seconds"],"joint_seconds":result["joint"]["full_trial_seconds"]})
        save(WORK/"paired-results.json",paired)
        if not quality["equivalent"]:
            save(WORK/"screen-report.json",{"status":"QUALITY_GATE_FAILED_AFTER_PAIRED_BLOCK",
                "native_baseline":winner,"pairs":paired,"optimizer_updates":0,
                "formal_rl_gain_confirmed":False,"performance_gate_passed":False,
                "limits":"Stop at first unequal paired block; preserve raw receipts and all costs."})
            raise ValueError("paired sample IDs/input/logprobs differ; diagnose before another GPU block")
    # Complete pending GPU ordinary and removal checks within this same window.
    # Ordinary requests were executed in all three native calibration trials.
    command=checked_command(gate,winner,BINARY,MODEL)
    with (WORK/"final-removal-gateway.log").open("ab") as log:
        proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,SGLANG_ENABLE_STABLE_TARGET_ROUTING="1"))
    ticks=Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")",1)[1].split()[19]
    owned={"pid":proc.pid,"pgid":os.getpgid(proc.pid),"start_ticks":ticks,"port":int(GATEWAY.rsplit(":",1)[1]),
           "identity_argument":str(BINARY),"command":command,"started":time.time()}
    save(WORK/"gateway-owned.json",owned);save(WORK/"final-removal-gateway-owned.json",owned)
    try:
        for _ in range(100):
            try:
                code,registry,_=await wire.call("final-registry",GATEWAY,"/workers")
                if code==200 and len(registry.get("workers",[]))==2:break
            except OSError:pass
            await asyncio.sleep(.1)
        else:raise TimeoutError("removal Gateway registry")
        worker=registry["workers"][0]
        code,_,_=await wire.call("remove-owned-worker",GATEWAY,"/workers/"+worker["id"],method="DELETE")
        assert code in (200,202)
        for _ in range(100):
            code,current,_=await wire.call("wait-owned-removal",GATEWAY,"/workers")
            if not any(x["id"]==worker["id"] for x in current.get("workers",[])):break
            await asyncio.sleep(.1)
        else:raise TimeoutError("worker removal")
        code,rejected,_=await wire.call("removed-worker-rejected",GATEWAY,"/generate",request_body(prompts[0],"remove",4),
             {"x-smg-target-worker-id":worker["id"],"x-sglang-native-target":"{}"})
        assert code==409 and rejected.get("error",{}).get("code")=="native_target_rejected_before_admission"
        save(WORK/"pending-gpu-checks.json",{"ordinary_generation_passed":True,"removal_rejection_passed":True,
                                            "removed_worker_http_status":409})
    finally:stop(proc)
    passed=all(x["quality"]["equivalent"] and x["relative_component_throughput_gain"]>=.05 for x in paired)
    save(WORK/"screen-report.json",{"status":"COMPONENT_PAIR_SCREEN_PASS_PENDING_ALL_COST_AND_RL" if passed else "COMPONENT_PAIR_SCREEN_DID_NOT_PASS",
        "native_baseline":winner,"pairs":paired,"screening_threshold":.05,"passed_steady_component_gate":passed,
        "optimizer_updates":0,"formal_rl_gain_confirmed":False,"dynamic_parameter_publication":False,
        "limits":"Synthetic fixed-weight screening only; all window/startup/idle costs are reported by outer supervisor and must be assessed before any actual RL confirmation."})


if __name__=="__main__":asyncio.run(main())
