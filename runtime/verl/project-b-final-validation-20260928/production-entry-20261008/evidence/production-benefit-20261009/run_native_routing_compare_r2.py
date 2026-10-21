"""Fixed ABBA comparison of native routing on the original business inputs."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import urllib.request

from verl.workers.rollout.sglang_gateway_client import GatewayClientConfig, SGLangGatewayClient


def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def http(url, data=None):
    request=urllib.request.Request(url,data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type':'application/json'},method='POST' if data is not None else 'GET')
    with urllib.request.urlopen(request,timeout=180) as response:
        raw = response.read()
        if url.endswith('/flush_cache'):
            return {'http_status':response.status, 'text':raw.decode()}
        return json.loads(raw)


async def window(args, rows, policy, sequence_index):
    out=Path(args.output)
    workers=args.worker_url
    for url in workers:
        body=http(url+'/flush_cache',{})
        if body.get('http_status') != 200 or not body.get('text','').startswith('Cache flushed.'):
            raise RuntimeError('actual engine did not confirm idle cache flush')
        if http(url+'/model_info').get('weight_version')!='0':
            raise RuntimeError('actual weights changed during frozen routing comparison')
    log=(out/f'gateway-{sequence_index}-{policy}.log').open('w')
    command=['smg','launch','--host','127.0.0.1','--port','32263','--policy',policy,
        '--enable-rl','--disable-retries','--worker-urls',*workers]
    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    client=SGLangGatewayClient(GatewayClientConfig(base_url='http://127.0.0.1:32263',
        max_model_len=16384,prompt_length=12288,response_length=1024,
        expected_worker_urls=tuple(workers),request_timeout_s=180,admission_timeout_s=30))
    observations=[]; outputs=[]
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError('owned native gateway exited before measurement')
            try:
                body=http('http://127.0.0.1:32263/workers')
                if len(body.get('workers',[]))==len(workers):
                    break
            except OSError:
                pass
            await asyncio.sleep(.5)
        else:
            raise TimeoutError('owned Gateway readiness deadline')
        started=time.monotonic()
        async def one(index,row):
            prompt=list(row['input_ids'])
            start=time.monotonic()
            item={'source_line_number':row['line_number'],'prompt_tokens':len(prompt),
                  'output_tokens':None,'cached_tokens':None,'elapsed_s':None,'error':None}
            try:
                output=await client.generate(f'fixed-routing-{sequence_index}-{index}',prompt_ids=prompt,
                    sampling_params={'max_new_tokens':128,'temperature':0,'logprobs':True,'sampling_seed':42})
                fields=output.extra_fields
                if not output.token_ids or len(output.log_probs)!=len(output.token_ids):
                    raise ValueError('actual output tokens and logprob do not align')
                if fields.get('global_steps')!=0 or not fields.get('salt_match'):
                    raise ValueError('actual native identity failed')
                cached=fields.get('num_cached_tokens')
                item.update(output_tokens=len(output.token_ids),cached_tokens=cached,
                    uncached_prompt_tokens_by_native_metadata=len(prompt)-cached if isinstance(cached,int) else None,
                    routed_worker_id=fields.get('gateway_worker_id'),engine_weight_version=fields.get('engine_weight_version'),
                    output_digest=sha(output.token_ids),stop_reason=output.stop_reason,
                    native_meta_info=fields.get('native_meta_info'))
                outputs.append({'source_line_number':row['line_number'],'output_ids':output.token_ids,'log_probs':output.log_probs})
            except Exception as exc:
                item['error']=type(exc).__name__
            finally:
                item['elapsed_s']=time.monotonic()-start
                observations.append(item)
        for batch in range(0,len(rows),args.concurrency):
            await asyncio.gather(*(one(index,rows[index]) for index in range(batch,min(batch+args.concurrency,len(rows)))))
        elapsed=time.monotonic()-started
        ordered=sorted(observations,key=lambda item:item['source_line_number'])
        durations=[item['elapsed_s'] for item in ordered]
        cached=[item['cached_tokens'] for item in ordered]
        result={'policy':policy,'sequence_index':sequence_index,'requests':len(rows),
            'concurrency':args.concurrency,'wall_s':elapsed,'request_p50_s':statistics.median(durations),
            'request_p95_s':sorted(durations)[max(0,int(.95*len(durations))-1)],
            'cached_tokens':sum(cached) if all(isinstance(x,int) for x in cached) else None,
            'generated_tokens':sum(item['output_tokens'] or 0 for item in ordered),
            'failed_requests':sum(item['error'] is not None for item in ordered),'observations':ordered,
            'ttft_s':None,'trainer_updates':0,'valid_training_samples_per_gpu_hour':None}
        (out/f'window-{sequence_index}-{policy}.json').write_text(json.dumps(result,indent=2))
        private=out/f'private-outputs-{sequence_index}.json'
        private.write_text(json.dumps(outputs)); os.chmod(private,0o600)
        return result
    finally:
        await client.close()
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=10)
        log.close()


async def run(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    data=json.loads(Path(args.input).read_text())
    rows=data['records'][:args.samples]
    if len(rows)!=args.samples:
        raise ValueError('fixed input count unavailable')
    for row in rows:
        if sha(row['input_ids'])!=row['token_ids_sha256']:
            raise ValueError('original token digest differs')
    (out/'freeze.json').write_text(json.dumps({'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        'ordered_source_rows':[row['line_number'] for row in rows], 'ordered_prompt_digest':sha([row['input_ids'] for row in rows]),
        'policies':['least_load','cache_aware','cache_aware','least_load'],'concurrency':args.concurrency,
        'generation_limit':128,'sampling_seed':42,'temperature':0,
        'only_treatment_difference':'native SMG policy; same engines, weights, input order, cache flush and sampling',
        'positive_requirement':'no failed requests; identical output tokens; consistent benefit across the two ABBA observations',
        'no_long_training_or_extra_data':True},indent=2))
    # Real first input, one discarded token per engine; finish kernel warmup
    # before identical cache flushes. This is neither a sample nor a benchmark.
    warm=[]
    for index,url in enumerate(args.worker_url):
        reply=http(url+'/generate',{'rid':f'kernel-warmup-{index}','input_ids':rows[0]['input_ids'],
            'sampling_params':{'max_new_tokens':1,'temperature':0,'sampling_seed':42},'cache_salt':'0','stream':False})
        warm.append({'worker_index':index,'actual_output_tokens':len(reply.get('output_ids',[])),'meta_info':reply.get('meta_info')})
    (out/'warmup-receipts.json').write_text(json.dumps(warm,indent=2))
    results=[]
    for index,policy in enumerate(('least_load','cache_aware','cache_aware','least_load')):
        results.append(await window(args,rows,policy,index))
    mappings=[{item['source_line_number']:item.get('output_digest') for item in result['observations']} for result in results]
    identical=all(mapping==mappings[0] for mapping in mappings[1:])
    no_errors=all(result['failed_requests']==0 for result in results)
    base=[result for result in results if result['policy']=='least_load']
    treatment=[result for result in results if result['policy']=='cache_aware']
    delta=[1-t['wall_s']/b['wall_s'] for b,t in zip(base,treatment)]
    summary={'status':'OBSERVATION_ONLY' if no_errors and identical else 'CORRECTNESS_OR_EXECUTION_FAILED',
        'routing_benefit':'CANDIDATE' if no_errors and identical and all(d>.05 for d in delta) else 'NO_STABLE_BENEFIT',
        'all_output_tokens_identical':identical,'all_requests_succeeded':no_errors,'paired_wall_fraction_saved':delta,
        'windows':[{key:value for key,value in result.items() if key!='observations'} for result in results],
        'proof_scope':'two repetitions of same-host native routing with real business first-turn inputs; no tool execution, training or production claim'}
    (out/'comparison.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--worker-url',required=True,action='append')
    parser.add_argument('--samples',type=int,default=32); parser.add_argument('--concurrency',type=int,default=4)
    asyncio.run(run(parser.parse_args()))
