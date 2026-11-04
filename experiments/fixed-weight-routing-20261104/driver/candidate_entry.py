"""Private, hash-bound checkpoint-policy override. Not parameter publication."""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

WORK=Path(__file__).resolve().parent
SOURCE=(WORK/'sglang_source').resolve()
for name,expected in json.loads((WORK/'runtime-source-manifest.json').read_text()).items():
    if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('actual frozen integration source changed: '+name)
sys.path.insert(0,str(SOURCE))
import sglang
if not Path(sglang.__file__).resolve().is_relative_to(SOURCE):
    raise RuntimeError('SGLang escaped frozen source')
from sglang.srt.managers import schedule_batch

ORIGINAL_PREPARE_FOR_DECODE=schedule_batch.ScheduleBatch.prepare_for_decode
ORIGINAL_PREPARE_FOR_EXTEND=schedule_batch.ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend
path=WORK/'source/sglang/srt/managers/schedule_batch.py'
manifest=json.loads((WORK/'policy-source-manifest.json').read_text())
if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['candidate_sha256']:
    raise RuntimeError('private checkpoint-policy candidate changed')
enabled=os.environ.get('PROJECT_B_PREFILL_CHECKPOINT_POLICY')=='1'
if enabled:
    tree=ast.parse(path.read_text())
    names={'prepare_for_decode','_mamba_radix_cache_v2_req_prepare_for_extend'}
    functions=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(functions)!=2:raise RuntimeError('unexpected candidate method set')
    for node in functions:
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),schedule_batch.__dict__)
        setattr(schedule_batch.ScheduleBatch,node.name,schedule_batch.__dict__[node.name])
print('PREFILL_CHECKPOINT_POLICY='+json.dumps({'enabled':enabled,'source':str(SOURCE),
      'candidate_sha256':manifest['candidate_sha256'],'scope':'fixed-load quality prototype, original weight authority retained'}),flush=True)
if os.environ.get('PROJECT_B_CHECKPOINT_QUALITY_TRACE')=='1':
    from sglang.srt.mem_cache.unified_cache.components.mamba_component import MambaComponent
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
    original_insert=MambaComponent.prepare_for_caching_req
    original_copy=MambaAttnBackendBase._track_mamba_state_extend
    seen=set()
    def traced_insert(self,req,insert_params,token_ids_len,is_finished):
        result=original_insert(self,req,insert_params,token_ids_len,is_finished)
        print('CHECKPOINT_INSERT='+json.dumps({'pid':os.getpid(),'rid':req.rid,
            'finished':is_finished,'token_ids_len':token_ids_len,'cache_len':result,
            'last_track_seqlen':req.mamba_last_track_seqlen,'keep_idx':req.mamba_last_track_idx}),flush=True)
        return result
    def traced_copy(self,forward_batch,h,ssm_states,forward_metadata):
        key=(str(ssm_states.dtype),None if h is None else str(h.dtype),
             forward_metadata.track_ssm_h_src.numel(),forward_metadata.track_ssm_final_src.numel())
        if key not in seen:
            seen.add(key)
            print('CHECKPOINT_COPY_PATH='+json.dumps({'pid':os.getpid(),'ssm_dtype':key[0],
                'intermediate_h_dtype':key[1],'intermediate_h_copies':key[2],'final_state_copies':key[3]}),flush=True)
        return original_copy(self,forward_batch,h,ssm_states,forward_metadata)
    MambaComponent.prepare_for_caching_req=traced_insert
    MambaAttnBackendBase._track_mamba_state_extend=traced_copy
if __name__=='__main__':
    runpy.run_module('sglang.launch_server',run_name='__main__')
