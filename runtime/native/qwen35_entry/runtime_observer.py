"""NOT_RUNTIME_VALIDATED: transparent observations around the frozen native fit.

Delegates every training/publication call once. Parameter samples are read in
the existing audit RPC, without a new RPC or collective. No missing value is
filled with zero. CPU tests exercise protocol/dispatch only, not model success.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import uuid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def unavailable(reason):
    if not isinstance(reason, str) or not reason:
        raise ValueError('Missing observations need a concrete reason')
    return {'value': None, 'reason': reason}


def plain(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'data') and not isinstance(value, (str, bytes, dict, list, tuple)):
        return plain(value.data)
    return value


def source_of(function):
    function = inspect.unwrap(function)
    path = inspect.getsourcefile(function)
    return {'module': function.__module__, 'qualified_name': function.__qualname__,
            'path': str(Path(path).resolve()) if path else None,
            'sha256': digest(path) if path and Path(path).is_file() else None}


def process_observation(*, expected_files=None, expected_repo=None, worker=None):
    """Observe already imported source and assigned devices; do not load a model."""
    modules = ['verl', 'run_native_training', 'native_training_audit', 'actor_worker',
               'training_update_audit', 'runtime_observer', 'business_adapter']
    engine = None
    if worker is not None:
        from actor_worker import resolve_training_worker
        selected = resolve_training_worker(worker)
        engine = selected.actor.engine if hasattr(selected, 'actor') else selected.engine
        modules.append(type(engine).__module__)
    source = []
    expected_files = expected_files or {}
    for name in dict.fromkeys(modules):
        module = sys.modules.get(name)
        filename = getattr(module, '__file__', None)
        if filename is None:
            source.append({'module': name, 'path': None, 'sha256': None, 'matches_frozen': None,
                           'reason': 'Module not imported in this process at this observation'})
            continue
        path = str(Path(filename).resolve()); actual = digest(path)
        expected = expected_files.get(path)
        if name == 'verl' and expected_repo and Path(path) != Path(expected_repo).resolve()/'verl/__init__.py':
            raise ValueError('Actual verl namespace is outside the frozen runtime repository')
        if expected is not None and actual != expected:
            raise ValueError('Actual imported source differs from frozen bytes: ' + name)
        if name.startswith('verl.') and expected_repo and not Path(path).is_relative_to(Path(expected_repo).resolve()):
            raise ValueError('Actual engine source is outside the frozen runtime repository')
        source.append({'module': name, 'path': path, 'sha256': actual,
                       'matches_frozen': actual == expected if expected else None,
                       'reason': None if expected else 'Loaded module not yet bound by frozen file list'})
    result = {'pid': os.getpid(), 'socket_hostname': socket.gethostname(), 'sources': source,
              'physical_mlp_hostname': unavailable('Join Ray node and GPU UUIDs with owned MLP inventory; socket hostname is not a platform host proof'),
              'ray_node_id': None, 'ray_actor_id': None, 'assigned_ray_gpu_ids': None,
              'gpu_uuid': unavailable('This process has no observed training CUDA device'),
              'rank': getattr(worker, 'rank', None), 'engine_class': type(engine).__qualname__ if engine else None}
    try:
        import ray
        if ray.is_initialized():
            context = ray.get_runtime_context()
            result['ray_node_id'] = str(context.get_node_id())
            result['ray_actor_id'] = str(context.get_actor_id())
            result['assigned_ray_gpu_ids'] = list(ray.get_gpu_ids())
    except Exception as error:
        result['ray_observation_error_type'] = type(error).__name__
    if worker is not None:
        try:
            import torch
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            gpu_uuid = getattr(props, 'uuid', None)
            result['cuda_device_index'] = device
            result['gpu_uuid'] = ({'value': str(gpu_uuid), 'reason': None} if gpu_uuid else
                unavailable('Actual torch device properties do not expose a physical UUID'))
        except Exception as error:
            result['gpu_uuid'] = unavailable('Actual CUDA identity query failed: ' + type(error).__name__)
    template = os.environ.get('NCCL_DEBUG_FILE')
    path = template.replace('%h', socket.gethostname()).replace('%p', str(os.getpid())) if template else None
    result['nccl_info'] = {'configured_level': os.environ.get('NCCL_DEBUG'), 'path': path,
        'file_exists': Path(path).is_file() if path else None,
        'bytes': Path(path).stat().st_size if path and Path(path).is_file() else None,
        'reason': None if path and Path(path).is_file() else 'No process-local NCCL INFO file observed yet'}
    return result


def read_update_with_observation(worker, options):
    """Same audit RPC, actual original post-hook state and local parameter probe."""
    from actor_worker import resolve_training_worker
    selected = resolve_training_worker(worker)
    # The native entry installs its optimizer audit through a local RPC outside
    # observed_factory. The first audited read is still before actor dispatch,
    # so attach here without adding a new RPC or calculating a gradient norm.
    if not hasattr(selected, '_qwen35_engine_step_returns'):
        engine = selected.actor.engine if hasattr(selected, 'actor') else selected.engine
        attach_engine_return_observer(engine, selected)
    from training_update_audit import read_worker_update_audit
    result = read_worker_update_audit(worker)
    info = process_observation(expected_files=options['expected_files'],
                               expected_repo=options['expected_repo'], worker=worker)
    parameters = None
    if options['sample_parameters']:
        from actor_worker import observe_original_parameters
        parameters = observe_original_parameters(worker)
    result['_observer'] = {'process': info, 'parameters': parameters,
        'native_engine_step_returns': getattr(selected, '_qwen35_engine_step_returns', None),
        'parameter_scope': 'Original local parameter shard sample; not a full weight digest'}
    return result


def install_audit_with_observation(worker, output_dir, options):
    from training_update_audit import install_worker_update_audit
    result = install_worker_update_audit(worker, output_dir)
    from actor_worker import resolve_training_worker
    selected = resolve_training_worker(worker)
    engine = selected.actor.engine if hasattr(selected, 'actor') else selected.engine
    attach_engine_return_observer(engine, selected)
    result['_observer'] = {'process': process_observation(expected_files=options['expected_files'],
        expected_repo=options['expected_repo'], worker=worker), 'parameters': None}
    return result


def attach_engine_return_observer(engine, selected):
    """Read the existing native scalar return; no norm calculation or collective."""
    if hasattr(selected, '_qwen35_engine_step_returns'):
        raise RuntimeError('Never attach an engine return observer twice')
    original = engine.optimizer_step
    selected._qwen35_engine_step_returns = []
    source = source_of(original)
    @functools.wraps(original)
    def observed(*args, **kwargs):
        row = {'trainer_step': selected._production_update_audit.current_step,
               'rank': selected._production_update_audit.rank, 'pid': os.getpid(), 'source': source}
        try:
            value = original(*args, **kwargs)
        except BaseException as error:
            row.update(status='ORIGINAL_ENGINE_EXCEPTION', error_type=type(error).__name__,
                       grad_norm=unavailable('Original optimizer_step did not return'))
            selected._qwen35_engine_step_returns.append(row)
            raise
        finite = type(value) in (int, float) and math.isfinite(value)
        row.update(status='ORIGINAL_ENGINE_RETURNED', finite=finite, nonzero=finite and value > 0,
                   grad_norm={'value':value,'reason':None} if finite else unavailable('Original engine return is absent, nonscalar or nonfinite'))
        selected._qwen35_engine_step_returns.append(row)
        return value
    engine.optimizer_step = observed


@contextmanager
def observe_initial_parameter_publication(observer):
    """Observe the one original initialization read, including its real receipt.

    Native run_original_fit imports this helper locally, before the factory's
    wrapped event callback exists at that call site. Do not read engines again
    or substitute a declared/configured starting version.
    """
    import run_v1_functional_gate as gate

    original = gate._initialized_parameter_observation

    @functools.wraps(original)
    def observed(*args, **kwargs):
        value = original(*args, **kwargs)
        observer.initial_publication = deepcopy(value)
        observer.event('native_initial_parameter_publication_returned',
                       publication=deepcopy(value), source=source_of(original))
        return value

    gate._initialized_parameter_observation = observed
    try:
        yield
    finally:
        gate._initialized_parameter_observation = original


def group_membership(keys):
    """Summarize actual native session keys, choosing each session's final chunk."""
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate actual batch key')
    final = {}; all_chunks = {}
    for row, key in enumerate(keys):
        fields = key.rsplit('_', 2)
        if len(fields) != 3 or not fields[0] or not fields[1].isdecimal() or not fields[2].isdecimal():
            raise ValueError('Invalid original uid_session_chunk key')
        uid, member, chunk = fields[0], int(fields[1]), int(fields[2])
        session = (uid, member)
        all_chunks.setdefault(session, []).append({'row': row, 'chunk': chunk, 'key': key})
        if session not in final or chunk > final[session]['chunk']:
            final[session] = {'row': row, 'chunk': chunk, 'key': key}
    groups = {}
    for (uid, member), entry in final.items():
        groups.setdefault(uid, []).append({'member_index': member, **entry,
                                           'chunk_count': len(all_chunks[(uid,member)])})
    for members in groups.values():
        if sorted(m['member_index'] for m in members) != list(range(8)):
            raise ValueError('Actual training group lacks exactly members 0..7')
    return {uid: sorted(members, key=lambda m:m['member_index']) for uid,members in groups.items()}


def input_observation(batch, dispatches):
    """Read original output metadata after stock advantage computation, without editing TQ."""
    import transfer_queue as tq
    groups = group_membership(batch.keys)
    tags = getattr(batch, 'tags', None)
    result = {'group_ids': list(groups), 'complete_groups': len(groups), 'members_per_group': 8,
              'batch_keys_sha256': fingerprint(sorted(batch.keys)), 'groups': [],
              'original_admission_scope': 'Actual batch consumed by stock replay_buffer.sample, after original advantage computation'}
    fields = ['uid','session_id','raw_prompt','extra_fields','prompts','responses','response_mask','rollout_log_probs']
    try:
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        for uid, members in groups.items():
            observed = []; initial_digests = set()
            for member in members:
                row = member['row']; value = data[row]
                raw = plain(value['raw_prompt']); initial = fingerprint(raw); initial_digests.add(initial)
                extra = plain(value['extra_fields'])
                if not isinstance(extra, dict):
                    raise ValueError('Actual extra_fields is not a mapping')
                segments = extra.get('trajectory_segments')
                provenance = None if segments is None else [{k:s.get(k) for k in
                    ('start','end','policy_version','sampling_config_fingerprint','execution_generation')} for s in segments]
                mask = value['response_mask']; logprobs = value['rollout_log_probs']
                import torch
                aligned = mask.ndim == logprobs.ndim == 1 and mask.numel() == logprobs.numel()
                finite = bool(torch.isfinite(logprobs).all()) if aligned else False
                tag = tags[row] if isinstance(tags, list) and len(tags)==len(batch.keys) else None
                actual_uid = plain(value['uid']); actual_member = plain(value['session_id'])
                if isinstance(actual_member, list) and len(actual_member)==1: actual_member=actual_member[0]
                if actual_uid != uid or actual_member != member['member_index']:
                    raise ValueError('Actual uid/session fields disagree with native membership keys')
                observed.append({**member, 'initial_raw_prompt_sha256': initial,
                    'training_prompt_tokens': int(value['prompts'].numel()),
                    'training_response_tokens': int(value['responses'].numel()),
                    'training_effective_model_tokens': int(mask.sum().item()),
                    'behavior_logprob_aligned_finite': aligned and finite,
                    'behavior_segments': provenance,
                    'behavior_segments_reason': None if provenance is not None else 'Actual source output did not expose trajectory_segments',
                    'original_tq_tag': {k:tag.get(k) for k in ('status','min_global_steps','max_global_steps','is_padding')} if tag else None})
            same_initial = len(initial_digests)==1
            dispatched = dispatches.get(uid)
            matched = same_initial and dispatched is not None and dispatched.get('initial_raw_prompt_sha256') in initial_digests
            result['groups'].append({'uid':uid,'members':observed,'initial_prompt_unique_count':len(initial_digests),
                'matches_actual_dispatch_prompt':matched,'dispatch':dispatched})
        result['status']='OBSERVED_ACTUAL_GROUP_METADATA'
    except Exception as error:
        # Observation gaps cannot trigger a repeated update. Preserve the gap and
        # let the unchanged original Trainer decide whether it can train.
        result.update(status='INCOMPLETE_METADATA_NOT_ZERO', error_type=type(error).__name__,
                      error_message_sha256=fingerprint(str(error)))
    result['actual_engine_prefill_tokens']=unavailable('Stored prompt lengths are not engine executed prefill counters')
    result['actual_engine_generated_tokens']=unavailable('Training response/mask lengths include persisted chunks; engine decode counters are not exposed here')
    result['retained_work']=unavailable('No linked source/target engine work-counter observations at this hook')
    result['recomputed_work']=unavailable('No actual token replay/prefill counter observations at this hook')
    return result


class NativeRunObservation:
    def __init__(self, output, *, control, frozen):
        self.output=Path(output); self.output.mkdir(parents=True,exist_ok=False)
        self.control=control; self.frozen=frozen; self.events=[]; self.dispatches={};self.input=None
        self.first_step=control['resume_step']+1; self.before=None; self.after=None;self.parameter_delta=None
        self.initial_publication=None;self.updated_publication=None;self.update_evidence=None
        self.first_exception=None;self.errors=[];self.installed_rank_processes=None;self.request_sources=[]

    def event(self, name, **fields):
        row={'schema_version':1,'event':name,'utc_unix_s':time.time(),'monotonic_ns':time.monotonic_ns(),
             'pid':os.getpid(),'socket_hostname':socket.gethostname(),
             'source_sha':self.control.get('runtime_source_sha256'),
             'stage_id':Path(self.control['output']).name,**fields}
        self.events.append(row)
        try:
            with (self.output/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
        except Exception as error:
            # Evidence I/O failure is reported; it cannot repeat or undo training.
            self.errors.append({'event':name,'error_type':type(error).__name__})
        return row

    @contextmanager
    def stage(self, name, *, source=None, **fields):
        sid=uuid.uuid4().hex;start=time.monotonic_ns()
        self.event('stage_begin',stage=name,span_id=sid,source=source,**fields)
        try:yield
        except BaseException as error:
            receipt={'stage':name,'span_id':sid,'error_type':type(error).__name__,
                     'error_message_sha256':fingerprint(str(error)),'elapsed_ns':time.monotonic_ns()-start}
            if self.first_exception is None:self.first_exception=receipt
            self.event('stage_exception',**receipt);raise
        else:self.event('stage_end',stage=name,span_id=sid,elapsed_ns=time.monotonic_ns()-start)

    def native_event(self, name, fields):
        self.event(name,source_event='frozen_native_audit',**fields)
        if getattr(self,'collector',None):self.collector.event(name,**fields)
        if name=='native_checkpoint_prefetch_restored':
            # Already matched to the actual native TQ restore by the short helper.
            # These are historical prompt dispatches; no cross-process clock subtraction.
            for uid, proof in fields['prompts'].items():
                if proof.get('origin') != 'actual_native_checkpoint_prefetch_not_new_process_dispatch':
                    raise ValueError('Unknown restored prompt provenance')
                self.dispatches[uid] = proof
        if name=='original_trainer_initialized':
            self.initial_publication=fields['publication']
            self.installed_rank_processes=[x.get('_observer',{}).get('process') for x in fields['observations']]
        if fields.get('trainer_step')==self.first_step:
            if name=='native_actor_update_verified':self.update_evidence=fields
            if name=='native_policy_publication_verified':
                self.updated_publication=fields
                self.write_first_update()

    def write_first_update(self):
        initial=(self.initial_publication or {}).get('actual_parameter_version')
        updated=(self.updated_publication or {}).get('actual_parameter_version')
        groups=self.input or {}
        group_checks=groups.get('status')=='OBSERVED_ACTUAL_GROUP_METADATA' and groups.get('complete_groups')==8
        for group in groups.get('groups',[]):
            group_checks=group_checks and group['matches_actual_dispatch_prompt']
            for member in group['members']:
                segments=member['behavior_segments']; tag=member['original_tq_tag']
                group_checks=group_checks and member['behavior_logprob_aligned_finite'] and bool(segments)
                group_checks=group_checks and all(s['policy_version'] is not None and s['sampling_config_fingerprint'] for s in segments or [])
                group_checks=group_checks and tag is not None and tag.get('status')=='success' and not tag.get('is_padding')
        update=self.update_evidence or {};gradient=update.get('gradient_norm')
        learning=update.get('learning_signal_verified') is True and isinstance(gradient,(int,float)) and math.isfinite(gradient) and gradient>0
        per_rank=[]
        for row in update.get('observations',[]):
            returns=row.get('_observer',{}).get('native_engine_step_returns')
            current=[x for x in returns or [] if x.get('trainer_step')==self.first_step]
            per_rank.append({'rank':row.get('rank'),'returns':current if returns is not None else None,
                             'reason':None if returns is not None else 'Native engine return observation missing'})
        rank_gradient=(sorted(x['rank'] for x in per_rank if type(x['rank']) is int)==list(range(4))
                       and len(per_rank)==4 and all(len(x['returns'] or [])==1 and x['returns'][0].get('finite') is True
                       and x['returns'][0].get('nonzero') is True for x in per_rank))
        delta=(self.parameter_delta or {}).get('observed_parameter_change') is True
        requests=[x for x in self.request_sources if x['requested_version']==self.first_step]
        passed=bool(group_checks and learning and rank_gradient and delta and initial==self.control['resume_step'] and updated==self.first_step
                    and update.get('rank_count')==4 and requests and not self.errors)
        end=self.updated_publication and next((e for e in reversed(self.events) if e['event']=='native_policy_publication_verified' and e.get('trainer_step')==self.first_step),None)
        starts=[g['dispatch']['monotonic_ns'] for g in groups.get('groups',[]) if g.get('dispatch') and type(g['dispatch'].get('monotonic_ns')) is int]
        elapsed=(end['monotonic_ns']-min(starts))/1e9 if end and starts else None
        report={'status':'PASS_REQUIRED_FIRST_UPDATE_OBSERVATIONS' if passed else 'INCOMPLETE_OR_FAILED_REQUIRED_FIRST_UPDATE_OBSERVATIONS',
            'actual_first_step':self.first_step,'original_trainer_fit_replaced':False,
            'original_group_metadata':self.input,'original_learning_signal':update.get('learning_signal_verified'),
            'gradient_norm':gradient,'gradient_scope':'Original actor/grad_norm reduced metric, separate from each native FSDP engine returned norm',
            'per_rank_native_engine_gradient_returns':per_rank,'all_rank_finite_nonzero_native_returns':rank_gradient,
            'all_rank_native_optimizer_proof':{k:update.get(k) for k in ('rank_count','verified_updates_per_rank','observations')},
            'same_coordinate_parameter_sample_delta':self.parameter_delta,
            'full_weight_equality_or_digest_proven':False,'engine_version_before':initial,'engine_version_after':updated,
            'actual_policy_publication':self.updated_publication,'actual_update_request_sources':requests,
            'first_legal_group_dispatch_to_publication_seconds':elapsed,
            'automatic_retry_by_observer':False,'failed_attempts_outside_this_process':unavailable('No linked prior-attempt lifetime receipts provided; cannot exclude them from cost'),
            'trainer_gpu_hours':unavailable('Actual owned trainer allocation lifetime and physical UUID inventory not joined for the full window'),
            'rollout_gpu_hours':unavailable('Actual rollout rank/node/UUID allocation lifetime is not exposed by the existing fit/audit RPC'),
            'total_actual_gpu_hours':unavailable('Both role allocation histories, including failed attempts, are required'),
            'weight_segments':{k:unavailable('Only original inclusive publication call spans observed; native profile collector must supply component receipts') for k in
                ('weight_preparation','transfer','apply','cache_invalidation','readmission')},
            'first_exception':self.first_exception,'observer_io_errors':self.errors,'model_runtime_verified_by_cpu_tests':False}
        self.first_update=report
        try:(self.output/'first-update.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
        except Exception as error:self.errors.append({'event':'first-update.json','error_type':type(error).__name__})
        return report


def wrap_publication_calls(trainer, observer):
    """Observe actual calls, retaining exact async/sync calling convention."""
    changes=[]; manager=getattr(trainer,'standalone_checkpoint_manager',None)
    if manager is None:return changes
    targets=[(manager,'update_weights')]
    control=getattr(manager,'gateway_control',None)
    if control is not None:
        targets += [(control,n) for n in ('prepare_policy_publication','mark_policy_update_started',
            'confirm_policy_publication_before_admission','start_admission') if hasattr(control,n)]
    for obj,name in targets:
        original=getattr(obj,name);source=source_of(original)
        def call_info(args,kwargs,*,name=name,source=source):
            version=kwargs.get('global_steps')
            if version is None and args and type(args[0]) is int:version=args[0]
            if name=='update_weights':
                request={'requested_version':version,'source':source,'driver_pid':os.getpid(),
                         'driver_hostname':socket.gethostname(),'utc_unix_s':time.time()}
                observer.request_sources.append(request);observer.event('native_weight_update_request_observed',**request)
            return version
        if inspect.iscoroutinefunction(original):
            @functools.wraps(original)
            async def wrapped(*args,_original=original,_name=name,_source=source,_info=call_info,**kwargs):
                version=_info(args,kwargs)
                with observer.stage('native_publication.'+_name,source=_source,requested_version=version):
                    return await _original(*args,**kwargs)
        else:
            @functools.wraps(original)
            def wrapped(*args,_original=original,_name=name,_source=source,_info=call_info,**kwargs):
                version=_info(args,kwargs)
                with observer.stage('native_publication.'+_name,source=_source,requested_version=version):
                    return _original(*args,**kwargs)
        changes.append((obj,name,original));setattr(obj,name,wrapped)
    return changes


def observed_factory(original_factory, observer, expected_files, expected_repo):
    def factory(base, **kwargs):
        old_rpc=kwargs['rpc'];old_event=kwargs['event'];old_signal=kwargs.get('read_learning_signal')
        def rpc(trainer,function,*args):
            name=function.__name__
            options={'expected_files':expected_files,'expected_repo':expected_repo,
                     'sample_parameters':trainer.global_steps==observer.first_step}
            if name=='read_worker_update_audit':
                rows=old_rpc(trainer,read_update_with_observation,options)
                if options['sample_parameters']:
                    values=[x['_observer']['parameters'] for x in rows]
                    if observer.before is None:observer.before=values
                    elif observer.after is None:
                        observer.after=values
                        from run_v1_functional_gate import parameter_sample_delta
                        observer.parameter_delta=parameter_sample_delta(observer.before,observer.after)
                        observer.event('first_update_parameter_delta',trainer_step=observer.first_step,
                                       original_parameter_sample_delta=observer.parameter_delta)
                return rows
            if name=='install_worker_update_audit':
                return old_rpc(trainer,install_audit_with_observation,*args,options)
            return old_rpc(trainer,function,*args)
        def event(name,**fields):
            old_event(name,**fields);observer.native_event(name,fields)
        def signal(trainer,batch):
            value=old_signal(trainer,batch)
            if trainer.global_steps==observer.first_step:
                observer.input=input_observation(batch,observer.dispatches)
                observer.event('training_group_admission_observed',trainer_step=trainer.global_steps,
                               metadata=observer.input,original_learning_signal=value)
            return value
        kwargs.update(rpc=rpc,event=event)
        old_publication=kwargs['read_publication']
        def read_publication(trainer):
            value=old_publication(trainer)
            profile=getattr(getattr(trainer,'standalone_checkpoint_manager',None),'last_weight_update_profile',None)
            if profile is not None:observer.event('native_weight_profile_snapshot',profile=profile)
            return value
        kwargs['read_publication']=read_publication
        if old_signal is not None:kwargs['read_learning_signal']=signal
        audited=original_factory(base,**kwargs)
        methods={}
        for name in ('init','_submit_batch_to_rollout','_compute_old_log_prob','_compute_ref_log_prob',
                     '_compute_advantage','_update_actor','on_step_end','_save_checkpoint'):
            original=getattr(audited,name,None)
            if original is None:continue
            @functools.wraps(original)
            def wrapped(self,*args,_original=original,_name=name,**kw):
                with observer.stage('trainer.'+_name,source=source_of(_original),trainer_step=getattr(self,'global_steps',None)):
                    if _name=='_submit_batch_to_rollout':
                        batch=args[0]
                        try:
                            uids=plain(batch['uid']);prompts=plain(batch['raw_prompt'])
                            for uid,prompt in zip(uids,prompts,strict=True):
                                row=observer.event('group_dispatch_observed',uid=uid,expected_members=8,
                                    initial_raw_prompt_sha256=fingerprint(prompt),trainer_step=self.global_steps)
                                observer.dispatches[uid]={k:row[k] for k in ('monotonic_ns','utc_unix_s','initial_raw_prompt_sha256','trainer_step')}
                        except Exception as error:
                            observer.event('group_dispatch_metadata_missing',error_type=type(error).__name__)
                    result=_original(self,*args,**kw)
                    if _name=='init':
                        self._observer_publication_changes=wrap_publication_calls(self,observer)
                    return result
            methods[name]=wrapped
        return type('ObservedNativeTrainer',(audited,),methods)
    return factory


def observed_run_original_fit(config,frozen,control):
    """Ray driver entry delegates the unchanged original fit exactly once."""
    import run_native_training as native
    expected={str(Path(x['path']).resolve()):x['sha256'] for x in frozen['files']}
    manifest=json.loads(Path(control['observer_source_manifest']).read_text())
    expected.update({str(Path(control['observer_repo']).resolve()/n):h for n,h in manifest['files'].items()})
    out=Path(control['output']);observer=NativeRunObservation(out.parent/(out.name+'-observation'),control=control,frozen=frozen)
    from weight_metrics_collect import WeightMetricsCollector
    observer.collector=WeightMetricsCollector(observer.output/'weight-metrics',run_id=out.name).install()
    import verl
    observer.event('driver_source_observed',process=process_observation(expected_files=expected,expected_repo=control['observer_repo']))
    original=native.make_audited_trainer
    if getattr(native, "_runtime_first_update_reader", None) is not None:
        raise RuntimeError("A native first-update producer is already bound in this process")
    native.make_audited_trainer=observed_factory(original,observer,expected,control['observer_repo'])
    report=None
    first_update_readback = {}
    def read_first_update():
        if "value" not in first_update_readback:
            first_update_readback["value"] = observer.write_first_update()
        return first_update_readback["value"]
    native._runtime_first_update_reader = read_first_update
    try:
        with observe_initial_parameter_publication(observer), observer.stage('original_native_fit',source=source_of(native.run_original_fit)):
            report=native.run_original_fit(config,frozen,control)
        report['first_update_observation']=read_first_update()['status']
        report['runtime_observer_directory']=str(observer.output)
        return report
    finally:
        del native._runtime_first_update_reader
        native.make_audited_trainer=original
        observer.collector.close()
        first=read_first_update()
        summary={'status':'OBSERVATION_COMPLETED' if report is not None else 'OBSERVATION_FAILED',
                 'native_result_status':report.get('status') if report else None,
                 'first_update_status':first['status'],'event_count':len(observer.events),
                 'first_exception':observer.first_exception,'observer_io_errors':observer.errors,
                 'algorithm_replaced':False,'new_collectives_or_probe_jobs':0,'automatic_retries':0,
                 'actual_gpu_hours':first['total_actual_gpu_hours']}
        try:(observer.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
        except Exception as error:observer.errors.append({'event':'summary.json','error_type':type(error).__name__})
