"""NOT_RUNTIME_VALIDATED: mount-time business input freeze and stock V1 fit adapter.

No random/synthetic builder is called. No training implementation is replaced.
Preparation reads files and Parquet metadata/rows, without model/Ray/tool calls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = '/mnt/beegfs-training4/hdd_pool/docker/user/hadoop-aipnlp/.friday/inference/zml_qwen3_5_35b_a3b/1/zml_qwen3_5_35b_a3b/1'
PENDING = '__READ_AFTER_CHECKPOINT_INIT__'
REQUIRED_ROLES = {'tools_config', 'tool_implementation', 'reward_implementation', 'business_loop_implementation'}
PRECOMPUTED_MANIFEST_SHA256 = '04923044b79b5efd390c07ed6a1541c4e7eb2a6800be4eb31dea361761d57587'
PRECOMPUTED_MODEL_SOURCE = '/mnt/shared/project-b'
PRECOMPUTED_MODEL_ID = 115995
PRECOMPUTED_MODEL_VERSION = 11886435


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def frozen_file(role, path):
    p = Path(path).resolve(strict=True)
    if not p.is_file():
        raise ValueError('Not a regular frozen input: ' + role)
    return {'role': role, 'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def load_native(entry_dir):
    """Pin actual reused adapters; imports here do not import Torch or Ray."""
    entry = Path(entry_dir).resolve()
    implementation = entry.parent
    pins = json.loads((HERE / 'native-dependency-pins.json').read_text())
    for name, expected in pins.items():
        if sha(implementation / name) != expected:
            raise ValueError('Reused native adapter bytes differ: ' + name)
    sys.path[:0] = [str(entry), str(implementation / 'update-audit')]
    result = []
    for name in ('prepare_native_training', 'run_native_training'):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != entry / (name + '.py'):
            raise ValueError('Foreign native adapter namespace: ' + name)
        result.append(module)
    return tuple(result)


def preserved_business_fields(config):
    """Exact fields retained from the owner-provided native business configuration."""
    a = config['actor_rollout_ref']; r = a['rollout']
    return copy.deepcopy({'algorithm': config['algorithm'], 'reward': config['reward'],
        'multi_turn': r['multi_turn'], 'agent': {k:v for k,v in r['agent'].items() if k != 'num_workers'},
        'sampling': {k:r.get(k) for k in ('temperature','top_p','top_k','do_sample')},
        'model_options': {k:v for k,v in a['model'].items() if k != 'path'},
        'data': {k:v for k,v in config['data'].items() if k not in ('train_batch_size','gen_batch_size')}})


def validate_business_base(config):
    """Raw FullyAsync/Megatron recipes require business V1 integration first."""
    trainer = config['trainer']; a = config['actor_rollout_ref']; r = a['rollout']
    if not trainer.get('use_v1') or trainer.get('v1', {}).get('trainer_mode') != 'separate_async':
        raise ValueError('Need a business-connected stock V1 separate_async base; raw FullyAsync recipe is not this interface')
    if a['actor'].get('strategy') != 'fsdp2':
        raise ValueError('This reused optimizer/parameter adapter is scoped to the FSDP2 native candidate; Megatron needs its own verified audit interface')
    if config['algorithm'].get('adv_estimator') != 'grpo' or r.get('n') != 8:
        raise ValueError('Native audit requires GRPO with eight real members per prompt')
    if r.get('name') != 'sglang' or r.get('mode') != 'async' or not r.get('calculate_log_probs'):
        raise ValueError('Need original asynchronous SGLang behavior log probabilities')
    if not r['multi_turn'].get('enable'):
        raise ValueError('Original business multi-turn loop must remain enabled')
    if not config['transfer_queue'].get('enable'):
        raise ValueError('Native V1 TransferQueue must be enabled')
    if r['agent'].get('default_agent_loop') in ('tool_agent', 'durable_tool_agent'):
        raise ValueError('Generic synthetic tool loop is not the original HTTPToolManager business integration')
    if r.get('trajectory_state', {}).get('acceptance_fixture', {}).get('enabled'):
        raise ValueError('Controlled acceptance fixture must not replace business trajectories')


def transform_config(base, *, native, model, target, checkpoint_root, durable_root, run_id, gateway_url):
    validate_business_base(base)
    parsed = urlsplit(gateway_url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.hostname.endswith('.invalid'):
        raise ValueError('Actual native Gateway URL required; URL syntax is not health evidence')
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('Unique plain run ID required')
    original = preserved_business_fields(base)
    config = native.prepare_config(base, target=target, checkpoint_root=checkpoint_root,
        durable_root=durable_root, run_id=run_id, entry_dir=HERE, audit_dir=Path(native.__file__).parent.parent/'update-audit')
    config['actor_rollout_ref']['model']['path'] = str(Path(model).resolve())
    config['actor_rollout_ref']['rollout']['sglang_gateway']['url'] = gateway_url
    if preserved_business_fields(config) != original:
        raise ValueError('Native preparation changed original business semantics')
    return config


def inspect_data(role, path):
    import pyarrow.parquet as pq
    p = Path(path).resolve(strict=True); parquet = pq.ParquetFile(p)
    # Ground truth remains the original reward's object; no Local-Life constraints injected.
    required = {'prompt', 'data_source', 'agent_name', 'reward_model', 'extra_info'}
    if not required <= set(parquet.schema_arrow.names) or parquet.metadata.num_rows == 0:
        raise ValueError('Original business Parquet lacks native reward/loop inputs: ' + role)
    sources = set(); agents = set(); checked = 0
    for batch in parquet.iter_batches(columns=sorted(required), batch_size=256):
        for row in batch.to_pylist():
            if not row['data_source'] or str(row['data_source']).startswith('synthetic'):
                raise ValueError('Synthetic or missing business data source rejected')
            if not row['agent_name'] or row['agent_name'] in ('tool_agent','durable_tool_agent'):
                raise ValueError('Row must select the original connected business loop')
            if not isinstance(row['prompt'], list) or not row['prompt'] or not isinstance(row['reward_model'], dict):
                raise ValueError('Incomplete original prompt/reward row')
            if any(not isinstance(m, dict) or not m.get('role') or not isinstance(m.get('content'), str) for m in row['prompt']):
                raise ValueError('Native business prompt schema rejected')
            sources.add(row['data_source']); agents.add(row['agent_name']); checked += 1
    return {**frozen_file(role,p),'rows':checked,'schema_sha256':hashlib.sha256(parquet.schema_arrow.serialize().to_pybytes()).hexdigest(),
            'data_sources':sorted(sources),'agent_names':sorted(agents),'original_business_reward_semantics_verified':False}


def validate_model_family(config):
    """This candidate covers the requested composite MoE, not a guessed route."""
    if config.get('model_type') == 'qwen3_5_moe_text':
        raise ValueError('Standalone Qwen3.5 MoE text outer configuration has an unverified native training route')
    if config.get('model_type') != 'qwen3_5_moe':
        raise ValueError('Mounted configuration is not the requested Qwen3.5 composite MoE family')
    text = config.get('text_config')
    if not isinstance(text, dict) or text.get('model_type') != 'qwen3_5_moe_text':
        raise ValueError('Actual nested text_config must be the Qwen3.5 qwen3_5_moe_text family')


def precomputed_model_manifest(root, manifest_path, manifest_sha256):
    """Opt-in reuse of a pinned, independently read-back receipt; no weight reread."""
    if not manifest_path or manifest_sha256 != PRECOMPUTED_MANIFEST_SHA256:
        raise ValueError('Explicit pinned precomputed model manifest and SHA required')
    if sha(manifest_path) != manifest_sha256:
        raise ValueError('Precomputed model manifest bytes differ')
    m = json.loads(Path(manifest_path).read_text())
    original = m.get('original_committed_manifest', {})
    for receipt in (m, original):
        if (receipt.get('status') != 'COMMITTED' or receipt.get('model_id') != PRECOMPUTED_MODEL_ID
                or receipt.get('model_version_id') != PRECOMPUTED_MODEL_VERSION):
            raise ValueError('Precomputed receipt model/version or committed status differs')
    if (m.get('independent_local_readback') is not True
            or original.get('shared_hdd_path_write_readback_pass') is not True
            or original.get('source') != PRECOMPUTED_MODEL_SOURCE
            or m.get('source') != original.get('target')):
        raise ValueError('Precomputed receipt lacks the complete independent readback chain')
    # Only the original model alias and the two verified copy destinations qualify.
    allowed = (original['source'], original['target'], m['target'])
    if Path(root).resolve(strict=True) not in {Path(x).resolve() for x in allowed}:
        raise ValueError('Current model path is outside the pinned verified copy chain')
    by_name = {}
    for receipt in (original, m):
        rows = receipt.get('files', [])
        records = {}
        for row in rows:
            name = row.get('name', '')
            digest = row.get('source_stream_sha256', '')
            if (not name or Path(name).name != name or name in records
                    or type(row.get('bytes')) is not int or row['bytes'] <= 0
                    or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                    or row.get('target_readback_sha256') != digest):
                raise ValueError('Incomplete verified file record in precomputed model receipt')
            records[name] = {'bytes': row['bytes'], 'sha256': digest}
        total = sum(x['bytes'] for x in records.values())
        if (len(records) != 24 or total != receipt.get('source_total_bytes')
                or total != receipt.get('total_bytes_copied_verified')):
            raise ValueError('Precomputed receipt does not verify every original file')
        if by_name and records != by_name:
            raise ValueError('Precomputed source/copy file identities differ')
        by_name = records
    shards = {n for n in by_name if n.endswith('.safetensors')}
    if len(shards) != 14 or original.get('source_shard_count') != 14:
        raise ValueError('Precomputed receipt must contain all fourteen weight shards')
    return by_name


def model_verification_provenance(manifest_path, manifest_sha256):
    if manifest_path:
        return {'provenance': 'precomputed_verified_manifest',
                'manifest_path': str(Path(manifest_path).resolve(strict=True)),
                'manifest_sha256': manifest_sha256, 'model_id': PRECOMPUTED_MODEL_ID,
                'model_version_id': PRECOMPUTED_MODEL_VERSION,
                'weight_content_read_by_this_process': False}
    if manifest_sha256:
        raise ValueError('Precomputed manifest path and SHA must be provided together')
    return {'provenance': 'full_content_sha256', 'weight_content_read_by_this_process': True}


def model_identity(root, *, precomputed_manifest=None, precomputed_manifest_sha256=None):
    root = Path(root).resolve(strict=True)
    model_verification_provenance(precomputed_manifest, precomputed_manifest_sha256)
    verified = (precomputed_model_manifest(root, precomputed_manifest, precomputed_manifest_sha256)
                if precomputed_manifest else None)
    config = json.loads((root/'config.json').read_text())
    validate_model_family(config)
    index = root/'model.safetensors.index.json'
    if not index.is_file():
        raise ValueError('Complete mounted safetensors index required')
    mapping = json.loads(index.read_text()).get('weight_map')
    if not isinstance(mapping,dict) or not mapping:
        raise ValueError('Empty mounted checkpoint weight index')
    names = sorted(set(mapping.values()))
    if any(Path(n).name != n or not n.endswith('.safetensors') for n in names):
        raise ValueError('Unsafe checkpoint shard name')
    weights = [frozen_file('model_config',root/'config.json'),frozen_file('model_index',index)]
    if verified is None:
        weights += [frozen_file('model_weight_'+str(i),root/n) for i,n in enumerate(names)]
    else:
        if set(names) != {n for n in verified if n.endswith('.safetensors')}:
            raise ValueError('Current model index differs from fourteen verified shards')
        for row, name in zip(weights, ('config.json', 'model.safetensors.index.json')):
            if {k: row[k] for k in ('bytes', 'sha256')} != verified.get(name):
                raise ValueError('Current model configuration/index bytes differ from verified receipt')
        for i, name in enumerate(names):
            path = root/name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != verified[name]['bytes']:
                raise ValueError('Current model shard missing, redirected or wrong size: ' + name)
            weights.append({'role': 'model_weight_'+str(i), 'path': str(path), **verified[name],
                            'digest_provenance': 'precomputed_verified_manifest',
                            'manifest_sha256': precomputed_manifest_sha256})
    tokenizer_names = ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','chat_template.jinja')
    tokens = [frozen_file('tokenizer_'+str(i),root/n) for i,n in enumerate(tokenizer_names) if (root/n).is_file()]
    if not (root/'tokenizer_config.json').is_file() or not any((root/n).is_file() for n in ('tokenizer.json','vocab.json')):
        raise ValueError('Actual tokenizer and template configuration required')
    return weights+tokens, canonical([{k:x[k] for k in ('role','sha256')} for x in weights]), canonical([{k:x[k] for k in ('role','sha256')} for x in tokens]), config


def verify_freeze(config_path, manifest_path, config=None, *, precomputed_manifest=None, precomputed_manifest_sha256=None):
    m = json.loads(Path(manifest_path).read_text())
    if m.get('schema_version') != 35 or m.get('data_kind') != 'production/business' or sha(config_path) != m.get('config_sha256'):
        raise ValueError('Actual Qwen35 business freeze required')
    roles = [x['role'] for x in m['files']]
    if len(roles) != len(set(roles)) or not REQUIRED_ROLES.union({'train_data','validation_data','runtime_source_manifest','model_config','model_index'}) <= set(roles):
        raise ValueError('Missing/duplicate frozen original assets')
    cached_weights = {}
    provenance = model_verification_provenance(precomputed_manifest, precomputed_manifest_sha256)
    if precomputed_manifest:
        if m.get('model_identity_verification') != provenance:
            raise ValueError('Frozen precomputed model provenance differs')
        current, weights, tokenizers, _ = model_identity(m['model_path'],
            precomputed_manifest=precomputed_manifest, precomputed_manifest_sha256=precomputed_manifest_sha256)
        cached_weights = {x['role']: x for x in current if x['role'].startswith('model_weight_')}
        if ({x['role'] for x in m['files'] if x['role'].startswith('model_weight_')} != set(cached_weights)
                or m['policy_identity']['model_weights_fingerprint'] != weights
                or m['policy_identity']['tokenizer_template_fingerprint'] != tokenizers):
            raise ValueError('Frozen model/tokenizer identity differs from pinned model receipt')
    for x in m['files']:
        if x['role'] in cached_weights:
            if x != cached_weights[x['role']]:
                raise ValueError('Frozen shard record differs from pinned model receipt: '+x['role'])
            continue
        if sha(x['path']) != x['sha256']:
            raise ValueError('Frozen input bytes differ: '+x['role'])
    if m.get('parameter_version_binding') != 'checkpoint_init_native_gateway_read' or m['policy_identity']['policy_version'] != PENDING:
        raise ValueError('Actual initialized version must come from the original Gateway readback')
    if config is not None:
        a=config['actor_rollout_ref']; r=a['rollout']; by_role={x['role']:x['path'] for x in m['files']}
        if a['model']['path'] != m['model_path'] or r['trajectory_state']['policy_identity'] != m['policy_identity']:
            raise ValueError('Prepared policy/model identity differs')
        if Path(r['multi_turn']['tool_config_path']).resolve() != Path(by_role['tools_config']) or Path(config['reward']['custom_reward_function']['path']).resolve() != Path(by_role['reward_implementation']):
            raise ValueError('Original tool/reward file binding differs')
        for role,key in (('train_data','train_files'),('validation_data','val_files')):
            values=config['data'][key]; values=[values] if isinstance(values,str) else list(values)
            if [str(Path(x).resolve()) for x in values] != [by_role[role]]:
                raise ValueError('Original business dataset binding differs')
    return m


def runtime_package_inventory():
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    source_packages = {}
    for name in ('torch', 'transformers', 'sglang', 'smg', 'transferqueue', 'tensordict'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            if name != 'sglang':
                raise
            import sglang
            module_path = Path(sglang.__file__).resolve()
            module_version = getattr(sglang, '__version__', None)
            if not module_path.is_file() or not isinstance(module_version, str) or not module_version:
                raise RuntimeError('Source SGLang requires its real module file and version')
            root = module_path.parent
            entries = [
                {'path': str(p.relative_to(root)), 'sha256': sha(p)}
                for p in sorted(root.rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')
            ]
            if not entries:
                raise RuntimeError('Source SGLang tree is empty')
            packages[name] = module_version
            source_packages[name] = {
                'identity_kind': 'source_tree', 'distribution_version': None,
                'module_version': module_version, 'module_path': str(module_path),
                'source_root': str(root), 'source_tree_sha256': canonical(entries),
                'source_files': entries,
            }
    return packages, source_packages


def prepare(args):
    if sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Mount-time preparation is Linux CPU-only with empty CUDA visibility')
    from omegaconf import OmegaConf
    native, runner = load_native(args.native_entry_dir)
    base=OmegaConf.to_container(OmegaConf.load(args.business_config),resolve=True)
    runner.verify_runtime_source(args.repo,args.runtime_source_manifest,args.runtime_source_sha256)
    env=base['ray_kwargs']['ray_init']['runtime_env']['env_vars']
    env['PYTHONPATH']=os.pathsep.join(dict.fromkeys([str(Path(args.repo).resolve())]+env['PYTHONPATH'].split(os.pathsep)))
    config=transform_config(base,native=native,model=args.model,target=args.target,checkpoint_root=args.checkpoint_root,
        durable_root=args.durable_root,run_id=args.run_id,gateway_url=args.gateway_url)
    assets=json.loads(Path(args.business_assets).read_text()); rows=assets['files']
    if not REQUIRED_ROLES <= {x['role'] for x in rows}:
        raise ValueError('Original HTTP tool, reward and business-loop sources must be supplied')
    if len({x['role'] for x in rows}) != len(rows):
        raise ValueError('Duplicate original asset roles')
    files=[frozen_file(x['role'],x['path']) for x in rows]
    train_files=config['data']['train_files']; val_files=config['data']['val_files']
    train_files=[train_files] if isinstance(train_files,str) else list(train_files)
    val_files=[val_files] if isinstance(val_files,str) else list(val_files)
    if len(train_files)!=1 or len(val_files)!=1 or Path(train_files[0]).resolve()==Path(val_files[0]).resolve():
        raise ValueError('Distinct original train/test Parquet pair required')
    train=inspect_data('train_data',train_files[0]); test=inspect_data('validation_data',val_files[0])
    if train['rows']<8:
        raise ValueError('Eight prompt groups need at least eight existing training rows')
    if set(train['agent_names']+test['agent_names']) != {config['actor_rollout_ref']['rollout']['agent']['default_agent_loop']}:
        raise ValueError('Dataset loop selection differs from the connected native business loop')
    model_options = {'precomputed_manifest': args.precomputed_model_manifest,
                     'precomputed_manifest_sha256': args.precomputed_model_manifest_sha256}
    model_files,weights,tokenizers,hfconfig=model_identity(args.model, **model_options)
    if args.precomputed_model_manifest:
        files.append(frozen_file('precomputed_model_manifest', args.precomputed_model_manifest))
    files += [train,test,*model_files,frozen_file('runtime_source_manifest',args.runtime_source_manifest),
              frozen_file('business_base_config',args.business_config),frozen_file('business_assets_descriptor',args.business_assets)]
    for name in json.loads((HERE/'native-dependency-pins.json').read_text()):
        files.append(frozen_file('native_adapter_'+name.replace('/','_'),Path(args.native_entry_dir).parent/name))
    # The fit adapter imports these original observation helpers at runtime.
    functional=Path(args.repo)/'project-b-final-validation-20260928/production-entry-20261008'
    files.append(frozen_file('native_parameter_observer',functional/'run_v1_functional_gate.py'))
    files += [frozen_file('qwen35_adapter',__file__),frozen_file('native_adapter_pins',HERE/'native-dependency-pins.json')]
    for filename,expected in json.loads((HERE/'observer-dependency-pins.json').read_text()).items():
        if sha(HERE/filename)!=expected:raise ValueError('Observer dependency bytes differ: '+filename)
        files.append(frozen_file('observer_'+filename.replace('.','_'),HERE/filename))
    files.append(frozen_file('observer_pins',HERE/'observer-dependency-pins.json'))
    packages, source_packages = runtime_package_inventory()
    for package, record in source_packages.items():
        for row in record['source_files']:
            files.append(frozen_file('runtime_source_'+package+'_'+canonical(row['path']), Path(record['source_root'])/row['path']))
    kv={'mode':'token_only','kv_ref':None,'kv_abi_verified':False,'tp':4,'dtype':config['actor_rollout_ref']['rollout']['dtype'],'runtime':packages}
    if source_packages:
        kv['runtime_source_digests'] = {name: record['source_tree_sha256'] for name, record in source_packages.items()}
    identity={'policy_version':PENDING,'model_weights_fingerprint':weights,'tokenizer_template_fingerprint':tokenizers,'kv_cache_schema_fingerprint':canonical(kv)}
    state=config['actor_rollout_ref']['rollout']['trajectory_state']; state['policy_identity']=identity
    state['model_configuration_fingerprint']=sha(Path(args.model)/'config.json')
    state['environment_version']=canonical([{'role':x['role'],'sha256':x['sha256']} for x in files if x['role'] in ('train_data','validation_data','reward_implementation','business_loop_implementation')])
    state['tool_version']=canonical([{'role':x['role'],'sha256':x['sha256']} for x in files if x['role'] in ('tools_config','tool_implementation')])
    environment=config['ray_kwargs']['ray_init']['runtime_env']['env_vars']
    # cloudpickle imports the reused driver module on remote workers. CWD alone
    # cannot provide run_native_training/native_training_audit namespaces.
    helper_parents=[str(Path(x['path']).parent) for x in files if x['role'] in
                    ('tool_implementation','reward_implementation','business_loop_implementation')]
    paths=[str(Path(args.repo).resolve()),str(functional.resolve()),str(Path(args.native_entry_dir).resolve()),
           str(Path(args.native_entry_dir).resolve().parent/'update-audit'),str(HERE),*helper_parents]
    environment['PYTHONPATH']=os.pathsep.join(dict.fromkeys(paths+environment['PYTHONPATH'].split(os.pathsep)))
    nccl_root=Path(args.checkpoint_root)/args.run_id/'nccl-info'/('target-'+str(args.target))
    nccl_root.mkdir(parents=True,exist_ok=True)
    environment['NCCL_DEBUG']='INFO'
    environment['NCCL_DEBUG_FILE']=str(nccl_root/'host-%h-pid-%p.log')
    runner.validate_config(config,target=args.target,resume_step=0 if args.target in (1,10) else 10)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    config_path=out/'frozen-business-native.yaml';OmegaConf.save(OmegaConf.create(config),config_path,resolve=True)
    manifest={'schema_version':35,'data_kind':'production/business','config_sha256':sha(config_path),'model_path':str(Path(args.model).resolve()),
        'model_identity_verification':model_verification_provenance(args.precomputed_model_manifest,args.precomputed_model_manifest_sha256),
        'parameter_identity_source':'actual_frozen_files_then_checkpoint_published_version' if not args.precomputed_model_manifest else 'precomputed_verified_manifest_then_checkpoint_published_version','parameter_version_binding':'checkpoint_init_native_gateway_read',
        'policy_identity':identity,'files':files,'data':{'train':train,'validation':test},'kv_fast_path':kv,'runtime_versions':packages,'runtime_package_sources':source_packages,
        'model_configuration_fingerprint':state['model_configuration_fingerprint'],'target_step':args.target,'native_resume_step':0 if args.target in (1,10) else 10,
        'experiment_scope':'original business tools/reward, bounded native updates; task benefit not yet measured',
        'production_task_trial_completed':False,'resource_profile':{'train_gpus':4,'standalone_generation_gpus':4,'capacity_verified':False,
            'actual_train_rank_node_gpu_uuid_mapping':None,'actual_rollout_rank_node_gpu_uuid_mapping':None,
            'rank_node_gpu_uuid_mapping_verified':False,'fixed_platform_worker_role_assignment':False},
        'mounted_hf_model_type':hfconfig.get('model_type'),'mounted_hf_architectures':hfconfig.get('architectures'),
        'qwen35_model_runtime_validated':False,'optimizer_updates_verified':0,'learning_signal_verified':False}
    manifest_path=out/'freeze.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    verify_freeze(config_path,manifest_path,config, **model_options)
    print(json.dumps({'status':'PREPARED_NOT_RUNTIME_VALIDATED','config':str(config_path),'freeze':str(manifest_path),'target':args.target,'models_loaded':0,'gpu_operations':0}))


def run(args):
    if not args.execute_native or sys.platform!='linux':
        raise RuntimeError('Explicit owned Linux native execution required')
    if args.target==100 and not args.prior_parameters:
        raise ValueError('Cold resume requires previous process actual parameter observations')
    from omegaconf import OmegaConf
    _, runner=load_native(args.native_entry_dir)
    repo=Path(args.repo).resolve(); sys.path[:0]=[str(repo),str(repo/'project-b-final-validation-20260928/production-entry-20261008'),str(HERE)]
    config=OmegaConf.load(args.config); plain=OmegaConf.to_container(config,resolve=True)
    frozen=verify_freeze(args.config,args.freeze_manifest,plain,
        precomputed_manifest=args.precomputed_model_manifest,
        precomputed_manifest_sha256=args.precomputed_model_manifest_sha256)
    source=[x for x in frozen['files'] if x['role']=='runtime_source_manifest'][0]
    runner.verify_runtime_source(repo,source['path'],args.runtime_source_sha256)
    resume=0 if args.target in (1,10) else 10
    if frozen['target_step']!=args.target or frozen['native_resume_step']!=resume:
        raise ValueError('Frozen phase differs')
    runner.validate_config(plain,target=args.target,resume_step=resume)
    import ray
    routing_spec = os.environ.get('ROUTING_EXPERIMENT_SPEC')
    routing_options = json.loads(Path(routing_spec).read_text()) if routing_spec else None
    runtime_env = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env,resolve=True)
    if routing_options:
        routing_options.update(native_config=str(Path(args.config).resolve()),
            native_freeze=str(Path(args.freeze_manifest).resolve()),
            owned_gateway_endpoint=os.environ['ROUTING_EXPERIMENT_GATEWAY_ENDPOINT'])
        runtime_env.setdefault('env_vars', {})['ROUTING_EXPERIMENT_OBSERVATION_DIR'] = routing_options['worker_observations']
    for observation_env in ('VERL_STORAGE_HANDOFF_CAMPAIGN_SPEC', 'VERL_STORAGE_HANDOFF_EVIDENCE_ROOT', 'VERL_STORAGE_HANDOFF_AT_STEP', 'VERL_WEIGHT_COMPONENT_PROFILE'):
        if observation_env in os.environ:
            runtime_env.setdefault('env_vars', {})[observation_env]=os.environ[observation_env]
    ray.init(address=os.environ['RAY_ADDRESS'],runtime_env=runtime_env)
    control={'output':str(Path(args.output).resolve()),'target':args.target,'resume_step':resume,
        'prior_parameters':str(Path(args.prior_parameters).resolve()) if args.prior_parameters else None,
        'runtime_source_sha256':args.runtime_source_sha256,'rpc_timeout_s':180,
        'observer_repo':str(repo),'observer_source_manifest':source['path']}
    if routing_options:
        control['routing_experiment'] = routing_options
    from runtime_observer import observed_run_original_fit
    result=ray.get(ray.remote(num_cpus=1)(observed_run_original_fit).remote(config,frozen,control))
    print(json.dumps({k:result[k] for k in ('status','verified_optimizer_updates_this_process','native_checkpoint')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');r=sub.add_parser('run')
    for s in (p,r):
        for flag in ('repo','native-entry-dir','output','runtime-source-sha256'):
            s.add_argument('--'+flag,required=True)
        s.add_argument('--target',type=int,choices=(1,10,100),required=True)
        s.add_argument('--precomputed-model-manifest', help='Explicit pinned COMMITTED receipt; skips repeated full weight reads only')
        s.add_argument('--precomputed-model-manifest-sha256', help='Must match the fixed independently verified manifest SHA')
    for flag in ('business-config','business-assets','checkpoint-root','durable-root','run-id','gateway-url'):
        p.add_argument('--'+flag,required=True)
    p.add_argument('--runtime-source-manifest','--source-manifest',dest='runtime_source_manifest',required=True)
    p.add_argument('--model',default=DEFAULT_MODEL)
    for flag in ('config','freeze-manifest'):
        r.add_argument('--'+flag,required=True)
    r.add_argument('--prior-parameters');r.add_argument('--execute-native',action='store_true')
    args=parser.parse_args();prepare(args) if args.action=='prepare' else run(args)


if __name__=='__main__':
    main()
