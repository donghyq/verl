"""Load the pinned original business manager without replacing community code."""
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

from business_rollout_contract import RolloutContract, legacy_driver_contract


@dataclass
class OriginalBusinessRuntime:
    manager: object
    initial_tool_names: list[str]
    receipt: dict
    function_schemas: dict
    api_parameter_schemas: dict


def _sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def extract_original_prompt_tool_schemas(text):
    decoder = json.JSONDecoder()
    found = {}
    for index, value in enumerate(text):
        if value != '{':
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get('name'), str) and isinstance(obj.get('parameters'), dict):
            if obj['name'] in found and found[obj['name']] != obj:
                raise ValueError('conflicting original prompt tool declarations')
            found[obj['name']] = obj
    if not found:
        raise ValueError('no complete original prompt tool declarations')
    return found


def _module(name, filename):
    existing=sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve()!=Path(filename).resolve():
            raise RuntimeError('original business module would replace another implementation')
        return existing
    spec=importlib.util.spec_from_file_location(name,filename)
    if spec is None or spec.loader is None:
        raise ImportError('original business module cannot be loaded')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return module


def load_original_business_runtime(*,source_root,http_config_path,agent_config_path,auth_file,
        rollout_contract=None):
    if rollout_contract is None:
        rollout_contract=legacy_driver_contract()
    elif not isinstance(rollout_contract,RolloutContract):
        raise TypeError('rollout_contract must be a validated RolloutContract')
    root=Path(source_root)
    http_path=Path(http_config_path).resolve()
    agent_path=Path(agent_config_path).resolve()
    mapping_path=http_path.parent/'agent_config.json'
    mapping=json.loads(mapping_path.read_text())
    if (http_path.parent/mapping['agent_config_path']).resolve()!=agent_path:
        raise ValueError('the original mapping does not select the supplied agent config')
    auth=Path(auth_file)
    if auth.stat().st_mode & 0o077:
        raise ValueError('tool auth file must be private')
    token=auth.read_text().strip()
    if not token:
        raise ValueError('original tool authentication is empty')
    files={'http_manager':root/'verl/envs/tool_manager/http_manager.py',
           'base':root/'verl/envs/tool_manager/base.py','cat_logger':root/'verl/utils/cat_logger.py'}
    if not all(file.is_file() for file in files.values()):
        raise FileNotFoundError('pinned original business runtime is incomplete')
    # Original CatLogger already supports disabled monitoring / missing pycat.
    # Import its actual source; no fake monitoring class or HTTP manager is used.
    _module('verl.utils.cat_logger',files['cat_logger'])
    package='_original_project_b_business'
    if package not in sys.modules:
        module=types.ModuleType(package); module.__path__=[str(files['base'].parent)]
        sys.modules[package]=module
    _module(package+'.base',files['base'])
    backend=_module(package+'.http_manager',files['http_manager'])
    declared=ast.literal_eval(http_path.read_text())
    expected=[name for item in declared for name in item.get('httpTools',{}).get('tool_names',[])]
    manager=backend.HTTPToolManager({'config_path':str(http_path),'HTTP_TOOL_AUTH_TOKEN':token,
        **rollout_contract.manager_settings(),'cat':{'enable':False}})
    schemas=manager._load_agent_config_schemas(str(http_path))
    api_parameter_schemas={name:manager.get_tool(name).get_function_schema()['parameters'] for name in manager.list_tools()}
    built={schema['name'] for schema in manager.functions}
    if set(expected)-built:
        raise RuntimeError('original manager silently failed to build a configured tool')
    agent=json.loads(agent_path.read_text())
    disclosed=[item['tool_id'] for item in agent.get('tools',[]) if item.get('disclosed') is True]
    initial=[schema.get('function',schema)['name'] for schema in manager.functions]
    if not initial or len(set(initial)) != len(initial) or set(initial)-set(schemas):
        raise ValueError('the actual original manager tool set has no unique complete schemas')
    receipt={'original_runtime_modules':{name:_sha(file) for name,file in files.items()},
        'http_config_sha256':_sha(http_path),'agent_config_sha256':_sha(agent_path),
        'mapping_sha256':_sha(mapping_path),'manager_class':manager.__class__.__name__,
        'configured_tools':expected,'built_http_tools':manager.list_tools(),
        'initial_tool_names':initial,'initial_tool_source':'original_manager.functions',
        'agent_config_disclosed_tool_names':disclosed,'declared_function_names':list(schemas),
        'credentials_in_receipt':False,'monitoring':'original optional monitoring disabled',
        'actual_manager_runtime_parameters':{**rollout_contract.manager_settings(),'cat_enable':False},
        'rollout_contract':rollout_contract.receipt(),
        'validation_scope':'original business implementation and configuration; bounded driver parameters are explicit, not a production launch replay',
        'actual_backend_parameter_schema_sha256':hashlib.sha256(json.dumps(api_parameter_schemas,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    return OriginalBusinessRuntime(manager,initial,receipt,schemas,api_parameter_schemas)
