"""Bind real business assets to the actual unpacked SOURCE; no tool/model calls.

The output is a candidate, not a production reward/algorithm acceptance receipt.
The input structure YAML must separately undergo the native entry's validation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

AGENT_NAME = "qwen35_business_trajectory"
RECIPE_SHA = "fb6bfcb11131c342ec5b83de82303b4b5c6a3f85b29fcb9257d43575e518cdb7"
SOURCE_AGENT_NAMES = {AGENT_NAME, "rl_factory_tool_agent", "rlfactory_agent"}


def canonicalize_agent_name(source_file, output_file, validate_record, context_extractor):
    """Map only technical loop dispatch metadata from an actual Parquet.

    All other Arrow columns are passed through unchanged. Missing dispatch uses
    the exact original recipe's default rl_factory_tool_agent; business fields
    (including coordinates/session/reward) must already exist and validate.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    source = Path(source_file).resolve(strict=True)
    output = Path(output_file)
    if output.exists() or output.resolve() == source:
        raise ValueError("Derived data must be a new file; original data cannot be overwritten")
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    present = "agent_name" in schema.names
    index = schema.get_field_index("agent_name") if present else len(schema)
    if present and not (pa.types.is_string(schema.field(index).type) or pa.types.is_large_string(schema.field(index).type)):
        raise ValueError("Original agent dispatch metadata is not text")
    target_schema = schema if present else schema.append(pa.field("agent_name",pa.string()))
    writer = None; counts = {}; total = 0
    try:
        for batch in parquet.iter_batches(batch_size=256):
            rows = batch.to_pylist()
            for row in rows:
                original = row.get("agent_name", "rl_factory_tool_agent")
                if original not in SOURCE_AGENT_NAMES:
                    raise ValueError("Unrecognized original business loop name: " + str(original))
                checked = copy.deepcopy(row); checked["agent_name"] = AGENT_NAME
                validate_record(checked, context_extractor)
                counts[original] = counts.get(original,0) + 1
            table = pa.Table.from_batches([batch])
            values = pa.array([AGENT_NAME]*len(rows),type=target_schema.field(index).type)
            table = table.set_column(index,target_schema.field(index),values) if present else table.append_column(target_schema.field(index),values)
            # Arrow equality is stronger than reconstructing unaffected values.
            for field_index,name in enumerate(schema.names):
                if name != "agent_name" and not table.column(field_index).equals(pa.chunked_array([batch.column(field_index)])):
                    raise ValueError("Canonical dispatch changed an original business column: " + name)
            if writer is None:
                output.parent.mkdir(parents=True,exist_ok=True)
                writer = pq.ParquetWriter(output,target_schema)
            writer.write_table(table); total += len(rows)
        if not total:
            raise ValueError("Actual business Parquet is empty")
    finally:
        if writer is not None:
            writer.close()
    return {"source_path":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "derived_path":str(output.resolve()),"derived_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows":total,"source_agent_name_counts":counts,"agent_name_column_present":present,
        "missing_column_default_source":"Original recipe default_agent_loop=rl_factory_tool_agent",
        "mapped_to":AGENT_NAME,"other_arrow_columns_unchanged":True,
        "new_business_coordinates_or_identity_or_reward_fields":False}


def source_files(root):
    manifest = json.loads((root / "payload-source-manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = (root / row["relative_path"]).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Business payload source differs: " + row["role"])
        rows.append({"role": row["role"], "path": str(path), "sha256": row["sha256"]})
    return rows


def bind_config(base, *, source, output, train, val, auth, model, python):
    """Replace old fixture interfaces; preserve the supplied training algorithm."""
    for path in (source, output, train, val, auth, model, python):
        if not Path(path).is_absolute():
            raise ValueError("Startup must supply actual absolute paths")
    if train == val:
        raise ValueError("Distinct original train/test Parquet paths are required")
    root = Path(source) / "qwen35_business"
    adapter = root / "business_adapter.py"
    config = copy.deepcopy(base)
    a = config["actor_rollout_ref"]; r = a["rollout"]; d = config["data"]
    if not config["trainer"]["use_v1"] or a["actor"]["strategy"] != "fsdp2":
        raise ValueError("A complete stock V1/FSDP2 configuration is required")
    if config["algorithm"]["adv_estimator"] != "grpo":
        raise ValueError("The requested native candidate requires an explicit GRPO configuration")
    # Original GRPO advantages and GSPO policy objective coexist. These fields
    # have native 6ad interfaces; the Megatron strategy itself is not copied.
    config["algorithm"]["use_kl_in_reward"] = False
    config["algorithm"]["kl_ctrl"]["kl_coef"] = 0
    a["actor"].update(loss_agg_mode="seq-mean-token-mean", clip_ratio_low=0.2,
        clip_ratio_high=0.28, clip_ratio_c=10.0, use_kl_loss=True, kl_loss_coef=0.02)
    a["actor"]["policy_loss"]["loss_mode"] = "gspo"
    a["model"].update(path=model, hf_config_path=model, tokenizer_path=model, custom_chat_template=None)
    d.update(train_files=[train], val_files=[val], tokenizer=None,
             tool_config_path=str(Path(output) / "business-native-tools.json"), function_tool_path=None,
             max_prompt_length=12288, max_response_length=65536, shuffle=True, seed=42,
             apply_chat_template_kwargs={"thinking": True})
    r.update(name="sglang", mode="async", n=8, calculate_log_probs=True,
             temperature=0.7, top_p=0.75, prompt_length=12288, response_length=65536, max_model_len=77824)
    # First-step memory baseline: one sample per GPU microbatch, without
    # dynamic packing. Keep the complete original model-length budget recorded
    # even though the static split does not use the dynamic token assertion.
    a["actor"].update(use_dynamic_bsz=False, ppo_micro_batch_size_per_gpu=1,
                      ppo_max_token_len_per_gpu=r["max_model_len"])
    r.update(log_prob_use_dynamic_bsz=False, log_prob_micro_batch_size_per_gpu=1,
             log_prob_max_token_len_per_gpu=r["max_model_len"])
    a["ref"].update(log_prob_use_dynamic_bsz=False, log_prob_micro_batch_size_per_gpu=1,
                    log_prob_max_token_len_per_gpu=r["max_model_len"])
    r["multi_turn"].update(enable=True, format="business_function_xml", max_assistant_turns=6,
        max_user_turns=6, max_parallel_calls=1, max_tool_response_length=8192,
        tool_response_truncate_side="right", tool_config_path=d["tool_config_path"], function_tool_path=None)
    r["agent"].update(default_agent_loop=AGENT_NAME, agent_loop_config_path=str(Path(output) / "business-agent.yaml"))
    # Remove identities of the old random input. The native prepare step replaces
    # these only after hashing the real model, datasets and business assets.
    state = r["trajectory_state"]
    for key in ("policy_identity", "model_configuration_fingerprint", "environment_version", "tool_version"):
        state[key] = None
    state.update(root_dir=str(Path(output) / "uninitialized-durable-state"), run_id="uninitialized-business-input")
    if "acceptance_fixture" in state:
        state["acceptance_fixture"]["enabled"] = False
    config["trainer"].update(default_local_dir=str(Path(output) / "uninitialized-checkpoints"),
                            resume_mode="disable", resume_from_path=None)
    reward_config = json.loads((root / "original-config/reward-config.json").read_text())
    config["reward"]["custom_reward_function"] = {**reward_config,
        "path": str(adapter), "name": "compute_score", "reward_kwargs": {
            "business_contract_path": str(Path(output) / "business-contract.json")}}
    config["reward"]["reward_manager"].update(source="importlib", name="BusinessRewardManager",
        module={"_target_": "verl.trainer.config.config.ModuleConfig", "path": str(adapter), "name": "business_reward_manager"})
    # Exact old reward/reward.yaml has no reward_kwargs. Never inherit an
    # unrelated structure template's overlong penalty into the business score.
    config["reward"]["reward_model"].pop("reward_kwargs", None)
    env = config["ray_kwargs"]["ray_init"].setdefault("runtime_env", {})
    env["py_executable"] = python
    variables = env.setdefault("env_vars", {})
    for key in list(variables):
        if key.startswith("LOCAL_LIFE"):
            del variables[key]
    variables["PYTHONPATH"] = str(Path(source)) + ":" + str(root)
    config["ray_kwargs"]["timeline_json_file"] = None
    return config


def training_batch_execution_contract(config):
    """Record a single-sample baseline; this makes no packing-speed claim."""
    a = config["actor_rollout_ref"]
    return {"mode": "single_sample_microbatch_baseline", "dynamic_packing_performance_claimed": False,
        "input_order": {"shuffle": config["data"]["shuffle"], "seed": config["data"]["seed"],
                        "reward_based_filtering": False, "original_parquet_unchanged": True},
        "max_model_len": a["rollout"]["max_model_len"],
        "actor": {key: a["actor"][key] for key in (
            "use_dynamic_bsz", "ppo_micro_batch_size_per_gpu", "ppo_max_token_len_per_gpu")},
        **{role: {key: a[role][key] for key in (
            "log_prob_use_dynamic_bsz", "log_prob_micro_batch_size_per_gpu", "log_prob_max_token_len_per_gpu")}
           for role in ("rollout", "ref")}}


def write_bundle(args):
    from omegaconf import OmegaConf
    source = Path(args.source).resolve(); root = source / "qwen35_business"
    rows = source_files(root)
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("Business output directory must be new")
    base = OmegaConf.to_container(OmegaConf.load(args.base_config), resolve=True)
    canonical_receipts = []
    original_inputs = [("train",args.train_file),("validation",args.val_file)]
    if args.canonicalize_agent_name:
        sys.path.insert(0,str(root)); sys.path.insert(0,str(root/"support"))
        import business_adapter
        from business_query_context import extract_business_query_context
        canonical_root = Path(str(output) + "-canonical-inputs")
        if canonical_root.exists():
            raise FileExistsError("Canonical data namespace must be new")
        # Data copies have their own namespace so failed preparation cannot
        # masquerade as a complete output bundle. Original Parquet is untouched.
        for role, source_file in original_inputs:
            canonical_receipts.append(canonicalize_agent_name(source_file,canonical_root/(role+".parquet"),
                business_adapter.validate_business_record,extract_business_query_context))
        args.train_file = canonical_receipts[0]["derived_path"]
        args.val_file = canonical_receipts[1]["derived_path"]
    config = bind_config(base, source=str(source), output=str(output), train=args.train_file,
        val=args.val_file, auth=args.auth_file, model=args.model, python=args.python)
    paths = {"business_support_root": str(root / "support"), "business_source_root": str(root / "original-manager"),
        "original_reward_root": str(root / "original-reward"), "http_config": str(root / "original-config/http_tools_config.pydata"),
        "agent_config": str(root / "original-config/example-project.json"),
        "reward_config": str(root / "original-config/reward-config.json"), "system_prompt": str(root / "original-config/default_system_prompt.txt")}
    catalog_verified = False
    interface_review = []
    derived_config_files = []
    derived_interface_changes = []
    if not args.tool_catalog and not args.tool_catalog_receipt and (root / "current-tool-catalog/catalog.json").is_file():
        args.tool_catalog = str(root / "current-tool-catalog/catalog.json")
        args.tool_catalog_receipt = str(root / "current-tool-catalog/receipt.json")
    if args.tool_catalog or args.tool_catalog_receipt:
        if not args.tool_catalog or not args.tool_catalog_receipt:
            raise ValueError("Actual tool catalog and its receipt must be supplied together")
        sys.path.insert(0, str(root))
        from business_http_routes import load_catalog, review_configured_interfaces, derive_current_agent_config
        catalog, _ = load_catalog(args.tool_catalog, args.tool_catalog_receipt)
        # Only load_skill/agent_handoff are the original static implementations.
        import ast
        declared = ast.literal_eval(Path(paths["http_config"]).read_text())
        required = {name for item in declared for name in item.get("httpTools", {}).get("tool_names", [])
                    if name not in {"load_skill", "agent_handoff"}}
        if required - {row["name"] for row in catalog["tools"]}:
            raise ValueError("Original configured business tools are absent from actual new/list")
        manager_tree = ast.parse((root / "original-manager/verl/envs/tool_manager/http_manager.py").read_text())
        constants = [node.value for node in ast.walk(manager_tree) if isinstance(node, ast.Assign)
                     and any(isinstance(target,ast.Name) and target.id == "IGNORED_KEYS" for target in node.targets)]
        if len(constants) != 1:
            raise ValueError("Original manager ignored-key boundary changed")
        agent, derived_interface_changes = derive_current_agent_config(
            catalog, paths["agent_config"], required, ast.literal_eval(constants[0]))
        interface_review = review_configured_interfaces(catalog, agent, required)
        incompatible = [row for row in interface_review if row["absent_original_fields"] or row["new_unprovided_required_fields"]]
        if incompatible:
            raise ValueError("Actual new API fields cannot honor the original protocol: " + json.dumps(incompatible))
        if derived_interface_changes:
            derived_root = output / "derived-http-config"
            original_agent_name = Path(paths["agent_config"]).name
            derived_config_files = [(derived_root / "http_tools_config.pydata", Path(paths["http_config"]).read_bytes()),
                (derived_root / "agent_config.json", (root / "original-config/agent_config.json").read_bytes()),
                (derived_root / original_agent_name, (json.dumps(agent,ensure_ascii=False,indent=2)+"\n").encode())]
            paths.update(http_config=str(derived_root / "http_tools_config.pydata"), agent_config=str(derived_root / original_agent_name))
            for index, (path, raw) in enumerate(derived_config_files):
                rows.append({"role":"current_api_derived_config_"+str(index),"path":str(path),
                             "sha256":hashlib.sha256(raw).hexdigest()})
        paths.update(current_tool_catalog=str(Path(args.tool_catalog).resolve()),
                     current_tool_catalog_receipt=str(Path(args.tool_catalog_receipt).resolve()))
        for role, path in (("current_http_catalog", paths["current_tool_catalog"]),
                           ("current_http_catalog_receipt", paths["current_tool_catalog_receipt"])):
            rows.append({"role":role,"path":path,"sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest()})
        catalog_verified = True
    rollout = {"prompt_length":12288,"response_length":65536,"max_model_len":77824,
        "max_assistant_turns":6,"max_user_turns":6,"max_tool_response_length":8192,"max_new_tokens":4096,
        "temperature":0.7,"top_p":0.75,"tool_timeout":100,"max_retries":3,
        "sampling_seed":config["actor_rollout_ref"]["rollout"]["seed"],"source_sha256":RECIPE_SHA,
        "proof_scope":"Original business recipe budgets; native V1/FSDP wiring candidate, no production replay"}
    contract = {"status":"PREPARED_BUSINESS_WIRING_CANDIDATE", "source_files":rows,"paths":paths,"rollout_contract":rollout,
        "training_batch_execution_contract":training_batch_execution_contract(config),
        "canonical_data":canonical_receipts,
        "http_route_mode":"official_new_route" if catalog_verified else "MISSING_ACTUAL_NEW_CATALOG",
        "gates":{"original_reward_default_overlong_configuration_verified":True,
                 "original_actor_loss_to_native_grpo_mapping_reviewed":True,
                 "actual_model_template_and_reasoning_render_verified":False,
                 "actual_http_auth_and_service_usable":False,
                 "original_http_route_compatible_with_current_business_api":catalog_verified},
        "http_route_boundary":{"pinned_original_list":"/api/tools/list", "pinned_original_call":"/api/tools/{toolName}",
            "current_official_list":"/api/tools/new/list","current_official_call":"/api/tools/new/{toolName}",
            "new_tools_namespace_and_schema_checked":catalog_verified,
            "interface_review":interface_review,
            "derived_interface_changes":derived_interface_changes,
            "adapter_origin":"Explicit SOP mapping over pinned original function code; not agentic_tools master"},
        "negative_history":"Original six 30B samples had reward=-1 and empty reasoning; two real tool errors remain.",
        "actual_optimizer_updates":0, "production_accepted":False}
    sys.path.insert(0, str(root))
    import business_adapter
    # Validate the exact contract parser offline; no manager instantiation.
    sys.path.insert(0, paths["business_support_root"])
    from business_rollout_contract import parse_rollout_contract
    parse_rollout_contract(rollout)
    data_checks = []
    if args.check_inputs:
        import pyarrow.parquet as pq
        from business_query_context import extract_business_query_context
        for role, filename in (("train",args.train_file),("validation",args.val_file)):
            count = 0
            parquet = pq.ParquetFile(filename)
            for batch in parquet.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    business_adapter.validate_business_record(row, extract_business_query_context)
                    count += 1
            if count == 0:
                raise ValueError("Actual dataset is empty: " + role)
            data_checks.append({"role":role,"rows_checked":count})
        auth = Path(args.auth_file)
        if auth.stat().st_mode & 0o077 or not auth.read_text().strip():
            raise ValueError("A private, nonempty authorized auth file is required")
    contract["input_checks"] = data_checks
    output.mkdir(parents=True, exist_ok=False)
    for path, raw in derived_config_files:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
    def write(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    write("business-contract.json", contract)
    source_identity = hashlib.sha256(json.dumps(
        [{"role":row["role"],"sha256":row["sha256"]} for row in rows],sort_keys=True).encode()).hexdigest()
    write("business-native-tools.json", {"tools":[],"business_source_identity_sha256":source_identity})
    # JSON is accepted by OmegaConf.load; the extension matches native loader.
    write("business-agent.yaml", [{"name":AGENT_NAME,"_target_":"business_adapter.create_business_loop",
        "business_contract_path":str(output / "business-contract.json"),"business_auth_file":args.auth_file}])
    assets = [{"role":"tools_config","path":str(output / "business-native-tools.json")},
        {"role":"tool_implementation","path":str(root / "business_http_routes.py")},
        {"role":"reward_implementation","path":str(root / "business_adapter.py")},
        {"role":"business_loop_implementation","path":str(root / "business_adapter.py")},
        {"role":"business_contract","path":str(output / "business-contract.json")},
        {"role":"business_agent_config","path":str(output / "business-agent.yaml")},
        *[{"role":"original_business_"+role+"_data","path":str(Path(path).resolve())} for role,path in original_inputs],
        *[{"role":row["role"],"path":row["path"]} for row in rows]]
    write("business-assets.json", {"files":assets})
    OmegaConf.save(OmegaConf.create(config), output / "business-config.yaml")
    write("business-preparation-receipt.json", {"status":"PREPARED_BUSINESS_WIRING_CANDIDATE",
        "source":str(source),"actual_updates":0,"input_checks":data_checks,
        "canonical_data":canonical_receipts,
        "credentials_in_receipt":False,"gates":contract["gates"],"files":[{"name":p.name,
        "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.rglob("*")) if p.is_file()]})


def main():
    parser = argparse.ArgumentParser()
    for name in ("source","base-config","train-file","val-file","auth-file","model","python","output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--check-inputs", action="store_true", help="Read all actual rows and private auth permissions; no HTTP")
    parser.add_argument("--tool-catalog", help="Actual authenticated new/list response; no request is issued here")
    parser.add_argument("--tool-catalog-receipt", help="GET endpoint/auth_supplied/HTTP200/body SHA receipt")
    parser.add_argument("--canonicalize-agent-name", action="store_true", help="Derive real Parquet copies changing only technical loop metadata")
    write_bundle(parser.parse_args())


if __name__ == "__main__":
    main()
