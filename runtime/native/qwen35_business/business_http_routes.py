"""Explicit SOP route adapter; reuse original function code, never patch globals.

Not the inaccessible agentic_tools master. Catalog must be a real authenticated
new/list readback, frozen before any model/manager creation. No HTTP at import.
"""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import FunctionType

HOST = "https://service.example.invalid"
LIST_URL = HOST + "/api/tools/new/list"
RECIPE_SHA = "fb6bfcb11131c342ec5b83de82303b4b5c6a3f85b29fcb9257d43575e518cdb7"


def load_catalog(catalog_path, receipt_path):
    raw = Path(catalog_path).read_bytes()
    receipt = json.loads(Path(receipt_path).read_text())
    # Support the actual system-trust curl receipt, not only the CPU fixture's
    # normalized fields. No credential contents are required or propagated.
    auth_supplied = receipt.get("auth_supplied")
    if auth_supplied is None:
        auth_supplied = bool(receipt.get("credential_source") and receipt.get("credential_in_logs_or_payload") is False
            and receipt.get("recipe_sha256") == RECIPE_SHA and receipt.get("tls_verification_enabled") is True
            and receipt.get("curl_exit_code") == 0)
    if (receipt.get("endpoint", receipt.get("url")) != LIST_URL or receipt.get("method") != "GET"
            or receipt.get("http_status") != 200
            or auth_supplied is not True
            or receipt.get("response_sha256") != hashlib.sha256(raw).hexdigest()):
        raise ValueError("Actual authenticated new/list readback does not match its receipt")
    payload = json.loads(raw)
    tools = payload.get("tools") if isinstance(payload, dict) else None
    if not isinstance(tools, list) or not tools:
        raise ValueError("Actual new/list shape differs from the original manager's tools protocol")
    names = []
    for row in tools:
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                or not row["name"] or not isinstance(row.get("description"), str)
                or not isinstance(row.get("inputSchema"), dict)):
            raise ValueError("New tool namespace/schema is incomplete; no schema inferred")
        names.append(row["name"])
    if len(names) != len(set(names)):
        raise ValueError("New tool catalog contains ambiguous duplicate names")
    if receipt.get("business_code") not in (None, 0, "0") or receipt.get("business_status") not in (None, 0, "0", "success"):
        raise ValueError("Tool-list receipt reports a business error")
    return payload, receipt


def review_configured_interfaces(catalog, agent_config_path, required_names):
    """Compare observed backend fields with the original exposed protocol."""
    source = agent_config_path if isinstance(agent_config_path, dict) else json.loads(Path(agent_config_path).read_text())
    original = {row.get("name") or row.get("tool_id"):row for row in source.get("tool_schemas", [])}
    observed = {row["name"]:row for row in catalog["tools"]}
    missing = sorted(set(required_names) - set(observed))
    if missing:
        raise ValueError("Original tools absent from actual new/list: " + ", ".join(missing))
    rows = []
    for name in sorted(required_names):
        if name not in original:
            raise ValueError("Configured tool has no original exposed schema: " + name)
        exposed = {field["name"]:field for field in original[name].get("input_fields", [])}
        actual = observed[name]["inputSchema"]
        properties = actual.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError("Actual backend properties are not a mapping: " + name)
        absent = sorted(set(exposed) - set(properties))
        # These defaults/context keys are added by the original HTTPTool and
        # same-record adapter; other new required arguments cannot be invented.
        provided = {"userId","uuid","lat","lng","originalQuery","useCache","searchSource"}
        required_unknown = sorted(set(actual.get("required", [])) - set(exposed) - provided)
        type_changes = [{"field":key,"original_type":field.get("type"),
            "actual_type":properties[key].get("type")} for key,field in exposed.items()
            if key in properties and isinstance(properties[key],dict) and field.get("type") != properties[key].get("type")]
        rows.append({"name":name,"absent_original_fields":absent,"new_unprovided_required_fields":required_unknown,
                     "type_differences":type_changes})
    return rows


def derive_current_agent_config(catalog, agent_config_path, required_names, ignored_keys):
    """Change only interfaces proven different by the actual catalog.

    Historical prompts, messages and rewards are not touched. Static skill
    content and unchanged tool entries are retained byte-for-value in JSON.
    """
    source = json.loads(Path(agent_config_path).read_text())
    review = review_configured_interfaces(catalog, agent_config_path, required_names)
    changed = {row["name"] for row in review if row["absent_original_fields"] or row["new_unprovided_required_fields"]}
    actual = {row["name"]:row for row in catalog["tools"]}
    differences = []
    derived = copy.deepcopy(source)
    for tool in derived.get("tool_schemas", []):
        name = tool.get("name") or tool.get("tool_id")
        if name not in changed:
            continue
        schema = actual[name]["inputSchema"]
        required = set(schema.get("required", []))
        fields = []
        for key, prop in schema.get("properties", {}).items():
            if key in ignored_keys:
                continue
            if not isinstance(prop, dict) or not isinstance(prop.get("type"), str):
                raise ValueError("Actual changed tool type is missing; no type inferred: " + name + "." + key)
            field = {"name":key, "type":prop["type"], "required":key in required}
            for attribute in ("description","enum","default","items"):
                if attribute in prop:
                    field[attribute] = copy.deepcopy(prop[attribute])
            fields.append(field)
        differences.append({"name":name,"before_input_fields":copy.deepcopy(tool.get("input_fields", [])),
            "after_input_fields":copy.deepcopy(fields),"source":"Actual authenticated new/list inputSchema"})
        tool["input_fields"] = fields
    return derived, differences


def _scoped_function(original, **bindings):
    if original.__closure__:
        raise ValueError("Original HTTP function has unexpected closure state")
    return FunctionType(original.__code__, {**original.__globals__, **bindings},
                        original.__name__, original.__defaults__)


class _CatalogResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None  # load_catalog already required the actual HTTP 200 receipt.

    def json(self):
        return copy.deepcopy(self.payload)


class _ScopedRequests:
    def __init__(self, original_requests, catalog):
        self.original = original_requests
        self.catalog = catalog
        self.allowed_names = {row["name"] for row in catalog["tools"]}

    def get(self, url, **kwargs):
        if url != HOST + "/api/tools/list":
            raise ValueError("Original catalog lookup changed host/path")
        return _CatalogResponse(self.catalog)

    def post(self, url, *args, **kwargs):
        prefix = HOST + "/api/tools/"
        if not isinstance(url, str) or not url.startswith(prefix):
            raise ValueError("Original tool request changed host/path")
        name = url[len(prefix):]
        if name not in self.allowed_names or "/" in name or "?" in name:
            raise ValueError("Tool is absent from the actual frozen new catalog")
        return self.original.post(HOST + "/api/tools/new/" + name, *args, **kwargs)


def load_routed_runtime(loader, *, catalog_path, catalog_receipt_path, **kwargs):
    catalog, receipt = load_catalog(catalog_path, catalog_receipt_path)
    root = Path(kwargs["source_root"])
    # The original loader uses exactly these private names and original files.
    base_path = root / "verl/envs/tool_manager/base.py"
    manager_path = root / "verl/envs/tool_manager/http_manager.py"
    logger_path = root / "verl/utils/cat_logger.py"
    loader._module("verl.utils.cat_logger", logger_path)
    import sys
    import types
    package = "_original_project_b_business"
    if package not in sys.modules:
        module = types.ModuleType(package); module.__path__ = [str(base_path.parent)]
        sys.modules[package] = module
    loader._module(package + ".base", base_path)
    backend = loader._module(package + ".http_manager", manager_path)
    scoped = _ScopedRequests(backend.requests, catalog)

    class RoutedHTTPTool(backend.HTTPTool):
        call = _scoped_function(backend.HTTPTool.call, requests=scoped)

    class RoutedHTTPToolManager(backend.HTTPToolManager):
        _get_tool_schema = _scoped_function(backend.HTTPToolManager._get_tool_schema, requests=scoped)
        _create_http_tool = _scoped_function(backend.HTTPToolManager._create_http_tool, HTTPTool=RoutedHTTPTool)

    # Keep the original loader's complete checks/config/source behavior. Only
    # its one constructor target is bound privately to our derived manager.
    tree = ast.parse(inspect.getsource(loader.load_original_business_runtime))
    count = 0
    for call in ast.walk(tree):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "HTTPToolManager" and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "backend"):
            call.func = ast.Name(id="manager_constructor", ctx=ast.Load()); count += 1
    if count != 1:
        raise ValueError("Original loader constructor boundary changed")
    ast.fix_missing_locations(tree)
    namespace = {**loader.load_original_business_runtime.__globals__, "manager_constructor":RoutedHTTPToolManager}
    exec(compile(tree, loader.__file__, "exec"), namespace)
    runtime = namespace["load_original_business_runtime"](**kwargs)
    runtime.receipt["current_http_route_adapter"] = {
        "implementation":"SOP route mapping; not agentic_tools master source",
        "list_endpoint":LIST_URL,"tool_endpoint_prefix":HOST + "/api/tools/new/",
        "catalog_sha256":receipt["response_sha256"],"original_function_code_reused":True,
        "source_files_modified":False,"original_requests_module_modified":False}
    return runtime
