"""Lossless views of the original business schemas, including omitted types.

The original XML parser treats a property without ``type`` as a string. This
view preserves that input instead of inventing JSON Schema types merely to fit
the community Pydantic model. It is private business-format compatibility.
"""
import copy
from collections.abc import Mapping


class _View:
    def __init__(self, raw):
        if not isinstance(raw, Mapping):
            raise ValueError('business schema must be an object')
        self._raw = copy.deepcopy(dict(raw))

    def model_dump(self, **_kwargs):
        return copy.deepcopy(self._raw)


class _Parameters(_View):
    def __init__(self, raw):
        super().__init__(raw)
        properties = self._raw.get('properties', {})
        required = self._raw.get('required', [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError('invalid business parameters shape')
        if set(required) - set(properties):
            raise ValueError('required business property is not declared')
        self.properties = {name: _View(value) for name, value in properties.items()}
        self.required = list(required)


class _Function(_View):
    def __init__(self, raw):
        super().__init__(raw)
        self.name = self._raw.get('name')
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('missing original business function name')
        self.parameters = _Parameters(self._raw.get('parameters', {}))


class OriginalBusinessSchemaView(_View):
    def __init__(self, raw):
        if not isinstance(raw, Mapping):
            raise ValueError('business function must be an object')
        wrapped = dict(raw) if 'function' in raw else {'type':'function','function':dict(raw)}
        if wrapped.get('type') != 'function':
            raise ValueError('unsupported original business tool kind')
        super().__init__(wrapped)
        self.type = 'function'
        self.function = _Function(self._raw['function'])
