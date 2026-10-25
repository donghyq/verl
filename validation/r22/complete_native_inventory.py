# Copyright 2026 Individual Contributor: donghyq
# SPDX-License-Identifier: Apache-2.0
def complete_native_inventory(row):
    """Match every native snapshot tensor to the actual hybrid model layout."""
    import math
    identity = row['identity'];layout = identity['cache_layout']
    full = layout['full_attention_layer_ids'];mamba = layout['mamba_layer_ids']
    if (any(type(i) is not int for i in full+mamba) or len(set(full+mamba)) != len(full+mamba)
            or set(full+mamba) != set(range(identity['num_hidden_layers']))):
        return False
    conv = layout['conv'];temporal = layout['temporal']
    specs = {f'.mamba_conv.{i}': x for i,x in enumerate(conv)}
    specs['.mamba_temporal'] = temporal
    if any(x['shape'][0] != len(mamba) for x in specs.values()):return False
    expected = {f'.attention_kv.{i}.0.{j}' for i in range(len(full)) for j in (0,1)} | set(specs)
    inventory = row['state_inventory'];names = [x['name'] for x in inventory]
    if len(names) != len(set(names)) or set(names) != expected:return False
    kv = [];sizes = {'torch.bfloat16':2,'torch.float16':2,'torch.float32':4}
    for x in inventory:
        shape,dtype,digest = x['shape'],x['dtype'],x['sha256']
        if (not shape or any(type(n) is not int or n <= 0 for n in shape)
                or dtype not in sizes or type(x['bytes']) is not int
                or x['bytes'] != math.prod(shape)*sizes[dtype]
                or not isinstance(digest,str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            return False
        if x['name'] in specs:
            spec = specs[x['name']]
            if shape != spec['shape'][:1]+spec['shape'][2:] or dtype != spec['dtype']:return False
        else:kv.append((tuple(shape),dtype))
    return bool(kv) and len(set(kv)) == 1 and set(row['state_kinds']) == {'attention_kv','mamba_conv','mamba_temporal'}
