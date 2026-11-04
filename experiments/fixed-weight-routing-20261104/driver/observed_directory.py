"""One request may reuse its verified native proof inside the existing view TTL.

No cross-request caching, new coverage, state installation or weight authority.
The unchanged backend still validates the target at generation admission.
"""
from copy import deepcopy
import time
from candidate.native_cache_directory_client import native_query_digest

class ObservedDirectory:
    def __init__(self,observations,query,*,clock=time.time,max_age_s=2):
        self.observations=deepcopy(observations);self.query_binding=deepcopy(query)
        self.clock=clock;self.max_age_s=max_age_s;self.closed=False
        if not observations or len(observations)!=2:raise ValueError('two real native proofs required')
    def query(self,query):
        if self.closed:raise TimeoutError('request observation closed')
        bound=self.query_binding
        for k in ('model','tenant_id','lora_name','block_size','token_ids','cache_salt','expected_weight_version','native_allowed_instances'):
            if query.get(k)!=bound.get(k):raise ValueError('observation binding changed: '+k)
        expected={k:v['native_query']['rank_generations'] for k,v in self.observations.items()}
        if query.get('native_expected_rank_generations')!=expected:raise ValueError('native generation changed')
        qdigest=native_query_digest(query['token_ids'],query['cache_salt'])
        now=self.clock()
        for value in self.observations.values():
            proof=value['native_query'];age=now-proof['observed_at_unix']
            if (proof['query_digest']!=qdigest or proof['actual_weight_version']!=query['expected_weight_version']
                or not 0<=age<=self.max_age_s):raise ValueError('observation expired or mismatched')
        return deepcopy(self.observations)
    def close(self):self.closed=True
