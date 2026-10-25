# Copyright 2026 Individual Contributor: donghyq
# SPDX-License-Identifier: Apache-2.0
def native_handoff_timeline(events, after_train):
    """Check observed driver operations without inventing allocation timestamps."""
    order = ['switch_to_trainer_begin','quiesced','replica_sleep_begin',
             'sleep_completed','switch_to_trainer_returned','native_actor_update_dispatched']
    result = {'verified':False,'timeline':{},'trainer_allocation_instant':None,
              'scope':'same_driver_handoff_to_native_update_order_not_allocation_timestamp'}
    starts = [i for i,e in enumerate(events) if e.get('event') == order[0]]
    if not starts:return result
    rows = [e for e in events[starts[-1]:] if e.get('event') in order]
    if [e['event'] for e in rows] != order:return result
    if any(e.get('controller_pid') != after_train.get('driver_pid') for e in rows):return result
    for key in ['sequence','monotonic_ns','observed_unix_s']:
        values = [e.get(key) for e in rows]
        if any(type(x) not in (int,float) for x in values):return result
        if any(a >= b for a,b in zip(values,values[1:])):return result
    end = after_train.get('driver_unix_s')
    if type(end) not in (int,float) or end <= rows[-1]['observed_unix_s']:return result
    result.update(verified=True,timeline={e['event']:e['observed_unix_s'] for e in rows})
    result['timeline']['native_update_returned'] = end
    return result
