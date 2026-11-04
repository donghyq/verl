"""Wait for the exact owned healthy worker set; HTTP health alone is insufficient."""
import asyncio,time

async def wait_workers(wire,gateway,urls,label,*,timeout=30):
    end=min(wire.deadline-2,time.time()+timeout)
    expected=set(urls)
    if len(expected)!=2:raise ValueError('two distinct owned endpoints required')
    while time.time()<end:
        status,registry,_=await wire.call(label+'-registry',gateway,'/workers')
        if status!=200:raise ValueError('registry HTTP failed: '+str(status))
        workers=registry.get('workers')
        if not isinstance(workers,list):raise ValueError('malformed worker registry')
        mapping={r.get('url'):r.get('id') for r in workers}
        ids=list(mapping.values())
        if (len(mapping)!=len(workers) or not set(mapping)<=expected
            or any(not isinstance(i,str) or not i for i in ids) or len(set(ids))!=len(ids)):
            raise ValueError('unexpected/duplicate/unidentified worker, refuse dispatch')
        if set(mapping)==expected and all(r.get('is_healthy') is True for r in workers):
            return mapping
        await asyncio.sleep(.05)
    raise TimeoutError('owned healthy workers not registered before absolute readiness deadline')
