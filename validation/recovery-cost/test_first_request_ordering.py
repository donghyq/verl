import pathlib,asyncio,contextlib,types,json,time
P=pathlib.Path(__file__).resolve().parent
ns={'asyncio':asyncio,'asynccontextmanager':contextlib.asynccontextmanager}
exec(compile((P/'actual_gate_factory.py.txt').read_text(),'actual_gate_factory','exec'),ns)
make=ns['make_gate']; passed=[]; snapshot=types.SimpleNamespace(kv_ref={'shape_fixture':True})
def check(ok,name):
    assert ok,name
    passed.append(name)
async def case(mode):
    turn,finish,future=await make(); sem=asyncio.Semaphore(1); seen=[]; error=ValueError('original source/owner failure fixture')
    async def worker(i):
        async with turn(i,None if mode=='missing' or i!=2 else snapshot),sem:
            seen.append(i)
            if i==2:
                await asyncio.sleep(0)
                if mode=='error': raise error
                if mode=='cancel': raise asyncio.CancelledError('selected member cancelled')
                if mode=='success': finish()
            return i
    out=await asyncio.wait_for(asyncio.gather(*(worker(i) for i in range(8)),return_exceptions=True),1)
    if mode=='success':
        check(seen[0]==2,'first actual body is designated member2')
        check(sorted(out)==list(range(8)),'all8 execute exactlyonce')
        check(future.done() and future.exception() is None,'first pair releases gate')
        async with turn(2,None): pass
        check(True,'latergroups do not require newpair snapshot')
    elif mode=='error':
        check(all(v is error for v in out),'same first error propagates to all8')
        check(seen==[2],'no other body starts after sourcefailure')
    elif mode=='missing':
        check(all(isinstance(v,ValueError) for v in out),'missing snapshot releases all8 withfailure')
        check(not seen,'no body onmissingdesignated snapshot')
    elif mode=='no_pair':
        check(all(isinstance(v,RuntimeError) for v in out),'return without pair doesnotdeadlock')
        check(seen==[2],'no other body beforepair')
    elif mode=='cancel':
        check(all(isinstance(v,asyncio.CancelledError) for v in out),'selected cancellation propagates withoutdeadlock')
async def main():
    for mode in ['success','error','missing','no_pair','cancel']: await case(mode)
    for bad in [None,True,-1,8,'2']:
        try: await make('owned-A',bad)
        except ValueError: check(True,'reject invalidselected index '+repr(bad))
        else: raise AssertionError('invalid indexaccepted')
    for mode in [None,'']:
        turn,finish,f=await make(mode,None)
        check(f is None,'default mode doesnotallocate gate '+repr(mode))
        async with turn(0,None): pass
    turn,finish,f=await make(); started=asyncio.Event(); release=asyncio.Event()
    async def selected():
        async with turn(2,snapshot):
            started.set(); await release.wait(); finish()
    async def waiting():
        async with turn(0,None): raise AssertionError('cancelled waiter entered body')
    a=asyncio.create_task(selected()); b=asyncio.create_task(waiting())
    await started.wait(); await asyncio.sleep(0); b.cancel(); await asyncio.gather(b,return_exceptions=True)
    check(not f.cancelled() and not f.done(),'one cancelled waiter preserves shared gate')
    release.set(); await asyncio.wait_for(a,1)
    check(f.done() and f.exception() is None,'selected pair continues afterotherwaitercancel')
t=time.monotonic(); asyncio.run(main())
r={'status':'PASS_ACTUAL_NEW_GATE_CONCURRENCY_CPU_NOT_LIVE_RUNTIME_LEASE','checks':len(passed),
   'elapsed_s':time.monotonic()-t,'passed':passed,'actual_HTTP_GPU_model_calls':0,
   'actual_lease_renewal_exercised':False,'scope':'actual gate source withCPUcallback/snapshotfixtures; originalrun_target/lease/nativechecks unchanged',
   'remaining':['dedicated freshA/B lifecycle','newcostbundle/sourceidentitypins','realtrainedversionbenchmark','wholewindowcostandqualityacceptance']}
(P/'CPU-ordering-receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:r[k] for k in ['status','checks','elapsed_s']}))
