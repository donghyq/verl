"""ROOT fallback from existing state helpers; source only, not MCC output or FSDP evidence."""
import copy,json,os,random,resource,signal,subprocess,sys,time,uuid
os.environ['CUDA_VISIBLE_DEVICES']='';os.umask(0o077)
MIB=1048576; STORAGE=256*MIB; CAP=3*1024**3
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def rng():
    n=np.random.get_state();return dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=(n[0],n[1].tolist(),*n[2:]))
def restore(r):
    torch.set_rng_state(r['torch']);random.setstate(r['python']);n=r['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),*n[2:]))
def make(backing=None):
    layer=torch.nn.Linear(4,4,bias=False)
    if backing is not None:layer.weight=torch.nn.Parameter(backing[:16].view(4,4))
    net=torch.nn.Sequential(layer,torch.nn.Dropout(.25));opt=torch.optim.Adam(net.parameters(),lr=.001)
    return net,opt,torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9)
def step(net,opt,sched):
    x=torch.randn(5,4)+float(np.random.normal())+random.random();opt.zero_grad(set_to_none=True)
    net(x).square().mean().backward();assert torch.count_nonzero(net[0].weight.grad).item()>0;opt.step();sched.step()
def small(net,opt,sched):
    return copy.deepcopy(dict(model={k:v.detach().clone() for k,v in net.state_dict().items()},optim=opt.state_dict(),scheduler=sched.state_dict(),rng=rng()))
def peak():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1024 if sys.platform=='linux' else 1)
def deadline(*_):raise TimeoutError('CPU preflight deadline')
def worker(arm,p):
    global torch,np
    import torch,numpy as np
    print(json.dumps(dict(stage='IMPORT',torch_version=str(torch.__version__),python_version=sys.version.split()[0],cuda_initialized=torch.cuda.is_initialized())),flush=True)
    assert sys.platform=='linux' and str(torch.__version__)=='2.13.0+cu129' and not torch.cuda.is_initialized()
    torch.set_num_threads(1);torch.manual_seed(20261025);np.random.seed(20261025);random.seed(20261025)
    backing=torch.rand(STORAGE//4,dtype=torch.float32,device='cpu');net,opt,sched=make(backing);step(net,opt,sched)
    payload=dict(backing=backing,views=dict(strided=backing[7:707:3],transposed=backing[:4096].view(64,64).t()),model=net.state_dict(),optim=opt.state_dict(),scheduler=sched.state_dict(),rng=rng())
    begin=time.monotonic();pre_peak=peak()
    if arm=='path':
        start=time.monotonic();torch.save(payload,p);returned=time.monotonic()
        with open(p,'r+b') as f:f.flush();os.fsync(f.fileno())
    else:
        assert arm=='buffer16MiB'
        with open(p,'wb',buffering=16*MIB) as f:
            start=time.monotonic();torch.save(payload,f);returned=time.monotonic();f.flush();os.fsync(f.fileno())
    end=time.monotonic();save_peak=peak();archive_bytes=os.stat(p).st_size
    loaded=torch.load(p,map_location='cpu',weights_only=True);ref=loaded['backing'];ptr=ref.untyped_storage().data_ptr()
    assert ref.untyped_storage().nbytes()==STORAGE and torch.equal(ref,backing)
    for name,old in payload['views'].items():
        new=loaded['views'][name];assert new.untyped_storage().data_ptr()==ptr and new.shape==old.shape and new.stride()==old.stride() and new.storage_offset()==old.storage_offset() and torch.equal(new,old)
    assert equal(loaded['model'],payload['model']) and loaded['model']._metadata==payload['model']._metadata
    weight=loaded['model']['0.weight'];assert weight.untyped_storage().data_ptr()==ptr and weight.storage_offset()==0 and weight.stride()==(4,1)
    before=net[0].weight.detach().clone();restore(payload['rng']);step(net,opt,sched);assert not torch.equal(before,net[0].weight);expected=small(net,opt,sched)
    clone,clone_opt,clone_sched=make(loaded['backing']);clone.load_state_dict(loaded['model']);clone_opt.load_state_dict(loaded['optim']);clone_sched.load_state_dict(loaded['scheduler'])
    assert clone[0].weight.untyped_storage().data_ptr()==ptr
    restore(loaded['rng']);step(clone,clone_opt,clone_sched);assert equal(expected,small(clone,clone_opt,clone_sched))
    total_peak=peak();assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASS_CPU_STATE_IO_ONLY' if total_peak<CAP else 'FAILED_RSS_BUDGET',arm=arm,seed=20261025,torch_version=str(torch.__version__),cuda_initialized=False,backing_bytes=STORAGE,alias_layout_values_equal=True,next_CPU_update_model_Adam_scheduler_RNG_equal=True,nonzero_gradient_and_parameter_delta=True,torch_save_s=returned-start,flush_fsync_close_s=end-returned,total_save_s=end-begin,archive_bytes=archive_bytes,pre_save_peak_RSS_bytes=pre_peak,save_peak_RSS_bytes=save_peak,validation_peak_RSS_bytes=total_peak,peak_scope='fresh child cumulative high-water; save peak precedes load; validation peak includes load',timing_scope='open/save/flush/fsync/close only; excludes UUID reservation, load, validation and cleanup')),flush=True)
    assert total_peak<CAP
def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':return worker(sys.argv[2],sys.argv[3])
    if len(sys.argv)!=3 or sys.argv[1]!='--execute':raise SystemExit('Source-only by default. Explicit execution: SCRIPT --execute OWN_EXISTING_HDD_DIR')
    root=os.path.abspath(sys.argv[2]);assert os.path.isdir(root) and os.access(root,os.W_OK)
    owned=[];rows=[];failure=None;signal.signal(signal.SIGALRM,deadline);signal.alarm(310)
    try:
        for arm in ('path','buffer16MiB'):
            p=os.path.join(root,'project-b-large-storage-'+uuid.uuid4().hex+'.pt');fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);owned.append(p);os.close(fd)
            try:r=subprocess.run([sys.executable,'-B',os.path.abspath(__file__),'--worker',arm,p],capture_output=True,text=True,timeout=150)
            except subprocess.TimeoutExpired as e:rows.append(dict(arm=arm,status='TIMEOUT',stdout=str(e.stdout or ''),stderr=str(e.stderr or '')));raise
            rows.append(dict(arm=arm,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
            if r.returncode:raise RuntimeError('CPU arm failed; no retry')
    except BaseException as e:failure=e
    finally:
        signal.alarm(0);begin=time.monotonic();errors=[]
        for p in owned:
            try:os.unlink(p)
            except FileNotFoundError:pass
            except OSError as e:errors.append(dict(path=p,error_type=type(e).__name__))
        removed=all(not os.path.lexists(p) for p in owned)
        print(json.dumps(dict(status='FAILED' if failure or not removed else 'PASS_CPU_PREFLIGHT_ONLY',implementation='ROOT_FALLBACK_EXISTING_HELPERS_NOT_MCC',scope='256MiB CPU serialization/state diagnostic; no GPU/FSDP/35B performance claim',arms=rows,owned_UUID_files=owned,own_UUID_files_removed=removed,cleanup_s=time.monotonic()-begin,cleanup_errors=errors,error_type=type(failure).__name__ if failure else None)),flush=True)
    if failure:raise failure
    if not removed:raise RuntimeError('Own UUID cleanup incomplete')
if __name__=='__main__':main()
