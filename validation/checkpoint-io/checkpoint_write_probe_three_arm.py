import os, sys, json, time, uuid, signal, tempfile
os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.umask(0o077)
def timeout(signum, frame): raise TimeoutError('85s I/O probe limit')
signal.signal(signal.SIGALRM, timeout); signal.alarm(85)
owned, paths, failure = [], {}, None
out = dict(status='FAILED', scope='CPU_IO_DIAGNOSTIC_ONLY_NOT_GPU_TRAINING_OR_FSDP', platform=sys.platform)
def own(directory):
    p = os.path.join(directory, uuid.uuid4().hex+'.pt')
    fd = os.open(p, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600); owned.append(p); os.close(fd)
    return p
def sync_path(p):
    with open(p, 'r+b') as f: f.flush(); os.fsync(f.fileno())
try:
    import torch
    assert not torch.cuda.is_initialized(); torch.set_num_threads(1)
    hdd = os.path.abspath(sys.argv[1]); local = os.path.abspath(sys.argv[2] if len(sys.argv)>2 else tempfile.gettempdir())
    if not all(os.path.isdir(p) and os.access(p, os.W_OK) for p in (hdd,local)): raise ValueError('require existing own writable HDD/local dirs')
    generator = torch.Generator(device='cpu'); generator.manual_seed(20261025)
    state = {f'k{i}':torch.rand(256*1024,dtype=torch.float32,generator=generator,device='cpu') for i in range(32)}
    state['extra'] = torch.rand(256*1024,dtype=torch.float32,generator=generator,device='cpu')
    out.update(torch_version=str(torch.__version__), hdd_dir=hdd, local_dir=local, input_logical_bytes=33*1048576,
        timer_scope='write/publish totals include creation, flush, fsync, close; validation/cleanup excluded', results={})
    print('META_PRE '+json.dumps(out), flush=True)
    for arm in ('path_string','buffered16MiB','local_stage_copy16MiB'):
        out['active_stage']=arm; begin=time.monotonic(); hp=own(hdd); paths[arm]=hp; details={}
        if arm=='path_string':
            start=time.monotonic(); torch.save(state,hp); returned=time.monotonic(); sync_path(hp); end=time.monotonic()
            details.update(torch_save_s=returned-start,fsync_close_s=end-returned)
        elif arm=='buffered16MiB':
            with open(hp,'wb',buffering=16*1048576) as f:
                start=time.monotonic(); torch.save(state,f); returned=time.monotonic(); f.flush(); os.fsync(f.fileno())
            end=time.monotonic(); details.update(torch_save_s=returned-start,fsync_close_s=end-returned)
        else:
            local_begin=time.monotonic(); lp=own(local); start=time.monotonic()
            torch.save(state,lp); returned=time.monotonic(); sync_path(lp); local_end=time.monotonic()
            copied=0; copy_begin=time.monotonic()
            with open(lp,'rb') as src, open(hp,'wb',buffering=16*1048576) as dst:
                while chunk:=src.read(16*1048576): copied+=dst.write(chunk)
                dst.flush(); os.fsync(dst.fileno())
            end=time.monotonic(); lb=os.stat(lp).st_size
            if copied!=lb or os.stat(hp).st_size!=lb: raise ValueError('local/copy/HDD byte count differs')
            details.update(torch_save_s=returned-start,local_fsync_close_s=local_end-returned,
                local_save_with_fsync_s=local_end-local_begin,chunk_copy_with_fsync_s=end-copy_begin,
                local_bytes=lb,copied_bytes=copied,local_path=lp)
        size=os.stat(hp).st_size; total=end-begin
        details.update(path=hp,actual_archive_bytes=size,total_save_publish_s=total,MiB_per_s_total=size/1048576/total)
        out['results'][arm]=details
    out['active_stage']='readback_validation'; validation_begin=time.monotonic(); equal={}
    for arm,p in paths.items():
        loaded=torch.load(p,map_location='cpu',weights_only=True)
        if set(loaded)!=set(state): raise ValueError('loaded key set differs')
        equal[arm]={k:bool(loaded[k].shape==state[k].shape and loaded[k].dtype==state[k].dtype and torch.equal(loaded[k],state[k])) for k in state}
        if not all(equal[arm].values()): raise ValueError('loaded tensor values differ')
    out.update(validation_s=time.monotonic()-validation_begin,per_key_equal=equal)
    assert not torch.cuda.is_initialized()
    out.update(status='PASS_CPU_IO_VALUE_CHECK_ONLY',cuda_initialized=False,
        limitation='33MiB diagnostic; local Torch2.11 is format/value only; no 35B/FSDP or global storage throughput claim')
except BaseException as error:
    failure=error; out.update(status='FAILED',error_type=type(error).__name__,error=str(error))
finally:
    signal.alarm(0); cleanup_begin=time.monotonic(); cleanup_errors=[]
    for p in owned:
        try: os.unlink(p)
        except FileNotFoundError: pass
        except OSError as error: cleanup_errors.append(dict(path=p,error_type=type(error).__name__))
    removed=all(not os.path.lexists(p) for p in owned)
    out.update(owned_UUID_files=owned,own_UUID_files_removed=removed,cleanup_s=time.monotonic()-cleanup_begin,cleanup_errors=cleanup_errors)
    if not removed: out['status']='FAILED'; failure=failure or RuntimeError('own UUID cleanup incomplete')
    print('FINAL '+json.dumps(out),flush=True)
if failure is not None: raise failure
