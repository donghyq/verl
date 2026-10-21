"""Optional CPU Gateway policy switch in the original bootstrap parent.

Python 3.6 compatible, standard library only. No model/Ray resource operations.
The child sends a binding, opaque request ID and one of two policies; it cannot
send commands or PIDs. Only the original OwnedPhase launches/reaps replacements.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
import uuid

POLICIES = ('least_load', 'cache_aware')
_FRAME = ('protocol', 'channel_id', 'binding', 'owner_hostname', 'kernel_boot_id', 'source_sha256')
_REQUEST_KEYS = set(_FRAME + ('request_id', 'operation', 'policy'))


class OwnedGatewayError(RuntimeError):
    pass


def _boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _read_cmdline(pid):
    return [x.decode('utf-8') for x in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if x]


def _cpu_seconds(pid):
    try:
        text = (Path('/proc') / str(pid) / 'stat').read_text()
        fields = text[text.rfind(')') + 2:].split()
        return (int(fields[11]) + int(fields[12])) / float(os.sysconf('SC_CLK_TCK'))
    except (OSError, ValueError, IndexError):
        return None


def _write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(str(tmp), str(path))


def _read(path, uid):
    path = Path(path)
    stat = path.lstat()
    if path.is_symlink() or stat.st_uid != uid or stat.st_mode & 0o022 or stat.st_size > 65536:
        raise OwnedGatewayError('Foreign, indirect or oversized owner control file')
    return json.loads(path.read_text())


class OwnedGatewayParent:
    """Only called by the original run_phase parent, inside its existing wait."""
    def __init__(self, owner, gateway, *, channel, gateway_url, native_argv,
                 launch_argv, source_sha256, head_ip, local_ip):
        import cleanup_qwen35
        self.process_tools = cleanup_qwen35
        self.owner, self.gateway = owner, gateway
        self.native_argv, self.launch_argv = list(native_argv), launch_argv
        self.gateway_url = gateway_url.rstrip('/')
        parsed = urlsplit(self.gateway_url)
        if head_ip != local_ip or parsed.hostname != head_ip or parsed.scheme != 'http' or not parsed.port:
            raise OwnedGatewayError('The original owner must run on the exact owned Gateway head')
        expected = ['smg', 'launch', '--host', '0.0.0.0', '--port', str(parsed.port),
                    '--policy', 'least_load', '--enable-rl', '--disable-retries']
        if self.native_argv != expected or not re.fullmatch('[0-9a-f]{64}', source_sha256):
            raise OwnedGatewayError('Only the exact original native launch and frozen source are allowed')
        if not any(proc is gateway for proc, _ in owner.handles) or owner.binding.get('rank') != 0:
            raise OwnedGatewayError('Gateway must be an actual head-parent OwnedPhase handle')
        self.channel = Path(channel)
        self.channel.mkdir(mode=0o700, exist_ok=False)
        self.uid = os.getuid(); self.host = socket.gethostname(); self.boot = _boot_id()
        self.owner_identity = self.process_tools.proc_identity(os.getpid())
        self.gateway_identity = self.process_tools.proc_identity(gateway.pid)
        if not self.owner_identity or not self.gateway_identity:
            raise OwnedGatewayError('Actual owner/Gateway identity is unavailable')
        self.policy = 'least_load'; self.deadline = None; self.unusable = False
        self.last_input_sha = None; self.last_restart = None
        self.frame = {'protocol': 1, 'channel_id': uuid.uuid4().hex,
                      'binding': dict(owner.binding), 'owner_hostname': self.host,
                      'kernel_boot_id': self.boot, 'source_sha256': source_sha256}
        self.endpoint = self.channel / 'endpoint.json'
        self._export()

    def _export(self):
        _write(self.endpoint, dict(self.frame, channel=str(self.channel.resolve()),
            gateway_url=self.gateway_url, owner_identity=self.owner_identity,
            gateway_identity=self.gateway_identity, native_gateway_argv=self.native_argv,
            deadline_monotonic=self.deadline, owner_uid=self.uid,
            active=self.deadline is not None and not self.unusable))
        return str(self.endpoint.resolve())

    def _remaining(self):
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise OwnedGatewayError('Original driver deadline exhausted')
        return left

    def _assert_parent(self):
        if socket.gethostname() != self.host or _boot_id() != self.boot:
            raise OwnedGatewayError('Head/kernel identity changed')
        if not self.process_tools.matches(self.owner_identity, self.process_tools.proc_identity(os.getpid())):
            raise OwnedGatewayError('Adapter is outside the original owner process')
        self.process_tools.assert_binding(_read(self.owner.roots, self.uid), self.frame['binding'])
        if self.owner.stop.exists() or self.owner.watcher.poll() is not None:
            raise OwnedGatewayError('Original ownership watcher is no longer running')

    def _gateway_tree(self):
        self._assert_parent()
        current = self.process_tools.proc_identity(self.gateway.pid)
        if not self.process_tools.matches(self.gateway_identity, current) or current['ppid'] != os.getpid():
            raise OwnedGatewayError('Gateway PID/start_ticks/UID or original parent changed')
        roots = _read(self.owner.roots, self.uid)['roots']
        if not any(self.process_tools.matches(self.gateway_identity, row) for row in roots):
            raise OwnedGatewayError('Gateway is not an explicitly registered original root')
        tree = self.process_tools.collect_owned([self.gateway_identity], {})
        # Only descendants of this one Gateway root may be passed to signalling.
        until = time.monotonic() + min(2, self._remaining())
        while True:
            registered = self.owner.registered()['identities']
            if all(self.process_tools.matches(row, registered.get(key)) for key, row in tree.items()):
                return tree
            if time.monotonic() >= until:
                raise OwnedGatewayError('Gateway descendant ownership registration is incomplete')
            time.sleep(0.05)

    def _workers(self):
        with urlopen(self.gateway_url + '/workers', timeout=min(5, self._remaining())) as response:
            if response.status != 200:
                raise OwnedGatewayError('Owned Gateway registry request failed')
            data = json.load(response)
        if not isinstance(data, dict) or not isinstance(data.get('workers'), list):
            raise OwnedGatewayError('Owned native registry schema is invalid')
        return data['workers']

    def _observed_policy(self):
        tree = self._gateway_tree()
        observations = []
        for row in tree.values():
            argv = _read_cmdline(row['pid'])
            if '--policy' in argv:
                if argv.count('--policy') != 1 or argv[argv.index('--policy') + 1] != self.policy:
                    raise OwnedGatewayError('Actual owned process native policy differs')
                observations.append({'identity': row, 'argv': argv, 'cpu_seconds_since_start': _cpu_seconds(row['pid'])})
        if not observations:
            raise OwnedGatewayError('No actual owned process argv confirms native policy')
        return observations

    def service_once(self):
        request_path = self.channel / 'request.json'
        if not request_path.exists():
            return None
        row = None
        try:
            row = _read(request_path, self.uid)
            sha = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            if sha == self.last_input_sha:
                return None
            self.last_input_sha = sha
            if set(row) != _REQUEST_KEYS or any(json.dumps(row[k], sort_keys=True) != json.dumps(self.frame[k], sort_keys=True) for k in _FRAME):
                raise OwnedGatewayError('Wrong binding/head/kernel/source or unauthorized request fields')
            if not re.fullmatch('[0-9a-f]{32}', row['request_id']) or row['policy'] not in POLICIES or row['operation'] not in ('restart', 'observe'):
                raise OwnedGatewayError('Only bounded native policy restart/observation requests are allowed')
            target = self.channel / (row['request_id'] + '.json')
            if target.exists():
                return _read(target, self.uid)  # Never execute an already completed request twice.
            if self.unusable:
                raise OwnedGatewayError('No retry after an unconfirmed owned restart')
            self._assert_parent(); self._remaining()
            started = time.monotonic(); cpu = resource.getrusage(resource.RUSAGE_SELF)
            if row['operation'] == 'restart':
                if self._remaining() < 5:
                    raise OwnedGatewayError('Insufficient original deadline for a safe native replacement')
                if self._workers():
                    raise OwnedGatewayError('Original lifecycle must remove every worker before CPU restart')
                tree = self._gateway_tree(); old = self.gateway_identity
                self._observed_policy()
                if not self.process_tools.terminate_identities(list(tree.values()), min(10, self._remaining()), probe=self.process_tools.proc_identity):
                    self.unusable = True
                    raise OwnedGatewayError('Owned Gateway processes did not exit')
                self.gateway.wait(timeout=min(2, self._remaining()))
                argv = list(self.native_argv); argv[argv.index('--policy') + 1] = row['policy']
                self.gateway = self.owner.launch(self.launch_argv(argv), self.channel / (row['request_id'] + '.log'))
                self.gateway_identity = self.process_tools.proc_identity(self.gateway.pid)
                self.native_argv, self.policy = argv, row['policy']
                self.last_restart = row['request_id']
                until = time.monotonic() + min(60, self._remaining())
                while True:
                    if self.gateway.poll() is not None:
                        self.unusable = True
                        raise OwnedGatewayError('Actual replacement Gateway exited')
                    try:
                        workers = self._workers()
                        observed = self._observed_policy()
                        if workers:
                            raise OwnedGatewayError('Replacement native Gateway is not empty')
                        break
                    except (OSError, ValueError, OwnedGatewayError):
                        if time.monotonic() >= until:
                            self.unusable = True
                            raise
                        time.sleep(0.05)
                result = {'status': 'REPLACED_OWNED_GATEWAY_EMPTY', 'old_gateway_pid_identity': old,
                          'old_owned_gateway_exited': True, 'empty_registry_confirmed': True}
            else:
                if row['policy'] != self.policy:
                    raise OwnedGatewayError('Observation does not refer to the current replacement')
                workers = self._workers(); observed = self._observed_policy()
                result = {'status': 'OBSERVED_OWNED_GATEWAY', 'workers': workers,
                          'replacement_request_id': self.last_restart}
            usage = resource.getrusage(resource.RUSAGE_SELF)
            result.update(self.frame, request_id=row['request_id'], actual_native_policy=self.policy,
                native_gateway_argv=list(self.native_argv), actual_process_argv=observed,
                new_gateway_pid_identity=self.gateway_identity, gateway_url=self.gateway_url,
                ownership_binding=dict(self.owner.binding), source_receipt=str(target.resolve()),
                elapsed_s=time.monotonic() - started, parent_cpu_user_s=usage.ru_utime-cpu.ru_utime,
                parent_cpu_system_s=usage.ru_stime-cpu.ru_stime, owner_observed_monotonic=time.monotonic(),
                remaining_driver_seconds=self._remaining(),
                independent_reset_receipt_id=self.last_restart)
            _write(target, result); self._export()
            return result
        except Exception as error:
            result = dict(self.frame, status='REJECTED_OR_FAILED_OWNED_REQUEST', error_type=type(error).__name__, reason=str(error))
            if isinstance(row, dict) and re.fullmatch('[0-9a-f]{32}', str(row.get('request_id', ''))):
                result['request_id'] = row['request_id']
                _write(self.channel / (row['request_id'] + '.json'), result)
            else:
                _write(self.channel / 'rejected.json', result)
            return result

    def wait_driver(self, driver, timeout):
        self.deadline = time.monotonic() + timeout  # Same start and deadline as the replaced Popen.wait.
        self._export()
        try:
            while driver.poll() is None:
                if time.monotonic() >= self.deadline:
                    raise subprocess.TimeoutExpired(driver.args, timeout)
                self.service_once()
                time.sleep(min(0.05, max(0, self.deadline-time.monotonic())))
            return driver.wait(timeout=max(0.001, self.deadline-time.monotonic()))
        finally:
            self.unusable = True; self._export()


class OwnedGatewayClient:
    """Constructed inside the Ray task; contains only a shared-file description."""
    def __init__(self, endpoint, *, source_sha256, gateway_url, timeout_s=90):
        self.endpoint = Path(endpoint); self.uid = os.getuid(); self.timeout_s = timeout_s
        self.description = _read(self.endpoint, self.uid)
        if self.description['source_sha256'] != source_sha256 or self.description['gateway_url'] != gateway_url.rstrip('/'):
            raise OwnedGatewayError('Owner endpoint differs from actual frozen source/Gateway')
        self.channel = Path(self.description['channel'])
        if (self.channel.is_symlink() or self.channel.stat().st_uid != self.uid or self.channel.stat().st_mode & 0o077 or
                self.endpoint.resolve().parent != self.channel.resolve()):
            raise OwnedGatewayError('Owner endpoint must be inside its private exact channel')
        if self.description['owner_uid'] != self.uid:
            raise OwnedGatewayError('Owner and Ray task must use the same actual UID')

    def _request(self, operation, policy):
        if policy not in POLICIES:
            raise OwnedGatewayError('Only original least_load/cache_aware policies are allowed')
        actual = _read(self.endpoint, self.uid)
        if not actual['active'] or any(actual[k] != self.description[k] for k in _FRAME):
            raise OwnedGatewayError('Original owned parent is inactive or its identity changed')
        request = {k: actual[k] for k in _FRAME}
        request.update(request_id=uuid.uuid4().hex, operation=operation, policy=policy)
        pending = self.channel / 'request.json'
        if pending.exists():
            previous = _read(pending, self.uid)
            if not (self.channel / (str(previous.get('request_id')) + '.json')).exists():
                raise OwnedGatewayError('A previous original-parent request has not completed; no overwrite/retry')
        _write(self.channel / 'request.json', request)
        target = self.channel / (request['request_id'] + '.json')
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if target.exists():
                result = _read(target, self.uid)
                if result.get('request_id') != request['request_id'] or any(result[k] != actual[k] for k in _FRAME):
                    raise OwnedGatewayError('Parent result does not match the exact request binding')
                if result['status'] not in ('REPLACED_OWNED_GATEWAY_EMPTY', 'OBSERVED_OWNED_GATEWAY'):
                    raise OwnedGatewayError(result.get('reason', 'Owned native policy operation failed'))
                return result
            time.sleep(0.05)
        raise OwnedGatewayError('No matching original parent response within the fixed control timeout; no retry')

    def __call__(self, policy, urls, actual_version):
        return self._request('restart', policy)

    def verify_after_registration(self, replacement):
        actual = self._request('observe', replacement['actual_native_policy'])
        if actual['replacement_request_id'] != replacement['request_id'] or actual['new_gateway_pid_identity'] != replacement['new_gateway_pid_identity']:
            raise OwnedGatewayError('Registered Gateway is not the exact new owned process')
        return actual

    def observe_original_budget(self):
        return self._request('observe', self.description['native_gateway_argv'][self.description['native_gateway_argv'].index('--policy') + 1])
