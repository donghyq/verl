#!/usr/bin/env python3
# Copyright 2026 Individual Contributor: donghyq
# SPDX-License-Identifier: Apache-2.0
"""Validate exported review files without importing project code or using CUDA."""
import ast
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text())
failures = []
python_files = 0
patterns = {
    'private_home': r'/(?:Users/[^/\s]+|home/hadoop[^/\s]*)',
    'private_mount': r'/mnt/' + r'dolphinfs/',
    'internal_domain': r'(?i)(?:sankuai|meituan)\.com',
    'platform_application': r'\bpsx[a-z0-9]{8,}\b',
    'private_ip': r'(?<![\d.])(?:10\.\d+|172\.(?:1[6-9]|2\d|3[01])|192\.168|33\.29)\.\d+\.\d+(?![\d.])',
    'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b',
    'aws_access_key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
}
expected = {'SOURCE_MANIFEST.json'}
for record in manifest['files']:
    name = record['path']
    rel = pathlib.PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or name in expected:
        failures.append([name, 'unsafe or duplicate manifest path'])
        continue
    expected.add(name)
    target = ROOT / name
    if target.is_symlink() or not target.is_file():
        failures.append([name, 'missing or nonregular file'])
        continue
    raw = target.read_bytes()
    if len(raw) != record['bytes'] or hashlib.sha256(raw).hexdigest() != record['public_sha256']:
        failures.append([name, 'hash/length mismatch'])
    if record['source'] not in manifest['sources']:
        failures.append([name, 'unknown source'])
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        failures.append([name, 'unexpected binary'])
        continue
    if target.suffix == '.py':
        python_files += 1
        try:
            ast.parse(text, filename=name)
        except SyntaxError as error:
            failures.append([name, f'syntax line {error.lineno}'])
    for label, pattern in patterns.items():
        if re.search(pattern, text):
            failures.append([name, label])
actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
          and '.git' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts}
for name in sorted(actual - expected):
    failures.append([name, 'unlisted file'])
for name in ('README.md', 'REVIEW_STATUS.md', 'THIRD_PARTY_NOTICES.md'):
    for url in re.findall(r'\]\(([^)]+)\)', (ROOT / name).read_text()):
        if '://' not in url and not url.startswith('#') and not (ROOT / url.split('#')[0]).exists():
            failures.append([name, f'broken review link {url}'])
result = {'status': 'PASS' if not failures else 'FAIL', 'files': len(expected),
          'python_syntax_files': python_files, 'failures': failures,
          'runtime_tests': 'NOT_RUN', 'model_or_service_calls': 0}
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(bool(failures))
