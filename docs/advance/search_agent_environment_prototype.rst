Search Agent Environment Prototype
==================================

Last updated: 07/13/2026.

Positioning
-----------

This prototype adds a first-stage search-oriented agent loop to VeRL's
experimental ``agent_loop`` package.

It is intentionally scoped to the M0/M1 boundary only:

- real interface audit against current VeRL code
- versioned Search Environment Contract
- deterministic mock / snapshot replay search adapters
- minimal multi-turn ``SearchAgentLoop``
- trajectory metadata and a deterministic baseline reward evaluator

It is **not** a full Search RL training system yet. In particular, this stage
does **not** implement real PPO/GRPO search training, partial rollout KV resume,
retrieval KV runtime reuse, SGLang kernel changes, or internal search service
integration.

How this differs from a normal Search-R1-style reproduction
-----------------------------------------------------------

The focus here is not just adding a ``<search>`` tag. The prototype adds a
versioned environment contract and replayable environment observations so later
reward and runtime work can remain reproducible.

Implemented components
----------------------

Search Environment Contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented in ``verl.experimental.agent_loop.search_environment``.

Key schemas:

- ``SearchAction``
- ``SearchDocument``
- ``SearchCost``
- ``SearchObservation``
- ``SearchTrace``
- ``SearchSnapshotRecord``

Contract properties:

- explicit schema versioning
- stable content hashing via SHA-256
- stable observation response hashing
- structured invalid-action / timeout / failure observations
- deterministic action identity for replay lookup

Mock and snapshot modes
~~~~~~~~~~~~~~~~~~~~~~~

Adapters implemented:

- ``InMemorySearchAdapter``: deterministic lexical / explicit-map mock backend
- ``SnapshotReplaySearchAdapter``: replay from JSON / JSONL snapshot records

M2 minimal additions in this branch:

- ``RecordingSearchAdapter``: wraps another adapter and records action /
  observation pairs into snapshot records
- ``SearchSnapshotStore``: in-memory snapshot collection for record workflows
- ``load_snapshot_store(...)``: lightweight loader for JSON / JSONL snapshots
- ``SearchAgentLoop`` config-driven record mode via
  ``rollout.custom.search_agent.record_mode`` and related snapshot settings

Snapshot replay is strict:

- schema version mismatch returns a structured error observation
- index version mismatch returns a structured error observation
- corrupted response hashes are rejected
- misses do not silently replay the wrong record

Agent loop data flow
--------------------

``SearchAgentLoop`` lives in ``verl.experimental.agent_loop.search_agent_loop``
and follows the current VeRL ``AgentLoopBase`` contract.

Minimal flow:

1. receive raw prompt messages
2. apply chat template to prompt ids
3. call ``LLMServerClient.generate`` with token-in / token-out generation
4. parse exactly one ``<search>{...}</search>`` or ``<answer>...</answer>`` action
5. if ``search``: validate action, call async search adapter, render observation
6. append rendered observation tokens with ``response_mask=0``
7. append LLM-generated action tokens with ``response_mask=1``
8. continue until answer or structured stop

Mask semantics
--------------

The semantics are the same as current VeRL multi-turn agent loops:

- ``response_mask = 1``: policy action tokens produced by the LLM
- ``response_mask = 0``: environment / search observation tokens

This means environment tokens do not participate in actor loss. It does **not**
mean the model is prevented from using or quoting search results.

Trajectory metadata
-------------------

The prototype keeps compatibility with current ``AgentLoopOutput`` by storing
search metadata in ``extra_fields`` rather than changing core training schemas.

Recorded fields include:

- ``trajectory_id`` / ``request_id``
- ``search_traces`` and summarized ``search_trace_summary``
- per-turn query / top_k / recall_profile
- doc ids and content hashes
- index version and response hash
- latency and ``SearchCost``
- degraded / error states
- ``num_searches`` and ``stop_reason``
- ``renderer_version``
- action-token and environment-token ranges

Baseline reward components
--------------------------

``verl.experimental.agent_loop.search_reward`` adds a deterministic evaluator
over recorded trajectory metadata.

Components:

- ``answer_correctness``
- ``groundedness``
- ``retrieval_utility``
- ``search_latency_cost``
- ``search_resource_cost``
- ``invalid_action_penalty``
- ``failure_penalty``
- ``format_penalty``

This is reward infrastructure scaffolding only. It is not a learned reward
model and not a PPO/GRPO algorithm change.

M2 reward-infrastructure extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The reward module now also provides deterministic batch helpers:

- ``evaluate_search_reward_batch(...)``
- ``summarize_search_reward_batch(...)``

These helpers aggregate the same per-trajectory component schema without
introducing any training-loop changes.

Run tests
---------

Example commands used for this prototype:

.. code-block:: bash

   python3 -m pytest tests/experimental/agent_loop/test_search_environment_on_cpu.py
   python3 -m pytest tests/experimental/agent_loop/test_search_agent_loop_on_cpu.py
   python3 -m pytest tests/experimental/agent_loop/test_search_reward_on_cpu.py

Testing TODO / blocked validation
---------------------------------

The following validation work remains mandatory before this prototype can be
reported as fully passing M1 verification:

1. Create a local Python environment with the repo runtime/test dependencies
   required by VeRL's agent loop stack, including at least:

   - ``pytest``
   - ``pytest-asyncio``
   - ``ray``
   - ``torch``
   - ``tensordict``
   - ``transformers``
   - ``codetiming``

2. Re-run the new CPU tests and record the real outcome:

   - ``tests/experimental/agent_loop/test_search_environment_on_cpu.py``
   - ``tests/experimental/agent_loop/test_search_agent_loop_on_cpu.py``
   - ``tests/experimental/agent_loop/test_search_reward_on_cpu.py``

3. Run the dry-run harness successfully:

   .. code-block:: bash

      PYTHONPATH=/path/to/verl python3 examples/tutorial/agent_loop_get_started/search_agent_dry_run.py

4. Run at least one relevant regression test from the existing agent-loop suite
   after the dependency environment is ready.

5. Run repository-standard lint / formatter / static validation on the modified
   files and record the exact commands and results.

6. Run ``git diff --check`` and record the final ``git status --short --branch``
   and ``git diff --stat`` outputs.

Until the above items are completed with real command output, the implementation
should be described as an ``environment_prototype`` code drop with validation
still in progress, not as a fully verified closed-loop search RL system.

Run minimal dry-run example
---------------------------

.. code-block:: bash

   python3 examples/tutorial/agent_loop_get_started/search_agent_dry_run.py

The example uses:

- a fake token-in / token-out LLM server client
- an in-memory search adapter
- the real ``SearchAgentLoop`` implementation
- the baseline reward evaluator

Current limitations
-------------------

- no real online search service integration
- no batch reward pipeline beyond deterministic local evaluation
- no KV pause / resume integration
- no retrieval-conditioned KV runtime reuse
- no SGLang runtime or kernel modifications
- text protocol uses explicit ``<search>`` / ``<answer>`` blocks instead of a
  native structured search tool schema because the current repo ships only the
  generic tool parsers

Next stages (M2-M6)
-------------------

Planned next work:

- M2: snapshot recording workflows and richer reward infrastructure
- M3: small-scale search RL closed-loop training validation
- M4: partial rollout / search-wait KV lifecycle integration
- M5: retrieval-KV identity integration experiments
- M6: higher-fidelity or real remote search backend validation
