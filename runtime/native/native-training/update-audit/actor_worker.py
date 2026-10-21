"""Select the one original training worker for either native resource layout."""


def resolve_training_worker(worker):
    workers = getattr(worker, "worker_dict", None)
    if workers is None:
        selected = worker
    else:
        if not isinstance(workers, dict):
            raise ValueError("Native worker roles are not a mapping")
        candidates = [workers[role] for role in ("actor", "actor_rollout", "actor_rollout_ref") if role in workers]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one original training role")
        selected = candidates[0]
        outer_rank, inner_rank = getattr(worker, "rank", None), getattr(selected, "rank", None)
        if outer_rank is not None and inner_rank is not None and outer_rank != inner_rank:
            raise ValueError("Native outer and training worker ranks differ")
    actor = getattr(selected, "actor", None)
    engine = getattr(actor, "engine", None) if actor is not None else getattr(selected, "engine", None)
    if engine is None or not hasattr(engine, "optimizer"):
        raise ValueError("Selected role does not own the original training optimizer")
    return selected


def observe_original_parameters(worker):
    """Reuse the existing native probe after selecting the actual role."""
    from run_v1_functional_gate import _parameter_probe

    return _parameter_probe(resolve_training_worker(worker))
