import numpy as np


def split_lenia_groups(n_params, dyn_dim=45):
    if dyn_dim <= 0 or dyn_dim >= n_params:
        raise ValueError(f"dyn_dim must be in (0, {n_params}), got {dyn_dim}")
    return slice(0, dyn_dim), slice(dyn_dim, n_params)


def _ensure_2d_seed_array(seeds):
    if isinstance(seeds, np.ndarray):
        if seeds.ndim == 1:
            seeds = seeds[None, :]
        elif seeds.ndim != 2:
            raise ValueError(f"seeds ndarray must be 1D or 2D, got shape {seeds.shape}")
        return seeds.astype(np.float32)

    if not isinstance(seeds, (list, tuple)) or len(seeds) == 0:
        raise ValueError("seeds must be a non-empty list/tuple of arrays or a numpy array")

    arrs = [np.asarray(s, dtype=np.float32) for s in seeds]
    first_shape = arrs[0].shape
    for i, a in enumerate(arrs):
        if a.shape != first_shape:
            raise ValueError(f"All seeds must have same shape, seed[0]={first_shape}, seed[{i}]={a.shape}")
    return np.stack(arrs, axis=0)


def _ensure_min_span(xl, xu, min_span):
    xl = np.asarray(xl, dtype=np.float32)
    xu = np.asarray(xu, dtype=np.float32)

    center = 0.5 * (xl + xu)
    half_span = 0.5 * (xu - xl)
    half_span = np.maximum(half_span, min_span / 2.0)

    new_xl = center - half_span
    new_xu = center + half_span
    return new_xl.astype(np.float32), new_xu.astype(np.float32)


def compute_group_bounds_from_seeds(
    seeds,
    dyn_dim=45,
    dyn_margin=0.5,
    init_margin=1.0,
    dyn_min_span=2.0,
    init_min_span=4.0,
):
    seed_stack = _ensure_2d_seed_array(seeds)
    n_seed, n_params = seed_stack.shape

    if n_seed < 1:
        raise ValueError("At least one seed is required")

    dyn_slice, init_slice = split_lenia_groups(n_params, dyn_dim=dyn_dim)

    xl = np.empty(n_params, dtype=np.float32)
    xu = np.empty(n_params, dtype=np.float32)

    # dyn group
    dyn_min = np.min(seed_stack[:, dyn_slice], axis=0) - dyn_margin
    dyn_max = np.max(seed_stack[:, dyn_slice], axis=0) + dyn_margin
    dyn_xl, dyn_xu = _ensure_min_span(dyn_min, dyn_max, dyn_min_span)

    # init group
    init_min = np.min(seed_stack[:, init_slice], axis=0) - init_margin
    init_max = np.max(seed_stack[:, init_slice], axis=0) + init_margin
    init_xl, init_xu = _ensure_min_span(init_min, init_max, init_min_span)

    xl[dyn_slice], xu[dyn_slice] = dyn_xl, dyn_xu
    xl[init_slice], xu[init_slice] = init_xl, init_xu

    if not np.all(xl < xu):
        raise ValueError("Invalid bounds: some dimensions have xl >= xu")

    return xl.astype(np.float32), xu.astype(np.float32)

#***************************************************************************************************************************
def sample_population_with_bounds(pop_size, xl, xu):#需要加载初始的种群
    xl = np.asarray(xl, dtype=np.float32)
    xu = np.asarray(xu, dtype=np.float32)

    if xl.shape != xu.shape:
        raise ValueError(f"xl and xu must have same shape, got {xl.shape} vs {xu.shape}")
    if xl.ndim != 1:
        raise ValueError(f"xl/xu must be 1D arrays, got ndim={xl.ndim}")
    if not np.all(xl < xu):
        raise ValueError("Each dimension must satisfy xl < xu")

    X = np.random.uniform(low=xl, high=xu, size=(pop_size, xl.shape[0]))
    return X.astype(np.float32)


def clip_population_with_bounds(X, xl, xu):
    X = np.asarray(X, dtype=np.float32)
    xl = np.asarray(xl, dtype=np.float32)
    xu = np.asarray(xu, dtype=np.float32)

    if xl.shape != xu.shape:
        raise ValueError(f"xl and xu must have same shape, got {xl.shape} vs {xu.shape}")
    if X.shape[-1] != xl.shape[0]:
        raise ValueError(f"Last dim of X must match bounds dim, got X.shape={X.shape}, bounds={xl.shape}")

    return np.clip(X, xl, xu).astype(np.float32)


def summarize_bounds(xl, xu, dyn_dim=45):
    xl = np.asarray(xl, dtype=np.float32)
    xu = np.asarray(xu, dtype=np.float32)

    dyn_slice, init_slice = split_lenia_groups(len(xl), dyn_dim=dyn_dim)
    span = xu - xl

    return {
        "n_params": int(len(xl)),
        "global_xl_min": float(np.min(xl)),
        "global_xl_max": float(np.max(xl)),
        "global_xu_min": float(np.min(xu)),
        "global_xu_max": float(np.max(xu)),
        "global_span_mean": float(np.mean(span)),
        "global_span_min": float(np.min(span)),
        "global_span_max": float(np.max(span)),
        "dyn_span_mean": float(np.mean(span[dyn_slice])),
        "dyn_span_min": float(np.min(span[dyn_slice])),
        "dyn_span_max": float(np.max(span[dyn_slice])),
        "init_span_mean": float(np.mean(span[init_slice])),
        "init_span_min": float(np.min(span[init_slice])),
        "init_span_max": float(np.max(span[init_slice])),
    }


def build_seeded_initial_population(
    seeds,
    pop_size,
    xl,
    xu,
    num_per_seed=10,
    noise_std=0.05,
):
    
    seed_stack = _ensure_2d_seed_array(seeds)
    n_seed, n_params = seed_stack.shape

    if pop_size <= 0:
        raise ValueError(f"pop_size must be positive, got {pop_size}")
    if num_per_seed <= 0:
        raise ValueError(f"num_per_seed must be positive, got {num_per_seed}")

    X_init = sample_population_with_bounds(pop_size, xl, xu)

    max_slots = n_seed * num_per_seed
    usable_slots = min(pop_size, max_slots)

    for seed_idx in range(n_seed):
        base_pos = seed_idx * num_per_seed
        if base_pos >= pop_size:
            break

        p = seed_stack[seed_idx]

        # 放入纯 seed
        X_init[base_pos] = clip_population_with_bounds(p, xl, xu)

        # 放入扰动样本
        for k in range(1, num_per_seed):
            pos = base_pos + k
            if pos >= usable_slots or pos >= pop_size:
                break
            perturbed = p + np.random.normal(0.0, noise_std, size=p.shape).astype(np.float32)
            X_init[pos] = clip_population_with_bounds(perturbed, xl, xu)

    return X_init.astype(np.float32)


def merge_resumed_with_random(resumed_X, pop_size, xl, xu):
    resumed_X = np.asarray(resumed_X, dtype=np.float32)
    if resumed_X.ndim != 2:
        raise ValueError(f"resumed_X must be 2D, got shape {resumed_X.shape}")

    n_existing, n_params = resumed_X.shape
    if n_existing == 0:
        return sample_population_with_bounds(pop_size, xl, xu)

    if n_existing >= pop_size:
        X_init = resumed_X[:pop_size].copy()
    else:
        X_init = sample_population_with_bounds(pop_size, xl, xu)
        X_init[:n_existing] = resumed_X

    X_init = clip_population_with_bounds(X_init, xl, xu)
    return X_init.astype(np.float32)


def diagnose_seed_edge_ratio(seeds, xl, xu, atol_ratio=0.02):
    seed_stack = _ensure_2d_seed_array(seeds)
    xl = np.asarray(xl, dtype=np.float32)
    xu = np.asarray(xu, dtype=np.float32)

    span = xu - xl
    atol = span * atol_ratio

    reports = []
    for i, p in enumerate(seed_stack):
        near_low = np.abs(p - xl) <= atol
        near_high = np.abs(p - xu) <= atol
        near_edge = np.logical_or(near_low, near_high)

        reports.append({
            "seed_index": i,
            "min": float(np.min(p)),
            "max": float(np.max(p)),
            "near_low_ratio": float(np.mean(near_low)),
            "near_high_ratio": float(np.mean(near_high)),
            "near_edge_ratio": float(np.mean(near_edge)),
        })

    return reports
