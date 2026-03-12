import os
import sys
import time
import glob
import pickle
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")

import plotly.graph_objects as go
from natsort import natsorted

# 强行关闭显存预分配
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import substrates
import foundation_models
from rollout import rollout_simulation
from pymoo.core.population import Population

from NSGAII_optimizers import PymooOptimizer
from eval_moo import get_batch_loss_fn
from moo_lenia_problem import (
    compute_group_bounds_from_seeds,
    sample_population_with_bounds,
    clip_population_with_bounds,
    build_seeded_initial_population,
    merge_resumed_with_random,
    summarize_bounds,
    diagnose_seed_edge_ratio,
)

# ================= 配置 =================
SAVE_DIR = "./data/results_pipeline_caterpillar_butterfly_moo"
PROMPTS = ["a caterpillar", "a butterfly"]
SUBSTRATE = "lenia"

POP_SIZE = 64
MOO_ITERS = 20000
SAVE_EVERY = 100

# 分组边界超参数
DYN_DIM = 45
DYN_MARGIN = 1.0
INIT_MARGIN = 2.0
DYN_MIN_SPAN = 4.0
INIT_MIN_SPAN = 8.0

# 初始化种群超参数
NUM_PER_SEED = 10
NOISE_STD = 0.05

# 阶段 1 的 seed 文件
SEED_PATHS = [
    "./data/results_pipeline_caterpillar_butterfly/seed_caterpillar/best.pkl",
    "./data/results_pipeline_caterpillar_butterfly/seed_butterfly/best.pkl",
]
# ======================================


# ================= evaluator setup =================
def setup_evaluator(substrate_name, prompts):
    fm = foundation_models.create_foundation_model("clip")
    substrate = substrates.create_substrate(substrate_name)
    substrate = substrates.FlattenSubstrateParameters(substrate)

    rollout_fn = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=fm,
        rollout_steps=substrate.rollout_steps,
        time_sampling=(max(1, len(prompts)), True),
        img_size=224,
        return_state=False,
    )
    return rollout_fn, fm, substrate


# ================= I/O helpers =================
def load_single_best(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Seed file not found: {path}")

    with open(path, "rb") as f:
        obj = pickle.load(f)

    # ASAL 的 best.pkl 通常保存为 (best_member, best_fitness)
    if isinstance(obj, (tuple, list)) and len(obj) >= 1:
        best_member = obj[0]
    else:
        best_member = obj

    return np.asarray(best_member, dtype=np.float32)


def load_existing_seeds(seed_paths):
    print("\n>>> 正在加载阶段 1 的种子解...")
    seeds = [load_single_best(p) for p in seed_paths]
    print(f">>> 种子加载成功，共 {len(seeds)} 个。")
    for i, s in enumerate(seeds):
        print(
            f"    seed[{i}] shape={s.shape}, "
            f"min={s.min():.6f}, max={s.max():.6f}, "
            f"mean(|x|>0.95)={np.mean(np.abs(s) > 0.95):.4%}"
        )
    return seeds


def find_latest_checkpoint():
    """
    返回:
        start_iter, resumed_X
    """
    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return 0, None

    files = natsorted(glob.glob(os.path.join(ckpt_dir, "pop_gen_*.pkl")))
    if not files:
        return 0, None

    last_file = files[-1]
    try:
        last_iter = int(os.path.basename(last_file).split("_")[-1].split(".")[0])
        with open(last_file, "rb") as f:
            data = pickle.load(f)
        last_X = np.asarray(data["X"], dtype=np.float32)

        print(f"\n>>> [断点恢复] 检测到历史进度，将从第 {last_iter} 代继续运行...")
        return last_iter, last_X
    except Exception as e:
        print(f">>> 读取 checkpoint 失败: {e}")
        return 0, None


def save_population_step(iter_num, x_pop, scores):
    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    save_path = os.path.join(ckpt_dir, f"pop_gen_{iter_num:05d}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"X": x_pop, "S": scores}, f)

    latest_path = os.path.join(SAVE_DIR, "latest_pop.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump({"X": x_pop, "S": scores}, f)


def save_archive(global_archive):
    archive_path = os.path.join(SAVE_DIR, "global_archive.pkl")
    with open(archive_path, "wb") as f:
        pickle.dump(global_archive, f)


def load_archive():
    archive_path = os.path.join(SAVE_DIR, "global_archive.pkl")
    if os.path.exists(archive_path):
        print(">>> 检测到全局历史 archive，正在加载...")
        with open(archive_path, "rb") as f:
            global_archive = pickle.load(f)
        global_id_counter = max(global_archive["id"]) + 1 if len(global_archive["id"]) > 0 else 0
    else:
        global_archive = {"X": [], "S": [], "gen": [], "id": []}
        global_id_counter = 0
    return global_archive, global_id_counter


# ================= visualization =================
def plot_all_evolution(global_archive):
    if len(global_archive["S"]) == 0:
        return

    scores = np.array(global_archive["S"])
    gens = np.array(global_archive["gen"])
    ids = np.array(global_archive["id"])

    if scores.ndim != 2 or scores.shape[1] < 2:
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=scores[:, 0],
        y=scores[:, 1],
        mode="markers",
        marker=dict(
            size=4,
            color=gens,
            colorscale="Viridis",
            opacity=0.25,
            colorbar=dict(title="Generation"),
        ),
        text=[f"ID: {i}<br>Gen: {g}" for i, g in zip(ids, gens)],
        hoverinfo="text",
        name="All Evaluated Solutions",
    ))

    gen0_mask = gens == 0
    if np.any(gen0_mask):
        fig.add_trace(go.Scatter(
            x=scores[gen0_mask, 0],
            y=scores[gen0_mask, 1],
            mode="markers",
            marker=dict(size=10, color="red", symbol="cross"),
            name="Initial Seeds (Gen 0)",
        ))

    fig.update_layout(
        title="Evolution History",
        xaxis_title=f"Score: {PROMPTS[0]}",
        yaxis_title=f"Score: {PROMPTS[1]}",
        template="plotly_white",
    )
    fig.write_html(os.path.join(SAVE_DIR, "evolution_history_interactive.html"))


# ================= main pipeline =================
def run_moo_with_seeds(seeds):
    jax.clear_caches()

    global_archive, global_id_counter = load_archive()
    start_iter, resumed_X = find_latest_checkpoint()

    rollout_fn, fm, substrate = setup_evaluator(SUBSTRATE, PROMPTS)
    eval_fn = get_batch_loss_fn(rollout_fn, fm, PROMPTS)

    # ---------- 计算边界 ----------
    xl, xu = compute_group_bounds_from_seeds(
        seeds=seeds,
        dyn_dim=DYN_DIM,
        dyn_margin=DYN_MARGIN,
        init_margin=INIT_MARGIN,
        dyn_min_span=DYN_MIN_SPAN,
        init_min_span=INIT_MIN_SPAN,
    )

    bound_stats = summarize_bounds(xl, xu, dyn_dim=DYN_DIM)
    print("\n>>> [边界统计]")
    for k, v in bound_stats.items():
        print(f"    {k}: {v}")

    edge_reports = diagnose_seed_edge_ratio(seeds, xl, xu, atol_ratio=0.02)
    print("\n>>> [Seed 贴边诊断]")
    for r in edge_reports:
        print(
            f"    seed[{r['seed_index']}]: "
            f"min={r['min']:.6f}, max={r['max']:.6f}, "
            f"near_low={r['near_low_ratio']:.4%}, "
            f"near_high={r['near_high_ratio']:.4%}, "
            f"near_edge={r['near_edge_ratio']:.4%}"
        )

    # ---------- 创建优化器 ----------
    opt = PymooOptimizer(
        algo_name="nsga2",
        pop_size=POP_SIZE,
        num_dims=substrate.n_params,
        num_objs=len(PROMPTS),
        xl=xl,
        xu=xu,
    )

    print("\n>>> [Problem 摘要]")
    for k, v in opt.get_problem_summary().items():
        print(f"    {k}: {v}")

    # ---------- 初始化种群 ----------
    if resumed_X is not None:
        X_init = merge_resumed_with_random(
            resumed_X=resumed_X,
            pop_size=POP_SIZE,
            xl=xl,
            xu=xu,
        )
        print(">>> 使用 checkpoint 恢复种群初始化。")
    else:
        X_init = build_seeded_initial_population(
            seeds=seeds,
            pop_size=POP_SIZE,
            xl=xl,
            xu=xu,
            num_per_seed=NUM_PER_SEED,
            noise_std=NOISE_STD,
        )
        print(">>> 使用 seed 引导初始化种群。")

    X_init = clip_population_with_bounds(X_init, xl, xu)

    pop_init = Population.new("X", X_init)
    opt.algorithm.setup(opt.problem, sampling=pop_init)

    # ---------- 初始纯净 seed 评估并记入 Gen 0 ----------
    rng = jax.random.PRNGKey(42)
    if start_iter == 0:
        rng_seed_eval = jax.random.split(rng, len(seeds))
        seed_scores, _ = eval_fn(rng_seed_eval, np.stack(seeds, axis=0))
        seed_scores = np.array(seed_scores)

        for i in range(len(seeds)):
            global_archive["X"].append(np.array(seeds[i]))
            global_archive["S"].append(seed_scores[i])
            global_archive["gen"].append(0)
            global_archive["id"].append(global_id_counter)
            global_id_counter += 1

        print(f"\n>>> [初始评估] 纯净种子已作为 Gen 0 登记。")
        for i, sc in enumerate(seed_scores):
            print(f"    seed[{i}] scores = {sc}")

        save_archive(global_archive)
        plot_all_evolution(global_archive)

    # ---------- 主循环 ----------
    for it in range(start_iter, MOO_ITERS):
        t0 = time.time()

        x_pop = np.array(opt.ask(), dtype=np.float32)

        rng, step_rng = jax.random.split(rng)
        rng_batch = jax.random.split(step_rng, len(x_pop))

        scores, aux = eval_fn(rng_batch, x_pop)
        scores = np.array(scores, dtype=np.float32)

        # pymoo 默认最小化，所以传负号
        opt.tell(x_pop, -scores)

        for i in range(len(x_pop)):
            global_archive["X"].append(np.array(x_pop[i]))
            global_archive["S"].append(np.array(scores[i]))
            global_archive["gen"].append(it + 1)
            global_archive["id"].append(global_id_counter)
            global_id_counter += 1

        if (it + 1) % SAVE_EVERY == 0 or it == MOO_ITERS - 1:
            dt = time.time() - t0
            it_speed = 1.0 / max(1e-9, dt)

            max_sims = np.max(scores, axis=0)
            avg_sims = np.mean(scores, axis=0)

            edge_ratio = np.mean(
                np.logical_or(
                    np.isclose(x_pop, xl[None, :], atol=(xu - xl)[None, :] * 0.02),
                    np.isclose(x_pop, xu[None, :], atol=(xu - xl)[None, :] * 0.02),
                )
            )

            print(
                f"Iter {it+1:5d} | Speed: {it_speed:.2f} it/s | "
                f"Max: {', '.join([f'{PROMPTS[j]}={max_sims[j]:.3f}' for j in range(len(PROMPTS))])} | "
                f"Mean: {', '.join([f'{PROMPTS[j]}={avg_sims[j]:.3f}' for j in range(len(PROMPTS))])} | "
                f"EdgeRatio: {edge_ratio:.4%}"
            )

            save_population_step(it + 1, x_pop, scores)
            save_archive(global_archive)
            plot_all_evolution(global_archive)

    plot_all_evolution(global_archive)
    save_archive(global_archive)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    seeds = load_existing_seeds(SEED_PATHS)
    run_moo_with_seeds(seeds)


if __name__ == "__main__":
    main()
