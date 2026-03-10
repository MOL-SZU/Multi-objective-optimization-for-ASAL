import os
import sys
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import glob
from natsort import natsorted

# 强行关闭显存预分配
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# 导入原有组件
import main_opt_moo
from optimizers import PymooOptimizer
from eval import get_batch_loss_fn
from pymoo.core.population import Population

# ================= 流程配置 =================
SAVE_DIR = "./data/results_pipeline_caterpillar_butterfly"
PROMPTS = ["a caterpillar", "a butterfly"]
SUBSTRATE = "lenia"

MOO_ITERS = 20000
POP_SIZE = 64
SAVE_EVERY = 100
# ============================================

# ------------------ 断点检测工具 ------------------
def find_latest_checkpoint():
    """检测现有的 checkpoint，返回 (起始代数, 初始种群参数)"""
    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return 0, None
    
    files = natsorted(glob.glob(os.path.join(ckpt_dir, "pareto_gen_*.pkl")))
    if not files:
        return 0, None
    
    last_file = files[-1]
    try:
        last_iter = int(os.path.basename(last_file).split('_')[-1].split('.')[0])
        with open(last_file, "rb") as f:
            data = pickle.load(f)
            last_X = data["X"]
        
        print(f"\n>>> [断点恢复] 检测到历史进度，将从第 {last_iter} 代继续运行...")
        return last_iter, last_X
    except Exception as e:
        print(f"读取Checkpoint失败: {e}")
        return 0, None

# ------------------ 工具函数：加载种子 ------------------
def load_existing_seeds():
    print(f"\n>>> 正在加载阶段 1 的种子解...")
    path_p1 = os.path.join(SAVE_DIR, "seed_caterpillar", "best.pkl")
    path_p2 = os.path.join(SAVE_DIR, "seed_butterfly", "best.pkl")

    if not os.path.exists(path_p1) or not os.path.exists(path_p2):
        print(f"错误：找不到种子文件！路径: {path_p1} 或 {path_p2}")
        sys.exit(1)

    with open(path_p1, "rb") as f:
        data1 = pickle.load(f)
        p1 = data1[0] if isinstance(data1, (tuple, list)) else data1

    with open(path_p2, "rb") as f:
        data2 = pickle.load(f)
        p2 = data2[0] if isinstance(data2, (tuple, list)) else data2

    print(f">>> 种子加载成功。")
    return [np.array(p1), np.array(p2)]
    
# ------------------ 核心保存逻辑 ------------------
def save_archive_step(iter_num, x_pop, scores):
    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    save_path = os.path.join(ckpt_dir, f"pop_gen_{iter_num:05d}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"X": x_pop, "S": scores}, f)

    latest_path = os.path.join(SAVE_DIR, "latest_pop.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump({"X": x_pop, "S": scores}, f)

# ------------------ 全局解的历史轨迹图 ------------------
def plot_all_evolution(global_archive):
    if not global_archive["S"]:
        return
    
    scores = np.array(global_archive["S"])
    gens = np.array(global_archive["gen"])
    ids = np.array(global_archive["id"])
    fig = go.Figure()

    # 层级 1：历史背景
    fig.add_trace(go.Scatter(
        x=scores[:, 0], y=scores[:, 1], mode='markers',
        marker=dict(size=4, color=gens, colorscale='Viridis', opacity=0.2, colorbar=dict(title="Generation")),
        text=[f"ID: {i}<br>Gen: {g}" for i, g in zip(ids, gens)],
        hoverinfo="text", name="All Evaluated Solutions"
    ))

    # 层级 2：初始种子
    gen0_mask = gens == 0
    if np.any(gen0_mask):
        fig.add_trace(go.Scatter(
            x=scores[gen0_mask, 0], y=scores[gen0_mask, 1], mode='markers',
            marker=dict(size=10, color='red', symbol='cross', line=dict(width=1, color='darkred')),
            name="Initial Seeds (Gen 0)"
        ))

    fig.update_layout(title="Evolution History", xaxis_title=f"Score: {PROMPTS[0]}", yaxis_title=f"Score: {PROMPTS[1]}", template="plotly_white")
    fig.write_html(os.path.join(SAVE_DIR, "evolution_history_interactive.html"))

# ------------------ 主流程 ------------------
def run_moo_with_seeds(seeds):
    jax.clear_caches()
    
    archive_path = os.path.join(SAVE_DIR, "global_archive.pkl")
    if os.path.exists(archive_path):
        print(">>> 检测到全局历史archive，正在加载...")
        with open(archive_path, "rb") as f:
            global_archive = pickle.load(f)
        global_id_counter = max(global_archive["id"]) + 1 if len(global_archive["id"]) > 0 else 0
    else:
        global_archive = {"X": [], "S": [], "gen": [], "id": []}
        global_id_counter = 0

    start_iter, resumed_X = find_latest_checkpoint()

    args = main_opt_moo.parse_args([])
    args.substrate = SUBSTRATE
    args.prompts = ";".join(PROMPTS)
    args.save_dir = SAVE_DIR

    rollout_fn, fm, substrate = main_opt_moo.setup_evaluator(args)
    eval_fn = get_batch_loss_fn(rollout_fn, fm, PROMPTS)
    opt = PymooOptimizer("nsga2", POP_SIZE, substrate.n_params, len(PROMPTS))

    # ================= 种群初始化 =================
    if resumed_X is not None:
        if len(resumed_X) >= POP_SIZE:
            X_init = resumed_X[:POP_SIZE]
        else:
            X_init = np.random.uniform(-1, 1, (POP_SIZE, substrate.n_params))
            X_init[:len(resumed_X)] = resumed_X

    else:
        X_init = np.random.uniform(-1, 1, (POP_SIZE, substrate.n_params))
        p1, p2 = seeds[0], seeds[1]
        num_per_seed, noise_std = 10, 0.005
        
        X_init[0] = p1
        X_init[num_per_seed] = p2
        
        for i in range(1, num_per_seed):
            X_init[i] = np.clip(p1 + np.random.normal(0, noise_std, p1.shape), -1, 1)
            X_init[num_per_seed + i] = np.clip(p2 + np.random.normal(0, noise_std, p2.shape), -1, 1)

    pop_init = Population.new("X", X_init)
    opt.algorithm.setup(opt.problem, sampling=pop_init)
    
    # ================= 评估并记录纯净种子分数 =================
    rng = jax.random.PRNGKey(42)
    rng, _rng = jax.random.split(rng)
    seed_scores, _ = eval_fn(_rng, np.stack([seeds[0], seeds[1]], axis=0))
    seed_scores = np.array(seed_scores)

    if start_iter == 0:
        for i in range(2):
            global_archive["X"].append(seeds[i])
            global_archive["S"].append(seed_scores[i])
            global_archive["gen"].append(0)
            global_archive["id"].append(global_id_counter)
            global_id_counter += 1
        print(f"\n>>> [初始评估] 纯净种子已作为 Gen 0 登记。得分: {seed_scores}")

    # ================= 主循环 =================
    for it in range(start_iter, MOO_ITERS):
        t0 = time.time()
        rng, _rng = jax.random.split(rng)

        x_pop = opt.ask()
        scores, _ = eval_fn(_rng, x_pop)
        scores = np.array(scores)
        opt.tell(x_pop, -scores)

        for i in range(len(x_pop)):
            global_archive["X"].append(x_pop[i])
            global_archive["S"].append(scores[i])
            global_archive["gen"].append(it)
            global_archive["id"].append(global_id_counter)
            global_id_counter += 1

        if (it + 1) % SAVE_EVERY == 0 or it == MOO_ITERS - 1:
            it_speed = 1.0 / max(1e-9, (time.time() - t0))
            
            max_sims = np.max(scores, axis=0)
            avg_sims = np.mean(scores, axis=0)
            print(f"Iter {it+1:4d} | Speed: {it_speed:.2f} it/s | Max: C={max_sims[0]:.3f}, B={max_sims[1]:.3f} | Mean: C={avg_sims[0]:.3f}, B={avg_sims[1]:.3f}")

            # 直接存下一整代的 64 个解，确保数据多样性不丢失
            save_archive_step(it + 1, x_pop, scores)

            with open(archive_path, "wb") as f:
                pickle.dump(global_archive, f)
            plot_all_evolution(global_archive)

    plot_all_evolution(global_archive)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    seeds = load_existing_seeds()
    run_moo_with_seeds(seeds)

if __name__ == "__main__":
    main()
