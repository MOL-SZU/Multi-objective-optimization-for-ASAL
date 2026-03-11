import os
import sys
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
# --- 环境适配：解决服务器无显示器报错 ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import re
from natsort import natsorted

# 强行关闭显存预分配
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import comocma

# 导入原有组件 (请确保这些文件在你的路径下)
import main_opt_moo
from eval import get_batch_loss_fn

# ================= 流程配置 =================
SAVE_DIR = "./data/results_pipeline_caterpillar_butterfly"
PROMPTS = ["a caterpillar", "a butterfly"]
SUBSTRATE_NAME = "lenia"

MOO_ITERS = 20000 
POP_SIZE = 64
NUM_KERNELS = 8
SIGMA = 0.15
SAVE_EVERY = 100
# ============================================

# ------------------ 工具函数：加载种子 ------------------
def load_existing_seeds():
    print(f"\n>>> 正在加载阶段 1 的种子解...")
    path_p1 = os.path.join(SAVE_DIR, "seed_caterpillar", "best.pkl")
    path_p2 = os.path.join(SAVE_DIR, "seed_butterfly", "best.pkl")

    if not os.path.exists(path_p1) or not os.path.exists(path_p2):
        print(f"Crit-Error: 种子丢失！请检查: {path_p1} 或 {path_p2}")
        sys.exit(1)

    def _extract(p):
        with open(p, "rb") as f:
            d = pickle.load(f)
        return np.array(d["params"] if isinstance(d, dict) else (d[0] if isinstance(d, (list, tuple)) else d))

    print(f">>> 种子加载成功。")
    return [_extract(path_p1), _extract(path_p2)]

# ------------------ 目标空间：非支配判断 ------------------
def dominates(a, b, eps=0.0):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a >= b - eps) and np.any(a > b + eps))

def non_dominated_mask(F, eps=0.0):
    F = np.asarray(F)
    n = F.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]: continue
        for j in range(n):
            if i == j or not mask[i]: continue
            if dominates(F[j], F[i], eps=eps):
                mask[i] = False
    return mask

# ------------------ 核心保存逻辑 ------------------
def save_checkpoint(moes, step):
    """保存优化器的全量状态 (原子写入防损坏)"""
    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"moo_state_step_{step:05d}.pkl")
    
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(moes, f)
    os.replace(tmp_path, path)
    
    all_ckpts = natsorted(glob.glob(os.path.join(ckpt_dir, "moo_state_step_*.pkl")))
    for old_f in all_ckpts[:-2]:
        os.remove(old_f)

def save_archive_step(iter_num, global_archive):
    """保存全局 Pareto 数据 (原子写入防损坏)"""
    ckpt_dir = os.path.join(SAVE_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 从全局档案中提取当前的所有点以计算前沿
    all_F = np.array(global_archive["S"])
    all_X = np.array(global_archive["X"])
    
    if len(all_F) == 0:
        return
        
    nd_mask = non_dominated_mask(all_F)
    X_nd, F_nd = all_X[nd_mask], all_F[nd_mask]
    
    # 1. 保存 Pareto .pkl 数据
    save_path = os.path.join(ckpt_dir, f"pareto_gen_{iter_num:05d}.pkl")
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"X": X_nd, "S": F_nd}, f)
    os.replace(tmp_path, save_path)
    
    # 更新 latest 指针
    latest_path = os.path.join(SAVE_DIR, "latest_pareto.pkl")
    tmp_latest = latest_path + ".tmp"
    with open(tmp_latest, "wb") as f:
        pickle.dump({"X": X_nd, "S": F_nd}, f)
    os.replace(tmp_latest, latest_path)

# ------------------ 全局解的历史轨迹图 (支持交互式和静态图) ------------------
# ------------------ 全局解的历史轨迹图 (支持交互式和静态图) ------------------
def plot_all_evolution(global_archive):
    """绘制分层渲染的演化图：包含原始种群、背景点、最终前沿与优化路径"""
    if not global_archive["S"]:
        return
        
    scores = np.array(global_archive["S"])
    gens = np.array(global_archive["gen"])
    ids = np.array(global_archive["id"])
    
    fig = go.Figure()

    # ---------- 层级 1：所有历史记录（作为半透明背景） ----------
    fig.add_trace(go.Scatter(
        x=scores[:, 0], 
        y=scores[:, 1],
        mode='markers',
        marker=dict(
            size=4,
            color=gens,
            colorscale='Viridis',
            opacity=0.2,  # 调低透明度，防止遮挡重要信息
            colorbar=dict(title="Generation")
        ),
        text=[f"ID: {i}<br>Gen: {g}" for i, g in zip(ids, gens)],
        hoverinfo="text",
        name="All Evaluated Solutions",
        showlegend=False
    ))

    # ---------- 层级 2：原始种群 / 种子起点 (Gen == 0) ----------
    gen0_mask = gens == 0
    if np.any(gen0_mask):
        fig.add_trace(go.Scatter(
            x=scores[gen0_mask, 0], 
            y=scores[gen0_mask, 1],
            mode='markers',
            marker=dict(size=10, color='red', symbol='cross', line=dict(width=1, color='darkred')),
            name="Initial Seeds (Gen 0)",
            hoverinfo="x+y"
        ))

    # ---------- 层级 3：最终代的 Pareto 前沿 ----------
    final_gen = np.max(gens)
    final_mask = gens == final_gen
    if np.any(final_mask):
        fig.add_trace(go.Scatter(
            x=scores[final_mask, 0], 
            y=scores[final_mask, 1],
            mode='markers',
            marker=dict(size=8, color='orange', symbol='diamond', line=dict(width=1, color='darkorange')),
            name=f"Final Front (Gen {final_gen})",
            text=[f"ID: {i}" for i in ids[final_mask]],
            hoverinfo="text+x+y"
        ))

    # ---------- 层级 4：进化轨迹 / 优化路径 ----------
    # 每隔一定代数，计算种群中表现最好的前10个个体的中心位置，连成线
    path_x, path_y = [], []
    step_size = max(1, final_gen // 15)  # 采样15个节点绘制路径
    
    for g in range(0, final_gen + 1, step_size):
        mask = gens == g
        if np.any(mask):
            g_scores = scores[mask]
            # 用两者得分之和作为简单的衡量标准，选出当前代最靠右上方的几个点
            top_idx = np.argsort(-(g_scores[:,0] + g_scores[:,1]))[:10]
            top_scores = g_scores[top_idx]
            path_x.append(np.mean(top_scores[:, 0]))
            path_y.append(np.mean(top_scores[:, 1]))
            
    # 加上最后一代的中心点保证路径完整
    if final_gen % step_size != 0:
        final_scores = scores[final_mask]
        top_idx = np.argsort(-(final_scores[:,0] + final_scores[:,1]))[:10]
        path_x.append(np.mean(final_scores[top_idx, 0]))
        path_y.append(np.mean(final_scores[top_idx, 1]))

    fig.add_trace(go.Scatter(
        x=path_x, 
        y=path_y,
        mode='lines+markers',
        line=dict(color='black', width=3, dash='dash'),
        marker=dict(size=8, color='black', symbol='arrow-right'),
        name="Optimization Trajectory"
    ))

    # ---------- 图表排版 ----------
    fig.update_layout(
        title="Enhanced Evolution History: Seeds, Trajectory, and Final Front",
        xaxis_title=f"Score: {PROMPTS[0]}",
        yaxis_title=f"Score: {PROMPTS[1]}",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white"
    )
    
    html_path = os.path.join(SAVE_DIR, "evolution_history_interactive.html")
    fig.write_html(html_path)
    print(f">>> 增强版交互图已生成: {html_path}")

# ------------------ 主流程 ------------------
def run_moo_comocma():
    print(f"[Log] 工作目录: {os.getcwd()}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    seeds = load_existing_seeds()
    p1, p2 = seeds[0], seeds[1]

    args = main_opt_moo.parse_args([])
    args.substrate, args.prompts, args.save_dir = SUBSTRATE_NAME, ";".join(PROMPTS), SAVE_DIR
    rollout_fn, fm, substrate = main_opt_moo.setup_evaluator(args)
    eval_fn = get_batch_loss_fn(rollout_fn, fm, PROMPTS)
    
    rng_key = jax.random.PRNGKey(42)
    seed_scores, _ = eval_fn(rng_key, np.stack([p1, p2], axis=0))
    seed_scores = np.array(seed_scores)
    print(f"\n>>> [初始评估] 种子分数 (Caterpillar, Butterfly): {seed_scores}")

    # ---------------- 初始化/读取全局档案 ----------------
    archive_path = os.path.join(SAVE_DIR, "global_archive.pkl")
    if os.path.exists(archive_path):
        print(">>> 检测到全局历史档案，正在加载...")
        with open(archive_path, "rb") as f:
            global_archive = pickle.load(f)
        global_id_counter = max(global_archive["id"]) + 1 if len(global_archive["id"]) > 0 else 0
    else:
        # X: 参数, S: 适应度分数, gen: 所在代数, id: 唯一编号用于对应图片
        global_archive = {"X": [], "S": [], "gen": [], "id": []}
        global_id_counter = 0

    # ---------- 魂：断点续传逻辑 (COMO-CMAES专属) ----------
    moes = None
    start_step = 0
    existing_states = natsorted(glob.glob(os.path.join(SAVE_DIR, "checkpoints", "moo_state_step_*.pkl")))
    
    if existing_states:
        latest_state = existing_states[-1]
        start_step = int(re.search(r"step_(\d+)", latest_state).group(1))
        with open(latest_state, "rb") as f:
            moes = pickle.load(f)
        print(f">>> [断点恢复] 从 {latest_state} 恢复进度...")
    else:
        print(f">>> [Fresh Start] 未发现存档，开始全新进化...")
        x0_list = [p1.tolist() if i < NUM_KERNELS // 2 else p2.tolist() for i in range(NUM_KERNELS)]
        inopts = {'popsize': POP_SIZE, 'seed': 42, 'verb_filenameprefix': os.path.join(SAVE_DIR, 'outcmaes')}
        kernels = comocma.get_cmas(x0_list, SIGMA, inopts=inopts)
        moes = comocma.Sofomore(kernels, reference_point=[0.0, 0.0], opts={'archive': True})

    # ---------- 核心拦截器：评估缓存与统一推入 global_archive ----------
    current_gen = start_step
    eval_cache = {}

    def evaluate_and_cache(x):
        nonlocal global_id_counter
        x_bytes = x.tobytes()
        
        # 拦截：防止 FitFun 对同一组参数评估两次
        if x_bytes not in eval_cache:
            scores, _ = eval_fn(jax.random.split(rng_key)[0], np.expand_dims(x, 0))
            f_val = np.array(scores[0])
            eval_cache[x_bytes] = f_val
            
            # 直接将解推入与 NSGA-II 格式一致的全局档案中
            global_archive["X"].append(x.copy())
            global_archive["S"].append(f_val.copy())
            global_archive["gen"].append(current_gen)
            global_archive["id"].append(global_id_counter)
            global_id_counter += 1
            
        return eval_cache[x_bytes]

    # COMO-CMAES的封装（隐式调用拦截器）
    fit_fun = comocma.FitFun(lambda x: -evaluate_and_cache(x)[0], lambda x: -evaluate_and_cache(x)[1])

    # ---------- 进化主循环 ----------
    if start_step >= MOO_ITERS:
        print(">>> 目标代数已达到，正在生成最终轨迹图...")
        plot_all_evolution(global_archive)
        return

    for it in range(start_step, MOO_ITERS):
        t0 = time.time()
        current_gen = it
        eval_cache.clear() # 每代清理缓存，用于获取本代指标并防止内存泄漏
        
        moes.optimize(fit_fun, iterations=1)
        
        step = it + 1
        it_speed = 1.0 / max(1e-9, (time.time() - t0))
        
        current_gen_scores = np.array(list(eval_cache.values()))
        if len(current_gen_scores) > 0:
            avg_sims = np.mean(current_gen_scores, axis=0)
            print(f"Iter {step:4d} | Evals: {global_id_counter} | Speed: {it_speed:.2f} it/s | Mean: Cat={avg_sims[0]:.4f}, Bfly={avg_sims[1]:.4f}")
        
        # 定期保存与出图
        if step % SAVE_EVERY == 0 or step == MOO_ITERS:
            save_checkpoint(moes, step)
            save_archive_step(step, global_archive)
            
            # --- 保存全局大档案 (安全写入) ---
            tmp_archive = archive_path + ".tmp"
            with open(tmp_archive, "wb") as f:
                pickle.dump(global_archive, f)
            os.replace(tmp_archive, archive_path)

    plot_all_evolution(global_archive)
    print(">>> 任务完成。")

if __name__ == "__main__":
    run_moo_comocma()
