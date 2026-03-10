import os, time, argparse, jax, pickle, sys
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
from functools import partial

import util
import main_opt
from rollout import rollout_simulation

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def lerp(v1, v2, alpha):
    return jax.tree_util.tree_map(lambda x, y: (1.0 - alpha) * x + alpha * y, v1, v2)

class ASALAdaptiveCompute:
    def __init__(self, args):
        self.args = args
        if not hasattr(self.args, 'time_sampling'): self.args.time_sampling = 1
        
        self.outdir = os.path.join(args.outdir, f"adaptive_archive_{time.strftime('%m%d_%H%M')}")
        os.makedirs(self.outdir, exist_ok=True)
        
        # 初始化环境
        self.rollout_fn_base, self.fm, self.substrate = main_opt.setup_evaluator(self.args)
        self.rollout_steps = args.rollout_steps or self.substrate.rollout_steps
        
        self.rollout_fn = jax.jit(partial(
            rollout_simulation, 
            s0=None, substrate=self.substrate, fm=self.fm, 
            rollout_steps=self.rollout_steps, 
            time_sampling=(self.rollout_steps, True), 
            img_size=224, return_state=False
        ))

    def get_features(self, params, rng):
        z_samples = []
        grid_imgs = []
        for i in range(10):
            rng, _rng = jax.random.split(rng)
            if i == 0:
                res = rollout_simulation(_rng, params, substrate=self.substrate, fm=self.fm,
                                         rollout_steps=self.rollout_steps, 
                                         time_sampling=(self.rollout_steps, False), 
                                         img_size=128, return_state=False)
                grid_imgs = [np.array(res['rgb'][t] * 255).astype(np.uint8) for t in 
                             np.linspace(0, self.rollout_steps - 1, 6).astype(int)]
                z_samples.append(res['z'][-1])
            else:
                res = self.rollout_fn(_rng, params)
                z_samples.append(res['z'][-1])
        
        z_mean = jnp.mean(jnp.stack(z_samples), axis=0)
        return z_mean, z_samples, grid_imgs

    def run_adaptive(self):
        base_path = "data/results_pipeline_caterpillar_butterfly"
        p_a, _ = util.load_pkl(os.path.join(base_path, "seed_caterpillar"), "best")
        p_b, _ = util.load_pkl(os.path.join(base_path, "seed_butterfly"), "best")
        
        z_t_a = jnp.squeeze(self.fm.embed_txt([self.args.prompt_a]))
        z_t_b = jnp.squeeze(self.fm.embed_txt([self.args.prompt_b]))
        
        self.results = {}
        rng = jax.random.PRNGKey(self.args.seed)

        def sample_point(alpha, rng):
            if alpha in self.results: return self.results[alpha]['z_mean']
            print(f"[*] 采样 Alpha = {alpha:.4f}")
            p_interp = lerp(p_a, p_b, alpha)
            z_mean, z_samples, grid_imgs = self.get_features(p_interp, rng)
            
            # 使用全称键名 sim_text_a/b 以匹配可视化脚本
            samples_record = [{
                "sim_text_a": float(jnp.dot(zs, z_t_a)),
                "sim_text_b": float(jnp.dot(zs, z_t_b))
            } for zs in z_samples]
            
            self.results[alpha] = {
                "z_mean": z_mean,
                "samples": samples_record,
                "grid_imgs": grid_imgs,
                "final_img": grid_imgs[-1]
            }
            return z_mean

        def recursive_step(a1, a2, rng, depth=0):
            if depth >= self.args.max_depth: return
            z1 = sample_point(a1, rng)
            z2 = sample_point(a2, rng)
            diff = jnp.linalg.norm(z1 - z2)
            if diff > self.args.threshold:
                mid = (a1 + a2) / 2
                print(f"[!] 变化率 {diff:.3f} > 阈值, 细分区间 [{a1:.2f}, {a2:.2f}]")
                recursive_step(a1, mid, rng, depth + 1)
                recursive_step(mid, a2, rng, depth + 1)

        initial_alphas = np.linspace(0, 1, 5) 
        for i in range(len(initial_alphas)-1):
            recursive_step(initial_alphas[i], initial_alphas[i+1], rng)

        sorted_alphas = sorted(self.results.keys())
        final_archive = []
        for a in sorted_alphas:
            res = self.results[a]
            final_archive.append({
                "alpha": float(a),
                "samples": res['samples'],
                "grid_imgs": res['grid_imgs'],
                "final_img": res['final_img']
            })

        save_path = os.path.join(self.outdir, "interpolation_archive.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({"args": vars(self.args), "data": final_archive}, f)
        print(f"[√] 自适应采样完成。点数: {len(final_archive)}。路径: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_a", type=str, required=True)
    parser.add_argument("--prompt_b", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="./interpolation_results")
    parser.add_argument("--substrate", type=str, default="lenia")
    parser.add_argument("--foundation_model", type=str, default="clip")
    parser.add_argument("--rollout_steps", type=int, default=None)
    parser.add_argument("--time_sampling", type=int, default=1)
    
    args = parser.parse_args()
    ASALAdaptiveCompute(args).run_adaptive()
