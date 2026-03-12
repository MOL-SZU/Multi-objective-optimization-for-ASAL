import jax
import jax.numpy as jnp
import asal_metrics

def get_batch_loss_fn(rollout_fn, fm, prompts):
    z_txt = fm.embed_txt(prompts)

    def eval_single(rng, params):
        rollout_data = rollout_fn(rng, params)
        z = rollout_data["z"]
       
        scores = []
        for i in range(len(prompts)):
            s = asal_metrics.calc_supervised_target_score(z, z_txt[i:i+1])
            scores.append(s)

        scores = jnp.stack(scores, axis=0)
        return scores, rollout_data

    eval_batch = jax.jit(jax.vmap(eval_single, in_axes=(0, 0)))
    return eval_batch