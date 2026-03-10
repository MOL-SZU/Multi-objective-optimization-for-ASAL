import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
from functools import partial

import jax
import jax.numpy as jnp
from jax.random import split
import numpy as np
import evosax
from tqdm.auto import tqdm

import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics
import util


parser = argparse.ArgumentParser()
group = parser.add_argument_group("meta")
group.add_argument("--seed", type=int, default=0, help="the random seed")
group.add_argument("--save_dir", type=str, default=None, help="path to save results to")

group.add_argument("--w1", type=float, default=1.0, help="weight for prompt 1 (e.g., butterfly)")
group.add_argument("--w2", type=float, default=1.0, help="weight for prompt 2 (e.g., caterpillar)")

group = parser.add_argument_group("substrate")
group.add_argument("--substrate", type=str, default="boids", help="name of the substrate")
group.add_argument(
    "--rollout_steps",
    type=int,
    default=None,
    help="number of rollout timesteps, leave None for the default of the substrate",
)

group = parser.add_argument_group("evaluation")
group.add_argument(
    "--foundation_model",
    type=str,
    default="clip",
    help="the foundation model to use (don't touch this)",
)
group.add_argument(
    "--time_sampling",
    type=int,
    default=1,
    help="number of images to render during one simulation rollout",
)
group.add_argument(
    "--prompts",
    type=str,
    default="a biological cell;two biological cells",
    help="prompts to optimize for seperated by ';'",
)
group.add_argument("--coef_prompt", type=float, default=1.0, help="coefficient for ASAL prompt loss")
group.add_argument(
    "--coef_softmax",
    type=float,
    default=0.0,
    help="coefficient for softmax loss (only for multiple temporal prompts)",
)
group.add_argument(
    "--coef_oe",
    type=float,
    default=0.0,
    help="coefficient for ASAL open-endedness loss (only for single prompt)",
)

group = parser.add_argument_group("optimization")
group.add_argument("--bs", type=int, default=1, help="number of init states to average simulation over")
group.add_argument("--pop_size", type=int, default=16, help="population size for Sep-CMA-ES")
group.add_argument("--n_iters", type=int, default=1000, help="number of iterations to run")
group.add_argument("--sigma", type=float, default=0.1, help="mutation rate")


# Optional helper (safe to keep; not used by main)
def setup_evaluator(args):
    """Extracted init logic for external scripts (optional)."""
    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)

    rollout_fn = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=fm,
        rollout_steps=args.rollout_steps or substrate.rollout_steps,
        time_sampling=(args.time_sampling, True),
        img_size=224,
        return_state=False,
    )
    return rollout_fn, fm, substrate


def parse_args(*args, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    for k, v in vars(args).items():
        if isinstance(v, str) and v.lower() == "none":
            setattr(args, k, None)
    return args


def main(args):
    prompts = args.prompts.split(";")
    if args.time_sampling < len(prompts):  # ensure enough frames if using multiple prompts
        args.time_sampling = len(prompts)
    print(args)

    fm = foundation_models.create_foundation_model(args.foundation_model)
    substrate = substrates.create_substrate(args.substrate)
    substrate = substrates.FlattenSubstrateParameters(substrate)

    if args.rollout_steps is None:
        args.rollout_steps = substrate.rollout_steps

    rollout_fn = partial(
        rollout_simulation,
        s0=None,
        substrate=substrate,
        fm=fm,
        rollout_steps=args.rollout_steps,
        time_sampling=(args.time_sampling, True),
        img_size=224,
        return_state=False,
    )

    z_txt = fm.embed_txt(prompts)  # (P, D)

    rng = jax.random.PRNGKey(args.seed)
    strategy = evosax.Sep_CMA_ES(
        popsize=args.pop_size, num_dims=substrate.n_params, sigma_init=args.sigma
    )
    es_params = strategy.default_params
    rng, _rng = split(rng)
    es_state = strategy.initialize(_rng, es_params)

    def calc_loss(rng, params):
        rollout_data = rollout_fn(rng, params)
        z = rollout_data["z"]

        # ---- Two prompts: loss = loss1 + loss2 ----
        loss1 = asal_metrics.calc_supervised_target_score(z, z_txt[0:1])
        loss2 = asal_metrics.calc_supervised_target_score(z, z_txt[1:2])
        w_sum = args.w1 + args.w2
        loss_prompt = (args.w1 * loss1 + args.w2 * loss2) / (w_sum + 1e-8)
        loss_each = jnp.stack([loss1, loss2])

        loss_softmax = asal_metrics.calc_supervised_target_softmax_score(z, z_txt)
        loss_oe = asal_metrics.calc_open_endedness_score(z)

        loss = (
            loss_prompt * args.coef_prompt
            + loss_softmax * args.coef_softmax
            + loss_oe * args.coef_oe
        )

        loss_dict = dict(
            loss=loss,
            loss_prompt=loss_prompt,
            loss_softmax=loss_softmax,
            loss_oe=loss_oe,
            loss_prompt_each=loss_each,
        )
        return loss, loss_dict



    @jax.jit
    def do_iter(es_state, rng):
        rng, _rng = split(rng)
        params, next_es_state = strategy.ask(_rng, es_state, es_params)

        # vmap over init-state rng (bs) and parameters (pop_size)
        calc_loss_vv = jax.vmap(jax.vmap(calc_loss, in_axes=(0, None)), in_axes=(None, 0))

        rng, _rng = split(rng)
        loss, loss_dict = calc_loss_vv(split(_rng, args.bs), params)

        # mean over bs dimension
        loss, loss_dict = jax.tree.map(lambda x: x.mean(axis=1), (loss, loss_dict))

        next_es_state = strategy.tell(params, loss, next_es_state, es_params)
        data = dict(best_loss=next_es_state.best_fitness, loss_dict=loss_dict)
        return next_es_state, data

    data = []
    pbar = tqdm(range(args.n_iters))
    for i_iter in pbar:
        rng, _rng = split(rng)
        es_state, di = do_iter(es_state, _rng)

        data.append(di)
        pbar.set_postfix(best_loss=es_state.best_fitness.item())

        if args.save_dir is not None and (
            i_iter % max(1, (args.n_iters // 10)) == 0 or i_iter == args.n_iters - 1
        ):
            data_save = jax.tree.map(lambda *x: np.array(jnp.stack(x, axis=0)), *data)
            util.save_pkl(args.save_dir, "data", data_save)
            best = jax.tree.map(lambda x: np.array(x), (es_state.best_member, es_state.best_fitness))
            util.save_pkl(args.save_dir, "best", best)


if __name__ == "__main__":
    main(parse_args())
