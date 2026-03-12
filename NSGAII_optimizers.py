import pickle
import numpy as np
import jax.numpy as jnp

from pymoo.core.problem import Problem


class BaseOptimizer:
    def ask(self):
        raise NotImplementedError

    def tell(self, x, fitness):
        raise NotImplementedError

    def get_pareto_front(self):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class BoxConstrainedProblem(Problem):
    """
    一个仅负责定义连续变量维度、目标数和 box constraints 的 pymoo Problem。

    注意：
    - 在当前 ask/tell 用法中，我们不在这里实现 _evaluate
    - 真正的评估在外部通过 eval_fn 完成
    """

    def __init__(self, num_dims, num_objs, xl, xu):
        xl = self._normalize_bounds(xl, num_dims, name="xl")
        xu = self._normalize_bounds(xu, num_dims, name="xu")

        if not np.all(xl < xu):
            bad_idx = np.where(~(xl < xu))[0]
            raise ValueError(
                f"Invalid bounds: some dimensions have xl >= xu. "
                f"First bad indices: {bad_idx[:10]}"
            )

        self.num_dims = int(num_dims)
        self.num_objs = int(num_objs)

        super().__init__(
            n_var=self.num_dims,
            n_obj=self.num_objs,
            xl=xl,
            xu=xu,
        )

    @staticmethod
    def _normalize_bounds(b, num_dims, name="bound"):
        """
        支持：
        - 标量边界：例如 -1.0
        - 一维向量边界：shape=(num_dims,)
        """
        if np.isscalar(b):
            arr = np.full(num_dims, b, dtype=np.float32)
        else:
            arr = np.asarray(b, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(
                    f"{name} must be a scalar or 1D array, got shape {arr.shape}"
                )
            if arr.shape[0] != num_dims:
                raise ValueError(
                    f"{name} length must equal num_dims={num_dims}, got {arr.shape[0]}"
                )
        return arr.astype(np.float32)

    def summarize(self):
        span = self.xu - self.xl
        return {
            "num_dims": self.num_dims,
            "num_objs": self.num_objs,
            "xl_min": float(np.min(self.xl)),
            "xl_max": float(np.max(self.xl)),
            "xu_min": float(np.min(self.xu)),
            "xu_max": float(np.max(self.xu)),
            "span_mean": float(np.mean(span)),
            "span_min": float(np.min(span)),
            "span_max": float(np.max(span)),
        }


class PymooOptimizer(BaseOptimizer):
    def __init__(self, algo_name, pop_size, num_dims, num_objs, xl, xu):
        self.algo_name = str(algo_name).lower()
        self.pop_size = int(pop_size)
        self.num_dims = int(num_dims)
        self.num_objs = int(num_objs)

        self.problem = BoxConstrainedProblem(
            num_dims=self.num_dims,
            num_objs=self.num_objs,
            xl=xl,
            xu=xu,
        )

        self.algorithm = self._build_algorithm(self.algo_name, self.pop_size)
        self.algorithm.setup(self.problem, termination=("n_gen", 1000))

        self.pop = None
#*******************************************************************************************************
    def _build_algorithm(self, algo_name, pop_size):
        if algo_name == "nsga2":
            from pymoo.algorithms.moo.nsga2 import NSGA2
            return NSGA2(pop_size=pop_size)

        raise ValueError(f"Unsupported algo_name: {algo_name}")

    def ask(self):
        self.pop = self.algorithm.ask()
        X = self.pop.get("X")
        return jnp.array(X)

    def tell(self, x, fitness):
        if self.pop is None:
            raise RuntimeError("tell() called before ask().")

        F = np.asarray(fitness, dtype=np.float32)

        if F.ndim == 1:
            # 单目标情况下允许 shape=(pop_size,)
            F = F[:, None]

        expected_shape = (len(self.pop), self.num_objs)
        if F.shape != expected_shape:
            raise ValueError(
                f"fitness shape mismatch: expected {expected_shape}, got {F.shape}"
            )

        self.pop.set("F", F)
        self.algorithm.tell(infills=self.pop)

    def get_pareto_front(self):
        res = self.algorithm.result()
        if res is None:
            return None, None
        return res.X, res.F

    def get_problem_summary(self):
        return self.problem.summarize()

    def save(self, path):
        payload = {
            "algo_name": self.algo_name,
            "pop_size": self.pop_size,
            "num_dims": self.num_dims,
            "num_objs": self.num_objs,
            "problem_xl": np.array(self.problem.xl, dtype=np.float32),
            "problem_xu": np.array(self.problem.xu, dtype=np.float32),
            "algorithm": self.algorithm,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.algo_name = payload["algo_name"]
        self.pop_size = payload["pop_size"]
        self.num_dims = payload["num_dims"]
        self.num_objs = payload["num_objs"]

        self.problem = BoxConstrainedProblem(
            num_dims=self.num_dims,
            num_objs=self.num_objs,
            xl=payload["problem_xl"],
            xu=payload["problem_xu"],
        )
        self.algorithm = payload["algorithm"]
        self.pop = None
