# DGP definitions adapted from:
# https://github.com/zrimseku/bootstrap-simulation/blob/main/generators.py
import numpy as np
import numpy.typing as npt


class DGP:
    def __init__(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)
        self.true_statistics = {}

    def sample(self, sample_size: int, nr_samples: int = 1) -> npt.NDArray:
        raise NotImplementedError()

    def get_true_value(self, statistic_name: str) -> float | npt.NDArray:
        if statistic_name not in self.true_statistics:
            raise ValueError(
                f"True value of {statistic_name} is not known. You should specify it at DGP initialization."
            )

        return self.true_statistics[statistic_name]

    def describe(self) -> str:
        return type(self).__name__


class DGPNorm(DGP):
    def __init__(
        self,
        seed: int,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__(seed)

        self.loc = loc
        self.scale = scale
        self.true_statistics["mean"] = loc
        self.true_statistics["median"] = loc
        self.true_statistics["std"] = scale
        self.true_statistics["percentile_5"] = loc - 1.645 * scale
        self.true_statistics["percentile_95"] = loc + 1.645 * scale

    def sample(self, sample_size: int, nr_samples: int = 1) -> npt.NDArray:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return self.rng.normal(loc=self.loc, scale=self.scale, size=size)

    def describe(self) -> str:
        return f"{type(self).__name__}_{self.loc}_{self.scale}"


class DGPExp(DGP):
    def __init__(
        self,
        seed: int,
        scale: float = 1.0,
    ) -> None:
        super().__init__(seed)

        self.scale = scale  # 1/lambda
        self.true_statistics["mean"] = scale
        self.true_statistics["median"] = scale * np.log(2)
        self.true_statistics["std"] = scale
        self.true_statistics["percentile_5"] = scale * np.log(20 / 19)
        self.true_statistics["percentile_95"] = scale * np.log(20)

    def sample(self, sample_size: int, nr_samples: int = 1) -> npt.NDArray:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return self.rng.exponential(scale=self.scale, size=size)

    def describe(self) -> str:
        return f"{type(self).__name__}_{self.scale}"


class DGPLogNorm(DGP):
    def __init__(
        self,
        seed: int,
        mean: float,
        sigma: float,
    ) -> None:
        super().__init__(seed)

        self.mean = mean
        self.sigma = sigma

        self.true_statistics["mean"] = np.exp(mean + (sigma**2) / 2)
        self.true_statistics["median"] = np.exp(mean)
        self.true_statistics["std"] = (
            np.exp(2 * mean + sigma**2) * (np.exp(sigma**2) - 1)
        ) ** 0.5
        self.true_statistics["percentile_5"] = np.exp(mean - 1.645 * sigma)
        self.true_statistics["percentile_95"] = np.exp(mean + 1.645 * sigma)

    def sample(self, sample_size: int, nr_samples: int = 1) -> npt.NDArray:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return self.rng.lognormal(mean=self.mean, sigma=self.sigma, size=size)

    def describe(self) -> str:
        return f"{type(self).__name__}_{self.mean}_{self.sigma}"


class DGPBiNorm(DGP):
    def __init__(
        self,
        seed: int,
        mean: npt.NDArray,
        cov: npt.NDArray,
    ):
        super().__init__(seed)

        # Means of both variables, 1D array of length 2
        self.mean = mean
        # Covariance matrix, 2D array (2x2)
        self.cov = cov
        self.true_statistics["mean"] = mean
        self.true_statistics["median"] = mean
        self.true_statistics["std"] = np.diag(cov) ** 0.5
        self.true_statistics["corr"] = (
            cov[0, 1] / (cov[0, 0] * cov[1, 1]) ** 0.5
        )

    def sample(self, sample_size: int, nr_samples: int = 1) -> npt.NDArray:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return self.rng.multivariate_normal(
            mean=self.mean, cov=self.cov, size=size
        )

    def describe(self) -> str:
        return f"{type(self).__name__}-{'_'.join(str(par) for par in [self.mean[0], self.mean[1], self.cov[0, 0], self.cov[0, 1], self.cov[1, 1]])}"
