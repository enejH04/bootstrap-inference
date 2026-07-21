# Copied from: https://github.com/zrimseku/Bootstrap-CI-analysis/blob/main/generators.py
import itertools

import numpy as np
import scipy.stats


class DGP:
    def __init__(self, seed: int, true_statistics: dict = None):
        np.random.seed(seed)
        if true_statistics is None:
            self.true_statistics = {}
        self.true_statistics = true_statistics

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        raise NotImplementedError()

    def get_true_value(self, statistic_name):
        if statistic_name not in self.true_statistics:
            raise ValueError(
                f"True value of {statistic_name} is not known. You should specify it at DGP "
                f"initialization."
            )
        return self.true_statistics[statistic_name]

    def describe(self):
        return type(self).__name__


class DGPNorm(DGP):
    def __init__(
        self,
        seed: int,
        loc: float = 0,
        scale: float = 1,
        true_statistics: dict = None,
    ):
        super(DGPNorm, self).__init__(seed, true_statistics)
        self.loc = loc
        self.scale = scale
        if true_statistics is None:
            self.true_statistics = {}
        self.true_statistics["mean"] = loc
        self.true_statistics["median"] = loc
        self.true_statistics["std"] = scale
        self.true_statistics["percentile_5"] = loc - 1.645 * scale
        self.true_statistics["percentile_95"] = loc + 1.645 * scale

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.normal(self.loc, self.scale, size=size)

    def describe(self):
        return type(self).__name__ + "_" + str(self.loc) + "_" + str(self.scale)


class DGPExp(DGP):
    def __init__(
        self, seed: int, scale: float = 1, true_statistics: dict = None
    ):
        super(DGPExp, self).__init__(seed, true_statistics)
        self.scale = scale  # 1/lambda
        if true_statistics is None:
            self.true_statistics = {}
        self.true_statistics["mean"] = scale
        self.true_statistics["median"] = scale * np.log(2)
        self.true_statistics["std"] = scale
        self.true_statistics["percentile_5"] = scale * np.log(20 / 19)
        self.true_statistics["percentile_95"] = scale * np.log(20)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.exponential(self.scale, size=size)

    def describe(self):
        return type(self).__name__ + "_" + str(self.scale)


class DGPBeta(DGP):
    def __init__(
        self,
        seed: int,
        alpha: float = 1,
        beta: float = 1,
        true_statistics: dict = None,
    ):
        super(DGPBeta, self).__init__(seed, true_statistics)
        self.alpha = alpha
        self.beta = beta
        if true_statistics is None:
            self.true_statistics = {}
        self.true_statistics["mean"] = alpha / (alpha + beta)
        self.true_statistics["std"] = np.sqrt(
            alpha * beta / (alpha + beta) ** 2 / (alpha + beta + 1)
        )
        self.true_statistics["percentile_5"] = scipy.stats.beta.ppf(
            0.05, alpha, beta
        )
        self.true_statistics["percentile_95"] = scipy.stats.beta.ppf(
            0.95, alpha, beta
        )
        self.true_statistics["median"] = scipy.stats.beta.ppf(0.5, alpha, beta)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.beta(self.alpha, self.beta, size=size)

    def describe(self):
        return (
            type(self).__name__ + "_" + str(self.alpha) + "_" + str(self.beta)
        )


class DGPLogNorm(DGP):
    def __init__(
        self, seed: int, mean: float, sigma: float, true_statistics: dict = None
    ):
        super(DGPLogNorm, self).__init__(seed, true_statistics)
        self.mean = mean
        self.sigma = sigma
        if true_statistics is None:
            self.true_statistics = {}
        self.true_statistics["mean"] = np.exp(mean + (sigma**2) / 2)
        self.true_statistics["median"] = np.exp(mean)
        self.true_statistics["std"] = (
            np.exp(2 * mean + sigma**2) * (np.exp(sigma**2) - 1)
        ) ** 0.5
        self.true_statistics["percentile_5"] = np.exp(mean - 1.645 * sigma)
        self.true_statistics["percentile_95"] = np.exp(mean + 1.645 * sigma)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.lognormal(self.mean, self.sigma, size=size)

    def describe(self):
        return (
            type(self).__name__ + "_" + str(self.mean) + "_" + str(self.sigma)
        )


class DGPLaplace(DGP):
    def __init__(
        self, seed: int, loc: float, scale: float, true_statistics: dict = None
    ):
        super(DGPLaplace, self).__init__(seed, true_statistics)
        self.loc = loc
        self.scale = scale
        if true_statistics is None:
            self.true_statistics = {}
        self.true_statistics["mean"] = loc
        self.true_statistics["median"] = loc
        self.true_statistics["std"] = scale * 2**0.5
        self.true_statistics["percentile_5"] = loc + scale * np.log(0.1)
        self.true_statistics["percentile_95"] = loc - scale * np.log(0.1)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.laplace(self.loc, self.scale, size=size)

    def describe(self):
        return type(self).__name__ + "_" + str(self.loc) + "_" + str(self.scale)


class DGPBernoulli(DGP):
    def __init__(self, seed: int, p: float, true_statistics: dict = None):
        super(DGPBernoulli, self).__init__(seed, true_statistics)
        self.p = p
        if true_statistics is None:
            self.true_statistics = {}
        self.true_statistics["mean"] = p
        if p == 0.5:
            self.true_statistics["median"] = 0.5
        elif p < 0.5:
            self.true_statistics["median"] = 0
        else:
            self.true_statistics["median"] = 1
        self.true_statistics["std"] = (p * (1 - p)) ** 0.5
        self.true_statistics["percentile_5"] = float(p >= 0.95)
        self.true_statistics["percentile_95"] = float(p >= 0.05)

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.binomial(1, self.p, size=size).astype(float)

    def describe(self):
        return type(self).__name__ + "_" + str(self.p)


class DGPBiNorm(DGP):
    def __init__(
        self,
        seed: int,
        mean: np.array,
        cov: np.array,
        true_statistics: dict = None,
    ):
        super(DGPBiNorm, self).__init__(seed, true_statistics)
        self.mean = mean  # means of both variables, 1D array of length 2
        self.cov = cov  # covariance matrix, 2D array (2x2)
        if true_statistics is None:
            self.true_statistics = {}
        self.true_statistics["mean"] = mean
        self.true_statistics["median"] = mean
        self.true_statistics["std"] = np.diag(cov) ** 0.5
        self.true_statistics["corr"] = (
            cov[0, 1] / (cov[0, 0] * cov[1, 1]) ** 0.5
        )  # extend it to multiple??

    def sample(self, sample_size: int, nr_samples: int = 1) -> np.array:
        size = (nr_samples, sample_size) if nr_samples != 1 else sample_size
        return np.random.multivariate_normal(self.mean, self.cov, size=size)

    def describe(self):
        return (
            type(self).__name__
            + "-"
            + "_".join(
                [
                    str(par)
                    for par in [
                        self.mean[0],
                        self.mean[1],
                        self.cov[0, 0],
                        self.cov[0, 1],
                        self.cov[1, 1],
                    ]
                ]
            )
        )
