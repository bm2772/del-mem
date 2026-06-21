"""
Beta Mixture Model (BMM) – Gated Memory Fusion
Implements Algorithm 1 from the FluxMem paper (Appendix B).

No external ML libraries required – EM is done in pure NumPy.
"""

from __future__ import annotations
import numpy as np
from scipy.special import gammaln       # only stdlib-level scipy dep


# ---------------------------------------------------------------------------
# Beta distribution helpers
# ---------------------------------------------------------------------------

def _log_beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Log-pdf of Beta(alpha, beta) at each point in x.
    log B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)
    """
    log_norm = gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
    return log_norm + (alpha - 1) * np.log(x) + (beta - 1) * np.log(1 - x)


def _beta_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


def _moments_to_params(mu: float, var: float) -> tuple[float, float]:
    """
    Convert Beta mean (mu) and variance (var) to (alpha, beta).
    kappa = mu*(1-mu)/var - 1
    alpha = mu * kappa,  beta = (1-mu) * kappa
    """
    var = max(var, 1e-8)
    kappa = mu * (1.0 - mu) / var - 1.0
    kappa = max(kappa, 0.1)                 # avoid degenerate
    alpha = mu * kappa
    beta = (1.0 - mu) * kappa
    alpha = max(alpha, 0.1)
    beta = max(beta, 0.1)
    return alpha, beta


# ---------------------------------------------------------------------------
# BMM Gate
# ---------------------------------------------------------------------------

class BMMGate:
    """
    Two-component Beta Mixture Model gate for memory fusion.

    Usage
    -----
    gate = BMMGate(tau=0.6, min_keep=1, em_iters=20)
    accepted_indices = gate.filter(raw_scores)   # list[int]
    """

    def __init__(
        self,
        tau: float = 0.6,               # posterior threshold (τ_BMM)
        min_keep: int = 1,              # minimum candidates to retain (m_min)
        em_iters: int = 20,             # fixed EM iterations
        eps: float = 1e-6,              # boundary guard for min-max norm
    ):
        self.tau = tau
        self.min_keep = min_keep
        self.em_iters = em_iters
        self.eps = eps

        # fitted parameters (populated after fit())
        self.pi_ = np.array([0.5, 0.5])
        self.alpha_ = np.array([1.0, 5.0])
        self.beta_ = np.array([5.0, 1.0])
        self.high_k_: int = 1           # index of high-compatibility component

    # ------------------------------------------------------------------
    # normalisation (Eq. 17)
    # ------------------------------------------------------------------

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            x = self.eps + (1 - 2 * self.eps) * (scores - s_min) / (s_max - s_min)
        else:
            x = np.full_like(scores, 0.5)
        return np.clip(x, self.eps, 1 - self.eps)

    # ------------------------------------------------------------------
    # EM fitting
    # ------------------------------------------------------------------

    def _init_params(self, x: np.ndarray) -> None:
        """Quantile-based initialisation (Appendix B)."""
        q0, q1 = np.quantile(x, [0.30, 0.70])
        mu0, mu1 = q0, q1
        default_var = 0.05
        a0, b0 = _moments_to_params(mu0, default_var)
        a1, b1 = _moments_to_params(mu1, default_var)
        self.pi_ = np.array([0.5, 0.5])
        self.alpha_ = np.array([a0, a1])
        self.beta_ = np.array([b0, b1])

    def fit(self, x: np.ndarray) -> "BMMGate":
        """
        Run EM on normalised scores x ∈ (0,1).
        Updates pi_, alpha_, beta_, and high_k_.
        """
        n = len(x)
        if n < 2:
            # degenerate case: keep single sample
            self.high_k_ = 1
            return self

        self._init_params(x)

        for _ in range(self.em_iters):
            # ---- E-step (log-space) --------------------------------
            log_r = np.zeros((n, 2))
            for k in range(2):
                log_r[:, k] = (
                    np.log(self.pi_[k])
                    + _log_beta_pdf(x, self.alpha_[k], self.beta_[k])
                )
            # softmax over components
            log_r -= log_r.max(axis=1, keepdims=True)
            r = np.exp(log_r)
            r /= r.sum(axis=1, keepdims=True)       # shape (n, 2)

            # ---- M-step --------------------------------------------
            Nk = r.sum(axis=0)                       # (2,)
            self.pi_ = Nk / n

            for k in range(2):
                if Nk[k] < 1e-9:
                    continue
                mu_k = (r[:, k] @ x) / Nk[k]
                var_k = (r[:, k] @ (x - mu_k) ** 2) / Nk[k]
                a, b = _moments_to_params(mu_k, var_k)
                self.alpha_[k] = a
                self.beta_[k] = b

        # identify high-compatibility component (Eq. 24)
        means = _beta_mean(self.alpha_[0], self.beta_[0]), _beta_mean(self.alpha_[1], self.beta_[1])
        self.high_k_ = int(np.argmax(means))
        return self

    # ------------------------------------------------------------------
    # posterior gate (Eq. 25)
    # ------------------------------------------------------------------

    def gate(self, x: np.ndarray) -> np.ndarray:
        """
        Compute posterior probability of belonging to high-compatibility
        component for each normalised score in x.
        Returns array of gate values in [0, 1].
        """
        k_star = self.high_k_
        log_num = np.log(self.pi_[k_star]) + _log_beta_pdf(x, self.alpha_[k_star], self.beta_[k_star])
        log_den_parts = np.stack([
            np.log(self.pi_[k]) + _log_beta_pdf(x, self.alpha_[k], self.beta_[k])
            for k in range(2)
        ], axis=1)
        log_den = log_den_parts.max(axis=1) + np.log(np.exp(log_den_parts - log_den_parts.max(axis=1, keepdims=True)).sum(axis=1))
        return np.exp(log_num - log_den)

    # ------------------------------------------------------------------
    # full filter (Algorithm 1)
    # ------------------------------------------------------------------

    def filter(self, raw_scores: list[float] | np.ndarray) -> list[int]:
        """
        Given raw matching scores, return indices of accepted candidates.

        Parameters
        ----------
        raw_scores : array-like of shape (n,)

        Returns
        -------
        list[int] – indices into raw_scores that pass the gate.
        """
        scores = np.asarray(raw_scores, dtype=float)
        n = len(scores)

        if n == 0:
            return []
        if n == 1:
            return [0]

        x = self._normalize(scores)
        self.fit(x)
        g = self.gate(x)

        # Eq. 26 – threshold
        accepted = [i for i in range(n) if g[i] >= self.tau]

        # min-keep fallback
        if len(accepted) < self.min_keep:
            order = np.argsort(scores)[::-1]
            accepted = list(order[: self.min_keep])

        return accepted

    # ------------------------------------------------------------------
    # convenience: return best single candidate (or None)
    # ------------------------------------------------------------------

    def best(self, raw_scores: list[float] | np.ndarray) -> int | None:
        """
        Returns the index of the single best candidate that passes the gate,
        or None if no candidates qualify (caller should create a new session).
        """
        accepted = self.filter(raw_scores)
        if not accepted:
            return None
        scores = np.asarray(raw_scores, dtype=float)
        # among accepted, take the one with highest raw score
        return max(accepted, key=lambda i: scores[i])
