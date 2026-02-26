import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

class PolicyOptimizer(ABC):
    def __init__(self, y_true, y_hat_s, y_hat_m, betas, xi, oracle=False, name=None):
        self.name = name
        self.gold_labels = np.array(y_true)
        # self.oracle_labels = None
        if oracle:
            self.gold_labels = np.array(y_hat_m)
            # assert np.array_equal(self.oracle_labels, self.gold_labels)
            assert np.array_equal(self.gold_labels, np.array(y_hat_m))

        self.s_predictions = np.array(y_hat_s)
        self.m_predictions = np.array(y_hat_m)
        self.betas = betas

        assert not np.array_equal(self.s_predictions, self.m_predictions)
        assert(len(self.gold_labels) == len(self.s_predictions) == len(self.betas))

        self.xi = xi
    
    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass


class SingleThresholdOptimizer(PolicyOptimizer):
    def __init__(self, y_true, y_hat_s, y_hat_m, betas, xi, 
                 cos_s, cos_m, oracle=False, name=None):
        super().__init__(y_true, y_hat_s, y_hat_m, betas, xi, oracle, name)

        self.cos_s = np.array(cos_s)
        self.cos_m = np.array(cos_m)

        
    
    def optimize(self, c_s, c_m):
        if self.name == 'oracle':
            return self.optimize_indicators_oracle_v2(c_s, c_m)
        elif 'linear' in self.name:
            return self.optimize_indicators_oracle_linear(c_s, c_m)
        elif 'continuous' in self.name:
            return self.optimize_continuous_loss_imperfect(c_s, c_m)
        else:
            return self.optimize_indicators_imperfect(c_s, c_m)

    def _min_achievable_error_single_threshold_non_oracle(self):
        n = len(self.betas)
        order = np.argsort(self.betas)
        b  = self.betas[order]
        e1 = (self.s_predictions[order] != self.gold_labels[order]).astype(int)
        e2 = (self.m_predictions[order] != self.gold_labels[order]).astype(int)

        # accept-all and defer-all errors
        err_accept_all = e1.sum() / n
        err_defer_all  = e2.sum() / n

        # thresholds at unique beta levels
        u, idx_start, counts = np.unique(b, return_index=True, return_counts=True)
        e1_per = np.add.reduceat(e1, idx_start)
        e2_per = np.add.reduceat(e2, idx_start)

        acc_m1 = e1_per[::-1].cumsum()[::-1]                  # accepted suffix
        def_m2 = np.concatenate(([0], e2_per.cumsum()[:-1]))  # deferred prefix

        err_grid = (acc_m1 + def_m2) / n

        # global minimum and argmin threshold
        min_err = min(err_accept_all, err_defer_all, float(err_grid.min()))
        if min_err == err_accept_all:
            tau_star = np.nextafter(b.min(), -np.inf)   # accept all
        elif min_err == err_defer_all:
            tau_star = np.nextafter(b.max(), np.inf)    # defer all
        else:
            j_star = int(np.argmin(err_grid))
            tau_star = float(u[j_star])

        return float(min_err), float(tau_star)

    def _compare_error_to_mllm(self, tau, accept_ge=True):
        """
        Returns (E_tau, E_m, delta=E_tau-E_m).
        Uses global (unconditional) MC error.
        """
        # betas = np.asarray(betas, float)
        # y_true = np.asarray(y_true)
        # yhat_s = np.asarray(yhat_s)
        # yhat_m = np.asarray(yhat_m)

        e_s = (self.s_predictions != self.gold_labels).astype(int)
        e_m = (self.m_predictions != self.gold_labels).astype(int)

        if accept_ge:
            accepted = self.betas >= float(tau)
        else:
            accepted = self.betas > float(tau)

        # cascade error at tau
        E_tau = (accepted * e_s + (~accepted) * e_m).mean()

        # defer-all baseline (mLLM alone)
        E_m = e_m.mean()

        # difference
        delta = E_tau - E_m  # positive means mLLM is better alone...
        to_print = 'positive' if delta > 0 else 'negative'
        print(f'Difference between min. achievable error and mLLM baseline error = {delta}, it is {to_print}.')

        # return float(E_tau), float(E_m), float(delta)

    # def optimize_indicators_oracle(self, c_s, c_m):
    #     n = len(self.betas)
    #     if not isinstance(c_s, list):
    #         assert c_s < c_m
    #         c_s = [c_s] * n
    #     if not isinstance(c_m, list):
    #         c_m = [c_m] * n

    #     total_mismatch = np.sum(self.gold_labels != self.s_predictions)
    #     err_at_0 = total_mismatch / n
    #     if err_at_0 <= 1 - self.xi:  # no need to defer, keep small model
    #         return 0.0, float(c_s[0]), round(float(err_at_0), 5)
    #     order = np.argsort(self.betas)
    #     beta_sorted = self.betas[order]
    #     mism_sorted = np.array(self.gold_labels[order] != self.s_predictions[order]).astype(float)
    #     cum_mis_right = np.cumsum(mism_sorted[::-1])[::-1]
    #     best = None
    #     for i in tqdm(range(n)):  # candidate threshold beta_i
    #         err = cum_mis_right[i] / n  # mean error based on threshold beta_i
    #         if err <= 1 - self.xi:
    #             cost = c_s[i] + c_m[i] * (i / n)
    #             best = (beta_sorted[i], cost, err)
    #             print(f'\nAccepted = {np.sum(self.betas >= beta_sorted[i])}, '
    #                   f'error = {np.sum(self.s_predictions[self.betas >= beta_sorted[i]] != self.gold_labels[self.betas >= beta_sorted[i]]) / len(self.gold_labels)}, '
    #                   f'cost = {c_s[i] + c_m[i] * (1.0 - np.array(self.betas >= beta_sorted[i]).mean())}')
    #             break
    #     if best is None:
    #         return 1.0, float(c_s[0] + c_m[0]), 0.0
    #     return tuple(map(float, best))

    def optimize_indicators_oracle_v2(self, c_s, c_m):
        """
        Pick τ for the policy 'accept if beta >= τ' under the unconditional-error
        constraint (# accepted mistakes)/N <= 1 - xi, assuming perfect oracle on deferrals.
        Returns (tau, cost_per_sample, error_per_sample).
        """
        n = len(self.betas)
        order = np.argsort(self.betas)  # ascending
        beta_sorted = self.betas[order]
        wrong_sorted = np.array(self.gold_labels[order] != self.s_predictions[order]).astype(int)

        # Right-cumulative accepted-mistake counts: E_right[i] = mistakes in i..N-1
        e_right = np.cumsum(wrong_sorted[::-1])[::-1]

        # --- Early 'accept all' check (τ below min β) ---
        err_all = e_right[0] / n  # all accepted
        if err_all <= 1 - self.xi:
            tau = np.nextafter(beta_sorted[0], -np.inf)  # strictly below min β
            cost = c_s + c_m * 0.0
            return float(tau), float(cost), float(err_all)

        # --- Scan candidates; use LEFT boundary of the τ-level (tie-aware) ---
        for i in tqdm(range(n)):
            tau = beta_sorted[i]  # candidate τ
            j = np.searchsorted(beta_sorted, tau, side='left')  # first index with β >= τ
            err = e_right[j] / n  # accepted mistakes / N
            if err <= 1 - self.xi:
                deferral = j / n  # #(β < τ)/N
                cost = c_s + c_m * deferral
                return float(tau), {'cost': float(cost), 'error': float(err)}

        # --- Infeasible: accept none (τ above max β) ---
        tau = np.nextafter(beta_sorted[-1], np.inf)  # strictly above max β
        cost = c_s + c_m * 1.0
        err = 0.0  # perfect-oracle assumption
        return float(tau), {'cost': float(cost), 'error': float(err)}

    # def optimize_indicators_linear(self, c_s, c_m, rng=None):
    #     """
    #     Perfect-oracle constraint. Finds tau for policy 'accept if beta >= tau'
    #     minimizing per-sample cost c_s + c_m * deferral subject to
    #         (# accepted mistakes) / N <= 1 - xi.
    #
    #     Returns (tau, cost_per_sample, error_per_sample).
    #     Expected O(N) time; worst-case O(N) with good pivots.
    #     """
    #     y = np.asarray(self.oracle_labels)
    #     yh = np.asarray(self.predicted_labels)
    #     b = np.asarray(self.betas, dtype=float)
    #     n = len(b)
    #     assert y.shape == yh.shape == (n,), "Mismatched lengths"
    #
    #     alpha_n = (1.0 - self.xi) * n
    #     wrong = np.array(yh != y).astype(np.int64)
    #
    #     rng = np.random.default_rng(rng)
    #
    #     # Active index set we are searching over for lower thresholds.
    #     # 'base_mistakes'/'base_size' are the mistakes/size of the already-accepted
    #     # tail we've locked in (β >= current pivot) as we push tau downward.
    #     active = np.arange(n)
    #     base_mistakes = 0
    #     base_size = 0
    #     # Keep the minimum β among currently accepted items; this will be tau*.
    #     accepted_beta_min = np.inf
    #
    #     # Edge case: accepting all (tau below min β) is already feasible
    #     total_wrong = wrong.sum()
    #     if total_wrong <= alpha_n:
    #         # accept everyone
    #         tau = np.nextafter(b.min() if n else 0.0, -np.inf)
    #         cost = c_s  # deferral = 0
    #         err = total_wrong / n
    #         return float(tau), float(cost), float(err)
    #
    #     # Main partition-search loop
    #     while active.size > 0:
    #         # --- choose a random pivot from the active set ---
    #         p = b[active[rng.integers(active.size)]]
    #
    #         # --- 3-way partition of 'active' by pivot p ---
    #         idx = active
    #         # Use exact equality for true ties; if you keep many decimals in β,
    #         # ties will be rare; that's fine too.
    #         l_mask = b[idx] < p
    #         e_mask = b[idx] == p
    #         g_mask = b[idx] > p
    #
    #         l = idx[l_mask]
    #         e = idx[e_mask]
    #         g = idx[g_mask]
    #
    #         # Accepted at tau=p is (E ∪ G), plus whatever we already accepted in 'base'
    #         mistakes_eg = wrong[e].sum() + wrong[g].sum()
    #         size_eg = e.size + g.size
    #         accepted_mistakes_at_p = base_mistakes + mistakes_eg
    #
    #         if accepted_mistakes_at_p <= alpha_n:
    #             # Feasible at p -> we can try to LOWER tau to reduce cost
    #             # Lock in E∪G to the base and continue searching in L
    #             base_mistakes += mistakes_eg
    #             base_size += size_eg
    #             if size_eg > 0:
    #                 # minimal β among accepted is the pivot's value p
    #                 accepted_beta_min = p if not np.isfinite(accepted_beta_min) else min(accepted_beta_min, p)
    #             active = l
    #         else:
    #             # Infeasible at p -> must RAISE tau: search only in G
    #             active = g
    #             # base_* unchanged
    #
    #     # If we never found feasibility during the loop, accept none (always feasible)
    #     if not np.isfinite(accepted_beta_min):
    #         tau = np.nextafter(b.max() if n else 1.0, np.inf)
    #         err = 0.0
    #         cost = c_s + c_m  # deferral = 1
    #         return float(tau), float(cost), float(err)
    #
    #     # Final τ is the smallest β that keeps feasibility: the min β in the accepted base
    #     tau = float(accepted_beta_min)
    #     defr = (n - base_size) / n
    #     err = base_mistakes / n
    #     cost = c_s + c_m * defr
    #     return float(tau), float(cost), float(err)

    def _median_of_medians(self, arr):
        """Return a pivot value using BFPRT (median-of-medians) in O(n)."""
        arr = np.asarray(arr)
        n = arr.size
        if n <= 5:
            return np.partition(arr, n // 2)[n // 2]
        # Partition into groups of 5, take medians
        groups = arr[: (n // 5) * 5].reshape(-1, 5)
        medians = np.partition(groups, 2, axis=1)[:, 2]
        if n % 5:
            tail = np.partition(arr[(n // 5) * 5:], arr[(n // 5) * 5:].size // 2)
            medians = np.concatenate([medians, [tail[tail.size // 2]]])
        # Recurse to get pivot as median of medians
        return self._median_of_medians(medians)

    def optimize_indicators_oracle_linear(self, c_s, c_m):
        """
        Perfect-oracle constraint. Policy: accept if beta >= tau.
        Returns (tau, cost_per_sample, error_per_sample).
        Worst-case O(N) via median-of-medians pivot selection.
        """
        y = np.asarray(self.gold_labels)
        yh = np.asarray(self.s_predictions)
        b = np.asarray(self.betas, dtype=float)
        n = b.size
        wrong = np.array(yh != y).astype(np.int64)
        alpha_n = (1.0 - self.xi) * n

        # Early 'accept all' check
        total_wrong = wrong.sum()
        if total_wrong <= alpha_n:
            tau = np.nextafter(b.min() if n else 0.0, -np.inf)
            err = total_wrong / max(1, n)
            cost = c_s  # deferral = 0
            return float(tau), float(cost), float(err)

        # Active indices and "accepted base" accumulated as we push tau downward
        active = np.arange(n)
        base_mistakes = 0
        base_size = 0
        accepted_beta_min = np.inf

        while active.size:
            # --- BFPRT pivot ---
            p = self._median_of_medians(b[active])

            idx = active
            l_mask = b[idx] < p
            e_mask = b[idx] == p
            g_mask = b[idx] > p

            l = idx[l_mask]
            e = idx[e_mask]
            g = idx[g_mask]

            mistakes_eg = wrong[e].sum() + wrong[g].sum()
            size_eg = e.size + g.size
            accepted_mistakes_at_p = base_mistakes + mistakes_eg

            if accepted_mistakes_at_p <= alpha_n:
                # Feasible at p ⇒ lock in E∪G, search lower τ in L
                base_mistakes += mistakes_eg
                base_size += size_eg
                if size_eg > 0:
                    accepted_beta_min = p if not np.isfinite(accepted_beta_min) else min(accepted_beta_min, p)
                active = l
            else:
                # Infeasible at p ⇒ need higher τ, search in G
                active = g

        # If nothing feasible found within loop, accept none
        if not np.isfinite(accepted_beta_min):
            tau = np.nextafter(b.max() if n else 1.0, np.inf)
            err = 0.0
            cost = c_s + c_m
            return float(tau), float(cost), float(err)

        tau = float(accepted_beta_min)  # tie-aware β ≥ τ
        defer = (n - base_size) / max(1, n)
        err = base_mistakes / max(1, n)
        cost = c_s + c_m * defer
        return float(tau), {'cost': float(cost), 'error': float(err)}
    

    def optimize_indicators_imperfect(self, c_s, c_m):
        """
        Non-oracle (imperfect m2) case.
        Policy: accept if beta >= tau, else defer to m2.
        Objective: minimize cost = c_s + c_m * deferral_rate
        s.t. global error = (accepted mistakes by m1 + deferred mistakes by m2)/N <= 1 - xi

        Returns (tau, cost_per_sample, error_per_sample).
        """
        # y     = np.asarray(y)
        # yhat1 = np.asarray(yhat1)
        # yhat2 = np.asarray(yhat2)
        # beta  = np.asarray(beta, dtype=float)
        n = len(self.betas)
        assert self.gold_labels.shape == self.s_predictions.shape == self.m_predictions.shape == (n,), "Mismatched lengths."

        alpha = 1.0 - self.xi  # error constraint
        min_error, min_theta = self._min_achievable_error_single_threshold_non_oracle()
        print('*****Sanity check*****')
        print(f'Min. achievable error = {min_error} with thres. = {min_theta}. Alpha (1-xi) = {alpha}.')
        self._compare_error_to_mllm(min_theta)
        print('*************************')

        # Sort ascending by beta
        order = np.argsort(self.betas)
        b = self.betas[order]
        e1 = (self.s_predictions[order] != self.gold_labels[order]).astype(int)
        e2 = (self.m_predictions[order] != self.gold_labels[order]).astype(int)

        # Group ties in beta (unique levels)
        u, idx_start, counts = np.unique(b, return_index=True, return_counts=True)

        # Sum errors per beta-level
        e1_per = np.add.reduceat(e1, idx_start)
        e2_per = np.add.reduceat(e2, idx_start)

        # For threshold tau = u[j] with rule accept beta >= tau:
        # accepted mistakes from m1 = sum_{levels j..end} e1_per
        acc_m1 = e1_per[::-1].cumsum()[::-1]                 # length L

        # deferred mistakes from m2 = sum_{levels 0..j-1} e2_per
        def_m2 = np.concatenate(([0], e2_per.cumsum()[:-1])) # length L

        # deferred count = total samples with beta < u[j]
        def_cnt = np.concatenate(([0], counts.cumsum()[:-1])) # length L

        err_grid  = (acc_m1 + def_m2) / n
        cost_grid = c_s + c_m * (def_cnt / n)

        # Also consider "accept all" (tau below min beta)
        err_accept_all  = e1.sum() / n
        cost_accept_all = float(c_s)

        # best_tau = None
        # best_cost = None
        # best_err = None

        # Check accept-all first (minimum possible cost)
        if err_accept_all <= alpha + 1e-12:
            tau = np.nextafter(b.min(), -np.inf)
            return float(tau), {'cost': float(cost_accept_all), 'error': float(err_accept_all)}

        # Otherwise scan thresholds in increasing tau; pick the first feasible (minimum cost)
        feasible = np.where(err_grid <= alpha + 1e-12)[0]
        if len(feasible) > 0:
            j = int(feasible[0])  # smallest tau among feasible -> smallest deferral -> min cost
            return float(u[j]), {'cost': float(cost_grid[j]), 'error': float(err_grid[j])}

        # If nothing feasible, return "accept none" (tau above max beta): everyone deferred
        # Then error = m2 error rate, cost = c_s + c_m
        tau = np.nextafter(b.max(), np.inf)
        err = e2.sum() / n
        cost = c_s + c_m
        return float(tau), {'cost': float(cost), 'error': float(err)}
    

    def optimize_continuous_loss_imperfect(self, c_s, c_m):
        """
        Continuous-loss deferral with imperfect m2.

        accept if beta >= tau, else defer to m2.

        Loss per sample from cosine similarity:
            loss = (1 - cos) / 2  in [0, 1]

        Constraint: average global loss <= alpha
            (accepted loss from m1 + deferred loss from m2) / N <= alpha

        Objective: minimize cost = c_s + c_m * deferral_rate

        Returns: (tau, cost_per_sample, avg_loss)
        """
        # beta = np.asarray(self.betas, dtype=float)
        # cos1 = np.asarray(cos1, dtype=float)
        # cos2 = np.asarray(cos2, dtype=float)
        N = len(self.betas)
        # print(self.betas, sigmoid(self.betas), np.sort(self.betas), sigmoid(np.sort(self.betas)))
        # self.betas = sigmoid(self.betas)
        assert self.cos_m.shape == (N,) and self.cos_s.shape == (N,), "Mismatched lengths."

        alpha = 1. - self.xi

        # map cosine [-1,1] -> loss [0,1]
        l1 = (1.0 - self.cos_s) / 2.0  # loss per sample
        l2 = (1.0 - self.cos_m) / 2.0

        # sort by beta ascending
        order = np.argsort(self.betas)
        b = self.betas[order]
        l1s = l1[order]
        l2s = l2[order]

        # group ties
        u, idx_start, counts = np.unique(b, return_index=True, return_counts=True)

        # sum losses per beta-level
        l1_per = np.add.reduceat(l1s, idx_start)
        l2_per = np.add.reduceat(l2s, idx_start)

        # For tau = u[j] with accept beta >= tau:
        acc_loss = l1_per[::-1].cumsum()[::-1]                 # sum loss of accepted by m1
        def_loss = np.concatenate(([0.0], l2_per.cumsum()[:-1]))# sum loss of deferred to m2
        def_cnt  = np.concatenate(([0], counts.cumsum()[:-1]))  # number deferred

        avg_loss_grid = (acc_loss + def_loss) / N
        cost_grid     = c_s + c_m * (def_cnt / N)

        # Also consider accept-all (tau below min beta)
        avg_loss_accept_all = l1.sum() / N
        if avg_loss_accept_all <= alpha + 1e-12:
            tau = np.nextafter(b.min(), -np.inf)
            return float(tau), {'cost': float(c_s), 'error': float(avg_loss_accept_all)}

        # First feasible threshold (min cost since cost increases with tau/deferral)
        feasible = np.where(avg_loss_grid <= alpha + 1e-12)[0]
        if len(feasible) > 0:
            j = int(feasible[0])
            return float(u[j]), {'cost': float(cost_grid[j]), 'error': float(avg_loss_grid[j])}

        # If nothing feasible: accept none (tau above max beta)
        avg_loss_all_defer = l2.sum() / N
        tau = np.nextafter(b.max(), np.inf)
        return float(tau), {'cost': float(c_s + c_m), 'error': float(avg_loss_all_defer)}
    

class MultiThresholdOptimizer(PolicyOptimizer):
    def __init__(self, y_true, y_hat_s, y_hat_m, betas, xi, oracle=False, name=None):
        super().__init__(y_true, y_hat_s, y_hat_m, betas, xi, oracle, name)
    
    def optimize(self, c_s, c_m):
        if self.name == 'oracle':
            return self.optimize_oracle_mc_dp(c_s, c_m)
        else:
            return self.optimize_imperfect_mc_dp(c_s, c_m)
            # return self.optimize_imperfect_mc_lagrangian(c_s, c_m)
    
    def _apply_multithreshold_from_betas(
        self,
        # yhat_s: np.ndarray,
        # betas: np.ndarray,
        theta: np.ndarray,
        accept_ge: bool = True) -> np.ndarray:
        """
        Accept if betas[i] >= theta[yhat_s[i]] (or > if accept_ge=False).
        """
        yhat_s = np.asarray(self.s_predictions)
        betas = np.asarray(self.betas, dtype=float)
        theta = np.asarray(theta, dtype=float)

        if accept_ge:
            return betas >= theta[yhat_s]
        else:
            return betas > theta[yhat_s]
    
    @staticmethod
    def _pareto_prune(states: Dict[int, int]) -> Dict[int, int]:
        """
        states: error_count -> deferral_count
        Keep only non-dominated (for increasing error, strictly improving min deferrals).
        """
        items = sorted(states.items())  # error asc
        pruned = {}
        best_def = 10**18
        for e, d in items:
            if d < best_def:
                pruned[e] = d
                best_def = d
        return pruned
    
    def _build_options_non_oracle(
        self,
        # y_true: np.ndarray,
        # yhat_s: np.ndarray,
        # betas: np.ndarray,
        # yhat_m: np.ndarray,
        K: Optional[int] = None,
    ) -> Tuple[List[Dict[str, np.ndarray]], int]:
        """
        Non-oracle: deferred samples use mLLM => error = accepted s-errors (suffix) + deferred m-errors (prefix).
        """
        y_true = np.asarray(self.gold_labels)
        yhat_s = np.asarray(self.s_predictions)
        yhat_m = np.asarray(self.m_predictions)
        # betas = np.asarray(self.betas, dtype=float)
        # n = len(betas)

        if K is None:
            K = int(yhat_s.max()) + 1

        e_s = (yhat_s != y_true).astype(int)
        e_m = (yhat_m != y_true).astype(int)

        class_opts: List[Dict[str, np.ndarray]] = []
        for k in range(K):
            idx = np.where(yhat_s == k)[0]
            if idx.size == 0:
                class_opts.append({
                    "k": k,
                    "tau": np.array([np.nextafter(0.0, -np.inf)], float),
                    "def_cnt": np.array([0], int),
                    "err_cnt": np.array([0], int),
                })
                continue

            b = self.betas[idx]
            es = e_s[idx]
            em = e_m[idx]

            order = np.argsort(b)
            b = b[order]
            es = es[order]
            em = em[order]

            u, idx_start, counts = np.unique(b, return_index=True, return_counts=True)
            es_per = np.add.reduceat(es, idx_start)
            em_per = np.add.reduceat(em, idx_start)

            # tau = u[j] => defer levels < j, accept >= j
            def_cnt = np.concatenate(([0], counts.cumsum()[:-1]))
            m_err_prefix = np.concatenate(([0], em_per.cumsum()[:-1]))
            s_err_suffix = es_per[::-1].cumsum()[::-1]
            err_cnt = m_err_prefix + s_err_suffix

            # defer-all endpoint
            tau_all_defer = np.nextafter(u.max(), np.inf)
            tau = np.concatenate([u, [tau_all_defer]])
            def_cnt = np.concatenate([def_cnt, [idx.size]])
            err_cnt = np.concatenate([err_cnt, [int(em.sum())]])

            class_opts.append({"k": k, "tau": tau, "def_cnt": def_cnt, "err_cnt": err_cnt})

        return class_opts, K
    
    def optimize_imperfect_mc_dp(
        self,
        # y_true: np.ndarray,
        # yhat_s: np.ndarray,
        # betas: np.ndarray,
        # yhat_m: np.ndarray,
        # error_budget: float,
        c_s: float,
        c_m: float,
        # K: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Exact DP for non-oracle MC objective.
        """
        y_true = np.asarray(self.gold_labels)
        yhat_s = np.asarray(self.s_predictions)
        yhat_m = np.asarray(self.m_predictions)
        betas = np.asarray(self.betas, dtype=float)
        n = len(betas)

        error_budget = 1. - self.xi

        class_opts, K = self._build_options_non_oracle()
        B = int(np.floor(error_budget * n + 1e-12))
        # B = n

        states: Dict[int, int] = {0: 0}
        back: List[Dict[int, Tuple[int, int]]] = []

        for opt in class_opts:
            tau, dcnt, ecnt = opt["tau"], opt["def_cnt"], opt["err_cnt"]
            new_states: Dict[int, int] = {}
            new_back: Dict[int, Tuple[int, int]] = {}

            for prev_e, prev_d in states.items():
                for j in range(len(tau)):
                    e = prev_e + int(ecnt[j])
                    if e > B: 
                        continue
                    d = prev_d + int(dcnt[j])
                    if (e not in new_states) or (d < new_states[e]):
                        new_states[e] = d
                        new_back[e] = (prev_e, j)

            new_states = self._pareto_prune(new_states)
            new_back = {e: new_back[e] for e in new_states.keys()}
            states = new_states
            back.append(new_back)

        if not states:
            theta = np.full(K, np.nextafter(1.0, np.inf), float)
            accepted = self._apply_multithreshold_from_betas(theta)
            defr = float((~accepted).mean())
            err = float((((accepted) & (yhat_s != y_true)).sum() + ((~accepted) & (yhat_m != y_true)).sum()) / n)
            cost = float(c_s + c_m * defr)
            return theta, {"err": err, "cost": cost, "deferral": defr, "coverage": 1-defr}

        # best_e = min(states.keys(), key=lambda e: (states[e], e))
        # lambda_ = 20 * c_m
        # es = []
        # for lambda_ in np.concatenate([[0.0], np.logspace(-4, 4, 300)]):
        #     best_e = min(states.keys(), key=lambda e: c_m * (states[e] / n) + lambda_ * (e / n))
        #     es.append(best_e)

        # best_e = min(states.keys(), key=lambda e: c_m * (states[e] / n) + lambda_ * (e / n))
        # best_e = min(es)

        # best_def = states[best_e]

        # --- Two-stage selection ---
        min_def = min(states.values())

        epsilon_rate = 0.0
        epsilon = int(np.floor(epsilon_rate * n))

        candidates = [
            e for e, d in states.items()
            if d <= min_def + epsilon
        ]

        best_e = min(candidates)
        best_def = states[best_e]


        chosen = [0] * K
        cur_e = best_e
        for t in range(len(class_opts)-1, -1, -1):
            prev_e, j = back[t][cur_e]
            k = class_opts[t]["k"]
            chosen[k] = j
            cur_e = prev_e

        theta = np.zeros(K, float)
        for opt in class_opts:
            k = opt["k"]
            theta[k] = float(opt["tau"][chosen[k]])

        accepted = self._apply_multithreshold_from_betas(theta)
        defr = float((~accepted).mean())
        err = float((((accepted) & (yhat_s != y_true)).sum() + ((~accepted) & (yhat_m != y_true)).sum()) / n)
        cost = float(c_s + c_m * defr)

        return theta, {"err": err, "cost": cost, "deferral": defr, "coverage": 1-defr,
                       "error_count": float(best_e), "deferred_count": float(best_def),}
    
    def optimize_imperfect_mc_lagrangian(
        self,
        # y_true: np.ndarray,
        # yhat_s: np.ndarray,
        # betas: np.ndarray,
        # yhat_m: np.ndarray,
        # error_budget: float,
        c_s: float,
        c_m: float,
        # K: Optional[int] = None,
        lambdas: Optional[np.ndarray] = None, 
        ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Lagrangian sweep for non-oracle MC objective.
        """
        y_true = np.asarray(self.gold_labels)
        yhat_s = np.asarray(self.s_predictions)
        yhat_m = np.asarray(self.m_predictions)
        betas = np.asarray(self.betas, dtype=float)
        n = len(betas)

        error_budget = 1. - self.xi

        # print('Lagrangian!!!!!')

        class_opts, K = self._build_options_non_oracle()
        B = int(np.floor(error_budget * n + 1e-12))

        if lambdas is None:
            lambdas = np.concatenate([[0.0], np.logspace(-4, 4, 81)])

        best_theta = None
        best_def = 10**18
        best_err = None

        for lam in lambdas:
            theta = np.zeros(K, float)
            total_def = 0
            total_err = 0
            for opt in class_opts:
                score = opt["def_cnt"] + lam * opt["err_cnt"]
                j = int(np.argmin(score))
                theta[opt["k"]] = float(opt["tau"][j])
                total_def += int(opt["def_cnt"][j])
                total_err += int(opt["err_cnt"][j])

            if total_err <= B and total_def < best_def:
                best_def = total_def
                best_theta = theta
                best_err = total_err

        if best_theta is None:
            theta = np.full(K, np.nextafter(1.0, np.inf), float)
        else:
            theta = best_theta

        accepted = self._apply_multithreshold_from_betas(theta)
        defr = float((~accepted).mean())
        err = float((((accepted) & (yhat_s != y_true)).sum() + ((~accepted) & (yhat_m != y_true)).sum()) / n)
        cost = float(c_s + c_m * defr)

        return theta, {"err": err, "cost": cost, "deferral": defr, "coverage": 1-defr,
                       "error_count_budget": float(B),
                       "chosen_error_count": float(best_err) if best_theta is not None else float((((accepted) & (yhat_s != y_true)).sum() + ((~accepted) & (yhat_m != y_true)).sum())),
                       "chosen_deferred_count": float(best_def) if best_theta is not None else float((~accepted).sum())}
    
    def _build_options_oracle(
        self,
        # y_true: np.ndarray,
        # yhat_s: np.ndarray,
        # betas: np.ndarray,
        K: Optional[int] = None,) -> Tuple[List[Dict[str, np.ndarray]], int]:
        """
        Per predicted class k, create options induced by unique beta levels.
        Oracle: deferred samples contribute 0 error, so error comes only from accepted sLLM mistakes.
        """
        y_true = np.asarray(self.gold_labels)
        yhat_s = np.asarray(self.s_predictions)
        betas = np.asarray(self.betas, dtype=float)
        # n = len(betas)

        if K is None:
            K = int(yhat_s.max()) + 1

        e_s = (yhat_s != y_true).astype(int)

        class_opts: List[Dict[str, np.ndarray]] = []
        for k in range(K):
            idx = np.where(yhat_s == k)[0]
            if idx.size == 0:
                class_opts.append({
                    "k": k,
                    "tau": np.array([np.nextafter(0.0, -np.inf)], float),
                    "def_cnt": np.array([0], int),
                    "err_cnt": np.array([0], int),
                })
                continue

            b = betas[idx]
            es = e_s[idx]

            order = np.argsort(b)
            b = b[order]
            es = es[order]

            u, idx_start, counts = np.unique(b, return_index=True, return_counts=True)
            es_per = np.add.reduceat(es, idx_start)

            # tau = u[j] => defer levels < j, accept >= j
            def_cnt = np.concatenate(([0], counts.cumsum()[:-1]))   # prefix deferred
            err_cnt = es_per[::-1].cumsum()[::-1]                   # suffix accepted s-errors

            # defer-all endpoint
            tau_all_defer = np.nextafter(u.max(), np.inf)
            tau = np.concatenate([u, [tau_all_defer]])
            def_cnt = np.concatenate([def_cnt, [idx.size]])
            err_cnt = np.concatenate([err_cnt, [0]])

            class_opts.append({"k": k, "tau": tau, "def_cnt": def_cnt, "err_cnt": err_cnt})

        return class_opts, K

    def optimize_oracle_mc_dp(
        self,
        # y_true: np.ndarray,
        # yhat_s: np.ndarray,
        # betas: np.ndarray,
        # error_budget: float,
        c_s: float,
        c_m: float,
        # K: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Exact DP: minimize deferrals subject to oracle global error <= error_budget.
        """
        y_true = np.asarray(self.gold_labels)
        yhat_s = np.asarray(self.s_predictions)
        betas = np.asarray(self.betas, dtype=float)
        n = len(betas)

        error_budget = 1. - self.xi

        class_opts, K = self._build_options_oracle()
        B = int(np.floor(error_budget * n + 1e-12))

        states: Dict[int, int] = {0: 0}
        back: List[Dict[int, Tuple[int, int]]] = []

        for opt in class_opts:
            tau, dcnt, ecnt = opt["tau"], opt["def_cnt"], opt["err_cnt"]
            new_states: Dict[int, int] = {}
            new_back: Dict[int, Tuple[int, int]] = {}

            for prev_e, prev_d in states.items():
                for j in range(len(tau)):
                    e = prev_e + int(ecnt[j])
                    if e > B:
                        continue
                    d = prev_d + int(dcnt[j])
                    if (e not in new_states) or (d < new_states[e]):
                        new_states[e] = d
                        new_back[e] = (prev_e, j)

            new_states = self._pareto_prune(new_states)
            new_back = {e: new_back[e] for e in new_states.keys()}
            states = new_states
            back.append(new_back)

        if not states:
            theta = np.full(K, np.nextafter(1.0, np.inf), float)
            accepted = self._apply_multithreshold_from_betas(theta)
            defr = float((~accepted).mean())
            err = float(((accepted) & (yhat_s != y_true)).sum() / n)
            cost = float(c_s + c_m * defr)
            return theta, {"err": err, "cost": cost, "deferral": defr, "coverage": 1-defr}

        best_e = min(states.keys(), key=lambda e: (states[e], e))
        best_def = states[best_e]

        chosen = [0] * K
        cur_e = best_e
        for t in range(len(class_opts)-1, -1, -1):
            prev_e, j = back[t][cur_e]
            k = class_opts[t]["k"]
            chosen[k] = j
            cur_e = prev_e

        theta = np.zeros(K, float)
        for opt in class_opts:
            k = opt["k"]
            theta[k] = float(opt["tau"][chosen[k]])

        accepted = self._apply_multithreshold_from_betas(theta)
        defr = float((~accepted).mean())
        err = float(((accepted) & (yhat_s != y_true)).sum() / n)
        cost = float(c_s + c_m * defr)

        return theta, {"err": err, "cost": cost, "deferral": defr, "coverage": 1-defr,
                       "error_count": float(best_e), "deferred_count": float(best_def)}





