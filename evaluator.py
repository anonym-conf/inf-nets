import numpy as np
import matplotlib.pyplot as plt
import evaluate, helpers, copy
from sklearn.metrics import classification_report
from typing import Any, Dict, Optional, Union

class Evaluator:
    def __init__(self, betas=None, threshold=None, xi=0., 
                 y_true=None, y_hat_s=None, y_hat_m=None, 
                 oracle=False, test=False):
        self.y_true = np.array(y_true) if y_true is not None else None
        self.y_hat_s = np.array(y_hat_s) if y_hat_s is not None else None
        self.y_hat_m = np.array(y_hat_m) if y_hat_m is not None else None

        if oracle:
            self.y_true = np.array(y_hat_m) if y_hat_m is not None else None
            assert np.array_equal(self.y_true, self.y_hat_m)

        self.betas = np.array(betas) if betas is not None else None
        self.threshold = threshold
        self.xi = xi

        self.test = test

    @staticmethod
    def classification_metrics(y_true, y_hat, model_name, oracle=None):
        if oracle:
            print(f'\nClassification report for {model_name} based on {oracle} as oracle:\n',
                  classification_report(y_true, y_hat, zero_division=1))
            
        else:
            print(f'\nClassification report for {model_name} based on gold truth:\n',
                  classification_report(y_true, y_hat, zero_division=1))
        
        return (np.asarray(y_hat) == np.asarray(y_true)).mean() # for potential use
    
    @staticmethod
    def generation_metrics(y_true, y_hat, model_name, sim=None, oracle=None):
        sim = np.array(sim)
        if oracle:
            print(f'\nMetrics for answer quality for {model_name} based on {oracle} as oracle:\n')
            rouge = evaluate.load("rouge")
            scores = rouge.compute(predictions=y_hat, references=y_true, use_stemmer=True)
            for k, v in scores.items():
                print(f"{k}: {v:.4f}")
            
            print(f'Average similarity: {sim.mean()}')
        else:
            print(f'\nMetrics for answer quality for {model_name} based on gold truth:\n')
            rouge = evaluate.load("rouge")
            scores = rouge.compute(predictions=y_hat, references=y_true, use_stemmer=True)
            for k, v in scores.items():
                print(f"{k}: {v:.4f}")
            
            print(f'Average similarity: {sim.mean()}')
            return sim.mean()
            




    def coverage_metrics(self, c_s, c_m):
        set_ = 'test' if self.test else 'train'
        print(f'\nEvaluating on {set_} in oracle setting...')
        print(f'Optimal threshold: {self.threshold}')

        if np.isscalar(self.threshold):
            accepted = self.betas >= self.threshold
        else:
            accepted = self.betas >= self.threshold[self.y_hat_s]
        coverage = accepted.mean()
        deferral = 1.0 - coverage
        error_masked = np.sum(self.y_hat_s[accepted] != self.y_true[accepted]) / len(self.y_true)
        cost_masked = c_s + c_m * deferral  # TODO: cost list (maybe)

        cascade_error = np.mean(np.where(accepted, self.y_hat_s, self.y_true) != self.y_true)

        print(f'Accepted samples on {set_}: {accepted.sum()}/{len(self.y_true)}')
        print(f'Coverage on {set_}: {coverage}')
        print(f'Deferral on {set_}: {deferral}')
        print(f'Error on {set_} = {error_masked} with cost = {cost_masked}')
        print(f'Full cascade error on {set_} = {cascade_error}')

        print(f'Accuracy on {set_} = {1. - error_masked}')

        # extreme cases
        print(f'Error on {set_} with none deferred = {np.sum(self.y_hat_s != self.y_true) / len(self.y_true)} '
              f'with cost = {c_s}')
        print(f'Error on {set_} with all deferred = {0.} with cost = {c_s + c_m}')

        cost_saved = ((c_s + c_m) - cost_masked) / (c_s + c_m) * 100
        print(f'We saved {(c_s + c_m - cost_masked):.3f} ({cost_saved:.3f}%) in cost '
              f'while having {error_masked:.3f} absolute error increase!')


        return error_masked, cost_masked

    def coverage_metrics_non_oracle(self, c_s, c_m, gold_truth):
        set_ = 'test' if self.test else 'train'
        gold_truth = np.array(gold_truth)
        print(f'\nEvaluating on {set_} in non-oracle setting...')
        print(f'Optimal threshold: {self.threshold}')
        
        if np.isscalar(self.threshold):
            accepted = self.betas >= self.threshold
        else:
            accepted = self.betas >= self.threshold[self.y_hat_s]
        coverage = accepted.mean()
        deferral = 1.0 - coverage

        y_hat_cascade = np.where(accepted, self.y_hat_s, self.y_hat_m)

        error_masked = np.sum(y_hat_cascade != gold_truth) / len(gold_truth)
        cost_masked = c_s + c_m * deferral  # TODO: cost list (maybe)

        cascade_error = np.mean(np.where(accepted, self.y_hat_s, self.y_hat_m) != gold_truth)

        print(f'Accepted samples on {set_}: {accepted.sum()}/{len(gold_truth)}')
        print(f'Coverage on {set_}: {coverage}')
        print(f'Deferral on {set_}: {deferral}')
        print(f'Error on {set_} = {error_masked} with cost = {cost_masked}')
        print(f'Full cascade error on {set_} = {cascade_error}')

        print(f'Accuracy on {set_} = {1. - error_masked}')

        # random deferral
        rng = np.random.default_rng(getattr(self, "random_seed", 42))

        p_defer = 0.5
        accepted_rand = rng.random(len(gold_truth)) >= p_defer  # accept w.p. 0.5, defer w.p. 0.5
        coverage_rand = accepted_rand.mean()
        deferral_rand = 1.0 - coverage_rand

        y_hat_random = np.where(accepted_rand, self.y_hat_s, self.y_hat_m)
        random_error = np.mean(y_hat_random != gold_truth)
        random_cost = c_s + c_m * deferral_rand

        print(f'Error on {set_} with random deferral (p_defer={p_defer}) = {random_error} '
              f'with cost = {random_cost} (coverage={coverage_rand}, deferral={deferral_rand})')
        print(f'Random Accuracy on {set_} = {1. - random_error}')

        # extreme cases
        print(f'Error on {set_} with none deferred = {np.sum(self.y_hat_s != gold_truth) / len(gold_truth)} '
              f'with cost = {c_s}')
        all_deferred_error = np.sum(self.y_hat_m != gold_truth) / len(gold_truth) 
        print(f'Error on {set_} with all deferred = {all_deferred_error} '
              f'with cost = {c_s + c_m}')

        cost_saved = ((c_s + c_m) - cost_masked) / (c_s + c_m) * 100
        loss_to_all_defer = cascade_error - all_deferred_error
        error_sign = 'increase' if loss_to_all_defer > 0 else 'increase'
        print(f'We saved {(c_s + c_m - cost_masked):.3f} ({cost_saved:.3f}%) in cost '
              f'while having {loss_to_all_defer:.3f} absolute error {error_sign} (based on gold truth)!\n')


        return error_masked, cost_masked

    def evaluate_policy(
        self,
        # y_true: np.ndarray,
        # yhat_s: np.ndarray,
        # proba_s: np.ndarray,
        # theta: Union[float, np.ndarray],
        *,
        mode: str = "non_oracle",  # "oracle" or "non_oracle"
        discr_: bool = True,
        # yhat_m: Optional[np.ndarray] = None,  # required if non_oracle
        # constant costs:
        c_s: Optional[float] = None,
        c_m: Optional[float] = None,
        # per-sample costs (override constants if provided):
        c_s_i: Optional[np.ndarray] = None,
        c_m_i: Optional[np.ndarray] = None,
        # accept rule:
        accept_ge: bool = True,  # accept if beta >= theta (True) else beta > theta
        gold_truth: Optional[list] = None # bypass oracle y_true
    ) -> Dict[str, Any]:
        """
        Evaluate a (single- or multi-) threshold deferral policy using pure MC indicators.
    
        Policy:
          - small model predicts yhat_s and has probabilities proba_s
          - beta_i = proba_s[i, yhat_s[i]]
          - accept if beta_i >= theta[yhat_s[i]] (or > if accept_ge=False)
          - if accept: output yhat_s
          - if defer: output yhat_m (non_oracle) or correct (oracle)
    
        Inputs:
          theta: float (single-threshold) or np.ndarray shape (K,) (multi-threshold)
    
        Returns a dict with:
          - coverage, deferral
          - global_error (oracle/non-oracle), conditional_error_on_accepted, conditional_error_on_deferred (non-oracle)
          - cost_total, cost_avg
          - per-class breakdown by predicted class of small model
          - counts: N, accepted_count, deferred_count
        """
        # y_true = np.asarray(y_true)
        # yhat_s = np.asarray(yhat_s)
        # proba_s = np.asarray(proba_s, dtype=float)  # array of (N,)
        # if proba_s.ndim > 1: N, K = proba_s.shape
        # else: N, K = proba_s.shape[0], 1
        N = len(self.y_hat_s)
        K = len(np.unique(self.y_true)) if discr_ else 0
    
        if mode not in {"oracle", "non_oracle"}:
            raise ValueError("mode must be 'oracle' or 'non_oracle'")
    
        if mode == "non_oracle":
            if self.y_hat_m is None:
                raise ValueError("yhat_m is required for mode='non_oracle'")
            # yhat_m = np.asarray(yhat_m)
    
        # Build theta vector
        # if np.isscalar(self.threshold):
        #     theta_vec = np.full(K, float(self.threshold), dtype=float)
        theta_vec = None
        if not np.isscalar(self.threshold):
            theta_vec = np.asarray(self.threshold, dtype=float)
            if theta_vec.shape != (K,):
                raise ValueError(f"theta must have shape (K,), got {theta_vec.shape}")
    
        # Confidences
        # beta = proba_s[np.arange(N)] if K < 2 else proba_s[np.arange(N), self.y_hat_s]
    
        # Accept mask
        if accept_ge:
            if np.isscalar(self.threshold): accepted = self.betas >= self.threshold
            else: accepted = self.betas >= theta_vec[self.y_hat_s]
        else:
            if np.isscalar(self.threshold): accepted = self.betas > self.threshold
            else: accepted = self.betas > theta_vec[self.y_hat_s]

        deferred = ~accepted

        y_hat_cascade = np.where(accepted, self.y_hat_s, self.y_hat_m)
        # random deferral
        rng = np.random.default_rng(getattr(self, "random_seed", 42))
        p_defer = 0.5
        accepted_rand = rng.random(len(self.y_true)) >= p_defer  # accept w.p. 0.5, defer w.p. 0.5
        coverage_rand = accepted_rand.mean()
        deferral_rand = 1.0 - coverage_rand
        y_hat_random = np.where(accepted_rand, self.y_hat_s, self.y_hat_m)
        random_error = np.mean(y_hat_random != self.y_true)
        random_cost = c_s + c_m * deferral_rand

        # keep texts...
        if not discr_:
        # if np.isscalar(self.threshold):
            gold_truth_txt =  copy.deepcopy(gold_truth)
            gold_truth = np.ones(shape=(len(self.y_true,))) if gold_truth is not None else None

            self.y_true_txt = copy.deepcopy(self.y_true)
            self.y_true = np.ones(shape=(len(self.y_true,)))

            self.y_hat_s_txt = copy.deepcopy(self.y_hat_s)
            self.y_hat_s = helpers.get_binary_labels(y_true=self.y_true_txt, y_hat=self.y_hat_s_txt)

            self.y_hat_m_txt = copy.deepcopy(self.y_hat_m)
            self.y_hat_m = helpers.get_binary_labels(y_true=self.y_true_txt, y_hat=self.y_hat_m_txt)

            self.y_hat_cascade_txt = copy.deepcopy(y_hat_cascade)
            y_hat_cascade = helpers.get_binary_labels(y_true=self.y_true_txt, y_hat=self.y_hat_cascade_txt)

            self.y_hat_random_txt = copy.deepcopy(y_hat_random)
            y_hat_random = helpers.get_binary_labels(y_true=self.y_true_txt, y_hat=self.y_hat_random_txt)

        old_y_true = None
        if gold_truth is not None: # bypass oracle argument to get statistics for non-oracle
            old_y_true = np.array(self.y_true)  # copy current y_true (y_hat_m in oracle setting)
            self.y_true = np.array(gold_truth)
    
        # Errors
        err_s = (self.y_hat_s != self.y_true)
        if mode == "oracle":
            err_total_count = int((accepted & err_s).sum())
            err_deferred_count = 0
        else:
            err_m = (self.y_hat_m != self.y_true)
            err_total_count = int((accepted & err_s).sum() + (deferred & err_m).sum())
            err_deferred_count = int((deferred & err_m).sum())
    
        err_accepted_count = int((accepted & err_s).sum())
    
        global_error = err_total_count / N

        if gold_truth is not None: assert global_error == np.sum(y_hat_cascade != gold_truth) / N
        else: assert global_error == np.sum(y_hat_cascade != self.y_true) / N

        accepted_count = int(accepted.sum())
        deferred_count = int(deferred.sum())
        coverage = accepted_count / N
        deferral = deferred_count / N
    
        # Conditional errors (sometimes useful but not what the constraint uses)
        # cond_err_accepted = (err_accepted_count / accepted_count) if accepted_count > 0 else 0.0
        # if mode == "non_oracle":
        #     cond_err_deferred = (err_deferred_count / deferred_count) if deferred_count > 0 else 0.0
        # else:
        #     cond_err_deferred = 0.0
    
        # Costs
        # Always pay small-model cost; pay big-model cost only when deferred.
        if c_s_i is not None:
            c_s_i = np.asarray(c_s_i, dtype=float)
            if c_s_i.shape != (N,):
                raise ValueError("c_s_i must have shape (N,)")
            cost_s_total = float(c_s_i.sum())
        else:
            if c_s is None:
                raise ValueError("Provide either c_s (scalar) or c_s_i (per-sample)")
            cost_s_total = float(c_s) * N
    
        if c_m_i is not None:
            c_m_i = np.asarray(c_m_i, dtype=float)
            if c_m_i.shape != (N,):
                raise ValueError("c_m_i must have shape (N,)")
            cost_m_total = float(c_m_i[deferred].sum())
        else:
            if c_m is None:
                raise ValueError("Provide either c_m (scalar) or c_m_i (per-sample)")
            cost_m_total = float(c_m) * deferred_count
    
        cost_total = cost_s_total + cost_m_total
        cost_avg = cost_total / N

        # comparisons
        all_deferred_error = np.sum(self.y_hat_m != self.y_true) / len(self.y_true)
        none_deferred_error = np.sum(self.y_hat_s != self.y_true) / len(self.y_true)

    
        # Per-class breakdown by small-model predicted class
        per_class = {}
        for k in range(K):
            idx = np.where(self.y_hat_s == k)[0]
            nk = idx.size
            if nk == 0:
                continue
            acc_k = accepted[idx]
            def_k = ~acc_k
            cov_k = float(acc_k.mean())
            defr_k = 1.0 - cov_k
    
            # global error contribution inside this group (still divided by nk here)
            err_s_k = (self.y_hat_s[idx] != self.y_true[idx])
            err_acc_k = int((acc_k & err_s_k).sum())
    
            if mode == "oracle":
                err_tot_k = err_acc_k
                err_def_k = 0
            else:
                err_m_k = (self.y_hat_m[idx] != self.y_true[idx])
                err_def_k = int((def_k & err_m_k).sum())
                err_tot_k = err_acc_k + err_def_k
    
            per_class[int(k)] = {
                "n": int(nk),
                "coverage": cov_k,
                "deferral": defr_k,
                "global_error_within_class": err_tot_k / nk,  # within group
                "accepted_error_within_class": (err_acc_k / max(int(acc_k.sum()), 1)),
                "deferred_error_within_class": (err_def_k / max(int(def_k.sum()), 1)) if mode == "non_oracle" else 0.0,
            }
        
        precision_all_deferred, recall_all_deferred, f1_all_deferred  = None, None, None
        precision_none_deferred, recall_none_deferred, f1_none_deferred = None, None, None
        precision_random_deferred, recall_random_deferred, f1_random_deferred = None, None, None
        precision, recall, f1 = None, None, None

        rouge_all_deferred, sim_all_deferred = None, None
        rouge_none_deferred, sim_none_deferred = None, None
        rouge_random_deferred, sim_random_deferred = None, None
        rouge, sim = None, None
        # rouge_scores = None
        if discr_:
        # if not np.isscalar(self.threshold):
            if gold_truth is not None:

                self.y_true = np.array(old_y_true)  # revert back to oracle y_true for possible next calculations
                del old_y_true
                report = classification_report(y_pred=y_hat_cascade, y_true=gold_truth, output_dict=True)
                precision = report['macro avg']['precision']
                recall = report['macro avg']['recall']
                f1 = report['macro avg']['f1-score']

                all_deferred_report = classification_report(y_pred=self.y_hat_m, y_true=gold_truth, output_dict=True)
                precision_all_deferred = all_deferred_report['macro avg']['precision']
                recall_all_deferred = all_deferred_report['macro avg']['recall']
                f1_all_deferred = all_deferred_report['macro avg']['f1-score']

                none_deferred_report = classification_report(y_pred=self.y_hat_s, y_true=gold_truth, output_dict=True)
                precision_none_deferred = none_deferred_report['macro avg']['precision']
                recall_none_deferred = none_deferred_report['macro avg']['recall']
                f1_none_deferred = none_deferred_report['macro avg']['f1-score']

                random_deferred_report = classification_report(y_pred=y_hat_random, y_true=gold_truth, output_dict=True)
                precision_random_deferred = random_deferred_report['macro avg']['precision']
                recall_random_deferred = random_deferred_report['macro avg']['recall']
                f1_random_deferred = random_deferred_report['macro avg']['f1-score']

            else:  # non-oracle
                report = classification_report(y_pred=y_hat_cascade, y_true=self.y_true, output_dict=True)
                precision = report['macro avg']['precision']
                recall = report['macro avg']['recall']
                f1 = report['macro avg']['f1-score']

                all_deferred_report = classification_report(y_pred=self.y_hat_m, y_true=self.y_true, output_dict=True)
                precision_all_deferred = all_deferred_report['macro avg']['precision']
                recall_all_deferred = all_deferred_report['macro avg']['recall']
                f1_all_deferred = all_deferred_report['macro avg']['f1-score']

                none_deferred_report = classification_report(y_pred=self.y_hat_s, y_true=self.y_true, output_dict=True)
                precision_none_deferred = none_deferred_report['macro avg']['precision']
                recall_none_deferred = none_deferred_report['macro avg']['recall']
                f1_none_deferred = none_deferred_report['macro avg']['f1-score']

                random_deferred_report = classification_report(y_pred=y_hat_random, y_true=self.y_true, output_dict=True)
                precision_random_deferred = random_deferred_report['macro avg']['precision']
                recall_random_deferred = random_deferred_report['macro avg']['recall']
                f1_random_deferred = random_deferred_report['macro avg']['f1-score']
        else:
            # need text for these...
            rouge_scorer = evaluate.load("rouge")
            if gold_truth is not None:
                rouge = float(rouge_scorer.compute(predictions=self.y_hat_cascade_txt, references=gold_truth_txt, use_stemmer=True)['rougeL'])
                sim = float(helpers.get_binary_labels(
                    y_true=gold_truth_txt, y_hat=self.y_hat_cascade_txt, return_sim=True
                ).mean())

                rouge_all_deferred = float(rouge_scorer.compute(predictions=self.y_hat_m_txt, references=gold_truth_txt, use_stemmer=True)['rougeL'])
                sim_all_deferred = float(helpers.get_binary_labels(
                    y_true=gold_truth_txt, y_hat=self.y_hat_m_txt, return_sim=True
                ).mean())

                rouge_none_deferred = float(rouge_scorer.compute(predictions=self.y_hat_s_txt, references=gold_truth_txt, use_stemmer=True)['rougeL'])
                sim_none_deferred = float(helpers.get_binary_labels(
                    y_true=gold_truth_txt, y_hat=self.y_hat_s_txt, return_sim=True
                ).mean())

                rouge_random_deferred = float(rouge_scorer.compute(predictions=self.y_hat_random_txt, references=gold_truth_txt, use_stemmer=True)['rougeL'])
                sim_random_deferred = float(helpers.get_binary_labels(
                    y_true=gold_truth_txt, y_hat=self.y_hat_random_txt, return_sim=True
                ).mean())
            else:
                rouge = float(rouge_scorer.compute(predictions=self.y_hat_cascade_txt, references=self.y_true_txt, use_stemmer=True)['rougeL'])
                sim = float(helpers.get_binary_labels(
                    y_true=self.y_true_txt, y_hat=self.y_hat_cascade_txt, return_sim=True
                ).mean())

                rouge_all_deferred = float(rouge_scorer.compute(predictions=self.y_hat_m_txt, references=self.y_true_txt, use_stemmer=True)['rougeL'])
                sim_all_deferred = float(helpers.get_binary_labels(
                    y_true=self.y_true_txt, y_hat=self.y_hat_m_txt, return_sim=True
                ).mean())

                rouge_none_deferred = float(rouge_scorer.compute(predictions=self.y_hat_s_txt, references=self.y_true_txt, use_stemmer=True)['rougeL'])
                sim_none_deferred = float(helpers.get_binary_labels(
                    y_true=self.y_true_txt, y_hat=self.y_hat_s_txt, return_sim=True
                ).mean())

                rouge_random_deferred = float(rouge_scorer.compute(predictions=self.y_hat_random_txt, references=self.y_true_txt, use_stemmer=True)['rougeL'])
                sim_random_deferred = float(helpers.get_binary_labels(
                    y_true=self.y_true_txt, y_hat=self.y_hat_random_txt, return_sim=True
                ).mean())

    
        return {
            "N": int(N),
            "K": int(K),
            "mode": mode,
            "accept_ge": bool(accept_ge),
            "theta": self.threshold.tolist() if not np.isscalar(self.threshold) else self.threshold,
            "policy_coverage": float(coverage),
            "policy_deferral": float(deferral),
            "policy_global_error": float(global_error),

            "all_deferred_error": float(all_deferred_error), 
            "all_deferred_accuracy": 1. - float(all_deferred_error),
            "all_deferred_cost": float(c_s + c_m),
            "all_deferred_precision": precision_all_deferred,
            "all_deferred_recall": recall_all_deferred,
            "all_deferred_f1": f1_all_deferred,
            "all_deferred_rouge": rouge_all_deferred,
            "all_deferred_sim": sim_all_deferred,
            "all_deferred_cost_saved": ((c_m) - float(c_s + c_m)) / (c_m) * 100,  # big-only baseline
            # "all_efficiency_ratio_f1": (f1_all_deferred or 0.) / float(c_s + c_m),
            # "all_efficiency_ratio_accuracy": ((1. - float(all_deferred_error) or 0.)) / float(c_s + c_m),
            # "all_efficiency_ratio_cosine": (sim_all_deferred or 0.) / float(c_s + c_m),
            # "all_utility_f1": (f1_all_deferred or 0.) - (float(c_s + c_m) / float(c_s + c_m)),
            # "all_utility_accuracy": ((1. - float(all_deferred_error)) or 0.) - (float(c_s + c_m) / float(c_s + c_m)),
            # "all_utility_cosine": (sim_all_deferred or 0.) - (float(c_s + c_m) / float(c_s + c_m)),
            

            "none_defereed_error": float(none_deferred_error),
            "none_deferred_accuracy": 1.-float(none_deferred_error),
            "none_deferred_cost": float(c_s),
            "none_deferred_precision": precision_none_deferred ,
            "none_deferred_recall": recall_none_deferred,
            "none_deferred_f1": f1_none_deferred,
            "none_deferred_rouge": rouge_none_deferred,
            "none_deferred_sim": sim_none_deferred,
            "none_deferred_cost_saved": ((c_m) - float(c_s)) / (c_m) * 100,  # big-only baseline
            # "none_efficiency_ratio_f1": (f1_none_deferred or 0.) / float(c_s),
            # "none_efficiency_ratio_accuracy": ((1. - float(none_deferred_error) or 0.)) / float(c_s),
            # "none_efficiency_ratio_cosine": (sim_none_deferred or 0.) / float(c_s),
            # "none_utility_f1": (f1_none_deferred or 0.) - (float(c_s) / float(c_s + c_m)),
            # "none_utility_accuracy": ((1. - float(none_deferred_error)) or 0.) - (float(c_s) / float(c_s + c_m)),
            # "none_utility_cosine": (sim_none_deferred or 0.) - (float(c_s) / float(c_s + c_m)),

            "random_defereed_error": float(random_error),
            "random_deferred_accuracy": 1.-float(random_error),
            "random_deferred_cost": float(random_cost),
            "random_deferred_precision": precision_random_deferred,
            "random_deferred_recall": recall_random_deferred,
            "random_deferred_f1": f1_random_deferred,
            "random_deferral": float(deferral_rand),
            "random_coverage": float(coverage_rand),
            "random_cost": float(random_cost),
            # "random_cost_saved": ((c_s + c_m) - float(random_cost)) / (c_s + c_m) * 100,
            "random_cost_saved": ((c_m) - float(random_cost)) / (c_m) * 100,
            # "random_cost_saved": ((c_s + c_m) - float(random_cost)) / (c_m) * 100,

            "random_deferred_rouge": rouge_random_deferred,
            "random_deferred_sim": sim_random_deferred,
            # "random_efficiency_ratio_f1": (f1_random_deferred or 0.) / float(random_cost),
            # "random_efficiency_ratio_accuracy": ((1. - float(random_error) or 0.)) / float(random_cost),
            # "random_efficiency_ratio_cosine": (sim_random_deferred or 0.) / float(random_cost),
            # "random_utility_f1": (f1_random_deferred or 0.) - (float(random_cost) / float(c_s + c_m)),
            # "random_utility_accuracy": ((1. - float(random_error)) or 0.) - (float(random_cost) / float(c_s + c_m)),
            # "random_utility_cosine": (sim_random_deferred or 0.) - (float(random_cost) / float(c_s + c_m)),

            "final_error_received": float(all_deferred_error-global_error),
            "final_accuracy_penalty": 1. - float(all_deferred_error) - (1. - float(global_error)),
            # "conditional_error_accepted": float(cond_err_accepted),
            # "conditional_error_deferred": float(cond_err_deferred),
            "accepted_count": int(accepted_count),
            "deferred_count": int(deferred_count),
            "error_count_total": int(err_total_count),
            "error_count_accepted": int(err_accepted_count),
            "error_count_deferred": int(err_deferred_count),
            "cost_total": float(cost_total),
            "policy_cost_avg": float(cost_avg),
            "policy_accuracy": 1. - float(global_error),
            "policy_precision": precision,
            "policy_recall": recall,
            "policy_f1": f1,
            "policy_rouge": rouge,
            "policy_sim": sim,
            "cost_small_total": float(cost_s_total),
            "cost_master_total": float(cost_m_total),
            # "policy_cost_saved": ((c_s + c_m) - float(cost_avg)) / (c_s + c_m) * 100,  # worst-case policy
            "policy_cost_saved": ((c_m) - float(cost_avg)) / (c_m) * 100,  # big-only baseline
            # "policy_cost_saved": ((c_s + c_m) - float(cost_avg)) / (c_m) * 100,  # avoidable cost
            # "policy_efficiency_ratio_f1": (f1_all_deferred or 0.) / float(cost_avg),
            # "policy_utility_f1": (f1_all_deferred or 0.) - (float(cost_avg) / float(c_s + c_m)),
            # "policy_efficiency_ratio_accuracy": ((1. - float(global_error) or 0.)) / float(cost_avg),
            # "policy_utility_accuracy": ((1. - float(global_error)) or 0.) - (float(cost_avg) / float(c_s + c_m)),
            # "policy_efficiency_ratio_cosine": (sim or 0.) / float(cost_avg),
            # "policy_utility_cosine": (sim or 0.) - (float(cost_avg) / float(c_s + c_m))
            # "per_class": per_class,
        }


    def threshold_plots(self, c_s, c_m, error_ind, cost_ind):

        set_ = 'Test' if self.test else 'Train'


        thetas = np.linspace(0, 1, 201)
        error_ind_curve = np.array([np.mean((self.betas >= t) & (self.y_true != self.y_hat_s)) for t in thetas])
        cost_curve = np.array([c_s + c_m * np.mean(self.betas < t) for t in thetas])


        # --- Plot 1: Error vs θ ---
        plt.figure(figsize=(9, 6))
        plt.plot(thetas, error_ind_curve, label='Error (MC indicators)')
        plt.axvline(self.threshold, linestyle=':', label=f'θ* indicator = {self.threshold:.3f} (error = {error_ind:.3f})')
        plt.axhline(1 - self.xi, linestyle='--', label=f'constraint = {1 - self.xi:.3f}')
        plt.scatter(self.threshold, error_ind, color='red', marker='x')
        plt.axhline(np.sum(self.y_hat_s != self.y_true) / len(self.y_true), linestyle='--', color='orange',
                    label=f'Error on 0% deferrals')
        plt.xlabel('θ')
        plt.ylabel('Error(θ)')
        plt.title(f'Error vs Threshold θ ({set_})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot 2: Cost vs θ ---
        plt.figure(figsize=(9, 6))
        plt.plot(thetas, cost_curve, label='Cost(θ) = c_s + c_m·P(β<θ)')
        plt.scatter(self.threshold, cost_ind, color='red', marker='x')
        plt.axvline(self.threshold, linestyle=':', label=f'θ* indicator = {self.threshold:.3f} (cost = {cost_ind:.3f})')
        plt.axhline(c_s + c_m, linestyle='--', color='orange', label=f'Cost on 100% deferrals')
        plt.axhline(c_s, linestyle='--', color='purple', label=f'Cost on 0% deferrals')
        plt.xlabel('θ')
        plt.ylabel('Cost(θ)')
        plt.title(f'Cost vs Threshold θ ({set_})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def threshold_plots_non_oracle(self, c_s, c_m, error_ind, cost_ind, gold_truth,
                                   filename_error=None, filename_cost=None, rouge=False):
        # gold_truth = np.array(gold_truth)



        error_ind = [error_ind] * len(self.threshold) if not np.isscalar(self.threshold) else error_ind
        cost_ind = [cost_ind] * len(self.threshold) if not np.isscalar(self.threshold) else cost_ind
        accepted = self.betas >= self.threshold if np.isscalar(self.threshold) else self.betas >= self.threshold[self.y_hat_s]

        thetas = np.linspace(0, 1, 201) # if np.isscalar(self.threshold) else [np.linspace(0, 1, 201) for _ in len(np.unique(self.y_true))]

        if rouge:
            # 1) cascade prediction for each sample (based on your existing `accepted`)
            # y_hat_cascade = np.where(accepted, self.y_hat_s_txt, self.y_hat_m_txt)

            # 2) ROUGE-based curve: use loss = 1 - ROUGE (so higher = worse, like error)
            # rouge_vals = helpers.rouge_per_sample(list(y_hat_cascade), list(gold_truth), rouge_types=("rougeL",))
            # loss = 1.0 - rouge_vals

            rouge_s = helpers.rouge_per_sample(self.y_hat_s_txt, gold_truth, rouge_types=("rougeL",))
            rouge_m = helpers.rouge_per_sample(self.y_hat_m_txt, gold_truth, rouge_types=("rougeL",))

            # print("rouge_s mean/min/max:", rouge_s.mean(), rouge_s.min(), rouge_s.max())
            # print("rouge_m mean/min/max:", rouge_m.mean(), rouge_m.min(), rouge_m.max())
            # print("ROUGE(t=0) should ~ mean(rouge_s):", rouge_curve[0], rouge_s.mean())
            # print("ROUGE(t=1) should ~ mean(rouge_m):", rouge_curve[-1], rouge_m.mean())

            error_ind_curve = np.array([
                np.mean(np.where(self.betas >= t, rouge_s, rouge_m))
                for t in thetas
            ])

            # if you want "error" instead of ROUGE:
            # loss_curve = 1.0 - rouge_curve

            # error_ind_curve = np.array([np.mean((self.betas >= t) * rouge_vals) for t in thetas])
        else:
            # y_hat_cascade = np.where(accepted, self.y_hat_s, self.y_hat_m)
            # error_ind_curve = np.array([np.mean((self.betas >= t) & (y_hat_cascade != np.ones(shape=(len(gold_truth,))))) for t in thetas])
            err_s = (self.y_hat_s != np.ones(shape=(len(gold_truth))))  # (N,)
            err_m = (self.y_hat_m != np.ones(shape=(len(gold_truth))))  # (N,)

            error_ind_curve = np.array([
                np.mean(np.where(self.betas >= t, err_s, err_m))
                for t in thetas
            ])
            
        cost_curve = np.array([c_s + c_m * np.mean(self.betas < t) for t in thetas])

        # --- Plot 1: Error vs θ ---
        plt.rcParams.update({
            'font.size': 12,
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titlesize': 14,
            'axes.labelsize': 14,
            'legend.fontsize': 10,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12, 
        })
        plt.figure(figsize=(9, 6))
        plt.plot(thetas, error_ind_curve, label=f'{"Error" if not rouge else "ROUGE"} (Monte Carlo sampling)', linewidth=3.5)
        if np.isscalar(self.threshold):
            plt.axvline(self.threshold, linestyle=':',
                        label=f'θ* = {self.threshold:.3f} ({"error" if not rouge else "ROUGE"} = {error_ind:.3f})', linewidth=3.5)
        if rouge: pass
        else: plt.axhline(1 - self.xi, linestyle='--', label=f'constraint on training = {1 - self.xi:.3f}', linewidth=3.5)
        plt.scatter(self.threshold, error_ind, color='red', marker='x', s=150)
        if rouge: 
            rouge_small = helpers.rouge_per_sample(list(self.y_hat_s_txt), list(gold_truth), rouge_types=("rougeL",)).mean()
            rouge_big = helpers.rouge_per_sample(list(self.y_hat_m_txt), list(gold_truth), rouge_types=("rougeL",)).mean()
            plt.axhline(rouge_small, linestyle='--', color='orange', label=f'ROUGE on 0% deferrals',linewidth=3.5)
            plt.axhline(rouge_big, linestyle='--', color='purple', label=f'ROUGE on 100% deferrals', linewidth=3.5)
        else:
            plt.axhline(np.sum(self.y_hat_s != np.ones(shape=(len(gold_truth,)))) / len(gold_truth), linestyle='--', color='orange',
                        label=f'Error on 0% deferrals', linewidth=3.5)
            plt.axhline(np.sum(self.y_hat_m != np.ones(shape=(len(gold_truth,)))) / len(gold_truth), linestyle='--', color='purple',
                        label=f'Error on 100% deferrals', linewidth=3.5)
        plt.xlabel('θ', )
        plt.ylabel('Error(θ)',)
        # plt.title(f'Error vs Threshold θ (Test - Non-oracle)')
        plt.legend(prop={'weight': 'bold'})
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        if filename_error: plt.savefig(filename_error)

        # --- Plot 2: Cost vs θ ---
        plt.figure(figsize=(9, 6))
        plt.plot(thetas, cost_curve, label='Cost (Monte Carlo sampling)', linewidth=3.5)
        plt.scatter(self.threshold, cost_ind, color='red', marker='x', s=150)
        if np.isscalar(self.threshold):
            plt.axvline(self.threshold, linestyle=':', label=f'θ* indicator = {self.threshold:.3f} (cost = {cost_ind:.3f})', linewidth=3.5)
        plt.axhline(c_s + c_m, linestyle='--', color='orange', label=f'Cost on 100% deferrals', linewidth=3.5)
        plt.axhline(c_s, linestyle='--', color='purple', label=f'Cost on 0% deferrals', linewidth=3.5)
        plt.xlabel('θ', )
        plt.ylabel('Cost(θ)', )
        # plt.title(f'Cost vs Threshold θ (Test - Non-oracle)')
        plt.legend(prop={'weight': 'bold'})
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        if filename_cost: plt.savefig(filename_cost)


    def reset(self):
        self.y_true = None
        self.y_hat_s = None
        self.y_hat_m = None
        self.betas = None
        self.xi = None
        self.threshold = None
