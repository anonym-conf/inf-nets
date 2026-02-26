import numpy as np
from sklearn.isotonic import IsotonicRegression
from abc import ABC, abstractmethod

class Calibrator(ABC):
    def __init__(self): pass

    @abstractmethod
    def fit(self, *args, **kwargs): pass

class TopConfidenceCalibrator(Calibrator):
    """
    Calibrate confidence = max probability using isotonic regression
    on correctness indicator.
    """

    def __init__(self):
        super().__init__()
        self.iso = None

    def fit(self, p_raw, y_true):
        """
        P_raw: shape (N, K) or shape (N,2)
        y_true: shape (N,)
        """
        # print(p_raw, p_raw.shape, np.array(y_true).shape)
        pred = np.argmax(p_raw, axis=1)
        # print(pred, pred.shape)
        conf_raw = np.max(p_raw, axis=1)
        correct = np.array(pred == np.array(y_true)).astype(int)
        print(f'Correct %: {correct.mean()}')
        self.iso = IsotonicRegression(out_of_bounds="clip")
        self.iso.fit(conf_raw, correct)
        return self

    def predict(self, p_raw):
        pred = np.argmax(p_raw, axis=1)
        conf_raw = np.max(p_raw, axis=1)
        conf_cal = self.iso.transform(conf_raw)
        return pred, conf_cal


class PerClassConfidenceCalibrator(Calibrator):
    """
    Per-predicted-class confidence calibration using isotonic regression on correctness.

    Fits one isotonic regressor per predicted class k on the subset {i: y_hat_i == k},
    with binary targets 1[y_true_i == k]. This estimates:
        g_k(s) ≈ P(Y = k | y_hat = k, score = s)
             = P(correct | y_hat = k, score = s)

    Supports two score choices:
      - "pmax": score = max softmax probability (expects p_raw input)
      - "margin": score = top1 - top2 margin (expects logits input)
                 (you can pass raw logits from HF generate; no softmax needed)

    If a class has too few samples, a global fallback isotonic regressor can be used.
    """

    def __init__(self, K, score_type="pmax", min_samples_per_class=20, use_global_fallback=False):
        """
        Args:
            K: number of classes
            score_type: "pmax" or "margin"
            min_samples_per_class: minimum points to fit a per-class isotonic model
            use_global_fallback: if True, fit a global isotonic model and use it when a class model is unavailable
        """
        super().__init__()
        assert score_type in ("pmax", "margin"), "score_type must be 'pmax' or 'margin'"
        self.K = int(K)
        self.score_type = score_type
        self.min_samples_per_class = int(min_samples_per_class)
        self.use_global_fallback = bool(use_global_fallback)

        self.iso_per_class = [None] * self.K
        self.iso_global = None

    @staticmethod
    def _pmax_and_pred_from_probs(p_raw):
        p = np.asarray(p_raw)
        pred = np.argmax(p, axis=1)  # (N, )
        score = np.max(p, axis=1)  # (N,)
        return pred, score

    @staticmethod
    def _margin_and_pred_from_logits(logits):
        z = np.asarray(logits)
        # top-2 logits per row
        top2 = np.partition(z, -2, axis=1)[:, -2:]  # unsorted two largest
        top1 = np.max(top2, axis=1)
        top2_small = np.min(top2, axis=1)
        margin = top1 - top2_small

        pred = np.argmax(z, axis=1)
        return pred, margin

    def _get_pred_and_score(self, arr):
        """
        Depending on score_type, interpret `arr` as:
          - probs p_raw if score_type=="pmax"
          - logits if score_type=="margin"
        """
        if self.score_type == "pmax":
            return self._pmax_and_pred_from_probs(arr)
        else:
            return self._margin_and_pred_from_logits(arr)

    def fit(self, arr, y_true, y_hat=None):
        """
        Fit isotonic regressors.

        Args:
            arr: shape (N, K)
                 - if score_type=="pmax": probabilities (after softmax)
                 - if score_type=="margin": raw logits
            y_true: shape (N,) integer labels in {0,...,K-1}
        """
        y_true = np.asarray(y_true).astype(int)
        pred, score = self._get_pred_and_score(arr)  # (N,), (N,)
        if y_hat is not None:
            pred = np.array(y_hat)  # override pred

        correct = (pred == y_true).astype(int)
        print(f"Overall correct %: {correct.mean():.4f} (N={len(y_true)})")

        # Optional global fallback: P(correct | score)
        if self.use_global_fallback:
            self.iso_global = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.iso_global.fit(score, correct)

        # Per-predicted-class: P(Y=k | y_hat=k, score)
        for k in range(self.K):
            idx = (pred == k)
            n_k = int(idx.sum())
            if n_k < self.min_samples_per_class:
                self.iso_per_class[k] = None
                continue

            target_k = (y_true[idx] == k).astype(int)  # correctness within predicted class k

            iso_k = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso_k.fit(score[idx], target_k)
            self.iso_per_class[k] = iso_k

            print(f"  Class {k}: fit isotonic on n={n_k}, acc={target_k.mean():.4f}")

        return self

    def predict(self, arr, y_hat=None):
        """
        Predict class labels and calibrated correctness probabilities.

        Args:
            arr: shape (N, K)
                 - probs if score_type=="pmax"
                 - logits if score_type=="margin"

        Returns:
            pred: shape (N,) argmax class
            beta_cal: shape (N,) calibrated correctness in [0,1]
        """
        pred, score = self._get_pred_and_score(arr)  # (N,), (N,)
        if y_hat is not None:
            pred = np.array(y_hat)  # override

        beta_cal = np.empty_like(score, dtype=float)

        for i, k in enumerate(pred):  # traverse N 
            iso_k = self.iso_per_class[int(k)]
            if iso_k is not None:
                beta_cal[i] = float(iso_k.transform([score[i]])[0])
            elif self.iso_global is not None:
                beta_cal[i] = float(self.iso_global.transform([score[i]])[0])
            else:
                # last-resort fallback: uncalibrated score (may not be in [0,1] for margin)
                beta_cal[i] = float(score[i])

        return pred, beta_cal

    def set_score_type(self, score_type):
        """
        Change scoring mode (requires refit).
        """
        assert score_type in ("pmax", "margin")
        self.score_type = score_type
        self.iso_per_class = [None] * self.K
        self.iso_global = None
        return self


class SequenceConfidenceCalibrator(Calibrator):
    """
    Calibrate a single scalar confidence score S to an estimated probability of success:
        beta_tilde = g(S) ≈ P(Z=1 | S)
    where Z is a binary indicator (1=acceptable/correct, 0=bad).
    """

    def __init__(self, increasing=True):
        super().__init__()
        self.increasing = bool(increasing)
        self.iso = None

    def fit(self, scores_raw, z_success):
        scores_raw = np.asarray(scores_raw, dtype=float).reshape(-1)
        z_success = np.asarray(z_success, dtype=float).reshape(-1)

        correct = np.array(np.ones_like(z_success) == z_success).astype(int)
        print(f'Correct %: {correct.mean()}')

        assert scores_raw.shape[0] == z_success.shape[0]
        # assert set(np.unique(z_success)).issubset({0, 1})

        self.iso = IsotonicRegression(
            out_of_bounds="clip",
            increasing=self.increasing,
            y_min=0.0,
            y_max=1.0,
        )
        self.iso.fit(scores_raw, z_success)
        return self

    def predict(self, scores_raw):
        scores_raw = np.asarray(scores_raw, dtype=float).reshape(-1)
        beta_cal = self.iso.transform(scores_raw)
        return beta_cal  # shape (N,), values in [0,1]

