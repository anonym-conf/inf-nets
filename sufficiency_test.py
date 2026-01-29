import numpy as np
import pandas as pd
from loader import StructuresLoader
from calibrators import *
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor 
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from tqdm import tqdm

import numpy as np

def augment_binary_probabilities(P, eps=1e-12):
    """
    Augment an (N, 2) array of probabilities with derived features.

    Parameters
    ----------
    P : np.ndarray of shape (N, 2)
        Each row contains probabilities for two classes.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    X_aug : np.ndarray of shape (N, M)
        Augmented feature matrix.
    feature_names : list of str
        Names of the augmented features.
    """

    assert P.ndim == 2 and P.shape[1] == 2, "P must be (N, 2)"

    p0 = P[:, 0]
    p1 = P[:, 1]

    # ---------- Information-theoretic ----------
    entropy = -(p0 * np.log(p0 + eps) + p1 * np.log(p1 + eps))
    gini = 1.0 - (p0**2 + p1**2)

    # ---------- Confidence / sharpness ----------
    p_max = np.maximum(p0, p1)
    p_min = np.minimum(p0, p1)
    margin = p_max - p_min
    abs_diff = np.abs(p0 - p1)
    dist_from_uniform = margin / 2.  # same as margin in binary case

    # ---------- Decision-related ----------
    argmax = np.argmax(P, axis=1)
    one_hot = np.eye(2)[argmax]

    # ---------- Probability transforms ----------
    log_p0 = np.log(p0 + eps)
    log_p1 = np.log(p1 + eps)
    log_odds = np.log((p1 + eps) / (p0 + eps))

    # ---------- Stack all features ----------
    X_aug = np.column_stack([
        # P,                 # original probabilities
        entropy,
        gini,
        p_max,
        p_min,
        margin,
        abs_diff,
        dist_from_uniform,
        log_p0,
        log_p1,
        log_odds,
        one_hot
    ])

    # feature_names = [
    #     # "p0", "p1",
    #     "entropy",
    #     "gini",
    #     "p_max",
    #     "p_min",
    #     "margin",
    #     "abs_diff",
    #     "log_p0",
    #     "log_p1",
    #     "log_odds",
    #     "argmax_is_0",
    #     "argmax_is_1"
    # ]

    return X_aug


train_loader = StructuresLoader(
    dataset_name='sst2', s_llm='gemma3-4b-it', m_llm='qwen3-4b-it'
    )
train_loader.load_dataset()  # loads dataset
train_loader.sLLM_results = train_loader.load_results_file('inference_files/sst2_train_gemma3-4b-it_profiler.pkl')
train_loader.mLLM_results = train_loader.load_results_file('inference_files/sst2_train_qwen3-4b-it_profiler.pkl')

gold_truth_train = [train_loader.dataset['train']['label'][i] for i in range(len(train_loader.dataset['train'].select(range(8000))))] # gold truth
train_texts = [train_loader.dataset['train']['sentence'][i] for i in range(len(train_loader.dataset['train'].select(range(8000))))]
print(f'Dataset length: texts={len(train_texts)}, labels={len(gold_truth_train)}')
sLLM_pred_train = np.array(train_loader.get_result_values(train_loader.sLLM_results, key='pred_idx'))  # pred A
mLLM_pred_train = np.array(train_loader.get_result_values(train_loader.mLLM_results, key='pred_idx'))  # pred B
print(f'Predictions shape: {sLLM_pred_train.shape} and {mLLM_pred_train.shape}')

probs = train_loader.obtain_binary_betas(return_both=True)
print(f'probs shape: {probs.shape}')
# calibrator = TopConfidenceCalibrator()
calibrator = PerClassConfidenceCalibrator(probs.shape[1])
print(f'\nCalibrating confidence...')
calibrator.fit(probs, gold_truth_train)
_, sLLM_betas_train = calibrator.predict(probs)
print(f'calibrated betas shape: {sLLM_betas_train.shape}')

sentence_transformer = SentenceTransformer('all-mpnet-base-v2')

x = sentence_transformer.encode(train_texts, show_progress_bar=True)
# x = augment_binary_probabilities(probs)
print(f'x shape: {x.shape}')

if x.shape[1] > 100:
    svd = TruncatedSVD(n_components=100)
    x = svd.fit_transform(x)
    print(f'svd"ed x shape: {x.shape}')


# Convert predictions to binary errors
E_A = np.array(sLLM_pred_train != gold_truth_train).astype(int)
E_B = np.array(mLLM_pred_train != gold_truth_train).astype(int)

# Delta is the signed difference in errors
Delta = E_A - E_B  # values in {-1, 0, 1}
print(f'Delta shape: {Delta.shape}')

# Use only score from Model A as s
s = sLLM_betas_train  # shape (n_samples,)
print(f's shape: {s.shape}')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

preds_null = np.zeros_like(Delta, dtype=float)
preds_full = np.zeros_like(Delta, dtype=float)

mse_null_folds = []
mse_full_folds = []

for train_idx, test_idx in tqdm(kf.split(x), total=kf.get_n_splits(), desc="cross-testing:"):
    # Null model: only s
    null_model = XGBRegressor(n_estimators=100, objective='reg:squarederror')
    null_model.fit(s[train_idx].reshape(-1, 1), Delta[train_idx])
    preds_null[test_idx] = null_model.predict(s[test_idx].reshape(-1, 1))
    preds_null_fold = null_model.predict(s[test_idx].reshape(-1, 1))
    
    # Full model: s + embeddings
    full_model = XGBRegressor(n_estimators=100, objective='reg:squarederror')
    full_model.fit(
        np.concatenate([s[train_idx].reshape(-1,1), x[train_idx]], axis=1),
        Delta[train_idx]
    )
    preds_full[test_idx] = full_model.predict(
        np.concatenate([s[test_idx].reshape(-1,1), x[test_idx]], axis=1)
    )
    preds_full_fold = full_model.predict(
        np.concatenate([s[test_idx].reshape(-1,1), x[test_idx]], axis=1)
    )

    mse_null_folds.append(np.mean((Delta[test_idx] - preds_null_fold)**2))
    mse_full_folds.append(np.mean((Delta[test_idx] - preds_full_fold)**2))


mse_null = np.mean((Delta - preds_null)**2)
mse_full = np.mean((Delta - preds_full)**2)
print("MSE null:", mse_null,)
print("MSE full:", mse_full,)
print()
print(f'CV mean MSE null (mean of folds) = {np.mean(mse_null_folds)} +- {np.std(mse_null_folds)}')
print(f'CV mean MSE full (mean of folds) = {np.mean(mse_full_folds)} +- {np.std(mse_full_folds)}')

n_permutations = 20
T_obs = mse_null - mse_full
perm_stats = []

for _ in tqdm(range(n_permutations), desc='permutations for t-test:'):
    perm_idx = np.random.permutation(len(x))
    X_perm = x[perm_idx]
    
    # fit full model with permuted X
    perm_preds_full = np.zeros_like(Delta, dtype=float)
    for train_idx, test_idx in kf.split(X_perm):
        full_model = XGBRegressor(n_estimators=100, objective='reg:squarederror')
        full_model.fit(
            np.concatenate([s[train_idx].reshape(-1,1), X_perm[train_idx]], axis=1),
            Delta[train_idx]
        )
        perm_preds_full[test_idx] = full_model.predict(
            np.concatenate([s[test_idx].reshape(-1,1), X_perm[test_idx]], axis=1)
        )
    mse_full_perm = np.mean((Delta - perm_preds_full)**2)
    perm_stats.append(mse_null - mse_full_perm)

# p-value
p_value = (np.sum(np.array(perm_stats) >= T_obs) + 1) / (n_permutations + 1)
print("p-value:", p_value, "+-", np.std(np.array(perm_stats), ddof=1))
print()


# Put them in a DataFrame for convenience
df = pd.DataFrame({
    's': s,
    'Delta': Delta
})


scaler = StandardScaler()
x = scaler.fit_transform(x)


# Add embedding columns
# for j in range(x_scaled.shape[1]):
#     df[f"x_{j}"] = x_scaled[:, j]

df = pd.concat(
    [df, pd.DataFrame(x, columns=[f"x_{j}" for j in range(x.shape[1])])],
    axis=1
)


# 1) Null model (only score s)
null_formula = "Delta ~ s"
null_model = smf.ols(null_formula, data=df).fit()

# 2) Full model (score + embeddings)
cols_Z = df.columns[df.columns.str.startswith("x_")]
full_formula = "Delta ~ s + " + " + ".join(cols_Z)
full_model = smf.ols(full_formula, data=df).fit()

print("Null model summary:\n", null_model.summary())
print("Full model summary:\n", full_model.summary())

# 3) Compare models using ANOVA (F-test)
import statsmodels.api as sm
anova_results = sm.stats.anova_lm(null_model, full_model)
print("ANOVA comparison (F-test):\n", anova_results)

# 1 = Model B better (Delta=+1)
# 0 = Model A better OR tie (Delta=0 or -1)
y_binary = (Delta == 1).astype(int)
df["y_bin"] = y_binary

# Null model (score alone)
null_logit = smf.logit("y_bin ~ s", data=df).fit(disp=False, method='bfgs')

# Full model (score + embeddings)
full_logit = smf.logit("y_bin ~ s + " + " + ".join(cols_Z), data=df).fit(disp=False, method='bfgs')

print("Null logistic regression results:\n", null_logit.summary())
print("Full logistic regression results:\n", full_logit.summary())

# Likelihood ratio test
lr_stat = 2 * (full_logit.llf - null_logit.llf)
df_diff = full_logit.df_model - null_logit.df_model
from scipy.stats import chi2
p_value_lr = chi2.sf(lr_stat, df_diff)

print(f"\nLikelihood ratio test statistic: {lr_stat:.3f}")
print(f"Degrees of freedom: {df_diff}")
print(f"p-value: {p_value_lr:.3g}")





