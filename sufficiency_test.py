import numpy as np
import pandas as pd
import helpers
from loader import StructuresLoader
from calibrators import *
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor 
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--sLLM', default='gemma3-1b-it', type=str,
                    choices=['gemma3-1b-it', 'gemma3-4b-it', 'gemma3-12b-it', 'qwen3-4b-it', 'ministral3-3b-it'], help='Name of sLLM.')
parser.add_argument('--mLLM', default='gemma3-4b-it', type=str,
                    choices=['gemma3-4b-it', 'gemma3-12b-it', 'gemma3-27b-it', 'qwen3-4b-it', 'ministral3-3b-it'], help='Name of mLLM.')
parser.add_argument('--dataset', default='sst2', type=str,
                    choices=['sst2', 'emotion', 'agnews', 'fakenews', 'squad', 'wmt'], help='Name of dataset.')
parser.add_argument('--calibration', default='multi', type=str, action='store_true', help='Method of calibration.',
                    choices=['multi', 'sequence', ])
parser.add_argument('--embeddings', default=False, action='store_true', help='Extra information (embeddings or logits statistics).')

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

def augment_multiclass_probabilities(P, eps=1e-12):
    """
    Augment an (N, K) array of class probabilities with derived features.

    Parameters
    ----------
    P : np.ndarray of shape (N, K)
        Each row contains probabilities over K classes.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    X_aug : np.ndarray of shape (N, M)
        Augmented feature matrix.
    feature_names : list of str
        Names of the augmented features.
    """

    assert P.ndim == 2, "P must be 2D (N, K)"
    N, K = P.shape

    # ---------- Basic quantities ----------
    P_safe = P + eps
    logP = np.log(P_safe)

    # ---------- Information-theoretic ----------
    entropy = -np.sum(P * logP, axis=1)                 # (N,)
    entropy_norm = entropy / np.log(K)                  # normalized entropy
    gini = 1.0 - np.sum(P**2, axis=1)                    # (N,)

    # ---------- Confidence / sharpness ----------
    p_max = np.max(P, axis=1)
    p_min = np.min(P, axis=1)
    p_sorted = np.sort(P, axis=1)
    margin_top2 = p_sorted[:, -1] - p_sorted[:, -2]     # top-1 vs top-2
    dist_from_uniform = np.linalg.norm(
        P - 1.0 / K, axis=1
    )                                                     # L2 distance

    # ---------- Decision-related ----------
    argmax = np.argmax(P, axis=1)
    one_hot = np.eye(K)[argmax]                          # (N, K)

    # ---------- Probability transforms ----------
    log_probs = logP                                     # (N, K)
    centered_log_probs = logP - np.mean(logP, axis=1, keepdims=True)

    # ---------- Aggregate log-odds-like measures ----------
    # log(p_max / mean(other probs))
    mean_other = (np.sum(P, axis=1) - p_max) / (K - 1)
    log_odds_max_vs_rest = np.log((p_max + eps) / (mean_other + eps))

    # ---------- Stack all features ----------
    X_aug = np.column_stack([
        entropy,
        entropy_norm,
        gini,
        p_max,
        p_min,
        margin_top2,
        dist_from_uniform,
        log_odds_max_vs_rest,
        log_probs,
        centered_log_probs,
        one_hot
    ])

    # Optional feature names (kept explicit for sanity)
    # feature_names = (
    #     [
    #         "entropy",
    #         "entropy_norm",
    #         "gini",
    #         "p_max",
    #         "p_min",
    #         "margin_top2",
    #         "dist_from_uniform",
    #         "log_odds_max_vs_rest",
    #     ]
    #     + [f"log_p_{k}" for k in range(K)]
    #     + [f"log_p_centered_{k}" for k in range(K)]
    #     + [f"argmax_is_{k}" for k in range(K)]
    # )

    return X_aug

def augment_bernoulli_probabilities(p, eps=1e-12):
    """
    Augment an (N,) array of Bernoulli probabilities with derived features.

    Parameters
    ----------
    p : np.ndarray of shape (N,)
        Probability of the positive class.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    X_aug : np.ndarray of shape (N, M)
        Augmented feature matrix.
    feature_names : list of str
        Names of the augmented features.
    """

    assert p.ndim == 1, "p must be (N,)"

    p1 = np.clip(p, eps, 1 - eps)
    p0 = 1.0 - p1

    # ---------- Information-theoretic ----------
    entropy = -(p0 * np.log(p0) + p1 * np.log(p1))
    gini = 1.0 - (p0**2 + p1**2)

    # ---------- Confidence / sharpness ----------
    p_max = np.maximum(p0, p1)
    p_min = np.minimum(p0, p1)
    margin = p_max - p_min
    abs_diff = np.abs(p1 - p0)
    dist_from_uniform = np.abs(p1 - 0.5)

    # ---------- Decision-related ----------
    argmax = (p1 >= 0.5).astype(int)
    one_hot = np.column_stack([1 - argmax, argmax])

    # ---------- Probability transforms ----------
    log_p0 = np.log(p0)
    log_p1 = np.log(p1)
    log_odds = np.log(p1 / p0)

    # ---------- Stack all features ----------
    X_aug = np.column_stack([
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
    #     "entropy",
    #     "gini",
    #     "p_max",
    #     "p_min",
    #     "margin",
    #     "abs_diff",
    #     "dist_from_uniform",
    #     "log_p0",
    #     "log_p1",
    #     "log_odds",
    #     "argmax_is_0",
    #     "argmax_is_1",
    # ]

    return X_aug



if __name__ == '__main__':
    args = parser.parse_args()
    help_map = {
        action.dest: action.help
        for action in parser._actions
        if action.help is not None
    }
    print("Parsed arguments:")
    for key, value in vars(args).items():
        help_text = help_map.get(key, "no help available")
        print(f"  {key} ({help_text}): {value}")

    train_loader = StructuresLoader(
    dataset_name=args.dataset, s_llm=args.sLLM, m_llm=args.mLLM
    )
    train_loader.load_dataset()  # loads dataset
    train_loader.sLLM_results = train_loader.load_results_file(f'inference_files/{args.dataset}_train_{args.sLLM}_profiler.pkl')
    train_loader.mLLM_results = train_loader.load_results_file(f'inference_files/{args.dataset}_train_{args.mLLM}_profiler.pkl')

    # choose based on dataset
    if train_loader.dataset_name == 'squad':
        train_texts = [train_loader.dataset['train']['context'][i] + '\n' + train_loader.dataset['train']['question'][i]
                       for i in range(len(train_loader.dataset['train'].select(range(len(train_loader.sLLM_results)))))]
        gold_truth_train = train_loader.get_result_values(train_loader.sLLM_results, key='gold_answer')  # gold truth
    elif train_loader.dataset_name == 'wmt':
        train_texts = [ ex['translation']['de'] for ex in train_loader.dataset['train'].select(range(len(train_loader.sLLM_results)))]
        gold_truth_train = train_loader.get_result_values(train_loader.sLLM_results, key='gold_answer')  # gold truth
    else:
        train_texts = [ex.get("text") or ex.get("sentence")
                       for ex in train_loader.dataset['train'].select(range(len(train_loader.sLLM_results)))]
        gold_truth_train = [train_loader.dataset['train']['label'][i] 
                            for i in range(len(train_loader.dataset['train'].select(range(len(train_loader.sLLM_results)))))] # gold truth
    
    assert len(train_texts) == len(train_loader.sLLM_results)

    # gold_truth_train = [train_loader.dataset['train']['label'][i] for i in range(len(train_loader.dataset['train'].select(range(len(train_loader.sLLM_results)))))] # gold truth
    # gold_truth_train = train_loader.get_result_values(train_loader.sLLM_results, key='gold_answer')  # gold truth
    # train_texts = [train_loader.dataset['train']['sentence']
                #    for i in range(len(train_loader.dataset['train'].select(range(len(train_loader.sLLM_results)))))]
    # train_texts = [ ex['translation']['de'] for ex in train_loader.dataset['train'].select(range(len(train_loader.sLLM_results)))]
    # train_texts = [train_loader.dataset['train']['context'][i] + '\n' + train_loader.dataset['train']['question'][i]
    #                for i in range(len(train_loader.dataset['train'].select(range(len(train_loader.sLLM_results)))))]
    print(f'Dataset length: texts={len(train_texts)}, labels={len(gold_truth_train)}')
    sLLM_pred_train = np.array(train_loader.get_result_values(train_loader.sLLM_results, key='pred'))  # pred A
    mLLM_pred_train = np.array(train_loader.get_result_values(train_loader.mLLM_results, key='pred'))  # pred B
    print(f'Predictions shape: {sLLM_pred_train.shape} and {mLLM_pred_train.shape}')

    # choose based on dataset...
    if train_loader.dataset_name == 'squad' or train_loader.dataset_name == 'wmt':
        probs = train_loader.obtain_generation_betas(quantile_based=True)
    elif train_loader.dataset_name == 'sst2' or train_loader.dataset_name == 'fakenews':
        probs = train_loader.obtain_binary_betas(return_both=True)
    else:
        probs = train_loader.obtain_multiclass_betas(return_all=True)
    
    print(f'probs shape: {probs.shape}')
    print(f'\nCalibrating confidence...')
    if args.calibration == 'sequence':
        assert train_loader.dataset_name == 'squad' or train_loader.dataset_name == 'wmt'
        calibrator = SequenceConfidenceCalibrator()
        calibrator.fit(probs, helpers.get_binary_labels(y_true=gold_truth_train, 
                                                        y_hat=train_loader.get_result_values(train_loader.sLLM_results, key='pred'), 
                                                        return_sim=True))
        sLLM_betas_train = calibrator.predict(probs)
    else:
        assert train_loader.dataset_name != 'squad' and train_loader.dataset_name != 'wmt'
        calibrator = PerClassConfidenceCalibrator(probs.shape[1])
        calibrator.fit(probs, gold_truth_train, y_hat=sLLM_pred_train)
        _, sLLM_betas_train = calibrator.predict(probs)
        

    print(f'calibrated betas shape: {sLLM_betas_train.shape}')

    if args.embeddings:
        sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
        if 'wmt' in train_loader.dataset_name:
            sentence_transformer = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            x = sentence_transformer.encode(train_texts, show_progress_bar=True)
    else:
        if train_loader.dataset_name == 'sst2' or train_loader.dataset_name == 'fakenews':
            x = augment_binary_probabilities(probs)
        elif train_loader.dataset_name == 'agnews' or train_loader.dataset_name == 'emotion':
            x = augment_multiclass_probabilities(probs)
        else:
            x = augment_bernoulli_probabilities(probs)
    
    print(f'x shape: {x.shape}')

    if x.shape[1] > 100:
        svd = TruncatedSVD(n_components=100)
        x = svd.fit_transform(x)
        print(f'svd"ed x shape: {x.shape}')


    # Convert predictions to binary errors
    E_A = np.array(helpers.get_binary_labels(y_true=gold_truth_train, y_hat=sLLM_pred_train) != [1] * len(gold_truth_train)).astype(int)
    E_B = np.array(helpers.get_binary_labels(y_true=gold_truth_train, y_hat=mLLM_pred_train) != [1] * len(gold_truth_train)).astype(int)

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

    for train_idx, test_idx in tqdm(kf.split(x), total=kf.get_n_splits(), desc="cross-testing"):
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
    print(f'Difference null-full = {np.mean(mse_null_folds) - np.mean(mse_full_folds)}')

    n_permutations = 10
    # T_obs = mse_null - mse_full
    T_obs = np.mean(mse_null_folds) - np.mean(mse_full_folds)
    perm_stats = []
    perm_mse_full_folds = []

    for _ in tqdm(range(n_permutations), desc='permutations for t-test'):
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

            perm_preds_full_fold = perm_preds_full[test_idx]
            perm_mse_full_folds.append(np.mean((Delta[test_idx] - perm_preds_full_fold)**2))


        mse_full_perm = np.mean((Delta - perm_preds_full)**2)
        # perm_stats.append(mse_null - mse_full_perm)
        perm_stats.append(np.mean(mse_null_folds) - np.mean(perm_mse_full_folds))

    # p-value
    print(perm_stats)
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
    
    



