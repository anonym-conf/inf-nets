import numpy as np, evaluate
from scipy import stats
from sentence_transformers import SentenceTransformer


embedder = SentenceTransformer('all-mpnet-base-v2', )


def rouge_per_sample(
    predictions,
    references,
    rouge_types=("rougeL",),     # e.g. ("rouge1",) or ("rouge1","rouge2","rougeL")
    use_stemmer=True,
    batch_size=256,
    combine="mean"              # "mean" across rouge_types or "first"
):
    """
    Compute per-sample ROUGE scores using `evaluate`'s ROUGE metric.

    Args:
        predictions: list[str] - model outputs
        references:  list[str] - ground truth texts
        rouge_types: tuple/list of ROUGE variants: "rouge1","rouge2","rougeL","rougeLsum"
        use_stemmer: bool
        batch_size:  int
        combine:     "mean" to average across rouge_types, or "first" to return first type

    Returns:
        np.ndarray shape (N,) with values in [0,1].
        If rouge_types has multiple entries and combine="mean", returns their average per sample.
    """
    
    _ROUGE = evaluate.load("rouge")

    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: len(predictions)={len(predictions)} vs len(references)={len(references)}")

    # Ensure strings (handle None safely)
    preds = ["" if p is None else str(p) for p in predictions]
    refs  = ["" if r is None else str(r) for r in references]

    rouge_types = tuple(rouge_types)
    per_type = {rt: [] for rt in rouge_types}

    for i in range(0, len(preds), batch_size):
        out = _ROUGE.compute(
            predictions=preds[i:i + batch_size],
            references=refs[i:i + batch_size],
            rouge_types=list(rouge_types),
            use_aggregator=False,   # <-- critical: returns per-example lists
            use_stemmer=use_stemmer
        )
        for rt in rouge_types:
            per_type[rt].extend(out[rt])

    if combine == "first":
        return np.asarray(per_type[rouge_types[0]], dtype=np.float32)

    # default: average multiple ROUGE types per sample
    arrs = [np.asarray(per_type[rt], dtype=np.float32) for rt in rouge_types]
    if len(arrs) == 1:
        return arrs[0]
    return np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)


def get_cost_model(cost_model):
    if 'gemma' in cost_model:
        cost_model = cost_model.replace('gemma', 'gemma-')
        cost_model = 'google/' + cost_model
    elif 'qwen' in cost_model:
        cost_model = 'Qwen/Qwen3-4B-Instruct-2507'
    elif 'ministral'in cost_model:
        cost_model = 'mistralai/Ministral-3-3B-Instruct-2512'
    
    print(f'Cost model name: {cost_model}')
    return cost_model


def min_max_normalize(log_probs):
    """
    Normalize a list of log probabilities to the range [0, 1] using min-max normalization.
    
    Parameters:
    log_probs (list or iterable): List of log probability values to normalize.
    
    Returns:
    list: List of normalized log probabilities in the range [0, 1].
    """
    # Ensure the input is a list or iterable
    if not isinstance(log_probs, (list, tuple, np.ndarray)):
        raise ValueError("log_probs should be a list, tuple or array of values.")
    
    # Calculate the minimum and maximum of the log probabilities
    min_log_prob = min(log_probs)
    max_log_prob = max(log_probs)
    
    # If all values are the same, return a list of 0.5 (or handle as needed)
    if min_log_prob == max_log_prob:
        return [0.5] * len(log_probs)
    
    # Apply min-max normalization
    normalized_probs = [(log_prob - min_log_prob) / (max_log_prob - min_log_prob) for log_prob in log_probs]
    
    return normalized_probs

def get_binary_labels(y_true=None, y_hat=None, sim=None, sim_threshold=0.7,
                      return_sim=False):
    if sim is not None:
        return (np.asarray(sim) >= sim_threshold).astype(int)
    assert y_true is not None and y_hat is not None
    # sim = []
    # for pred, gold in tqdm(zip(y_hat, y_true), total=len(y_hat)):
    #     pred_emb = embedder.encode([pred], normalize_embeddings=True).flatten()
    #     gold_emb = embedder.encode(([gold]), normalize_embeddings=True)
    #     # print(pred_emb.shape, gold_emb.shape)
    #     sim.append(gold_emb @ pred_emb)

    pred_embs = embedder.encode(y_hat, normalize_embeddings=True)
    gold_embs = embedder.encode(y_true, normalize_embeddings=True)
    # print(pred_emb.shape, gold_embs.shape)
    # print(pred_emb.shape, gold_embs.shape)
    # sims = gold_embs @ pred_emb  # because normalized => cosine
    sim = np.sum(pred_embs * gold_embs, axis=1)
    # sim = np.array(sim)
    # del embedder
    if return_sim: return sim
    return (sim >= sim_threshold).astype(int) 


def build_posthoc_quantile_features(token_logprobs: np.ndarray) -> np.ndarray:
    """
    token_logprobs: shape (T,), log p(y_t | context) for generated answer tokens
    returns: shape (22,) = [sum, avg] + 20 quantiles
    """
    if token_logprobs.size == 0:
        token_logprobs = np.array([-1e9], dtype=np.float32)
    
    alphas = [0.0] + [i / 100 for i in range(1, 11)] + [i / 10 for i in range(2, 11)]
    s_sum = float(token_logprobs.sum())
    s_avg = float(token_logprobs.mean())
    qs = np.quantile(token_logprobs, alphas).astype(np.float32)

    feats = np.concatenate([[s_sum, s_avg], qs], axis=0).astype(np.float32)
    # feats = qs.astype(np.float32)
    return feats


def cost_distribution_signature(cost_list, weights=None):
    """
    Convert a list of per-query costs into a unique scalar signature
    that captures the distribution characteristics.

    Parameters:
    cost_list: list of float - per-query costs for an LLM
    weights: dict - optional weights for different components

    Returns:
    float - unique scalar signature
    """

    if weights is None:
        weights = {
            'mean': 1.0,
            'median': 1.2,
            'std': 0.8,
            'iqr': 1.1,
            'skew': 0.6,
            'kurtosis': 0.5,
            'p95': 1.3,
            'cv': 0.9  # coefficient of variation
        }

    arr = np.array(cost_list)

    # Basic statistics
    mean_val = np.mean(arr).astype(float)
    median_val = np.median(arr).astype(float)
    std_val = np.std(arr).astype(float)

    # Robust statistics
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_val = q75 - q25
    p95_val = np.percentile(arr, 95).astype(float)

    # Shape statistics
    try:
        if not np.all(arr == arr[0]):
            skew_val = stats.skew(arr)
            kurtosis_val = stats.kurtosis(arr)
        else:
            skew_val = 0
            kurtosis_val = 0
    except:
        # Fallback for small samples
        skew_val = 0
        kurtosis_val = 0

    # Coefficient of variation (normalized variability)
    cv_val = std_val / mean_val if mean_val > 0 else 0

    # Combine into weighted signature
    signature = (
            weights['mean'] * mean_val +
            weights['median'] * median_val +
            weights['std'] * std_val +
            weights['iqr'] * iqr_val +
            weights['skew'] * abs(skew_val) +  # Use absolute for consistency
            weights['kurtosis'] * abs(kurtosis_val) +
            weights['p95'] * p95_val +
            weights['cv'] * cv_val
    )

    return signature


def cost_distribution_signature_param_aware(cost_list, parameter_count, weights=None):
    """
    Convert per-query costs into a scalar signature that incorporates
    the intuition that higher parameter counts should lead to higher costs.

    Parameters:
    cost_list: list of float - per-query costs
    parameter_count: int - number of parameters in billions (e.g., 7 for 7B model)
    weights: dict - optional custom weights

    Returns:
    float - cost signature that respects parameter count intuition
    """

    # Base weights that emphasize metrics correlated with higher parameter costs
    if weights is None:
        # weights = {
        #     'mean': 1.4,  # Higher weight - directly reflects average cost
        #     'median': 1.3,  # High weight - robust central tendency
        #     'std': 0.7,  # Moderate - variability matters but less than central tendency
        #     'iqr': 0.8,  # Moderate - robust variability
        #     'skew': 0.4,  # Low - distribution shape secondary
        #     'kurtosis': 0.3,  # Low - tail behavior secondary
        #     'p95': 1.5,  # Highest - worst-case costs often scale with parameters
        #     'cv': 0.6,  # Moderate - normalized variability
        #     'param_boost': 100  # Direct parameter count influence
        # }
        weights = {
            'mean': 2.0,
            'median': 1.2,
            'std': 0.8,
            'iqr': 1.1,
            'skew': 0.6,
            'kurtosis': 0.5,
            'p95': 1.3,
            'cv': 0.9,
            'param_boost': 100
        }

    arr = np.array(cost_list)

    # Core distribution statistics
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    std_val = np.std(arr)

    q75, q25 = np.percentile(arr, [75, 25])
    iqr_val = q75 - q25
    p95_val = np.percentile(arr, 95)

    try:
        skew_val = stats.skew(arr)
        kurtosis_val = stats.kurtosis(arr)
    except:
        skew_val = 0
        kurtosis_val = 0

    cv_val = std_val / mean_val if mean_val > 0 else 0

    # Parameter count normalization
    # Assume typical range: 1B-100B parameters, normalize to 0-1 scale
    normalized_params = np.log10(parameter_count) / 2.0  # log10(100) = 2

    # Combined signature with parameter awareness
    distribution_component = (
            weights['mean'] * mean_val +
            weights['median'] * median_val +
            weights['std'] * std_val +
            weights['iqr'] * iqr_val +
            weights['skew'] * abs(skew_val) +
            weights['kurtosis'] * abs(kurtosis_val) +
            weights['p95'] * p95_val +
            weights['cv'] * cv_val
    )

    # Add parameter boost - scales with the distribution component
    param_component = weights['param_boost'] * normalized_params * distribution_component

    signature = distribution_component + param_component

    return signature
