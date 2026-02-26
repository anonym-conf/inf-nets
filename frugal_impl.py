import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from sklearn.metrics import f1_score

import torch
import evaluate
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments, EarlyStoppingCallback,
    Trainer, DataCollatorWithPadding
)


# ----------------------------
# Reproducibility
# ----------------------------
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ============================================
# 1) Generator Wrapper (LLM classifier via generation)
# ============================================

@dataclass
class GenResult:
    yhat: int
    answer_text: str
    in_tokens: int
    out_tokens: int

class HFGenerator:
    """
    Wraps a HF model (seq2seq or causal) that generates answers.
    """
    def __init__(
        self,
        model_name: str = None,
        max_new_tokens: int = 6,
        temperature: float = 0.0,
        results: list = None,
    ):
        if model_name is None: 
            assert results is not None
            self.results = results
            return

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        cfg = AutoConfig.from_pretrained(model_name)
        self.is_encoder_decoder = bool(getattr(cfg, "is_encoder_decoder", False))

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Load model
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")

        self.model.eval()

        if self.tok.pad_token is None:
          self.tok.add_special_tokens({'pad_token': '[PAD]'})
          self.model.resize_token_embeddings(len(self.tok))
        
        # label_ids = []
        # for lbl in LABELS:
        #   toks = self.tok(f"{lbl}", add_special_tokens=False).input_ids
        #   print(toks[0])
        #   assert len(toks) == 1, f"'{lbl}' mapped to multiple tokens! --> {toks}"
        #   label_ids.append(toks[0])
        
        # self.logits_processor = KWayLogitsProcessor(label_ids)
        self.logits_processor = None

    @torch.no_grad()
    def predict_one(self, text: str) -> GenResult:
        # prompt = make_prompt_agnews(text)
        prompt = text

        enc = self.tok(prompt, return_tensors="pt",)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        in_tokens = int(enc["input_ids"].shape[-1])

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=(self.temperature > 0),
            temperature=self.temperature if self.temperature > 0 else None,
            logits_processor=[self.logits_processor] if self.logits_processor is not None else [], 
            # output_scores=True,
            # return_dict_in_generate=True
        )
        # remove None keys safely
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.inference_mode():
          out = self.model.generate(**enc, **gen_kwargs)
        # print(out, out.sequences)

        if self.is_encoder_decoder:
            gen_ids = out[0]
            # gen_ids = out.sequences
            out_tokens = int(gen_ids.shape[-1]) - 1  # + decoder_start_token_id 
            assert out_tokens == self.max_new_tokens
            decoded = self.tok.decode(gen_ids, skip_special_tokens=True)
            # print(decoded)
            ans = decoded.strip()
        else:
            # causal includes prompt + generated
            gen_ids = out[0]
            # gen_ids = out.sequences
            out_tokens = int(gen_ids.shape[-1] - in_tokens)
            assert out_tokens == self.max_new_tokens
            decoded = self.tok.decode(gen_ids[in_tokens:], skip_special_tokens=True)
            ans = decoded.strip()

        # yhat = parse_label_from_text(ans)
        # yhat = label2id[ans]
        if yhat is None:
            # fallback: default to most frequent label (World) or random;
            # here we pick World for stability.
            yhat = 0

        return GenResult(yhat=yhat, answer_text=ans, in_tokens=in_tokens, out_tokens=out_tokens)

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[GenResult]:
        return [self.predict_one(t) for t in texts]

    def get_predictions_from_results(self) -> List[GenResult]:
        gen_results = []  # GenResult list
        for entry in self.results: # each entry is a dict
            res = GenResult(
                yhat=entry['pred_idx'] if 'pred_idx' in entry else entry['binary_with_gold'],
                answer_text=entry['pred'],
                in_tokens=entry['text_length'], out_tokens=entry['output_length']
            )
            gen_results.append(res)
        return gen_results



# ============================================
# 2) Cost model: token-based or fixed-cost
# ============================================

@dataclass
class TokenCostModel:
    # cost = c_in*in_tokens + c_out*out_tokens + c0
    c_in: float = 1.0
    c_out: float = 1.0
    c0: float = 0.0

    fixed_cost: Optional[float] = None

    def cost(self, in_tokens: int = 0, out_tokens: int = 0) -> float:  # for single query
        if self.fixed_cost is not None: return self.fixed_cost
        assert in_tokens != 0 and out_tokens != 0
        return self.c_in * in_tokens + self.c_out * out_tokens + self.c0


# ============================================
# 3) Scorer g(q,a): predict correctness prob from (q,a)
#    We'll train one scorer per generator stage (as in FrugalGPT)
# ============================================

def build_scorer_text(query_text: str, answer_text: str) -> str:
    # Simple cross-encoder input
    return f"[QUERY]\n{query_text}\n\n[ANSWER]\n{answer_text}\n"

class CorrectnessScorer:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        if self.tok.pad_token is None:
          self.tok.add_special_tokens({'pad_token': '[PAD]'})
          self.model.resize_token_embeddings(len(self.tok))

    def _tokenize_dataset(self, ds: Dataset) -> Dataset:
        def tok_fn(ex):
            enc = self.tok(ex["pair_text"], truncation=True)
            enc["labels"] = int(ex["label"])
            return enc
        return ds.map(tok_fn, remove_columns=ds.column_names)

    def fit(self, train_ds: Dataset, val_ds: Dataset, out_dir: str, epochs: int = 1):
        data_collator = DataCollatorWithPadding(tokenizer=self.tok)
        train_tok = self._tokenize_dataset(train_ds)
        # print(train_tok['input_ids'])
        # val_tok = self._tokenize_dataset(val_ds) if val_ds is not None else None

        split_tok = train_tok.train_test_split(test_size=0.15, shuffle=False, seed=1)

        metric_acc = evaluate.load("accuracy")

        self.model.train()

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return metric_acc.compute(predictions=preds, references=labels)

        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            learning_rate=2e-5,
            num_train_epochs=epochs,
            eval_strategy="epoch",
            # eval_strategy="no",
            save_strategy="epoch",
            logging_steps=50,
            report_to="none",
            fp16=torch.cuda.is_available(),
            load_best_model_at_end=True,  # Essential: loads best model when training ends
            metric_for_best_model='eval_loss',  # Metric to determine "best" model
            greater_is_better=False,      # Lower loss is better (reverse for accuracy)
            # use_cpu=True
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=split_tok['train'],
            eval_dataset=split_tok['test'],
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()

    @torch.no_grad()
    def score_batch(self, pair_texts: List[str]) -> np.ndarray:
        """
        Returns probability of correctness P(correct=1 | query, answer) in [0,1].
        """
        self.model.eval()
        probs = []
        bs = 32
        for i in range(0, len(pair_texts), bs):
            batch = pair_texts[i:i+bs]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            logits = self.model(**enc).logits  # (bs, 2)
            # p = torch.softmax(logits, dim=-1)[:, 1]  # prob(correct=1)
            p = torch.softmax(logits, dim=-1).max(dim=-1).values  # prob(predicted)
            probs.append(p.detach().cpu().numpy())
        return np.concatenate(probs, axis=0)


# ============================================
# 4) Frugal Cascade: accept if score >= tau else defer
# ============================================

@dataclass
class CascadePolicy:
    thresholds: List[float]  # length = num_stages-1 (last stage always accept)

class FrugalCascade:
    def __init__(
        self,
        generators: List[HFGenerator],
        scorers: List[CorrectnessScorer],
        cost_models: List[TokenCostModel],
        policy: CascadePolicy,
    ):
        assert len(generators) == len(scorers) == len(cost_models)
        assert len(policy.thresholds) == len(generators) - 1
        self.generators = generators
        self.scorers = scorers
        self.cost_models = cost_models
        self.policy = policy

    def run(self, 
            ds: Dataset,
            # texts: List[str], 
            # y_true: List[int]
            ) -> Dict[str, float]:
        """
        Executes cascade, returns accuracy + average cost + deferral rates.
        """
        texts = [ex.get("text") or ex.get("sentence") or ex.get("question") or ex.get('translation').get('de') for ex in ds]
        y_true = np.array([int(ex["label"]) if "label" in ex else 1 for ex in ds], dtype=int)

        n = len(texts)
        # stage = 0
        active_idx = np.arange(n)

        final_pred = np.full(n, -1, dtype=int)
        total_cost = np.zeros(n, dtype=float)

        deferral_counts = np.zeros(len(self.generators), dtype=int)

        c_s, c_m = self.cost_models[0].cost(), self.cost_models[1].cost()

        # sequential stages
        for i in range(len(self.generators)):
            if len(active_idx) == 0:
                break

            # generate answers for active queries
            batch_texts = [texts[j] for j in active_idx]
            # gen_results = self.generators[i].predict_batch(batch_texts)
            gen_results = self.generators[i].get_predictions_from_results()
            assert n == len(gen_results)

            # accumulate cost
            for k, j in enumerate(active_idx):
                # total_cost[j] += self.cost_models[i].cost(gen_results[k].in_tokens, gen_results[k].out_tokens)
                total_cost[j] += self.cost_models[i].cost()

            # last stage always accept
            if i == len(self.generators) - 1:
                for k, j in enumerate(active_idx):
                    # final_pred[j] = gen_results[k].yhat
                    final_pred[j] = gen_results[j].yhat
                deferral_counts[i] += len(active_idx)
                break

            # score
            pair_texts = [build_scorer_text(batch_texts[k], gen_results[k].answer_text) for k in range(len(active_idx))]
            scores = self.scorers[i].score_batch(pair_texts)

            theta = self.policy.thresholds[i]
            accept_mask = scores >= theta

            # accept now
            accepted_idx = active_idx[accept_mask]
            for k, j in enumerate(accepted_idx):
                # map accepted_idx back to local index k2
                local_pos = np.where(active_idx == j)[0][0]
                final_pred[j] = gen_results[local_pos].yhat

            # defer remaining
            active_idx = active_idx[~accept_mask]
            deferral_counts[i] += len(accepted_idx)

        acc = (final_pred == np.array(y_true)).mean()
        avg_cost = float(total_cost.mean())

        # fraction reaching each stage: approximate from costs is messy; we report acceptance counts
        return {
            "accuracy": float(acc), "f1": float(f1_score(final_pred, np.array(y_true))),
            "avg_cost": avg_cost,
            "accepted_stage_counts": deferral_counts.tolist(),
            "cost_saved": ((c_s + c_m) - float(avg_cost)) / (c_s + c_m) * 100
        }


# ============================================
# 5) Demo: data + training scorers + threshold sweep
# ============================================

# def load_agnews_subsets(n_train=800, n_val=400, n_test=400) -> Tuple[Dataset, Dataset, Dataset]:
#     ds = load_dataset("ag_news")
#     train = ds["train"].shuffle(seed=SEED).select(range(n_train))
#     test = ds["test"].shuffle(seed=SEED).select(range(n_test))

#     # carve val from train
#     val = train.select(range(n_val))
#     train2 = train.select(range(n_val, n_train))

#     return train2, val, test

def generate_scorer_dataset(
    base_ds: Dataset,
    generator: HFGenerator,
    max_items: int = 400,
) -> Dataset:
    """
    Build scorer dataset for one stage:
      pair_text = f(query, answer)
      label = 1(correct) / 0(incorrect)
    """
    base_ds = base_ds.select(range(min(max_items, len(base_ds))))
    # texts = [ex["text"] for ex in base_ds]
    texts = [ex.get("text") or ex.get("sentence") or ex.get("question") or ex.get('translation').get('de') for ex in base_ds]
    y_true = [int(ex["label"]) if "label" in ex else 1 for ex in base_ds]  # always 1 for generation tasks

    # gen_results = generator.predict_batch(texts)
    gen_results = generator.get_predictions_from_results()
    pair_texts = [build_scorer_text(texts[i], gen_results[i].answer_text) for i in range(len(texts))]
    y_correct = [1 if gen_results[i].yhat == y_true[i] else 0 for i in range(len(texts))]

    return Dataset.from_dict({"pair_text": pair_texts, "label": y_correct})

def threshold_sweep_two_stage(
    ds: Dataset,
    # y_true: List[int],
    gen_small: HFGenerator,
    gen_big: HFGenerator,
    scorer_small: CorrectnessScorer,
    cost_small: TokenCostModel,
    cost_big: TokenCostModel,
    thetas: np.ndarray,
    budget: float,
    oracle: bool = False
) -> Dict[str, float]:
    """
    Two-stage sweep:
      always run small
      if score >= tau: accept small
      else: run big

    Choose best accuracy s.t. avg_cost <= budget_tokens
    """
    # Precompute small outputs
    # small_res = gen_small.predict_batch(texts)
    small_res = gen_small.get_predictions_from_results()
    small_pred = np.array([r.yhat for r in small_res], dtype=int)
    # small_cost = np.array([cost_small.cost(r.in_tokens, r.out_tokens) for r in small_res], dtype=float)
    # small_cost = np.array([cost_small.cost() for _ in small_res], dtype=float)  # same cost across
    small_cost = cost_small.cost()
    
    texts = [ex.get("text") or ex.get("sentence") or ex.get("question") or ex.get('translation').get('de') for ex in ds]
    pair_texts = [build_scorer_text(texts[i], small_res[i].answer_text) for i in range(len(texts))]
    scores = scorer_small.score_batch(pair_texts)

    # Precompute big outputs (only needed for deferred, but for simplicity cache once)
    # big_res = gen_big.predict_batch(texts)
    big_res = gen_big.get_predictions_from_results()
    big_pred = np.array([r.yhat for r in big_res], dtype=int)
    # big_cost = np.array([cost_big.cost(r.in_tokens, r.out_tokens) for r in big_res], dtype=float)
    # big_cost = np.array([cost_big.cost() for _ in big_res], dtype=float)  # same cost across
    big_cost = cost_big.cost()

    best = {"theta": 0., "acc": -1.0, "avg_cost": None, "defer_rate": None}

    # y_true_arr = np.array(y_true, dtype=int)
    y_true_arr = np.array(
        [int(ex["label"]) if "label" in ex else 1 for ex in ds], dtype=int
    )  # non-oracle

    if oracle:
        y_true_arr = np.asarray(big_pred)

    for theta in thetas:
        accept = scores >= theta
        final_pred = np.where(accept, small_pred, big_pred)

        acc = (final_pred == y_true_arr).mean()
        avg_cost = (float(small_cost) + (~accept).astype(float) * float(big_cost)).mean()
        defer_rate = float((~accept).mean())

        if avg_cost <= budget and acc > best["acc"]:
            best = {"theta": float(theta), "acc": float(acc), 
                    "avg_cost": float(avg_cost), "defer_rate": defer_rate}

    return best
