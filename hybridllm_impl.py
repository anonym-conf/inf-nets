import torch, evaluate, numpy as np
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, 
    TrainingArguments, EarlyStoppingCallback)
from datasets import Dataset
from typing import *
from dataclasses import dataclass
from frugal_impl import HFGenerator, TokenCostModel
from string2string.similarity import BARTScore


@dataclass
class BARTScorer:
    model_name: str = "facebook/bart-large-cnn"
    device: Optional[str] = None
    max_length: int = 1024

    def __post_init__(self):
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok, self.model = None, None
        self.bart_scorer = BARTScore(model_name_or_path=self.model_name, device=self.device)

    @torch.no_grad()
    def score(self, cands: List[str], refs: List[str], batch_size: int = 8) -> torch.Tensor:
        """
        Returns BARTScore-like scores: average log p(cand | ref) per cand token.
        Higher is better (less negative).

        Implementation: NLL from seq2seq with encoder_input=ref, decoder_labels=cand.
        """
        assert len(cands) == len(refs)
        self.tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        scores = []

        for i in range(0, len(cands), batch_size):
            c_batch = cands[i:i + batch_size]
            r_batch = refs[i:i + batch_size]

            enc = self.tok(
                r_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            # labels: candidate tokens
            lab = self.tok(
                c_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).input_ids.to(self.device)

            # Ignore padding in loss
            pad_id = self.tok.pad_token_id
            labels = lab.clone()
            labels[labels == pad_id] = -100

            out = self.model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                labels=labels,
            )
            # out.loss is mean over non-ignored tokens across batch.
            # We want per-example average log-prob per token.
            # Compute per-example token-average NLL by manual token loss.
            logits = out.logits  # [B, T, V]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # mask for valid tokens
            valid = shift_labels != -100
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            # gather log p(label_t)
            token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            token_logp = token_logp * valid  # zero out ignored

            sum_logp = token_logp.sum(dim=1)
            denom = valid.sum(dim=1).clamp(min=1)
            avg_logp = sum_logp / denom  # per-token average log-prob
            scores.append(avg_logp.detach().cpu())

        return torch.cat(scores, dim=0)
    
    def score_ready(self, cands: List[str], refs: List[str], batch_size: int = 8) -> torch.Tensor:        
        score = self.bart_scorer.compute(refs, cands, agg="mean", batch_size=batch_size)
        return torch.from_numpy(score['score'])


class Router:
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        if self.tok.pad_token is None:
          self.tok.add_special_tokens({'pad_token': '[PAD]'})
          self.model.resize_token_embeddings(len(self.tok))

    def _tokenize_dataset(self, ds: Dataset) -> Dataset:
        def tok_fn(ex):
            enc = self.tok(ex["text"], truncation=True, padding=True)
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
            load_best_model_at_end=True,  # loads best model when training ends
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
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """
        Returns probability of correctness P(correct=1 | query, answer) in [0,1].
        """
        self.model.eval()
        probs = []
        bs = 32
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            logits = self.model(**enc).logits  # (bs, 2)
            p = torch.softmax(logits, dim=-1)[:, 1]  # class 1 -> route-to-small
            # p = torch.softmax(logits, dim=-1).max(dim=-1).values  # prob(predicted)
            probs.append(p.detach().cpu().numpy())
        return np.concatenate(probs, axis=0)


def generate_router_dataset(
    base_ds: Dataset,
    generator_small: HFGenerator,
    generator_big: HFGenerator,
    bart_scorer: BARTScorer,
    label_tokens: list,
    max_items: int = 400,
) -> Dataset:
    """
    Build router dataset:
      text = f(query)
      label = 1(correct) / 0(incorrect)
    """
    base_ds = base_ds.select(range(min(max_items, len(base_ds))))
    print(base_ds)
    # texts = [ex["text"] for ex in base_ds]
    texts = [ex.get("text") or ex.get("sentence") or ex.get("question") or ex.get('translation').get('de') for ex in base_ds]
    refs = [
        label_tokens[int(ex.get('label'))] if 'label' in ex else 
        (ex.get('answers').get('text')[0] if 'answers' in ex else ex.get('translation').get('en')) 
        for ex in base_ds
    ]

    # gen_results = generator.predict_batch(texts)
    gen_sm_results = generator_small.get_predictions_from_results()
    gen_bg_results = generator_big.get_predictions_from_results()
    
    y_s = [e.answer_text for e in gen_sm_results]
    y_l = [e.answer_text for e in gen_bg_results]

    q_s = bart_scorer.score_ready(y_s, refs, batch_size=32).tolist()
    q_l = bart_scorer.score_ready(y_l, refs, batch_size=32).tolist()
    y_small_wins = [1 if a >= b else 0 for a, b in zip(q_s, q_l)]  # based on BART score

    return Dataset.from_dict({"text": texts, "label": y_small_wins, 
                              "quality_small": q_s, "quality_large": q_l})

def router_threshold_sweep(
    ds: Dataset,
    # gen_small: HFGenerator,
    # gen_big: HFGenerator,
    router: Router,
    cost_small: TokenCostModel,
    cost_big: TokenCostModel,
    thetas: np.ndarray,
    delta: float,
    # oracle: bool = False
) -> Dict[str, float]:
    

    # small_res = gen_small.get_predictions_from_results()
    # small_pred = np.array([r.yhat for r in small_res], dtype=int)

    small_cost = cost_small.cost()

    texts = [ex.get("text") or ex.get("sentence") or ex.get("question") or ex.get('translation').get('de') for ex in ds]
    scores = router.score_batch(texts)

    # big_res = gen_big.get_predictions_from_results()
    # big_pred = np.array([r.yhat for r in big_res], dtype=int)

    big_cost = cost_big.cost()

    # best = {"theta": 0., "savings": None, 
    #         "frac_small": None, "defer_rate": None,"quality_h": None, 
    #         "cost_h": None}
    best = None

    # y_true_arr = np.array(y_true, dtype=int)
    # y_true_arr = np.array(
    #     [int(ex["label"]) if "label" in ex else 1 for ex in ds], dtype=int
    # )  # non-oracle

    # if oracle:
    #     y_true_arr = np.asarray(big_pred)
    
    q_s = np.array(ds['quality_small'], dtype=float)
    q_l = np.array(ds['quality_large'], dtype=float)

    # Baseline
    large_only_q = q_l.mean()

    for theta in thetas:
        route_small = scores >= theta
        # hybrid quality = average BARTScore of chosen output
        q_h = np.where(route_small, q_s, q_l).mean()
        # hybrid cost
        cost_h = (small_cost * route_small + big_cost * (~route_small)).mean()
        frac_small = route_small.mean()

        ok = (q_h >= large_only_q - delta)
        if ok:
            # maximize savings / or maximize fraction_small under constraint
            savings = 1.0 - (cost_h / big_cost)
            cand = (savings, frac_small, q_h, cost_h, theta)
            if (best is None) or (cand[0] > best['savings']):
                best = {"theta": float(theta), "savings": savings, 
                        "frac_small": frac_small, "defer_rate": 1. - frac_small, 
                        "quality_h": q_h, "cost_h": cost_h}
    
    return best
