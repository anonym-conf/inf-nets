import pickle
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import login
from typing import *
from transformers import (
    AutoModelForCausalLM, FineGrainedFP8Config, Mistral3ForConditionalGeneration
)

class CostCalculator:
    def __init__(self, model_id, results=None, results_filename=None):
        if results is not None:
            self.results = results
        else:
            assert results_filename is not None
            self.results = pickle.load(open(results_filename, 'rb'))
        model = self.load_model(model_id)
        self.config = model.config
        self.lengths = []
        del model  # free up memory
    
    def load_model(self, model_id):
        # access_token = "..."
        # login(token=access_token, )  # possibly will need HF login
        if 'ministral' in str(model_id).lower():
            return Mistral3ForConditionalGeneration.from_pretrained(model_id, device_map="cpu",
                                                                    quantization_config=FineGrainedFP8Config(dequantize=True))
        return AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", )

    @staticmethod
    def _tri(L_in, L_out):
        # attention sweep over prefill + generation with KV cache
        return (L_in * (L_in + 1) / 2.0) + (L_out * L_in) + ((L_out - 1) * L_out / 2.0)

    @staticmethod
    def _arch_constants_from_config(cfg):
        """
        Infer c_mlp (d^2-heavy) and c_attn (d·L) from HF config.
        - c_mlp = (projections) + (MLP)
          projections: Q,O full-size + K,V reduced by r_kv = n_kv_heads / n_heads
            => proj_factor ≈ 2 + 2 * r_kv   (≈4 when MHA; smaller when GQA/MQA)
          MLP: use expansion r = intermediate_size / hidden_size
            GeLU-like => mlp_factor ≈ 2 * r
            SwiGLU/SiLU-like => mlp_factor ≈ 3 * r
        - c_attn: ~2 (QK^T + AV). Keep a small mid-range default.
        """
        d = getattr(cfg, "hidden_size", None)
        if d is None:
            d = getattr(cfg.text_config, "hidden_size")
        n = getattr(cfg, "num_hidden_layers", None)
        if n is None:
            n = getattr(cfg.text_config, "num_hidden_layers")
        inter = getattr(cfg, "intermediate_size", 4 * d)
        r = inter / d

        act = (getattr(cfg, "hidden_activation", "") or "").lower()
        if "swiglu" in act or "silu" in act or "swish" in act:
            mlp_factor = 3.0 * r
        else:
            mlp_factor = 2.0 * r

        h = getattr(cfg, "num_attention_heads", None)
        h_kv = getattr(cfg, "num_key_value_heads", h)
        r_kv = (h_kv / h) if (h and h_kv) else 1.0
        proj_factor = 2.0 + 2.0 * r_kv  # Q,O full (2) + K,V scaled by r_kv

        c_mlp = proj_factor + mlp_factor   # total d^2-heavy constant
        c_attn = 2.5                       # mild mid-point for attention kernels
        return n, d, c_mlp, c_attn

    def proxy_query_cost(self, L_in: int, L_out: int, mode: str = "tflops",
                         count_mac_as_2flop: bool = True):
        """
        Returns a single scalar per-query cost.
        mode="units"  -> unitless, param-free, architecture-agnostic
        mode="tflops" -> weighted & scaled (uses config to estimate constants)
        """

        L_in = int(L_in)
        L_out = int(L_out)
        n = getattr(self.config, "num_hidden_layers", None)
        d = getattr(self.config, "hidden_size", None)
        if n is None:
            n = getattr(self.config.text_config, "num_hidden_layers")
        if d is None:
            d = getattr(self.config.text_config, 'hidden_size')
        tri = self._tri(L_in, L_out)

        if mode == "units":
            return n * ((L_in + L_out) * (d ** 2) + d * tri)

        # tflops: architecture-aware constants + MAC->FLOPs + scale
        n, d, c_mlp, c_attn = self._arch_constants_from_config(self.config)
        a = n * c_mlp * (d ** 2)     # d^2-heavy piece
        b = n * c_attn * d           # d·L attention piece
        flops = (L_in + L_out) * a + b * tri
        if count_mac_as_2flop:
            flops *= 2.0
        return flops / 1e12  # TFLOPs-ish


    # --------- constants from model.config (architecture-aware) ----------
    # def _arch_constants_from_config(cfg):
    #     """
    #     Infer c_mlp (d^2-heavy) and c_attn (d·L) from a HuggingFace config.
    #     - c_mlp = projections + MLP
    #       projections: Q,O full-size + K,V scaled by r_kv = n_kv_heads / n_heads
    #         => proj_factor ≈ 2 + 2 * r_kv   (≈4 when MHA; smaller when GQA/MQA)
    #       MLP expansion r = intermediate_size / hidden_size
    #         GeLU-like => mlp_factor ≈ 2 * r
    #         SwiGLU/SiLU-like => mlp_factor ≈ 3 * r
    #     - c_attn: ~2–4 (QK^T + AV); we use 2.5 as a reasonable midpoint.
    #     """
    #     d = getattr(cfg, "hidden_size")
    #     n = getattr(cfg, "num_hidden_layers")
    #     inter = getattr(cfg, "intermediate_size", 4 * d)
    #     r = inter / d

    #     act = (getattr(cfg, "hidden_act", "") or "").lower()
    #     if "swiglu" in act or "silu" in act or "swish" in act:
    #         mlp_factor = 3.0 * r
    #     else:
    #         mlp_factor = 2.0 * r

    #     h = getattr(cfg, "num_attention_heads", None)
    #     h_kv = getattr(cfg, "num_key_value_heads", h)
    #     r_kv = (h_kv / h) if (h and h_kv) else 1.0
    #     proj_factor = 2.0 + 2.0 * r_kv  # Q,O full (2) + K,V scaled by r_kv

    #     c_mlp = proj_factor + mlp_factor   # d^2-heavy constant
    #     c_attn = 2.5                       # d·L constant (midpoint)
    #     return n, d, c_mlp, c_attn

    # --------- per-item "ideal" (no padding) costs ----------
    @staticmethod
    def _ideal_prefill_ab(a: float, b: float, Lin: int) -> float:
        # a * Lin + b * Lin*(Lin+1)/2
        return a * Lin + b * (Lin * (Lin + 1) / 2.0)
    
    @staticmethod
    def _ideal_gen_ab(a: float, b: float, Lin: int, Lout: int) -> float:
        # sum_{t=0}^{Lout-1} [a + b*(Lin + t)]
        return (a * Lout) + b * (Lout * Lin + (Lout - 1) * Lout / 2.0)

    # --------- batch-level costs, closer to what profiler counts ----------
    @staticmethod
    def _batch_prefill_ab(a: float, b: float, Lin_batch: Sequence[int]) -> float:
        # Prefill typically uses padded seq_len = max(Lin) across the batch
        B = len(Lin_batch)
        Lmax = max(Lin_batch) if B else 0
        return B * (a * Lmax + b * (Lmax * (Lmax + 1) / 2.0))
    
    @staticmethod
    def _batch_gen_ab_static(a: float, b: float, Lin_batch: Sequence[int], Lout_batch: Sequence[int]) -> float:
        """
        Static batch (HF default): batch stays full until all finish.
        At step t=0..Tmax-1, we compute for all B sequences; attention sees (Lin_i + t).
        """
        B = len(Lin_batch)
        T_max = max(Lout_batch) if B else 0
        sum_Lin = sum(Lin_batch)
        return (a * B * T_max) + b * (T_max * sum_Lin + B * (T_max - 1) * T_max / 2.0)
    
    @staticmethod
    def _batch_gen_ab_dynamic(a: float, b: float, Lin_batch: Sequence[int], Lout_batch: Sequence[int]) -> float:
        """
        Dynamic batch: shrink as sequences finish.
        At step t, only alive items (t < Lout_i) compute; attention uses (Lin_i + t) for alive i.
        """
        if not Lin_batch:
            return 0.0
        T_max = max(Lout_batch)
        total = 0.0
        for t in range(T_max):
            alive = [i for i, Lout in enumerate(Lout_batch) if t < Lout]
            B_t = len(alive)
            if B_t == 0:
                break
            total += a * B_t + b * sum(Lin_batch[i] + t for i in alive)
        return total

# --------- main post-hoc estimator ----------
    def approximate_profiler_cost(
        self,
        Lin_list: List[int],
        Lout_list: List[int],
        batches: Optional[List[List[int]]] = None,
        mode: str = "tflops",               # "tflops" or "units"
        mac_as_2flop: bool = True,
        generation_batch_mode: str = "static"  # "static" (default) or "dynamic"
    ) -> Dict[int, float]:
        """
        Post-hoc, profiler-like per-query cost estimator.

        Inputs:
          - Lin_list, Lout_list: per-query input/output token counts
          - batches: list of batches, each a list of query indices processed together.
                     If None, each query is treated as its own batch (no padding overhead).
          - mode: "tflops" (architecture-aware; weighted & scaled) or "units" (dimensionless).
          - generation_batch_mode:
              "static": batch stays full until all sequences finish (HF default behavior).
              "dynamic": batch shrinks as sequences finish.

        Returns:
          dict: query_index -> per-query cost (TFLOPs-ish if mode="tflops"; Units if mode="units").
        """
        assert len(Lin_list) == len(Lout_list)
        N = len(Lin_list)
        if batches is None:
            # fall back to one-by-one (no batching overhead)
            batches = [[i] for i in range(N)]

        # architecture-aware constants (or unitless)
        n, d, c_mlp, c_attn = self._arch_constants_from_config(self.config)
        if mode == "units":
            a = n * (d ** 2)      # c_mlp = 1
            b = n * d             # c_attn = 1
            scale = 1.0           # no MAC->FLOPs scaling, no 1e12
        else:
            a = n * c_mlp * (d ** 2)
            b = n * c_attn * d
            scale = 2.0 if mac_as_2flop else 1.0
            scale /= 1e12  # TFLOPs-ish

        out = {i: 0.0 for i in range(N)}

        for batch in batches:
            Lin_b = [int(Lin_list[i]) for i in batch]
            Lout_b = [int(Lout_list[i]) for i in batch]

            # Batch-level costs (close to profiler totals)
            prefill_batch = self._batch_prefill_ab(a, b, Lin_b)
            if generation_batch_mode == "dynamic":
                gen_batch = self._batch_gen_ab_dynamic(a, b, Lin_b, Lout_b)
            else:
                gen_batch = self._batch_gen_ab_static(a, b, Lin_b, Lout_b)

            # Ideal (no-pad) per-item weights for attribution
            ideal_pref = [self._ideal_prefill_ab(a, b, Lin_b[k]) for k in range(len(batch))]
            ideal_gen  = [self._ideal_gen_ab(a, b, Lin_b[k], Lout_b[k]) for k in range(len(batch))]
            sum_pref_w = sum(ideal_pref) or 1.0
            sum_gen_w  = sum(ideal_gen)  or 1.0

            # Allocate batch totals proportionally to ideal costs
            for local_k, qidx in enumerate(batch):
                share_pref = prefill_batch * (ideal_pref[local_k] / sum_pref_w)
                share_gen  = gen_batch    * (ideal_gen[local_k]  / sum_gen_w)
                out[qidx] += (share_pref + share_gen) * scale

        return out

    # ---------- main MC post-hoc estimator ----------
    def approximate_profiler_cost_mc(
        self,
        Lin_list: List[int],
        Lout_list: List[int],
        batch_size: int,
        R: int = 200,                      # simulations
        mode: str = "tflops",              # "tflops" or "units"
        batch_mode: str = "static",        # "static" or "dynamic"
        seed: int = 0
    ) -> Tuple[Dict[int, float], float]:
        """
        Returns (per_query_cost, global_overhead_factor).
        - per_query_cost: TFLOPs-ish (mode="tflops") or Units (mode="units")
        - global_overhead_factor: mean(batch_total) / sum(ideal_per_query), i.e., average pad/static overhead multiplier.
        """
        import random
        assert len(Lin_list) == len(Lout_list)
        N = len(Lin_list)
        n, d, c_mlp, c_attn = self._arch_constants_from_config(self.config)

        if mode == "units":
            a, b, scale = n * (d**2), n * d, 1.0
        else:
            a, b, scale = n * c_mlp * (d**2), n * c_attn * d, (2.0 / 1e12)  # MAC→FLOPs, TFLOPs scale

        # ideal (no padding) cost per query for attribution + for overhead baseline
        ideal = [self._ideal_prefill_ab(a,b,Lin_list[i]) + self._ideal_gen_ab(a,b,Lin_list[i],Lout_list[i]) for i in range(N)]
        ideal_sum = sum(ideal) or 1.0

        rng = random.Random(seed)
        accum = [0.0] * N
        overhead_mult_sum = 0.0

        for _ in range(R):
            idx = list(range(N))
            rng.shuffle(idx)  # random assignment to batches
            # chunk into batches of size <= batch_size
            for s in range(0, N, batch_size):
                batch = idx[s:s+batch_size]
                Lins = [Lin_list[i] for i in batch]
                Louts = [Lout_list[i] for i in batch]

                pre = self._batch_prefill_ab(a, b, Lins)
                gen = self._batch_gen_ab_static(a, b, Lins, Louts) if batch_mode=="static" else self._batch_gen_ab_dynamic(a, b, Lins, Louts)
                batch_total = pre + gen

                # allocate batch_total to items by their ideal shares within the batch
                w = [self._ideal_prefill_ab(a,b,Lins[k]) + self._ideal_gen_ab(a,b,Lins[k],Louts[k]) for k in range(len(batch)) ]
                wsum = sum(w) or 1.0
                for k, q in enumerate(batch):
                    accum[q] += scale * batch_total * (w[k] / wsum)

            # accumulate batch overhead factor vs. the same items' ideal sum
            overhead_mult_sum += (scale * sum(accum) / (scale * ideal_sum))  # they cancel; keep simple

        per_query = {i: accum[i] / R for i in range(N)}
        overhead_factor = overhead_mult_sum / R
        return per_query, overhead_factor
    
    @staticmethod
    def simulate_batches_by_len(Lin_list, max_batch_size=8):
        """
        Heuristic: sort by L_in (prompt length), then chunk into batches of size <= max_batch_size.
        This reduces padding in prefill and approximates real packing policies.
        Returns a list of batches, each a list of query indices.
        """
        idx = list(range(len(Lin_list)))
        idx.sort(key=lambda i: Lin_list[i])     # ascending prompt length
        batches = [idx[k:k+max_batch_size] for k in range(0, len(idx), max_batch_size)]
        return batches

    
    def calculate(self):
        l_in_list, l_out_list = [], []
        # self.lengths = []
        self.lengths = pickle.load(open('text_lengths.pkl', 'rb')) if Path('text_lengths.pkl').exists() else []
        for i, entry in enumerate(tqdm(self.results, desc='Calculating proxy cost:')):
            # print(entry)
            if 'text_length' in entry:
                l_in = entry['text_length']
                self.lengths.append(entry['text_length'])  # for safety
            else:
                l_in = self.lengths[i]
                entry['text_length'] = l_in
            l_in_list.append(l_in)
            if 'output_length' in entry:
                l_out = entry['output_length']  
            else:
                l_out = 1
                entry['output_length'] = l_out
            l_out_list.append(l_out)
            cost = self.proxy_query_cost(l_in, l_out)
            # cost = self.approximate_profiler_cost(l_in, l_out)
            entry['proxy_post_hoc_cost'] = cost
        if len(self.lengths) > 0: pickle.dump(self.lengths, open('text_lengths.pkl', 'wb'))
        
        # batches_hat = self.simulate_batches_by_len(l_in_list, max_batch_size=32)
        # approx_prof_cost = self.approximate_profiler_cost(l_in_list, l_out_list, 
        #                                                   batches=batches_hat)
        approx_prof_cost, _ = self.approximate_profiler_cost_mc(l_in_list, l_out_list, batch_size=32)
        # print(approx_prof_cost)
        for i, entry in enumerate(tqdm(self.results, desc='Calculating approx. profiler cost:')):
            entry['prof_post_hoc_cost'] = approx_prof_cost[i]
