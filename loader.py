import pickle, torch, helpers
import numpy as np
from datasets import load_dataset

class StructuresLoader:
    def __init__(self, dataset_name, s_llm, m_llm, ):
        self.dataset_name = dataset_name
        self.sLLM = s_llm
        self.mLLM = m_llm

        self.dataset = None
        self.sLLM_results = None
        self.mLLM_results = None

    def load_dataset(self):
        if self.dataset_name == 'sst2':
            self.dataset = load_dataset('nyu-mll/glue', 'sst2',)
        elif self.dataset_name == 'emotion':
            self.dataset = load_dataset('dair-ai/emotion',)
        elif self.dataset_name == 'agnews':
            self.dataset = load_dataset('fancyzhx/ag_news')
        elif self.dataset_name == 'fakenews':
            self.dataset = load_dataset('Pulk17/Fake-News-Detection-dataset')
        elif self.dataset_name == 'squad':
            self.dataset = load_dataset('rajpurkar/squad')
        else:
            self.dataset = load_dataset('wmt/wmt_t2t')

    @staticmethod
    def load_results_file(path): return pickle.load(open(path, 'rb'))

    @staticmethod
    def get_result_values(results, key): return [entry[key] for entry in results]

    def obtain_binary_betas(self, return_both=False):
        assert self.sLLM_results is not None
        positive_probs = self.get_result_values(self.sLLM_results, 'prob_positive')  # list
        negative_probs = self.get_result_values(self.sLLM_results, 'prob_negative')  # list
        arr = [np.array(l) for l in [negative_probs, positive_probs]] 
        if return_both:
            return np.vstack(arr).T  # (N, 2)
        return np.vstack(arr).T.max(axis=1)  # (N,)
    
    def obtain_multiclass_betas(self, return_all=False):
        assert self.sLLM_results is not None
        probs = self.get_result_values(self.sLLM_results, 'probs')  # list of lists
        if return_all:
            # print(np.array(probs).shape)
            return np.array(probs)  # for calibration
        # print(np.array(probs).max(axis=1).shape)
        return np.array(probs).max(axis=1)
    
    def obtain_generation_betas(self, mean=True, quantile_based=False, sigmoid=False):
        assert self.sLLM_results is not None
        token_logprobs = self.get_result_values(self.sLLM_results, 'logprobs')  # list of arrays
        if quantile_based:
            return np.array(
                self.get_result_values(self.sLLM_results, 'pred_confidence_reg')
            )
        if sigmoid:
            sigmoid = lambda z : 1/(1 + np.exp(-z))
            if mean: return sigmoid(np.array(token_logprobs).mean(axis=1)) # (N, max_tokens) --> (N,)
            return sigmoid(np.array(token_logprobs).sum(axis=1)) # (N, max_tokens) --> (N,)
        else:
            if mean: return helpers.min_max_normalize(np.array(token_logprobs).mean(axis=1))
            return helpers.min_max_normalize(np.array(token_logprobs).sum(axis=1))
        

    
