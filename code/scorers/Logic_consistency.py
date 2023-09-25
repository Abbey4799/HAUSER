from scorers.scorer import Scorer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import scorers.utils as utils
import pdb



class Logic_Scorer(Scorer):
    def __init__(self):
        super(Scorer,self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("mujeensung/roberta-base_mnli_bc")
        self.model = AutoModelForSequenceClassification.from_pretrained("mujeensung/roberta-base_mnli_bc", num_labels=3, ignore_mismatched_sizes=True).to("cuda")
        

    def get_subscore(self, Sample):
        scores = self.get_mnli(Sample.cands, Sample.origin_sent)
        Sample.logic_consistency_score = self.get_final_score([score['1-contradiction'] for score in scores], mode='mnli')
    
    def get_mnli(self, cands, txt):
        scores = []
        for cand in cands:
            _ = txt  + ' ' +  cand.strip('\'')
            inputs = self.tokenizer(_, return_tensors="pt")
            for ii in inputs:
                inputs[ii] = inputs[ii].to("cuda")
            with torch.no_grad():
                logits = self.model(**inputs).logits

            scores.append(
                {
                    'entailment': torch.softmax(logits, dim = 1)[0][0].cpu().item(),
                    '1-contradiction': 1 - torch.softmax(logits, dim = 1)[0][2].cpu().item()
                }
            )
        return scores
