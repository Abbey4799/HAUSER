from scorers.scorer import Scorer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import scorers.utils as utils



class Sent_Scorer(Scorer):
    def __init__(self):
        super(Scorer,self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("mujeensung/roberta-base_mnli_bc")
        self.model = AutoModelForSequenceClassification.from_pretrained("mujeensung/roberta-base_mnli_bc", num_labels=3, ignore_mismatched_sizes=True).to("cuda")
        

    def get_subscore(self, Sample):
        try:
            scores = self.get_emotion_patio(Sample.cands, Sample.origin_sent, Sample.comps)
        except:
            scores = self.get_emotion_all(Sample.cands, Sample.origin_sent)

        Sample.sentiment_consistency_score = self.get_final_score([score['Delta_sentiment_score'] for score in scores], mode='emo')
    
    def get_emotion_all(self, cands, txt):
        inputs = self.tokenizer(txt, return_tensors="pt")
        for ii in inputs:
            inputs[ii] = inputs[ii].to("cuda")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        txt_emo = torch.softmax(logits, dim = 1)
        predicted_class_id_txt = logits.argmax().item()

        scores = []
        for cand in cands:
            _ = cand.strip('\'').strip('’')
            inputs = self.tokenizer(_, return_tensors="pt")
            for ii in inputs:
                inputs[ii] = inputs[ii].to("cuda")

            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_class_id = logits.argmax().item()
            _ = torch.softmax(logits, dim = 1) - txt_emo
            # 原句的结果、预测的Deltas、原句的label、预测的label
            scores.append({
                'origin_sentiment_score': txt_emo[0][predicted_class_id_txt].cpu().item(),
                'Delta_sentiment_score':  _[0][predicted_class_id_txt].cpu().item(),
                'Origin_label': predicted_class_id_txt,
                'Prediction_label': predicted_class_id
            })
        return scores



    def get_emotion_patio(self, cands, txt, comps):
        txt = txt[:txt.index(comps[0][2]) + len(comps[0][2])]
        inputs = self.tokenizer(txt, return_tensors="pt")
        for ii in inputs:
            inputs[ii] = inputs[ii].to("cuda")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        txt_emo = torch.softmax(logits, dim = 1)
        predicted_class_id_txt = logits.argmax().item()

        scores = []
        for idx,cand in enumerate(cands):
            cand = cand[:cand.index(comps[idx][1]) + len(comps[idx][1])]
            _ = cand.strip('\'').strip('’')
            inputs = self.tokenizer(_, return_tensors="pt")
            for ii in inputs:
                inputs[ii] = inputs[ii].to("cuda")
            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_class_id = logits.argmax().item()
            _ = torch.softmax(logits, dim = 1) - txt_emo
            # 原句的结果、预测的Deltas、原句的label、预测的label
            scores.append({
                'origin_sentiment_score': txt_emo[0][predicted_class_id_txt].cpu().item(),
                'Delta_sentiment_score':  _[0][predicted_class_id_txt].cpu().item(),
                'Origin_label': predicted_class_id_txt,
                'Prediction_label': predicted_class_id
            })
        return scores
