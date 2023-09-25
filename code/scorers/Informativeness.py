from scorers.scorer import Scorer
import nltk
import numpy as np


class Inform_Scorer(Scorer):
    def __init__(self):
        super(Scorer,self).__init__()
    

    def get_subscore(self, Sample):
        comps = Sample.comps
        lens = []
        for comp in comps:
            len_ = []
            for _ in comp:
                vv = _[1]
                len_.append(len(nltk.word_tokenize(vv)))
            lens.append(np.average(np.array(len_)))
        Sample.inform_score = self.get_final_score(lens)
    
    