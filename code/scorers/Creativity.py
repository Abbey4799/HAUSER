from scorers.scorer import Scorer
import pandas as pd
import numpy as np
import scorers.utils as utils

class Creativity_Scorer(Scorer):
    def __init__(self, KB_path):
        super(Scorer,self).__init__()
        self.KB = pd.read_csv(KB_path)
    

    def get_subscore(self, Sample):
        comps = Sample.comps
        score_list = []
        for comp in comps:
            scores = []
            for cc in comp:
                scores.append(np.sum(np.array(self.KB[self.KB.new_vehicle == utils.clean(cc[1])]['count'])))
            scores = np.array(scores)
            score_list.append(np.mean(scores))
        Sample.creativity_score = self.get_final_score(score_list, mode='creative_log')
    
    