from scorers.scorer import Scorer
import pandas as pd
import numpy as np
import scorers.utils as utils

class Relevance_Scorer(Scorer):
    def __init__(self, KB_path):
        super(Scorer,self).__init__()
        self.KB = pd.read_csv(KB_path)
    

    def get_subscore(self, Sample):
        comps = Sample.comps
        score_list = []
        for comp in comps:
            scores = []
            for cc in comp:
                _ = []
                for index,row in self.KB[(self.KB.new_topic == utils.clean(cc[0])) & (self.KB.new_vehicle == utils.clean(cc[1]))].iterrows():
                    _.append(row['plausibility'] * row['count'])
                if len(_) == 0:
                    scores.append(0)
                else:
                    scores.append(np.sum(np.array(_)))
            score_list.append(np.average(np.array(scores)))
        Sample.relevance_score = self.get_final_score(score_list, mode='KB')
    
    