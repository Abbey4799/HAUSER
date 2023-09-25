import numpy as np 
import pdb

class Scorer:
    def __init__(self):
        self.final_score = 0


    def get_relative_scores(self, data, mode = None):    
        _range = np.max(data) - np.min(data)
        if np.sum(_range) == 0:
            if mode == 'emo':
                if data[0] < 0:
                    return np.array([0] * len(data))
                else:
                    return np.array([0.5] * len(data))
            return np.array([0.5] * len(data))
        return ((data - np.min(data)) / _range)

    def get_final_score(self, score_list, mode = None):
        score_list = np.array(score_list)
        score_list[np.isnan(score_list)] = 0

        if mode == 'mnli' or mode == 'emo':
            return self.get_relative_scores(np.array(score_list), mode = mode)
        elif mode == 'creative_log':
            return self.get_relative_scores(- np.log(np.array(score_list) + 1))
        else:
            return self.get_relative_scores(np.array(score_list))