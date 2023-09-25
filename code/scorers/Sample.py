import scorers.utils as utils
import numpy as np

class Sample:
    def __init__(self, origin_sentence, candidates):
        self.origin_sent = origin_sentence
        self.cands = candidates
        self.comps = self.get_components()
        self.relevance_score, self.creativity_score, self.inform_score, self.logic_consistency_score, self.sentiment_consistency_score = np.nan, np.nan, np.nan, np.nan, np.nan
        self.final_score = np.nan

    def get_components(self):
        comps_list = []
        for cand in self.cands:
            comps_list.append(utils.get_component_tree(cand))
        return comps_list
        
    def print_score(self, score_name, scores, idx):
        if not isinstance(scores, float):
            print(f'{score_name}: {scores[idx]} \n')
        else:
            print(f'{score_name} is not calculated..')

    def display_score(self):
        print(f'=============')
        print(f'literal sentence:{self.origin_sent}')
        print('-----')
        for idx in range(len(self.cands)):
            print(f'Simile candidate: ', self.cands[idx])
            print(f'Extracted Simile components: ', self.comps[idx])
            print('-----')
            self.print_score('relevance_score', self.relevance_score, idx)
            self.print_score('logic_consistency_score', self.logic_consistency_score, idx)
            self.print_score('sentiment_consistency_score', self.sentiment_consistency_score, idx)
            self.print_score('creativity_score', self.creativity_score, idx)
            self.print_score('inform_score', self.inform_score, idx)
        print(f'=============\n')