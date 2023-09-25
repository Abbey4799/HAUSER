from scorers.Relevance import Relevance_Scorer
from scorers.Creativity import Creativity_Scorer
from scorers.Sentiment_consistency import Sent_Scorer
from scorers.Logic_consistency import Logic_Scorer
from scorers.Informativeness import Inform_Scorer
from scorers.Sample import Sample
from scorers import utils
from tqdm import tqdm
import argparse
import pdb




def main(args):
    samples = utils.readjson(args.data_path)

    print('relevance scorer loading...')
    relevance_scorer = Relevance_Scorer(KB_path=args.KB_path)
    print('sentiment consistency scorer loading...')
    sent_scorer = Sent_Scorer()
    print('logic consistency scorer loading...')
    logic_scorer = Logic_Scorer()
    print('Creativity scorer loading...')
    creativity_scorer = Creativity_Scorer(KB_path=args.KB_path)
    print('Informativeness scorer loading...')
    inform_scorer = Inform_Scorer()
    
    new_samples = []
    for sample in tqdm(samples[:5]):
        sample = Sample(sample['origin_sentence'], sample['cands'])
        relevance_scorer.get_subscore(sample)
        sent_scorer.get_subscore(sample)
        logic_scorer.get_subscore(sample)
        creativity_scorer.get_subscore(sample)
        inform_scorer.get_subscore(sample)
        new_samples.append(sample)
        sample.display_score()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--KB_path", type=str, default='dependency/MAPS-KB.csv')
    parser.add_argument("--data_path", type=str, default='data/simile_candidates_raw.json')
    parser.add_argument("--saved_score_path", type=str, default='scores/')
    args = parser.parse_args()

    main(args)