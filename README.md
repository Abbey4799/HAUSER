# HAUSER
Code and data for the paper "HAUSER: Towards Holistic and Automatic Evaluation of Simile Generation" (ACL 2023)

## Install Dependencies

```
conda create -n hauser python=3.6.13
conda activate hauser
pip install -r requirements.txt
```

Install the necessary dependencies for simile component extraction.
```
cd HAUSER/dependency
bash install.sh
python test_syntax.py
```
If test_syntax.py runs successfully, it proves that the dependency installation is successful.

Cache the necessary pre-trained models
```
cd HAUSER/dependency
python test_models.py
```

Download the Million-scale Simile Knowledge Base [MAPS-KB](https://arxiv.org/abs/2212.05254) from [Google Drive](https://drive.google.com/file/d/1d-Xn9OygjxhMoGPoMXSv48-etnOHjkve/view).

Put the file **MAPS-KB.csv**  into the folder [HAUSER/dependency].


## HAUSER

### Data

First, we finetune a pre-trained sequence-to-sequence model, BART, based on the simile generation datasets from [MAPS-KB](https://github.com/Abbey4799/MAPS-KB).
Then, we generate five simile candidates for 2500 literal sentences in the test set.
The data is in `HAUSER/data/simile_candidates_raw.json`.

Three annotators labeled 150 data samples, which can be found in `HAUSER/data/human_annotated.csv`. The annotations follow this format:
- `label1`, `label2`, `label3`: represent annotations from the three annotators
- Dimension suffixes:
  - `_q`: quality score
  - `_c`: creativity score
  - `_i`: informativeness score

### Metric

You can get the `relevance_score`, `logic_consistency_score`, `sentiment_consistency_score`, `creativity_score` and `informativeness_score`using the following script `score.sh`:

```
cd HAUSER
python code/score.py
```

You can reimplement the correlation analysis in `code/calculate_correlation.py`.

## Citation
```
@article{he2023hauser,
  title={HAUSER: Towards Holistic and Automatic Evaluation of Simile Generation},
  author={He, Qianyu and Zhang, Yikai and Liang, Jiaqing and Huang, Yuncheng and Xiao, Yanghua and Chen, Yunwen},
  journal={arXiv preprint arXiv:2306.07554},
  year={2023}
}
```