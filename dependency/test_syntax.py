import re
import benepar, spacy
import nltk

import json
import os
import re

from stanfordcorenlp import StanfordCoreNLP


# Syntax Parsing
nlp = spacy.load('en_core_web_sm')
if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("./benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "./benepar_en3"})

nlp_rep = StanfordCoreNLP('./stanford-corenlp-4.2.0')

print(nlp('As for Paganel, he wept like a little boy who does not think of hiding his emotion.').sents)
print(nlp_rep.coref('As for Paganel, he wept like a little boy who does not think of hiding his emotion.'))
