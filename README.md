# TopicModeling_LDA
This project decribes topic modeling with LDA in Python3 language. I has been develeoped using a trivial example for low configured computer. 

Language: Python3

Packages: 
# Data Preprocessing 
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

Spacial Install Command:
> pip3 install spacy
> sudo python3 -m spacy download en
