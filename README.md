# Example Created By 
May 8, 2020
Md Saeed Siddik, 
IIT University of Dhaka, BD
https://sites.google.com/view/saeedsiddik

# TopicModeling_LDA
This project decribes topic modeling with LDA in Python3 language. I has been develeoped using a trivial example for low configured computer. 

# Language: 
Python3

# Library Packages: 
For Data Preprocessing 

import re

import numpy as np

import pandas as pd

from pprint import pprint


For Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel


For spacy for lemmatization

import spacy


For Plotting tools

import pyLDAvis

import pyLDAvis.gensim  # don't skip this

import matplotlib.pyplot as plt


For Spacial Install Command:
> pip3 install spacy

> sudo python3 -m spacy download en

Supporting Source: 
https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
https://www.kaggle.com/rahulvks/topic-modeling-using-gensim
