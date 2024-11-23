import numpy as np

import gensim.downloader as api
from gensim.models import KeyedVectors

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from nltk.tag.crf import CRFTagger
import pycrfsuite

import warnings
warnings.filterwarnings("ignore")

def load(filename):
    listeDoc = list()
    with open(filename, "r") as f:
        doc = list()
        for ligne in f:
            #print "l : ",len(ligne)," ",ligne
            if len(ligne) < 2: # fin de doc
                listeDoc.append(doc)
                doc = list()
                continue
            mots = ligne.replace("\n","").split(" ")
            doc.append((mots[0],mots[2])) # mettre mots[2] à la place de mots[1] pour le chuncking
    return listeDoc


def randomvec():
    default = np.random.randn(300)
    default = default  / np.linalg.norm(default)
    return default


def vectorize(word, wv_model, mean=False):
    """
    This function should vectorize one review

    input: str
    output: np.array(float)
    """    
    text_vectorized = []

    if word in wv_model.key_to_index :
        text_vectorized.append(wv_model[word])
    
            
    return np.array(text_vectorized).reshape(-1,)
