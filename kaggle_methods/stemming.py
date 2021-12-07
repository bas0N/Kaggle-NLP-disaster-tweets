import numpy as np
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from pathlib import Path
from processing import text_processing 


#import of porter stemmer from the nltk library
p = nltk.PorterStemmer()

#stemming method that is applied to consecutively occuring sentenes in a series
def stemming_prep(sentence):
    return ' '.join(list(map(lambda word: p.stem(word), sentence)))

#ultimate steming method taking a dataframe column od strings and desired file name as an input 
def stemming(text,name):
    processed=text.apply(text_processing)
    
    output_file = name+'.csv'
    output_dir = Path('out')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stemmed = processed.apply(stemming_prep)
    stemmed.to_csv(output_dir / output_file)
    return stemmed