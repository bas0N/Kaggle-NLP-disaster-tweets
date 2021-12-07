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


l = nltk.WordNetLemmatizer()

def convert_nltk_tag_to_wordnet(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
    
def lemmatizing_prep(sentence):
    result =[]
    for word, tag in nltk.pos_tag(sentence):
        pos_tag = convert_nltk_tag_to_wordnet(tag)
        if pos_tag == '':
            result.append(word)
        else:
            result.append(l.lemmatize(word, pos = pos_tag))
    return ' '.join(result)

def lemmatizing(text,name):
    processed=text.apply(text_processing)
    output_file = name +'.csv'
    output_dir = Path('out')
    output_dir.mkdir(parents=True, exist_ok=True)
    lemmatized = processed.apply(lemmatizing_prep)
    lemmatized.to_csv(output_dir / output_file)
    return lemmatized
    
    