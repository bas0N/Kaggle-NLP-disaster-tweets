import numpy as np
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


#general processing function
def text_processing(tweet):
 
    #Searching and deleting punctuation
    nopunc = [char for char in tweet if char not in string.punctuation]
   
    #Joining the sentence where punctuation was deleted AND setting all letters to lowercase
    nopunc = ''.join(nopunc).lower()
   
    
    # Removing stopwords and returning the sentence
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]