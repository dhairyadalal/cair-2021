import spacy 
from spacy.lang.en.stop_words import STOP_WORDS
import re
import swifter
import pandas as pd  
import pickle

data = pd.read_csv("data/parsed_docs.csv")

nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "ner"])

def clean_lemmatize(text):
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    
    toks = []
    for tok in nlp(clean_text):
        if tok.text not in STOP_WORDS and tok.text.strip() != "":
            toks.append(tok.lemma_)
    return toks

data["tokens"] = data["text"].swifter.apply(lambda x: clean_lemmatize(x))

pickle.dump(data, open("data/parsed_docs_with_toks.pkl", "wb"))
