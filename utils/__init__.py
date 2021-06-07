from bs4 import BeautifulSoup
import pytrec_eval
import spacy 
from typing import List
import pickle 

nlp = spacy.load('en_core_web_sm', disable=["tagger", "ner", "parser"])

class DocumentDB():

    def __init__(self, loc: str="data/parsed_docs_with_toks.pkl"):
        self.df = pickle.load(open(loc, "rb"))
    
    def lookup_idx(self, idx):
        docno = self.df.iloc[idx]["docno"]
        return docno

    def batch_idx_lookup(self, idxs):
        docnos =  self.df.iloc[idxs]["docno"].tolist()
        return docnos

    def lookup_docno(self, docno):
        text = self.df.query("docno == @docno")["text"].item()
        text = " ".join(text.split())
        return text 
    
    def batch_docno_lookup(self, ids):
        docs = [self.lookup_docno(id) for id in ids]
        return docs 
    
    def get_docs_from_ids(self, idxs):
        docnos = self.batch_idx_lookup(idxs)
        texts = self.batch_docno_lookup(docnos)
        return texts
    
    def get_doc_from_id(self, idx):
        docno = self.lookup_idx(idx)
        text = self.lookup_docno(docno)
        return text

def preprocess_query(text):
    cleaned_query = []
    for tok in nlp(text):
        tok = tok.lemma_.lower()
        if tok.isalpha():
            cleaned_query.append(tok)
    return cleaned_query

def read_qrel_from_file(file_path: str) -> dict:
    """ Method return json representation of qrel file """
    
    qrel_json = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            line_split = line.split("\t")
            query_id = line_split[0]
            doc = line_split[2]
            rel = int(line_split[3].strip())

            if query_id in qrel_json:
                qrel_json[query_id][doc]=rel
            else:
                qrel_json[query_id]={doc:rel}
                
    return qrel_json


def evaluate_run(run: dict, qrel: dict,  metrics: set = {'map', 'ndcg'}) -> dict:
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
    return evaluator.evaluate(run)

def extract_topics_from_file(file: str) -> dict:
    soup = BeautifulSoup(open(file, "r"), 'xml')

    extracted_topics = []
    for topic in soup.find_all("top"):
        number = str(topic.num.text)
        title = topic.title.text
        narr = topic.narr.text
        extracted_topics.append({"number": number, "title": title, "narrative": narr})

    return extracted_topics

