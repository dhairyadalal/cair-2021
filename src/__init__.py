import numpy as np 
import pandas as pd  
from annoy import AnnoyIndex
import pickle 
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import SentenceTransformer
import spacy 
from spacy.lang.en.stop_words import STOP_WORDS

class BM25Index():
    def __init__(self, 
                 idx_loc: str="saved_indices/bm250kapi_idx.pkl", 
                 idx_type: str = "bm250kpi"):
        self.idx = pickle.load(open(idx_loc, "rb"))
        self.idx_type = idx_type
        self.nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "ner"])
        
        
    def query(self, text, num_results: int=50):
        query_toks = self.preprocess_query(text)
        scores = self.idx.get_scores(query_toks)
        sorted_doc_ids = np.argsort(scores)[::-1][:num_results]
        sorted_scores = [scores[idx] for idx in sorted_doc_ids]
        return sorted_doc_ids.tolist(), sorted_scores  

    def preprocess_query(self, text):
        clean_text = re.sub(r'[^\w\s]', '', text.lower())        
        toks = []
        for tok in self.nlp(clean_text):
            if tok.text not in STOP_WORDS and tok.text.strip() != "":
                toks.append(tok.lemma_)
        return toks

class SemanticIndex():
    def __init__(self, 
                 semantic_weights = "msmarco-distilbert-base-v3",
                 idx_dim=768,
                 cache_loc="saved_indices/msmarco-distilbert-base-idx-cleaned.ann"):
        self.model = SentenceTransformer(semantic_weights)

        self.idx = AnnoyIndex(idx_dim, "angular")
        self.idx.load(cache_loc, True)
    
    def cosine_similarity_transform(self, angular_distance):
        return (2-(angular_distance**2))/2
    
    def query(self, text, num_results: int=50):
        encoded_query = self.model.encode(text)
        doc_idxs, distances = self.idx.get_nns_by_vector(encoded_query, num_results, search_k=-1, include_distances=True)
        scores = [self.cosine_similarity_transform(dist) for dist in distances]
        return doc_idxs, scores