#%%
import pandas as pd 
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer, util
import pickle 
import swifter

print("load data")
df = pickle.load(open("data/parsed_docs_with_toks.pkl", "rb"))
df = df.query("bd_news_flag != 1")

print("df filt len", len(df))

print("loading model")
model = SentenceTransformer('msmarco-distilbert-base-v3')

print("embedding")
passage_embeddings = model.encode(df["text_clean"].tolist(), show_progress_bar=True)

print("dumping embeddings")
pickle.dump(passage_embeddings, open("data/passages-msmarco-distilbert-base-v3-cleaned.pkl", "wb"))


# passage_embeddings = pickle.load(open("data/passages-msmarco-distilbert-base-v3.pkl", "rb"))

print('building index')
dim = 768
idx = AnnoyIndex(dim, "angular")

for i,e in enumerate(passage_embeddings):
    idx.add_item(i, e)
    
print("building index")    
idx.build(1000)

idx.save("saved_indices/msmarco-distilbert-base-idx-cleaned.ann")
# %%
