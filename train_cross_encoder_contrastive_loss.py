#%%
from utils import DocumentDB
import pytrec_eval
from utils import read_qrel_from_file, evaluate_run, extract_topics_from_file
import re
import numpy as np
from nltk.tokenize import sent_tokenize
import numpy as np
import pickle 

db = DocumentDB()
print("loaded db")


topics = extract_topics_from_file("qrels/2020/topics_test.txt")
qrel = read_qrel_from_file("qrels/2020/cair2020_qrel.txt")

metrics: set = {'map', 'ndcg', 'P_5'}
evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)

def print_cum_stats(run):
    run_results = evaluator.evaluate(run)

    map_scores = [v["map"] for k,v in run_results.items()]
    p_scores  = [v["P_5"] for k,v in run_results.items()]
    ndcg_scores = [v['ndcg'] for k,v in run_results.items()]

    print("Aggregate results")
    print("Average MAP: ", np.mean(map_scores))
    print("Average P_5: ", np.mean(p_scores))
    print("Average NDCG: ", np.mean(ndcg_scores))
    

from sentence_transformers import CrossEncoder
ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Base loss
from sentence_transformers import SentencesDataset, losses
from sentence_transformers.readers import InputExample

examples = []

for topic in topics:
    gold = qrel[topic["number"]].items()
    query = topic["title"].strip()

    for item in gold:
        try:
            doc = db.lookup_docno(item[0])
            examples.append(InputExample(texts=[query, doc], label=item[1]))
        except:
            continue
print("finished", len(examples))

#%%
from torch.utils.data import DataLoader
train_dataset = SentencesDataset(examples, ranker)
train_dl = DataLoader(train_dataset, shuffle=True, batch_size = 16)
train_loss = losses.OnlineContrastiveLoss(model=ranker)

ranker.fit(train_dataloader=train_dl, 
           epochs=20,  
           output_path="ranker/constrastive_loss/",
           save_best_model = True)

pickle.dump(ranker, open("ranker/constrastive_loss/ranker_contrastive_loss_20_epochs.pkl", "wb"))

from tqdm.notebook import tqdm

run = {}
for topic in tqdm(topics):
    number = topic["number"]
    query = topic["title"]

    extracted_ids = [k for k in qrel[number].keys()]

    doc_ids = []
    for id in extracted_ids:
        try:
            db.lookup_docno(id)
            doc_ids.append(id)
        except:
            continue

    texts = db.batch_docno_lookup(doc_ids)
    pairs = list(zip([query] * len(texts), texts))

    scores = ranker.predict(pairs)
    scores = scores.tolist()
    
    doc_scores = dict(zip(doc_ids, scores))
    run[number] = doc_scores
print_cum_stats(run)