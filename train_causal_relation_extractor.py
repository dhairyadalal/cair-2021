#%%
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TransformerWordEmbeddings, CharacterEmbeddings, FlairEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


columns = {0: "text", 1: "pos", 2: "ner"}
data_folder = "data/causenet/"

corpus = ColumnCorpus(data_folder, 
                      columns, 
                      train_file="train.txt",
                      test_file="test.txt",
                      dev_file="val.txt")

tag_type = "ner"
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)


embedding_types = [CharacterEmbeddings(),
                   FlairEmbeddings('news-forward'), 
                   FlairEmbeddings('news-backward')]

embeddings : StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


tagger : SequenceTagger = SequenceTagger(hidden_size=128, 
                                         embeddings=embeddings,
                                         tag_dictionary=tag_dictionary,
                                         tag_type=tag_type,
                                         dropout=0.25,
                                         rnn_layers=4,
                                         use_crf=True)




trainer : ModelTrainer = ModelTrainer(tagger, corpus)
trainer.train('resources/causal-tagger',
              embeddings_storage_mode='none',
              learning_rate=0.2,
              mini_batch_size=32,
              max_epochs=30,
              save_final_model=True,
              patience=3,
              num_workers=8)



# %%
