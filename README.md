# Overview
CAIR (Causality-driven Adhoc Information Retrieval) is a shared task organized by the Forum for Information Retrieval (FIRE 2021). The goal is to develop a causal search system which will retrieve documents that provide information n the likely causes leading to an query event. 


More information can be found here [CAIR Website] (https://cair-miners.github.io/CAIR-2021-website/#task).

## Data setup
Data will need to download the `parsed_docs.csv`. Parsed docs is csv with following columns:

    - docno: the id of the document
    - text: extracted text of the document 

To replicate the processed data: 
To download the original dataset: 
1. Download the data: 
The data can be found here:
http://www.isical.ac.in/~fire/data/docs/adhoc/en.docs.2011.tar.gpg

2. Decrypt the dataset: 
It is stored as an encrypted gpg file. To decrypt run the following:
`` gpg --decrypt en.docs.2011.tar.gpg -o en.docs.2011.tar``

3. Run extract_data.py 
`` python data/extract_data.py``
