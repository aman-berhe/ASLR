from transformers import AutoModel, AutoTokenizer,AutoModelForTokenClassification
from transformers import pipeline

import pandas as pd
import json
import os
import sys
from tqdm import tqdm

model_path="/home/aberhe/Projects/SANTAL/ASLR/data/ner/models/"


entity_list=["O","B-Species","I-Species"]
entity_list_d=["O","B-Disease","I-Disease"]
entity_list_c=["O","B-Chemical","I-Chemical"]

label2Index={e:i for i,e in enumerate(entity_list)}
index2Label={i:e for i,e in enumerate(entity_list)}

label2Index_d={e:i for i,e in enumerate(entity_list_d)}
index2Label_d={i:e for i,e in enumerate(entity_list_d)}

label2Index_c={e:i for i,e in enumerate(entity_list_c)}
index2Label_c={i:e for i,e in enumerate(entity_list_c)}



model=AutoModelForTokenClassification.from_pretrained(os.path.join(model_path,"NER_Model_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_Species_s800_dataset_ner_cleaned"),num_labels=3, id2label=index2Label, label2id=label2Index)
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model_d=AutoModelForTokenClassification.from_pretrained(os.path.join(model_path,"NER_Model_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_Disease_NCBI-disease_dataset_ner_cleaned"),num_labels=3, id2label=index2Label_d, label2id=label2Index_d)
model_c=AutoModelForTokenClassification.from_pretrained(os.path.join(model_path,"NER_Model_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_Chemical_bc5cdr_dataset_ner_cleaned"),num_labels=3, id2label=index2Label_c, label2id=label2Index_c)


pipe = pipeline("ner", model=model,tokenizer=tokenizer)
pipe_d = pipeline("ner", model=model_d,tokenizer=tokenizer)
pipe_c = pipeline("ner", model=model_c,tokenizer=tokenizer)

ssi=["","",""]
for text in ssi:
    res=pipe(text,aggregation_strategy="max")
    res_d=pipe_d(text,aggregation_strategy="max")
    res_c=pipe_c(text,aggregation_strategy="max")
    for r in res:
        if r["entity_group"]!="O":
            print(r)
    for r in res_d:
        if r["entity_group"]!="O":
            print(r)
    for r in res_c:
        if r["entity_group"]!="O":
            print(r)