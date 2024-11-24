from transformers import AutoTokenizer, AutoModelForTokenClassification,DataCollatorForTokenClassification, EarlyStoppingCallback
from transformers import TrainingArguments, Trainer, pipeline
import datasets

import pandas as pd
import numpy as np
import os
import sys
import torch

#Loggings
import logging
from tqdm import tqdm,trange

#metrics for evaluation
from sklearn.metrics import confusion_matrix, classification_report, precision_score

MODELS_DIR="models"
class AlibertForTokenClassification():
    def __init__(self,model_name="PubmedBERT",tokenizer_name="PubMedBERT",list_entities=[],train=True):
        """
        Initializes the AlibertForTokenClassification model.

        Args:
            model_name (str): The model name should be the model name in which the folder name and the model inside it is the same (huggingface style): e.g. AliBERT/AliBERT. Otherwise the full path to the model
            tokenizer_name (str): The tokenizer name should be the tokenizer to be used in which the folder name and the model inside it is the same: e.g. AliBERT/AliBERT
            cased (bool): Flag indicating whether the model is cased or uncased.
            load (bool): Flag indicating whether to load the model and tokenizer. Model will be used for inference. Make sure the model classes and the inpute number_classes are equal
            number_classes (int): Number of classes for token classification.
            train (bool): Flag indicating whether to train the model.
        """

        self.train=train
        self.model_name=model_name
        self.tokenizer_name=tokenizer_name
        self.list_entities=list_entities# The order of the entities in the list is important: since it will be used for prediction later
        self.number_classes=len(list_entities)
        print(self.tokenizer_name)
        self.tokenizer=AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model=AutoModelForTokenClassification.from_pretrained(self.model_name,num_labels=self.number_classes)
        #Load model to GPU if there is any
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)

        self.data_collator=DataCollatorForTokenClassification(tokenizer=self.tokenizer)
            

    def train_ner_model(self,dataset_path,save_mode=True):
        #self.number_classes=number_classes
        """
        Prepare the dataset fro training:
        it takes each dataset (train,validation, test) and tokenize the sequence of tokens (list of words, or list of strings): make sure each list have exact the same number of labels as its length
        padding, and trancation will be used, is_split_into_words will ne True is the inpute string is represented in a list which each element is an entity (can be many classes but one for one element) or non entity
        """
        self.dataset_path=dataset_path
        self.save_mode=save_mode

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(examples["tokens"], max_length = 512, padding=True, truncation=True, is_split_into_words=True)
            
            labels = []
            for i, label in enumerate(examples["ner_token_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(label[word_idx])
                    previous_word_idx = word_idx

                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        self.dataset=datasets.load_from_disk(self.dataset_path) 
        self.tokenized_dataset=self.dataset.map(tokenize_and_align_labels,batched=True)

        self.training_args = TrainingArguments(
            output_dir                  = 'test',
            overwrite_output_dir        = True, # To overwrite a checkpoint while training
            evaluation_strategy         = 'steps',
            eval_steps                  = 50,
            learning_rate               = 2e-5,
            save_total_limit            = 3       , # Only last 3 models are saved. Older ones are deleted
            per_device_train_batch_size = 4,
            per_device_eval_batch_size  = 4,
            logging_strategy            = 'steps',
            logging_steps               = 10,
            num_train_epochs            = 5,
            weight_decay                = 0.01,
            max_steps                   = 300, 
            gradient_accumulation_steps = 2,
            gradient_checkpointing      = False,
            warmup_steps                = 20,
            metric_for_best_model       = 'eval_loss',
            load_best_model_at_end      = True
                )
        
        self.trainer = Trainer(
            model = self.model,
            args=self.training_args,
            train_dataset = self.tokenized_dataset['train'],
            eval_dataset = self.tokenized_dataset['valid'],
            tokenizer = self.tokenizer,
            data_collator = self.data_collator,
            compute_metrics=self.compute_metrics_per_class,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
            )

        #Training the NER model
        logging.info("Ready to train on GPU device name: {}".format(torch.cuda.get_device_name(0)))
        self.trainer.train()

        logging.info("Evaluating finetunned model on the evaluation dataset!")

        self.evaluation_results=self.trainer.predict(self.tokenized_dataset["test"])
        
        self.df_results=self.result_to_DF()

        if self.save_mode:
            logging.info("Saving finetunned model")
            self.trainer.save_model("/home/aberhe/Projects/SANTAL/ASLR/data/ner/models/NER_Model_"+self.model_name.split("/")[-1]+"_"+self.dataset_path.split("/")[-1])
        else:
            logging.warning("Attention: Model is not saved yet")
    
    def perclass_metric(self, y_true,y_pred,labels):
        #the input (list of labels) class must be the same sequence of list
        y_true=[labels[i] for i in y_true]
        y_pred=[labels[i] for i in y_pred]
        
        classes=list(set(y_true))

        cm = confusion_matrix(y_true, y_pred)
        res_reports=classification_report(y_true, y_pred, target_names=classes, digits=3,output_dict=True)

        res={}# # A dictionary to store the results
        # Iterate over each class
        for k in classes:
            for kk in res_reports[k].keys():
                res[k+"_"+kk]=res_reports[k][kk]

        #Store additional overall metrics
        res["overall_accuracy"]=res_reports["accuracy"]
        res["macro_precision"]=res_reports["macro avg"]["precision"]
        res["macro_recall"]=res_reports["macro avg"]["recall"]
        res["macro_f1-score"]=res_reports["macro avg"]["f1-score"]

        """
        Returns a dictionary of metrics: All metric for each entity is returned 
        """
        return res

    def compute_metrics_per_class(self,p):
        predictions, labels = p
        prediction = np.argmax(predictions, axis=-1)

        # Filter out predictions and labels where label is -100
        true_predictions = [[p for p, l in zip(pred, label) if l!=-100] for pred,label in zip(prediction, labels)]
        true_labels = [[l for p, l in zip(pred, label) if l!=-100] for pred,label in zip(prediction, labels)]

        # Flatten the lists
        true_predictions = [l for sublabel in true_predictions for l in sublabel]
        true_labels = [l for sublabel in true_labels for l in sublabel]
        
        """
        precision  = precision_score(true_predictions,true_labels,average='micro')
        recall = recall_score(true_predictions,true_labels,average='micro')
        f1 = f1_score(true_predictions,true_labels,average='macro')
        res = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                #"accuracy": accuracy,
            } 
        #trainer.log_metrics
        """
        res=self.perclass_metric(true_labels,true_predictions,self.list_entities)
        return res
    
    
    def result_to_DF(self):
        precision=[]
        recall=[]
        f1_score=[]
        df=pd.DataFrame(columns=["Entity","Precision","Recall","F1_Score"])
        for en in self.list_entities:
            precision.append(self.evaluation_results[-1]["test_"+en+"_precision"])
            recall.append(self.evaluation_results[-1]["test_"+en+"_recall"])
            f1_score.append(self.evaluation_results[-1]["test_"+en+"_f1-score"])

        df.Entity=self.list_entities
        df.Precision=precision
        df.Recall=recall
        df.F1_Score=f1_score 

        df.loc[len(df)]=["Over_all_macro",self.evaluation_results[-1]['test_macro_precision'],self.evaluation_results[-1]['test_macro_recall'],self.evaluation_results[-1]['test_macro_f1-score']]
        
        return df
                            
    
    



        
