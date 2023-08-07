import numpy as np
import pandas as pd
import os

# We need the sys package to load modules from another directory:
import sys
sys.path.append('../')
from preprocessing.preprocessors import *

import evaluate

from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from tqdm import tqdm


def get_bert(dataset, level, bert = "distilbert-base-uncased", models_dir = "../models/distilbert/", results_dir = "../results/distilbert/"):
    if dataset == "clustered_df":
        if level == "level0":
            id2label = {0: 'sadness', 1: 'neutral', 2: 'love', 3: 'gratitude', 4: 'disapproval',
                    5: 'amusement', 6: 'disappointment', 7: 'realization', 8: 'admiration', 9:
                    'annoyance', 10: 'confusion', 11: 'optimism', 12: 'excitement', 13: 'caring',
                    14: 'remorse', 15: 'joy', 16: 'approval', 17: 'embarrassment', 18: 'surprise',
                    19: 'curiosity', 20: 'anger', 21: 'grief', 22: 'disgust', 23: 'pride', 24: 'desire',
                    25: 'relief', 26: 'fear', 27: 'nervousness'}
            label2id = {'sadness': 0, 'neutral': 1, 'love': 2, 'gratitude': 3, 'disapproval': 4,
                    'amusement': 5, 'disappointment': 6, 'realization': 7, 'admiration': 8,
                    'annoyance': 9, 'confusion': 10, 'optimism': 11, 'excitement': 12, 'caring': 13,
                    'remorse': 14, 'joy': 15, 'approval': 16, 'embarrassment': 17, 'surprise': 18,
                    'curiosity': 19, 'anger': 20, 'grief': 21, 'disgust': 22, 'pride': 23, 'desire': 24,
                    'relief': 25, 'fear': 26, 'nervousness': 27}
        elif level == "level1":
            id2label = {0: 'dis_sad', 1: 'neutral', 2: 'love', 3: 'gra_rel', 4: 'disapproval',
                    5: 'amusement', 6: 'app_rea', 7: 'pri_adm', 8: 'ang_ann', 9: 'cur_con', 10: 'des_opt',
                    11: 'exc_joy', 12: 'caring', 13: 'rem_emb', 14: 'embarrassment', 14: 'surprise',
                    16: 'grief', 17: 'disgust', 18: 'fea_ner'}
            label2id = {'dis_sad': 0, 'neutral':1, 'love':2, 'gra_rel':3, 'disapproval':4,
                    'amusement':5, 'app_rea':6, 'pri_adm':7, 'ang_ann':8, 'cur_con':9, 'des_opt':10,
                    'exc_joy':11, 'caring':12, 'rem_emb':13, 'embarrassment':14, 'surprise':15,
                    'grief':16, 'disgust':17, 'fea_ner':18}
        elif level == "level2":
            id2label = {0: 'dis_sad_gri', 1:'neutral', 2:'exc_joy_lov', 3:'pri_adm_gra_rel',
                    4:'disapproval', 5:'amusement', 6:'app_rea', 7:'dis_ang_ann',
                    8:'sur_cur_con', 9:'des_opt_car', 10:'rem_emb', 11:'embarrassment',
                    12:'fea_ner'}
            label2id = {'dis_sad_gri':0, 'neutral':1, 'exc_joy_lov':2, 'pri_adm_gra_rel':3,
                    'disapproval':4, 'amusement':5, 'app_rea':6, 'dis_ang_ann':7,
                    'sur_cur_con':8, 'des_opt_car':9, 'rem_emb':10, 'embarrassment':11,
                    'fea_ner':12}
        elif level == "level3":
            id2label = {0: 'rem_emb_dis_sad_gri', 1:'neutral', 2:'amu_exc_joy_lov',
                    3: 'pri_adm_gra_rel_app_rea', 4: 'dis_ang_ann_dis', 5: 'sur_cur_con',
                    6: 'des_opt_car', 7: 'embarrassment', 8: 'fea_ner'}
            label2id = {'rem_emb_dis_sad_gri':0, 'neutral':1, 'amu_exc_joy_lov':2,
                    3:'pri_adm_gra_rel_app_rea', 4:'dis_ang_ann_dis', 5: 'sur_cur_con',
                    6: 'des_opt_car', 7: 'embarrassment', 8: 'fea_ner'}
        else:
            print("false level-input for clustered_df")
    elif dataset == "plutchik_df":
        if level == "plutchik":
            id2label = {0: 'betrübt', 1:'neutral', 2:'verliebt', 3:'ehrfürchtig', 4:'angewidert',
                5: 'begeistert', 6: 'enttäuscht', 7: 'klar', 8: 'bewundernd', 9: 'wütend',
                10:'erstaunt', 11:'optimistisch', 12:'bereuend', 13:'embarrassment',
                14: 'streitlustig', 15:'erschrocken'}
            label2id = {'betrübt':0, 'neutral':1, 'verliebt':2, 'ehrfürchtig':3, 'angewidert':4,
                'begeistert':5, 'enttäuscht':6, 'klar':7, 'bewundernd':8, 'wütend':9,
                'erstaunt':10, 'optimistisch':11, 'bereuend':12, 'embarrassment':13,
                'streitlustig':14, 'erschrocken':15}
        else:
            print("false level-input for plutchik_df, level must be plutchik")
    else:
        print("false dataset-input, dataset must be clustered_df or plutchik_df")       

    tokenizer = AutoTokenizer.from_pretrained(bert)

    dataset["label"] = dataset[level].map(label2id.get)

    training_data = dataset.groupby(level).sample(frac=0.8, random_state=25) # stratified sampling
    testing_data = dataset.drop(training_data.index)

    training_data = Dataset.from_pandas(training_data) # create transformers compatible dataset from dataframe
    testing_data = Dataset.from_pandas(testing_data)

    def tokenize_function(examples): # replace representation of data, convert column text to tensor-based representation
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_training_data = training_data.map(tokenize_function, batched=True) # convert text to tensor form
    tokenized_testing_data = testing_data.map(tokenize_function, batched=True)

    classCounts = dataset.level.value_counts() 
    numberOfDocuments = len(dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Padding -> map all tensors to the same size
    accuracy = evaluate.load("accuracy") # define evaluation method -> quality

    def compute_metrics(eval_pred): # function calculation metric
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        bert, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    
    # create Models directory if it's not already there
    if os.path.exists(models_dir) == False:
        os.makedirs(models_dir)
    # create Results directory if it's not already there
    if os.path.exists(results_dir) == False:
        os.makedirs(results_dir)

    # training
    training_args = TrainingArguments(
        output_dir= models_dir+"model_"+level,
        learning_rate=2e-5,  # standard
        per_device_train_batch_size=16, # size in which chunks are entered into the network, on how many data parallel weights are trained
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        evaluation_strategy="epoch", # save model per epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False #,
        #label_names=["level0"],
    )

    # IMPORTANT: Set: Model, dataset, ... , define learning process, metrics, ...

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_training_data,
        eval_dataset=tokenized_testing_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics  
    )

    #checkpointing
    #use cuda
    trainer.train()

    trainer.save_model(models_dir+"model_"+level)

    classifier = pipeline("text-classification", model=models_dir+"model_"+level, device=0) # method pipeline -> sting for textclassificaton, folder, device (graphics card)
    results = [classifier(text,truncation=True) for text in tqdm(dataset.text.to_list())] # listcomprehension over all texts, tokenization in model, truncation -> padding too long texts

    results = [tmp[0] for tmp in results]
    pd.DataFrame(results).to_pickle(results_dir+"results_"+level+".pkl")  # convert as dataframe, pick, safe

    return dataset, results, tokenized_testing_data
