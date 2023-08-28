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


def get_bert(dataset, level, bert, models_dir, results_dir):
    '''
    Returns the results for a chosen BERT-model.
    
    Parameters: 
    dataset (dataframe): Input data containing the used text and emotion labels
    level (string): cluster level of emotions that should be analyzed
    bert (string): chosen BERT-model for training
    models_dir (string): directory where the models are saved
    results_dir (string): directory where the results are saved
    
    Returns:
    dataset (dataframe): input data with mapped emotion labels
    results (list): Model classification results for each text
    tokenized_testing_data (Dataset): tokenized text data in tensor form
    '''
    
    if level == "level0":
        id2label = {0: 'sadness', 1: 'neutral', 2: 'love', 3: 'gratitude', 4: 'disapproval',
                    5: 'amusement', 6: 'disappointment', 7: 'realization', 8: 'admiration', 9:
                    'annoyance', 10: 'confusion', 11: 'optimism', 12: 'excitement', 13: 'caring',
                    14: 'remorse', 15: 'joy', 16: 'approval', 17: 'embarrassment', 18: 'surprise',
                    19: 'curiosity', 20: 'anger', 21: 'grief', 22: 'disgust', 23: 'pride', 24: 'desire',
                    25: 'relief', 26: 'fear', 27: 'nervousness'}
        label2id = {value: key for key, value in id2label.items()}
    elif level == "level1":
        id2label = {0: 'dis_sad', 1: 'neutral', 2: 'love', 3: 'gra_rel', 4: 'disapproval',
                    5: 'amusement', 6: 'app_rea', 7: 'pri_adm', 8: 'ang_ann', 9: 'cur_con', 10: 'des_opt',
                    11: 'exc_joy', 12: 'caring', 13: 'rem_emb', 14: 'surprise',
                    15: 'grief', 16: 'disgust', 17: 'fea_ner'}
        label2id = {value: key for key, value in id2label.items()}
    elif level == "level2":
        id2label = {0: 'dis_sad_gri', 1:'neutral', 2:'exc_joy_lov', 3:'pri_adm_gra_rel',
                    4:'disapproval', 5:'amusement', 6:'app_rea', 7:'dis_ang_ann',
                    8:'sur_cur_con', 9:'des_opt_car', 10:'rem_emb',
                    11:'fea_ner'}
        label2id = {value: key for key, value in id2label.items()}
    elif level == "level3":
        id2label = {0: 'rem_emb_dis_sad_gri', 1:'neutral', 2:'amu_exc_joy_lov',
                    3: 'pri_adm_gra_rel_app_rea', 4: 'dis_ang_ann_dis', 5: 'sur_cur_con',
                    6: 'des_opt_car', 7: 'fea_ner'}
        label2id = {value: key for key, value in id2label.items()}
    elif level == "plutchik":
        id2label = {0: 'grief', 1:'neutral', 2:'love', 3:'awe', 4:'loathing', 5:'ecstasy',
                    6:'disapproval', 7:'vigilance', 8:'admiration', 9:'rage', 10:'amazement',
                    11:'optimism', 12:'remorse', 13:'aggressiveness', 14:'terror'}
        label2id = {value: key for key, value in id2label.items()}
    else:
        print("false level-input")       

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
        per_device_train_batch_size=12, # size in which chunks are entered into the network, on how many data parallel weights are trained
        per_device_eval_batch_size=12,
        num_train_epochs=3,
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

    return dataset, results, tokenized_testing_data, testing_data, label2id


def load_bert(dataset, level, bert, models_dir, results_dir):
    """
    Loads a pre-trained BERT model, applies it to the provided dataset, and returns the classification results.

    Parameters:
    dataset (dataframe): Input data containing text and emotion labels
    level (string): Cluster level of emotions that should be analyzed
    bert (string): Chosen BERT-model for inference
    models_dir (string): Directory where the trained models are saved
    results_dir (string): Directory where the model results are saved

    Returns:
    dataset (dataframe): Input data with mapped emotion labels
    results (list): Model classification results for each text
    tokenized_testing_data (Dataset): Tokenized text data in tensor form
    """

    # Replace 'models_dir' and 'level' with your actual paths and level
    model_path = models_dir + "model_" + level

    # Load the saved model
    loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    if level == "level0":
        id2label = {0: 'sadness', 1: 'neutral', 2: 'love', 3: 'gratitude', 4: 'disapproval',
                    5: 'amusement', 6: 'disappointment', 7: 'realization', 8: 'admiration', 9:
                    'annoyance', 10: 'confusion', 11: 'optimism', 12: 'excitement', 13: 'caring',
                    14: 'remorse', 15: 'joy', 16: 'approval', 17: 'embarrassment', 18: 'surprise',
                    19: 'curiosity', 20: 'anger', 21: 'grief', 22: 'disgust', 23: 'pride', 24: 'desire',
                    25: 'relief', 26: 'fear', 27: 'nervousness'}
        label2id = {value: key for key, value in id2label.items()}
    elif level == "level1":
        id2label = {0: 'dis_sad', 1: 'neutral', 2: 'love', 3: 'gra_rel', 4: 'disapproval',
                    5: 'amusement', 6: 'app_rea', 7: 'pri_adm', 8: 'ang_ann', 9: 'cur_con', 10: 'des_opt',
                    11: 'exc_joy', 12: 'caring', 13: 'rem_emb', 14: 'surprise',
                    15: 'grief', 16: 'disgust', 17: 'fea_ner'}
        label2id = {value: key for key, value in id2label.items()}
    elif level == "level2":
        id2label = {0: 'dis_sad_gri', 1:'neutral', 2:'exc_joy_lov', 3:'pri_adm_gra_rel',
                    4:'disapproval', 5:'amusement', 6:'app_rea', 7:'dis_ang_ann',
                    8:'sur_cur_con', 9:'des_opt_car', 10:'rem_emb',
                    11:'fea_ner'}
        label2id = {value: key for key, value in id2label.items()}
    elif level == "level3":
        id2label = {0: 'rem_emb_dis_sad_gri', 1:'neutral', 2:'amu_exc_joy_lov',
                    3: 'pri_adm_gra_rel_app_rea', 4: 'dis_ang_ann_dis', 5: 'sur_cur_con',
                    6: 'des_opt_car', 7: 'fea_ner'}
        label2id = {value: key for key, value in id2label.items()}
    elif level == "plutchik":
        id2label = {0: 'grief', 1:'neutral', 2:'love', 3:'awe', 4:'loathing', 5:'ecstasy',
                    6:'disapproval', 7:'vigilance', 8:'admiration', 9:'rage', 10:'amazement',
                    11:'optimism', 12:'remorse', 13:'aggressiveness', 14:'terror'}
        label2id = {value: key for key, value in id2label.items()}
    else:
        print("false level-input") 

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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Padding -> map all tensors to the same size
    accuracy = evaluate.load("accuracy") # define evaluation method -> quality

    def compute_metrics(eval_pred): # function calculation metric
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    classifier = pipeline("text-classification", model=models_dir+"model_" + level, device=0) # method pipeline -> sting for textclassificaton, folder, device (graphics card)
    results = [classifier(text,truncation=True) for text in tqdm(dataset.text.to_list())] # listcomprehension over all texts, tokenization in model, truncation -> padding too long texts

    results = [tmp[0] for tmp in results]
    pd.DataFrame(results).to_pickle(results_dir+"results_"+level+".pkl")  # convert as dataframe, pick, safe
    
    return dataset, results, tokenized_testing_data, testing_data, label2id