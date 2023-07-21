import numpy as np
import pandas as pd

emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

def clean_df(df):
    '''
    Returns the labeled data with only the emotion columns and ids.
    Not in this dataframe: column "other", any comments with multiple labels

    Parameters: 
    df (pandas.Dataframe): The input data
    
    Returns:
    clean_df (pandas.Dataframe): The preprocessed data
    '''

    # Drop all classified as "other"
    clean_df = df.drop(df[df['example_very_unclear'] == True].index)
    
    # Drop unnecessary columns
    clean_df = clean_df.drop(['link_id', 'parent_id', 'created_utc', 'example_very_unclear'], axis=1)

    # Drop all multi-label observations (more than one emotion coded) 
    clean_df['num_labels'] = clean_df.loc[:, emotions].sum(axis=1)  # new column with count how many labels where assigned
    clean_df = clean_df.drop(clean_df[clean_df['num_labels'] != 1].index)  # filter according to num_labels
    clean_df = clean_df.drop(['num_labels'], axis=1)  # drop this helper column again
    
    return clean_df


def map_plutchik(emotion):
    '''
    Returns Plutchik emotion adjective for any emotion.
    This is based on our own understanding of Plutchik's emotions.
    
    Parameters: 
    emotion (string): input emotion
    
    Returns:
    new_emotion (string): corresponding Plutchik emotion
    '''
    if emotion in ["amusement", "excitement", "joy"]:
        return "begeistert"
    elif emotion in ["love", "desire"]:
        return "verliebt"
    elif emotion in ["admiration", "caring", "approval"]:
        return "bewundernd"
    elif emotion in ["gratitude"]:
        return "ehrfürchtig"
    elif emotion in ["fear", "nervousness"]:
        return "erschrocken"
    elif emotion in ["embarassment"]:
        return "ehrfürchtig"
    elif emotion in ["surprise", "confusion"]:
        return "erstaunt"
    elif emotion in ["disappointment"]:
        return "enttäuscht"
    elif emotion in ["grief", "sadness"]:
        return "betrübt"
    elif emotion in ["remorse"]:
        return "bereuend"
    elif emotion in ["disgust", "disapproval"]:
        return "angewidert"
    elif emotion in []:
        return "hassend"
    elif emotion in ["anger", "annoyance"]:
        return "wütend"
    elif emotion in ["pride"]:
        return "streitlustig"
    elif emotion in ["realization", "curiosity"]:
        return "klar"
    elif emotion in ["optimism", "relief"]:
        return "optimistisch"
    else:
        return emotion # only "neutral"


def map_level1(emotion):
    '''
    Returns level1 clustering label for any emotion ("level0"). 
    The clustering is based on the sentiment analysis by Demszky et al. (2020)
    
    Parameters: 
    emotion (string): input emotion
    
    Returns:
    new_emotion (string): corresponding level1 emotion cluster
    '''
    if emotion in ["excitement", "joy"]:
        return "exc_joy"
    elif emotion in ["desire", "optimism"]:
        return "des_opt"
    elif emotion in ["pride", "admiration"]:
        return "pri_adm"
    elif emotion in ["gratitude", "relief"]:
        return "gra_rel"
    elif emotion in ["approval", "realization"]:
        return "app_rea"
    elif emotion in ["curiosity", "confusion"]:
        return "cur_con"
    elif emotion in ["fear", "nervousness"]:
        return "fea_ner"
    elif emotion in ["remorse", "embarassment"]:
        return "rem_emb"
    elif emotion in ["disappointment", "sadness"]:
        return "dis_sad"
    elif emotion in ["anger", "annoyance"]:
        return "ang_ann"
    else:
        return emotion


def map_level2(emotion):
    '''
    Returns level2 clustering label for any emotion ("level0"). 
    The clustering is based on the sentiment analysis by Demszky et al. (2020)

    Parameters: 
    emotion (string): input emotion
    
    Returns:
    new_emotion (string): corresponding level2 emotion cluster
    '''
    if emotion in ["excitement", "joy", "love"]:
        return "exc_joy_lov"
    elif emotion in ["desire", "optimism", "caring"]:
        return "des_opt_car"
    elif emotion in ["pride", "admiration", "gratitude", "relief"]:
        return "pri_adm_gra_rel"
    elif emotion in ["approval", "realization"]:
        return "app_rea"
    elif emotion in ["surprise", "curiosity", "confusion"]:
        return "sur_cur_con"
    elif emotion in ["fear", "nervousness"]:
        return "fea_ner"
    elif emotion in ["remorse", "embarassment"]:
        return "rem_emb"
    elif emotion in ["disappointment", "sadness", "grief"]:
        return "dis_sad_gri"
    elif emotion in ["disgust", "anger", "annoyance"]:
        return "dis_ang_ann"
    else:
        return emotion



def map_level3(emotion):
    '''
    Returns level3 clustering label for any emotion ("level0"). 
    The clustering is based on the sentiment analysis by Demszky et al. (2020)

    Parameters: 
    emotion (string): input emotion
    
    Returns:
    new_emotion (string): corresponding level3 emotion cluster
    '''
    if emotion in ["amusement", "excitement", "joy", "love"]:
        return "amu_exc_joy_lov"
    elif emotion in ["desire", "optimism", "caring"]:
        return "des_opt_car"
    elif emotion in ["pride", "admiration", "gratitude", "relief", "approval", "realization"]:
        return "pri_adm_gra_rel_app_rea"
    elif emotion in ["surprise", "curiosity", "confusion"]:
        return "sur_cur_con"
    elif emotion in ["fear", "nervousness"]:
        return "fea_ner"
    elif emotion in ["remorse", "embarassment", "disappointment", "sadness", "grief"]:
        return "rem_emb_dis_sad_gri"
    elif emotion in ["disgust", "anger", "annoyance", "disapproval"]:
        return "dis_ang_ann_dis"
    else:
        return emotion # it remains only "neutral"


def create_clustered_df(df):
    '''
    Pivots the emotion columns into long format, so there is just one "level0" column. 
    Adds "level1", "level2" and "level3" emotion columns --> Hierachical clustering of the original emotions.

    Paramaters:
    df (pandas.Dataframe): The input data to transform
    
    Returns:
    plutchik_df (pandas.Dataframe): The transformed data frame with emotions clustered on different levels
    '''
    return clustered_df


def create_plutchik_df(df):
    '''
    Pivots the emotion columns into long format, so there is just one "level0" column. Adds another "plutchik" emotion column.

    Paramaters:
    df (pandas.Dataframe): The input data to transform
    
    Returns:
    plutchik_df (pandas.Dataframe): The data frame mapped onto plutchik emotions
    '''
    
    helper1 = df.copy()
    helper2 = df.copy()
    plutchik_df = df.copy()

    # First we write the 'level0' column, which shows the original emotion coded as a string value. 
    # Mask() is applied on the whole helper dataframe, which is a copy of the original df.
    for emotion in emotions:
        helper1 = helper1.mask(helper1[emotion] == 1, emotion)
        # Now copy the level0 column on the output df:
        plutchik_df['level0'] = helper1['text'] # it doesn't matter, which column, as it mapped the emotions on every column

    # Now we do the same thing for plutchi emotion, using the mapfunction defined above.
    for emotion in emotions:
        helper2 = helper2.mask(helper2[emotion] == 1, map_plutchik(emotion))
        # Now copy the plutchik column on the output df:
        plutchik_df['plutchik'] = helper2['text'] # it doesn't matter, which column, as it mapped the emotions on every column

    # Now we drop all the original emotion columns:
    plutchik_df = plutchik_df.drop(emotions, axis = 1)
    return plutchik_df

