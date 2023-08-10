# import numpy as np
# import pandas as pd
import pandas as pd

# 27 + 1 emotions from paper on "GoEmotions"
emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
            'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise', 'neutral']


def clean_df(df):
    """
    Returns the labeled data with only the emotion columns and ids.
    Not in this dataframe: column "other", any comments with multiple labels

    Parameters:
    df (pandas.Dataframe): The input data

    Returns:
    cleaned_df (pandas.Dataframe): The preprocessed data
    """

    # Drop all ratings classified as "very unclear"
    cleaned_df = df.drop(df[df['example_very_unclear'] == True].index)

    # Drop unnecessary columns
    cleaned_df = cleaned_df.drop(['link_id', 'parent_id', 'created_utc', 'example_very_unclear'], axis=1)

    # Drop all multi-label observations (more than one emotion coded) as we want to do multi-class
    # classification and not multi-label classification
    cleaned_df['num_labels'] = cleaned_df.loc[:, emotions].sum(axis=1)  # count how many labels where assigned
    cleaned_df = cleaned_df.drop(cleaned_df[cleaned_df['num_labels'] != 1].index)  # filter according to num_labels
    cleaned_df = cleaned_df.drop(['num_labels'], axis=1)  # drop this helper column again

    return cleaned_df


def map_plutchik(emotion):
    """
    Returns Plutchik emotion adjective for any emotion.
    This is based on our own understanding of emotions as classified by Plutchik.
    Refer to https://en.wikipedia.org/wiki/Emotion_classification#Lists_of_emotions

    Parameters:
    emotion (string): input emotion

    Returns:
    new_emotion (string): corresponding Plutchik emotion
    """
    if emotion in ["amusement", "excitement", "joy"]:
        return "ecstasy"
    elif emotion in ["love", "desire"]:
        return "love"
    elif emotion in ["admiration", "caring", "approval"]:
        return "admiration"
    elif emotion in ["gratitude", "embarrassment"]:
        return "awe"
    elif emotion in ["fear", "nervousness"]:
        return "terror"
    elif emotion in ["surprise", "confusion"]:
        return "amazement"
    elif emotion in ["disappointment"]:
        return "disapproval"
    elif emotion in ["grief", "sadness"]:
        return "grief"
    elif emotion in ["remorse"]:
        return "remorse"
    elif emotion in ["disgust", "disapproval"]:
        return "loathing"
    elif emotion in []:
        return "contempt"
    elif emotion in ["anger", "annoyance"]:
        return "rage"
    elif emotion in ["pride"]:
        return "aggressiveness"
    elif emotion in ["realization", "curiosity"]:
        return "vigilance"
    elif emotion in ["optimism", "relief"]:
        return "optimism"
    else:
        return emotion  # only "neutral"


def map_level1(emotion):
    """
    Returns level1 clustering label for any emotion ("level0").
    The clustering is based on the sentiment analysis by Demszky et al. (2020)

    Parameters:
    emotion (string): input emotion

    Returns:
    new_emotion (string): corresponding level 1 emotion cluster
    """
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
    elif emotion in ["remorse", "embarrassment"]:
        return "rem_emb"
    elif emotion in ["disappointment", "sadness"]:
        return "dis_sad"
    elif emotion in ["anger", "annoyance"]:
        return "ang_ann"
    else:
        return emotion  # amusement, caring, disapproval, disgust, grief, love, surprise, neutral


def map_level2(emotion):
    """
    Returns level2 clustering label for any emotion ("level0").
    The clustering is based on the sentiment analysis by Demszky et al. (2020)

    Parameters:
    emotion (string): input emotion

    Returns:
    new_emotion (string): corresponding level 2 emotion cluster
    """
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
    elif emotion in ["remorse", "embarrassment"]:
        return "rem_emb"
    elif emotion in ["disappointment", "sadness", "grief"]:
        return "dis_sad_gri"
    elif emotion in ["disgust", "anger", "annoyance"]:
        return "dis_ang_ann"
    else:
        return emotion  # amusement, disapproval, neutral


def map_level3(emotion):
    """
    Returns level3 clustering label for any emotion ("level0").
    The clustering is based on the sentiment analysis by Demszky et al. (2020)

    Parameters:
    emotion (string): input emotion

    Returns:
    new_emotion (string): corresponding level 3 emotion cluster
    """
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
    elif emotion in ["remorse", "embarrassment", "disappointment", "sadness", "grief"]:
        return "rem_emb_dis_sad_gri"
    elif emotion in ["disgust", "anger", "annoyance", "disapproval"]:
        return "dis_ang_ann_dis"
    else:
        return emotion  # only "neutral" remains


def create_pivoted_df(df):
    """
    Pivots the emotion columns into long format, so there is just one "level0" column.
    Adds "level1", "level2" and "level3" emotion columns --> hierarchical clustering of the original emotions.
    according to GoEmotions Paper

    Adds "Plutchik" emotion column with a mapping of 27+1 emotions to Plutchik emotions

    Parameters:
    df (pandas.Dataframe): The input data to transform

    Returns:
    clustered_df (pandas.Dataframe): The transformed data frame with emotions clustered on different levels
    """
    pivoted_df = df.copy()
    pivoted_df['level0'] = pivoted_df[emotions].idxmax(axis=1)
    # Now we drop all the original emotion columns as we do not need them any longer:
    pivoted_df = pivoted_df.drop(emotions, axis=1)
    return pivoted_df


def add_hierarchical_levels(clustered_df):
    """
    Add the hierarchical columns

    :param clustered_df:
    :return:
    """
    clustered_df['level1'] = clustered_df.level0.apply(map_level1)
    clustered_df['level2'] = clustered_df.level0.apply(map_level2)
    clustered_df['level3'] = clustered_df.level0.apply(map_level3)
    clustered_df['plutchik'] = clustered_df.level0.apply(map_plutchik)
    return clustered_df


def majority_voted_df(df):
    """
    Do a majority vote on the emotions for each set of rows in the dataframe with the same id.
    Keep only ids which have a clear majority vote result i.e. ONE most common emotion.
    :param df:
    :return:
    """
    result_list = []
    df_reduced = df[['id', 'level0']]
    df_groups_ser = df_reduced.groupby('id')['level0'].apply(list)
    for text_id, emotions_list in df_groups_ser.items():
        action, item1 = majority_vote(emotions_list)
        if action == 'keep':
            result_list.append([text_id, item1])
    return pd.DataFrame(result_list, columns=['id', 'level0'])


def majority_vote(emotion_list: list):
    """
    Do a strict majority voting on a list of emotions.
    :param emotion_list:
    :return:
    """
    action = 'keep'
    item1 = max(set(emotion_list), key=emotion_list.count)
    freq1 = emotion_list.count(item1)
    emotion_list = list(filter(lambda a: a != item1, emotion_list))
    if emotion_list:
        item2 = max(set(emotion_list), key=emotion_list.count)
        freq2 = emotion_list.count(item2)
        if freq1 == freq2:
            action = "delete"
    return action, item1
