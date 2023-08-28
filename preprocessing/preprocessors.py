import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
        return emotion  # amusement, love, caring, surprise, grief, disgust, disapproval, neutral


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

    Parameters:
    df (pandas.Dataframe): The input data to transform. Needs to be ONE-hot encoded on the emotions
    columns, so that this step is lossless and predictable for Multi-class classification

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
    Adds "level1", "level2", and "level3" emotion columns through hierarchical clustering of the original emotions
    according to the GoEmotions Paper.

    Also adds a "Plutchik" emotion column with a mapping of 27+1 emotions to Plutchik emotions.

    Parameters:
    clustered_df (pandas.Dataframe): DataFrame containing emotions and other data

    Returns: 
    clustered_df (pandas.Dataframe): DataFrame with added hierarchical and Plutchik emotion columns
    """
    clustered_df['level1'] = clustered_df.level0.apply(map_level1)
    clustered_df['level2'] = clustered_df.level0.apply(map_level2)
    clustered_df['level3'] = clustered_df.level0.apply(map_level3)
    clustered_df['plutchik'] = clustered_df.level0.apply(map_plutchik)
    return clustered_df


def majority_voted_df(df):
    """
    Performs majority voting on emotions for each set of rows in the dataframe with the same id.
    Keeps only ids which have a clear majority vote result, i.e. a single most common emotion.

    Parameters: 
    df (pandas.DataFrame): DataFrame containing emotions and other data

    Returns: 
    DataFrame with majority voted emotions
    """
    result_list = []
    # only keep columns: id and level0, this suffices for the majority vote
    df_reduced = df[['id', 'level0']]
    # group by id, list the level0 emotions
    df_groups_ser = df_reduced.groupby('id')['level0'].apply(list)
    # now do majority vote using another function
    for text_id, emotions_list in df_groups_ser.items():
        action, item1 = majority_vote(emotions_list)
        if action == 'keep':
            result_list.append([text_id, item1])
    return pd.DataFrame(result_list, columns=['id', 'level0'])


def majority_vote(emotion_list: list):
    """
    Performs strict majority voting on a list of emotions.
    Keeps an item if it overrules the others; deletes if two most frequent items are equally frequent.

    Parameters:
    emotion_list (list): List of emotions to perform majority voting on

    Returns: 
    Tuple containing action ("keep" or "delete") and the majority voted item
    """
    action = 'keep'
    # find most frequent entry and its frequency
    item1 = max(set(emotion_list), key=emotion_list.count)
    freq1 = emotion_list.count(item1)
    # only keep the rest
    emotion_list = list(filter(lambda a: a != item1, emotion_list))
    # if something remains
    if emotion_list:
        # find 2nd most frequent entry and its frequency
        item2 = max(set(emotion_list), key=emotion_list.count)
        freq2 = emotion_list.count(item2)
        # if they match in frequency: they rule other out => delete
        if freq1 == freq2:
            action = "delete"
    return action, item1


def back_translate(text, src_tokenizer, src_model, tgt_tokenizer, tgt_model):
    """
    Translates source text to the target language using machine translation and then translates it 
    back to the source language to create a back-translated version of the text.

    Parameters:
    text (string): Source text to be back-translated
    src_tokenizer: Tokenizer for the source language
    src_model: Machine translation model for the source language
    tgt_tokenizer: Tokenizer for the target language
    tgt_model: Machine translation model for the target language

    Returns: 
    backtranslated version of the input text
    """

    # Translate source text to the target language
    src_input = src_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    tgt_translation = src_model.generate(**src_input)
    tgt_translation_text = tgt_tokenizer.decode(tgt_translation[0], skip_special_tokens=True)

    # Translate target translation back to the source language
    tgt_input = tgt_tokenizer(tgt_translation_text, return_tensors="pt", padding=True, truncation=True)
    src_back_translation = tgt_model.generate(**tgt_input)
    src_back_translation_text = src_tokenizer.decode(src_back_translation[0], skip_special_tokens=True)

    return src_back_translation_text  # , tgt_translation_text


def back_translate_emo(df, language, src_model_name, tgt_model_name):
    """
    Performs backtranslation on the 'text' column of the input DataFrame using given translation models.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'text' column to be back-translated
    language (string): Short language code for the target language
    src_model_name (string): Pre-trained translation model for the source language
    tgt_model_name (string): Pre-trained translation model for the target language

    Returns: 
    DataFrame with back-translated 'text' column and modified 'id' column
    """

    src_tokenizer = AutoTokenizer.from_pretrained(src_model_name)
    src_model = AutoModelForSeq2SeqLM.from_pretrained(src_model_name)

    tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_model_name)
    tgt_model = AutoModelForSeq2SeqLM.from_pretrained(tgt_model_name)

    # Apply back translation to the 'text' column
    df['text'] = df['text'].apply(lambda x: back_translate(x, src_tokenizer, src_model, tgt_tokenizer, tgt_model))

    # remove ▁ from subword tokenization
    df['text'] = df['text'].str.replace("▁", " ")

    # Add "_fr" to the id column for back translated rows
    df['id'] = df['id'] + language

    return df


def back_translated_df(df):
    """
    Creates a DataFrame with back-translated data by applying back translation to different emotions and languages.

    Parameters:
    df (pandas.DataFrame): DataFrame containing original data for backtranslation

    Returns: 
    result_df (pandas.DataFrame): DataFrame with back-translated data
    """

    # two pre-trained translation models: source language and target language
    src_model_name = ["Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-en-es",
                      "Helsinki-NLP/opus-mt-en-da", "Helsinki-NLP/opus-mt-en-sv", "Helsinki-NLP/opus-mt-en-ru",
                      "Helsinki-NLP/opus-mt-en-id", "Helsinki-NLP/opus-mt-en-nl", "Helsinki-NLP/opus-mt-en-cs"]
    tgt_model_name = ["Helsinki-NLP/opus-mt-fr-en", "Helsinki-NLP/opus-mt-de-en", "Helsinki-NLP/opus-mt-es-en",
                      "Helsinki-NLP/opus-mt-da-en", "Helsinki-NLP/opus-mt-sv-en", "Helsinki-NLP/opus-mt-ru-en",
                      "Helsinki-NLP/opus-mt-id-en", "Helsinki-NLP/opus-mt-nl-en", "Helsinki-NLP/opus-mt-cs-en"]
    language_short = ["_fr", "_de", "_es", "_da", "_sv", "_ru", "_id", "_nl", "_cs"]

    # Create back translation and concatenate DataFrames
    embarrassment_fr = back_translate_emo(df[df['level0'] == 'embarrassment'], language_short[0], src_model_name[0],
                                          tgt_model_name[0])
    result_df = pd.concat([df, embarrassment_fr], ignore_index=True)
    relief_fr = back_translate_emo(df[df['level0'] == 'relief'], language_short[0], src_model_name[0],
                                   tgt_model_name[0])
    result_df = pd.concat([result_df, relief_fr], ignore_index=True)
    relief_de = back_translate_emo(df[df['level0'] == 'relief'], language_short[1], src_model_name[1],
                                   tgt_model_name[1])
    result_df = pd.concat([result_df, relief_de], ignore_index=True)
    relief_es = back_translate_emo(df[df['level0'] == 'relief'], language_short[2], src_model_name[2],
                                   tgt_model_name[2])
    result_df = pd.concat([result_df, relief_es], ignore_index=True)
    nervousness_fr = back_translate_emo(df[df['level0'] == 'nervousness'], language_short[0], src_model_name[0],
                                        tgt_model_name[0])
    result_df = pd.concat([result_df, nervousness_fr], ignore_index=True)
    nervousness_de = back_translate_emo(df[df['level0'] == 'nervousness'], language_short[1], src_model_name[1],
                                        tgt_model_name[1])
    result_df = pd.concat([result_df, nervousness_de], ignore_index=True)
    nervousness_es = back_translate_emo(df[df['level0'] == 'nervousness'], language_short[2], src_model_name[2],
                                        tgt_model_name[2])
    result_df = pd.concat([result_df, nervousness_es], ignore_index=True)
    pride_fr = back_translate_emo(df[df['level0'] == 'pride'], language_short[0], src_model_name[0], tgt_model_name[0])
    result_df = pd.concat([result_df, pride_fr], ignore_index=True)
    pride_de = back_translate_emo(df[df['level0'] == 'pride'], language_short[1], src_model_name[1], tgt_model_name[1])
    result_df = pd.concat([result_df, pride_de], ignore_index=True)
    pride_es = back_translate_emo(df[df['level0'] == 'pride'], language_short[2], src_model_name[2], tgt_model_name[2])
    result_df = pd.concat([result_df, pride_es], ignore_index=True)
    pride_da = back_translate_emo(df[df['level0'] == 'pride'], language_short[3], src_model_name[3], tgt_model_name[3])
    result_df = pd.concat([result_df, pride_da], ignore_index=True)
    pride_sv = back_translate_emo(df[df['level0'] == 'pride'], language_short[4], src_model_name[4], tgt_model_name[4])
    result_df = pd.concat([result_df, pride_sv], ignore_index=True)
    grief_fr = back_translate_emo(df[df['level0'] == 'grief'], language_short[0], src_model_name[0], tgt_model_name[0])
    result_df = pd.concat([result_df, grief_fr], ignore_index=True)
    grief_de = back_translate_emo(df[df['level0'] == 'grief'], language_short[1], src_model_name[1], tgt_model_name[1])
    result_df = pd.concat([result_df, grief_de], ignore_index=True)
    grief_es = back_translate_emo(df[df['level0'] == 'grief'], language_short[2], src_model_name[2], tgt_model_name[2])
    result_df = pd.concat([result_df, grief_es], ignore_index=True)
    grief_da = back_translate_emo(df[df['level0'] == 'grief'], language_short[3], src_model_name[3], tgt_model_name[3])
    result_df = pd.concat([result_df, grief_da], ignore_index=True)
    grief_sv = back_translate_emo(df[df['level0'] == 'grief'], language_short[4], src_model_name[4], tgt_model_name[4])
    result_df = pd.concat([result_df, grief_sv], ignore_index=True)
    grief_ru = back_translate_emo(df[df['level0'] == 'grief'], language_short[5], src_model_name[5], tgt_model_name[5])
    result_df = pd.concat([result_df, grief_ru], ignore_index=True)
    grief_id = back_translate_emo(df[df['level0'] == 'grief'], language_short[6], src_model_name[6], tgt_model_name[6])
    result_df = pd.concat([result_df, grief_id], ignore_index=True)
    grief_nl = back_translate_emo(df[df['level0'] == 'grief'], language_short[7], src_model_name[7], tgt_model_name[7])
    result_df = pd.concat([result_df, grief_nl], ignore_index=True)
    grief_cs = back_translate_emo(df[df['level0'] == 'grief'], language_short[8], src_model_name[8], tgt_model_name[8])
    result_df = pd.concat([result_df, grief_cs], ignore_index=True)

    result_df.to_csv('../data/back_translated_df.csv', index=False)  # save dataframe as csv

    return result_df
