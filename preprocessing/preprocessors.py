def clean_df():
    '''
    Returns the labeled data with only the emotion columns and ids.
    Not in this dataframe: column "other", any comments with multiple labels

    Returns:
    df (pandas.Dataframe): The preprocessed data
    '''
    
    return "This was helper1"


def map_plutchik(emotion):
    '''
    Returns Plutchik emotion adjective for any emotion.
    This is based on our own understanding of Plutchik's emotions.
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
    Returns xy
    '''
    return clustered_df


def create_plutchik_df(df):
    '''
    Returns xy
    '''
    return plutchik_df

