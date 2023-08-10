import pandas as pd
import preprocessing.preprocessors
import pytest


@pytest.fixture(scope="module")
def cleaned_df():
    df = pd.read_csv("../data/GoEmotions.csv")
    cleaned_df = preprocessing.preprocessors.clean_df(df)
    return cleaned_df


def test_pivoted_df(cleaned_df):
    pivoted_df = preprocessing.preprocessors.create_pivoted_df(cleaned_df)
    # check that one-hot encoded emotion columns are removed
    assert (not set(preprocessing.preprocessors.emotions).issubset(pivoted_df.columns))


def test_add_hierarchical_levels(cleaned_df):
    pivoted_df = preprocessing.preprocessors.create_pivoted_df(cleaned_df)
    clustered_df = preprocessing.preprocessors.add_hierarchical_levels(pivoted_df)
    # check that columns exists
    assert ('level0' in clustered_df.columns)
    assert ('level1' in clustered_df.columns)
    assert ('level2' in clustered_df.columns)
    assert ('level3' in clustered_df.columns)
    assert ('plutchik' in clustered_df.columns)


def test_majority_vote():
    action, item1 = preprocessing.preprocessors.majority_vote(['anger', 'anger', 'disapproval', 'disapproval'])
    assert (action == 'delete')
    action, item1 = preprocessing.preprocessors.majority_vote(['anger', 'anger', 'disapproval'])
    assert (action == 'keep')
    assert (item1 == 'anger')
    action, item1 = preprocessing.preprocessors.majority_vote(['anger', 'anger'])
    assert (action == 'keep')
    assert (item1 == 'anger')
    action, item1 = preprocessing.preprocessors.majority_vote(['anger', 'disapproval'])
    assert (action == 'delete')
    action, item1 = preprocessing.preprocessors.majority_vote(['anger'])
    assert (action == 'keep')
    assert (item1 == 'anger')


def test_majority_voted_df(cleaned_df):
    pivoted_df = preprocessing.preprocessors.create_pivoted_df(cleaned_df)
    maj_voted_df = preprocessing.preprocessors.majority_voted_df(pivoted_df)
    classCounts = maj_voted_df.level0.value_counts()
    print(classCounts)
    # check that columns exists
    assert ('id' in maj_voted_df.columns)
    assert ('level0' in maj_voted_df.columns)
    assert (len(maj_voted_df) == 43379)
