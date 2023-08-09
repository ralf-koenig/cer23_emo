import pandas as pd
import preprocessing.preprocessors
import pytest


@pytest.fixture(scope="module")
def df():
    return pd.read_csv("../data/GoEmotions.csv")


def test_clustered_df(df):
    cleaned_df = preprocessing.preprocessors.clean_df(df)
    clustered_df = preprocessing.preprocessors.create_clustered_df(cleaned_df)
    # check that columns exists
    assert ('level0' in clustered_df)
    assert ('level1' in clustered_df)
    assert ('level2' in clustered_df)
    assert ('level3' in clustered_df)
    assert ('plutchik' in clustered_df)
    # check that one-hot encoded emotion columns are removed
    assert (not set(preprocessing.preprocessors.emotions).issubset(clustered_df.columns))
