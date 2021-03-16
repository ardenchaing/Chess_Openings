"""
Takes in a dataset and cleans it so it only
pertains relevant rows and columns.
"""
import pandas as pd


def processing(filepath, skill=1200):
    """
    Returns two Dataframes. One with all the chess games from the
    filepath. The other with all the chess games that fit specific
    constraints to eliminate underskilled players and casual games.
    Input: filepath to chess dataset (CSV)
    """
    df = pd.read_csv(filepath)
    constraint = ['rated', 'turns', 'victory_status', 'winner',
                  'white_rating', 'black_rating', 'moves',
                  'opening_name']
    data = df.loc[:, constraint]
    data = data[(data['rated']) & (data['white_rating'] >= skill) &
                (data['black_rating'] >= skill)]
    return (df, data)
