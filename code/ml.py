"""
Constructs and trains a machine learning model
on the featured data.
"""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def opening_model(cleaned):
    """
    Returns the trained DecisionTreeClassifier model, the values for
    hyperparameters (max_depth, min_samples_split, min_samples_leaf),
    and the testing accuracy of the trained model. Cleans, splits,
    and trains the dataset into a DecisionTreeClassifier model using
    optimal hyperparameters. Also calculates testing accuracy.
    Input: a dataframe of chess games.
    """
    cleaned = cleaned.reset_index()
    cleaned['White rank'] = simplify_elo_ratings(cleaned['white_rating'])
    cleaned['Black rank'] = simplify_elo_ratings(cleaned['black_rating'])
    cleaned = cleaned.drop(columns=['index', 'rated', 'victory_status',
                                    'turns', 'moves', 'white_rating',
                                    'black_rating'])
    features = cleaned.loc[:, cleaned.columns != 'opening_name']
    features = pd.get_dummies(features)
    labels = simplify_opening_names(cleaned['opening_name'])
    features_model, features_test, labels_model, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    features_train, features_dev, labels_train, labels_dev = \
        train_test_split(features_model, labels_model, test_size=0.25)
    hyper_name, hyper_value = optimal_parameter(features_train, labels_train,
                                                features_dev, labels_dev)
    depth, depth_num, split, split_num, leaf, leaf_num = hyper_name
    model = DecisionTreeClassifier()
    model = DecisionTreeClassifier(max_depth=depth_num,
                                   min_samples_split=split_num,
                                   min_samples_leaf=leaf_num)
    model = model.fit(features_train, labels_train)
    test_predictions = model.predict(features_test)
    test_acc = accuracy_score(labels_test, test_predictions)
    return model, depth_num, split_num, leaf_num, test_acc


def optimal_parameter(features_train, labels_train, features_dev, labels_dev):
    """
    Returns a string of the best hyperparameters to utilize for a DTC and its
    corresponding value. Calculates the optimal hyperparameters with
    max_depth, min_samples_split, and min_samples_leaf. Takes about 8 mins.
    Input: Two dataframes and two series consisting of the features and labels
    for the training and development datasets.
    """
    hyperparameters = {}
    for i in range(3, 20):
        for j in range(2, 41):
            for k in range(1, 21):
                model = DecisionTreeClassifier(max_depth=i,
                                               min_samples_split=j,
                                               min_samples_leaf=k)
                model = model.fit(features_train, labels_train)
                dev_predictions = model.predict(features_dev)
                dev_acc = accuracy_score(labels_dev, dev_predictions)
                hyperparameters['max_depth', i,
                                'min_samples_split', j,
                                'min_samples_leaf', k] = dev_acc
    hyper_name = max(hyperparameters, key=hyperparameters.get)
    hyper_value = max(hyperparameters.values())
    return hyper_name, hyper_value


def simplify_opening_names(labels):
    """
    Returns a series consisting of opening names. Eliminates opening
    variations by renaming them as their standard non-varient names.
    Input: series consisting of full opening names.
    """
    new_openings = []
    for opening in labels:
        words = opening.split()
        new_name = ''
        for word in words:
            if word[len(word)-1:] == ':':
                new_name += word[:len(word)-1]
                break
            elif (word[0:1] == '#') | (word == '|'):
                break
            new_name += (word + ' ')
        new_openings.append(new_name)
    return pd.Series(new_openings)


def simplify_elo_ratings(ratings):
    """
    Returns a series consisting of rating categories. Groups elo
    ratings for players into their respective ranking categories.
    Input: dataframe consisting of elo ratings of players
    """
    new_rating = []
    for rating in ratings:
        category = ''
        if rating < 1200:
            category = 'Novice'
        elif rating < 1400:
            category = 'Class D'
        elif rating < 1600:
            category = 'Class C'
        elif rating < 1800:
            category = 'Class B'
        elif rating < 2000:
            category = 'Class A'
        elif rating < 2300:
            category = 'CM'
        elif rating < 2400:
            category = 'FM'
        elif rating < 2500:
            category = 'IM'
        else:
            category = 'GM'
        new_rating.append(category)
    return pd.Series(new_rating)


def best_opening(model):
    """
    Returns a string of the best opening line to follow.
    Asks the user what critical questions to create a
    hypothetical game. Then computes the optimal opening
    line in order for the user to win.
    Input: DecisionTreeClassifier model.
    """
    valid = 0
    while valid == 0:
        rating_op = input('Opponent\'s ELO rating:')
        color_op = input('Opponent\'s starting color: ').lower()
        rating_pl = input('Your ELO rating:')
        if (rating_op.isdigit() & ((color_op == 'white') |
           (color_op == 'black')) & rating_pl.isdigit()):
            valid = 1
        else:
            print('Input valid responses')
    data = {
        'winner_black': 0,
        'winner_draw': 0,
        'winner_white': 0,
        'White rank_GM': 0,
        'White rank_IM': 0,
        'White rank_FM': 0,
        'White rank_CM': 0,
        'White rank_Class A': 0,
        'White rank_Class B': 0,
        'White rank_Class C': 0,
        'White rank_Class D': 0,
        'White rank_FM': 0,
        'White rank_IM': 0,
        'White rank_Novice': 0,
        'Black rank_GM': 0,
        'Black rank_IM': 0,
        'Black rank_FM': 0,
        'Black rank_CM': 0,
        'Black rank_Class A': 0,
        'Black rank_Class B': 0,
        'Black rank_Class C': 0,
        'Black rank_Class D': 0,
        'Black rank_FM': 0,
        'Black rank_IM': 0,
        'Black rank_Novice': 0
    }
    rating_op = pd.DataFrame([[int(rating_op)]],
                             columns=['rating_op'])
    ranking_op = simplify_elo_ratings(rating_op['rating_op'])
    rating_pl = pd.DataFrame([[int(rating_pl)]],
                             columns=['rating_pl'])
    ranking_pl = simplify_elo_ratings(rating_pl['rating_pl'])
    if color_op == 'white':
        data['winner_black'] = 1
        data = elo_rating_conversion(data, ranking_op, ranking_pl)
    else:
        data['winner_white'] = 1
        data = elo_rating_conversion(data, ranking_pl, ranking_op)
    df = pd.DataFrame(data, index=[0])
    best_opening_name = model.predict(df)
    return best_opening_name


def elo_rating_conversion(data, ranking_white, ranking_black):
    """
    Returns a dictionary with updated values. Updates the dictionary
    to reflect the rank/rating of both players.
    Input: dictionary of single game info, rank of white (string),
    and the rank of black (string)
    """
    white = 'White rank_' + (ranking_white.values)[0]
    black = 'Black rank_' + (ranking_black.values)[0]
    for key in data:
        if key == white:
            data[key] = 1
        if key == black:
            data[key] = 1
    return data
