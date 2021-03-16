"""
Provides a test suite to validate the program.
"""
from cse163_utils import assert_equals
import pandas as pd
import data_cleanse
import data_analysis
import ml


def test_data_cleanse(ori_test30, clean_test30):
    """
    Test the processing method from data_cleanse.
    """
    assert_equals(30, len(ori_test30))
    assert_equals(16, len(ori_test30.columns))
    assert_equals(8, len(clean_test30.columns))
    assert_equals(20, len(clean_test30))
    print('Passed: import and clean data')


def test_opening_move(clean_test30):
    """
    Tests the opening_move method from data_analysis.
    """
    data, first_moves, white_win, win_ratio, best_opening = \
        data_analysis.opening_move(clean_test30)
    assert_equals(len(data), sum(first_moves.values()))
    assert_equals({'e4': 4, 'd4': 5, 'Nf3': 4, 'c4': 3},
                  first_moves)
    assert_equals({'e4': 4, 'd4': 2, 'Nf3': 1},
                  white_win)
    assert_equals({'e4': 1.0, 'd4': 0.4, 'Nf3': 0.25, 'c4': 0.0},
                  win_ratio)
    assert_equals('e4', best_opening)
    print('Passed: optimal opening move')


def test_piece_value(chess_games):
    """
    Tests the piece_value method from data_analysis.
    """
    pieces = data_analysis.piece_value(chess_games['moves'])
    for piece, score in pieces.items():
        if (piece == 'P') | (piece == 'N'):
            assert(score != 0)
        else:
            assert(score == 0)
    print('Passed: piece value analysis')


def test_simplify_opening_names(clean_test9):
    """
    Tests the simplify_opening_names method in ml.
    """
    test_list = ['A', 'B ', 'C ', 'D E', 'F G H', 'I ', 'J ', 'K', 'L M ']
    test_series = pd.Series(test_list)
    names = ml.simplify_opening_names(clean_test9['opening_name'])
    if test_series.equals(other=names):
        print('Passed: simplify opening names')


def test_simplify_elo_ratings(clean_test9):
    """
    Tests the simplify_elo_ratings method in ml.
    """
    test_list = ['Class D', 'Class C', 'Class B', 'Class A', 'CM',
                 'FM', 'IM', 'GM', 'GM']
    test_series = pd.Series(test_list)
    ranks = ml.simplify_elo_ratings(clean_test9['white_rating'])
    if test_series.equals(other=ranks):
        print('Passed: simplify elo ratings')


def main():
    # import the test files
    filepath_test30 = ('datasets/chess_games_30.csv')
    filepath_test9 = ('datasets/chess_games_9.csv')
    filepath_data = ('datasets/chess_games_20k.csv')
    # cleaning the dataset and test files
    ori_data, clean_data = data_cleanse.processing(filepath_data)
    # cleaning the test files
    ori_test30, clean_test30 = data_cleanse.processing(filepath_test30)
    ori_test9, clean_test9 = data_cleanse.processing(filepath_test9)
    # testing data_analysis methods
    print('Beginning tests')
    test_data_cleanse(ori_test30, clean_test30)
    test_opening_move(clean_test30)
    test_piece_value(clean_test30)
    test_simplify_opening_names(clean_test9)
    test_simplify_elo_ratings(clean_test9)
    print('All tests passed!')


if __name__ == '__main__':
    main()
