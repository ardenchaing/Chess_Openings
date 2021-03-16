"""
Runs functions for the Chess Opening analysis.
"""
import data_cleanse
import data_analysis
import ml


def run_plot_opening_move(data):
    """
    Runs the opening_move and plot_opening_move method
    from data_analysis. Prints the best opening moves.
    """
    data, first_moves, white_win, win_ratio, best_opening = \
        data_analysis.opening_move(data)
    print('')
    print('Results from opening move method')
    print('Total games analyzed:', sum(first_moves.values()))
    print('First moves times played:', first_moves)
    print('First moves where White wins:', white_win)
    print('Win ratios:', win_ratio)
    print('Best opening:', best_opening)
    print('')
    data_analysis.plot_opening_move(first_moves, white_win, win_ratio)


def run_piece_value(data):
    """
    Runs the piece_value and plot_piece_value method
    from data_analysis. Can uncomment print statements in
    piece_value method to validate the approach and algorithm
    used to calcualte each piece value. Best validation on
    clean_test1 data since a lot of information is shown.
    Runtime with clean_data_ma (132 games) took 20 minutes.
    """
    evaluation = data_analysis.piece_value(data['moves'])
    print('')
    print('Results from piece_value method')
    print('Piece evaluation:', evaluation)
    print('')
    data_analysis.plot_piece_value(evaluation)


def run_opening_model(data):
    """
    Runs the opening_model method from ml.
    """
    model, depth_num, split_num, leaf_num, test_acc = ml.opening_model(data)
    print('')
    print('Results from opening_model method')
    print('Parameters used:')
    print('max_depth =', depth_num)
    print('min_samples_split =', split_num)
    print('min_samples_leaf =', leaf_num)
    print('Test accuracy:', test_acc)
    print('')
    best_opening_name = ml.best_opening(model)
    print('')
    print('Results from best_opening model')
    print('Best opening line to follow:', best_opening_name)


def main():
    # import the dataset and test files
    filepath_data = ('datasets/chess_games_20k.csv')
    filepath_test1 = ('datasets/chess_games_1.csv')
    # cleaning the dataset and test files
    ori_data, clean_data = data_cleanse.processing(filepath_data)
    ori_data_ma, clean_data_ma = data_cleanse.processing(filepath_data, 2200)
    ori_data_all, clean_data_all = data_cleanse.processing(filepath_data, 800)
    ori_test1, clean_test1 = data_cleanse.processing(filepath_test1)
    # running data_analysis methods on final dataset
    run_plot_opening_move(clean_data)
    # also can input clean_test1 to see algorithm validation
    # run_piece_value(clean_data_ma)
    run_opening_model(clean_data_all)


if __name__ == '__main__':
    main()
