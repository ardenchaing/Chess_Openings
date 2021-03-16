"""
Contains functions for analyzing the optimal opening move
and true piece type values for chess.
"""
import chess
import chess.engine
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})


def opening_move(cleaned):
    """
    Returns the cleaned dataframe with specific constraints, dictionary of all
    the first moves, dictionary of all the first moves where white wins,
    dictionary of the first moves ratio of winning (at least 3% of people
    must play that opening in the dataset to be included), and the opening
    with the highest winning percentage (string). Computes the starting move
    that has the highest probablity of winning for white.
    Input: a dataframe of chess games.
    """
    data = cleaned[(cleaned['turns'] >= 19) &
                   ((cleaned['victory_status'] == 'mate') |
                   (cleaned['victory_status'] == 'resign'))]
    first_moves = {}
    white_win = {}
    for index, row in data.iterrows():
        plays = row['moves'].split()
        if row['winner'] == 'white':
            if plays[0] in white_win:
                white_win[plays[0]] += 1
            else:
                white_win[plays[0]] = 1
        if plays[0] in first_moves:
            first_moves[plays[0]] += 1
        else:
            first_moves[plays[0]] = 1
    win_ratio = {}
    for move, freq in first_moves.items():
        if freq >= sum(first_moves.values()) * 0.03:
            num = 0
            if move in white_win:
                num = white_win[move]
            win_ratio[move] = num / freq
    best_opening = max(win_ratio, key=win_ratio.get)
    return data, first_moves, white_win, win_ratio, best_opening


def plot_opening_move(first_moves, white_win, win_ratio):
    """
    Plots bar graphs for all the first moves played by at least
    3% of the players in the data and its corresponding
    number of times played, number of times won, and winning
    percentage. Saves plot as opening_move.png.
    Input: dictionary of opening moves and amount played,
    dictionary of opening moves and times won, and dictionary
    of opening moves and its winning percentage.
    """
    first_moves_short = {}
    white_win_short = {}
    for move in win_ratio.keys():
        first_moves_short[move] = first_moves[move]
        white_win_short[move] = white_win[move]
    keys_first = first_moves_short.keys()
    values_first = first_moves_short.values()
    keys_white = white_win_short.keys()
    values_white = white_win_short.values()
    keys_ratio = win_ratio.keys()
    values_ratio = win_ratio.values()
    fig, axs = plt.subplots(3, figsize=(15, 10))
    fig.tight_layout(pad=3.00)
    axs[0].bar(keys_first, values_first)
    axs[0].set_title('Distribution of Opening Moves')
    axs[0].set_xlabel('Opening Moves')
    axs[0].set_ylabel('Times Played')
    axs[1].bar(keys_white, values_white)
    axs[1].set_title('Distribution of Opening Moves where White Wins')
    axs[1].set_xlabel('Opening Moves')
    axs[1].set_ylabel('Times Won')
    axs[2].bar(keys_ratio, values_ratio)
    axs[2].set_title('Percentage of Wins for Opening Moves')
    axs[2].set_xlabel('Opening Moves')
    axs[2].set_ylabel('Winning Percentage')
    plt.savefig('opening_move.png')


def piece_value(games):
    """
    Returns a dictionary of all the piece types and their aggregate positional
    evaluation scores from all the games in the given dataset. Utilizes
    the chess engine stockfish to perform positional analysis. Takes into
    account casting (giving points to both the King and Rook) and promotions
    (giving points to the pawn). Also takes into account color moving ensuring
    proper point evaluation.
    Input: a dataframe of chess games.
    """
    engine = chess.engine.SimpleEngine.popen_uci('stockfish')
    pieces = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0}
    board = chess.Board()
    for game in games:
        moves = game.split()
        # visual indication of the current board position
        # print(board)
        info = engine.analyse(board, chess.engine.Limit(time=0.1))
        final_score = score_conversion(info)
        white_adv = final_score
        # checks the starting score of the game from White's POV
        # print('Starting score White POV:', white_adv)
        # print('')
        for move in moves:
            piece_type = []
            play = board.parse_san(move)
            # checks move conversion is correct from algebriac to chess.Move
            # print('Move in algebriac notation:', move)
            # print('Move in from a sq to a sq notation:', play)
            if board.is_castling(play):
                piece_type.append('K')
                piece_type.append('R')
                # checks if move is castling
                # print('Castling')
            elif len(str(play)) > 4:
                piece_type.append('P')
                # checks if move is promotion
                # print('Promotion')
            else:
                location_int = chess.parse_square(str(play)[:2])
                piece_int = board.piece_type_at(location_int)
                piece_name = chess.piece_symbol(piece_int).upper()
                piece_type.append(piece_name)
                # checks the current piece type moving
                # print('Piece moving:', piece_name)
            board.push_san(move)
            # visual indication of the current board position
            # print(board)
            info = engine.analyse(board, chess.engine.Limit(time=0.1))
            final_score = score_conversion(info)
            # checks score from that move from white's POV
            # print('Score from White POV:', final_score)
            piece_score = final_score-white_adv
            if board.turn == chess.WHITE:
                piece_score *= -1
            for i in piece_type:
                pieces[i] += piece_score
            # checks piece type score calculated correctly
            # and also placed into correct location in dictionary
            # print('Score for that move:', piece_score)
            # print(pieces)
            # print('')
            white_adv = final_score
        board.reset()
    engine.quit()
    return pieces


def score_conversion(info):
    """
    Returns the score of the new position from white's POV. Scores
    recieved as either Cp (centi-pawns) or Mate.
    Example of order can be seen as
    Mate(-0) < Mate (-2) < Cp(-50) < Cp(2000) < Mate(12) < Mate(0).
    Conversions from Mate to Cp is 5000 - mate score.
    Example being Mate(5) = Cp(4995).
    Input: a dictionary of aggregated information produced by
    the chess engine.
    """
    score = str(info["score"])
    score_color = score[len(score)-6:len(score)-1]
    score_value = score[9:len(score)-8]
    # checks string slicing is correct
    # print('Color:', score_color)
    # print('Score:', score_value)
    final_score = 0
    if score_value[0] == 'M':
        if score_value[5:6] == '0':
            final_score = 5000-int(score_value[5:len(score_value)-1])
        else:
            final_score = 5000-int(score_value[6:len(score_value)-1])
            if score_value[5:6] == '-':
                final_score *= -1
    else:
        if score_value[3:4] == '0':
            final_score = int(score_value[3:len(score_value)-1])
        else:
            final_score = int(score_value[4:len(score_value)-1])
            if score_value[3:4] == '-':
                final_score *= -1
    if score_color == 'WHITE':
        return final_score
    else:
        return final_score * -1


def plot_piece_value(pieces):
    """
    Plots a bar graph of all the chess piece types and their
    respective chess engine score evaluation aggregates.
    Saves plot as piece_vale.png.
    Input: dictionary of all the chess piece types and their
    respective scores.
    """
    keys = pieces.keys()
    values = pieces.values()
    fig, axs = plt.subplots(1, figsize=(18, 10))
    plt.bar(keys, values)
    plt.title('Chess Piece Type Value Evaluation')
    plt.xlabel('Piece Type')
    plt.ylabel('Score evaluation')
    plt.savefig('piece_value.png')
