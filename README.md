# Chess_Openings
Author: Arden Chaing

## Table of Contents
- `data_cleanse.py`: cleans the dataset(s)
- `data_analysis.py`: contains functions for analyzing the optimal opening move and true piece type values for chess
- `ml.py`: constructs and trains a machine learning model on the featured data
- `test.py`: provides a test suite to validate the program
- `main.py`: runs functions for the chess opening analysis
- `cse163_utils.py`: module to house functions used in testing
- `stockfish.exe`: chess engine used for analysis

## Reproducing Results
1. Download all the table of contents listed above
2. Download all the csv files from the datasets folder above
3. All code should run properly with the following Python libraries:
   - Chess
   - Pandas
   - Matplotlilb
   - Scikit-learn
6. Install stockfish either from above or from the [website](https://stockfishchess.org/)
   1. unzip/extract all the download (for reference I downloaded the Window's 64-bit version)
   2. rename the application file as stockfish (for reference mine was initially name: stockfish_12_win_x64, type: application)
   3. make sure line 96 in `data_analysis.py` references where you've downloaded the application stockfish
7. Run `main.py` to reproduce similar results from the report
   - Best Opening Move for White
     - results will be the exact same as the report
   - Bishop vs Knight
     - results may not be the same as the report
     - this is because it's running an analysis through a chess engine (stockfish) for every single move in the 132 games
     - took my computer 20 minutes to finish compliling
   - Optimal Opening Line for the Opponent
     - results may not be the same as the report 
     - this is because it's discovering different optimal hyperparameter combinations every time it runs in order to maximize testing accuracy
     - chess is a very complicated game thus comes up with a different model each time
     - testing accuracy almost always greater than 10%
     - took my computer 8 minutes to finish compliling
