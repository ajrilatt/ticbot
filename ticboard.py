# Adam Rilatt
# 09 / 18 / 20
# Tic Tac Toe

'''
This script implements a game of Ultimate Tic-Tac-Toe. Besides being a game
player, this script also allows for generation of training samples.
'''

import multiprocessing
import numpy as np
import random
import time
import h5py
import os

''' Adjustable parameters. '''
GAME_COUNT = 100_000

# while playing another human, save the board states as training data
TRAIN_MODE  = False

# run as many autonomous games as possible
CORE_MODE   = True

# print interface and play against another person, or against a bot if PLAY_BOT
HUMAN_MODE  = False

# play against a bot. Must be paired with HUMAN_MODE
PLAY_BOT    = False

# ask the bot who's going to win after each move
BOT_OPINION = False

RECORD_FILE = h5py.File('tictac_record_test.h5', 'a')
BOT_NAME    = 'brain2.h5'
PLAYER_X    = 'X'
PLAYER_Y    = 'O'
WIN_COMBOS  = ((0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6))

class TicBoard():

    def __init__(self):

        self.board = [[k for k in range(i, i + 9)] for i in range(0, 81, 9)]
        self.meta  = [False for i in range(9)]
        self.turn  = 0

    def draw_grid(self):
        ''' Draws the current grid after clearing the terminal screen.       '''

        os.system('cls')

        # this is painful to look at. never touch this again, and if you do,
        # succeed where I failed.

        s = "\n"
        s += "\t\t    %s TURN -- MOVE %d" % (PLAYER_Y if self.turn % 2 else PLAYER_X, self.turn)
        s += "\n\n" + (" " * 18 + "|") * 2

        for row in range(0, 9, 3):

            if row in (3, 6):
                # row dividers
                s += "\n" + ("-" * 18 + "+") * 2 + "-" * 18
                s += "\n" + (" " * 18 + "|") * 2

            for col in range(0, 9, 3):

                s += "\n" + "|  ".join([
                            "  %-6s%-6s%-4s" % tuple(self.board[row][col : col + 3]),
                            "%-6s%-6s%-4s" % tuple(self.board[row + 1][col : col + 3]),
                            "%-6s%-6s%-4s" % tuple(self.board[row + 2][col : col + 3])
                            ])

                # column dividers
                s += "\n" + (" " * 18 + "|") * 2

        print(s)


    def set_square(self, move):
        ''' Attempts to take a square, designated by its index 0-80, for the
            current player.                                                  '''

        # set piece on board
        self.board[move // 9][move % 9] = PLAYER_Y if self.turn % 2 else PLAYER_X

        # set metaboard to reflect whether this has won a sub-square or not
        subsquare = self.board[move // 9]
        self.meta[move // 9] = (PLAYER_Y if self.turn % 2 else PLAYER_X) if any(
                                subsquare[r[0]] == subsquare[r[1]] and
                                subsquare[r[1]] == subsquare[r[2]]
                                for r in WIN_COMBOS) else self.meta[move // 9]

    def win_check(self):
        ''' Returns the symbol for X if X has won, the symbol for Y if Y has won
            and 0 if the match should continue.                              '''

        # check for win conditions on the metaboard created above.
        if any(self.meta[r[0]]==self.meta[r[1]] and
               self.meta[r[1]]==self.meta[r[2]] and self.meta[r[0]] for r in WIN_COMBOS):

            return self.meta[0]

        else:
            return 0

    def space_options(self, previous_move):
        ''' In Ultimate Tic-Tac-Toe, the previous move dictates which spaces
            the current move may take. This returns the squares available to the
            current player based on the previous move.                       '''

        # first move condition / special condition: anything goes with None
        if previous_move is None:
            return [k for k in self.get_flat() if k not in (PLAYER_X, PLAYER_Y)]

        # previous player forced current player into taken sub-square => play
        # anywhere except for a taken square
        if self.meta[previous_move % 9]:

            valid = []
            for i, l in enumerate(self.board):
                for k in l:
                    if k not in (PLAYER_X, PLAYER_Y) and not self.meta[i]: valid.append(k)

            return valid

        # normal case: previous player forced current player in to open sub-
        # square, so they must pick from that sub-square
        return [k for k in self.board[previous_move % 9] if k not in (PLAYER_X, PLAYER_Y)]

    def get_flat(self):
        ''' The neural net takes in data in one dimension, so we store it as
            such. Flattens the game board.                                   '''

        return sum(self.board, [])


def train_tac_toe(return_dict, game_count):
    ''' Worker process for multiprocessing. Plays a designated number of random
        games, then adds the results to return_dict.                         '''

    game_record = []
    game_out    = []

    t1 = time.perf_counter()

    for game_num in range(game_count):

        board = TicBoard()
        previous_move = None

        while True:

            options = board.space_options(previous_move)

            # out of moves? the enemy wins
            if len(options) < 1:
                break

            move = random.choice(options)

            board.set_square(move)
            previous_move = move

            board_state = []
            for square in board.get_flat():
                if square == PLAYER_X:
                    board_state.append(1)

                elif square == PLAYER_Y:
                    board_state.append(-1)

                else:
                    board_state.append(0)

            game_record.append(board_state)

            if board.win_check() != 0:
                break

            board.turn += 1

        # loop broken -> a win condition has been reached
        t2 = time.perf_counter()

        options = board.space_options(previous_move)
        win     = board.win_check()

        for i in range(board.turn):
            if len(options) < 1:
                game_out.append([0, 0, 1])

            elif win == PLAYER_X:
                game_out.append([1, 0, 0])

            else:
                game_out.append([0, 1, 0])

    name = multiprocessing.current_process().name

    return_dict[name] = (game_record, game_out)

    print("%s completed %d tasks in %.3f seconds." % (name, game_count, (t2 - t1)))


if __name__ == "__main__":

    # This script will generate the data files used in training the neural net.
    # For now, we'll play two randomized players agaist each other.

    # an HDF5 file was created previously. It contains two datasets:
    #   - X, 81 columns and extendable rows, containing all board states
    #   - Y,  3 columns and extendable rows, containing who won each game

    '''
    SYSTEM NOTES:
    - 5MB per 20k games in about 5s
    - Definitely a RAM bottleneck... rewrite to optimize for memory use
    '''

    if PLAY_BOT or BOT_OPINION:
        from tensorflow import keras
        brain = keras.models.load_model(BOT_NAME, compile = True)
        bot_thought = [0, 0]

    # generate samples as quickly as possible, no visual
    if CORE_MODE:

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        jobs = []
        for i in range(multiprocessing.cpu_count()):
            p = multiprocessing.Process(target = train_tac_toe, args = (
                                        return_dict, GAME_COUNT // multiprocessing.cpu_count()
                                        ))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        for i, proc in enumerate(return_dict.keys()):

            print("Saving %d / %d to the record file..." % (i + 1, len(return_dict.keys())), end='\r')

            # game_record contains X train, and game_out contains Y train.

            game_record = np.array(return_dict[proc][0])
            game_out    = np.array(return_dict[proc][1])

            RECORD_FILE['X'].resize(RECORD_FILE['X'].shape[0] + len(game_record), axis = 0)
            RECORD_FILE['Y'].resize(RECORD_FILE['Y'].shape[0] + len(game_out),    axis = 0)

            RECORD_FILE['X'][-len(game_record):] = game_record
            RECORD_FILE['Y'][-len(game_out):]    = game_out

            print("... Saved.\t\t\t\t", end='\r')


    # human and / or train mode
    else:
        game_record = []
        game_out    = []

        import time
        t1 = time.perf_counter()

        for game_num in range(GAME_COUNT):

            board = TicBoard()
            previous_move = None

            while True:

                if TRAIN_MODE and not HUMAN_MODE:
                    print("Game %-5d / %d \t Move %-2d / 81" %
                          (game_num, GAME_COUNT, board.turn), end='\r')

                options = board.space_options(previous_move)

                # out of moves? the enemy wins
                if len(options) < 1:
                    break

                if HUMAN_MODE:

                    board.draw_grid()

                    if previous_move is not None:
                        print("%s move was to square %d." % (PLAYER_X if board.turn % 2
                              else PLAYER_Y, previous_move))

                        if BOT_OPINION:
                            print("Bot thought: %s, %.2f%%" % (bot_thought[0], bot_thought[1] * 100))
                    try:
                        move = int(input("Enter move, %s. > " % (
                                 PLAYER_Y if board.turn % 2 else PLAYER_X
                            )))
                    except ValueError:
                        continue

                    if move not in options:
                        continue

                else:
                    # robot player
                    move = random.choice(options)

                board.set_square(move)
                previous_move = move

                board_state = []
                for square in board.get_flat():
                    if square == PLAYER_X:
                        board_state.append(1)

                    elif square == PLAYER_Y:
                        board_state.append(-1)

                    else:
                        board_state.append(0)

                if TRAIN_MODE:
                    game_record.append(board_state)

                if BOT_OPINION:

                    # given the current board state, predict who's going to win
                    percents = brain.predict(np.array(board_state).reshape(-1, 81))
                    pred = int(np.argmax(percents, axis = 1))
                    bot_thought = [["X", "O", "DRAW"][pred], np.max(percents)]

                if board.win_check() != 0:
                    break

                board.turn += 1

            # loop broken -> a win condition has been reached

            options = board.space_options(previous_move)
            win     = board.win_check()

            if HUMAN_MODE:
                board.draw_grid()

                if len(options) < 1:
                    print("DRAW.")

                else:
                    print("PLAYER %s WIN! Final move to %d." % (win, previous_move))


                repeat = input("Play again? (y)es, (n)o >").lower()

                if repeat == 'n':
                    break

            if TRAIN_MODE:

                for i in range(board.turn):
                    if len(options) < 1:
                        game_out.append([0, 0, 1])

                    elif win == PLAYER_X:
                        game_out.append([1, 0, 0])

                    else:
                        game_out.append([0, 1, 0])


        t2 = time.perf_counter()
        print("\nCompleted in %.4f seconds." % (t2 - t1))

        print("Saving to the record file...")

        # game_record contains X train, and game_out contains Y train.

        game_record = np.array(game_record)
        game_out    = np.array(game_out)

        RECORD_FILE['X'].resize(RECORD_FILE['X'].shape[0] + len(game_record), axis = 0)
        RECORD_FILE['Y'].resize(RECORD_FILE['Y'].shape[0] + len(game_out),    axis = 0)

        RECORD_FILE['X'][-len(game_record):] = game_record
        RECORD_FILE['Y'][-len(game_record):] = game_out

        print("... Saved.")
