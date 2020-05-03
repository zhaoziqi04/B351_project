#!/usr/bin/python3

### CSCI-B 351 / COGS-Q 351 Spring 2020
### Framework code copyright 2020 B351/Q351 instruction team.
### Do not copy or redistribute this code without permission
### and do not share your solutions outside of this class.
### Doing so constitutes academic misconduct and copyright infringement.

import sys

from board import Board
from player import *

class Game:
    def __init__(self, trace, player1, player2):
        self.trace = trace
        self.player1 = player1
        self.player2 = player2
        self.winner = 0

    def runGame(self):
        board = Board(self.trace)

        while not board.game_over:
            # finds the move to make
            if board.turn == 0:
                #if not isinstance(self.player1, ManualPlayer):
                    #print("Player 1 is thinking")

                move = self.player1.findMove(board.trace)
            else:
                #if not isinstance(self.player2, ManualPlayer):
                    #print("Player 2 is thinking")
                #print("Player 2 is thinking, heuristic value = " + str(self.player2.heuristic(board)))
                move = self.player2.findMove(board.trace)

            # makes the move
            board.makeMove(move)
            #board.print()

        # determines if the game is over or not
        self.winner = board.winner
        #if board.winner == -1:
        #    print("It is a draw!")
        #elif board.winner == 0:
        #    print("Player 1 wins!")
        #elif board.winner == 1:
        #    print("Player 2 wins!")

if __name__ == "__main__":
    # String representing a sequence of moves from a board.
    if len(sys.argv) >= 2:
        trace = sys.argv[1]
    else:
        trace = ''

    # Create player one by calling the
    # player class corresponding to the
    # search algorithm the player uses.
    #p1 = RemotePlayer(3)

    # Same for player 2
    #p2 = ManualPlayer()

    # Create the game instance using the
    # board and players you've made.
    #g = Game(trace, p1, p2)

    #g.runGame()
