#!/usr/bin/python3

### CSCI-B 351 / COGS-Q 351 Spring 2020
### Framework code copyright 2020 B351/Q351 instruction team.
### Do not copy or redistribute this code without permission
### and do not share your solutions outside of this class.
### Doing so constitutes academic misconduct and copyright infringement.

import math
from board import Board
import random
import math
import numpy as np
import tensorflow as tf

class BasePlayer:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    ##################
    #      TODO      #
    ##################
    # Assign integer scores to the three terminal states
    # P2_WIN_SCORE < TIE_SCORE < P1_WIN_SCORE
    # Access these with "self.TIE_SCORE", etc.
    P1_WIN_SCORE = 100
    P2_WIN_SCORE = -100
    TIE_SCORE =  0

    # Returns a heuristic for the board position
    # Good positions for 0 pieces should be positive and
    # good positions for 1 pieces should be negative
    # for all boards, P2_WIN_SCORE < heuristic(b) < P1_WIN_SCORE
    def heuristic(self, board):
        h1 = board.p1_pits[0]
        h2 = sum(board.p1_pits) - sum(board.p2_pits)
        h3 = 0
        for pit in board.p1_pits:
            if pit != 0:
                h3 += 1
        for pit in board.p2_pits:
            if pit != 0:
                h3 -= 1
        h4 = board.p1_pot
        h6 = -board.p2_pot
        return h1 * 0.2 + h2 * 0.2 + h3 * 0.4 + h4 * 1 + h6 * 0.6
        raise NotImplementedError

    def findMove(self, trace):
        raise NotImplementedError

class RandomPlayer(BasePlayer):
    def __init__(self, max_depth=None):
        BasePlayer.__init__(self, max_depth)
        self.random = random.Random(13487951347859)
    def findMove(self, trace):
        board = Board(trace)
        options = list(board.getAllValidMoves())
        return self.random.choice(options)

class ManualPlayer(BasePlayer):
    def __init__(self, max_depth=None):
        BasePlayer.__init__(self, max_depth)

    def findMove(self, trace):
        board = Board(trace)
        opts = "  "
        for c in range(6):
            opts += " "+(str(c+1) if board.isValidMove(c) else ' ')+"  "

        while True:
            if(board.turn == 0):
                print("\n")
                board.printSpaced()
                print(opts)
                pit = input("Pick a pit (P1 side): ")
            else:
                print("\n")
                print(" " + opts[::-1])
                board.printSpaced()
                pit = input("Pick a pit (P2 side): ")
            try: pit = int(pit) - 1
            except ValueError: continue
            if board.isValidMove(pit):
                return pit

class PlayerMM(BasePlayer):
    def max_v(self, board, depth):
        moves = board.getAllValidMoves()
        bestVal = None #self.P2_WIN_SCORE
        bestMove = None
        for move in moves:
            board.makeMove(move)
            v = self.minimax(board, depth - 1)[1]
            board.undoMove()
            if bestMove == None or v > bestVal or (v == self.P2_WIN_SCORE and bestVal == self.P2_WIN_SCORE):
                bestVal = v
                bestMove = move
        return bestMove, bestVal

    def min_v(self, board, depth):
        moves = board.getAllValidMoves()
        bestVal = None #self.P1_WIN_SCORE
        bestMove = None
        for move in moves:
            board.makeMove(move)
            v = self.minimax(board, depth - 1)[1]
            board.undoMove()
            if bestMove == None or v < bestVal or (v == self.P1_WIN_SCORE and bestVal == self.P1_WIN_SCORE):
                bestVal = v
                bestMove = move
        return bestMove, bestVal

    # performs minimax on board with depth.
    # returns the best move and best score as a tuple
    def minimax(self, board, depth):
        hval = self.heuristic(board)
        if board.game_over or hval >= self.P1_WIN_SCORE or hval <= self.P2_WIN_SCORE:
            if board.p1_pot == board.p2_pot:
                return None, self.TIE_SCORE
            elif board.p1_pot > board.p2_pot:
                return None, self.P1_WIN_SCORE
            elif board.p1_pot < board.p2_pot:
                return None, self.P2_WIN_SCORE
            else:
                raise NotImplementedError
        if depth == 0:
            return None, hval
        #children = board.getAllValidMoves()
        if board.turn == 0: # Maximizing Player 1's turn
            return self.max_v(board, depth)
        else: # Minimizing Player 2's turn
            return self.min_v(board, depth)
        raise NotImplementedError

    def findMove(self, trace):
        board = Board(trace)
        move, score = self.minimax(board, self.max_depth)
        if move == None:
            board.print()
            print("game over: " + str(board.game_over))
        return move

class PlayerAB(BasePlayer):
    # performs minimax with alpha-beta pruning on board with depth.
    # alpha represents the score of max's current strategy
    # beta  represents the score of min's current strategy
    # in a cutoff situation, return the score that resulted in the cutoff
    # returns the best move and best score as a tuple
    def alphaBeta(self, board, depth, alpha, beta):
        hval = self.heuristic(board)
        if board.game_over or hval >= self.P1_WIN_SCORE or hval <= self.P2_WIN_SCORE:
            if board.p1_pot == board.p2_pot:
                return None, self.TIE_SCORE
            elif board.p1_pot > board.p2_pot:
                return None, self.P1_WIN_SCORE
            elif board.p1_pot < board.p2_pot:
                return None, self.P2_WIN_SCORE
            else:
                raise NotImplementedError
        if depth == 0:
            return None, hval
        #children = board.getAllValidMoves()
        if board.turn == 0: # Maximizing Player 1's turn
            return self.max_v(board, depth, alpha, beta)
        else: # Minimizing Player 2's turn
            return self.min_v(board, depth, alpha, beta)
        raise NotImplementedError

    def max_v(self, board, depth, alpha, beta):
        moves = board.getAllValidMoves()
        bestMove = None
        bestScore = None
        for move in moves:
            board.makeMove(move)
            v = self.alphaBeta(board, depth - 1, alpha, beta)[1]
            board.undoMove()
            if bestMove == None or v > bestScore:
                bestScore = v
                bestMove = move
                if v > alpha:
                    alpha = v
                    if beta <= alpha:
                        bestMove = None
                        break
        #return bestMove, alpha
        return bestMove, bestScore

    def min_v(self, board, depth, alpha, beta):
        moves = board.getAllValidMoves()
        bestMove = None
        bestScore = None
        for move in moves:
            board.makeMove(move)
            v = self.alphaBeta(board, depth - 1, alpha, beta)[1]
            board.undoMove()
            if bestMove == None or v < bestScore:
                bestScore = v
                bestMove = move
                if v < beta:
                    beta = v
                    if beta <= alpha:
                        bestMove = None
                        break
        return bestMove, bestScore

    def findMove(self, trace):
        board = Board(trace)
        move, score = self.alphaBeta(board, self.max_depth, -math.inf, math.inf)
        if move == None:
            board.print()
            print("No Move, score: " + str(score) + " hval: " + str(self.heuristic(board)))
        return move

class PlayerDP(PlayerAB):
    ''' A version of PlayerAB that implements dynamic programming
        to cache values for its heuristic function, improving performance. '''
    def __init__(self, max_depth):
        PlayerAB.__init__(self, max_depth)
        self.resolved = {}

    # if a saved heuristic value exists in self.resolved for board.state, returns that value
    # otherwise, uses BasePlayer.heuristic to get a heuristic value and saves it under board.state
    def heuristic(self, board):
        if board.state in self.resolved.keys():
            return self.resolved[board.state]
        else:
            val = super().heuristic(board)
            self.resolved[board.state] = val
            return val
        raise NotImplementedError

class PlayerABMC(BasePlayer):
    def alphaBeta(self, board, depth, alpha, beta, levels_expanded):
        hval = self.heuristic(board)
        if board.game_over or hval >= self.P1_WIN_SCORE or hval <= self.P2_WIN_SCORE:
            if board.p1_pot == board.p2_pot:
                return None, self.TIE_SCORE
            elif board.p1_pot > board.p2_pot:
                return None, self.P1_WIN_SCORE
            elif board.p1_pot < board.p2_pot:
                return None, self.P2_WIN_SCORE
            else:
                raise NotImplementedError
        if depth == 0:
            return None, hval
        if board.turn == 0: # Maximizing Player 1's turn
            #randomly end or continue search
            if (1/(1 + math.exp(-hval*self.sigmoid_coeff)))*self.expand_prob < random.random() and levels_expanded > self.min_depth:
                randMove = random.choice(list(board.getAllValidMoves()))
                return randMove, hval
            return self.max_v(board, depth, alpha, beta, levels_expanded+1)
        else: # Minimizing Player 2's turn
            #randomly end or continue search
            if (1/(1 + math.exp(hval*self.sigmoid_coeff)))*self.expand_prob < random.random() and levels_expanded > self.min_depth:
                randMove = random.choice(list(board.getAllValidMoves()))
                return randMove, hval
            return self.min_v(board, depth, alpha, beta, levels_expanded+1)
        raise NotImplementedError

    def max_v(self, board, depth, alpha, beta, levels_expanded):
        moves = board.getAllValidMoves()
        bestMove = None
        bestScore = None
        for move in moves:
            board.makeMove(move)
            v = self.alphaBeta(board, depth - 1, alpha, beta, levels_expanded)[1]
            board.undoMove()
            if bestMove == None or v > bestScore:
                bestScore = v
                bestMove = move
                if v > alpha:
                    alpha = v
                    if beta <= alpha:
                        bestMove = None
                        break
        return bestMove, bestScore

    def min_v(self, board, depth, alpha, beta, levels_expanded):
        moves = board.getAllValidMoves()
        bestMove = None
        bestScore = None
        for move in moves:
            board.makeMove(move)
            v = self.alphaBeta(board, depth - 1, alpha, beta, levels_expanded)[1]
            board.undoMove()
            if bestMove == None or v < bestScore:
                bestScore = v
                bestMove = move
                if v < beta:
                    beta = v
                    if beta <= alpha:
                        bestMove = None
                        break
        return bestMove, bestScore

    def findScore(self, trace):
        board = Board(trace)
        move, score = self.alphaBeta(board, self.max_depth, -math.inf, math.inf, 0)
        return score
    
    #min depth is the minimum depth of the tree that will be explored
    #expand_prob is the chance for each node beyond the min_depth to be expanded
    def __init__(self, max_depth, expand_prob, min_depth, sigmoid_coeff):
        self.max_depth = max_depth
        self.expand_prob = expand_prob
        self.min_depth=2
        self.sigmoid_coeff = sigmoid_coeff

class PlayerDPMC(PlayerABMC):
    ''' A version of PlayerMCAB that implements dynamic programming
        to cache values for its heuristic function, improving performance. '''
    def __init__(self, max_depth, expand_prob, min_depth, sigmoid_coeff):
        PlayerABMC.__init__(self, max_depth, expand_prob, min_depth, sigmoid_coeff)
        self.resolved = {}

    def heuristic(self, board):
        if board.state in self.resolved.keys():
            return self.resolved[board.state]
        else:
            val = super().heuristic(board)
            self.resolved[board.state] = val
            return val

class PlayerMCTS(BasePlayer):
    #A Player that repeatedly runs alpha beta players that probabilistically expand
    #nodes up to great depth
    def __init__(self, expand_prob=.3, limit = 15, max_depth=40, min_depth=2, sigmoid_coeff=.1):
        self.resolved = {}
        self.limit = limit
        self.player = PlayerDPMC(max_depth, expand_prob, min_depth, sigmoid_coeff)

    def heuristic(self,trace):
        dict={}
        total=0
        board = Board(trace)
        moves = list(board.getAllValidMoves())
        for i in moves:
            dict[i] = 0
            testboard = Board(trace)
            testboard.makeMove(i)
            new_trace = testboard.trace
            for l in range(self.limit):
                score = self.player.findScore(new_trace)
                if board.turn == 0: # Maximizing Player 1's turn
                    dict[i] = np.amax((score, dict[i]))
                else:
                    dict[i] = np.amin((score, dict[i]))
        self.resolved[trace]=dict

    def findMove(self,trace):
        if trace not in self.resolved.keys():
            self.heuristic(trace)
        board = Board(trace)
        dict=self.resolved[trace]
        if board.turn == 0: # Maximizing Player 1's turn
            max_v = self.P2_WIN_SCORE
            best_m = list(dict.keys())[0]
            for i in dict.keys():
                if max_v <= dict[i]:
                    max_v = dict[i]
                    best_m = i
        else: # Minimizing Player 2's turn
            min_v = self.P1_WIN_SCORE
            best_m = list(dict.keys())[0]
            for i in dict.keys():
                if min_v >= dict[i]:
                    min_v = dict[i]
                    best_m = i
        return best_m

#A Player that returns the move suggested by the given neural network model
class NNplayer():
  def __init__(self, model, name="NNplayer"):
    self.model = model
    self.name = name
  #the network was trained to output the optimal move for player 1
  #when the network is player 2, rearrange the input to the network so that the
  #outputted move for player 1 will be equivalent to the optimal move for player 2
  def findMove(self,trace):
    board = Board(trace)
    X = [board.board[:14]]
    if board.turn == 1: #Player 2's turn, so flip the board inputs
      X = [np.concatenate([X[0][0:7],X[0][7:14]]).tolist()]
    a = self.model.predict(X)
    moves_ranked = np.argsort(a)[:][::-1][0]
    potential_children = board.getAllValidMoves(preorder=list(moves_ranked))
    return list(potential_children)[0]
