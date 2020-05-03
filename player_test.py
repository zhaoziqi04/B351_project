from board import Board
from a4 import Game
from player import *
from random import randint
from random import choice
import time
import tensorflow as tf
from itertools import combinations
#create num_boards number of strings representing randomly generated boards
def get_traces(num_boards):
  traces = []
  num_random_moves_in_generated_board = 20
  while len(traces) < num_boards:
    board = Board()
    trace = ""
    #generate a trace for a game with a given number of random moves previously played
    #num_random_moves_in_generated_board specifies the number of random moves for the board state
    for j in range(num_random_moves_in_generated_board):
        moves = list(board.getAllValidMoves())
        if len(moves) != 0:
            move = choice(moves)
            trace += str(move)
            board.makeMove(move)
    #continue generating random moves for the board until the next player of the game is player 1
    while(board.turn == 1):
      moves = list(board.getAllValidMoves())
      if len(moves) != 0:
          move = choice(moves)
          trace += str(move)
          board.makeMove(move)
      else:
        break
    #only use the trace to generate training data if the game for the generated board has not already ended(there is not yet a winner) and a correct next move exists
    if(board.game_over == False):
      traces.append(trace)
  return traces

def player_battle(player_one, player_two, traces):
  p2_wins = 0
  total_games = len(traces)
  start_time = time.perf_counter()
  for trace in traces:
    game = Game(trace, player_one, player_two)
    game.runGame()
    if game.winner == 1:
        p2_wins += 1
    elif game.winner == -1:
      p2_wins += 0.5
  #swap which player starts first and rerun all games to prevent imbalanced boards from affecting the final result
  for trace in traces:
    game = Game(trace, player_two, player_one)
    game.runGame()
    if game.winner == 0:
      p2_wins += 1
    elif game.winner == -1:
      p2_wins += 0.5
  percent_wins = p2_wins / total_games * 100 / 2
  print(str(np.round(time.perf_counter() - start_time, decimals=2)) + " seconds to complete")
  return percent_wins

def get_player_name(player):
    p_name = player.__class__.__name__
    if p_name == "PlayerDP":
        p_name = p_name + str(player.max_depth)
    if p_name == "NNplayer":
        p_name = player.name
        return p_name

  #run a round robin with each mancala agent in the players array and print the results
def round_robin(players):
  all_players = combinations(players, 2)
  traces = get_traces(50)
  print("Round Robin Begun")
  for player_set in all_players:
    percent_wins = player_battle(player_set[0], player_set[1], traces)
    p1_name = get_player_name(player_set[0])
    p2_name = get_player_name(player_set[1])
    print(p2_name + " wins " + str(percent_wins) + " percent of " + str(len(traces) * 2) + " games against " + p1_name)
  print("Round Robin Completed")

  # create neural network players
mc_model = tf.keras.models.load_model('MC_model')
ab_model = tf.keras.models.load_model('AB_model')
MC_NNPlayer = NNplayer(mc_model, name="MC_NNPlayer")
AB_NNPlayer = NNplayer(ab_model, name="AB_NNPlayer")

#create all other players
AB_low_player = PlayerDP(2)
AB_med_player = PlayerDP(5)
AB_high_player = PlayerDP(10)
MC_player = PlayerMCTS(expand_prob=0.3, limit=20)
random_player = RandomPlayer()

players = [MC_NNPlayer, AB_NNPlayer, AB_low_player, AB_med_player,AB_high_player, MC_player, random_player]
players = [random_player, MC_NNPlayer, AB_NNPlayer, MC_player]
round_robin(players)
