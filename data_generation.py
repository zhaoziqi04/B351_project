from board import Board
from a4 import Game
from player import *
from random import randint
from random import choice
import time
import psutil
import tensorflow as tf
from itertools import combinations
import csv


data_generation_begun = False

csv_file_name = 'AB_young_boards_data.csv'
PlayerMC = PlayerMCTS(expand_prob=1.0)
PlayerAB = PlayerDP(10)

#num_boards specifies the number of data points to generate
num_boards = 500
num_random_moves_in_generated_board = 20

#player stores the Mancala Agent used to generate the training data
#if generating data for the alpha beta agent, set player equal to PlayerAB
#if generating data for the monte carlo agent, set player equal to PlayerMC
player = PlayerAB


#generate a list of traces that represent random board states to use as input for the network
#create num_boards number of strings representing randomly generated boards
def get_traces(num_boards, num_random_moves_in_generated_board):
  traces = []
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

def get_row(player, trace):
  board = Board(trace)
  optimalMove = player.findMove(trace)
  row = {}
  for i in range(0,6):
    if i == optimalMove:
      row_val = 1
    else:
      row_val = 0
    row["move_" + str(i)] = row_val
  pos_number = 0
  for pos in board.board[0:14]:
    row["pos_" + str(pos_number)] = pos
    pos_number += 1
  return row
#generate data for the neural network
traces = get_traces(num_boards, num_random_moves_in_generated_board)
#create a csv file with 20 columns, recording the number of stones in each of the 14 board positions, and the one hot encoded optimal move(6 possible values)
print("Start generating data")
start_time = time.perf_counter()
with open(csv_file_name, 'a', newline='') as file:
    #initialize the array of column names for the data faile
    fieldnames = ["move_0","move_1","move_2","move_3","move_4","move_5"]
    for i in range(0,14):
      fieldnames.append("pos_" + str(i))
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    #add the header containing the column names if the header has not already been created
    if(data_generation_begun == False):
      writer.writeheader()
      file.flush()
      data_generation_begun = True
    countval = 0
    print("Created header")
    for trace in traces:
      countval += 1
      writer.writerow(get_row(player, trace))
      #save file progress periodically in case the computer runs out of RAM partway through the generation process
      if countval % 100 == 0:
          print(countval, " data points generated so far")
          file.flush()
print("It took " + str(time.perf_counter()-start_time) + " seconds to generate " + str(len(traces)) + " data points")
