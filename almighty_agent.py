import argparse
import pygame
import time
import tensorflow as tf
import sys

# from almighty import Almighty_move as am
from environment import *


class Almighty_move:
    def __init__(self, width, height):
        self.cell_size = 20
        self.width = width // 20
        self.height = height // 20
        self.row = 120
        self.col = 120
        self.max_score = self.width * self.height - 3
        self.score = 0
        self.prev_coord = [0,0]
        self.left = 0
        self.right = 1
        self.up = 2
        self.down = 3

    def almighty_move(self):
        num_up_down = (self.width - 2) // 2
        yield self.initial_move_down()
        for i in range(2):
            yield self.move_up()
            yield self.move_down()
        yield self.move_up_last_col()
        yield self.move_back_origin()

        while self.score < self.max_score:
            yield self.move_down_first_col()
            for i in range(0, num_up_down):
                yield self.move_up()
                yield self.move_down()
            yield self.move_up_last_col()
            yield self.move_back_origin()
            self.score += 1

    def get_max_score(self):
        return self.max_score

    def get_cooridnate(self):
        return [self.row, self.col]

    def set_prev_coord(self):
        self.prev_coord[0] = self.row
        self.prev_coord[1] = self.col

    def get_direction(self):
        if self.prev_coord[0] > self.row:
            return self.left
        elif self.prev_coord[0] < self.row:
            return self.right
        elif self.prev_coord[1] > self.col:
            return self.up
        return self.down 

    def can_terminate(self):
        return self.score >= self.max_score

    def move_back_origin(self):
        for i in range(0, self.width - 1):
            self.set_prev_coord()
            self.row -= self.cell_size
            yield [self.get_cooridnate(), self.get_direction()]

    def move_down_first_col(self):
        for i in range(0, self.height - 1):
            self.set_prev_coord()
            self.col += self.cell_size
            yield [self.get_cooridnate(), self.get_direction()]
        self.set_prev_coord()
        self.row += self.cell_size
        yield [self.get_cooridnate(), self.get_direction()]       
        
    def move_up_last_col(self):
        for i in range(0, self.height -1 ):
            self.set_prev_coord()
            self.col -= self.cell_size
            yield [self.get_cooridnate(), self.get_direction()]

    def move_down(self):
        for i in range(0, self.height - 2):
            self.set_prev_coord()
            self.col += self.cell_size
            yield [self.get_cooridnate(), self.get_direction()]
        self.set_prev_coord()
        self.row += self.cell_size
        yield [self.get_cooridnate(), self.get_direction()]

    def initial_move_down(self):
        for i in range(5):
            self.set_prev_coord()
            self.col += self.cell_size
            yield [self.get_cooridnate(), self.get_direction()]
        self.set_prev_coord()
        self.row += self.cell_size
        yield [self.get_cooridnate(), self.get_direction()]

    def move_up(self):
        for i in range(0, self.height - 2):
            self.set_prev_coord()
            self.col -= self.cell_size
            yield [self.get_cooridnate(), self.get_direction()]
        self.set_prev_coord()
        self.row += self.cell_size
        yield [self.get_cooridnate(), self.get_direction()]

    def initial_move_up(self):
        for i in range(4):
            self.set_prev_coord()
            self.col -= self.cell_size
            yield [self.get_cooridnate(), self.get_direction()]
        self.set_prev_coord()
        self.row += self.cell_size
        yield [self.get_cooridnate(), self.get_direction()]

    def show_location(self):
        print(f'({self.row}, {self.col})')
# Almighty_move



class Almighty_agent():
    def __init__(self, env, num_games):
        self.cell_size = 20
        self.algo = Almighty_move(240,240)
        self.terminate = False
        self.moves = self.algo.almighty_move()
        self.env = env
        self.max_score = self.algo.get_max_score()
        self.effs = []
        self.efficient_total = 0
        self.num_games = num_games

    def move_snake(self, coord):
        print(f'({coord[0]}, {coord[1]})')

    def set_env(self, env):
        self.env = env

    def get_effs(self):
        return self.effs

    def play(self):
        done = False
        total_score = 0
        steps = 0

        for loop in self.moves:
            for coord in loop:
                # time.sleep(0.00625)
                # time.sleep(0.025)
                self.env.render()

                if done:  # Game over, start a new game
                    self.env.reset()
                    print(f'Total score: {total_score}')
                    return
                current_score = self.env.get_total()
                if total_score < current_score:
                    total_score = current_score
                    efficient = current_score / steps
                    self.effs.append(efficient)
                    self.efficient_total += efficient
                    steps = 0
                if self.max_score == total_score:
                    print(f'Total score: {total_score}')
                    return

                for event in pygame.event.get():  # Stop the program if we quit the game
                    if event.type == pygame.QUIT:
                        running = False
                        break
                
                action = coord[1]
                _, _, done, _, _ = self.env.step(action)
                steps += 1

    # def show_efficiencies(self):
    #     i = 0
    #     for eff in self.effs:
    #         i += 1
    #         print(f'Efficiency of score {i}: {eff}')
    #     print(f'Total efficiencies: {self.efficient_total}')

def show_efficiencies(effs, num_games):
    output_file = None
    filename = 'almighty_out.txt'
    try:
        output_file = open(filename, 'x')
    except FileExistsError:
        output_file = open(filename, 'w')

    i = 0
    efficient_total = 0.0
    for eff in effs:
        i += 1
        efficient_total += eff
        output_file.write(f'Efficiency of score {i}: {eff / num_games}\n')
    output_file.write(f'Total efficiencies: {efficient_total / num_games}\n')

            
if __name__ == "__main__":
    output_file = None
    filename = 'almighty_out.txt'
    try:
        output_file = open(filename, 'x')
    except FileExistsError:
        output_file = open(filename, 'w')

    num_games = int(sys.argv[1])
    max_score = 141
    effs = [0.0] * max_score
    for i in range(num_games):
        env = Environment()
        aa = Almighty_agent(env, num_games)
        aa.play()
        result = aa.get_effs()
        eff_total = 0.0
        for j in range(max_score):
            effs[j] += result[j]
            eff_total += result[j]
        output_file.write(f'Efficiency of score of game {i+1}: {eff_total}\n')
    
    output_file.write(f'\nAverage efficiency of score of total {num_games}\n')
    i = 0
    efficient_total = 0.0
    for eff in effs:
        i += 1
        efficient_total += eff
        output_file.write(f'Average efficiency of score {i}: {eff / num_games}\n')
    output_file.write(f'Average total efficiencies: {efficient_total / num_games}\n')
