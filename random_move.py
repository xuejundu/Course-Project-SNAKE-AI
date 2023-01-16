import argparse
import pygame
import time
import tensorflow as tf
import sys

from environment import *
from random import randrange

class Random_Move:
    def __init__(self, env, num_games):
        self.terminate = False
        self.env = env
        self.max_score = 141
        self.num_games = num_games

    def random_move(self):
        return randrange(4)

    def play(self):
        output_file = None
        filename = 'random_out.txt'
        try:
            output_file = open(filename, 'x')
        except FileExistsError:
            output_file = open(filename, 'w')

        done = False
        running = True
        accumulated_score = 0
        accumulated_efficiency = 0.0
        # num_games = 100
        max_score_achieved = 0
        max_efficient_total = 0

        for i in range(self.num_games):
            total_score = 0
            steps = 0
            efficient_total = 0
            while total_score < self.max_score:
                # time.sleep(0.00625)
                # time.sleep(0.1)
                self.env.render()

                if done:  # Game over, start a new game
                    self.env.reset()
                    done = False
                    print(f'Total score: {total_score}')
                    break

                current_score = self.env.get_total()
                if total_score < current_score:
                    total_score = current_score
                    efficient = current_score / steps
                    efficient_total += efficient
                    steps = 0

                if self.max_score == total_score:
                    print(f'Total score: {total_score}')
                    break

                for event in pygame.event.get():  # Stop the program if we quit the game
                    if event.type == pygame.QUIT:
                        running = False
                # if not running:
                #     break
                
                action = self.random_move()
                _, _, done, _, _ = self.env.step(action)
                steps += 1
            accumulated_score += total_score
            accumulated_efficiency += efficient_total
            if total_score > max_score_achieved:
                max_score_achieved = total_score
            if efficient_total > max_efficient_total:
                max_efficient_total = efficient_total
            output_file.write(f'Game {i+1} - score: {total_score}, average score: {accumulated_score / (i+1)}, max score: {max_score_achieved}, efficiency: {efficient_total}, average efficiency: {accumulated_efficiency / (i+1)}, max efficiency: {max_efficient_total}\n')


if __name__ == "__main__":
    num_games = int(sys.argv[1])
    env = Environment()
    rm = Random_Move(env, num_games)
    rm.play()

