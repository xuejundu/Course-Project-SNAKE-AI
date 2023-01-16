import argparse
import os
import time

import pygame
import tensorflow as tf
import numpy as np

from actorCritic import ActorCritic
from environment import *


games_scores = []  # List that will contain the score of each game played by the gamebot
games_steps = []
games_effectiveness = []

parser = argparse.ArgumentParser(description='DQN-snake testing.')

parser.add_argument('--numberOfGames', type=int, required=False, default=500,
                    help='Number of test games.')

parser.add_argument('--slowDownFactor', type=float, required=False, default=0.06,
                    help='The factor to make the game slow down. A value of 0 means the games is at full speed.')

parser.add_argument('--modelName', type=str, required=True,
                    help='The name of the model.')

parser.add_argument('--loadModel', type=str, required=True, default="",
                    help='The file saves the models.')

args = parser.parse_args()
file = args.loadModel
n_games = args.numberOfGames
slow_down_factor = args.slowDownFactor
model_name = args.modelName

def make_agent_play_games(env, agent, n_games, slow_down_factor):
    """
    Make the agent play a given number of games

    :param n_games: The number of games to play.
    :param slow_down_factor: Throttling to make the snake move less rapidly.
    :return: A list containing the score of each game played.
    """
    episode = 0
    iterations_without_progress = 0
    max_without_progress = 200
    best_total = 0
    steps = 0
    effective = 0

    while episode < n_games:  # Number of games that we want the robot to play
        time.sleep(slow_down_factor)     # Make the game slow down

        env.render(display=True)
        observation = env.screenshot()
        cur_state = env.get_last_frames(observation)

        action = agent.predict(cur_state)
        new_state, reward, done, eat_apple, length = env.step(action)
        steps += 1

        if eat_apple:
            effective += env.snake.total / steps

        # Check if the snake makes progress in the game
        if env.snake.total > best_total:
            best_total = env.snake.total
            iterations_without_progress = 0
        else:
            iterations_without_progress += 1
        # If the snake gets stuck, the game is over
        if iterations_without_progress >= max_without_progress:
            done = True

        if done:   # Game over, start a new game
            time.sleep(1)
            games_scores.append(env.snake.total)
            games_steps.append(steps)
            games_effectiveness.append(effective)
            env.reset()
            episode += 1  # Increment the number of games played
            iterations_without_progress = 0
            best_total = 0
            steps = 0
            effective = 0

        for event in pygame.event.get():  # Stop the program if we quit the game
            if event.type == pygame.QUIT:
                print("pygame quit")
                break;

    return games_scores

def writeToFile(model_name):
    with open(model_name + '_test_score.txt', 'w') as f:
        for item in games_scores:
            f.write("%s\n" % str(item))

    with open(model_name + '_test_step.txt', 'w') as f:
        for item in games_steps:
            f.write("%s\n" % str(item))

    with open(model_name + '_test_efficiency.txt', 'w') as f:
        for item in games_effectiveness:
            f.write("%s\n" % str(item))

if __name__ == '__main__':

    pygame.init()   # Intializes the pygame

    env = Environment()


    agent = ActorCritic()
    agent.load_model(file)

    games_scores = make_agent_play_games(env, agent, n_games, slow_down_factor)
    writeToFile("tests/" + model_name)
    mean_score = np.mean(games_scores)
    std = np.std(games_scores)
    max_score = np.max(games_scores)


    print("Max score {:.2f}\tMean score {:.2f}\tStandard deviation {:.2f} ".format(max_score, mean_score, std))
