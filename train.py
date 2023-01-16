import argparse

import pygame
import tensorflow as tf

from actorCritic import ActorCritic
from environment import *

# from time import time

parser = argparse.ArgumentParser(description='DQN-snake testing.')

parser.add_argument('--modelName', type=str, required=True,
                    help='The name of the model.')

parser.add_argument('--learningRate', type=float, required=False, default=0.001,
                    help='Learning rate of the training of the agent.')

parser.add_argument('--memorySize', type=int, required=False, default=500000,
                    help='The number of past events remembered by the agent.')

parser.add_argument('--discountRate', type=float, required=False, default=0.99,
                    help={'The discount rate is the parameter that indicates how many actions '
                          'will be considered in the future to evaluate the reward of a given action.'
                          'A value of 0 means the agent only considers the present action, '
                          'and a value close to 1 means the agent considers actions very far in the future.'})

parser.add_argument('--epsilonMin', type=float, required=False, default=0.00,
                    help='The percentage of random actions take by the agent.')

parser.add_argument('--numberOfSteps', type=int, required=False, default= 100000,
                    help='The total number of training games.')

parser.add_argument('--writtenFile', type=str, required=False, default="out.txt",
                    help='The file saves the steps.')

parser.add_argument('--loadFile', type=str, required=False, default="",
                    help='The file saves the steps.')


args = parser.parse_args()
model_name = args.modelName
learning_rate = args.learningRate
memory_size = args.memorySize
discount_rate = args.discountRate
eps_min = args.epsilonMin
n_steps = args.numberOfSteps
file = args.writtenFile
loadFile = args.loadFile

observe = 5000


#session = tf.compat.v1.Session()


def train(env, agent):
    # file_writer = get_file_writer(model_name=model_name, session=session)
    # checkpoint_path = get_checkpoint_path(model_name=model_name)

    running = True
    done = False
    #iteration = 0
    n_games = 0
    mean_score = 0
    curr_max_score = 0

    curr_best_steps = 0
    curr_steps = 0
    curr_score = 0
    pygame_quit = False

    for i in range(n_steps):
        curr_steps = 0
        curr_score = 0
        fail_to_eat_apple_steps = 0
        while True:
            env.render()

            if done:  # Game over, start a new game
                curr_score = env.snake.total

                if curr_score > curr_max_score:
                    curr_max_score = env.snake.total

                if curr_steps > curr_best_steps:
                    curr_best_steps = curr_steps

                env.reset()
                agent.finish_game(start_decay = (i > observe))

                n_games += 1
                mean_score = env.total_rewards / n_games
                done = False
                break

            for event in pygame.event.get():  # Stop the program if we quit the game
                if event.type == pygame.QUIT:
                    pygame_quit = True

            if (pygame_quit):
                break

            observation = env.screenshot()
            cur_state = env.get_last_frames(observation)

            action = agent.act(cur_state)
            new_state, reward, done, eat_apple, length = env.step(action)
            curr_steps += 1

            if (not eat_apple):
                fail_to_eat_apple_steps += 1
            else:
                fail_to_eat_apple_steps = 0

            agent.remember(cur_state, action, reward, new_state, done, eat_apple, length, fail_to_eat_apple_steps)

            agent.replay()

            # do not have to call target train that often
            # maintain stability
            # https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
            if (i % 100 == 0):
                agent.target_train()

        if (pygame_quit):
            break


        f_score=open(model_name + '_score_' + file,'a')
        f_score.write(str(curr_score) + "\n")
        f_score.close()

        f_step=open(model_name + '_step_' + file,'a')
        f_step.write(str(curr_steps) + "\n")
        f_step.close()

        if i % 50 == 0:
            agent.save_model('models/' + model_name)
            print("\rTraining step {}/{} ({:.1f})%\tMean score {:.2f}\tMax score {}\tBest Surviving Step {}".format(
                    i, n_steps, i / n_steps * 100, mean_score, curr_max_score, curr_best_steps), end="")


if __name__ == '__main__':
    pygame.init()  # Intializes the game

    environment = Environment()
    training_agent = ActorCritic(learning_rate=learning_rate,
                                 memory_size=memory_size, discount_rate=discount_rate,
                                 eps_min=eps_min)
    training_agent.start(loadFile)

    train(environment, training_agent)
