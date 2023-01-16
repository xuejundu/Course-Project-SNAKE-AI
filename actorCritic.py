import os

import numpy as np
import tensorflow as tf
import random

from utils import ReplayMemory, GameReplayMemory

INPUT_HEIGHT = 64
INPUT_WIDTH = 64
CHANNELS = 4
N_OUTPUTS = 4  # Number of possible actions that the agent can make (the four directions)
REWARD_THRES = 0.5
#SAMPLING_PROP = 0.8

# training gap params
p = 0.4
q = 2
k = 10

class ActorCritic:

    def __init__(self, learning_rate=10 ** (-6), memory_size=100000, discount_rate=0.99, eps_min=0.001, eta_min = 0.5):
        self.learning_rate = learning_rate

        # self.tau = 0.05

        # self.model = self.cnn_model()
        # self.target_model = self.cnn_model()

        self.memory_size = memory_size
        self.memory1 = ReplayMemory(int(self.memory_size / 2))
        self.memory2 = ReplayMemory(int(self.memory_size / 2))
        self.game_memory1 = GameReplayMemory()
        self.game_memory2 = GameReplayMemory()
        self.observed = True

        self.eta_min = eta_min
        self.eta = 0.5
        #self.eta_linear_decay = 10 ** (-6)

        '''
        The discount rate is the parameter that indicates how many actions will be considered in the future to evaluate
        the reward of a given action.
        A value of 0 means the agent only considers the present action, and a value close to 1 means the agent
        considers actions very far in the future.
        '''
        self.discount_rate = discount_rate

        self.epsilon_min = eps_min
        self.epsilon = 0.5
        #self.epsilon_decay = 0.9995
        self.epsilon_linear_decay = 10**(-5)

        self.training_gap_M = 0

    def start(self, fn = ""):
        if (fn != ""):
            self.load_model(fn)
        else:
            self.model = self.cnn_model()
            self.target_model = self.cnn_model()

    def cnn_model(self):
        """
        Creates a CNN network with two convolutional layers followed by two fully connected layers.

        :param X_state: Placeholder for the state of the game
        :param name: Name of the network (actor or critic)
        :return : The output (logits) layer and the trainable variables
        """

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(32, (7, 7), activation = 'relu', strides=4, padding='SAME', input_shape=(INPUT_HEIGHT, INPUT_WIDTH, CHANNELS)))
        model.add(tf.keras.layers.Conv2D(64, (5, 5), activation = 'relu', strides=2, padding='SAME'))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation = 'relu', strides=2, padding='SAME'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation = 'relu'))
        model.add(tf.keras.layers.Dense(N_OUTPUTS, activation = 'linear'))

        opt = tf.keras.optimizers.Adam(lr = self.learning_rate)
        model.compile(optimizer = opt, loss = self.custom_loss_function)

        return model

    def custom_loss_function(self, y_actual, y_predict):
        q_value = tf.reduce_sum(y_predict * tf.one_hot(self.actions, N_OUTPUTS), axis=1, keepdims=True)
        error = y_actual - q_value
        loss = tf.reduce_mean(tf.square(error))
        return loss

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, fn):
        self.model.save(fn)

    def load_model(self, fn):
        self.model = tf.keras.models.load_model(fn, custom_objects = {'custom_loss_function' : self.custom_loss_function})
        self.target_model = tf.keras.models.load_model(fn, custom_objects = {'custom_loss_function' : self.custom_loss_function})
        print("load model successfully")

    def training_gap_func(self, length):
        if (length <= k):
            return 6
        else:
            return np.ceil(p * length + q)

    def time_out(self, length, steps):
        if (steps >= np.ceil(0.7*length +10)):
            return True
        else:
            return False

    def remember(self, cur_state, action, reward, new_state, done, eat_apple, length, steps):

        if (eat_apple):
            self.training_gap_M = self.training_gap_func(length)
            self.training_gap_M -= 1
            return

        if (self.training_gap_M != 0):
            self.training_gap_M -= 1
            return

        if (reward >= REWARD_THRES):
            self.game_memory1.append([cur_state, action, reward, new_state, done])
        else:
            self.game_memory2.append([cur_state, action, reward, new_state, done])

        if (self.time_out(length, steps)):
            punishment = -0.5/length

            j = self.game_memory2.length-1
            while(j>=0):
                self.game_memory2.buf[j][2] += punishment
                self.game_memory2.buf[j][2] = min(self.game_memory2.buf[j][2], -1)
                j -= 1

            i = self.game_memory1.length-1
            remove_index = []
            while(i>=0):
                self.game_memory1.buf[i][2] += punishment
                self.game_memory1.buf[i][2] = min(self.game_memory1.buf[i][2], -1)

                if (self.game_memory1.buf[i][2] <= REWARD_THRES):
                    self.game_memory2.append(self.game_memory1.buf[i])
                    remove_index.append(i)
                i -= 1

            for k in remove_index:
                self.game_memory1.buf.pop(k)
                self.game_memory1.length -= 1

    def finish_game(self, start_decay):
        i = self.game_memory1.length-1
        while(i>=0):
            self.memory1.append(self.game_memory1.buf[i])
            i -= 1
        self.game_memory1.reset()

        j = self.game_memory2.length-1
        while(j>=0):
            self.memory2.append(self.game_memory2.buf[j])
            j -= 1
        self.game_memory2.reset()

        if start_decay:
            self.observed = False
            self.epsilon -= self.epsilon_linear_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

            # self.eta -= self.eta_linear_decay
            # self.eta = max(self.eta, self.eta_min)

    def act(self, cur_state):
        """
        :param cur_state: Current state of the game
        :param step: Training step
        :return: Action selected by the agent
        """

        if (self.observed or np.random.random() < self.epsilon):
            return np.random.randint(N_OUTPUTS)  # Random action
        return np.argmax(self.model.predict(np.array([cur_state]))[0])

    def predict(self, cur_state):
        return np.argmax(self.model.predict(np.array([cur_state]))[0])

    def replay(self, batch_size=32):
        states, actions, rewards, new_states, dones = self.sample_memories(batch_size)
        self.actions = actions
        next_q_values = self.target_model.predict(new_states)
        max_next_q_values = np.max(next_q_values, axis=1)
        targets = rewards + np.multiply(1-dones, self.discount_rate * max_next_q_values)
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

    def sample_memories(self, batch_size=32):
        """
        Extracts memories from the agent's memory.
        Credits goes to https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb.

        :param batch_size: Size of the batch that we extract form the memory
        :return: State, action, reward, next_state, and done values as np.arrays
        """

        p1=0
        if (self.game_memory1.length+self.memory1.length > 0):
            p1 = self.memory1.length/(self.game_memory1.length+self.memory1.length)

        p2=0
        if (self.game_memory2.length+self.memory2.length > 0):
            p2 = self.memory2.length/(self.game_memory2.length+self.memory2.length)

        batch_size1 = int(np.round(self.eta * batch_size))
        batch_size2 = batch_size - batch_size1

        mem1_batch_size = sum([random.random() < p1 for i in range(batch_size1)])
        game_mem1_batch_size = batch_size1 - mem1_batch_size

        mem2_batch_size = sum([random.random() < p2 for i in range(batch_size2)])
        game_mem2_batch_size = batch_size2 - mem2_batch_size

        cols = [[], [], [], [], []]  # state, action, reward, next_state, done

        # get experiences from memory pool 1
        sample_from_mem1 = self.memory1.sample(mem1_batch_size, with_replacement=False)
        if (sample_from_mem1 is not None):
            for memory in sample_from_mem1:
                for col, value in zip(cols, memory):
                    col.append(value)

        # get experiences from memory pool 2
        sample_from_mem2 = self.memory2.sample(mem2_batch_size, with_replacement=False)
        if (sample_from_mem2 is not None):
            for memory in self.memory2.sample(mem2_batch_size):
                for col, value in zip(cols, memory):
                    col.append(value)

        # get experiences from current game memory pool 1
        sample_from_game_mem1 = self.game_memory1.sample(game_mem1_batch_size, with_replacement=False)
        if (sample_from_game_mem1 is not None):
            for memory in sample_from_game_mem1:
                for col, value in zip(cols, memory):
                    col.append(value)

        # get experiences from current game memory pool 2
        sample_from_game_mem2 = self.game_memory2.sample(game_mem2_batch_size, with_replacement=False)
        if (sample_from_game_mem2 is not None):
            for memory in sample_from_game_mem2:
                for col, value in zip(cols, memory):
                    col.append(value)

        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1), cols[3], cols[4].reshape(-1)
