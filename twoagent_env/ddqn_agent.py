from utils import *
import random
import numpy as np

from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Dense, Conv2D, Flatten, Input, Reshape, Lambda, Add, RepeatVector
from keras.models import Sequential, Model
from collections import deque


class DDQNAgent(RLDebugger):
    def __init__(self, observation_space, action_space):
        RLDebugger.__init__(self)
        # get size of state and action
        self.state_size = observation_space[0]
        self.action_size = action_space
        # hyper parameters
        self.learning_rate = .00001
        self.model = self.build_model()
        self.target_model = self.model
        self.gamma = 0.999
        self.epsilon_max = 1.
        self.epsilon = 1.
        self.t = 0
        self.epsilon_min = 0.1
        self.n_first_exploration_steps = 800
        self.epsilon_decay_len = 500000
        self.batch_size = 8
        self.train_start = 16
        # create replay memory using deque
        self.memory = deque(maxlen=100000)
        self.target_model = self.build_model(trainable=False)

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self, trainable=True):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu', trainable=trainable))
        model.add(Dense(32, activation='relu', trainable=trainable))
        model.add(Dense(self.action_size, activation='relu', trainable=trainable))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

    # get action from model using greedy policy
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    # decay epsilon
    def update_epsilon(self):
        self.t += 1
        self.epsilon = self.epsilon_min + max(0., (self.epsilon_max - self.epsilon_min) *
                            (self.epsilon_decay_len - max(0.,
                                     self.t - self.n_first_exploration_steps)) / self.epsilon_decay_len)

    # train the target network on the selected action and transition
    def train_model(self, action, state, next_state, reward, done):

        # save sample <s,a,r,s'> to the replay memory
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) >= self.train_start:
            states, actions, rewards, next_states, dones = self.create_minibatch()

            targets = self.model.predict(states)
            target_values = self.target_model.predict(next_states)

            for i in range(self.batch_size):
                # Approx Q Learning
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + self.gamma * (np.amax(target_values[i]))

            # and do the model fit!
            loss = self.model.fit(states, targets, verbose=0).history['loss'][0]

            for i in range(self.batch_size):
                self.record(actions[i], states[i], targets[i], target_values[i], loss / self.batch_size, rewards[i])

    def create_minibatch(self):
        # pick samples randomly from replay memory (using batch_size)

        batch_size = min(self.batch_size, len(self.memory))
        samples = random.sample(self.memory, batch_size)

        states = np.array([_[0][0] for _ in samples])
        actions = np.array([_[1] for _ in samples])
        rewards = np.array([_[2] for _ in samples])
        next_states = np.array([_[3][0] for _ in samples])
        dones = np.array([_[4] for _ in samples])

        return (states, actions, rewards, next_states, dones)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())