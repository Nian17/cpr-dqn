from utils import RLEnvironment, RLDebugger
import gym
import random
import numpy as np

from keras.optimizers import Adam, RMSprop, SGD
from keras.layers import Dense, Conv2D, Flatten, Input, Reshape, Lambda, Add, RepeatVector
from keras.models import Sequential, Model
from keras import backend as K

class DQNAgent(RLDebugger):
    def __init__(self, observation_space, action_space):
        RLDebugger.__init__(self)
        # get size of state and action
        self.state_size = observation_space[0]
        self.action_size = action_space
        # hyper parameters 
        self.learning_rate = .00025
        self.model = self.build_model()  
        self.target_model = self.model
        self.gamma = 0.999
        
    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self, trainable=True):
        model = Sequential()
        #This is a simple one hidden layer model, thought it should be enough here,
        #it is much easier to train with different achitectures (stack layers, change activation)
        model.add(Dense(32, input_dim=self.state_size, activation='relu', trainable=trainable))
        model.add(Dense(32, activation='relu', trainable=trainable))
        model.add(Dense(self.action_size, activation='linear', trainable=trainable))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary()
        # 1/ You can try different losses. As an logcosh loss is a twice differenciable approximation of Huber loss
        # 2/ From a theoretical perspective Learning rate should decay with time to guarantee convergence 
        return model

    # get action from model using greedy policy. 
    def get_action(self, state):
        q_value = self.model.predict(state)
        best_action = np.argmax(q_value[0]) #The [0] is because keras outputs a set of predictions of size 1
        return best_action

    # train the target network on the selected action and transition
    def train_model(self, action, state, next_state, reward, done):
        target = self.model.predict(state)
        # We use our internal model in order to estimate the V value of the next state 
        target_val = self.target_model.predict(next_state)
        # Q Learning: target values should respect the Bellman's optimality principle
        if done: #We are on a terminal state
            target[0][action] = reward
        else:
            target[0][action] = reward + self.gamma * (np.amax(target_val))

        # and do the model fit!
        loss = self.model.fit(state, target, verbose=0).history['loss'][0]
        self.record(action, state, target, target_val, loss, reward)

class DQNAgentWithExploration(DQNAgent):
    def __init__(self, observation_space, action_space):
        super(DQNAgentWithExploration, self).__init__(observation_space, action_space)
        # exploration schedule parameters 
        self.epsilon_max = 1.
        self.epsilon = 1.
        self.t = 0
        self.epsilon_min = 0.1
        self.n_first_exploration_steps = 500
        self.epsilon_decay_len = 1000000
        # TODO store your additional parameters here 

    # decay epsilon
    def update_epsilon(self):
        # TODO write the code for your decay  
        self.t += 1
        self.epsilon = self.epsilon_min + max(0., (self.epsilon_max-self.epsilon_min) *
                           (self.epsilon_decay_len - max(0., self.t - self.n_first_exploration_steps)) / self.epsilon_decay_len)
        #self.learning_rate = self.learning_rate**np.sqrt(self.t-self.n_first_exploration_steps) if self.t > self.n_first_exploration_steps else self.learning_rate

    # get action from model using greedy policy
    def get_action(self, state):
        #TODO add the exploration 
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

from collections import deque

class DQNAgentWithExplorationAndReplay(DQNAgentWithExploration):
    def __init__(self, observation_space, action_space):
        super(DQNAgentWithExplorationAndReplay, self).__init__(observation_space, action_space)
        self.batch_size = 64
        self.train_start = 128
        # create replay memory using deque
        self.memory = deque(maxlen=1000000)

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

class DoubleDQNAgentWithExplorationAndReplay(DQNAgentWithExplorationAndReplay):
    def __init__(self, observation_space, action_space):
        super(DoubleDQNAgentWithExplorationAndReplay, self).__init__(observation_space, action_space)
        # TODO: initialize a second model
        self.target_model = self.build_model(trainable=False)

    def update_target_model(self):
        # TODO copy weights from the model used for action selection to the model used for computing targets
        self.target_model.set_weights(self.model.get_weights())