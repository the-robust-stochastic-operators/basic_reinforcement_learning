'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning
Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import gym
import numpy
import random
import pandas
import matplotlib.pyplot as plt
import sys
import time
import os
import csv

epochs = 10000

class QLearn:
    def __init__(self, actions, alpha, epsilon, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def updateQBellman(self, currentState, action, reward, nextState):
        Qvalue = self.q.get((currentState, action), None)
        if Qvalue is None:
            self.q[(currentState, action)] = reward
        else:
            rvalue = reward + self.gamma*max([self.getQ(nextState, a) for a in self.actions])
            self.q[(currentState, action)] = Qvalue + self.alpha * (rvalue - Qvalue)

    def updateQConsistent(self, currentState, action, reward, nextState):
        Qvalue = self.q.get((currentState, action), None)
        if Qvalue is None:
            self.q[(currentState, action)] = reward
        else:
            if currentState != nextState:
                rvalue = reward + self.gamma*max([self.getQ(nextState, a) for a in self.actions])
            else:
                rvalue = reward + self.gamma*Qvalue
            self.q[(currentState, action)] = Qvalue + self.alpha * (rvalue - Qvalue)

    def updateQRSO(self, currentState, action, reward, nextState):
        Qvalue = self.q.get((currentState, action), None)
        if Qvalue is None:
            self.q[(currentState, action)] = reward
        else:
            beta = numpy.random.uniform(0,2)
            bellmanValue = reward + self.gamma*max([self.getQ(nextState, a) for a in self.actions])
            rvalue = bellmanValue - beta*(max([self.getQ(currentState, a) for a in self.actions]) - Qvalue)
            self.q[(currentState, action)] = Qvalue + self.alpha * (rvalue - Qvalue)

    def updateQRSO2(self, currentState, action, reward, nextState):
        Qvalue = self.q.get((currentState, action), None)
        if Qvalue is None:
            self.q[(currentState, action)] = reward
        else:
            beta = 1
            bellmanValue = reward + self.gamma*max([self.getQ(nextState, a) for a in self.actions])
            rvalue = bellmanValue - beta*(max([self.getQ(currentState, a) for a in self.actions]) - Qvalue)
            self.q[(currentState, action)] = Qvalue + self.alpha * (rvalue - Qvalue)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))] 
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

def build_state(features):    
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

def update():
    env = gym.make('MountainCar-v0')

    global alpha 
    global epsilon
    global gamma
    global alpha_decay
    global epsilon_decay
    global update_type
    global num_trials
    max_number_of_steps = 200
    n_bins = 40

    number_of_features = env.observation_space.shape[0]

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    feature1_bins = pandas.cut([-1.2, 0.6], bins=n_bins, retbins=True)[1][1:-1]
    feature2_bins = pandas.cut([-0.07, 0.07], bins=n_bins, retbins=True)[1][1:-1]

    # The Q-learn algorithm
    qlearn = QLearn(range(env.action_space.n), alpha, epsilon, gamma)
    # alpha = 0.005, gamma = 0.90, epsilon = 0.1

    last_time = time.time()
    total_reward = 0
    for j in range(num_trials):
        for i_episode in range(epochs):
            observation = env.reset()

            feature1, feature2 = observation            
            state = build_state([to_bin(feature1, feature1_bins),
                             to_bin(feature2, feature2_bins)])

            if alpha_decay:
                alpha = alpha * 0.999
            if epsilon_decay:
                qlearn.epsilon = qlearn.epsilon * 0.999


            for t in range(max_number_of_steps):            
                action = qlearn.chooseAction(state)
                observation, reward, done, info = env.step(action)

                # Digitize the observation to get a state
                feature1, feature2 = observation            
                nextState = build_state([to_bin(feature1, feature1_bins),
                                 to_bin(feature2, feature2_bins)])

                if (update_type == 0):
                    qlearn.updateQBellman(state, action, reward, nextState)
                elif (update_type == 1):
                    qlearn.updateQConsistent(state, action, reward, nextState)
                elif (update_type == 2):
                    qlearn.updateQRSO(state, action, reward, nextState)
                else:
                    qlearn.updateQRSO2(state, action, reward, nextState)

                state = nextState
                if done:
                    break

        cumulated_reward = 0

        qlearn.epsilon=0
        for i_episode in range(1000):
            observation = env.reset()

            feature1, feature2 = observation            
            state = build_state([to_bin(feature1, feature1_bins),
                             to_bin(feature2, feature2_bins)])

            for t in range(max_number_of_steps):            
                action = qlearn.chooseAction(state)
                observation, reward, done, info = env.step(action)

                # Digitize the observation to get a state
                feature1, feature2 = observation            
                nextState = build_state([to_bin(feature1, feature1_bins),
                                 to_bin(feature2, feature2_bins)])

                state = nextState
                cumulated_reward += reward
                if done:
                    break
        total_reward += (cumulated_reward / float(1000))
        print(time.time() - last_time)
    print(total_reward / float(num_trials))

if __name__ == '__main__':
    global alpha
    global epsilon 
    global gamma 
    global alpha_decay
    global epsilon_decay 
    global num_trials
    global update_type
    alpha = float(sys.argv[1])
    epsilon = float(sys.argv[2])
    gamma = float(sys.argv[3])
    alpha_decay = int(sys.argv[4])
    epsilon_decay = int(sys.argv[5])
    num_trials = int(sys.argv[6])
    update_type = int(sys.argv[7])
    start_time = time.time()

    fn = "alpha = " + str(alpha) + " epsilon = " + str(epsilon) + " gamma = " + str(gamma) + " alpha decay = " + str(alpha_decay) + " epsilon decay = " + str(epsilon_decay) + "num_trials = " + str(num_trials) + ".csv"

    update()
    #save_file_name = "alpha=" + str(alpha) + " epsilon=" + str(epsilon) + " gamma=" + str(gamma)

    # plt.plot(numpy.arange(epochs), bellman_update_arr, label='bellman')
    # plt.plot(numpy.arange(epochs), consistent_bellman_update_arr, label='consistent')
    # plt.plot(numpy.arange(epochs), rso_update_arr, label='rso')
    # plt.legend(loc='upper left')
    # plt.show()

#undo plots
#undo prints that are not to log (tracking episode reward stuff)
#change back all parameters at top to what they should be
