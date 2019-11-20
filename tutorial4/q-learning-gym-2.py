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

epochs = 10000
num_trials = 20

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

def update(update, alpha, epsilon, gamma):
    env = gym.make('MountainCar-v0')

    max_number_of_steps = 200
    last_time_steps = numpy.ndarray(0)
    n_bins = 10

    number_of_features = env.observation_space.shape[0]
    last_time_steps = numpy.ndarray(0)

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    feature1_bins = pandas.cut([-1.2, 0.6], bins=n_bins, retbins=True)[1][1:-1]
    feature2_bins = pandas.cut([-0.07, 0.07], bins=n_bins, retbins=True)[1][1:-1]

    # The Q-learn algorithm
    qlearn = QLearn(range(env.action_space.n), alpha, epsilon, gamma)
    # alpha = 0.005, gamma = 0.90, epsilon = 0.1

    award_arr = numpy.zeros(epochs)

    for i_episode in range(epochs):
        observation = env.reset()

        feature1, feature2 = observation            
        state = build_state([to_bin(feature1, feature1_bins),
                         to_bin(feature2, feature2_bins)])

        qlearn.epsilon = qlearn.epsilon * 0.999 # added epsilon decay
        cumulated_reward = 0

        for t in range(max_number_of_steps):            
            action = qlearn.chooseAction(state)
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            feature1, feature2 = observation            
            nextState = build_state([to_bin(feature1, feature1_bins),
                             to_bin(feature2, feature2_bins)])

            if (update == 0):
                qlearn.updateQBellman(state, action, reward, nextState)
            elif (update == 1):
                qlearn.updateQConsistent(state, action, reward, nextState)
            else:
                qlearn.updateQRSO(state, action, reward, nextState)

            state = nextState
            cumulated_reward += reward
            if done:
                break
        award_arr[i_episode] = cumulated_reward
        #print("Episode {:d} reward score: {:0.2f}".format(i_episode, cumulated_reward))

    moving_average_reward = numpy.zeros(epochs)
    for i in range(0, epochs):
        moving_average_reward[i] = numpy.mean(award_arr[max(i-1000, 0):(i+1)])
    return moving_average_reward

if __name__ == '__main__':
    alpha = float(sys.argv[1])
    epsilon = float(sys.argv[2])
    gamma = float(sys.argv[3])

    bellman_update_arr = numpy.zeros(epochs)
    for i in range(num_trials):
        bellman_update_arr = bellman_update_arr + update(0, alpha, epsilon, gamma)
    bellman_update_arr = bellman_update_arr / float(num_trials)

    consistent_bellman_update_arr = numpy.zeros(epochs)
    for i in range(num_trials):
        consistent_bellman_update_arr = consistent_bellman_update_arr + update(1, alpha, epsilon, gamma)
    consistent_bellman_update_arr = consistent_bellman_update_arr / float(num_trials)

    rso_update_arr = numpy.zeros(epochs)
    for i in range(num_trials):
        rso_update_arr = rso_update_arr + update(2, alpha, epsilon, gamma)
    rso_update_arr = rso_update_arr / float(num_trials)

    bellman_update_arr = numpy.expand_dims(bellman_update_arr, 1)
    consistent_bellman_update_arr = numpy.expand_dims(consistent_bellman_update_arr, 1)
    rso_update_arr = numpy.expand_dims(rso_update_arr, 1)

    save_file = numpy.hstack((bellman_update_arr, consistent_bellman_update_arr, rso_update_arr))
    for a in range(save_file.shape[0]):
        print(save_file[a, :])
    #save_file_name = "alpha=" + str(alpha) + " epsilon=" + str(epsilon) + " gamma=" + str(gamma)

    # plt.plot(numpy.arange(epochs), bellman_update_arr, label='bellman')
    # plt.plot(numpy.arange(epochs), consistent_bellman_update_arr, label='consistent')
    # plt.plot(numpy.arange(epochs), rso_update_arr, label='rso')
    # plt.legend(loc='upper left')
    # plt.show()

#undo plots
#undo prints that are not to log (tracking episode reward stuff)
#change back all parameters at top to what they should be


