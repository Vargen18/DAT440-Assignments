import numpy as np

class Agent(object): #Q-learning agent
    def __init__(self, state_space, action_space, alpha=0.7, epsilon=0.05, gamma=0.95):
        self.action_space = action_space
        self.state_space = state_space
        self.Q = np.zeros([state_space, action_space])
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state = None
        self.action = None

    def observe(self, observation, reward, done):
        if done:
            self.Q[self.state,self.action] = self.Q[self.state,self.action] + self.alpha*(reward - self.Q[self.state,self.action]) #Ignore future reward, since there is no future
        else:
            action = self.act(observation)
            E = 0
            E += (1 - self.epsilon)*self.Q[observation,action]
            num = np.sum(self.Q[observation,:] < np.max(self.Q[observation,:]))
            for Q in self.Q[observation,:]:
                if Q < self.Q[observation,action]:
                    E += self.epsilon/num*Q
            self.Q[self.state,self.action] = self.Q[self.state,self.action] + self.alpha*(reward + self.gamma*E- self.Q[self.state,self.action])

    def act(self, observation):
        if type(observation) != int: #for some reason, first state is returned as (0, {'prob': 1}), so we must take the first value in the tuple
            self.state = observation[0]
        else: 
            self.state = observation

        if np.random.uniform() < self.epsilon:
            self.action = np.random.randint(self.action_space) #random action
            
        else: 
            self.action = np.random.choice(np.flatnonzero(self.Q[self.state,:] == self.Q[self.state,:].max())) #greedy action, break ties randomly
        
        return self.action
