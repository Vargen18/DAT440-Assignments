import numpy as np

class Agent(object): #Q-learning agent
    def __init__(self, state_space, action_space, alpha=0.7, epsilon=0.05, gamma=0.95):
        self.action_space = action_space
        self.state_space = state_space
        self.QA = np.zeros([state_space, action_space])
        self.QB = np.zeros([state_space, action_space])
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state = None
        self.action = None

    def observe(self, observation, reward, done):
        if np.random.uniform() < 0.5: # update QA 
            if done:
                self.QA[self.state,self.action] = self.QA[self.state,self.action] + self.alpha*(reward - self.QA[self.state,self.action]) #Ignore future reward, since there is no future
            else:
                a = np.argmax(self.QA[observation,:])
                self.QA[self.state,self.action] = self.QA[self.state,self.action] + self.alpha*(reward + self.gamma*self.QB[observation,a]- self.QA[self.state,self.action])
        else:                         # update QB 
            if done:
                self.QB[self.state,self.action] = self.QB[self.state,self.action] + self.alpha*(reward - self.QB[self.state,self.action]) #Ignore future reward, since there is no future
            else:
                b = np.argmax(self.QB[observation,:])
                self.QB[self.state,self.action] = self.QB[self.state,self.action] + self.alpha*(reward + self.gamma*self.QA[observation,b]- self.QB[self.state,self.action])
       
    def act(self, observation):
        if type(observation) != int: #for some reason, first state is returned as (0, {'prob': 1}), so we must take the first value in the tuple
            self.state = observation[0]
        else: 
            self.state = observation

        if np.random.uniform() < self.epsilon:
            self.action = np.random.randint(self.action_space) #random action
            
        else:
            avg_Q = (self.QA + self.QB) / 2
            self.action = np.random.choice(np.flatnonzero(avg_Q[self.state,:] == avg_Q[self.state,:].max())) #greedy action, break ties randomly
        
        return self.action