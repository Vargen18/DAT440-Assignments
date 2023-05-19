import numpy as np

class Agent(object): #Q-learning agent
    def __init__(self, state_space, action_space, alpha=0.3, epsilon=0.05, gamma=0.95):
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
            if type(observation) != int and type(observation) != np.int32: #for some reason, first state is returned as (0, {'prob': 1}), so we must take the first value in the tuple. RiverSwim uses np.int32 instead of int, hence the and. 
                tmp_state = observation[0]
            else: 
                tmp_state = observation
            tmp_action = np.random.randint(self.action_space) #random action
            if np.random.uniform() > self.epsilon:
                tmp_action = np.random.choice(np.where(self.Q[self.state,:] == self.Q[self.state,:].max())[0]) #greedy action, break ties randomly
        
            E = 0
            E += (1 - self.epsilon)*self.Q[tmp_state,tmp_action]
            num = np.sum(self.Q[tmp_state,:] < np.max(self.Q[tmp_state,:]))
            for Q in self.Q[tmp_state,:]:
                if Q < self.Q[tmp_state,tmp_action]:
                    E += self.epsilon/num*Q
            self.Q[self.state,self.action] = self.Q[self.state,self.action] + self.alpha*(reward + self.gamma*E- self.Q[self.state,self.action])

    def act(self, observation):
        if type(observation) != int and type(observation) != np.int32: #for some reason, first state is returned as (0, {'prob': 1}), so we must take the first value in the tuple. RiverSwim uses np.int32 instead of int, hence the and. 
            self.state = observation[0]
        else: 
            self.state = observation

        if np.random.uniform() < self.epsilon:
            self.action = np.random.randint(self.action_space) #random action
            
        else: 
            self.action = np.random.choice(np.where(self.Q[self.state,:] == self.Q[self.state,:].max())[0]) #greedy action, break ties randomly
        return self.action

    def get_Q(self):
        return self.Q