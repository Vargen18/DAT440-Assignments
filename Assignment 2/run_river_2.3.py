import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="q_agent.py")
parser.add_argument("--env", type=str, help="Environment", default="riverswim:RiverSwim")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)


try:
    env = gym.make(args.env)
    print("Loaded ", args.env)
except:
    file_name, env_name = args.env.split(":")
    gym.envs.register(
        id=env_name + "-v0",
        entry_point=args.env,
    )
    env = gym.make(env_name + "-v0")
    print("Loaded", args.env)


rewards = []
action_dim = env.action_space.n
state_dim = env.observation_space.n

agent = agentfile.Agent(state_dim, action_dim)

observation = env.reset()
for _ in range(100000): 
    #env.render()
    action = agent.act(observation) # your agent here (this currently takes random actions) (can be chosen as an argument when calling from terminal, for example python run_experiment.py --agentfile q_agent.py)
    observation, reward, done, truncated, info = env.step(action)
    rewards.append(reward)
    agent.observe(observation, reward, done)

    if done:
        observation, info = env.reset() 

env.close()

import matplotlib.pyplot as plt


def river_Q(Q):
    col=[]
    for i in range(6):
        col.append("col"+str(i))
    row=[]
    for i in range(1):
        row.append("row"+str(i))
    cell_text = []
    tmp = 0
    for i in range(1):
        cell_i = []
        for j in range(6):
            state = Q[tmp]
            action = np.argmax(state)
            if action == 0:
                cell_i.append("left")
            if action == 1:
                cell_i.append("right")
            tmp += 1
        cell_text.append(cell_i)
    plt.table(cellText=cell_text, 
              colLabels=col, 
                rowLabels=row,
              loc='center', 
              cellLoc='center',
              rowLoc='center')
    plt.axis('off')
    plt.show()
river_Q(agent.Q)
#only for junyu: C:/Users/Dreamsong/anaconda3/python.exe "c:/Users/Dreamsong/Desktop/advanced_ml_assignment-main/DAT440-Assignments/Assignment 2/run_experiment.py" --agentfile "c:/Users/Dreamsong/Desktop/advanced_ml_assignment-main/DAT440-Assignments/Assignment 2/q_agent.py" 


print(agent.get_Q()) #Checking that the result is not completely absurd
