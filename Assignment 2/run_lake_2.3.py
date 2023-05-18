import argparse
import gymnasium as gym
import importlib.util
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="q_agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
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
def lake_Q(Q): #Action Space 0: LEFT 1: DOWN 2: RIGHT 3: UP
    col=[]
    for i in range(4):
        col.append("col"+str(i))
    row=[]
    for i in range(4):
        row.append("row"+str(i))

    cell_text = []
    tmp = 0
    for i in range(4):
        cell_i = []
        for j in range(4):
            state = Q[tmp]
            action = np.argmax(state)
            if state[action] == 0:
                cell_i.append("obstacle")
            else:
                if action == 0:
                    cell_i.append("left")
                if action == 1:
                    cell_i.append("down")
                if action == 2:
                    cell_i.append("right")
                if action == 3:
                    cell_i.append("up")
            tmp += 1
        cell_text.append(cell_i)
    cell_text[-1][-1] = 'end'
    plt.table(cellText=cell_text, 
              colLabels=col, 
                rowLabels=row,
              loc='center', 
              cellLoc='center',
              rowLoc='center')
    plt.axis('off')
    plt.show()

    
def lake_heatmaps(Q): #Generates one heatmap for each action for the frozen lake environment. The first heatmap shows value of taking left action for each cell, the second heatmap shows down, then right, finally up. 
    if(Q.shape != (16,4)):
        print("Wrong dimensions of  Q-table! Did you run the correct environment?")
        return None
    fig, axes = plt.subplots(2, 2)
    for i in range(len(Q[0])):
        sublist = list(x[i] for x in Q)
        actions_as_matrix = np.reshape(sublist, [4, 4])
        sns.set()
        sns.heatmap(actions_as_matrix, vmin=np.min(Q), vmax=np.max(Q), ax=axes[i%2,int(i/2)], annot=True) #Use min/max of the whole table as anchor for every heatmap, to make comparisons between them easier
    axes[0,0].set_title("Left")
    axes[1,0].set_title("Down")
    axes[0,1].set_title("Right")
    axes[1,1].set_title("Up")
    plt.show()


np.set_printoptions(formatter={'float_kind':"{:.2f}".format}) #just formatting the output
print(agent.get_Q()) #Checking that the result is not completely absurd
lake_Q(agent.Q)
lake_heatmaps(agent.get_Q())