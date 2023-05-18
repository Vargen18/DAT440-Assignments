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

# For river we do 10000 step 
import matplotlib.pyplot as plt
import scipy.stats as st
reward_5 = []
test_num = 100000
for k in range(5):
    rewards = []
    action_dim = env.action_space.n
    state_dim = env.observation_space.n

    agent = agentfile.Agent(state_dim, action_dim)

    observation = env.reset()
    for i in range(test_num): 
        #env.render()
        action = agent.act(observation) # your agent here (this currently takes random actions) (can be chosen as an argument when calling from terminal, for example python run_experiment.py --agentfile q_agent.py)
        observation, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        agent.observe(observation, reward, done)

        if done:
            observation, info = env.reset() 

    reward_5.append(rewards)

reward = np.array(reward_5)
expect = np.mean(reward, 0)

low_CI_bound, high_CI_bound = st.t.interval(0.95, test_num -1,
                                            loc=np.mean(reward , 0),
                                            scale=st.sem(reward))
x = np.linspace(0, test_num-1, num=test_num)
plt.plot(expect, label='reward')
plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.5,
                label='confidence interval')
plt.legend()
plt.title(args.env +' '+ args.agentfile.split('/')[-1]+ ' average reward ')
plt.show()

env.close()


# #only for junyu: "c:/Users/Dreamsong/Desktop/DAT440-Assignments-main/Assignment 2/run_river_2.2.py" --agentfile "c:/Users/Dreamsong/Desktop/DAT440-Assignments-main/Assignment 2/sarsa_agent.py"