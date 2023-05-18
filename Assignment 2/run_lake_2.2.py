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

# For lake we do 3500 episodes
import matplotlib.pyplot as plt
import scipy.stats as st
episode_reward = []
test_num = 2000
for k in range(5):
    reward_by_episode = []
    rewards = []
    action_dim = env.action_space.n
    state_dim = env.observation_space.n

    agent = agentfile.Agent(state_dim, action_dim)

    observation = env.reset()
    i = 0 # we do 2000 tests
    while i < test_num: 
        #env.render()
        action = agent.act(observation) # your agent here (this currently takes random actions) (can be chosen as an argument when calling from terminal, for example python run_experiment.py --agentfile q_agent.py)
        observation, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        agent.observe(observation, reward, done)

        if done:
            observation, info = env.reset() 
            reward_by_episode.append(np.sum(rewards))
            rewards = []
            i += 1
    episode_reward.append(reward_by_episode)

episode_reward = np.array(episode_reward)
episode_expect = np.mean(episode_reward, 0)

low_CI_bound, high_CI_bound = st.t.interval(0.95, test_num -1,
                                            loc=np.mean(episode_reward , 0),
                                            scale=st.sem(episode_reward))
x = np.linspace(0, test_num - 1, num=test_num)
plt.plot(episode_expect, label='reward')
plt.fill_between(x, low_CI_bound, high_CI_bound, alpha=0.5,
                label='confidence interval')
plt.legend()
plt.title(args.env +' '+ args.agentfile.split('/')[-1]+ ' average reward ')
plt.show()

env.close()


# #only for junyu: C:/Users/Dreamsong/anaconda3/python.exe "c:/Users/Dreamsong/Desktop/DAT440-Assignments-main/Assignment 2/run_lake_2.2.py" --agentfile "c:/Users/Dreamsong/Desktop/DAT440-Assignments-main/Assignment 2/q_agent.py"
