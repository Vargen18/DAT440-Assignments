import matplotlib.pyplot as plt


import re

numbers = re.compile('-?\d+.?\d+')


DDPG_LLC = 'ddpg_llc'
DDPG_MCC = 'ddpg_mcc'
PPO_LLC = 'ppo_llc'
PPO_MMC = 'ppo_mmc'

path = PPO_LLC

with open(path + '/episode_reward1.txt') as f:
    lines = f.readlines()
f.close()


means1 = []
deviations1 = []

for line in lines:
    result = list(map(float, numbers.findall(line)))
    means1.append(result[1])
    deviations1.append(result[2])

with open(path +'/episode_reward2.txt') as f:
    lines = f.readlines()
f.close()


means2 = []
deviations2 = []


for line in lines:
    result = list(map(float, numbers.findall(line)))
    means2.append(result[1])
    deviations2.append(result[2])


with open(path +'/episode_reward3.txt') as f:
    lines = f.readlines()
f.close()

means3 = []
deviations3 = []

for line in lines:
    result = list(map(float, numbers.findall(line)))
    means3.append(result[1])
    deviations3.append(result[2])

print(f'm1: {len(means1)}')
print(f'm2: {len(means2)}')
print(f'm3: {len(means3)}')

xs = [x for x in range(len(means1))]
mean_mean = [(x1+x2+x3)/3 for (x1,x2,x3) in zip(means1,means2,means3) ]


#plt.plot(xs, means, label="run 1")
#plt.plot(xs, means2, label="run 2")
#plt.plot(xs, means3, label="run 3")
plt.plot(xs, mean_mean, label="avg mean reward", color='r')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend(loc="lower right")

plt.show()