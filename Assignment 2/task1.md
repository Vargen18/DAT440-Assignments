# Q-Learning

Q-Learning is an off-policy control method for TD-learning. 
It works by assigning a Q-value to each (state, action)-pair. The Q-value is an estimation of the q(s,a)-value, which is of how good an action is in the given state. (high q-value -> high quality action) 

The algorithm works by first assigning arbitrary Q-values to each (state-action)-pair, and then chooses an action in accordance with the ε-greedy policy (On policy: (1-ε) porbability), until a terminal state is reached.

When an action **a** is chosen in state **s**, the immediate reward **r** is observed, and the agent moves into state **s’**. The value Q(**s**,**a**) is updated according to the following formula:
$$ Q(s,a) \leftarrow Q(s,a) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t , A_t)]$$

The updated Q(s,a) value is thus based on the immediate reward, as well as the highest discounted Q-value of the actions available in state s'.


# Double Q-Learning

As the Q-values are just estimates of the real q-values, there is a risk of maximization bias, ie. shaping the policy to pick the best estimates, when the estimates are wrong and misleading. To counter this, one can use double learning, or more specifically, Double Q-Learning.

In Double Q-learning, two Q-estimates are used, instead of just one. Each time an action is chosen, we update either Q_1 or Q_2 (with a 0.5 probability for each), according to:
$$ Q_1(s,a) \leftarrow Q_1(s,a) + \alpha [R_{t+1} + \gamma \max_a Q_2(S_{t+1}, a) - Q_1(S_t , A_t)]$$
if Q_1 is chosen. 

In the update-function above, Q_1 uses the Q_2-value of the action chosen by the maximization.

Using Double Q-Learning requires more memory than regular Q-learning, but it also converges faster.

# Sarsa
Sarsa (State-Action-Reward-State-Action) is an on-policy control method, where the learning policy is the same as the target policy. 

Like Q-learning, Sarsa uses Q(s,a) as an estimate of the actual q-values. 
First, the Q-values for all state-action pairs are initiated arbitrarily, except for the terminal states, which are assigned Q-values of 0.

Starting in a state s, an action a is chosen according to the policy, and the reward is observed. From the “pending” state s’, action a’ is chosen according to policy. 
Now, the value Q(s,a) is updated using the immediate reward r as well as the discounted Q-value Q(s’,a’). 

As a’ is chosen according to the policy, the policy is updated on the fly.


# Expected SARSA

Expected SARSA is similar to regular Sarsa, but instead of updating Q(s,a) only based on Q(s’,a’),  we update according the expected value of Q(s’, A), ie by using the following formula:

e-Sarsa excels in applications where there might be a large stochastic variance, ie. states with large action spaces. One drawback to e-Sarsa is that it requires more resources than regular sarsa, much like D-QL requires more resources than QL


