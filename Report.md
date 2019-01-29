[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: ./images/plot_of_rewards.png "plot of rewards"

# Navigation Project Report
## Double Dueling DQN Agent with Prioritized Experience Replay

## 1. Introduction

For this project, we implemented a dueling double DQN with prioritized experience replay agent to solve the environment of the first project of Udacity's Deep Reinforcement Learning Nanodegree.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## 2. Learning Algorithm

In order to solve the environment we implemented a DQN agent with three improvements:

1. Double DQN : where we use a local and target network. We train the local network using predicted Q values (for next states) from the target network then make soft updates to the target network.

2. Dueling DQN : where we divided the output of the model into two parts : value and advantages

3. Prioritized Experience Replay: Here we used a SumTree which is a modified version of [Morvan Zhou' code](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py)
    
## 3. Model Architecture

The DQN agent has a target and local networks having the same architecture implemented in [model.py](model.py):

- 1 fully connected layer of size 32.

- 1 fully connected layer of size 32 "value_fc" and 1 fully connected layer of size 32 "advantage_fc", both connected to the first layer.

- 1 fully connected layer of size 1 "value" connected to "value_fc" and 1 fully connected layer of size 4 (action size) "advantage" connected to "advantage_fc".

- 1 output layer of size 4 (action size) having the following equation: value + (advantage - mean(advantage))

## 4. Hyperparameters Tuning

The hyperparameters where chosen using trial and error.

We used a :

- Buffer size of 100 000 for the experience replay

- Update rate of 4 (Update the agent after every 4 time steps)

- Batch size of 64 for training

- Learning rate of 5 e-4

- Tau coefficient of 0.001 for soft update of target parameters

- Dicount factor Gamma of 0.99

## 5. Results

The agent was able to solve the environment after 817 episodes with an average score of 13.04 for the last 100 episodes.

![plot of rewards][image2]

## 6. Ideas for Future Work

Even though the results where satisfying, we still can improve them by implementing all of the rainbow improvements and tuning the hyperparameters.

Also, the prioritized experience replay implemented [here](per.py) uses SumTree in order to get a better computation speed. However, the agent can not start the training until the tree is populated. Hence, another improvement that could be done is to run the computations needed to update the buffer on GPU.