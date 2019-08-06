# Report - Continuous Control with Deep Deterministic Policy Gradients

The work desribed in this report was completed as part of the submission for the Deep Reinforcement Learning nanodegree by Udacity.


## Introduction

In this project we use a modification of the Deep Deterministic Policy Gradient algortihm (DDPG) from [DDPG] to simulataneuous train 20 Reacher agents in a Unity environment. We made use of the implementation of DDPG provided in the Deep Reinforcement Learning course repository [DRLGIT], and, following the guidelines in the benchmark solution,  modifieid the code to work for multiple agents.

## Implementation Details

### Modifications to use with multiple agents
The DDPG algorithm from [DDPG] is shown below.

![DDPG Algorithm](DDG_Algorithm.png)
Format: ![DDPG Algorithm](url)

In this project we made use of the implementation of DDPG provided in the Deep Reinforcement Learning course repository [DRLGIT] and modifieid the code to work for multiple agents.


In the modified implementation the replay buffer is shared between all 33 agents. In the step method (that is called every time step) we loop through each agent and add the most recently collected experience to the buffer.

```python

for i in range (self.num_agents):
  state,action,reward,next_state,done = multi_states[i,:], multi_actions[i,:],multi_rewards[i],multi_next_states[i,:],           multi_done[i]
  self.memory.add(state, action, reward, next_state, done)
```

One of the issues we face when collecting experience from multiple agents is that the learning algortihm can become very unstable, particularly if we make an update for every agent at every time step. Following the recomenations in the benchmark solution we instead only make 10 updates to the networks every 20 timesteps.

```python
if time_step%20== 0 and len(self.memory) > BATCH_SIZE:
    for j in range(10):
        experiences = self.memory.sample()
        self.learn(experiences, GAMMA)
```

Another recomended step we followed to reduce instability was to clip the gradient of the  crtic network.

```python
self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
```

Another difference from the provided code is that we removed the clipping of each action. This was originally a temporary measure because of uncertainty around the best values to clip to in a new environment. As learning was stable without clipping the actions it was decided that no action clipping was necessary.



### Network Architecture

The network architecture is unchanged from the provided code and is desribed below.

The actor network maps states to actions with the following simple architecture.

* A fully connected layer with 256 dimension followed by Relu activation

* An output layer to the action dimension follwed by tanh activation.


The critic network maps the state-action space to a single value using the following architecture;

* The 33 dimensional state is fed through a fully connected layer of 256 dimensions followed by Leaky Relu activation.

* The 256 dimesional output is combined with the 4 dimensional action and fed into a fully connected layer of 256 dimeniosn followed by Leaky Reul activation.

* A 128 dimensional fully connected layer followed by Leaky Relue

* A 1 dimensional output layer.




### Hyperparameters

The hyperparameters used are listed below. These were kept the same as the provided code except that we; 

* Decreased the buffer size. This was originally e^6 and was reduced as an attempt to speed up training.
* Increased the batch size. This was done on the recommendation of user Max G in the course forum.
* Set learning rate of the critic to be equal to that of the actor.
* Did not use weight decay.


HyperParameter | Description | Value
------------ | ------------- | -------------  
BUFFER_SIZE | Size of the Replay buffer| 2e5
BATCH_SIZE | Size of each minibatch for gradient update| 1024
GAMMA| The discounting factor| 0.99
TAU | Soft update parameter| 1e^-3
LR_ACTOR | Learning rate of the actor | 1e^-4
LR_CRITIC | Learning rate of the critic | 1e^-4
N_TIMESTEP | Number of timesteps between learning from experiences | 20
N_GRAD_UPDATE| Number of updates performed every N_TIMESTEPS | 10
MU | Drift parameter of the OU process used to add noise to the actions | 0
THETA | Speed parameter of the OU process used to add noise to the actions | 0.15
SIGMA | Volatility parameter of the OU process used to add noise to the actions | 0.2


## Results

A single training run was made with the final agent.

The environment is considered solved when the average score of the 33 agents over 100 episodes is maintained at or above 30.

Our agent achieved this after episode 202.

The score (averaged over all agents) for each episode and the average score over the previous 100 episodes is plotted below.

![OnlineTraining](online_training.png)
Format: ![online_training](url)


The unity environment includes tools for visualisation. Below is a short clip of the trained agent(after 400 episodes) in the environment. The full episode has been posted on [YouTube](https://www.youtube.com/watch?v=OseH3sEPzuI)

![](Trained.gif)


## Ideas for Future Work
The current implementation could be improved by considering the following
* Systematic hyper parameter optimisation
* Using batch normalisation
* Reducing the weight applied to noise in the actions as training progressed.
* Introducing clipping to the actions.

It would also be interesting to implement the distributed version of the algorithm described in [D4PG]

## References

[DDPG] - [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

[DRLGIT] - [DDPG implementation from Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)

[D4PG] - [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/pdf/1804.08617.pdf)
