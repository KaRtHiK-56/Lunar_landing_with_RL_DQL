# Land your Chandrayaan rover on the moon using Artificial Intelligence

## **1. Introduction**
The Lunar Lander is a classic reinforcement learning (RL) problem where an agent (the lander) must navigate and land safely on the moon's surface. This project implements a Deep Q-Learning (DQL) model to train the agent to achieve optimal control in the Lunar Lander environment using PyTorch.

Deep Q-learning is an extension of the traditional Q-Learning algorithm, which uses deep neural networks to approximate the Q-value function. By using neural networks, the agent can handle high-dimensional state spaces and learn complex policies.

## **2. Problem Description**
The Lunar Lander environment is part of the OpenAI Gym toolkit and is a well-known benchmark for RL algorithms. The goal of the agent is to land the lunar module smoothly at the landing pad. The environment has four discrete actions:
- Do nothing.
- Fire left orientation engine.
- Fire main engine.
- Fire right orientation engine.

The state space consists of 8 variables:
- X and Y positions.
- X and Y velocities.
- Angle and angular velocity.
- Boolean values for left and right leg contact with the ground.

The reward function includes:
- A positive reward for successful landing.
- A negative reward for crashing or moving away from the landing pad.
- Smaller penalties for using fuel (firing the engines).

## **3. Deep Q-Learning Overview**
Deep Q-Learning uses a deep neural network to approximate the Q-value function \( Q(s, a) \). The goal is to estimate the expected future rewards given a state-action pair. Unlike traditional Q-Learning, where a lookup table is used, DQL generalizes across large or continuous state spaces.

### **Key Concepts:**
- **Experience Replay:** Stores agent experiences (state, action, reward, next state) in a replay buffer, enabling the model to learn from past experiences, breaking the correlation between consecutive samples.
- **Target Network:** A separate network with the same architecture as the Q-network, which is updated periodically and used to compute target Q-values. This stabilizes training.
- **\(\epsilon\)-Greedy Policy:** Balances exploration (trying new actions) and exploitation (choosing the best-known action).

## **4. Implementation Details**
The implementation is done in Python using PyTorch, and the following components are included:

### **4.1. Environment Setup**
The Lunar Lander environment is loaded from OpenAI Gym. The state and action space information is extracted to build the DQN.

### **4.2. Neural Network Architecture**
A simple fully connected neural network with ReLU activations approximates the Q-values. The architecture is as follows:
- Input layer: 8 neurons (corresponding to the state variables).
- Hidden layers: Two fully connected layers with 84,72 neurons each.
- Output layer: 4 neurons (corresponding to the action space).

### **4.3. Training Loop**
The training loop involves:
- Sampling experiences from the replay buffer.
- Calculating the target Q-value using the target network.
- Computing the loss (mean squared error) between the predicted and target Q-values.
- Backpropagating the loss to update the Q-network weights.

The process is repeated over multiple episodes until the agent learns an optimal policy.

### **4.4. Hyperparameters**
Key hyperparameters include:
- Learning rate: 0.001
- Discount factor (\(\gamma\)): 0.995
- Replay buffer size: 100,000
- Batch size: 100
- Exploration rate (\(\epsilon\)) decay schedule
- Target network update frequency: every 100 episodes

## **5. Results and Analysis**
The model is trained over several thousand episodes. Performance metrics include:
- Average reward per episode.
- Success rate (percentage of successful landings).
- Visualization of landing trajectories.

The agent's performance is visualized through reward plots and landing simulations, showing progressive learning and improvement in control.

## **6. Conclusion**
This project demonstrates how Deep Q-Learning can be effectively applied to solve complex control tasks like the Lunar Lander. By leveraging deep learning and experience replay, the model learns an optimal landing strategy through trial and error.

## **7. Future Work**
Potential improvements include:
- Implementing Double DQN to mitigate overestimation bias.
- Incorporating Prioritized Experience Replay for sampling more important experiences.
- Exploring alternative architectures like dueling DQNs for better learning efficiency.

## **8. References**
- Mnih, V., et al. "Playing Atari with Deep Reinforcement Learning." arXiv preprint arXiv:1312.5602 (2013).
- OpenAI Gym Documentation.
- PyTorch Official Documentation.

## Acknowledgements

 - [Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
 - [Pytorch](https://pytorch.org/)


## Authors

- [@Github](https://www.github.com/KaRtHiK-56)
- [@LinkedIn](https://www.linkedin.com/in/l-karthik/)


## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)


## Demo

https://github.com/KaRtHiK-56/Lunar_landing_with_RL_DQL

I have experimented with my different hidden layer neuron hyperparameter nodes and have included videos resulting in multiple landings, please refer/download videos to see the experiments downloads(6).mp4 has efficient landing compared to others.



https://github.com/user-attachments/assets/2f8a3440-5487-464f-8265-7ef25b068082


https://github.com/user-attachments/assets/e10bfa25-a99d-4b20-a367-12db04b0b0dc


https://github.com/user-attachments/assets/b6eb1049-046b-4082-8663-3887d677eb07


https://github.com/user-attachments/assets/322f79a3-70d5-4b42-9767-8c19736f894b


https://github.com/user-attachments/assets/daaee22c-4066-493c-b820-92ad64c174a2


https://github.com/user-attachments/assets/fb828b55-1dba-459f-a2b5-b1747eb8eb04


https://github.com/user-attachments/assets/40e077cd-53aa-4b1b-aeef-bcc1a0685d6f

