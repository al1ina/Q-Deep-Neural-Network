
## Q-Deep Learning Neural Network
This project implements a Deep Q-Learning solution to the classic CartPole reinforcement-learning problem — fully from scratch, without using frameworks like TensorFlow or PyTorch.
It includes:
* A non-neural-network baseline using raw Q-Learning
* A custom neural network implementation trained to approximate the Q-function
* Core RL mechanisms such as epsilon-greedy exploration, replay buffers, mini-batch updates, and the Adam optimizer
* Manual implementation of forward pass, backpropagation, and activation functions
* The goal of this repository is to clearly demonstrate how Q-Learning transitions into Deep Q-Learning using only basic Python and NumPy.
## What is the CartPole Problem?
The CartPole problem is a classic control challenge in reinforcement learning.
A pole is attached to a cart that can move left or right along a track. The system is inherently unstable — gravity constantly tries to pull the pole down.
The agent's objective is to apply left or right forces to keep the pole balanced upright for as long as possible.
At each time-step, the agent receives a state vector consisting of:
* Cart position
* Cart velocity
* Pole angle
* Pole angular velocity

Based on these states, the agent must choose one of two actions:
* Move the cart left
* Move the cart right

For every step the pole remains upright, the agent receives a reward of +1.
## Features
In this repository you will find a q_learning.py file, in it contains the CartPole non-neural network problem solution, you can run it and see the average reward. In the other files, there is a neural_network.py file which contains a class based neural network python solution to the problem. It contains classes like NeuralNetwork, TrainingLoop and ReplayBuffer. This neural network implements mini-batch updates where it takes select Q-values and uses them to calculate the next best weights and so on. The file also implements the raw math behind back propagation, forward pass, the sigmoid activation function, epsilon greedy and the adam optimizer.



