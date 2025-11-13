# GOAL: reach 500 for total reward

import numpy as np
import pandas as pd
import gymnasium as gym

from neural_network import NeuralNetwork, ReplayBuffer, TrainingLoop
# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode=None)

n_in = env.observation_space.shape[0]   # 4 for CartPole
n_out = env.action_space.n               # 2 actions: left/right
n_hidden = 32
gamma = 0.99
l_rate = 0.001
buffer_size = 5000
batch_size = 64

# Initialize the trainer object
trainer = TrainingLoop(
    n_in=n_in,
    n_out=n_out,
    n_hidden=n_hidden,
    gamma=gamma,
    l_rate=l_rate,
    target_update_freq=20,
)


def epsilon_greedy_action(state, epsilon=1.0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_out)       # random action
    else:
        q_values = trainer.nn.predict(state.reshape(1, -1))
        return np.argmax(q_values)            # greedy action

num_episodes = 300
epsilon = 1.0

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # select action
        action = epsilon_greedy_action(state, epsilon)
        
        # step in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # store transition
        trainer.replay.add(state.reshape(1,-1), action, reward, next_state.reshape(1,-1), done)
        
        # train if enough samples
        if len(trainer.replay.storage) >= 500: #cannot be >= batch_size (32) too early to pick randomly
            trainer.train_step(batch_size)
        
        state = next_state
        total_reward += reward
    
    if episode % 10 == 0:
        print(f"Episode {episode}, total reward: {total_reward}")
    if total_reward >= 500:
        print(f"\n🎉 Goal reached at episode {episode} with total reward = {total_reward}")
        print("\nWeights and biases of the trained network:")
        print("Layer 1 (input → hidden) weights:\n", trainer.nn.w1)
        print("Layer 1 (input → hidden) biases:\n", trainer.nn.b1)
        print("\nLayer 2 (hidden → output) weights:\n", trainer.nn.w2)
        print("Layer 2 (hidden → output) biases:\n", trainer.nn.b2)
        break
    
    epsilon = max(0.05, epsilon * 0.99)
