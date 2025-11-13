import numpy as np
import pandas as pd
import gymnasium as gym
import csv

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode=None)

df = pd.read_csv("cartpole_transitions.csv")

# Initialize variables
num_bins = 10
learning_rate = 0.1
n_actions = 2   # CartPole has 2 actions: left (0), right (1)
n_rows = num_bins ** 4  # 4 state variables

# Extract arrays
states = df[["cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity"]].to_numpy()
actions = df["action"].to_numpy()
rewards = df["reward"].to_numpy()
next_states = df[["next_cart_position", "next_cart_velocity", "next_pole_angle", "next_pole_angular_velocity"]].to_numpy()
dones = df["done"].to_numpy()

# Define ranges for each variable
cart_pos_range = [-2.4, 2.4]
cart_vel_range = [-1, 1]
pole_angle_range = [-12 * np.pi/180, 12 * np.pi/180]  # convert degrees to radians
pole_vel_range = [-2, 2]

# Create the bins, split it into sections
cart_pos_bins = np.linspace(cart_pos_range[0], cart_pos_range[1], num_bins)
cart_vel_bins = np.linspace(cart_vel_range[0], cart_vel_range[1], num_bins)
pole_angle_bins = np.linspace(pole_angle_range[0], pole_angle_range[1], num_bins)
pole_vel_bins = np.linspace(pole_vel_range[0], pole_vel_range[1], num_bins)

q_table = np.zeros((n_rows, n_actions))

# creates/returns indices for each bin
def get_state_index(state):
    cart_pos, cart_vel, pole_angle, pole_vel = state
    # Find bin indices
    bin0 = np.digitize(cart_pos, cart_pos_bins) - 1
    bin1 = np.digitize(cart_vel, cart_vel_bins) - 1
    bin2 = np.digitize(pole_angle, pole_angle_bins) - 1
    bin3 = np.digitize(pole_vel, pole_vel_bins) - 1
    
    index = bin0 + bin1*num_bins + bin2*num_bins**2 + bin3*num_bins**3
    return index
def epsilon_greedy_action(state_index, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Explore: random action
    else:
        return np.argmax(q_table[state_index])  # Exploit: best action from Q-table

# Training parameters
total_reward = 0
episodes = 2000
count = 0 # after every epoch count increases by 1 (episode count)
episode_rewards = []
for epoch in range(episodes): # ten episodes
    #create/reset the enviornment
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        state_index = get_state_index(state)
        # Epsilon-greedy action selection
        if count == 0:
            epsilon = 1
        else:
            epsilon = max(0.1, 1.0 - epoch / 500) # we'll see if this works i guess!
        action = epsilon_greedy_action(state_index, epsilon)

        # so now take the action i.e send it to the enviornment and get the next state and reward
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        total_reward += reward
        next_state_index = get_state_index(next_observation)

        # Q-learning update
        # update table

        best_next_q = np.max(q_table[next_state_index])
        q_table[state_index, action] += learning_rate * (reward + 0.99*best_next_q - q_table[state_index, action])


        best_next_action = np.argmax(q_table[next_state_index])
        
        state = next_observation


        if terminated or truncated:
            count += 1
            done = True
    episode_rewards.append(episode_reward)

# bad average reward <20, good >195
print("Average reward per episode: ", total_reward/episodes)
print("Accuracy:", total_reward/(episodes*200))
print("Final average:", np.mean(episode_rewards))

