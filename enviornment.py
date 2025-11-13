import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import gymnasium as gym
import csv


# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode=None)

with open("cartpole_transitions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "cart_position", "cart_velocity", "pole_angle", "pole_angular_velocity",  # state
        "action", "reward",  # action and reward
        "next_cart_position", "next_cart_velocity", "next_pole_angle", "next_pole_angular_velocity",  # next_state
        "done"
    ])

    for episode in range(10):  # number of episodes
        state, _ = env.reset() # resets the cartpole environment, each episode starts fresh with the pole upright
        # also returns the initial state (a vector of 4 #'s)
        done = False # flag that says whether the episode is over

        while not done:
            # Pick random action (exploration)
            action = env.action_space.sample() # randomly picks 0 or 1, left or right

            # Step environment
            # env.step(action)  --> pushes the cart
            # terminated --> true if the pole fell over
            # truncated --> true if max steps are reached
            next_state, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated

            # Write transition
            writer.writerow(list(state) + [action, reward] + list(next_state) + [done])

            # Move to next state
            state = next_state # update the state so the cart continues from the last state
 
env.close()