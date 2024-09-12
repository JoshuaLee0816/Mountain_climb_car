import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time

env = gym.make('MountainCar-v0') #render_mode = 'human' not shown first
observation, info = env.reset()

# hyperparameters setting, include exponential decay factor
alpha, gamma, epsilon, min_epsilon, epsilon_decay = 0.15, 0.999, 0.02, 0.001, 0.9999
episodes, save_interval = 100000, 1000
success_count = 0
failure_count = 0
q_table_update_count = 0
q_table_update_count_100 = 0

# create q table , the size need to be (position x * possible speed velocity value *  possible action which is 3)
position_bins = np.linspace(-1.2, 0.6, 35)
velocity_bins = np.linspace(-0.7, 0.7, 35)
# linspace converts the consecutive data into discrete data , which names as bins
# initialize q table by zero , by using PyTorch and move it to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_table_path = "q_table.pth"

if os.path.exists(q_table_path):
    q_table = torch.load(q_table_path)
    print("Loaded previous Q-table from file.")
else:
    q_table = torch.zeros((len(position_bins), len(velocity_bins), env.action_space.n))
    print("Initialized new Q-table.")


def discretize_state(state, position_bins, velocity_bins):
    position, velocity = state

    position_idx = np.digitize(position, position_bins) - 1
    velocity_idx = np.digitize(velocity, velocity_bins) - 1

    return position_idx, velocity_idx

#Prepare to plot rewards
rewards_per_episode = []
plt.ion() #Interactive mode on for real-time plotting
#Prepare to plot average
average = []
total_reward_100 = 0

#start time recording
start_time = time.time()

#Epsilon turn back to 0.3 after hitting first 0.1 2000 times
hit_min_epsilon = False
hit_min_epsilon_count = 0

for episode in range(1,episodes+1):
    if epsilon <= 0.05:
        hit_min_epsilon = True

    if hit_min_epsilon:
        hit_min_epsilon_count += 1
        if hit_min_epsilon_count >= 2000:
            epsilon = 0.3
            hit_min_epsilon_count = 0
            hit_min_epsilon = False

    observation, info = env.reset()
    terminated = False
    total_reward = 0

    while not terminated:
        #env.render()

        # Discretize the state
        position_idx, velocity_idx = discretize_state(observation, position_bins, velocity_bins)

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # exploration
        else:
            action = torch.argmax(q_table[position_idx, velocity_idx]).item()  # exploitation

        # Take the next action
        next_observation, reward, terminated, truncation, info = env.step(action)

        #reward mechanism define
        if next_observation[0] >= 0.5:
            reward = 100
        elif next_observation[0] >= 0.46 or next_observation[0] < -1.15:
            reward = 5
        elif next_observation[0] >= 0.3:
            reward = 2
        elif next_observation[0] >= 0 or next_observation[0] < -0.95:
            reward = 0.2
        else:
            reward = -1 #based on time step

        if abs(next_observation[1]) <= 0.01:
            reward = -2

        next_position_idx, next_velocity_idx = discretize_state(next_observation, position_bins, velocity_bins)

        #store old q values for comparison
        old_q_value = q_table[position_idx, velocity_idx, action].clone()

        # Q_value updates
        best_next_action = torch.argmax(q_table[next_position_idx, next_velocity_idx]).item()
        q_table[position_idx, velocity_idx, action] = q_table[position_idx, velocity_idx, action] + alpha * (
                    reward + gamma * q_table[next_position_idx, next_velocity_idx, best_next_action] - q_table[
                position_idx, velocity_idx, action])

        #Compare old Q value with the updated one to track updates
        if abs(old_q_value - q_table[position_idx, velocity_idx, action]) > 1e-6:
            q_table_update_count += 1

        total_reward = total_reward + reward
        # Update observation
        observation = next_observation

        if terminated or truncation:
            rewards_per_episode.append(total_reward)
            total_reward_100 += total_reward

            if total_reward < -590:
                failure_count += 1

            if q_table_update_count >= 190: #Set that if update count >= 190, means it is not converge yet.
                q_table_update_count_100 += 1
                q_table_update_count = 0

            # Check if the episode was successful (if car reached the goal)
            if observation[0] >= 0.5:
                success_count += 1

            if episode % 100 == 0:
                avg_reward = total_reward_100 /100
                average.append(avg_reward)
                print("episode: ", episode, "  ,total_reward: ", int(total_reward), "alpha: ", round(alpha,2), "epsilon: ", round(epsilon, 3), "failure_count: ", failure_count, "success_count: ",success_count, "update_count: ", q_table_update_count_100)
                success_count = 0
                failure_count = 0
                total_reward_100 = 0
                q_table_update_count_100 = 0

            terminated = True

    #Epsilond decay
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    #Save the Q-table periodically
    if episode % save_interval == 0 and episode != 0:
        torch.save(q_table, q_table_path)
        print(f"Q-table saved at episode {episode}.")

    #Update the plot of reward
    if episode % 100 == 0:
        plt.figure(1)
        plt.clf()
        plt.plot(rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.pause(0.01)

    #Update the plot of average
    if episode % 100 == 0:
        plt.figure(2)
        plt.clf()
        plt.plot(average)
        plt.xlabel('Episode(x100)')
        plt.ylabel('Average')
        plt.title('Average per Episode')

        plt.tight_layout() #Automatically adjust layout to prevent overlap
        plt.pause(0.01)


# Record the end time and calculate total time taken
end_time = time.time()
total_time = end_time - start_time

env.close()

#Final save after all episodes
torch.save(q_table, q_table_path)
print(f"Final Q-table saved after {episodes} episodes.")
print(f"Total training time: {total_time // 3600} hours, {(total_time % 3600) // 60} minutes, {int(total_time % 60)} seconds")

#Keep the plot open after training
plt.ioff() # Turn off interactive mode
plt.show()
