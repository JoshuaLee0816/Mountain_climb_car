Build environment by using the module of openAI.

import gymnaism as gym
env = gym.make("Name", "render_mode)
observation, info = env.reset(seed = ?)

for _ in range(1000):
    action = env.action_space.sample() # This is where to insert the policy I created
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
