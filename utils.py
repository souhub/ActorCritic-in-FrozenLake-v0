import matplotlib.pyplot as plt
import numpy as np


def show_success_rate(episode_rewards):
    episodes = len(episode_rewards)
    episode_unit = 1000
    rates = np.average(episode_rewards.reshape(
        [episodes//episode_unit, episode_unit]), axis=1)*100
    x = [(i+1)*episode_unit for i in range(episodes//episode_unit)]
    plt.plot(x, rates)
    plt.title(f'Success rate per {episode_unit} episodes')
    plt.xlabel('Episode (th times)')
    plt.ylabel('Success rate (%)')
    plt.savefig("success_rate.png")
