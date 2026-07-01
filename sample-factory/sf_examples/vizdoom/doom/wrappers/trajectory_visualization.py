import os
import numpy as np

import gymnasium as gym

import matplotlib
matplotlib.use("Agg")
print("Matplotlib backend:", matplotlib.get_backend())
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection

def generate_trajectory_map(sector_lines, agent_trajectory, medikits, poisons, fig_name:str):
    plt.figure(figsize=(8, 8))

    arr = np.array(sector_lines)
    x_min = arr[:, [0, 2]].min()
    x_max = arr[:, [0, 2]].max()
    y_min = arr[:, [1, 3]].min()
    y_max = arr[:, [1, 3]].max()
    margin = 20
    texture = mpimg.imread("./assets/acid_floor.png")
    plt.imshow(
        texture,
        extent=[x_min-margin, x_max+margin, y_min-margin, y_max+margin],
        origin="upper",
        zorder=0
    )

    for line in sector_lines:
        plt.plot([line[0], line[2]], [line[1], line[3]], color="white", linewidth=4, zorder=10)

    if len(medikits) > 0:
        mk = np.array(medikits)
        plt.scatter(
            mk[:, 0],
            mk[:, 1],
            marker="+",
            s=120,
            linewidths=3,
            color="lime",
            zorder=15,
            label="Medkit"
        )

    if len(poisons) > 0:
        ps = np.array(poisons)
        plt.scatter(
            ps[:, 0],
            ps[:, 1],
            marker="x",
            s=100,
            linewidths=3,
            color="magenta",
            zorder=15,
            label="Poison"
        )

    traj = np.array(agent_trajectory)
    if len(traj) > 1:
        # Build line segments
        segments = np.stack([traj[:-1], traj[1:]], axis=1)

        # Normalize time [0, 1]
        t = np.linspace(0, 1, len(segments))

        lc = LineCollection(
            segments,
            cmap=cm.plasma,     # plasma / inferno / turbo
            array=t,
            linewidth=3,
            zorder=20
        )

        plt.gca().add_collection(lc)

        plt.scatter(*traj[0], color="cyan", s=40, zorder=30, label="Start")
        plt.scatter(*traj[-1], color="red", s=40, zorder=30, label="End")

    plt.axis("equal")
    plt.title("Episode Trajectory")
    plt.axis("off")
    plt.grid(False)
    plt.savefig(fig_name)
    plt.close()

class TrajectoryVisualizationWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env, image_directory:str):
        # env
        self.env = env
        super(TrajectoryVisualizationWrapper, self).__init__(env)

        self.image_directory = image_directory

        self.images_count = 0

        os.makedirs(self.image_directory, exist_ok=True)

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            generate_trajectory_map(info["sector_lines"], info["agent_trajectory"], info["medikits"], info["poisons"], f"{self.image_directory}/{self.images_count:06d}")
            self.images_count += 1
            
        return observation, reward, terminated, truncated, info
    