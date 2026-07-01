import numpy as np
import gymnasium as gym

def is_inside(x, y, n, m):
    return x >= 0 and x < n and y >= 0 and y < m

class GlaucomaWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env, steps_with_hungry_to_glaucoma:int, steps_glaucoma_level:int):
        """
        steps_with_hungry_to_glaucoma: how much steps the agent will be with hungry before glaucoma begins
        steps_glaucoma_level: how much pixels the glaucoma will take when the agent is hungry
        """
        # env
        self.env = env
        self.num_envs = env.num_envs
        super(GlaucomaWrapper, self).__init__(env)

        # steps heuristic
        self.steps_with_hungry_to_glaucoma = steps_with_hungry_to_glaucoma
        self.steps_with_hungry = np.zeros(self.num_envs)
        self.max_steps_with_hungry  = np.zeros(self.num_envs)
        self.steps_glaucoma_level = steps_glaucoma_level

        # pixel stuffs
        self.pixels = self.generate_spiral(env.observation_space.shape[1], env.observation_space.shape[2])
        self.pixels_rows, self.pixels_cols = zip(*self.pixels)
        self.erased_pixel = np.zeros(self.num_envs)

        self.last_medkits = np.zeros(self.num_envs)
        self.last_poisons = np.zeros(self.num_envs)

    def reset(self, seed=None, options=None):
        self.steps_with_hungry = np.zeros(self.num_envs, dtype=np.uint16)
        self.erased_pixel = np.zeros(self.num_envs, dtype=np.uint16)
        self.last_medkits = np.zeros(self.num_envs, dtype=np.uint16)
        self.last_poisons = np.zeros(self.num_envs, dtype=np.uint16)
        self.max_steps_with_hungry  = np.zeros(self.num_envs, dtype=np.uint16)
        return self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.glaucoma_policy(info)

        ok = truncated | terminated
        info["max_steps_with_hungry"] = np.where(
            ok,
            self.max_steps_with_hungry,
            0 
        )

        return self.erase_pixels(observation), reward, terminated, truncated, info
    
    def glaucoma_policy(self, info):
        info["USER2"] = info["USER2"].astype(np.uint32)

        # POISON POLICY
        new_poisons = info["USER2"]>>10
        poisons_used = new_poisons > self.last_poisons
        self.last_poisons = new_poisons
        self.erased_pixel += poisons_used.astype(np.uint16) * 5 * self.steps_glaucoma_level


        # MEDIKIT POLICY
        new_medikits = info["USER2"]&0b1111111111
        medkit_used = new_medikits > self.last_medkits
        self.last_medkits = new_medikits
        self.steps_with_hungry = np.where(
            medkit_used,
            -1,
            self.steps_with_hungry
        )
        self.erased_pixel = np.where(
            medkit_used,
            0,
            self.erased_pixel
        )
        self.steps_with_hungry += 1

        # HUNGRY POLICY
        glaucoma_mask = self.steps_with_hungry > self.steps_with_hungry_to_glaucoma
        self.erased_pixel = np.where(
            glaucoma_mask,
            self.erased_pixel + self.steps_glaucoma_level,
            self.erased_pixel
        )

        # MAX STEPS WITH HUNGRY LOG
        self.max_steps_with_hungry = np.maximum(
            self.max_steps_with_hungry,
            self.steps_with_hungry
        )


    # def erase_pixels(self, observation):
    #     if self.erased_pixel > 0:
    #         rows = self.pixels_rows[:self.erased_pixel]
    #         cols = self.pixels_cols[:self.erased_pixel]
    #         observation[:, rows, cols] = 0
    #     return observation
        
    def erase_pixels(self, observation):
        N, C, H, W = observation.shape

        # nothing to erase
        if np.all(self.erased_pixel <= 0):
            return observation

        # for each env, erase its first erased_pixel[i] pixels
        for i in range(N):
            k = self.erased_pixel[i]
            if k > 0:
                rows = self.pixels_rows[:k]
                cols = self.pixels_cols[:k]
                observation[i, :, rows, cols] = 0

        return observation

    def generate_spiral(self, n, m):
        x = n//2
        y = x
        # RIGHT, DOWN, LEFT, UP
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        pixels = [(x, y)]
        flag = True
        count = 1
        count_of_count = 0
        direction = 0
        while flag:
            for c in range(count):
                x += dx[direction]
                y += dy[direction]
                if is_inside(x, y, n, m):
                    pixels.append((x, y))
                else:
                    flag = False      
                    break
            count_of_count =  (count_of_count+1)%2
            count += (count_of_count==0)
            direction = (direction+1)%4
        return pixels

def glaucoma_wrapper(start, level):
    return lambda env: GlaucomaWrapper(env, start, level)