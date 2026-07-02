import gymnasium as gym

def is_inside(x, y, n, m):
    return x >= 0 and x < n and y >= 0 and y < m

class GlaucomaWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env, steps_with_hungry_to_glaucoma:int, steps_glaucoma_level:int, 
                 blidness_reward:float):
        """
        steps_with_hungry_to_glaucoma: how much steps the agent will be with hungry before glaucoma begins
        steps_glaucoma_level: how much pixels the glaucoma will take when the agent is hungry
        blidness_reward: the reward the agent will win after blidness, the vision after blidness will be reseted
        """
        # env
        self.env = env
        super(GlaucomaWrapper, self).__init__(env)

        # steps heuristic
        self.steps_with_hungry_to_glaucoma = steps_with_hungry_to_glaucoma
        self.steps_with_hungry = 0
        self.max_steps_with_hungry  = 0
        self.steps_glaucoma_level = steps_glaucoma_level

        # blidness reward
        self.blindness_reward = blidness_reward
        self.blind = False

        # pixel stuffs
        self.pixels = self.generate_spiral(env.observation_space.shape[1], env.observation_space.shape[2])
        self.pixels_rows, self.pixels_cols = zip(*self.pixels)
        self.erased_pixel = 0
        self.max_erased_pixel = 0

        self.last_medkits = 0
        self.last_poisons = 0

    def reset(self, seed=None, options=None):
        self.blind = False
        self.steps_with_hungry = 0
        self.erased_pixel = 0
        self.last_medkits = 0
        self.last_poisons = 0
        self.max_erased_pixel = 0
        self.max_steps_with_hungry  = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.glaucoma_policy(info)

        if terminated or truncated:
            info["max_glaucoma_len"] = self.max_erased_pixel
            info["max_steps_with_hungry"] = self.max_steps_with_hungry

        return self.erase_pixels(observation), reward, terminated, truncated, info

    def glaucoma_policy(self, info):
        medkit_used = False 
        if info["MEDIKITS"] > self.last_medkits:
            medkit_used = True
        self.last_medkits = info["MEDIKITS"]

        poison_used = False
        if "POISONS" in info:
            if info["POISONS"] > self.last_poisons:
                poison_used = True
            self.last_poisons = info["POISONS"]

        if poison_used:
            self.erased_pixel += self.steps_glaucoma_level

        if medkit_used:
            self.steps_with_hungry = -1
            self.erased_pixel = 0
        self.steps_with_hungry += 1

        if self.steps_with_hungry > self.steps_with_hungry_to_glaucoma:
            self.erased_pixel += self.steps_glaucoma_level
            if self.erased_pixel > len(self.pixels):
                self.blind = True

        if self.erased_pixel > self.max_erased_pixel:
            self.max_erased_pixel = self.erased_pixel

        if self.steps_with_hungry > self.max_steps_with_hungry:
            self.max_steps_with_hungry = self.steps_with_hungry

    def reward_policy(self, reward):
        if self.blind:
            reward = self.blindness_reward
            self.erased_pixel = 0
        self.blind = False
        return reward

    def erase_pixels(self, observation):
        if self.erased_pixel > 0:
            rows = self.pixels_rows[:self.erased_pixel]
            cols = self.pixels_cols[:self.erased_pixel]
            observation[:, rows, cols] = 0
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
