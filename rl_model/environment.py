import gym
from gym import spaces
import numpy as np

class DiabetesEnv(gym.Env):
    def __init__(self, patient_df):
        super(DiabetesEnv, self).__init__()
        self.df = patient_df
        self.current_step = 0
        self.df.fillna(0, inplace=True)
        self.df['glucose'] = self.df['glucose'] / 400
        self.df['insulin'] = self.df['insulin'] / 10
        self.df['carbs'] = self.df['carbs'] / 150
        self.df['steps'] = self.df['steps'] / 10000
        self.df['heart_rate'] = self.df['heart_rate'] / 200
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row['glucose'], row['carbs'], row['insulin'], row['steps'], row['heart_rate']
        ], dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        glucose = self.df.iloc[self.current_step]['glucose'] * 400
        reward = -abs(glucose - 100)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, {}
