class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance, window_size):
        self.data = data
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.current_step = self.window_size

        self.action_space = spaces.Discrete(3)  # Buy, hold, sell
        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=(self.window_size, 5), dtype=np.float32)

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.done = False
        return self._get_observation()

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']

        if action == 0:  # Buy
            self.balance -= current_price
        elif action == 1:  # Hold
            pass
        elif action == 2:  # Sell
            self.balance += current_price

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        obs = self._get_observation()
        reward = self.balance - self.initial_balance
        done = self.done
        info = {}

        return obs, reward, done, info

    def _get_observation(self):
        obs = self.data.iloc[self.current_step - self.window_size:self.current_step].copy()
        obs.reset_index(drop=True, inplace=True)
        return obs.values

    def render(self, mode='human'):
        pass
