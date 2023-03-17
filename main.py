from fetch_data import initialize_exchange, fetch_historical_data
from trading_env import TradingEnvironment
from dqn_agent import DQNAgent
from collections import deque
from random import sample

def train_agent(agent, env, episodes, buffer_size, batch_size, update_freq):
    replay_buffer = deque(maxlen=buffer_size)

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))

            if len(replay_buffer) >= batch_size:
                minibatch = sample(replay_buffer, batch_size)
                agent.train(minibatch)

            state = next_state
            episode_reward += reward

            if done:
                agent.update_target_net()
                print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}")

            if episode % update_freq == 0:
                agent.update_target_net()

def evaluate_agent(agent, env):
    state = env.reset()
    done = False
    cumulative_profit = 0

    while not done:
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done, _ = env.step(action)
        cumulative_profit += reward
        state = next_state

    return cumulative_profit


if __name__ == "__main__":
    #api_key = 'YOUR_API_KEY'
    #secret_key = 'YOUR_SECRET_KEY'
    #exchange = initialize_exchange(api_key, secret_key)
    exchange = initialize_exchange()

    symbol = 'BTC/USDT'
    timeframe = '15m'
    start_date = '2021-01-01T00:00:00Z'
    end_date = '2021-12-31T23:59:59Z'
    data = fetch_historical_data(exchange, symbol, timeframe, start_date, end_date)

    # Create the trading environment
    env = TradingEnvironment(data, initial_balance=1000, window_size=10)


    # Create and train the DQN agent
    agent = DQNAgent(env)
    train_agent(agent, env, episodes=1000, buffer_size=10000, batch_size=64, update_freq=100)

    # Load evaluation data
    eval_data = ...  # Fetch evaluation data similar to training data

    # Create a new trading environment using the evaluation data
    eval_env = TradingEnvironment(eval_data)

    # Evaluate the agent's performance
    cumulative_profit = evaluate_agent(agent, eval_env)
    print(f"Cumulative profit: {cumulative_profit}")

    # Live trading (once you're satisfied with the agent's performance)
    live_data = ...  # Fetch live data for trading

    # Create a new trading environment using the live data
    live_env = TradingEnvironment(live_data)

    # Use the trained agent to make trading decisions
    action = agent.select_action(live_env.current_state())

    # Execute the trading decision using the CCXT library