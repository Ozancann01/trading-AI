import pickle
from threading import Thread
from dashboard import run_dashboard
import json
from fetch_data import initialize_exchange, fetch_historical_data
from trading_env import TradingEnvironment
from dqn_agent import DQNAgent
from collections import deque
from random import sample
import os
import torch
from queue import Queue


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU (index 0)

# Limit the number of CPU threads used by PyTorch
torch.set_num_threads(4)  # Use 4 CPU threads
trading_data=[]



def evaluate_agent(agent, env, trading_data):
    state = env.reset()
    done = False
    cumulative_profit = 0

    while not done:
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done, _ = env.step(action)
        cumulative_profit += reward
        state = next_state
    # Update the trading_data queue with new trade data
    trading_data[env.current_step] = {"timestamp": env.current_step, "action": action, "balance": env.balance}


    return cumulative_profit


def train_agent(agent, env, episodes, buffer_size, batch_size, update_freq, trading_data):
    replay_buffer = deque(maxlen=buffer_size)

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            print(f"Agent selected action: {action}")
            next_state, reward, done, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            
            trading_data[env.current_step] = {"timestamp": env.current_step, "action": action, "balance": env.balance}



            if len(replay_buffer) >= batch_size:
                minibatch = sample(replay_buffer, batch_size)
                agent.train(minibatch)

            state = next_state
            episode_reward += reward
            print(f"Current episode reward: {episode_reward}")

            if done:
                agent.update_target_net()
                print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}")
                

        if episode % update_freq == 0:
            agent.update_target_net()
            print(f"Target network updated at episode {episode + 1}")

        # Save the model every 10 episodes
        if (episode + 1) % 200 == 0:
            agent.save_model(f"models/model_checkpoint_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")



def save_trade(trading_data):
    with open('trading_data.json', 'w') as f:
        json.dump(trading_data, f)

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

      # Start the dashboard 
    dashboard_thread = Thread(target=run_dashboard, args=(trading_data,), daemon=True)
    dashboard_thread.start()


    # Get input_dim, action_size, and device
    input_dim = env.observation_space.shape[0]
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create and train the DQN agent
    agent = DQNAgent(env, input_dim, action_size, device)

   # Check if you want to load a pre-trained model and epsilon value
    load_model = False  # Set this to True if you want to load a pre-trained model
    if load_model:
        agent.load_model('path/to/load/model.pth')
        with open('agent_epsilon.pkl', 'rb') as f:
            agent.epsilon = pickle.load(f)
    else:
        train_agent(agent, env, episodes=1000, buffer_size=10000, batch_size=64, update_freq=100, trading_data=trading_data)
        # Save the model and epsilon value after training
        agent.save_model('path/to/save/model.pth')
        with open('agent_epsilon.pkl', 'wb') as f:
            pickle.dump(agent.epsilon, f)


    # Load evaluation data
    #eval_data = ...  # Fetch evaluation data similar to training data

    # Create a new trading environment using the evaluation data
    #eval_env = TradingEnvironment(eval_data_with_chart_patterns)
    eval_env = TradingEnvironment()

    # Evaluate the agent's performance
    evaluate_agent(agent, eval_env, trading_data=trading_data)
  #  print(f"Cumulative profit: {cumulative_profit}")

    # Live trading (once you're satisfied with the agent's performance)
    live_data = ...  # Fetch live data for trading

    # Create a new trading environment using the live data
    #live_env = TradingEnvironment(live_data)

    # Use the trained agent to make trading decisions
    #action = agent.select_action(live_env.current_state())

    # Execute the trading decision using the CCXT library
    dashboard_thread.join()