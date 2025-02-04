import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
from gym import spaces
import tqdm
from typing import List
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# 1. Data Handling
def load_stock_data(folder_path):
    """Load stock data from individual csv files and returns dict of dataframes"""
    all_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            symbol = filename[:-4] # remove .csv
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath, parse_dates=['datetime']) # read in csv
            df['Date'] = df['datetime'].dt.date # create a Date column
            df.set_index('Date', inplace=True) #set date as the index
            df.drop(columns=['datetime','barcount','interval'], inplace=True, errors='ignore') # remove uneeded columns, ignore error if not present
            all_data[symbol] = df
    return all_data

def calculate_forward_returns(df, days_forward=1):
  """Calculate simple forward returns using future open prices."""
  df['Forward_Return'] = df['open'].shift(-days_forward) / df['open'] - 1
  df.dropna(inplace=True) # drop rows with no future data
  return df

def create_dataset(dataframes, start_date, end_date, days_forward=1):
    """ Create one pandas dataframe using a dictionary of dataframes."""
    combined_df = pd.concat([df.assign(symbol=symbol) for symbol, df in dataframes.items()], ignore_index=False)
    combined_df = combined_df.reset_index()

    # Convert columns to numeric, handling non-numeric values
    cols_to_convert = ['open', 'high', 'low', 'close', 'adj close', 'volume']
    for col in cols_to_convert:
      combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    combined_df = combined_df.dropna()
    #Convert start and end dates into the correct format before slicing
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    combined_df = combined_df.loc[combined_df['Date'].isin(pd.date_range(start_date, end_date).date)]
    combined_df = combined_df.groupby('symbol', group_keys=True).apply(lambda x: calculate_forward_returns(x, days_forward=days_forward))
    combined_df.reset_index(inplace=True, drop=True) # drop the previous level 0 index as it is not needed
    combined_df = combined_df.set_index(['Date', 'symbol'])
    return combined_df

# 2. Sentiment Analysis (Placeholder)
def get_sentiment_scores(text_data):
    """Placeholder - Integrate with FinGPT later"""
    # This should return a sentiment score
    return np.random.uniform(-1, 1, len(text_data))

# 3. Trading Environment (Custom)
class StockTradingEnv(gym.Env):
    """Custom stock trading environment."""

    def __init__(
        self,
        all_stocks_data,
        start_date,
        end_date,
        initial_amount=1000000,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        reward_scaling=1e-4
        ):
        super().__init__()
        self.all_stocks_data = all_stocks_data
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.tickers = list(self.all_stocks_data.keys())
        self.num_stocks = len(self.tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.data = create_dataset(self.all_stocks_data, start_date, end_date)

        # Define action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)

        # Define observation space: this requires both historical and real-time data
        self.observation_space = spaces.Dict({
            "hlc_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.num_stocks * 2,), dtype=np.float32
            ),
            "llc_state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.num_stocks, 3), dtype=np.float32
            ),
            "stock_prices": spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.num_stocks,), dtype=np.float32
            ),
        })

        self.current_step = 0
        self.max_steps = len(self.data.index.get_level_values(0).unique())  # number of timesteps
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to the beginning of a new episode."""
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        self.holdings = np.zeros(self.num_stocks)  # Initialize holdings to 0
        self.balance = self.initial_amount

        # Initialize with first available prices
        self.last_prices = self.data.loc[
            self.data.index.get_level_values(0).unique()[0]
        ]['open'].values
        self.last_prices = np.array(self.last_prices)

        # Initialize state
        self._update_state()

        return self.state, {}

    def _update_state(self):
        """Update the current state."""
        current_date = self.data.index.get_level_values(0).unique()[self.current_step]
        current_data = self.data.loc[current_date]

        forward_returns = current_data['Forward_Return'].values
        sentiment_scores = get_sentiment_scores([f"News for {ticker}" for ticker in self.tickers])  # placeholder
        self.stock_prices = current_data['open'].values
        self.stock_prices = np.array(self.stock_prices)

        hlc_state = np.concatenate([forward_returns, sentiment_scores])
        llc_states = []
        for i in range(self.num_stocks):
            llc_states.append([self.stock_prices[i], self.holdings[i], self.balance])
        llc_states = np.array(llc_states)

        self.state = {
            "hlc_state": hlc_state.astype(np.float32),
            "llc_state": llc_states.astype(np.float32),
            "stock_prices": self.stock_prices.astype(np.float32),
        }

    def step(self, actions):
        """Take a step in the environment based on provided actions."""
        previous_portfolio_value = self.portfolio_value
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # 1. Get current price from data
        if not done:
          self._update_state()
        else:
          return self.state, 0, done, {}

        # Scale actions
        scaled_actions = np.clip(actions, -1, 1)
        trades = np.zeros(self.num_stocks) # initialise an empty array
        # 2. Calculate trades
        for i, ticker in enumerate(self.tickers):
          price = self.stock_prices[i]
          trade_amount = scaled_actions[i] * self.hmax
          trade_amount = np.round(trade_amount) # make integer values
          trades[i] = trade_amount
          if trade_amount > 0: #Buy stocks
              cost = price * trade_amount * (1 + self.buy_cost_pct)
              if self.balance >= cost:
                self.balance -= cost
                self.holdings[i] += trade_amount
              else: # can't afford this amount
                trades[i] = 0
          elif trade_amount < 0: #sell stocks
              trade_amount = abs(trade_amount)
              if self.holdings[i] >= trade_amount:
                  proceeds = price * trade_amount * (1 - self.sell_cost_pct)
                  self.balance += proceeds
                  self.holdings[i] -= trade_amount
              else:
                 trades[i] = 0

        # Update portfolio value (current total)
        self.portfolio_value = self.balance + np.sum(self.holdings * self.stock_prices)

        # Calculate reward
        reward = (self.portfolio_value - previous_portfolio_value) * self.reward_scaling
        reward = np.clip(reward, -1, 1) #clip to help stabilise the training process

        return self.state, reward, done, {}

# 4. HLC (High-Level Controller)
class HLC_PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HLC_PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class HLC_ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HLC_ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def ppo_update(hlc_policy, hlc_value, optimizer_policy, optimizer_value, states, actions, returns, advantages, clip_param=0.2, entropy_beta=0.01):
    """Standard PPO update step"""
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    returns = torch.tensor(np.array(returns), dtype=torch.float32).to(device)
    advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(device)
    #calculate new policy
    new_actions = hlc_policy(states)
    new_action_probs = torch.sigmoid(new_actions)

    old_action_probs = torch.sigmoid(actions)

    ratio = torch.exp(torch.log(new_action_probs) - torch.log(old_action_probs))
    clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
    surrogate_objective = torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    #entropy regularization - prevent policy from becoming deterministic
    entropy = -(new_action_probs * torch.log(new_action_probs + 1e-8)).mean() # add a small number to avoid log of 0

    #policy gradient update
    policy_loss = - (surrogate_objective - entropy_beta * entropy)
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    #calculate value loss
    values = hlc_value(states).squeeze()
    value_loss = ((values - returns)**2).mean()
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()
    return policy_loss.item(), value_loss.item()

def create_advantage_estimator(rewards, values, gamma=0.99, gae_lambda=0.95):
  """Generalized advantage estimation"""
  advantages = []
  advantage = 0
  for i in reversed(range(len(rewards))):
    delta = rewards[i] + gamma * values[i+1] - values[i] if i < len(rewards) - 1 else rewards[i] - values[i]
    advantage = delta + gamma * gae_lambda * advantage
    advantages.insert(0, advantage) # insert at beginning so we have correct order for training
  return advantages

# 5. LLC (Low-Level Controller)
class LLC_ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LLC_ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class LLC_CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(LLC_CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def ddpg_update(actor, critic, target_actor, target_critic, optimizer_actor, optimizer_critic, replay_buffer, batch_size, discount_factor=0.99, tau=0.005):
    """Standard DDPG update step"""
    if len(replay_buffer) < batch_size: return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)

    target_actions = target_actor(next_states)
    target_q_values = target_critic(next_states, target_actions).squeeze()
    expected_q_values = rewards + (discount_factor * target_q_values * (1 - dones))

    #critic loss
    q_values = critic(states, actions).squeeze()
    critic_loss = ((q_values - expected_q_values)**2).mean()
    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

    #actor loss
    policy_actions = actor(states)
    actor_loss = -critic(states, policy_actions).mean()
    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    #update target networks
    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    return critic_loss.item(), actor_loss.item()

# 6. HRT Agent
class ReplayBuffer:
    """A simple replay buffer."""
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
      batch = np.random.choice(len(self.buffer), batch_size, replace=False)
      states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
      return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class HRTAgent:
    def __init__(self, env, hlc_config, llc_config, device):
        # Create High Level Controller (HLC)
        self.num_stocks = env.num_stocks
        self.hlc_policy = HLC_PolicyNetwork(hlc_config["input_size"], hlc_config["hidden_size"], hlc_config["output_size"]).to(device)
        self.hlc_value = HLC_ValueNetwork(hlc_config["input_size"], hlc_config["hidden_size"]).to(device)
        self.hlc_optimizer_policy = optim.Adam(self.hlc_policy.parameters(), lr=hlc_config["learning_rate_policy"])
        self.hlc_optimizer_value = optim.Adam(self.hlc_value.parameters(), lr=hlc_config["learning_rate_value"])
        # Create Low Level Controller (LLC)
        self.llc_actor = LLC_ActorNetwork(llc_config["input_size"], llc_config["hidden_size"], llc_config["output_size"]).to(device)
        self.llc_critic = LLC_CriticNetwork(llc_config["input_size"], llc_config["hidden_size"], llc_config["output_size"]).to(device)
        self.llc_target_actor = LLC_ActorNetwork(llc_config["input_size"], llc_config["hidden_size"], llc_config["output_size"]).to(device)
        self.llc_target_critic = LLC_CriticNetwork(llc_config["input_size"], llc_config["hidden_size"], llc_config["output_size"]).to(device)
        self.llc_optimizer_actor = optim.Adam(self.llc_actor.parameters(), lr=llc_config["learning_rate_actor"])
        self.llc_optimizer_critic = optim.Adam(self.llc_critic.parameters(), lr=llc_config["learning_rate_critic"])
        self.llc_replay_buffer = ReplayBuffer(llc_config["replay_buffer_size"])
        # Initialize Target Networks
        self.llc_target_actor.load_state_dict(self.llc_actor.state_dict())
        self.llc_target_critic.load_state_dict(self.llc_critic.state_dict())

        self.device = device
        self.env = env
        self.current_step = 0

    def select_action(self, state, exploration_noise=None):
      with torch.no_grad():
        hlc_state = torch.tensor(state["hlc_state"], dtype=torch.float32).to(self.device)
        hlc_action = self.hlc_policy(hlc_state).cpu().numpy()
        llc_actions = []
        for i in range(self.num_stocks):
          if hlc_action[i] > 0.5: #buy action
            llc_state = torch.tensor(state["llc_state"][i], dtype=torch.float32).to(self.device)
            action = self.llc_actor(llc_state)
            if exploration_noise is not None:
              noise = torch.normal(torch.zeros(action.shape), exploration_noise).to(self.device)
              action = (action + noise).clamp(-1, 1)
            llc_actions.append(action.cpu().numpy())
          elif hlc_action[i] < -0.5: #sell action
              llc_state = torch.tensor(state["llc_state"][i], dtype=torch.float32).to(self.device)
              action = self.llc_actor(llc_state)
              if exploration_noise is not None:
                noise = torch.normal(torch.zeros(action.shape), exploration_noise).to(self.device)
                action = (action + noise).clamp(-1, 1)
              llc_actions.append(action.cpu().numpy())
          else: # hold action
              llc_actions.append(np.array(0))

        return hlc_action, llc_actions


    def calculate_hlc_reward(self, last_state, next_state, hlc_action, llc_reward, at, price_changes):
      """Calculate alignment and overall reward"""
      #calculate alignment reward
      alignment_rewards = []
      for i in range(self.num_stocks):
        action_i = 1 if hlc_action[i] > 0.5 else -1 if hlc_action[i] < -0.5 else 0
        price_change_sign = np.sign(price_changes[i])
        alignment_reward = np.sign(action_i) * price_change_sign if action_i != 0 else 0
        alignment_rewards.append(alignment_reward)
      alignment_rewards = np.array(alignment_rewards)

      return at * np.sum(alignment_rewards) + (1 - at) * llc_reward

    def phased_alternating_train(self, num_episodes, env, training_config, writer, at_initial_value = 1, at_decay = 0.001):
      """ Train using the phased alternating algorithm."""

      self.current_step = 0
      for episode in tqdm.tqdm(range(num_episodes), desc="Training Progress", colour="green"):
          at = at_initial_value * np.exp(-at_decay * episode)
          # Phase 1: HLC Training
          if episode < training_config["hlc_episodes"]:
            print(f"Phase 1: HLC Training, Episode {episode}")
            states, actions, rewards, log_probs, values = self.train_hlc(env, training_config, writer, at, hlc_only=True)
            for step, reward in enumerate(rewards):
              writer.add_scalar('Rewards/HLC_only', reward, self.current_step + step)
            self.current_step += len(states)
          # Phase 2: LLC Training
          elif episode >= training_config["hlc_episodes"] and episode < training_config["llc_episodes"]:
            print(f"Phase 2: LLC Training, Episode {episode}")
            self.train_llc(env, training_config, writer)
          # Phase 3: Alternating Training
          else:
            print(f"Phase 3: Alternating Training, Episode {episode}")
            states, actions, rewards, log_probs, values = self.train_hlc(env, training_config, writer, at, hlc_only=False)
            for step, reward in enumerate(rewards):
              writer.add_scalar('Rewards/Combined', reward, self.current_step + step)
            self.current_step += len(states)
            self.train_llc(env, training_config, writer)

    def train_hlc(self, env, training_config, writer, at, hlc_only=False):
      """Train the HLC network"""
      states, actions, rewards, log_probs, values = [], [], [], [], []

      state = env.reset()[0] # get only the state from the reset
      done = False
      t = 0
      while not done:
        hlc_state = state["hlc_state"]
        llc_state = state["llc_state"]
        hlc_action, llc_actions = self.select_action(state, exploration_noise=None)
        next_state, reward, done, _ = env.step(llc_actions)

        price_changes = []
        if t > 0: # calculate if there were price changes
            for i in range(self.num_stocks):
                price_changes.append(next_state["stock_prices"][i] - state["stock_prices"][i])
        else: #set to 0 on first iteration
            price_changes = np.zeros(self.num_stocks)

        # calculate reward
        if hlc_only:
          r = self.calculate_hlc_reward(state, next_state, hlc_action, 0, at, price_changes)
        else:
          r = self.calculate_hlc_reward(state, next_state, hlc_action, reward, at, price_changes)
        states.append(hlc_state)
        actions.append(hlc_action)
        values.append(self.hlc_value(torch.tensor(np.array(hlc_state), dtype=torch.float32).to(self.device)).detach().cpu().numpy())
        rewards.append(r)
        state = next_state
        t += 1

      values.append(self.hlc_value(torch.tensor(np.array(next_state["hlc_state"]), dtype=torch.float32).to(self.device)).detach().cpu().numpy())
      advantages = create_advantage_estimator(rewards, values, gamma=training_config["gamma"], gae_lambda=training_config["gae_lambda"])
      for _ in range(training_config["ppo_epochs"]):
        policy_loss, value_loss = ppo_update(self.hlc_policy, self.hlc_value, self.hlc_optimizer_policy, self.hlc_optimizer_value, states, actions, rewards, advantages)
        writer.add_scalar('Losses/HLC_PolicyLoss', policy_loss, self.current_step)
        writer.add_scalar('Losses/HLC_ValueLoss', value_loss, self.current_step)

      return states, actions, rewards, log_probs, values

    def train_llc(self, env, training_config, writer):
        """Train the LLC network"""
        state = env.reset()[0]
        done = False
        while not done:
          hlc_state = state["hlc_state"]
          llc_state = state["llc_state"]
          hlc_action, llc_actions = self.select_action(state, exploration_noise=training_config["exploration_noise"])
          next_state, reward, done, _ = env.step(llc_actions)

          # Store in replay buffer
          for i in range(self.num_stocks):
            if hlc_action[i] > 0.5 or hlc_action[i] < -0.5: #only add to buffer if we take action
              self.llc_replay_buffer.push(llc_state[i], llc_actions[i], reward, next_state["llc_state"][i], done)

          critic_loss, actor_loss = ddpg_update(self.llc_actor, self.llc_critic, self.llc_target_actor, self.llc_target_critic, self.llc_optimizer_actor, self.llc_optimizer_critic, self.llc_replay_buffer, training_config["batch_size"], discount_factor=training_config["gamma"], tau=training_config["tau"])
          writer.add_scalar('Losses/LLC_CriticLoss', critic_loss, self.current_step)
          writer.add_scalar('Losses/LLC_ActorLoss', actor_loss, self.current_step)
          state = next_state

        writer.add_scalar('Rewards/LLC', reward, self.current_step)

# 7. Training and Testing
def train_hrt_agent(agent, env, num_episodes, training_config, writer):
  """Train the HRT agent"""
  agent.phased_alternating_train(num_episodes, env, training_config, writer)
  return agent

def test_hrt_agent(agent, env, num_episodes, writer):
  """ Test the HRT agent and calculate performance metrics."""
  state = env.reset()[0]
  done = False
  total_reward = 0
  trades = 0
  while not done:
    hlc_action, llc_actions = agent.select_action(state, exploration_noise=None)
    next_state, reward, done, _ = env.step(llc_actions)
    state = next_state
    total_reward += reward
    trades +=1
  writer.add_scalar('Results/Test_TotalReward', total_reward, 1)
  writer.add_scalar('Results/Test_NumTrades', trades, 1)
  print(f"Test Result - Total reward: {total_reward}, Trades taken {trades}")


# Main Execution
if __name__ == "__main__":
  # Load Data and setup the environment
  data_folder = "/Users/vandanchopra/Vandan_Personal_Folder/CODE_STUFF/Projects/mathematricks/db/data/ibkr/1d"  # Replace with your data folder path
  all_stocks_data = load_stock_data(data_folder)
  start_date_train = '2015-01-01'
  end_date_train = '2019-12-31'
  start_date_test = '2021-01-01'
  end_date_test = '2021-12-31'

  # Create the custom environment
  env_train = StockTradingEnv(all_stocks_data, start_date_train, end_date_train)
  env_test = StockTradingEnv(all_stocks_data, start_date_test, end_date_test)

  # Configuration
  hlc_config = {
    "input_size": env_train.observation_space["hlc_state"].shape[0],  # Use the observation space of your specific environment.
    "hidden_size": 128,
    "output_size": len(all_stocks_data), # buy/sell/hold for each stock
    "learning_rate_policy": 3e-4,
    "learning_rate_value": 1e-3
  }
  llc_config = {
    "input_size": env_train.observation_space["llc_state"].shape[2], # observation for each stock
    "hidden_size": 128,
    "output_size": 1,  # trading volume for each stock
    "learning_rate_actor": 1e-3,
    "learning_rate_critic": 1e-3,
    "replay_buffer_size": 200000
  }

  training_config = {
    "hlc_episodes": 10,
    "llc_episodes": 20,
    "ppo_epochs": 5,
    "exploration_noise": 0.1,
    "batch_size": 256,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "tau": 0.005
  }

  num_training_episodes = 30 # Adjust as needed
  writer = SummaryWriter()
  # Create and Train HRT Agent
  hrt_agent = HRTAgent(env_train, hlc_config, llc_config, device)
  trained_agent = train_hrt_agent(hrt_agent, env_train, num_training_episodes, training_config, writer)
  # Test the agent
  test_hrt_agent(trained_agent, env_test, num_training_episodes, writer)

  writer.close()