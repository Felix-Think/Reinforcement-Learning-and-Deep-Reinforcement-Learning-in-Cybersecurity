import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.distributions as distributions 
import torch.optim as optim 

import numpy as np
import matplotlib.pyplot as plt 
import gymnasium as gym

train_env = gym.make("CartPole-v1")
val_env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")


class ReplayBuffer():
    def __init__(self, capacity, n_dim, device="cpu"):
        self.capacity = capacity 
        self.n_dim = n_dim 
        self.device = device 
        self.state_buff = torch.zeros((capacity, *n_dim), dtype=torch.float32)
        self.action_buff = torch.zeros((capacity, ), dtype=torch.int64)
        self.return_buff = torch.zeros((capacity, ), dtype=torch.float32)
        self.advantage_buff = torch.zeros((capacity, ), dtype=torch.float32)
        self.size = 0
        self.ptr = 0 
    def store(self, state, action, returns, advantage):
        self.state_buff[self.ptr] = state 
        self.action_buff[self.ptr] = action 
        self.return_buff[self.ptr] = returns 
        self.advantage_buff[self.ptr] = advantage 
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample_batch(self, batch_size):
        idxs = np.random.choice(self.size, batch_size)
        batch = [
            self.state_buff[idxs],
            self.action_buff[idxs],
            self.return_buff[idxs],
            self.advantage_buff[idxs],
        ]
        return [torch.tensor(x, device=self.device) for x in batch]
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor 
        self.critic = critic 

    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred 

INPUT_DIM = train_env.observation_space.shape[0]
HIDDEN_DIM = 128 
OUTPUT_DIM = train_env.action_space.n 

actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)
policy = ActorCritic(actor, critic)

LEARNING_RATE = 1e-3
optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)


def init_weight(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

policy.apply(init_weight)

def compute_gae(values, rewards, dones, discounted_factor, trace_decay, next_value):
    advantages = []
    gae = 0
    for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = reward + discounted_factor * next_value * (1 - done) - value.item()
        gae = delta + discounted_factor * trace_decay * (1-done) *gae 
        next_value = value.item()
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype = torch.float32)
    values = torch.tensor(values, dtype = torch.float32)
    returns = advantages + values
    return returns, advantages
    
def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def collect_trajectories(env, policy, buffer, discounted_factor, trace_decay, n_steps = 2048):
    policy.eval()
    ep_rewards_list = []
    episode_reward = 0
    state, _ = env.reset()
    states, actions, rewards, values, dones = [], [], [], [], []

    for _ in range(n_steps):
        state = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            action_pred, value_pred = policy(state)
        dist = distributions.Categorical(logits = action_pred)
        action = dist.sample().item()
        next_state, reward, done, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(value_pred.squeeze(-1).item())
        dones.append(float(done))

        episode_reward += reward 
        if done or truncated:
            state, _ = env.reset()
            ep_rewards_list.append(episode_reward) 
            episode_reward = 0 
        else:
            state = next_state 
    with torch.no_grad():
        state = torch.tensor(state).unsqueeze(0)
        _, next_value = policy(state)
        next_value = next_value.squeeze(-1).item()

    values = np.array(values)
    returns, advantages = compute_gae(values, rewards, dones, discounted_factor, trace_decay, next_value)

    for i in range(n_steps):
        buffer.store(states[i],
        actions[i],
        returns[i],
        advantages[i])

    return np.mean(ep_rewards_list) if len(ep_rewards_list) > 0 else episode_reward

        

    
def update_policy(policy, optimizer, buffer, batch_size, beta, weight_clip, n_epochs = 10):
    policy.train()

    if buffer.size < batch_size:
        return 0.0, 0.0
    
    total_actor_loss, total_critic_loss = 0, 0 

    update_count = 0
    
    # Tính số lần update gradient (ví dụ n_steps=2048, batch_size=256 => 1 epoch ~ 8 updates)
    num_updates = n_epochs * 8

    for _ in range(num_updates):
        states, actions, returns, advantages = buffer.sample_batch(batch_size)
        
        # Chuẩn hóa advantage ngay tại lúc sample batch thay vì trước khi đưa vào buffer
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actions_pred, values_pred = policy(states)
        values_pred = values_pred.squeeze(-1)

        weights = torch.exp(1 / beta * advantages)
        weights = torch.clamp(weights, max = weight_clip)

    
        # Đưa vào distribution
        dist = distributions.Categorical(logits = actions_pred)
        log_probs = dist.log_prob(actions)

        # tinh loss
        actor_loss = -(weights * log_probs).mean()
        critic_loss = F.mse_loss(returns, values_pred)
        loss = actor_loss + 0.5 * critic_loss 

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()
        update_count += 1 
        
    return total_actor_loss / update_count, total_critic_loss / update_count

def evaluate(env, policy):
    policy.eval()
    state, _ = env.reset()
    done, truncated = False, False 
    ep_reward = 0
    while not done and not truncated:
        
        with torch.no_grad():
            state  = torch.tensor(state).unsqueeze(0)
            action_pred, _ = policy(state)
        action = F.softmax(action_pred, dim = -1)
        action = torch.argmax(action_pred, dim = -1)
        state, reward, done, truncated, _ = env.step(action.item())
        ep_reward += reward 

    return ep_reward 



class AWR_train():
    
    def __init__(self, max_episode, discount_factor, n_trials, reward_threshold, print_every, buffer_capacity, batch_size, tau, beta, weight_clip, n_epochs, n_steps, trace_decay):
        self.max_episode = max_episode 
        self.discount_factor = discount_factor 
        self.n_trials = n_trials 
        self.reward_threshold = reward_threshold 
        self.print_every = print_every 
        self.buffer_capacity = buffer_capacity 
        self.batch_size = batch_size 
        self.tau = tau 
        self.beta = beta 
        self.weight_clip = weight_clip
        self.n_epochs = n_epochs
        self.n_steps = n_steps 
        self.trace_decay = trace_decay

    def run(self, train_env, test_env):
        train_rewards = []
        test_rewards = []

        INPUT_DIM = train_env.observation_space.shape[0]
        HIDDEN_DIM = 256
        OUTPUT_DIM = train_env.action_space.n 

        actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)
        policy = ActorCritic(actor, critic)
        policy.apply(init_weight)
        
        optimizer = optim.Adam(policy.parameters(), lr=1e-3)

        target_actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        target_critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)
        target_policy = ActorCritic(target_actor, target_critic)
        target_policy.load_state_dict(policy.state_dict())
        target_policy.eval()

        buffer = ReplayBuffer(self.buffer_capacity, train_env.observation_space.shape)
        for episode in range(1, self.max_episode+1):
                
            train_reward = collect_trajectories(train_env, policy, buffer, self.discount_factor, self.trace_decay, self.n_steps)
            policy_loss, value_loss = update_policy(policy, optimizer, buffer, self.batch_size, self.beta, self.weight_clip, self.n_epochs)
            soft_update(policy, target_policy, self.tau)

            test_reward = evaluate(test_env, policy)
            train_rewards.append(train_reward)
            test_rewards.append(test_reward)
                
            mean_train_rewards = np.mean(train_rewards[-self.n_trials:])
            mean_test_rewards = np.mean(test_rewards[-self.n_trials:])
                
            if episode % self.print_every == 0:
                print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
                
            if mean_test_rewards >= self.reward_threshold:
                print(f'Reached reward threshold in {episode} episodes')
                break
        return train_rewards, test_rewards

if __name__== "__main__":

    MAX_EPISODES = 500
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 500
    PRINT_EVERY = 10

    BUFFER_CAPACITY = 50000
    BATCH_SIZE = 256
    TAU = 0.005
    BETA = 0.99
    WEIGHT_CLIP = 20.0

    train_rewards = []
    test_rewards = []


    INPUT_DIM = train_env.observation_space.shape
    HIDDEN_DIM = 128 
    OUTPUT_DIM = train_env.action_space.n 


    target_actor = MLP(INPUT_DIM[0], HIDDEN_DIM, OUTPUT_DIM)
    target_critic = MLP(INPUT_DIM[0], HIDDEN_DIM, 1)
    target_policy = ActorCritic(target_actor, target_critic)
    target_policy.load_state_dict(policy.state_dict())
    target_policy.eval()

    buffer = ReplayBuffer(BUFFER_CAPACITY, INPUT_DIM)
    
