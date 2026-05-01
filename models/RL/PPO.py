import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import torch.distributions as distributions 

import numpy  as np
import matplotlib.pyplot as plt 
import gymnasium as gym 
test_env = gym.make("CartPole-v1")
train_env = gym.make("CartPole-v1")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.0):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor 
        self.critic = critic 
    
    def forward(self, x):
        action_pred = self.actor(x)
        value_pred = self.critic(x)
        return action_pred, value_pred 
INPUT_DIM = train_env.observation_space.shape[0]
HIDDEN_DIM = 64
OUTPUT_DIM = test_env.action_space.n 

actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

def init_weight(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

policy = ActorCritic(actor, critic)
policy.apply(init_weight)
optimizer = optim.Adam(policy.parameters(), lr = 1.5e-3)


def train(env, policy, optimizer, discounted_factor, ppo_steps, ppo_clip, trace_decay, n_steps = 2048, batch_size = 64):
    policy.train()

    states = []
    rewards = []
    values = []
    dones = []
    log_prob_actions = []
    actions = []
    done, truncated = False, False 

    state, _ = env.reset() 
    
    ep_rewards_list = []
    episode_reward = 0 
    
    for _ in range(n_steps):
        state_tensor = torch.tensor(state).unsqueeze(0)
        states.append(state_tensor)
        with torch.no_grad():
            action_pred, value_pred = policy(state_tensor)

        dist = distributions.Categorical(logits = action_pred)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)

        next_state, reward, done, truncated, _ = env.step(action.item())

        episode_reward += reward 
        if done or truncated:
            state, _ = env.reset()
            ep_rewards_list.append(episode_reward)
            episode_reward = 0
        else:
            state = next_state  
        rewards.append(reward)
        dones.append(done)
        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)


    #chuyen sang tensor
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)
    states = torch.cat(states)
    actions = torch.cat(actions)

    # Tính next_value ở step cuối cùng để phục vụ GAE Bootstrap
    _, next_value = policy(torch.tensor(state).unsqueeze(0))
    next_value = next_value.item()
    
    # TÍNH GAE VÀ CHUẨN HÓA TRÊN TOÀN BỘ N-STEPS
    advantages, returns = compute_gae(values, rewards, dones, discounted_factor, trace_decay, next_value)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    #CHIA MINI-BATCH ĐỂ HỌC
    indices = np.arange(n_steps)
    total_policy_loss = 0
    total_value_loss = 0 

    for _ in range(ppo_steps):
        np.random.shuffle(indices)
        for start in range(0, n_steps, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_log_prob_actions = log_prob_actions[batch_indices]
            batch_returns = returns[batch_indices]

            policy_loss, value_loss = update_policy(policy, 
                                                    optimizer, 
                                                    batch_states, 
                                                    batch_actions, 
                                                    batch_returns, 
                                                    batch_advantages, 
                                                    batch_log_prob_actions, 
                                                    ppo_clip)

            total_policy_loss += policy_loss
            total_value_loss += value_loss
            
    mean_train_reward = np.mean(ep_rewards_list) if len(ep_rewards_list) > 0 else episode_reward
    return total_policy_loss, total_value_loss, mean_train_reward



def compute_gae(values, rewards, dones, discounted_factor, trace_decay, next_value):
    advantages = []
    gae = 0
    for reward, value, done in  zip(reversed(rewards), reversed(values), reversed(dones)):
        # Tinhs 1-step TD error
        delta = reward + discounted_factor * next_value * (1 - done) - value.item()
        
        # Tich luy TD error voi trace_decay 
        gae = delta + discounted_factor * trace_decay * (1 - done) * gae 
        next_value = value.item()
        advantages.insert(0, gae)
    advantages = torch.tensor(advantages, dtype = torch.float32) # N

    returns = advantages + values.detach() # N + N broadcast

    #normalize cho on dinh khi train 
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns

def update_policy(policy, optimizer, states, actions, returns, advantages, log_prob_actions, ppo_clip):
    total_policy_loss = 0
    total_value_loss = 0
    
    #detacch 
    states = states.detach()
    advantages = advantages.detach()
    returns = returns.detach()
    actions = actions.detach()
    log_prob_actions = log_prob_actions.detach()


    # get new log prob action for all input states
    action_pred, value_pred = policy(states)
    value_pred = value_pred.squeeze(-1)

    action_prob = F.softmax(action_pred, dim = -1)
    dist = distributions.Categorical(action_prob)
    entropy = dist.entropy().mean()
    next_log_prob_actions = dist.log_prob(actions)

    # tinh ratio between policy and old_policy 
    policy_ratio = (next_log_prob_actions - log_prob_actions).exp()
    policy_loss1 = policy_ratio * advantages 
    policy_loss2 = advantages * torch.clamp(policy_ratio,min = 1.0 - ppo_clip,max = 1.0 + ppo_clip)

    policy_loss = - torch.min(policy_loss1, policy_loss2).mean()
    policy_loss = policy_loss - 0.01 * entropy

    value_loss = F.smooth_l1_loss(returns, value_pred).mean()

    optimizer.zero_grad()
    loss = policy_loss + 0.5 * value_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()

    total_policy_loss += policy_loss.item() 
    total_value_loss += value_loss.item() 

    return total_policy_loss, total_value_loss 


def evaluate(env, policy):
    policy.eval()
    done, truncated = False, False 
    episode_reward = 0 

    state, _ = env.reset()
    while not done and not truncated:
        with torch.no_grad():
            state = torch.tensor(state).unsqueeze(0)
            action_pred, _ = policy(state)
        action_prob = F.softmax(action_pred, dim = -1)
        action = torch.argmax(action_prob, dim = -1)
        state, reward, done, truncated, _ = env.step(action.item())
        episode_reward += reward 
    return episode_reward 

class PPO_train():
    def __init__(self, max_episode, discount_factor, n_trials, reward_threshold, print_every, ppo_steps, ppo_clip, trace_decay):
        self.max_episode = max_episode 
        self.discount_factor = discount_factor 
        self.n_trials = n_trials 
        self.reward_threshold = reward_threshold 
        self.print_every = print_every 
        self.ppo_steps = ppo_steps 
        self.ppo_clip = ppo_clip 
        self.trace_decay = trace_decay 


    def run(self, train_env, test_env):
        train_rewards = []
        test_rewards = []

        INPUT_DIM = train_env.observation_space.shape[0]
        HIDDEN_DIM = 256
        OUTPUT_DIM = test_env.action_space.n 

        actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

        def init_weight(m):
            if type(m) == nn.Linear:
                torch.nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0)

        policy = ActorCritic(actor, critic)
        policy.apply(init_weight)
        optimizer = optim.Adam(policy.parameters(), lr = 1.5e-3)

        for episode in range(1, self.max_episode+1):
                
            policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, self.discount_factor, self.ppo_steps, self.ppo_clip, self.trace_decay)
                
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

if __name__ == "__main__":
    MAX_EPISODES = 500
    DISCOUNT_FACTOR = 0.99
    N_TRIALS = 25
    REWARD_THRESHOLD = 500
    PRINT_EVERY = 10
    PPO_STEPS = 3
    PPO_CLIP = 0.2
    TRACE_DECAY = 0.99

    train_rewards = []
    test_rewards = []

    for episode in range(1, MAX_EPISODES+1):
        
        policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP, TRACE_DECAY)
        
        test_reward = evaluate(test_env, policy)
        
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        
        mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
        mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
        
        if episode % PRINT_EVERY == 0:
        
            print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')
        
        if mean_test_rewards >= REWARD_THRESHOLD:
            
            print(f'Reached reward threshold in {episode} episodes')
            
            break