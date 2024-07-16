# encoding: utf-8
# desc: 训练
from datetime import datetime
from parse_arg import args
import torch
import torch.nn as nn
import os
import logger
from pettingzoo.mpe import simple_adversary_v3
import numpy as np
from model import Agent

# 设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

# 创建工作目录
while True:
    experiment_name = input("Enter experiment name : ")
    if os.path.exists(os.path.join("./output", experiment_name)):
        answer = input("work dir exists! overwrite? [Y/N]:")
        if answer.lower() == "y":
            break
    else:
        break
os.makedirs(os.path.join("./output/", experiment_name), exist_ok=True)
os.makedirs(os.path.join("./output/", experiment_name, "csv"), exist_ok=True)
os.makedirs(os.path.join("./output/", experiment_name, "figure"), exist_ok=True)
os.makedirs(os.path.join("./output/", experiment_name, "log"), exist_ok=True)
os.makedirs(os.path.join("./output/", experiment_name, "checkpoint"), exist_ok=True)

# 创建日志
log_file = os.path.join("./output/", experiment_name, "log", "training.log")
logger = logger.logger(log_file)

# 打印日志
logger.log("experiment name: {}".format(experiment_name))
logger.log("device: {}".format(device))
logger.log("max_epochs: {}".format(args.max_epochs))
logger.log("batch_size: {}".format(args.batch_size))

# 初始化环境
env = simple_adversary_v3.parallel_env(N=2, max_cycles=args.max_steps, continuous_actions=True)
multi_obs, _ = env.reset()
num_agent = env.num_agents
agent_name_list = env.agents
logger.log("num_agent: {}".format(num_agent))

# 观测信息维度
obs_dim = []  # [8, 10, 10]
for agent_obs in multi_obs.values():
    obs_dim.append(agent_obs.shape[0])
logger.log("obs_dim: {}".format(obs_dim))

# 状态空间维度
state_dim = sum(obs_dim)  # state_dim = 28
logger.log("state_dim: {}".format(state_dim))

# 动作空间维度
action_dim = []  # [5, 5, 5]
for agent_name in agent_name_list:
    action_dim.append(env.action_space(agent_name).sample().shape[0])
logger.log("action_dim: {}".format(action_dim))

agents = []
for agent_i in range(num_agent):
    agent = Agent(memo_size=args.memory_size,
                  obs_dim=obs_dim[agent_i],  # 5, 单个智能体的观测维度
                  state_dim=state_dim,  # 28, 全局信息维度
                  n_agent=num_agent,  # 3, 智能体数量
                  action_dim=action_dim[agent_i],  # 动作维度
                  alpha=args.lr_actor,
                  beta=args.lr_critic,
                  fc1_dim=args.hidden_dim,
                  fc2_dims=args.hidden_dim,
                  gamma=args.gamma,
                  tau=args.tau,
                  batch_size=args.batch_size)
    agents.append(agent)
logger.log("memory_size: {}".format(args.memory_size))
logger.log("lr_actor: {}".format(args.lr_actor))
logger.log("lr_critic: {}".format(args.lr_critic))
logger.log("hidden_dim: {}".format(args.hidden_dim))
logger.log("gamma: {}".format(args.gamma))
logger.log("tau: {}".format(args.tau))

# 训练
for episode_i in range(args.max_epochs):
    start_time = datetime.now()
    multi_obs, _ = env.reset()
    episode_reward = 0
    multi_done = {agent_name: False for agent_name in agent_name_list}
    for step_i in range(args.max_steps):
        total_step = episode_i * args.max_steps + step_i
        multi_actions = {}
        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_obs[agent_name]
            signal_action = agent.action(single_obs)
            multi_actions[agent_name] = signal_action
        multi_next_obs, multi_reward, multi_done, multi_truncation, infos = env.step(multi_actions)

        # 根据 multi_obs 构建 state
        state = np.array([])
        for obs in multi_obs.values():
            state = np.concatenate([state, obs])

        # 根据 multi_next_obs 构建 next_state
        next_state = np.array([])
        for next_obs in multi_next_obs.values():
            next_state = np.concatenate([next_state, next_obs])

        # 判断是否结束
        if step_i >= args.max_steps - 1:
            multi_done = {agent_name: True for agent_name in agent_name_list}

        # 存储经验
        for agent_i, agent_name in enumerate(agent_name_list):
            agent = agents[agent_i]
            single_obs = multi_obs[agent_name]
            single_next_obs = multi_next_obs[agent_name]
            single_action = multi_actions[agent_name]
            single_reward = multi_reward[agent_name]
            single_done = multi_done[agent_name]
            agent.replay_buffer.add_memo(single_obs, single_next_obs, state, next_state, single_action, single_reward, single_done)

        # 更新网络参数
        if (total_step + 1) % args.update_interval == 0:
            # 取数据
            multi_batch_obs = []
            multi_batch_next_obs = []
            multi_batch_state = []
            multi_batch_next_state = []
            multi_batch_action = []
            multi_batch_next_action = []
            multi_batch_online_action = []
            multi_batch_reward = []
            multi_batch_done = []

            # Sample a batch of memories
            current_memo_size = min(args.memory_size, total_step + 1)
            if current_memo_size < args.batch_size:
                batch_idx = range(0, current_memo_size)
            else:
                batch_idx = np.random.choice(current_memo_size, args.batch_size)

            for agent_j in range(num_agent):
                agent = agents[agent_j]
                batch_obs, batch_next_obs, batch_state, batch_next_state, batch_action, batch_reward, batch_done = agent.replay_buffer.sample(batch_idx)

                # batch to tensor
                batch_obs_tensor = torch.tensor(batch_obs, dtype=torch.float).to(device)
                batch_next_obs_tensor = torch.tensor(batch_next_obs, dtype=torch.float).to(device)
                batch_state_tensor = torch.tensor(batch_state, dtype=torch.float).to(device)
                batch_next_state_tensor = torch.tensor(batch_next_state, dtype=torch.float).to(device)
                batch_action_tensor = torch.tensor(batch_action, dtype=torch.float).to(device)
                batch_reward_tensor = torch.tensor(batch_reward, dtype=torch.float).to(device)
                batch_done_tensor = torch.tensor(batch_done, dtype=torch.float).to(device)

                # multi_batch
                multi_batch_obs.append(batch_obs_tensor)
                multi_batch_next_obs.append(batch_next_obs_tensor)
                multi_batch_state.append(batch_state_tensor)
                multi_batch_next_state.append(batch_next_state_tensor)
                multi_batch_action.append(batch_action_tensor)

                single_batch_next_action = agent.target_actor.forward(batch_next_obs_tensor)
                multi_batch_next_action.append(single_batch_next_action)
                single_batch_online_action = agent.actor.forward(batch_obs_tensor)
                multi_batch_online_action.append(single_batch_online_action)

                multi_batch_reward.append(batch_reward_tensor)
                multi_batch_done.append(batch_done_tensor)

            multi_batch_action_tensor = torch.cat(multi_batch_action, dim=1).to(device)  # 1 * 28
            multi_batch_next_action_tensor = torch.cat(multi_batch_next_action, dim=1).to(device)  # 1 * 15
            multi_batch_online_action_tensor = torch.cat(multi_batch_online_action, dim=1).to(device)

            # Update critic and actor
            for agent_i in range(num_agent):
                agent = agents[agent_i]
                batch_obs_tensor = multi_batch_obs[agent_i]
                batch_state_tensor = multi_batch_state[agent_i]
                batch_next_state_tensor = multi_batch_next_state[agent_i]  # 1 * 8
                batch_reward_tensor = multi_batch_reward[agent_i]
                batch_done_tensor = multi_batch_done[agent_i]
                batch_action_tensor = multi_batch_action[agent_i]

                # target critic
                critic_target_q = agent.target_critic.forward(batch_next_state_tensor, multi_batch_next_action_tensor.detach())
                y = (batch_reward_tensor + (1 - batch_done_tensor) * agent.gamma * critic_target_q).flatten()
                critic_q = agent.critic.forward(batch_state_tensor, multi_batch_action_tensor.detach()).flatten()

                critic_loss = nn.MSELoss()(y, critic_q)
                agent.critic.optimizer.zero_grad()
                critic_loss.backward()
                agent.critic.optimizer.step()

                # update actor
                actor_loss = agent.critic.forward(batch_state_tensor, multi_batch_online_action_tensor.detach()).flatten()
                actor_loss = - torch.mean(actor_loss)
                agent.actor.optimizer.zero_grad()
                actor_loss.backward()
                agent.actor.optimizer.step()

                # update target critic
                for target_param, param in zip(agent.target_critic.parameters(), agent.critic.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

                # update target actor
                for target_param, param in zip(agent.target_actor.parameters(), agent.actor.parameters()):
                    target_param.data.copy_(agent.tau * param.data + (1.0 - agent.tau) * target_param.data)

        multi_obs = multi_next_obs
        episode_reward += sum([single_reward for single_reward in multi_reward.values()])
    end_time = datetime.now()
    execution_time = end_time - start_time
    logger.log('episode: {}\t\t lr: {}\t\t reward: {}\t\t time: {}'.format(episode_i, args.lr_actor, episode_reward, execution_time.total_seconds()))

env.close()