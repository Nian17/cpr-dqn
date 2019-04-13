from ddqn_agent import *
from environment import *

env = GameEnv()
observation_space = env.reset()

agent1 = DDQNAgent(observation_space[0].shape, 7)
agent2 = DDQNAgent(observation_space[0].shape, 7)

agents = [agent1, agent2]

state_size = observation_space[0].shape[0]
last_rewards = []
episode = 0
max_episode_len = 1000
print(50*'#')
print("Printing agent's hyperparameters:")
print('Learning rate:', agent1.learning_rate, 'Batch size:', agent1.batch_size, 'Eps decay len:', agent1.epsilon_decay_len)
print("UPDATE EVERY 3")
print(50*'#')
while episode < 2000:
    episode += 1
    state = env.reset()
    state_n = [np.reshape(i, [1, state_size]) for i in state]
    #if episode % 100 == 0:
    env.render_env()
    agent1_reward = 0
    agent2_reward = 0
    cumulative_reward = 0

    step = 0
    gameover = False
    while not gameover:
        step += 1
        #if episode % 100 == 0:
        env.render_env()
        action_n = [agent.get_action(state) for agent, state in zip(agents, state_n)]
        reward, next_state, done = env.step(action_n)
        next_state = [np.reshape(i, [1, state_size]) for i in next_state]
        agent1_reward += reward[0]
        agent2_reward += reward[1]
        cumulative_reward += reward[0] + reward[1]
        for i, agent in enumerate(agents):
            agent.train_model(action_n[i], state_n[i], next_state[i], reward[i], done)
            agent.update_epsilon()
        state_n = next_state
        terminal = (step >= max_episode_len)
        if done or terminal:
            last_rewards.append([agent1_reward, agent2_reward, cumulative_reward])
            if episode % 3 == 0:
                agent1.update_target_model()
                agent2.update_target_model()
            gameover = True

    print('episode:', episode, 'cumulative reward: ', cumulative_reward, 'epsilon:', agent1.epsilon, 'step', step)
'''
print(50*'#')
print('Average training reward', np.mean(last_rewards))
print('** EVALUATION PHASE **')
print(50*'#')
eval_rewards = []
agent.epsilon = 0
for i in range(100):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    # if episode % 100 == 0:
    #   env.render_env()
    total_reward = 0

    step = 0
    gameover = False
    while not gameover:
        step += 1
        action = agent.get_action(state)
        reward, next_state, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        state = next_state
        terminal = (step >= max_episode_len)
        if done or terminal:
            eval_rewards.append(total_reward)
            gameover = True

    print('episode:', i, 'cumulative reward: ', total_reward, 'epsilon:', agent.epsilon, 'step', step)
print(50*'#')
print('Average evaluation reward', np.mean(eval_rewards))

c=10
mean_rew = []
while c <= len(last_rewards):
        mean_rew.append(np.mean(last_rewards[c-10:c]))
        c+=1
plt.plot([i for i in range(len(mean_rew))], mean_rew, label='DQN; last 10 average')
plt.xlabel('episode')
plt.ylabel('average reward')
plt.title("DQN training")
plt.legend()
plt.show()'''