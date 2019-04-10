from ddqn_agent import *
from environment import *

env = GameEnv()
observation_space = env.reset()

agent = DDQNAgent(observation_space.shape, 7)

state_size = observation_space.shape[0]
last_rewards = []
episode = 0
max_episode_len = 1000
print(50*'#')
print("Printing agent's hyperparameters:")
print('Learning rate:', agent.learning_rate, 'Batch size:', agent.batch_size, 'Eps decay len:', agent.epsilon_decay_len)
print("UPDATE EVERY 3")
print(50*'#')
while episode < 20000:
    episode += 1
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    #if episode % 100 == 0:
     #   env.render_env()
    total_reward = 0

    step = 0
    gameover = False
    while not gameover:
        step += 1
        #if episode % 100 == 0:
         #   env.render_env()
        action = agent.get_action(state)
        reward, next_state, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        agent.train_model(action, state, next_state, reward, done)
        agent.update_epsilon()
        state = next_state
        terminal = (step >= max_episode_len)
        if done or terminal:
            last_rewards.append(total_reward)
            #if episode % 3 == 0:
             #   agent.update_target_model()
            gameover = True

    print('episode:', episode, 'cumulative reward: ', total_reward, 'epsilon:', agent.epsilon, 'step', step)
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
plt.show()
'''
