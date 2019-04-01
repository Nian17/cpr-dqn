from dqn_agent import *
from environment import *

env = GameEnv()
observation_space = env.reset()

agent = DDQNAgent(observation_space.shape, 7)

state_size = observation_space.shape[0]
last_rewards = []
episode = 0
max_episode_len = 1000
while episode < 10100:
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
            agent.update_target_model()
            gameover = True

    print('episode:', episode, 'cumulative reward: ', total_reward, 'epsilon:', agent.epsilon, 'step', step)