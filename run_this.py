import numpy as np
from maze_env import Maze
from dqn import DeepQNetwork


def run_maze(n_episode, training_start, training_interval):
    global_step = 0

    for episode in range(n_episode):
        state = env.reset()
        local_step = 0
        total_max_q = 0
        mean_max_q = 0.0
        loss_val = np.infty

        while True:
            local_step += 1
            global_step += 1

            # Online DQN evaluates what to do
            action, max_q_values = model.choose_action(state, global_step)

            # Online DQN plays
            next_state, reward, done = env.step(action)
            env.render()

            # Let's memorize what happened
            model.replay_memory.append((state, action, reward, next_state, 1.0 - done))
            state = next_state
            total_max_q += max_q_values
            mean_max_q = total_max_q / local_step

            if (global_step > training_start) and (global_step % training_interval == 0):
                loss_val = model.learn(global_step)

            print("\rGlobal step {}\tEpisode {}\tlocal_step {}\tLoss {:5f}\tMean Max-Q {:5f}\taction {}\n".format(
                global_step, episode, local_step, loss_val, mean_max_q, env.action_space[action]), end="")

            if done:
                break
    # end of game
    print('game over')


if __name__ == "__main__":
    # maze game
    env = Maze()
    model = DeepQNetwork(n_observation=env.n_features,
                         n_action=env.n_actions,
                         learning_rate=0.01,
                         gamma=0.9,
                         replay_memory_size=10000,
                         batch_size=50,
                         eps_min=0.1,
                         eps_max=1.0,    # the higher, more random
                         eps_decay_steps=50000,
                         checkpoint_path="checkpoint/maze.ckpt",
                         save_steps=2000,
                         copy_steps=5000)
    # trainie
    run_maze(
        n_episode=5000,
        training_start=1000,
        training_interval=4)

    model.plot_loss()
