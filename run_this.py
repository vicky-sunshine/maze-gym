from maze_env import Maze
from dqn import DeepQNetwork


def run_maze():
    step = 0
    episode = 0
    for episode in range(5000):
        # initial observation
        observation = env._reset()

        while True:
            # fresh env
            env._render()

            # model choose action based on observation
            action = model.choose_action(step, observation)

            # model take action and get next observation and reward
            observation_, reward, continue_ = env._step(action)

            model.store_transition(observation, action, reward, observation_, continue_)

            if (step > 1000) and (step % 5 == 0):
                model.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if continue_ == 0:
                break
            step += 1

        # episode += 1
        print("episode: "+str(episode))

    # end of game
    print('game over')


if __name__ == "__main__":
    # maze game
    env = Maze()
    model = DeepQNetwork(env.n_actions, env.n_features,
                         batch_size=64,
                         learning_rate=0.01,
                         reward_decay=0.9,
                         replace_trainee_iter=5000,
                         save_trainee_iter=2000,
                         memory_size=10000,
                         eps_min=0.1,
                         eps_max=0.9,    # the higher, more random
                         eps_decay_steps=2000000,
                         checkpoint_path="checkpoint/maze.ckpt",
                         output_graph=True
                         )
    run_maze()
    # model.plot_cost()
