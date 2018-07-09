import numpy as np
import tensorflow as tf
import os

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.01,
        reward_decay=0.9,
        eps_min=0.1,
        eps_max=1,
        eps_decay_steps=2000000,
        memory_size=10000,
        batch_size=32,
        checkpoint_path=None,
        replace_trainee_iter=300,
        save_trainee_iter=300,
        output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay_steps = eps_decay_steps
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.replace_trainee_iter = replace_trainee_iter
        self.save_trainee_iter = save_trainee_iter

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.replay_memory = ReplayMemory(self.memory_size)

        # consist of [trainee_network, evaluate_net]
        self._build_dqn()
        self.sess = tf.Session()

        self.checkpoint_path = checkpoint_path

        self.saver = tf.train.Saver()
        if os.path.isfile(checkpoint_path+".index"):
            self.saver.restore(self.sess, checkpoint_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        if output_graph:
            # $ tensorboard --logdir=logs
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self, x_state, scope, n_observation, n_action):
        w_initializer = tf.variance_scaling_initializer()
        b_initializer = tf.zeros_initializer()
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(inputs=x_state, units=60, activation=tf.nn.relu,
                                 kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='l1')
            l2 = tf.layers.dense(inputs=l1, units=40, activation=tf.nn.relu,
                                 kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='l2')
            outputs = tf.layers.dense(inputs=l2, units=n_action, activation=tf.nn.softmax,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer, name='outputs')
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=scope)
        return outputs, trainable_vars

    # return outputs
    def _build_dqn(self):
        # all inputs
        self.s = tf.placeholder(tf.float32, shape=[None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, shape=[None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, shape=[None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, shape=[None, ], name='a')  # input Action

        # build network
        self.q_coach_output, coach_params = self._build_net(self.s_, 'coach',
                                                            self.n_features, self.n_actions)
        self.q_trainee_output, trainee_params = self._build_net(self.s, 'trainee',
                                                                self.n_features, self.n_actions)

        with tf.variable_scope('q_coach'):
            q_coach = self.r + self.gamma * \
                      tf.reduce_max(self.q_coach_output, axis=1,
                                    name='Qmax_s_')    # shape=(None, )            # shape=(None, )
        with tf.variable_scope('q_trainee'):
            q_trainee = tf.reduce_sum(self.q_trainee_output * tf.one_hot(self.a, self.n_actions),
                                      axis=1, keepdims=True)
        with tf.variable_scope('loss'):
            error = tf.abs(q_coach - q_trainee)
            clipped_error = tf.clip_by_value(error, 0.0, 1.0)
            linear_error = 2 * (error - clipped_error)
            self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        with tf.variable_scope('soft_replacement'):
            self.coach_replace_op = [tf.assign(c, t) for c, t in zip(coach_params, trainee_params)]

    def store_transition(self, s, a, r, s_):
        # store transiyion to replay memory
        self.replay_memory.append((s, a, r, s_))

    def sample_memories(self, batch_size):
        # sample batch_size memory for trainee network
        cols = [[], [], [], []]  # state, action, reward, next_state
        for memory in self.replay_memory.sample(batch_size):
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        # return s, a, r, s_
        return cols[0].reshape(-1, self.n_features), cols[1], cols[2], cols[3].reshape(-1, self.n_features)

    def choose_action(self, step, observation):
        # epsilon-greedy policy
        epsilon = max(self.eps_min,
                      self.eps_max - (self.eps_max - self.eps_min) * step / self.eps_decay_steps)
        if np.random.uniform() < epsilon:
            # random choose action for exploration
            action = np.random.randint(0, self.n_actions)
        else:
            # use coach network to choose action
            actions_value = self.sess.run(self.q_coach_output, feed_dict={
                                          self.s_: observation})
            action = np.argmax(actions_value)

        return action

    def learn(self):
        # check to replace trainee parameters
        if self.learn_step_counter % self.replace_trainee_iter == 0:
            self.sess.run(self.coach_replace_op)
            print('\ntrainee_params_replaced\n')
        if self.learn_step_counter % self.save_trainee_iter == 0:
            self.save_parameter()
            print('\ntrainee_params_saved\n')

        X_state_val, X_action_val, rewards, X_next_state_val = (
            self.sample_memories(self.batch_size))
        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: X_state_val,
                self.a: X_action_val,
                self.r: rewards,
                self.s_: X_next_state_val,
            })

        self.cost_his.append(cost)

        self.learn_step_counter += 1
        print("trainee learning step:" + str(self.learn_step_counter))

    def save_parameter(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size)  # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]

