import numpy as np
import tensorflow as tf
import os


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
        self,
        n_observation,
        n_action,
        learning_rate,
        gamma,
        replay_memory_size,
        batch_size,
        eps_min,
        eps_max,
        eps_decay_steps,
        checkpoint_path,
        save_steps,
        copy_steps,
    ):
        self.n_observation = n_observation
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay_steps = eps_decay_steps
        self.replay_memory = ReplayMemory(replay_memory_size)
        self.checkpoint_path = checkpoint_path
        self.save_steps = save_steps
        self.copy_steps = copy_steps

        self._build_dqn()
        self.sess = tf.Session()

        if os.path.isfile(self.checkpoint_path + ".index"):
            self.saver.restore(self.sess, self.checkpoint_path)
        else:
            self.sess.run(self.variable_init)
            self.sess.run(self.copy_trainee_to_coach)

        self.loss_history = []

    def _q_network(self, X_state, scope, n_action):
        w_initializer = tf.variance_scaling_initializer()
        b_initializer = tf.zeros_initializer()
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(inputs=X_state, units=20, activation=tf.nn.relu,
                                 kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='l1')
            l2 = tf.layers.dense(inputs=l1, units=10, activation=tf.nn.relu,
                                 kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='l2')
            outputs = tf.layers.dense(inputs=l2, units=n_action, activation=tf.nn.softmax,
                                      kernel_initializer=w_initializer,
                                      bias_initializer=b_initializer, name='outputs')
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=scope)
        return outputs, trainable_vars

    def _build_dqn(self):
        # prepare coach network to interact with environment
        self.coach_X_state = tf.placeholder(tf.float32, shape=[None, self.n_observation])
        self.coach_q_values, self.coach_params = self._q_network(
            self.coach_X_state, scope="coach", n_action=self.n_action)

        # prepare trainne network to train for better Q (use memory)
        self.trainee_X_state = tf.placeholder(tf.float32, shape=[None, self.n_observation])
        self.trainee_q_values, self.trainee_params = self._q_network(
            self.trainee_X_state, scope="trainee", n_action=self.n_action)

        # training pipe
        with tf.variable_scope("train"):
            self.trainee_X_action = tf.placeholder(tf.int32, shape=[None])
            self.trainee_y = tf.placeholder(tf.float32, shape=[None, 1])
            q_value = tf.reduce_sum(self.trainee_q_values * tf.one_hot(self.trainee_X_action, self.n_action),
                                    axis=1, keepdims=True)
            error = tf.abs(self.trainee_y - q_value)
            clipped_error = tf.clip_by_value(error, 0.0, 1.0)
            linear_error = 2 * (error - clipped_error)
            self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.training_op = optimizer.minimize(self.loss)

        # parameter initiater
        self.variable_init = tf.global_variables_initializer()

        # parameter saver
        self.saver = tf.train.Saver()

        # copy trainee to coach
        with tf.variable_scope('soft_replacement'):
            self.copy_trainee_to_coach = [tf.assign(c, t) for c, t in zip(self.coach_params, self.trainee_params)]

    def _sample_memories(self, batch_size):
        cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
        for memory in self.replay_memory.sample(batch_size):
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

    def _epsilon_greedy(self, q_values, global_step):
        # use epsilon greedy policy
        epsilon = max(self.eps_min, self.eps_max -
                      (self.eps_max - self.eps_min) * global_step / self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_action)  # random action
        else:
            return np.argmax(q_values)  # optimal action

    def choose_action(self, state, global_step):
        q_values = self.sess.run(self.trainee_q_values, feed_dict={self.trainee_X_state: [state]})
        action = self._epsilon_greedy(q_values, global_step)
        return action, q_values.max()

    def learn(self, global_step):
        # Sample memories and use the target DQN to produce the target Q-Value
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            self._sample_memories(self.batch_size))
        next_q_values = self.sess.run(self.coach_q_values, feed_dict={
                                      self.coach_X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * self.gamma * max_next_q_values

        # Train the online DQN
        _, loss_val = self.sess.run([self.training_op, self.loss], feed_dict={
                                    self.trainee_X_state: X_state_val,
                                    self.trainee_X_action: X_action_val,
                                    self.trainee_y: y_val})

        # Regularly copy the online DQN to the target DQN
        if global_step % self.copy_steps == 0:
            print("copy_trainee_to_coach")
            self.sess.run(self.copy_trainee_to_coach)

        # And save regularly
        if global_step % self.save_steps == 0:
            print("save parameter")
            self.saver.save(self.sess, self.checkpoint_path)

        self.loss_history.append(loss_val)
        return loss_val

    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.loss_history)), self.loss_history)
        plt.ylabel('Loss')
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
