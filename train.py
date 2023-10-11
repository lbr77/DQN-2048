import tensorflow as tf
import random
import sys
import time
from collections import deque

# from rl.agents import DQNAgent
# from rl.memory import SequentialMemory
# from rl.policy import BoltzmannQPolicy
from keras.optimizers import adam_v2
import numpy as np

from gym_2048.envs import Game2048Env

env = Game2048Env()


def make_model():
    ipt = tf.keras.layers.Input(shape=(4, 4, 1))
    # x = tf.keras.layers.Conv2D(32, 3, activation="relu")(ipt)
    # x = tf.keras.layers.Conv2D(64, 1, activation="relu")(x)
    # # x = tf.keras.layers.Conv2D(128, 2, activation="relu")(x)
    # # x = tf.keras.layers.Conv2D(256, activation="relu")(ipt)
    x = tf.keras.layers.Dense(256, activation="relu")(ipt)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    actions = tf.keras.layers.Dense(4, activation="linear")(x)
    model = tf.keras.Model(inputs=ipt, outputs=actions)
    return model


model = make_model()
try:
    model.load_weights("models/dqn_weights_best.pkl")
    print("Loaded Weights from dqn_weights_best.pkl")
except:
    print("No Weights Found")
    pass
tmodel = make_model()

print(model.summary())

optimizer = adam_v2.Adam()
model.compile(optimizer, loss="mse")
# tmodel.compile(optimizer, loss="mse")
memory = deque(maxlen=5000)


def convert_to_batch(samples):
    states = np.array([sample[0] for sample in samples]).reshape(-1, 4, 4, 1)
    actions = np.array([sample[1] for sample in samples])
    rewards = np.array([sample[2] for sample in samples])
    next_states = np.array([sample[3] for sample in samples]).reshape(-1, 4, 4, 1)
    return states, actions, rewards, next_states


def model_save(model, score):
    print("Saving model...")
    model.save("models/dqn.h5")
    model.save_weights("models/dqn_weights_{}.pkl".format(int(score)))


def train():
    batch_size = 128
    epochs = 1000
    gamma = 0.95
    epsilon = 1.0
    global score
    score = 0
    print("Started Training")
    tot_times = 0
    env.reset()
    try:
        for i in range(epochs):  # we train for n rounds of game.
            tot_times += 1
            print("\nEpoch {}".format(i))
            env.reset()
            t = 0
            times = time.time()
            while True:
                t += 1
                sys.stdout.write(
                    f"\r{t} steps played.Speed {1000*(time.time()-times)/t:.2f} ms/step. Score={env.score} .eps={epsilon:.2f} "
                )
                sys.stdout.flush()
                state = np.array(env.get_board()).reshape(-1, 4, 4, 1)
                if np.random.rand() < epsilon:  # random action
                    action = np.random.randint(0, 4)
                else:  # model action
                    q_values = model.predict(state)[0]
                    action = np.argmax(q_values)
                # print(action)
                next_state, reward, done, _ = env.step(action)
                memory.append((state, action, reward, next_state, done))
                batch = []
                if len(memory) < batch_size:
                    batch = memory
                else:
                    batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states = convert_to_batch(batch)
                # print(states.shape, actions.shape, rewards.shape, next_states.shape)
                next_q_values = tmodel.predict(next_states)
                max_next_q_values = np.max(next_q_values, axis=1).reshape(-1, 1)
                target_q_values = rewards + (1 - done) * gamma * max_next_q_values

                with tf.GradientTape() as tape:
                    q_values = model(states)
                    action_q_values = tf.gather(q_values, actions, batch_dims=1)
                    loss = tf.reduce_mean(tf.square(target_q_values - action_q_values))
                    # print(
                    #     "Step {},score={},Step={},max score={}".format(
                    #         i, env.score, action, score
                    #     )
                    # )
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if env.isend():
                    print(
                        "\nEpoch {},Game End.Score {} max score={} max_tile={}, Status\n {}".format(
                            i, env.score, score, env.max_tile, env.get_board()
                        )
                    )
                    if score < env.score:
                        # model.save("dqn_{}.h5".format(env.score))

                        score = env.score
                    break

                if epsilon > 0.1:
                    epsilon *= 0.999
                if tot_times % 100 == 0:
                    tmodel.set_weights(model.get_weights())  # update target model
            if i % 5 == 0:
                epsilon = 1.0  # reset epsilon

    except Exception as e:
        print(e)
        print("Saving Model")
        model.save("models/dqn.h5")
        model.save_weights("models/dqn_weights_best.pkl")


try:
    train()
    model_save(model, score)
except KeyboardInterrupt as e:
    print(e)
    print("Saving Model")
    model.save("models/dqn.h5")
    model.save_weights("models/dqn_weights_best.pkl")
