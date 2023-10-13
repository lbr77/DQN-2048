import os

# 设置TensorFlow日志级别
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from gym_2048.envs import Game2048Env
import sys

tf.get_logger().setLevel("FATAL")
# Functions


def convert_to_batch(batch):
    states = np.array([i[0] for i in batch]).reshape(-1, 4, 4, 1)
    actions = np.array([i[1] for i in batch])
    rewards = np.array([i[2] for i in batch])
    next_states = np.array([i[3] for i in batch]).reshape(-1, 4, 4, 1)
    return states, actions, rewards, next_states


def save_model():
    model.save("model.keras")
    model.save_weights("model.pkl")
    print("Saved model")
    return


# Hyperparameters
BATCH_SIZE = 64  # size of data used to experience replay
GAMMA = 0.99  # discount factor
EPSILON = 1.0  # exploration rate
EPSILON_MIN = 0.1  # minimum exploration rate
EPSILON_DECAY = 0.999  # decay rate of exploration rate
TARGET_UPDATE = 100  # update target network every 5 episodes
MEMORY_SIZE = 50000  # size of experience replay memory
LEARNING_RATE = 0.001  # learning rate
EPOCHES = 1000  # number of training epochs
RESET_EPSILON_TIMES = 10  # reset epsilon for 10 times.
# Environment
env = Game2048Env()
memory = deque(maxlen=MEMORY_SIZE)


# Model
def make_model():
    inputs = tf.keras.Input(shape=(4, 4, 1))
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(4, activation="linear")(x)
    model = Model(inputs=inputs, outputs=x)
    return model


model = make_model()
target_model = make_model()

try:
    model.load_weights("model.pkl")
    print("Loaded model")
except:
    print("No model found")

target_model.set_weights(model.get_weights())

# Optimizer
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="mse")

# Variables
tot_times = 0
# Training

try:
    for epoch in range(EPOCHES):
        env.reset()
        t = 0
        while True:
            t += 1
            tot_times += 1
            sys.stdout.write(
                f"\r\rEpoch: {epoch}. Steps: {t}. Exploration rate: {EPSILON:.2f}. Current Score: {env.score}. "
            )
            sys.stdout.flush()
            state = np.array(env.get_board()).reshape(-1, 4, 4, 1)
            if np.random.rand() <= EPSILON:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(model.predict(state, verbose=0)[0])
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state).reshape(-1, 4, 4, 1)
            memory.append((state, action, reward, next_state, done))
            batch = []
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
            else:
                batch = memory
            states, actions, rewards, next_states = convert_to_batch(batch)
            next_q_val = target_model.predict(next_states, verbose=0)
            max_next_q_val = np.max(next_q_val, axis=1)
            target_q_val = rewards + (1 - done) * GAMMA * max_next_q_val
            with tf.GradientTape() as tape:
                q_values = model(states)
                actions_q_values = tf.gather(q_values, actions, batch_dims=1)
                loss = tf.reduce_mean(tf.square(target_q_val - actions_q_values))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if env.isend():
                break
            if EPSILON > EPSILON_MIN:
                EPSILON *= EPSILON_DECAY
            if tot_times % TARGET_UPDATE == 0:  # update target network
                target_model.set_weights(model.get_weights())
        if (
            epoch % RESET_EPSILON_TIMES == 0 and epoch <= 100
        ):  # We use the first 0.1 epoches to explore with high probability.
            EPSILON = 1.0  # RESET EPSILON
        print(
            f"\nGame over. Score: {env.score}. Steps: {t}. Total Trained steps: {tot_times}. \n End status:\n{env.get_board()}"
        )
        save_model()


except:
    print(f"\nInterupted. Total Trained steps: {tot_times}.")
    save_model()
