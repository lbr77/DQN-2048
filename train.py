import os

# 设置TensorFlow日志级别
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Flatten, Dense, concatenate
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from gym_2048.envs import Game2048Env
import sys

tf.get_logger().setLevel("FATAL")
# Functions


def state_process(state):
    state = state.reshape(4, 4)
    state_ = np.where(state <= 0, 1, state)
    _state = np.log2(state_) / np.log2(65536)
    return _state


def convert_to_batch(batch):
    states = np.array([i[0] for i in batch]).reshape(-1, 4, 4, 1)
    actions = np.array([i[1] for i in batch])
    rewards = np.array([i[2] for i in batch])
    next_states = np.array([i[3] for i in batch]).reshape(-1, 4, 4, 1)
    return states, actions, rewards, next_states


def save_model():
    model.save("model1.keras")
    model.save_weights("model1.pkl")
    print("Saved model")
    return


# Hyperparameters
BATCH_SIZE = 256  # size of data used to experience replay
GAMMA = 0.9  # discount factor
EPSILON = 1.0  # exploration rate
EPSILON_MIN = 0.1  # minimum exploration rate
EPSILON_DECAY = 0.99  # decay rate of exploration rate
TARGET_UPDATE = 100  # update target network every 5 episodes
MEMORY_SIZE = 50000  # size of experience replay memory
LEARNING_RATE = 0.0001  # learning rate
EPOCHES = 1000  # number of training epochs
REFRESH_EPSILON_TIMES = 10  # reset epsilon for 10 times.


# Environment
env = Game2048Env()
memory = deque(maxlen=MEMORY_SIZE)

# VARIABLES
INPUT_SHAPE_CNN = (4, 4, 1)
FILTER_NUM_L1 = 512
FILTER_NUM_L2 = 4096

FILTER_SIZE_L1 = 3
FILTER_SIZE_L2 = 1
ACTIVE_FUNC_CNN = "relu"
ACTIVE_FUNC_DENSE = "relu"
ACTIVE_FUNC_OUTPUT = "linear"

OUTPUT_SIZE = 4


# Model
def make_model():
    inputs = Input(shape=INPUT_SHAPE_CNN)
    conv_a = Conv2D(
        filters=FILTER_NUM_L1,
        kernel_size=FILTER_SIZE_L1,
        strides=(2, 1),
        padding="valid",
        activation=ACTIVE_FUNC_CNN,
    )(inputs)
    conv_b = Conv2D(
        filters=FILTER_NUM_L1,
        kernel_size=FILTER_SIZE_L1,
        strides=(1, 2),
        padding="valid",
        activation=ACTIVE_FUNC_CNN,
    )(inputs)
    conv_aa = Conv2D(
        filters=FILTER_NUM_L2,
        kernel_size=FILTER_SIZE_L2,
        strides=(2, 1),
        padding="valid",
        activation=ACTIVE_FUNC_CNN,
    )(conv_a)
    conv_ab = Conv2D(
        filters=FILTER_NUM_L2,
        kernel_size=FILTER_SIZE_L2,
        strides=(1, 2),
        padding="valid",
        activation=ACTIVE_FUNC_CNN,
    )(conv_a)
    conv_ba = Conv2D(
        filters=FILTER_NUM_L2,
        kernel_size=FILTER_SIZE_L2,
        strides=(2, 1),
        padding="valid",
        activation=ACTIVE_FUNC_CNN,
    )(conv_b)
    conv_bb = Conv2D(
        filters=FILTER_NUM_L2,
        kernel_size=FILTER_SIZE_L2,
        strides=(1, 2),
        padding="valid",
        activation=ACTIVE_FUNC_CNN,
    )(conv_b)
    merge = concatenate(
        [Flatten()(x) for x in [conv_aa, conv_ab, conv_ba, conv_bb, conv_a, conv_b]]
    )
    outputs = Dense(OUTPUT_SIZE, activation=ACTIVE_FUNC_OUTPUT)(merge)
    model = Model(inputs=inputs, outputs=outputs)
    return model


model = make_model()
target_model = make_model()
print(model.summary())
try:
    model.load_weights("model1.pkl")
    print("Loaded model")
except:
    print("No model found")

target_model.set_weights(model.get_weights())

# Optimizer
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="mse")

# Training
writer = tf.summary.create_file_writer("./logs")

# EPSILON = 0.1
global tot_times
# tot_times = 53460 + 99700
tot_times = 0
try:
    for epoch in range(EPOCHES):
        env.reset()
        t = 0
        while True:
            t += 1
            tot_times += 1
            sys.stdout.flush()
            # Get Action
            state = np.array(state_process(env.get_board())).reshape(-1, 4, 4, 1)
            if np.random.rand() <= EPSILON:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(model.predict(state, verbose=0)[0])
            # Do Action
            next_state, reward, done, _ = env.step(action)
            # Save Action
            next_state = np.array(state_process(next_state)).reshape(-1, 4, 4, 1)
            memory.append((state, action, reward, next_state, done))
            # Get Batches from Action.
            batch = []
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
            else:
                batch = memory
            states, actions, rewards, next_states = convert_to_batch(batch)
            # Lower Q's from previous models.
            next_q_val = target_model.predict(next_states, verbose=0)
            max_next_q_val = np.max(next_q_val, axis=1)
            # Target Q from Bellman Equation.
            # print(rewards.shape, max_next_q_val.shape, next_q_val.shape, states.shape)
            target_q_val = rewards + (1 - done) * GAMMA * max_next_q_val
            # Model fit ?
            history = model.fit(states, target_q_val, verbose=0)
            # Finish !
            if env.isend():
                break
            if EPSILON > EPSILON_MIN:
                EPSILON *= EPSILON_DECAY
            if tot_times % TARGET_UPDATE == 0:  # update target network
                target_model.set_weights(model.get_weights())
            sys.stdout.write(
                f"\r\rEpoch: {epoch}. Steps: {t}. Exploration rate: {EPSILON:.2f}. Current Score: {env.score}. "
            )
            if tot_times % 5 == 0:
                with writer.as_default():
                    tf.summary.scalar("score", env.score, step=tot_times)
                    tf.summary.scalar("epsilon", EPSILON, step=tot_times)
                    tf.summary.scalar(
                        "learning_rate", optimizer.learning_rate.numpy(), step=tot_times
                    )
                    tf.summary.scalar(
                        "loss", history.history["loss"][0], step=tot_times
                    )
                    tf.summary.scalar(
                        "Q expected", np.sum(target_q_val), step=tot_times
                    )
                    tf.summary.scalar("Q caculated", np.sum(next_q_val), step=tot_times)
                    # tf.summary.scalar("reward", t, step=tot_times)
                    writer.flush()
        if (
            epoch % REFRESH_EPSILON_TIMES == 0 and epoch < 50
        ):  # We use the first 0.1 epoches to explore with high probability.
            EPSILON = 1.0  # RESET EPSILON
        print(
            f"\nGame over. Score: {env.score}. Steps: {t}. Total Trained steps: {tot_times}. \n End status:\n{env.get_board()}"
        )
        save_model()


except Exception as e:
    print(f"\nInterupted. Total Trained steps: {tot_times}.")
    print(f"{e}")
    save_model()
# # train()
