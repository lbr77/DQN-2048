import tensorflow as tf
import numpy as np
import os
import time
from gym_2048.envs import Game2048Env


# def make_model():
#     ipt = tf.keras.layers.Input(shape=(4, 4, 1))
#     x = tf.keras.layers.Conv2D(256, 2, activation="relu")(ipt)
#     x = tf.keras.layers.Conv2D(256, 2, activation="relu")(x)
#     x = tf.keras.layers.Flatten()(x)
#     x = tf.keras.layers.Dense(256, activation="relu")(x)
#     actions = tf.keras.layers.Dense(4, activation="linear")(x)
#     model = tf.keras.Model(inputs=ipt, outputs=actions)
#     return model


# model = make_model()
model = tf.keras.models.load_model("models/dqn.h5")
model.load_weights("models/dqn_weights_best.pkl")

print(model.summary())

env = Game2048Env()

env.reset()

# env.render(mode="human")
while True:
    env.render(mode="human")
    state = np.array(env.get_board()).reshape(-1, 4, 4, 1)
    # action = np.argmax(model(state)[0])
    action = 0
    if np.random.rand() > 0.9:
        action = np.argmax(model(state)[0])

    else:
        action = np.random.randint(0, 4)
    env.step(action)
    time.sleep(0.5)

    if env.isend():
        exit(0)
    os.system("cls")
