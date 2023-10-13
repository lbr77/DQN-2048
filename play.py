import tensorflow as tf
import numpy as np
import os
import time
from gym_2048.envs import Game2048Env


# 设置TensorFlow日志级别
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# model = make_model()
model = tf.keras.models.load_model("model.keras")

model.load_weights("model.pkl")

# print(model.summary())

env = Game2048Env()

env.reset()

# env.render(mode="human")
while True:
    os.system("cls")
    env.render(mode="human")
    state = np.array(env.get_board()).reshape(-1, 4, 4, 1)
    if np.random.rand() <= 0.1:
        action = np.random.randint(0, 4)
    else:
        action = np.argmax(model.predict(state)[0])
    env.step(action)
    time.sleep(0.5)
    if env.isend():
        print("Game Over!")
        break
