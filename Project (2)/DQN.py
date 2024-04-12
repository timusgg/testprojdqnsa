import collections
import numpy as np
import random
import tensorflow as tf


class DQNClass:

  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.mainNet = self.createNetwork()
    self.targetNet = self.createNetwork()
    self.updateTargetModel()
    self.memory = deque(maxlen=2000)
    self.batch_size = 32
    self.gamma = 0.95
    self.epsilon = 1.0
    self.epsilonDecay = 0.4
    self.epsilonMin = 0.01
    self.loss = []
    self.actionsPerformed = []
    self.updateFrequency = 200

  def decreaseEpsilonRate(self, episode):
    self.epsilon = self.epsilonMin + (1.0 - self.epsilonMin) * np.exp(
        -self.epsilonDecay * episode)

  def meanSquareError(self, predictedQ, y):
    indices = np.zeros(shape=(self.batch_size, 2))
    indices[:, 0] = np.arange(self.batch_size)
    indices[:, 1] = self.actionsPerformed

    loss = tf.keras.losses.mean_squared_error(
        tf.gather_nd(predictedQ, indices=indices.astype(np.int64)),
        tf.gather_nd(y, indices=indices.astype(np.int64)))
    return loss

  def createNetwork(self):
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Dense(43,
                              input_shape=(self.state_size, ),
                              activation='relu'))
    model.add(tf.keras.layers.Dense(88, activation='relu'))
    model.add(tf.keras.layers.Dense(22, activation='relu'))
    model.add(tf.keras.layers.Dense(11, activation='relu'))
    model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=self.meanSquareError)

    return model

  def updateTargetModel(self):
    self.targetNet.set_weights(self.mainNet.get_weights())

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.randint(self.action_size)
    else:
      return np.argmax(self.mainNet.predict(np.array(state), verbose=0)[0])

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def train(self):
    if len(self.memory) < self.batch_size:
      return

    batch = random.sample(self.memory, self.batch_size)

    currentStatesBatch = np.zeros(shape=(self.batch_size, self.state_size))
    nextStatesBatch = np.zeros(shape=(self.batch_size, self.state_size))

    for i, expTuple in enumerate(batch):
      currentStatesBatch[i, :] = expTuple[0]
      nextStatesBatch[i, :] = expTuple[3]

    currentQvalues = self.mainNet.predict(currentStatesBatch)
    nextQvalues = self.targetNet.predict(nextStatesBatch)

    input = currentStatesBatch
    predictedValues = np.zeros(shape=(self.batch_size, self.action_size))

    self.actionsPerformed = []
    for i, (state, action, reward, next_state, done) in enumerate(batch):
      if done:
        y = reward
      else:
        y = reward + self.gamma * np.amax(nextQvalues[i])

      self.actionsPerformed.append(action)
      predictedValues[i] = currentQvalues[i]
      predictedValues[i, action] = y

    obj = self.mainNet.fit(input,
                           predictedValues,
                           batch_size=self.batch_size,
                           epochs=1,
                           verbose=1)
    loss_values = obj.history['loss']

    return loss_values


class trainedDQN():

  def __init__(self, model):
    self.trainedModel = tf.keras.models.load_model(
        model, custom_objects={'meanSquareError': DQNClass.meanSquareError})

  def act(self, state):
    action = self.trainedModel.act(state)
    return action
