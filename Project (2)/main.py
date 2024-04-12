
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import functions
import tensorflow as tf
from collections import deque

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.buildModel()
        self.target_model = self.buildModel()
        self.updateTargetModel()
        self.memory = deque(maxlen=300)  # Experience replay buffer
        self.batch_size = 50
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilonDecay = 0.1  # Decay factor
        self.epsilonMin = 0.01  # Minimum exploration rate
        self.loss = []

    def decreaseEpsilonRate(self, episode):
        self.epsilon = self.epsilonMin + (1.0 - self.epsilonMin) * np.exp(-self.epsilonDecay*episode)



    def buildModel(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_shape=(self.state_size, ), activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.model.predict(np.array(state), verbose=0)[0])

    '''
    def train(self, state, action, reward, next_state,done):
        target = reward if done else reward + 0.95 * np.amax(self.target_model.predict(np.array([next_state]),verbose=0)[0])
        target_full = self.model.predict(np.array(state))
        target_full[0][action] = target
        self.model.fit(np.array(state), target_full, epochs=1, verbose=0)
    '''


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)

        #loss = []

        for state, action, reward, next_state, done in batch:
            target = reward if done else reward + 0.95 * np.amax(self.target_model.predict(np.array(next_state), verbose=0)[0])
            target_full = self.model.predict(np.array(state), verbose = 0)
            target_full[0][action] = target
            obj = self.model.fit(np.array(state), target_full, epochs=1, verbose=1)
            lossValue = obj.history['loss'][0]
            self.loss.append(lossValue)
        

data = pd.read_csv('topology5node.csv', index_col=0)
#print(data.values)


g = nx.Graph(data.values)
nodes = g.nodes
requests = []


#nx.draw_networkx(g, with_labels=True)
#plt.draw()

#plt.show()

epsilon = 1.0
numberofRequest = 500

spectrumSlots = 100

episodeSize = 25
episodeNum = 0

requestNum = 0
threshold = 0.1


agent = DQN(12, 2)

blockedRquests = 0
lossPerBatch = []

averageReward = []
BRperEpisode = []
totalBR = 0
totalReward = 0
totalBlocked = 0



for _ in range(numberofRequest):
    sdPair = random.sample(nodes, 2)
    slots = random.randint(1,4)
    sdPair.append(slots)
    requests.append(sdPair)



edges = []

for i in range(len(nodes)):
    for edge in g.edges(i):
        edges.append(edge)

#print(edges)

edgelinks = list(enumerate(edges))

#print(edgelinks)



#slot_table
slotTable = []
for x in range(len(edges)):
    slots = [0 for _ in range(spectrumSlots)]
    slotTable.append(slots)

#print(slotTable)


def findPaths(graph, s, d):
    P = nx.shortest_simple_paths(graph, s, d)
    kSPs = []
    k = 3
    for number, path in enumerate(P):
        links = [l for l in nx.path_graph(path).edges()]
        linkNumber= []
        for link in links:
            linkNumber.append(edges.index(link))
        kSPs.append(linkNumber)
        if number == k-1:
            break
    return kSPs


def getPathsFromST(pathList):
    pathSlots = []
    for path in pathList:
        slotTableLink = []
        for link in path:
            slotTableLink.append(slotTable[link])
        pathSlots.append(slotTableLink)
    return pathSlots

def getLinksFromST(path):
    slotTableLink = []
    for link in path:
        slotTableLink.append(slotTable[link])

    return slotTableLink



history = [['Source'], ['Destination'], ['Required Slots'], ['State'], ['Action'], ['Reward'], ['Next State'], ['Done']]



for r in requests:
    #print(r)
    requestNum += 1
    source = r[0]
    destination = r[1]
    slots_required = r[2]
    pathList = findPaths(g, source, destination)

    #print(pathList)
    paths = getPathsFromST(pathList) 


    linkFragmentation = []
    for num, path in enumerate(paths):
        availability = functions.checkAvailability(path, slots_required)
        #print(availability)
        if availability is not None:
            fragmentation = functions.fragmentation(path)
            linkFragmentation.append([fragmentation, pathList[num]])

    if linkFragmentation:
        pathFragmentation = [sum(item[0]) for item in linkFragmentation]
        lowestFragmentation = [item for item in linkFragmentation if sum(item[0]) == min(pathFragmentation)]
    
        #print(lowestFragmentation)
    
        if lowestFragmentation:
            lowestFragmentation = lowestFragmentation[0]
            path = getLinksFromST(lowestFragmentation[1])


            state = np.array(functions.defineState(lowestFragmentation[0], lowestFragmentation[1], len(edges))).reshape(1, 12)
            action = agent.act(state)
    
          #print(state)
        
        action = 0

        if action == 0:
            reward = functions.firstFit(path, slots_required)
            #print('firstfit', reward) 
        elif action == 1:
            reward = functions.lastFit(path, slots_required)
            #print('lastfit', reward)



        #print(path)
        newFragmentation = functions.fragmentation(path)
        #print(reward, newFragmentation)

        nextState = np.array(functions.defineState(newFragmentation, lowestFragmentation[1], len(edges))).reshape(1, 12)
        #print(nextState)
    else:
        reward = -1
    
    totalReward += reward
    
    if reward == -1:
        blockedRquests += 1

    BR = blockedRquests/episodeSize
    totalBlocked += blockedRquests
    
    done = False
    #network_fragmentation = functions.networkFragmentation(slotTable)
    #print(network_fragmentation) 
    totalReward += reward
    
    if requestNum == episodeSize or BR > threshold:
        
        BRperEpisode.append(totalBR) 
        episodeNum += 1
        done = True
        averageReward.append(totalReward)
        requestNum = 0
        blockedRquests = 0
        agent.decreaseEpsilonRate(episodeNum)

    
    agent.remember(state, action, reward, nextState, done)
    details = [source, destination, slots_required, state, action, reward, nextState, done]
    for i in range(len(details)):
        history[i].append(details[i])

    
    
    if len(agent.memory) > 150:
        agent.replay()
        

    if episodeNum == 8:
        agent.updateTargetModel()
        episodeNum = 0
    #print(slotTable)
    


dictData = {item[0]:item[1:] for item in data}

dataframe = pd.DataFrame(dictData)

dataframe.to_excel('test.xlsx', index=True)


#print(totalBlocked)






'''
plt.plot(loss)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.show()

plt.plot(agent.loss)
plt.xlabel('request')
plt.ylabel('loss')
plt.show()

plt.plot(BRperEpisode)
plt.xlabel('episode')
plt.ylabel('reward')
plt.show()


'''




'''

for num,link in enumerate(slotTable):
    print(num,link)


'''