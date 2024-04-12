import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import copy
import functions
from DQN import DQNClass
from graph_functions import networkGraph
from state import defineState

data = pd.read_csv('nsfnet.csv', index_col=0)
#print(data.values)

graph = networkGraph(data)
nodes = graph.nodes

requests = []

totalRequests = 200

spectrumSlots = 100

episodeSize = 100

episodeNum = 0

requestNum = 0

threshold = 0.1

arrivalRate = 12

serviceRate = 14

bandwidth = [25, 50, 75, 100]

k = 3

stateSize = len(nodes)*2 + k*5

actionSize = k*2

actionInfo = []

for n in range(k):
    actionInfo.append((functions.firstFit,n))
    actionInfo.append((functions.exactFit,n))

agent = DQNClass(stateSize, actionSize)


lossPerEpisode = []

BRperEpisode = {'Episode': [], 'FF' : [], 'EF' : [], 'RF' : [], 'DQNSA' : []}

rewardPerEpisode = []

totalBlocked = 0
history = [['Source'], ['Destination'], ['Required Slots'], ['State'], ['Action'], ['Reward'], ['Next State']]


for _ in range(int(totalRequests / episodeSize)):
    re = []
    arrivalTime = np.random.poisson(arrivalRate, episodeSize)
    arrivalTime.sort()
    holdingTime = np.random.poisson(serviceRate, episodeSize)
    #bandwidth = random.choices(bandwidthRequirement, k = episodeSize)
    for i in range(episodeSize):
        sdPair = random.sample(nodes, 2)
        #value = [3,6,9,12,15,18,21]
        bandwidthRequirement = random.choice(bandwidth)
        sdPair.extend([bandwidthRequirement, arrivalTime[i], holdingTime[i]])
        re.append(sdPair)    
    requests.append(re)

edgelinks = list(enumerate(graph.edges))



slotTableDQNSA = []
for x in range(len(graph.edges)):
    slots = [0 for _ in range(spectrumSlots)]
    slotTableDQNSA.append(slots)

slotTableFF = copy.deepcopy(slotTableDQNSA)
slotTableEF = copy.deepcopy(slotTableDQNSA)
slotTableRF = copy.deepcopy(slotTableDQNSA)

slotTables = {'FF': slotTableFF, 'EF': slotTableEF, 'RF': slotTableRF}

policies = ['FF', 'EF', 'RF']


for episode in requests:
    episode = list(enumerate(episode,1))
    episodeRewards = []
    loss = []
    BR = 0
    blockedRequestsDQNSA = 0
    blockedRequests = {'FF' : 0, 'EF' : 0, 'RF' : 0}
    episodeNum += 1
    done = False
    activeRequestsDQNSA = {}
    activeRequests = {'FF':{}, 'EF':{}, 'RF':{}}
    for id, r in episode:
        requestNum += 1
        currentRequest = r
        source, destination = r[0], r[1]
        current_time = r[3]
        holding_time = r[4]
        state, kpaths, slotsRequired, pathSlotsDQNSA = defineState(currentRequest, graph, k, slotTableDQNSA)
        pathSlots = {}
        if id < episodeSize:
            nextRequest = episode[id][1]
        else:
            nextRequest = currentRequest
        
        
        for policy in policies:

            pathSlots[policy] = functions.getPathSlotsfromST(kpaths, slotTables[policy])
            
            isBlocked = functions.performAllocation(policy, id, current_time, holding_time, kpaths, pathSlots[policy], slotsRequired, slotTables[policy], activeRequests[policy])
            if isBlocked:
                blockedRequests[policy] += 1

        ''' 
        pathSlots['FF'] = functions.getPathSlotsfromST(pathList, slotTables['FF'])
            
        isBlocked = functions.performAllocation('FF', id, current_time, holding_time, pathList, pathSlots['FF'], slotsRequired, slotTables['FF'], activeRequests['FF'])
        if isBlocked:
            blockedRequests['FF'] += 1
        '''
    
        action =  agent.act(state)
        
        selectedPolicy, selectedPath = actionInfo[action]

        linkNumbers = kpaths[selectedPath]

        reward, occupiedSlotsIndexes = selectedPolicy(pathSlotsDQNSA[selectedPath], slotsRequired[selectedPath])
        

        if occupiedSlotsIndexes is not None:
            functions.addToActiveRequests(activeRequestsDQNSA, id, current_time, holding_time, linkNumbers, occupiedSlotsIndexes)
            
        if reward == -0.25:
            blockedRequestsDQNSA += 1
            #blocknew += 1

        
        if requestNum%episodeSize == 0 and BR <= threshold: 
            reward += 1
            done = True


        #network_fragmentation = functions.networkFragmentation(slotTable)
        #print(network_fragmentation)
        episodeRewards.append(reward)
        nextState = defineState(nextRequest, graph, k, slotTableDQNSA)[0]
        
        agent.remember(state, action, reward, nextState, done)
        
        details = [source, destination, slotsRequired[selectedPath], state, action, reward, nextState]
        for i in range(len(details)):
            history[i].append(details[i])

        if BR >= threshold:
            agent.decreaseEpsilonRate(episodeNum)        

        if len(agent.memory) > episodeSize*2:
            loss.append(agent.train())

        if requestNum%agent.updateFrequency == 0:
            agent.updateTargetModel()
        
        functions.checkActiveRequests(current_time, activeRequestsDQNSA, slotTableDQNSA)
 
        '''
        for policy in policies:
            functions.checkActiveRequests(current_time, activeRequests[policy], slotTables[policy])
        '''
        #print(blockedRequests)

    #print(blockedRequests)

    agent.decreaseEpsilonRate(episodeNum)
    print('Episode Number ', episodeNum)

    rewardPerEpisode.append(np.mean(episodeRewards))
    lossPerEpisode.append(np.mean(loss))


    BR = blockedRequestsDQNSA/episodeSize
    
    for policy in policies:
        ratio = blockedRequests[policy]/episodeSize
        BRperEpisode[policy].append(ratio)
    
    BRperEpisode['Episode'].append(episodeNum)
    BRperEpisode['DQNSA'].append(BR)

    #print(BRperEpisode)
    functions.reset(slotTableDQNSA)
    
    #print(slotTables['FF'])    
    for policy in policies:
        #print(slotTables[policy])
        functions.reset(slotTables[policy])

    #print(BRperEpisode)
    #counter =+1



dictData = {item[0]:item[1:] for item in history}

dataframe = pd.DataFrame(dictData)

dataframe.to_excel('history.xlsx', index=True)

lossRewardData = pd.DataFrame({'LossPerEpisode': lossPerEpisode, 'RewardPerEpisode': rewardPerEpisode})

lossRewardData.to_excel("AverageLossAndReward.xlsx", index=False)

print(BRperEpisode)
blockingRatioData = pd.DataFrame(BRperEpisode)

blockingRatioData.to_excel('BlockingRatios.xlsx', index=True)

agent.mainNet.save('trainedNetwork.h5')

blockingRatioData.head()
blockingRatioData.plot(x = 'Episode', title='Blocking Ratios')

plt.show()

plt.figure(1)
plt.plot(lossPerEpisode)
plt.xlabel('Episode')
plt.ylabel('Average Loss')

plt.figure(2)
plt.plot(rewardPerEpisode)
plt.xlabel('Episode')
plt.ylabel('Average Reward')

plt.show() 
