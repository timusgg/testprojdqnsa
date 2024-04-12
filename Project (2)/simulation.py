import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import copy
import functions
from DQN import DQNClass, trainedDQN
from graph_functions import networkGraph
from state import defineState

def simulation(topology:str, totalRequests:int, spectrumSlots:int, arrivalRate:int, serviceRate:int, k:int, model=False):
    if model != False:
        try:
            trainedAgent = trainedDQN(model)
        except:
            print("Problem Encountered while loading the model, please check Model file, exiting......")
            exit()

    try:
        data = pd.read_csv(topology, index_col=0)
    except:
        print("Check whether the filename is correct and file is present the current directory !!! Exiting with error.")
        exit()

    #print(data.values)

    graph = networkGraph(data)
    nodes = graph.nodes

    requests = []
    requestNum = 0

    bandwidth = [25, 50, 75, 100]
    stateSize = len(nodes)*2 + k*5

    actionSize = k*2

    actionInfo = []

    for n in range(k):
        actionInfo.append((functions.firstFit,n))
        actionInfo.append((functions.exactFit,n))

    for i in range(totalRequests):
        re = []
        arrivalTime = np.random.poisson(arrivalRate, episodeSize)
        arrivalTime.sort()
        holdingTime = np.random.poisson(serviceRate, episodeSize)
        #bandwidth = random.choices(bandwidthRequirement, k = episodeSize)
        sdPair = random.sample(nodes, 2)
        #value = [3,6,9,12,15,18,21]
        bandwidthRequirement = random.choice(bandwidth)
        sdPair.extend([bandwidthRequirement, arrivalTime[i], holdingTime[i]])
        re.append(sdPair)
        requests.append(re)

    #edgelinks = list(enumerate(graph.edges))


    slotTableDQNSA = []
    for _ in range(len(graph.edges)):
        slots = [0 for _ in range(spectrumSlots)]
        slotTableDQNSA.append(slots)

    slotTableFF = copy.deepcopy(slotTableDQNSA)
    slotTableEF = copy.deepcopy(slotTableDQNSA)
    slotTableRF = copy.deepcopy(slotTableDQNSA)

    slotTables = {'FF': slotTableFF, 'EF': slotTableEF, 'RF': slotTableRF}

    policies = ['FF', 'EF', 'RF']


    activeRequestsDQNSA = {}
    activeRequests = {'FF':{}, 'EF':{}, 'RF':{}}

    blockedRequestsDQNSA = 0
    blockedRequests = {'FF' : 0, 'EF' : 0, 'RF' : 0}

    BlockingRatios = {"NumberOfRequests" : totalRequests}

    for id, r in enumerate(requests):
        requestNum += 1
        currentRequest = r
        current_time = r[3]
        holding_time = r[4]
        state, kpaths, slotsRequired, pathSlotsDQNSA = defineState(currentRequest, graph, k, slotTableDQNSA)
        pathSlots = {}

        for policy in policies:
            pathSlots[policy] = functions.getPathSlotsfromST(kpaths, slotTables[policy])
            isBlocked = functions.performAllocation(policy, id, current_time, holding_time, kpaths, pathSlots[policy], slotsRequired, slotTables[policy], activeRequests[policy])
            if isBlocked:
                blockedRequests[policy] += 1

        action = trainedAgent.act(state)

        selectedPolicy, selectedPath = actionInfo[action]

        linkNumbers = kpaths[selectedPath]

        reward, occupiedSlotsIndexes = selectedPolicy(pathSlotsDQNSA[selectedPath], slotsRequired[selectedPath])

        if occupiedSlotsIndexes is not None:
            functions.addToActiveRequests(activeRequestsDQNSA, id, current_time, holding_time, linkNumbers, occupiedSlotsIndexes)

        if reward == -0.25:
            blockedRequestsDQNSA += 1


        #network_fragmentation = functions.networkFragmentation(slotTable)
        #print(network_fragmentation)
        '''
        details = [source, destination, slotsRequired[selectedPath], state, action, reward, nextState]
        for i in range(len(details)):
            history[i].append(details[i])
        '''

        functions.checkActiveRequests(current_time, activeRequestsDQNSA, slotTableDQNSA)
        for policy in policies:
            functions.checkActiveRequests(current_time, activeRequests[policy], slotTables[policy])

        #print(blockedRequests)

    #print(blockedRequests)
    BR = blockedRequestsDQNSA/totalRequests

    for policy in policies:
        ratio = blockedRequests[policy]/totalRequests
        BlockingRatios[policy] = ratio

    BlockingRatios['DQNSA'] = BR 

    return BlockingRatios

    '''
    dictData = {item[0]:item[1:] for item in history}

    dataframe = pd.DataFrame(dictData)

    dataframe.to_excel('history.xlsx', index=True)
    '''
    
def run():
    BlockingRatios = {}
    policies = ["FF", "EF", "RF", "DQNSA"]
    for num in range(100, 1000, 100):
        result = simulation("nsfnet.csv", num, spectrumSlots=50, arrivalRate=12, serviceRate=14, k=5, model="trainedNetwork.h5")
        for key, value in result.items():
            if key not in BlockingRatios:
                BlockingRatios[key] = value
            else:
                BlockingRatios[key].append(value)

    data = pd.DataFrame(BlockingRatios)
    data.to_excel("SimulationBlockingRatios.xlsx", index=False)

    for policy in policies:
        plt.plot(data["NumberOfRequests"], data[policy])
    

    plt.xlabel("Requests")
    plt.ylabel("Policies")
    plt.legend()
    plt.show()

run()
