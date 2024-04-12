import math
import random
import numpy as np

#nestedList = [[0 for _ in range(16)] for _ in range(3)] 
link = list(np.random.randint(0,2, size=16))
path = [link for _ in range(3)]

BitRateAndDistances = { (0, 150): 333,
    (150, 300): 277.5,
    (300, 600): 222,
    (600, 1200): 166.5,
    (1200, 3500): 111,
    (3500, 6300): 55.5,
    (6300, 20000): 55.5}

def performOR(available, lst):
    available = [x or y for x,y in zip(available, lst)]
    return available

def getFreeSlots(listofLinks):
    zeroList = [0 for _ in range(len(listofLinks[0]))]
    for link in listofLinks:
        zeroList = performOR(zeroList, link)
    return zeroList

def getFSIndexes(link):
    x = 0
    freeslots = []
    slotsBlock = []
    for slot in range(len(link)):
        if link[slot] == 0 :
            x += 1
            slotsBlock.append(slot)
        else :
            if x > 0:
                freeslots.append(slotsBlock)
                slotsBlock = []
            x = 0
    if x > 0:
        freeslots.append(slotsBlock)


    return freeslots


#result = getFreeSlots(nestedList)
#print('result =\n', result)

#print(getFSIndexes(result))


def fragmentation(path):
    #print(path)
    pathFragmentation = []
    for link in path:
        linkFragmentation = 0
        sigmaBS = 0
        freeSlotsBlocks = getFSIndexes(link)
        fMax = [i for i in range((len(link)-1), 0, -1) if link[i] == 1]
        if len(fMax) == 0:
            fMax = 1
        else:
            fMax = max(fMax) + 1

        #print('freeslotsblock', freeSlotsBlocks)
        alpha = len(freeSlotsBlocks)
        #print(alpha, 'alpha')

        if alpha == 0:

            alpha = 1
            sigmaBS = 1
        else:
            
            #print('fmax',fMax)
            for block in freeSlotsBlocks:
                sigmaBS += len(block) ** 2
            #print('sigmsBS', sigmaBS)
        
        linkFragmentation = (fMax * alpha) / (math.sqrt(sigmaBS / alpha))

        #print('linkfrag', linkFragmentation)
        pathFragmentation.append(linkFragmentation)
    
    #print(pathFragmentation, 'pathfrag')
    return pathFragmentation


def checkAvailability(path, requiredSlots):
    freeSlots = getFSIndexes(getFreeSlots(path))
    freeBlock = next((slotBlock for slotBlock in freeSlots if len(slotBlock) >= requiredSlots), None)
    return freeBlock



def firstFit(path, requiredSlots):
    firstfreeBlock = checkAvailability(path, requiredSlots)
    slots_index = []
    if firstfreeBlock:
        for link in path:
            for slot in firstfreeBlock[:requiredSlots]:
                link[slot] = 1
                slots_index.append(slot)
            #print(link)
        return 1/(1+sum(fragmentation(path))) , set(slots_index)
    else:
        return -0.25, None 
   
'''
def lastFit(path, requiredSlots):
    freeSlots = getFSIndexes(getFreeSlots(path))
    lastfreeBlock = next((slotBlock for slotBlock in reversed(freeSlots) if len(slotBlock) >= requiredSlots), None)
    slots_index = []
    if lastfreeBlock:
        for link in path:
            for slot in range(len(lastfreeBlock)-1,len(lastfreeBlock)-requiredSlots-1,-1):
                link[lastfreeBlock[slot]] = 1
                slots_index.append(slot)
            #print(link)
        return 1/(1+sum(fragmentation(path))) , set(slots_index)
'''

def randomFit(path, requiredSlots):
    freeSlots = getFSIndexes(getFreeSlots(path))
    freeBlocks = [slotBlock for slotBlock in freeSlots if len(slotBlock) >= requiredSlots]
    slots_index = []

    if freeBlocks:
        randomBlock = random.choice(freeBlocks)
        startIndex = random.randint(0, len(randomBlock) - requiredSlots)
        randomFreeSlots = randomBlock[startIndex:startIndex + requiredSlots]
        for link in path:
            for slot in randomFreeSlots:
                link[slot] = 1
                slots_index.append(slot)
                
        return 1/(1+sum(fragmentation(path))) , set(slots_index)

def exactFit(path, requiredSlots):
    freeSlots = getFSIndexes(getFreeSlots(path))
    exactBlock = next((slotBlock for slotBlock in freeSlots if len(slotBlock) == requiredSlots), None)
    if exactBlock:
        slots_index = []
        for link in path:
            for slot in exactBlock:
                link[slot] = 1
                slots_index.append(slot)    
        #print('EX')
        return 1 / (1 + sum(fragmentation(path))), set(slots_index)
    else:
        #print('FF')
        reward, slots = firstFit(path, requiredSlots)
        return reward, slots
    

def networkFragmentation(slotTable):
    sigmaFe = sum(fragmentation(slotTable))
    smax = 0
    for index in range((len(slotTable)-1), -1, -1):
        lst = [index for link in slotTable if link[index] == 1]   
        if len(lst) != 0:
            smax = lst[0] + 1
            break

    S = len(slotTable[0])
    E = len(slotTable)
    #print(smax, S, E, sigmaFe)

    return (sigmaFe/E)*(smax/S)


def reset(slotTable):
    for link in slotTable:
        for slot in range(len(link)):
            link[slot] = 0
    return slotTable


def releaseSlots(links, indexes):
    for link in links:
        for index in indexes:
            link[index] = 0
    

def getPathSlotsfromST(pathList, slotTable):
    pathSlots = []
    for path in pathList:
        slotTableLink = []
        for link in path:
            slotTableLink.append(slotTable[link])
        pathSlots.append(slotTableLink)
    return pathSlots

def getLinkSlotsfromST(path, slotTable):
    slotTableLink = []
    for link in path:
        slotTableLink.append(slotTable[link])

    return slotTableLink

def addToActiveRequests(activeRequestsList, id, currentTime, holdingTime, linkNumbers, occupiedIndexes):
    activeRequestsList[id] = [currentTime, holdingTime]
    activeRequestsList[id].append(linkNumbers)
    activeRequestsList[id].append(occupiedIndexes)

def checkActiveRequests(currentTime, activeRequestList, slotTable):
    for req in list(activeRequestList):
        arrival_time, holding_time = activeRequestList[req][0], activeRequestList[req][1]
        if currentTime >= (arrival_time + holding_time):
            linksUsed = getLinkSlotsfromST(activeRequestList[req][2], slotTable)
            slotsIndexes = activeRequestList[req][3]
            releaseSlots(linksUsed, slotsIndexes)
            #print('released request Number', req)
            del activeRequestList[req]

def performAllocation(policy, id, currentTime, holdingTime, pathList, pathSlots, slots_required, slotTable, activeRequests):
    availableShortestPath = []
    slots_req = 0
    occupiedSlotsIndexes = []
    for num, path in enumerate(pathSlots):
        availability = checkAvailability(path, slots_required[num])
        if availability is not None:
            availableShortestPath = pathList[num]
            slots_req = slots_required[num]
            break

    #print(availableShortestPath)

    if availableShortestPath:
        path = getLinkSlotsfromST(availableShortestPath, slotTable)
        if policy == 'FF':
            occupiedSlotsIndexes = firstFit(path, slots_req)[1]
            #print('served Request', id)
        elif policy == 'RF':
            #print('served Request', id)
            occupiedSlotsIndexes = randomFit(path, slots_req)[1]
        elif policy == 'EF':
            occupiedSlotsIndexes = randomFit(path, slots_req)[1]

        addToActiveRequests(activeRequests, id, currentTime, holdingTime, availableShortestPath, occupiedSlotsIndexes)
    else:
        return True  



def calculateRequiredSlots(distances, bandwidth):
    slotsRequired = {}
    for path in distances:
        for transDistance, bitRate in BitRateAndDistances.items():
            if transDistance[0] <= distances[path] < transDistance[1]:
                #print(bitRate)
                slotsRequired[path] = math.ceil(bandwidth/bitRate) * 3
    #print(slotsRequired)
    return slotsRequired






'''
def performAllocationstatic(policy, id, pathList, pathSlots, slots_required, slotTable):
    availableShortestPath = []
    for num, path in enumerate(pathSlots):
        availability = checkAvailability(path, slots_required)
        if availability is not None:
            availableShortestPath = pathList[num]
            break

    #print(availableShortestPath)

    if availableShortestPath:
        path = getLinkSlotsfromST(availableShortestPath, slotTable)
        if policy == 'FF':
            occupiedSlotsIndexes = firstFit(path, slots_required)[1]
            print('served Request', id)
        elif policy == 'LF':
            occupiedSlotsIndexes = lastFit(path, slots_required)[1]
            print('served Request', id)
        elif policy == 'RF':
            occupiedSlotsIndexes = randomFit(path, slots_required)[1]

        
    else:
        return True  


slots = random.randint(1,4)

nestedList1 = copy.deepcopy(nestedList)
nestedList2 = copy.deepcopy(nestedList)

for l in nestedList:
    print(l, 'Before')

for l in nestedList1:
    print(l, 'Before')

for l in nestedList2:
    print(l, 'Before')


for r in range(5):
    iFF = firstFit(nestedList,r)[1]
    iRF = randomFit(nestedList2,r)[1]
    iLF = lastFit(nestedList1, r)[1]

for l in nestedList:
    print(l)


for l in nestedList1:
    print(l)

for l in nestedList2:
    print(l)

'''


