import functions
import numpy as np

def defineState(request, ng, k, slotTable):
    stateSize = len(ng.nodes)*2 + k*5
    state = []
    source = request[0]
    destination = request[1]
    bandwidthRequirement = request[2]
    kpaths = ng.findPaths(source, destination, k)
    slotsRequired =  functions.calculateRequiredSlots(ng.findDistances(kpaths), bandwidthRequirement)
    kPathSlots = functions.getPathSlotsfromST(kpaths, slotTable)
    srcOneHot = np.array([1 if source == node else 0 for node in ng.nodes])
    destOneHot = np.array([1 if destination == node else 0 for node in ng.nodes])
    state.extend(srcOneHot)
    state.extend(destOneHot)
    
    for num,path in enumerate(kPathSlots):
        link = path[0]
        freeBlocks = functions.getFSIndexes(link)
        #print(freeBlocks)
        if freeBlocks:
            firstFreeBlockIndex = freeBlocks[0][0]
            sizeOfFirstBlock = len(freeBlocks[0]) 
            averageBlockSize = np.mean([len(block) for block in freeBlocks])
        else:
            firstFreeBlockIndex,sizeOfFirstBlock,averageBlockSize = len(link),0,0.0

        pfragmentation = sum(functions.fragmentation(path))
        state.extend([slotsRequired[num], firstFreeBlockIndex, sizeOfFirstBlock, averageBlockSize, pfragmentation])
    
    state = np.array(state).reshape(1, stateSize)
    return state, kpaths, slotsRequired, kPathSlots
