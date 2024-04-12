import networkx as nx

class networkGraph:
    def __init__(self, data):
        self.g = nx.Graph(data.values)
        self.nodes = self.g.nodes
        self.edges = self.extractEdges()
        
    def extractEdges(self):
        edges = []
        for i in range(len(self.nodes)):
            for edge in self.g.edges(i):
                edges.append(edge)
        return edges        
    
    def findPaths(self, source, destination, k):
        P = nx.shortest_simple_paths(self.g, source, destination)
        kSPs = []
        for number, path in enumerate(P):
            links = [l for l in nx.path_graph(path).edges()]
            linkNumber= []
            for link in links:
                linkNumber.append(self.edges.index(link))
            kSPs.append(linkNumber)
            if number == k-1:
                break
        return kSPs

    def findDistances(self, paths):
        distances = {}
        for num,path in enumerate(paths):
            p = []
            for link in path:
                p.extend(self.edges[link])
            #print(p)
            p = list(dict.fromkeys(p))
            distances[num] = nx.path_weight(self.g, p, "weight")

        return distances    


'''

edgelinks = list(enumerate(edges))

def findPaths(graph, s, d):
    P = nx.shortest_simple_paths(graph, s, d)
    kSPs = []
    for number, path in enumerate(P):
        links = [l for l in nx.path_graph(path).edges()]
        linkNumber= []
        for link in links:
            linkNumber.append(edges.index(link))
        kSPs.append(linkNumber)
        if number == k-1:
            break
    return kSPs

def findDistances(paths):
    distances = {}
    for num,path in enumerate(paths):
        p = []
        for link in path:
            p.extend(edges[link])
        #print(p)
        p = list(dict.fromkeys(p))
        distances[num+1] = nx.path_weight(g, p, "weight")

    return distances
'''