import math
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time 

def euclid(p,q):
    x = p[0]-q[0]
    y = p[1]-q[1]
    return math.sqrt(x*x+y*y)
                
class Graph:

    # See the attached graphs for examples of the input format. We take care of two cases:
    # the -1 case, where we read points in the Euclidean plane, and
    # the n>0 case, where we read a general graph in a different format.
    # self.perm, self.dists, self.n are the key variables to be set up.
    def __init__(self,n,filename):
        
        F = open(filename)
        lines = F.readlines()
        
        if (n == (-1)):
            # print("Euclidean TSP")
            
            self.n = len(lines)
            
            # Converting the lines into lists of length 2
            coords = [[int(s) for s in lines[i].split()] for i in range(self.n)]
            
            # Initializing a "table" of distances
            self.dists = [[euclid(coords[i] , coords[j]) for i in range(self.n)] for j in range(self.n)]
        else:
            # print("General TSP")
            
            self.n = n
            
            # Converting the lines into lists of length 3
            triples = [[int(s) for s in lines[i].split()] for i in range((self.n * (self.n - 1)) // 2)]
            
            # Initializing the distances with 0s for all pairs of nodes, to be updated once we consider edges.
            # The 0s will only remain on the main diagonal. 
            self.dists = [[0 for i in range(self.n)] for j in range(self.n)]
            
            
            for triple in triples:
                self.dists[triple[0]][triple[1]] = triple[2]
                self.dists[triple[1]][triple[0]] = triple[2]
            
        # Initialize self.perm to be the identity, regardless of what type of TSP we are dealing with 
        self.perm = [i for i in range(self.n)]

    # Complete as described in the spec, to calculate the cost of the
    # current tour (as represented by self.perm).
    def tourValue(self):
       return sum( [self.dists[self.perm[i]][self.perm[(i+1) % self.n]] for i in range (self.n)] )
       
    # Attempt the swap of cities i and i+1 in self.perm and commit
    # commit to the swap if it improves the cost of the tour.
    # Return True/False depending on success.
    def trySwap(self,i):
        # for n greater than 2, we swap vertices i and (i+1) if the cost to get from (i-1) to (i+2) is reduced by swapping
        # the two vertices. The rest of the cost will be unaffected. This does not generally work for n=2. 
        # For n < 2, swaping anything doesn't make any sense whatsoever, so we just return 0 (false).
        if self.n < 2:
            
            return 0
        
        else: 
            if self.n == 2:
            
                if self.dists[self.perm[0]][self.perm[1]] > self.dists[self.perm[1]][self.perm[0]]:
                    tempVar = self.perm[0]
                    self.perm[0] = self.perm[1]
                    self.perm[1] = tempVar
                    return 1
                else:
                    return 0
            
            else: # n > 2
            
                # We compute the relevant step costs.
                minus1ToI     = self.dists[self.perm[(i-1) % self.n]][self.perm[i]]
                iToPlus1      = self.dists[self.perm[i]][self.perm[(i+1) % self.n]]
                plus1ToPlus2  = self.dists[self.perm[(i+1) % self.n]][self.perm[(i+2) % self.n]]
                minus1ToPlus1 = self.dists[self.perm[(i-1) % self.n]][self.perm[(i+1) % self.n]]
                plus1ToI      = self.dists[self.perm[(i+1) % self.n]][self.perm[i]]
                iToPlus2      = self.dists[self.perm[i]][self.perm[(i+2) % self.n]]
            
                # We compare the costs to get from vertex (i-1) to (i+2)
                oldCost = minus1ToI + iToPlus1 + plus1ToPlus2
                newCost = minus1ToPlus1 + plus1ToI + iToPlus2
        
                if newCost < oldCost:
                    tempVar = self.perm[i]
                    self.perm[i] = self.perm[(i+1) % self.n] 
                    self.perm[(i+1) % self.n] = tempVar
                    return 1
                else:
                    return 0
            
        


    # Consider the effect of reversiing the segment between
    # self.perm[i] and self.perm[j], and commit to the reversal
    # if it improves the tour value.
    # Return True/False depending on success.              
    def tryReverse(self,i,j):
        # we replace the edges from (i-1) to i and j to (j+1) by the ones from (i-1) to j and i to (j+1).
        # By symmetry, all other costs stay the same.
        oldCost = self.dists[self.perm[(i-1) % self.n]][self.perm[i]] + self.dists[self.perm[j]][self.perm[(j+1) % self.n]]
        newCost = self.dists[self.perm[(i-1) % self.n]][self.perm[j]] + self.dists[self.perm[i]][self.perm[(j+1) % self.n]]
        
        if newCost < oldCost:
            startUnchanged = [self.perm[k] for k in range(i)] 
            rev = [self.perm[j - k] for k in range(j - i + 1)] 
            endUnchanged = [self.perm[j + k] for k in range(1 , self.n - j)]
            self.perm = startUnchanged + rev + endUnchanged
            return 1
        else:
            return 0
        
    def swapHeuristic(self):
        better = True
        while better:
            better = False
            for i in range(self.n):
                if self.trySwap(i):
                    better = True

    def TwoOptHeuristic(self):
        better = True
        while better:
            better = False
            for j in range(self.n-1):
                for i in range(j):
                    if self.tryReverse(i,j):
                        better = True
                

    # The Greedy heuristic builds a tour starting
    # from node 0, taking the closest (unused) node as 'next'
    # each time.
    def Greedy(self):
        unusedSet = set()
        unusedSet.update(self.perm)
        self.perm[0] = 0
        unusedSet.remove(0) # unusedSet now contains all nodes except 0. 
        
        for i in range(1 , self.n):
            
            # First, we get the distance from our current node
            for x in unusedSet:
                break  
            # x is some element in our set. It will serve as initial "closest city"
            minDistance = self.dists[self.perm[i-1]][x]
            closestCity = x
            
            # we then find the  city that is actually closest to our perm[i-1]
            for y in unusedSet:
                if self.dists[self.perm[i-1]][y] < minDistance:
                    minDistance = self.dists[self.perm[i-1]][y]
                    closestCity = y
           
            # (greedily) add the closest city to our list. we will not be able to use it again, so we remove it from the set
            unusedSet.remove(closestCity)
            self.perm[i] = closestCity 
            
    # This helper function computes the cost of an arbitrary tour, not just the current one. 
    def customTourValue(self, p):
       return sum( [self.dists[p[i]][p[(i+1) % self.n]] for i in range (self.n)] )       


    # See the attached report pdf for an explanation of the Lin-Kernighan algorithm I implemented.
    def lk(self, seed):
        
        alpha = 5
        improveCounter = 0
        maxImprovements = self.n**2.5
        
        # If a seed is provided, we wish to shuffle the starting permutation according to that seed. 
        if seed != -1:
            random.seed(seed)
            self.perm = [i for i in range(self.n)]
            random.shuffle(self.perm)
            random.seed()
        else: 
            random.seed()
            random.shuffle(self.perm)
            
        idleIterations = 0
        cyclicCounter = 0
        
        # We keep trying to improve our tour until we were unsuccessful n times in a row
        while (idleIterations < self.n):
            
            if improveCounter > maxImprovements: 
                print("Max. number of improvements reached!")
                return  
            
            # We rotate our path, so that we consider all nodes as "end nodes" once before giving up 
            cyclicCounter = (cyclicCounter + 1) % self.n 
            pathWithoutWraparound = [self.perm[(cyclicCounter + k) % self.n] for k in range(self.n)]
            
            res = self.improvePath(pathWithoutWraparound, 1, set(), alpha, improveCounter, maxImprovements)
            attemptedImprovement = res[0]
            improveCounter = res[1] # improvePath returns a list of 2 elements, so that we can keep track of
                                    # our improve counter, without having to add an instance variable to Graph
            
            if attemptedImprovement is None:
                idleIterations = idleIterations + 1
            else: # if an improvement is found, we accept it and reset the "idle counter"
                if (self.customTourValue(attemptedImprovement) < self.tourValue()):
                    idleIterations = 0
                    self.perm = attemptedImprovement
                else:
                    idleIterations = idleIterations + 1
        
                
    def improvePath(self, P, depth, restrictedSet, alpha, improveCounter, maxImprovements):
        # if our recursion depth is at least n, no unrestricted nodes are left
        if depth >= self.n: 
            return [P, improveCounter]
        
        improveCounter = improveCounter + 1
        if improveCounter > maxImprovements: 
            print("Max. number of improvements reached!")
            return [P, improveCounter]
        
        e = P[self.n-1] # our "end node" e
        currentCost = self.customTourValue(P)
        
        if depth < alpha:
            
            improvements = []
            
            for i in [i for i in range(self.n-1) if not (P[i] in restrictedSet)]:
                x = P[i]
                y = P[i+1]
                g = self.dists[x][y] - self.dists[e][x] # compute the "provisional gain" g
                if g > 0: 
                    startUnchanged = [P[k] for k in range(i+1)] 
                    restReversed   = [P[self.n-1-k] for k in range(self.n - 1 - i)]
                    newPath = startUnchanged + restReversed
                    newCost = self.customTourValue(newPath)
                    
                    if newCost < currentCost:
                        return [newPath, improveCounter] # accept the first 2-opt improvement we find
                    else:
                        nextRestrictedSet = copy.deepcopy(restrictedSet)
                        nextRestrictedSet.add(x)
                        res = self.improvePath(newPath, depth+1, nextRestrictedSet, alpha, improveCounter, maxImprovements)
                        newPath = res[0]
                        improveCounter = res[1]
                       
                        # if no 2-opt move is found, we take the best (recursively computed) k-opt move with k > 2
                        if self.customTourValue(newPath) < currentCost:
                            improvements.append(newPath)
            
            if improvements == []:
                return [P, improveCounter]
            
            costs = [self.customTourValue(t) for t in improvements]
            i = costs.index(min(costs))
            return [improvements[i], improveCounter]
            # return the improvement with the smallest cost 
            
        else: 
            # if our depth is at least alpha, we take the path with the largest "provisional gain" and return it,
            # after trying to improve it if we don't immediately find an improvement.
            
            gains = []
            
            for i in [i for i in range(self.n-1) if not (P[i] in restrictedSet)]:
                x = P[i]
                y = P[i+1]
                g = self.dists[x][y] - self.dists[e][x] 
                if g > 0:
                    gains.append([g , i])
            
            if gains == []:
                return [P, improveCounter]
            
            i = max(gains)[1]
            
            x = P[i] 
            startUnchanged = [P[k] for k in range(i+1)] 
            restReversed   = [P[self.n-1-k] for k in range(self.n - 1 - i)]
            newPath = startUnchanged + restReversed
            newCost = self.customTourValue(newPath)
                
            if newCost < self.customTourValue(P):
                return [newPath, improveCounter]
            else:
                nextRestrictedSet = copy.deepcopy(restrictedSet)
                nextRestrictedSet.add(x)
                return self.improvePath(newPath, depth+1, nextRestrictedSet, alpha, improveCounter, maxImprovements)
                        