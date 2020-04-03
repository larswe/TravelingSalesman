import math
from graph import *
import random
import time

class Tester:
    
    # To run my tests, you could initialize a graph and call test functions on it in the following main method.
    # Then simply run Tester.main() in the terminal.
    # The current main method contains examples of how graphs may be tested. 
    @staticmethod
    def main():
        #g = Tester.generalTSP(100, 50, 0.1 , 0.25, 100)
        g = Tester.metricTSP(100, 50, 0.1, 0.25, 100)
        #g = Graph(-1, "50cities")
        
        
        #print(Tester.testSwap(g))
        print(Tester.testTwoOpt(g))
        #print(Tester.testSwapAndTwoOpt(g))
        print(Tester.testGreedy(g))
        #print(Tester.testSingleLK(g , -1))
        print(Tester.testLK(g))
    
    # The following 4 functions return the cost of the solution found by the corresponding heuristic,
    # as well as the time in milliseconds it took to find that solution. 
    
    @staticmethod
    def testSwap(g):
        g.perm = [i for i in range(g.n)]
        start = time.time()
        g.swapHeuristic()
        end = time.time()
        ms = 1000 * (end - start)
        result = g.tourValue()
        return [result, ms]

    @staticmethod
    def testTwoOpt(g):
        g.perm = [i for i in range(g.n)]
        start = time.time()
        g.TwoOptHeuristic()
        end = time.time()
        ms = 1000 * (end - start)
        result = g.tourValue()
        return [result, ms]

    @staticmethod
    def testSwapAndTwoOpt(g):
        g.perm = [i for i in range(g.n)]
        start = time.time()
        g.swapHeuristic()
        g.TwoOptHeuristic()
        end = time.time()
        ms = 1000 * (end - start)
        result = g.tourValue()
        return [result, ms]
    
    @staticmethod
    def testGreedy(g):
        g.perm = [i for i in range(g.n)]
        start = time.time()
        g.Greedy()
        end = time.time()
        ms = 1000 * (end - start)
        result = g.tourValue()
        return [result, ms]
    
    # This function returns the cheapest, median and most expensive solutions (or rather, their costs)
    # found by LK on 100 iterations. 
    # Intermediate results are printed along the way.
    @staticmethod
    def testLK(g):
        tuples = []
        for i in range(100):
            g.improveCounter = 0
            seed = math.ceil( g.n**3 * random.random() )
            start = time.time()
            g.lk(seed)
            end = time.time()
            ms = 1000 * (end - start)
            result = g.tourValue()
            
            print("Seed: " + str(seed))
            print([result , ms])
            print()
            
            tuples.append([result, ms]) 
                
        tuples.sort()
        return [tuples[0] , tuples[50], tuples[99]]
    
    @staticmethod
    # Choose parameter seed = -1 for random seed
    def testSingleLK(g, seed):
        g.perm = [i for i in range(g.n)]
        start = time.time()
        g.lk(seed)
        end = time.time()
        ms = 1000 * (end - start)
        result = g.tourValue()
        return [result, ms]
    
    @staticmethod
    # Given a Euclidean graph and the "coordinates" file it is based on, this function plots the graph and
    # the tour corresponding to its current permutation of nodes. 
    # Since a graph does not "know" of its coordinates, we need to provide the matching filename for this to work. 
    def plotTour(g, filename):
        F = open(filename)
        lines = F.readlines()
        
        if len(lines) != self.n:
            return 
            
        # Converting the lines into lists of length 2
        coords = [[int(s) for s in lines[i].split()] for i in range(self.n)]
        
        xs = [point[0] for point in coords]
        ys = [point[1] for point in coords]
        
        plt.scatter(xs, ys)
        
        for i in range(g.n - 1):
            plt.plot([xs[g.perm[i]] , xs[g.perm[i+1]]], [ys[g.perm[i]] , ys[g.perm[i+1]]], 'k-', lw=2)
        
        plt.plot([xs[g.perm[g.n-1]] , xs[g.perm[0]]], [ys[g.perm[g.n-1]] , ys[g.perm[0]]], 'k-', lw=2)
        
        plt.show()
    
    # We would now like to generate TSPs for which we know the optimal solution, in order to test our algorithms 
    
    # The following function, generalTSP, returns a Traveling Salesman Problem, which does not have to be metric. 
    # n = number of nodes
    # R = maximum distance 
    # 0 <= rho <= sigma <= 0.5 restrict alpha_i for node i, 
    # s.t. 2R*rho <= d(i,j) <= 2R*sigma <= R for nodes i,j on the optimal tour and
    # s.t. 2R*rho <= d(i,j) <= R for nodes not on the optimal tour. 
    # A small rho will lead to some edges being short, even if they are not optimal.
    # Large rho and sigma will lead to all distances being close to R. 
    # Thus, if we would like a large diversity of edge costs, rho and sigma should
    # neither be very small nor very big, e.g. rho = 0.1, sigma = 0.25
    
    # Proof that this works:
    # Suppose [a_1, ..., a_n] is our proclaimed "cheapest tour". It has cost 2*(alpha_1+...+alpha_n). 
    # Since distance d_i,j >= alpha_i + alpha_j, the distance of a tour will be a sum of d_i,j where every number
    # from 1 to n will appear once as a i and once as a j. It follows that any tour through the graph has distance
    # at least 2*(alpha_1 + ... + alpha_n). q.e.d.
    @staticmethod
    def generalTSP(n, R, rho, sigma, seed):
         
        if rho > sigma or rho < 0:
            print("Rho must be in [0,Sigma]")
            return
        if sigma > 0.5:
            print("Sigma must be <= 0.5")
            return
            
        bestTour = [i for i in range(n)]
        
        if seed != -1:
            random.seed(seed)
        
        random.shuffle(bestTour)    
        alphas = [random.uniform(R*rho , R*sigma) for i in range(n)]
            
        distanceMatrix = []
        for i in range(n):
            new = []
            for j in range(n):
                new.append(0)
            distanceMatrix.append(new)
        
        for i in range(n):
            for j in range(i):
                randomDistance = random.uniform(alphas[i] + alphas[j] , R)
                distanceMatrix[i][j] = math.ceil(randomDistance)
                
        bestCost = 0
        for i in range(n):
            minDistance = math.floor(alphas[bestTour[i]]+alphas[bestTour[(i+1)%n]])
            bestCost = bestCost + minDistance
            # We want to construct a lower triangular matrix here. But instead of using an if-else we might as well
            # fill both entries, exactly one of which lies in the lower triangle we will then "map" to the text file. 
            distanceMatrix[bestTour[i]][bestTour[(i+1)%n]] = minDistance
            distanceMatrix[bestTour[(i+1)%n]][bestTour[i]] = minDistance
            
                
        problemCode = math.ceil(random.random() * 1000)
        random.seed() # this is where we discard the given seed
        filename = "general_graph_" + str(problemCode)
        goalFile = open(filename , "w")
        for i in range(n):
            for j in range(i):
                goalFile.write(str(i) + " " + str(j) + " " + str(distanceMatrix[i][j]) + "\n")

        goalFile.write("Best Cost: " + str(bestCost) + "\n")
        goalFile.write("Best Tour" + str(bestTour))
            
        goalFile.close()    
            
        g = Graph(n , filename)
            
        return(g)
      
    # To ensure that the triangle inequality is satisfied by our distance matrix, we add one small detail to our general algorithm:
    # We compute a gamma, which equals twice the smallest alpha_k. 
    # We then ensure that all distances d_i,j lie in the interval [alpha_i + alpha_j , min(alpha_i + alpha_j + gamma , R)]. 
    # This will lead to a metric scenario, since it implies that
    # d_i,j <= alpha_i + alpha_j + gamma <= alpha_i + alpha_j + 2*alpha_k for all k. 
    # -> alpha_i + alpha_k <= d_i,k and alpha_j + alpha_k <= d_j,k by construction.
    # -> Therefore, d_i,j <= d_i,k + d_j,k for all k, as required. 
    @staticmethod
    def metricTSP(n, R, rho, sigma, seed):
        
        if rho > sigma or rho < 0:
            print("Rho must be in [0,Sigma]")
            return
        if sigma > 0.5:
            print("Sigma must be <= 0.5")
            return
            
        bestTour = [i for i in range(n)]
        
        if seed != -1:
            random.seed(seed)
        
        random.shuffle(bestTour)    
        alphas = [random.uniform(R*rho , R*sigma) for i in range(n)]
        gamma = 2*min(alphas)
            
        distanceMatrix = []
        for i in range(n):
            new = []
            for j in range(n):
                new.append(0)
            distanceMatrix.append(new)
        
      
        for i in range(n):
            for j in range(i):
                upperBound = min([R , alphas[i] + alphas[j] + gamma])
                randomDistance = random.uniform(alphas[i] + alphas[j] , upperBound)
                distanceMatrix[i][j] = math.ceil(randomDistance)
                
                
        bestCost = 0
        for i in range(n):
            minDistance = math.floor(alphas[bestTour[i]]+alphas[bestTour[(i+1)%n]])
            bestCost = bestCost + minDistance
            
            distanceMatrix[bestTour[i]][bestTour[(i+1)%n]] = minDistance
            distanceMatrix[bestTour[(i+1)%n]][bestTour[i]] = minDistance
            
                
        problemCode = math.ceil(random.random() * 1000)
        random.seed() # this is where we discard the given seed
        filename = "metric_graph_" + str(problemCode)
        goalFile = open(filename , "w")
        for i in range(n):
            for j in range(i):
                goalFile.write(str(i) + " " + str(j) + " " + str(distanceMatrix[i][j]) + "\n")

        goalFile.write("Best Cost: " + str(bestCost) + "\n")
        goalFile.write("Best Tour" + str(bestTour))
            
        goalFile.close()    
            
        g = Graph(n , filename)
            
        return(g)
           