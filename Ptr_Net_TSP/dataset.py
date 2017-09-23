import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
from scipy.spatial.distance import pdist, squareform 
from sklearn.decomposition import PCA
# from tsp_with_ortools import Solver



class DataGenerator(object):

    # Initialize a DataGenerator
    def __init__(self,solver):
        self.solver=solver  # reference solver for TSP (Google_OR_tools)


    # Solve an instance with reference solver
    def solve_instance(self, sequence):

        # Calculate dist_matrix
        dist_array = pdist(sequence)
        dist_matrix = squareform(dist_array)
        # Call OR Tools to solve instance
        route, opt_tour_length=self.solver.run(dist_matrix)
        # Corresponding tour
        ordered_seq = sequence[route]

        return ordered_seq[:-1], opt_tour_length


    # Solve an instance with Nearest Neighbor heuristic
    def solve_NN_policy(self, seq):

        dist_array = pdist(seq)
        dist_matrix = squareform(dist_array)

        best_length=1000
        best_tour=[]

        # Evaluate NN policy starting from start_point
        for start_point in range(len(seq)):

            ii=start_point
            seq_order=[start_point]
            sup = np.max(dist_matrix)

            NN_policy_length = 0

            while len(seq_order)!=len(seq):
                c_copy = np.copy(dist_matrix[ii]) # get adjacency list of current point
                for j in seq_order: # mask visited nodes
                    c_copy[j]=sup+1 

                nearest_neighbor_i = np.argmin(c_copy,axis=0) # find nearest neighbor
                ii=nearest_neighbor_i
                seq_order.append(ii)
                
                NN_policy_length+=c_copy[ii] # update tour length

            NN_policy_length+=dist_matrix[ii][start_point] # go back to start_point

            if NN_policy_length<best_length:
                best_length=NN_policy_length
                best_tour=seq[seq_order]

        return best_tour, best_length


    # Generate random TSP instance
    def gen_instance(self, max_length, dimension, test_mode=True, seed=0):
        if seed!=0: np.random.seed(seed)

        # Randomly generate (max_length) cities with (dimension) coordinates in [0,100]
        seq = np.random.randint(100, size=(max_length, dimension))

        # Principal Component Analysis to center & rotate coordinates
        pca = PCA(n_components=dimension)
        sequence = pca.fit_transform(seq)

        # Scale to [0,1[
        input_ = sequence/100

        if test_mode == True:
            return input_, seq
        else:
            return input_

    # Generate random batch for training procedure
    def train_batch(self, batch_size, max_length, dimension):
        input_batch = []

        for _ in range(batch_size):
            # Generate random TSP instance
            input_ = self.gen_instance(max_length, dimension, test_mode=False)

            # Store batch
            input_batch.append(input_)

        return input_batch


    # Generate random batch for testing procedure
    def test_batch(self, batch_size, max_length, dimension, seed=0):
        input_batch = []

        # Generate random TSP instance
        input_, or_sequence = self.gen_instance(max_length, dimension, test_mode=True, seed=seed)

        for _ in range(batch_size):
            # Shuffle sequence
            sequence = np.copy(input_)
            np.random.shuffle(sequence)

            # Store batch
            input_batch.append(sequence)

        return input_batch, or_sequence


    # Randomly shuffle an input_batch
    def shuffle_batch(self, coord_batch):
        seq = coord_batch[0]
        input_batch = []

        for _ in range(len(coord_batch)):
            # Shuffle sequence
            sequence = np.copy(seq)
            np.random.shuffle(sequence)

            # Store batch
            input_batch.append(sequence)

        return input_batch


    # Plot a tour
    def visualize_2D_trip(self, trip):
        plt.figure(figsize=(30,30))
        rcParams.update({'font.size': 22})

        # Plot cities
        plt.scatter(trip[:,0], trip[:,1], s=200)

        # Plot tour
        tour=np.array(list(range(len(trip))) + [0])
        X = trip[tour, 0]
        Y = trip[tour, 1]
        plt.plot(X, Y,"--", markersize=100)

        # Annotate cities with order
        labels = range(len(trip))
        for i, (x, y) in zip(labels,(zip(X,Y))):
            plt.annotate(i,xy=(x, y))  

        plt.xlim(-0.75,0.75)
        plt.ylim(-0.75,0.75)
        plt.show()



if __name__ == "__main__":

    # Config
    batch_size=3
    max_length=5
    dimension=2

    # Create Solver and Data Generator
    solver = [] # Solver(max_length)
    dataset = DataGenerator(solver)

    # Generate some data
    # input_batch = dataset.train_batch(batch_size, max_length, dimension, seed=0)
    input_batch, or_seq = dataset.test_batch(batch_size, max_length, dimension, seed=0)

    # Some print
    #print('Coordinates: \n',input_batch[0])

    # Solve to optimality
    opt_seq, opt_length = dataset.solve_instance(or_seq)
    print('Optimal tour length: \n',opt_length)

    NN_seq, NN_length = dataset.solve_NN_policy(or_seq)
    print('NN tour length: \n',NN_length)
    dataset.visualize_2D_trip(NN_seq)
    dataset.visualize_2D_trip(opt_seq)

    # 2D plot for coord batch
    #dataset.visualize_2D_trip(input_batch[0])
    #dataset.visualize_2D_trip(input_batch[0])