import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
import tensorflow as tf
print(tf.config.list_physical_devices()) 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import pandas as pd
import random
import mlflow

from itertools import product
from hilbert import encode
from sklearn.neighbors import NearestNeighbors
from data.data_loader import load_saddle_data
from curve_saddle.constants_saddle import N_BITS, N_DIMS, MLFLOW_HOST, MLFLOW_PORT, QUERY_SIZES, N_POINTS_PER_AXIS


random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)

N_BLOCKS          = 40
k                 = 6
ALPHA             = 0.7
EPOCHS            = 500 #400 #370 <-latest
LR                = 1e-3

N_CORE_QUERIES  = {"S":150, "M":200,"L":20}
N_VAL_QUERIES   = {"S":350, "M":600, "L":60}
N_TRAIN_QUERIES = {"S":2, "M":3, "L":2}

TEMP       = 1.0
TEMP_MIN   = 0.00001
SOFTM_TEMP_DECAY = 0.98
SOFTR_TEMP_DECAY = 0.98

CLUSTER_LOSS_WEIGHT = 6

example_distributions = []
distr_vis_indices     = [0, 63, 122, 140]

GLOBAL_START_TEMP = 1.0



##########################################################
                    #Evaluation
##########################################################
def perform_evaluation(x_tf:tf.Tensor, datapoints:np.ndarray, data_pi_and_rad:np.ndarray) -> None:
    '''Evaluates the performance of the trained model versus the performance of the Hilbert curve.

    Args:
        - x_tf (tf.Tensor): Data points as tf.Tensor
        - datapoints (np.ndarray): Data points plain as ndarray
        - data_pi_and_rad (np.ndarray): π and radius values for data points. 
    Returns:
        - None
    '''

    val_queries = tf.constant(create_queries(N_VAL_QUERIES, QUERY_SIZES, min(data_pi_and_rad[:,1]), max(data_pi_and_rad[:,1])))
    val_qp_matrix = tf.constant(create_qp_matrix(val_queries, data_pi_and_rad))
    
    eval_model_vs_hilbert(model, x_tf, datapoints, val_qp_matrix)
    plot_query_size_distribution(val_qp_matrix)
    #plot_softrank_evolution(soft_rank_values) #TODO: Plot only art of ranks for more than n ranks -> otherwise graphic will not be expressive anymore
    plot_point_block_distribution_evolution(example_distributions)


def log_training_metrics_to_mlflow(metrics:dict)->None:
    ''' Logs training matrics to mlflow

    Args:
        -metrics (dict): Dict of metrics to log
    Returns:
        -None
    '''

    print("Log training metrics...")
    for ep in range(EPOCHS):
        mlflow.log_metrics({metric:metrics[metric][ep] for metric in metrics}, ep)


def plot_query_size_distribution(qp_matrix:np.ndarray) -> None:
    '''Plot distribution of query sizes for a given qp_matrix.

    Args:
        - qp_matrix(nd.array): QP_Matrix with input queries for distribution
    Returns:
        - None
    '''
    points_per_query = sorted(np.sum(qp_matrix, axis=1))
    fig, ax = plt.subplots()
    plt.hist(points_per_query, bins=10)
    ax.set_title(f"Querysize Histogram")
    ax.set_xlabel("Number of Points")
    ax.set_ylabel("Number of Queries")
    mlflow.log_figure(fig, "querysize_histogram.png")
    plt.close(fig)
    
    fig, ax = plt.subplots()
    plt.bar(range(len(qp_matrix)), points_per_query)
    ax.set_title("Points per Query")
    ax.set_xlabel("Query")
    ax.set_ylabel("Number of Points")
    mlflow.log_figure(fig, "points_per_query.png")
    plt.close(fig)
    

def log_params_to_mlflow() -> None:
    ''' Logs model parameters to MLFlow
    
    Args:
        -None
    Returns:
        -None
    '''
    mlflow.log_param("Learning Rate", LR)
    mlflow.log_param("Epochs", EPOCHS)
    mlflow.log_param("Number of Core Queries", sum([N_CORE_QUERIES[key] for key in N_CORE_QUERIES]))
   

def visualize_block_point_assignment(distribution:np.ndarray, ax:plt.Axes.axes):
    '''Plots the evolution of the probability distribution (point to block) for a single point

    Args:
        - distribution(np.ndarray): Distribution of values
        - ax (plt.Axes.axes) (np.ndarray): Axes to plot on
    Returns:
        - None
    '''
    distr = pd.DataFrame(distribution)
    distr_long = distr.T.melt(var_name="Epoche", value_name="Probability")
    sns.lineplot(data=distr_long, x=distr_long.groupby("Epoche").cumcount(), y="Probability", hue="Epoche", ax=ax)
    ax.set_xlabel("Block")
    ax.set_ylabel("Probability")
    

def plot_curve_of_trained_model(ranks:tf.Tensor, x_tf:tf.Tensor, model_name:str) -> None:
    ''' Plots the curve through the points of a dataset by using the ranks of the trained model

    Args:
        - ranks (tf.Tensor): Ranks for each data point 
        - x_tf (tf.Tensor): Data points as tf.Tensor
        - model_name (str): Name of the model
    Returns:
        - None
    '''
    x_sorted_np = tf.gather(x_tf, ranks).numpy()
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(x_sorted_np[:,0], x_sorted_np[:,1], 'o-', linewidth=0.8, alpha=0.8, color='gray')
    ax.set_title(f"{model_name} Sort Order")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    mlflow.log_figure(fig, f"{model_name}_sort_order.png")
    
    plt.close(fig)
    
    
def plot_point_block_assignment(bocks_for_points:np.ndarray, model_name:str) -> None:
    '''Plots the assignments of single points to blocks

    Args:
        - bocks_for_points(np.ndarray): Assignment of blocks for each point
        - model_name (str): Name of the model

    Returns: 
        - None
    '''
    fig, ax = plt.subplots()
    ax.plot(bocks_for_points)
    ax.set_xlabel("Point")
    ax.set_ylabel("Block")
    ax.set_title("Point to Block Assignment")
    
    mlflow.log_figure(fig, f"{model_name}_point_block_assignment.png")
    plt.close(fig)


def plot_block_weight_hist(blocks_in_queries:np.ndarray, model_name:str) -> None:
    '''Plots the distribution of points within blocks. 
    This is especially useful to find out how the model learns to distribute points.
    
    Args:
        - blocks_in_queries (np.ndarray): A matrix mapping blocks to queries
    Returns:
        - None
    '''

    blocks, counts = np.unique(blocks_in_queries, return_counts=True)
    
    fig, ax = plt.subplots()
    ax.bar(blocks[1:], counts[1:])
    ax.set_xlabel("Block")
    ax.set_ylabel("Count")
    ax.set_title("Block Weight Distribution")
    mlflow.log_figure(fig, f"{model_name}_block_weight_hist.png")
    plt.close(fig)
    

def calc_query_statistics(blocks_in_queries: np.ndarray, points_in_blocks_hist:dict[int:int]) -> tuple[list,float,float]:
    ''' Caculates how many blocks are used in total, how many blocks are touched by a query on average and
    how man clusters can be found within a query. A cluster is a set of consecutive numbered blocks. 
    Also, the number of tuples being read is calculates by number of tuples within a block. 
    
    Args:
        - blocks_in_queries (np.ndarray): A matrix mapping blocks to queries
        - points_in_blocks_hist (dict[int:int]): Number of points in each block
    Returns:
        - results (tuple): used_blocks (set), mean_blocks_per_query (float), mean_clusters_per_query (float)
    '''

    used_blocks, blocks_per_query, clusters_for_query, traversed_tuples = [],[],[],[]

    for query in blocks_in_queries:
        query = query[query != -1]          #Take only the blocks that are touched by a query. -1 say its not part of the query
        unique_blocks = np.unique(query)    #Erease double values
        used_blocks.extend(unique_blocks)   # Add blocks of this query to used_blocks --> Overview which blocks are used in general
        blocks_per_query.append(len(unique_blocks)) #No blocks touched by query
        traversed_tuples.append(sum([points_in_blocks_hist[block] for block in unique_blocks]))

        #Cluster means consecutive block values. Sequential read is always better than jumping between blocks
        if len(unique_blocks) == 0:
            clusters = 0
        else:
            sorted_blocks = np.sort(unique_blocks)
            diffs = np.diff(sorted_blocks) #Diffs between consecutive block numbers
            clusters = np.count_nonzero(diffs > 1) + 1 #If there is a gap>1 --> new cluster --> no sequential read
        clusters_for_query.append(clusters)
        results = set(used_blocks), np.mean(blocks_per_query), np.mean(clusters_for_query), np.average(traversed_tuples)

    return results 


def assig_points_to_blocks(ranks:np.ndarray) -> np.ndarray:
    '''For every rank of every point in the grid/dataset, compute the corresponding block 
    by dividing the ranks by block size.

    Args:
        - ranks (np.ndarray): List of ranks for points in the dataset

    Returns:
        - blocks_for_points (np.ndarray): List of blocks, to which points are assigned
    '''
    N = len(ranks)
    blocks_for_points = (ranks * N_BLOCKS) // N   
    blocks_for_points = np.minimum(blocks_for_points, N_BLOCKS - 1) #Avoid rounding errors that lead to values > N_BLOCKS-1
    return blocks_for_points.astype(np.int32)


def assign_blocks_to_queries(qp_matrix:np.ndarray, blocks_for_points:np.ndarray) -> np.ndarray:
    '''Each query embraces 1 or more blocks. This relationship is reflected by blocks_in_queries, resulting from this function.
    For this, a mask is created, that repeats blocks_for_points N_QUERIES-times.
    The mask then has the same dimensionality as the qp_matrix. qp_matrix: N_QUERIES x N_POINTS
    The mask is applied to the qp_matrix (which is 1 if a point is in a query, 0 else)
    If there is a in the qp_matrix -> insert the corresponding block of the this point, else set it to -1
    
    Args:
        - qp_matrix         (np.ndarray): Matrix, mapping points to queries
        - blocks_for_points (np.ndarray): Matrix mapping blocks to points

    Returns:
        - blocks_in_queries (np.ndarray): Matrix mapping blocks to queries
    
    '''

    mask = np.tile(blocks_for_points[None,:], (qp_matrix.shape[0], 1)) 
    blocks_in_queries = np.where(qp_matrix.numpy() > 0, mask, -1)
    
    return blocks_in_queries


def eval_model_vs_hilbert(model:tf.keras.Model, x_tf:tf.Tensor, datapoints:np.ndarray, qp_matrix:np.ndarray) -> None:
    '''Evaluates the performance of the neural net against the performance of the Hilbert curve.

    Args:
        - model (tf.keras.Model): Trained Neural Net
        - grid  (np.ndarray)    : Gridpoints/datapoints
        - qp_matrix (np.ndarray): Matrix assigning points to queries

    Returns:
        - None
    '''

    #model_ranks = soft_rank(tf.squeeze(model(x_tf), axis=1),1,1).numpy() #Calculate the 1D value for all points given
    model_embeddings = tf.squeeze(model(x_tf), axis=1)
    model_ranks = tf.argsort(tf.argsort(model_embeddings)).numpy() #For all points the rank is the position of the point in the ordered list of embeddings
    h_idx = encode(datapoints, num_dims=N_DIMS, num_bits=N_BITS)
    h_ranks = np.argsort(np.argsort(h_idx))
    
    np.savetxt("data_points.csv", x_tf.numpy(), delimiter=",")
    #
    # np.savetxt("model_ranks.csv", model_embeddings.numpy(), delimiter=",")
    np.savetxt("hilbert_ranks.csv", h_ranks, delimiter=",")
    
    mlflow.log_artifact("data_points.csv", artifact_path="data")
    #mlflow.log_artifact("model_ranks.csv", artifact_path="data")
    mlflow.log_artifact("hilbert_ranks.csv", artifact_path="data")
          
    analyze_ranking_performance(model_ranks, qp_matrix, "NEURAL_NET")
    analyze_ranking_performance(h_ranks, qp_matrix, "HILBERT")
    #plot_curve_of_trained_model(model_ranks, x_tf, "NEURAL_NET") 
   # plot_curve_of_trained_model(h_ranks, x_tf, "HILBERT") 


def get_block_to_point_histogram(blocks_for_points:np.ndarray) -> dict:
    '''Computes the histogram of points in blocks.

    Args:
        - blocks_for_points (np.ndarray): List with one entry for every point, with the number of the corresponding block

    Returns:
        - dict: Values for the blocks and counts = number of tuples in block.
    '''
    vals, counts = np.unique(blocks_for_points, return_counts=True)
    
    return dict(zip(vals,counts))


def analyze_ranking_performance(ranks:np.ndarray, qp_matrix:np.ndarray, model_name:str, is_validation=False) -> None:
    '''Computes the number of used blocks, average blocks per query and average clusters per query
    for a given list of ranks.
    
    Args:
        - ranks (np.ndarray)     : Ranks from a model
        - qp_matrix  (np.ndarray): Matrix assigning queries to points
        - model_name (str)       : Name of the model, that is evaluated

    Returns:
        if not validation after each epoche:
            -None 
        else:
            -str: String of intermediate results for logging print after each epoche.
    '''
    blocks_for_points = assig_points_to_blocks(ranks)
    points_in_blocks_hist = get_block_to_point_histogram(blocks_for_points)
    blocks_in_queries = assign_blocks_to_queries(qp_matrix, blocks_for_points)
    n_used_blocks, avg_blocks_per_query, avg_clusters_per_query, traversed_tuples = calc_query_statistics(blocks_in_queries, points_in_blocks_hist)
    if is_validation == False:
        plot_block_weight_hist(blocks_in_queries, model_name)
        plot_point_block_assignment(blocks_for_points, model_name)
        
        mlflow.log_param(f"{model_name} - Used blocks", set(n_used_blocks))
        mlflow.log_metric(f"{model_name} - Avg blocks per query", avg_blocks_per_query)
        mlflow.log_metric(f"{model_name} - Avg clusters per query", avg_clusters_per_query)
        mlflow.log_metric(f"{model_name} - Traversed Tuples", traversed_tuples)
        print(f"{model_name}: Used blocks: {str(set(n_used_blocks))}")
        print(f"Avg blocks per query: {avg_blocks_per_query:.2f} |  Avg clusters per query: {avg_clusters_per_query:.2f}")
        print(f"Points traversed: {traversed_tuples}")
    else:
        return f"Avg blocks per query: {avg_blocks_per_query:.2f} |  Avg clusters per query: {avg_clusters_per_query:.2f} Points traversed: {traversed_tuples}"
    

def save_point_to_cluster_probs(indices:list, storage:list):
    ''' Decorator function that saves intermediate results from the block to point assignment probability function.

    Args:
        - indices(list): List of indices for points, from which we want to save the intermediate distribution results
        - storage(list): List that holds intermediate results. Validation functions will draw them from this list. 
    Returns:
        - Basically it returns the wrapper and decorator - just check the code. Type hints are no expressive here. 
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            block_weight_matrix, center_dists_proba = func(*args, **kwargs)
            for i, index in enumerate(indices):
                if len(storage) < i+1:
                    storage.append([])
                storage[i].append(center_dists_proba[index,:].numpy())
            return block_weight_matrix
        
        return wrapper
    return decorator
    
    
def plot_softrank_evolution(soft_rank_values:np.ndarray) -> None:
    '''Plots the evolution of softranks for currently all ranks. 
    

    Args:
        -soft_rank_values(np.ndarray): Array of softrank values for each epoche
    
    Returns
        - None
    '''

    rank_distr = pd.DataFrame(soft_rank_values)   # shape: epochs x points
    rank_distr["Epoch"] = rank_distr.index

    ranks_long = rank_distr.melt(
        id_vars="Epoch", 
        var_name="Point", 
        value_name="Rank"
    )
    
    fig, ax = plt.subplots()
    sns.lineplot(
        data=ranks_long,
        x="Point", y="Rank", hue="Epoch",
        estimator=None, lw=0.8, alpha=0.7
    )

    ax.set_xlabel("Point")
    ax.set_title("Softranks")
    fig.tight_layout()
    
    mlflow.log_figure(fig, "softrank_evolution.png")

    plt.close(fig)



def plot_point_block_distribution_evolution(distributions:np.ndarray) -> None:
    '''Plots the evolution of point to block distributions for a set of epoche, contained in the input array 
    
    Args:
        - distributions(np.ndarray): List of distributions - one for each epoche - for a single point.
    Returns:
        - None
    '''
    side_length = int(len(distributions) / 2) if len(distributions) % 2 == 0 else int(len(distributions)//2+1)
    fig, axs = plt.subplots(side_length, 2)
    img_index = 0
    for i in range(0,2):
        for j in range(0, side_length):
            visualize_block_point_assignment(distributions[img_index], ax=axs[j,i])
            img_index += 1
            
    fig.tight_layout()
    mlflow.log_figure(fig, "point_to_block_distr_evolution.png")
    plt.close(fig)



##########################################################
                         #Setup
##########################################################

def build_knn_graph(grid:np.ndarray, k:int=4) -> tuple[list, list]:
    '''Computes the knn graph for every point in the grid.
      Args:
        - grid (np.ndarray): List of gridpoints
        - k    (int)       : Number of neigbors for knn computation
    Returns:
        - results (tuple): List of indices and list of distances
    '''
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(grid)
    dists, idxs = nn.kneighbors(grid)
    return idxs[:,1:], dists[:,1:] 


def create_queries(n_queries_dict:dict, queries_sizes:dict, min_radius:int, max_radius:int) -> list[list[float, float, float, float]]:
    '''Creates queries based on values for pi and radius --> cutting out segments of the saddle with one whole
    
    Args:
        - n_queries (dict)    : Number of queries for query type (S,M,L)
        - queries_sizes (dict): PI-value span for each query type
    
    Returns: 
        -queries (list[list[min_pi,max_pi,min_radius, max_radius]]): List of queries.
    '''
    queries = []
    for q_type in n_queries_dict:
        n_queries = n_queries_dict[q_type]
        for _ in range(n_queries):

            lower_max_pi, upper_max_pi = queries_sizes[q_type]
            min_pi, max_pi = 0, lower_max_pi+(random.random()*upper_max_pi)
            rotation_factor = random.random()*2 # rotates the selected range of pi to a different position on the circle

            #cover values > 2pi
            shifted_min_pi = min_pi + rotation_factor if min_pi + rotation_factor < 2 * np.pi else min_pi + rotation_factor - 2 * np.pi
            shifted_max_pi = max_pi + rotation_factor if max_pi + rotation_factor < 2 *np.pi else max_pi + rotation_factor - 2 * np.pi

            #select radius/ring_width range
            selected_min_radius = min_radius + random.random() * (max_radius - min_radius) + 0.001
            selected_max_radius = selected_min_radius + random.random() * (max_radius - selected_min_radius) + 0.001

            query = np.array([shifted_min_pi, shifted_max_pi, selected_min_radius, selected_max_radius])
            
            if random.random() <= 0.05:     # cover full circle with 5% chance
                query[0] = 0
                query[1] = 2*np.pi
            if random.random() <= 0.05:     # cover full width with 5% chance
                query[2] = min_radius
                query[3] = max_radius
            
            queries.append(query)
    
    return queries


def calc_grid_points_for_intervals(ranges:list[list[int, int]]) -> np.ndarray[np.ndarray[int, int]]:
    ''' Computes the grid points for a set of ranges (these are the queries in this case).
    
    Args:
        - ranges (list[list[int, int]]): Ranges for which encapsulated points need to be computed
    Returns:
        -points  (list[list[int, int]]): List of points, ecapsulated by ranges
    '''
    
    ranges = [range(start, end+1) for start,end in ranges]
    points = np.array(list(product(*ranges)))
    return points



def define_model() -> tuple[tf.keras.Model, tf.keras.optimizers.Adam]:
    '''Defines the neural network. 10 selu layers with one skip-connection.

    Args:
        -
    Returns:
        -model (tf.keras.Model): Model definition
        -opt   (tf.keras.Optimizer): Optimizer to train with
    '''
    activation = "selu"
    inputs = tf.keras.Input(shape=(3,))
    x = tf.keras.layers.Dense(8, activation=activation)(inputs)
    x_1 = tf.keras.layers.Dense(16, activation=activation)(x)
    x = tf.keras.layers.Dense(128, activation=activation)(x_1)
    x = tf.keras.layers.Dense(256, activation=activation)(x)
    x = tf.keras.layers.Dense(512, activation=activation)(x)
    x = tf.keras.layers.Dense(512, activation=activation)(x)
    x = tf.keras.layers.Dense(256, activation=activation)(x)
    x = tf.keras.layers.Dense(128, activation=activation)(x)
    x = tf.keras.layers.Dense(16, activation=activation)(x)
    x_2= tf.keras.layers.Dense(8, activation=activation)(x + x_1)
    
    outputs = tf.keras.layers.Dense(1)(x_2)
    model = tf.keras.Model(inputs, outputs)
    model.summary()

    opt = tf.keras.optimizers.AdamW(
        learning_rate=LR,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    return model, opt

def create_qp_matrix(queries:tf.Tensor, selection_vals:np.ndarray) -> tf.Tensor:
    '''Computes the qp_matrix by numpy broadcasting - much faster than using loops.
    1: Separate dimensions of the queries, in x1,x2 and y1,y2
    2: Same approach for the grid
    3: Create a mask for x, that is 1 if an x1 in a query is smaller than an x in the grid and the x2 is greater.
       Do the same for y. If both are True --> point is in query --> add a one to qp_matrix for the respective query and the respective point
    
    Args:
        queries (np.ndarray) : List of queries
        grid    (np.ndarray) : List of gridpoints
    
    Returns:
        qp_matrix (np.ndarray): Matrix mapping points to queries   
    '''
    
    q_pi_1, q_pi_2 = np.expand_dims(queries[:,0], axis = 1),  np.expand_dims(queries[:,1], axis = 1)
    q_rad_1, q_rad_2 = np.expand_dims(queries[:,2], axis = 1), np.expand_dims(queries[:,3], axis = 1)
    pi,  rad = np.expand_dims(selection_vals[:,0], axis = 0), np.expand_dims(selection_vals[:,1], axis = 0)
    
    mask_pi_normal = (q_pi_1 <= pi) & (pi <= q_pi_2)
    mask_pi_wrapped = (q_pi_1 >= pi) | (pi <= q_pi_2)

    mask_pi = np.where(q_pi_1 > q_pi_2 , mask_pi_wrapped, mask_pi_normal)
    mask_rad = (q_rad_1 <= rad) & (rad <= q_rad_2)

    qp_matrix = (mask_pi & mask_rad).astype(np.float32)

    return tf.constant(qp_matrix, dtype=tf.float32)


def setup_model_training() -> list[np.ndarray, np.ndarray, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    '''Sets up the necessary elements, needed for training the model.
        - Loads data
        - Creates queries & qp_matrix
        - Defines blocks
        - Computes knn graph, in case knn loss is used -> Might be dispensable for future trainings

    Args:
        - None
    Returns:
        - data_pi_and_rad (np.array): Holds a value for π and radius for every data point
        - datapoints (np.array): Coordinates of data points
        - x_tf (tf.Tensor): Data points as tf.Tensor
        - qp_matrix (tf.Tensor): QP Matrix - one row for every query, one column for every point. 1 if point in query, 0 else
        - blocks (tf.Tensor): Blocks to assign points to
        - knn_dists (tf.Tensor): Knn distances for every point
        - knn_idx_tf (tf.Tensor): Indices for knns for every point
    '''
    datapoints, data_pi_and_rad = load_saddle_data(N_POINTS_PER_AXIS-1)
    x_tf = tf.constant(datapoints, dtype=tf.float32)
    core_queries = tf.constant(create_queries(N_CORE_QUERIES, QUERY_SIZES, min(data_pi_and_rad[:,1]), max(data_pi_and_rad[:,1])), dtype=tf.float32)
    qp_matrix = create_qp_matrix(core_queries, data_pi_and_rad)
    blocks = (tf.range(N_BLOCKS, dtype = tf.float32) + 0.5) / N_BLOCKS
    knn_idx, knn_dists = build_knn_graph(datapoints, k=4)
    knn_idx_tf = tf.constant(knn_idx, dtype=tf.int32)
    knn_dists = knn_dists.astype(np.float32)
    
    return data_pi_and_rad, datapoints, x_tf, qp_matrix, blocks, knn_dists, knn_idx_tf
    

##########################################################
                         #Training
##########################################################

def extend_qp_matrix_by_random_queries(qp_matrix: tf.Tensor, data_pi_and_rad: np.ndarray) -> tf.Tensor:
    '''Every epoch, a set of randomly generated training queries are added.
    Those queries need to be combined with the core training queries.
    For this reason, the qp_matrix needs to be updated every epoche. 

    Args:
        - qp_matrix (tf.Tensor): QP Matrix - one row for every query, one column for every point. 1 if point in query, 0 else 
        - data_pi_and_rad (tf.Tensor): π and radius values for data points
    Returns:
        - updated_qp_matrix (tf.Tensor): Qp_matrix updated with randomly generated queries for current epoch
    '''
    rnd_queries = tf.constant(create_queries(N_TRAIN_QUERIES, QUERY_SIZES, min(data_pi_and_rad[:,1]), max(data_pi_and_rad[:,1])))
    rnd_qp_matrix = tf.constant(create_qp_matrix(rnd_queries, data_pi_and_rad))
    updated_qp_matrix = tf.concat([qp_matrix, rnd_qp_matrix], axis=0)

    mask = tf.reduce_sum(updated_qp_matrix, axis=1) >= 3
    updated_qp_matrix = tf.boolean_mask(updated_qp_matrix, mask)

    return updated_qp_matrix

    
def soft_rank(model_out:tf.Tensor, epoche:int, start_temp:float) -> tuple[tf.Tensor, float]:
    """ To compute a differentiable rank, I use soft_rank with temperature annealing. 
    It computes distances between subsequent model outputs and applies the sigmoid function. 
    If the difference is 0 -> value is 0.5
    If it is super large   -> value goes towards 1
    If it is very small    -> values goes towards 0
    Computing the sum of sigmoid diffs leads to a great value for values with a great rank, and vice versa for small values.
    Even if model outputs are concentrating on as small interval, ranks are spreaded wider.
    
    Args:
        - model_out (tf.Tensor): Raw model outputs
        - epoche    (int)      : Current epoche
        - soft_rank_temp       : Temperature value that defines the "softness" of the ranking. 
        
    Returns:
        - ranks (tf.Tensor): (Soft) ranks
    """
   # current_temp =5.98E-02 * 0.999**epoche if 5.98E-02 * 0.999**epoche > 3.5e-2 else  3.5e-2*1.001**(epoche/10)  #if epoche%100!=0 else 5.98E-02 # start_temp * SOFTR_TEMP_DECAY**epoche 
    current_temp =5.2E-02 * 0.999**epoche if 5.98E-02 * 0.999**epoche > 3.9e-2 else 3.95e-2
    if epoche > 520:
       current_temp = 4.5e-2
    soft_rank_temp = current_temp#current_temp if current_temp > 0.00001 else 0.00001
    diff = tf.expand_dims(model_out, axis = 0) - tf.expand_dims(model_out, axis = 1)
    diff = tf.nn.sigmoid(diff/soft_rank_temp)
    ranks = 0.5 + tf.reduce_sum(diff, axis = 1)
    
    ranks_min, ranks_max = tf.reduce_min(ranks), tf.reduce_max(ranks)
    ranks_normalized = (ranks - ranks_min)/(ranks_max - ranks_min + 1e-8)
    
    return ranks_normalized, soft_rank_temp

    
def compute_knn_loss(embeddings_normalized:tf.Tensor, knn_idx:np.ndarray, knn_dists:np.ndarray, alpha=1.0) -> tf.Tensor:
    """KNN Loss penalizes if two embeddings are to far way, compared the points in the data/grid.
    Alpha defines the ratio between data-point distance and the 1D distance
    
    Args: 
        - embeddings_normalized (tf.Tensor): Embeddings in 1D
        - knn_idx (np.ndarray)             : List of indices of knns for a point
        - knn-dists (np.ndarray)           : List of distances for a point and its knns
        
    Returns:
        - loss (tf.tensor) : knn loss
    """
    h_i = tf.gather(embeddings_normalized, tf.range(tf.shape(embeddings_normalized)[0]))[:, None]     
    h_j = tf.gather(embeddings_normalized, knn_idx)                               
    
    d_h = tf.abs(h_i - h_j)                                   
    d_x = tf.constant(knn_dists, dtype=tf.float32)           
    
    return tf.reduce_mean(tf.square(d_h - alpha * d_x))


def compute_cluster_loss(block_weight_matrix:tf.Tensor) -> tf.Tensor:
    """Uses the block_weight_matrix to compute the cluster loss.
    The block_weight_matrix maps basically queries to blocks (the probabilities, that a point of a query is in a block)
    The dimensionality of block_weight_matrix is N_QUERIES x N_BLOCKS 
    It computes the position-wise distance between subsequent entries for one query
    If there is a lot of up and down in the probabilities -> many blocks are touched.
    If e.g. only the first position is big and the subsquent very low -> distances are low -> sum stays small
    
    Args:
        - block_weight_matrix (tf.Tensor): Assignes the probabilities, that a point belongs to certain block, to the query, that contains this point.
        
    Returns:
        - cluster_loss (tf.Tensor)
    """
    pos_diffs = tf.nn.relu(block_weight_matrix[:,1:] - block_weight_matrix[:,:-1])
    clusters = block_weight_matrix[:, 0] + tf.reduce_sum(pos_diffs, axis=1) 
    return tf.reduce_mean(clusters) 


def compute_block_purity_loss(block_weight_matrix:tf.Tensor) -> tf.Tensor:
    '''Computes column-wise Gini coefficients for block_weight_matrix to determine, how balanced the distribution of weight is.

    Args:
        - block_weight_matrix (tf.Tensor): Assigns the probability, that a certain point belongs to a block to the query, that contains this point.
    Returns:
        - mean_of_column_ginis (tf.Tensor): Mean value over all column-ginis
    '''
    weights = tf.transpose(tf.reduce_sum(block_weight_matrix, axis=0)[:, None])+1e-08
    sqrt_ratios = tf.square(block_weight_matrix/weights)
    column_ginis = 1 - tf.reduce_sum(sqrt_ratios, axis=0)
    
    return tf.reduce_mean(column_ginis)

def compute_rank_span_loss(ranks:tf.Tensor, qp_matrix:tf.Tensor) -> tf.Tensor:
    '''Rank span means: Distance between the minimum and maximum Rank for points within a query. 
    This needs to be optimized in order to keep the amount of blocks being touched by a query small.
    Low rank span -> low number of blocks touched. 
    
    Args:
        - ranks (tf.Tensor): Ranks for data points
        - qp_Matrix (tf.Tensor):  QP Matrix - one row for every query, one column for every point. 1 if point in query, 0 else 
    Returns:
        -mean_rank_span (tf.Tensor): Rank span averaged over all queries
    '''
    ranks_in_query = qp_matrix*ranks[None, :]
    #points_per_query = tf.reduce_sum(qp_matrix, axis=1)
    masked_ranked_in_query_min = tf.where(qp_matrix > 0, ranks[None, :], tf.constant(np.inf, dtype=ranks.dtype))
    max_vals = tf.reduce_max(ranks_in_query, axis=1)
    min_vals = tf.reduce_min(masked_ranked_in_query_min, axis=1)
    rank_span = (max_vals-min_vals)
    with open("test.txt", "a") as myfile:
        np.savetxt(myfile, rank_span.numpy().reshape(1, -1), fmt="%.6f")
        
    return tf.reduce_mean(rank_span)


def compute_query_purity_loss(block_weight_matrix:tf.Tensor, qp_matrix:tf.Tensor) -> tf.Tensor:
    '''Computes gini for every block. The more a query distributes over many blocks, the higher is the gini coefficient.
    Args:
        - block_weight_matrix (tf.Tensor): Assigns the probability, that a certain point belongs to a block to the query, that contains this point.
        - qp_matrix (tf.Tensor): QP Matrix - one row for every query, one column for every point. 1 if point in query, 0 else
    Returns:
        - mean_of_row_ginis (tf.Tensor)
    '''
    weights = tf.transpose(tf.reduce_sum(block_weight_matrix, axis=1)[None,:])+1e-08 #Sum over rows, 1+e-08 to avoid division by 0
    sqrt_ratios = tf.square(block_weight_matrix/weights) #Turns rows to probability distributions
    row_ginis = 1 - tf.reduce_sum(sqrt_ratios, axis=1) #Compute gini
    points_per_query = tf.reduce_sum(qp_matrix, axis=1) #Normalize gini by query size
    # sqrt_ratios = tf.square(block_weight_matrix/weights)
    # row_entropy =-tf.reduce_sum(sqrt_ratios * tf.math.log(sqrt_ratios + 1e-9), axis=1)
    
    return tf.reduce_mean(row_ginis/points_per_query)

    
def compute_lipschitz_regularization(model:tf.keras.Model) -> None:
    '''Computes the biggest singular value for each weight via power iteration
     to perform spectral normalization. This forces the network to deliver 1-Lipschitz output.
     
     Args:
        model (tf.keras.Model): Model that needs regularization 
    '''
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            W = layer.kernel
            v = tf.random.normal([W.shape[1]])
            v /= tf.norm(v)
        
            for _ in range(0,10):
                u = tf.linalg.matvec(W,v)
                u /= tf.norm(u)
                v = tf.linalg.matvec(tf.transpose(W),u)
                v /= tf.norm(v)

            sigma = tf.tensordot(u, tf.linalg.matvec(W, v), axes=1)
            layer.kernel.assign(W /(sigma))
            
            
@save_point_to_cluster_probs(indices=distr_vis_indices, storage=example_distributions)          
def compute_block_weight_matrix(center_dists:tf.Tensor, temp:int, qp_matrix:tf.Tensor) -> tf.Tensor:
    ''' Computes the block_weight_matrix, that assignes the probabilities, that a point belongs to certain block, to the query, that contains this point.
    The block_weight_matrix, results from the center_dists_proba(bilities): NP x NB and the qp_matrix: NQ x NP
    NQ x NP * NP x NB --> NQ x NB
    '''
    
    center_dists_proba = tf.nn.softmax(-(center_dists*center_dists)/temp, axis=1) #N_Points x N_Blocks 
    block_weight_matrix = tf.matmul(qp_matrix, center_dists_proba) #N_Queries x NPoints * N_Points x N_Blocks --> N_Queries x N_Blocks
    #block_weight_matrix = 1.0 - tf.exp(-block_weight_matrix) #Scales values to a range between 0 and 1
    
    return block_weight_matrix, center_dists_proba

temp = GLOBAL_START_TEMP
model, optimizer = define_model()
soft_rank_values =[] #For visualization purpose

mlflow.set_tracking_uri(uri=f"http://{MLFLOW_HOST}:{MLFLOW_PORT}")
mlflow.set_experiment("NeuralCurve_Saddle")

data_pi_and_rad, datapoints, x_tf, qp_matrix, blocks, knn_dists, knn_idx_tf = setup_model_training()
training_metrics = {"Rank Span Loss":[], "Soft Rank Temp":[]} #For logging to MLFlow

lr=LR
with mlflow.start_run() as run:
    for ep in range(EPOCHS):
        qp_matrix_train = extend_qp_matrix_by_random_queries(qp_matrix, data_pi_and_rad)
        if ep > 360 and lr > 1e-6: 
            lr *=0.98
            #lr*=1.01 
            optimizer.learning_rate.assign(lr)
        if ep > 520:
            lr = 1e-5
            optimizer.learning_rate.assign(lr)
        with tf.GradientTape() as tape:
            embeddings = tf.squeeze(model(x_tf), axis=1) #1 x N_Data
            soft_ranks_normalized, soft_rank_temp = soft_rank(embeddings, epoche=ep, start_temp=GLOBAL_START_TEMP)
            soft_rank_values.append(soft_ranks_normalized.numpy())
            
            center_dists = tf.expand_dims(soft_ranks_normalized, axis=1) - tf.expand_dims(blocks, axis=0) #1 x N
            #if ep > 50:
            temp = temp * SOFTM_TEMP_DECAY if (temp * SOFTM_TEMP_DECAY) > TEMP_MIN else TEMP_MIN     
               
            block_weight_matrix = compute_block_weight_matrix(center_dists, temp, qp_matrix_train)
            scan_loss = tf.reduce_mean(tf.reduce_sum(block_weight_matrix, axis=1))
            query_purity_loss = 100*compute_query_purity_loss(block_weight_matrix, qp_matrix_train)
            block_purity_loss = 10*compute_block_purity_loss(block_weight_matrix)
            
            rank_span_loss = 100*compute_rank_span_loss(soft_ranks_normalized, qp_matrix_train)
        
            global_loss = rank_span_loss#qvl+block_purity_loss#100*qvl#rank_span_loss#rank_span_loss+block_purity_loss#scan_loss# + rank_span_loss #rank_span_loss#+query_purity_loss+block_purity_loss#+block_purity_loss#+ cluster_loss #+purity_loss#+ knn_loss #+ uni_loss##+ knn_loss# + uni_loss
            
        gradients = tape.gradient(global_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        training_metrics["Rank Span Loss"].append(rank_span_loss.numpy())
        training_metrics["Soft Rank Temp"].append(soft_rank_temp)
        val_string = analyze_ranking_performance(tf.argsort(tf.argsort(embeddings)).numpy(), qp_matrix_train, "Neural Net:Eval", is_validation=True)
        example_distr_max_example = max(example_distributions[2][ep])
        print(f"Ep: {ep}| RSL: {rank_span_loss:.3f} | QPL: {block_purity_loss:.3f} | SRT: {soft_rank_temp:.2E} | LR: {lr} | Max_Prob: {example_distr_max_example:.3f} | Val: {val_string} ")#| Rank_Span:{rank_span_loss:.3f} |Total:{global_loss:.3f} | LR:{optimizer.lr.numpy():.2E} | SM_TEMP: {temp:.2E} | SR_TEMP:{soft_rank_temp:.2E} | Val: {val_string}")
    
    mlflow.log_artifact("./curve_saddle/neural_curve_saddle.py")
    perform_evaluation(x_tf, datapoints,data_pi_and_rad)
    mlflow.tensorflow.log_model(model, name="Model")
    log_params_to_mlflow()
    log_training_metrics_to_mlflow(training_metrics)
