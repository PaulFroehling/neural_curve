import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import pandas as pd
import random

from itertools import product
from hilbert import encode
from sklearn.neighbors import NearestNeighbors


N_DIMS = 2 
N_BITS = 4
N_POINTS_PER_AXIS = 2**N_BITS
N_GRID_POINTS     = (N_POINTS_PER_AXIS)**N_DIMS
N_BLOCKS        = 40
k               = 6
ALPHA           = 0.7
EPOCHS          = 370 #400 #300 <-latest
LR              = 1e-4 #1e-5
N_QUERIES       = 1000
MIN_QUERY_WIDTH = 2
MAX_QUERY_WIDTH = 15

TEMP        = 2.0
TEMP_MIN    = 0.00001
TEMP_DECAY  = 0.98 

CLUSTER_LOSS_WEIGHT = 12

example_distributions = []
distr_vis_indices = [0,63,122,255]

##########################################################
                    #Evaluation
##########################################################
   
def visualize_cluster_point_assignment(distribution:np.ndarray, ax):
    distr = pd.DataFrame(distribution)
    distr_long = distr.T.melt(var_name="Epoche", value_name="Probability") 
    sns.lineplot(data=distr_long, x=distr_long.groupby("Epoche").cumcount(),y="Probability", hue="Epoche", ax=ax)

def plot_curve_of_trained_model(ranks, x_tf):
    x_sorted_np = tf.gather(x_tf, ranks).numpy()
    plt.figure(figsize=(6,6))
    plt.plot(x_sorted_np[:,0], x_sorted_np[:,1], 'o-', linewidth=0.8, alpha=0.8, color='gray')
    plt.title("Neural Curve")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
def plot_point_block_assignment(bocks_for_points:np.ndarray) -> None:
    plt.plot(bocks_for_points)
    plt.show()

def plot_block_weight_hist(blocks_in_queries:np.ndarray) -> None:
    """
    Plots the distribution of points within blocks. 
    This is especially useful to find out how the model learns to distribute points.
    Args:
        - blocks_in_queries (np.ndarray): A matrix mapping blocks to queries
    Returns:
        -
    """
    blocks, counts = np.unique(blocks_in_queries, return_counts=True)
    plt.bar(blocks[1:], height=counts[1:])
    plt.show()
    

def calc_query_statistics(blocks_in_queries: np.ndarray) -> tuple[list,float,float]:
    """
    Caculates how many blocks are used in total, how many blocks are touched by a query on average and
    how man clusters can be found within a query. A cluster is a set of consecutive numbered blocks
    
    Args:
        - blocks_in_queries (np.ndarray): A matrix mapping blocks to queries
    Returns:
        - results (tuple): used_blocks (set), mean_blocks_per_query (float), mean_clusters_per_query (float)
    """
    
    used_blocks = []
    blocks_per_query = []
    clusters_for_query = []

    for query in blocks_in_queries:
        query = query[query != -1]          #Take only the blocks that are touched by a query. -1 say its not part of the query
        unique_blocks = np.unique(query)    #Erease double values
        used_blocks.extend(unique_blocks)   # Add blocks of this query to used_blocks --> Overview which blocks are used in general
        blocks_per_query.append(len(unique_blocks)) #No blocks touched by query

        #Cluster means consecutive block values. Sequential read is always better than jumping between blocks
        if len(unique_blocks) == 0:
            clusters = 0
        else:
            sorted_blocks = np.sort(unique_blocks)
            diffs = np.diff(sorted_blocks) #Diffs between consecutive block numbers
            clusters = np.count_nonzero(diffs > 1) + 1 #If there is a gap>1 --> new cluster --> no sequential read
        clusters_for_query.append(clusters)
        results = set(used_blocks), np.mean(blocks_per_query), np.mean(clusters_for_query)

    return results 


def assig_points_to_blocks(ranks:np.ndarray) -> np.ndarray:
    """
    For every rank of every point in the grid/dataset, compute the corresponding block 
    by dividing the ranks by block size.
    Args:
        - ranks (np.ndarray): List of ranks for points in the dataset
    Returns:
        - blocks_for_points (np.ndarray): List of blocks, to which points are assigned
    """
    N = len(ranks)
    blocks_for_points = (ranks * N_BLOCKS) // N   
    blocks_for_points = np.minimum(blocks_for_points, N_BLOCKS - 1) 
    return blocks_for_points.astype(np.int32)

def assign_blocks_to_queries(qp_matrix:np.ndarray, blocks_for_points:np.ndarray) -> np.ndarray:
    """Each query embraces 1 or more blocks. This relationship is reflected by blocks_in_queries, resulting from this function.
    For this, a mask is created, that repeats blocks_for_points N_QUERIES-times.
    The mask then has the same dimensionality as the qp_matrix. qp_matrix: N_QUERIES x N_POINTS
    The mask is applied to the qp_matrix (which is 1 if a point is in a query, 0 else)
    If there is a in the qp_matrix -> insert the corresponding block of the this point, else set it to -1
    
    Args:
        - qp_matrix         (np.ndarray): Matrix, mapping points to queries
        - blocks_for_points (np.ndarray): Matrix mapping blocks to points
    Returns:
        - blocks_in_queries (np.ndarray): Matrix mapping blocks to queries
    
    """
    mask = np.tile(blocks_for_points[None,:], (qp_matrix.shape[0], 1)) 
    blocks_in_queries = np.where(qp_matrix.numpy() > 0, mask, -1)
    
    return blocks_in_queries


def eval_model_vs_hilbert(model:tf.keras.Model, x_tf:tf.Tensor, grid:np.ndarray, qp_matrix:np.ndarray) -> None:
    """
    Evaluates the performance of the neural net against the performance of the Hilbert curve.
    Args:
        - model (tf.keras.Model): Trained Neural Net
        - grid  (np.ndarray)    : Gridpoints/datapoints
        - qp_matrix (np.ndarray): Matrix assigning queries to points
    Returns:
        -
    """
    model_embeddings = tf.squeeze(model(x_tf), axis=1) #Calculate the 1D value for all points given
    model_ranks = tf.argsort(tf.argsort(model_embeddings)).numpy() #For all points the rank is the position of the point in the ordered list of embeddings
    hilber_ranks = encode(grid, num_dims=N_DIMS, num_bits = N_BITS)          
    analyze_ranking_performance(model_ranks, qp_matrix, "NEURAL_NET")
    analyze_ranking_performance(hilber_ranks, qp_matrix, "HILBERT")
    plot_curve_of_trained_model(model_ranks, x_tf)


def analyze_ranking_performance(ranks:np.ndarray, qp_matrix:np.ndarray, model_name:str) -> None:
    """Computes the number of used blocks, average blocks per query and average clusters per query
    for a given list of ranks.
    
    Args:
        - ranks (np.ndarray)     : Ranks from a model
        - qp_matrix  (np.ndarray): Matrix assigning queries to points
        - model_name (str)       : Name of the model, that is evaluated
    Returns:
        -
    
    """
    blocks_for_points = assig_points_to_blocks(ranks)
    blocks_in_queries = assign_blocks_to_queries(qp_matrix, blocks_for_points)
    n_used_blocks, avg_blocks_per_query, avg_clusters_per_query = calc_query_statistics(blocks_in_queries)
    plot_block_weight_hist(blocks_in_queries)
    plot_point_block_assignment(blocks_for_points)
    
    print(f"{model_name}: Used blocks: {set(n_used_blocks)}")
    print(f"{model_name}: Avg blocks per query: {avg_blocks_per_query:.2f}")
    print(f"{model_name}: Avg clusters per query: {avg_clusters_per_query:.2f}")
    
    
def save_point_to_cluster_probs(indices, storage):
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


##########################################################
                         #Setup
##########################################################

def build_knn_graph(grid:np.ndarray, k:int=4) -> tuple[list, list]:
    """ Computes the knn graph for every point in the grid.
      Args:
        - grid (np.ndarray): List of gridpoints
        - k    (int)       : Number of neigbors for knn computation
    Returns:
        - results (tuple): List of indices and list of distances
    """
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(grid)
    dists, idxs = nn.kneighbors(grid)
    return idxs[:,1:], dists[:,1:] 


def create_queries(n_queries:int, n_dims:int, n_bits:int) -> list[list[int, int]]:
    """ Creates queries that are used for model training. Each query is randomly placed and has random width (within lower and upper border)
    All queries are squared.
    A query is defines as [lower_left, upper_right] -> [[x,y],[x,y]]
    
    Args:
        - n_queries (int): Number of queries
        - n_dims    (int): Number of dimensions
        - n_bits    (int): Number of bits
    
    Returns: 
        -queries (list[list[int, int]]): List of queries.
    """
    queries = []
    values_per_axis = 2**n_bits
    for _ in range(n_queries):
        query_width = MIN_QUERY_WIDTH + int(((MAX_QUERY_WIDTH-MIN_QUERY_WIDTH)*random.random()))
        query = np.array([[0,query_width]]*n_dims)
        query_shifted = [dim+int(random.random()*(values_per_axis-query_width)) for dim in query]
        queries.append(query_shifted)
    
    return queries


def calc_grid_points_for_intervals(ranges:list[list[int, int]]) -> np.ndarray[np.ndarray[int, int]]:
    """ Computes the grid points for a set of ranges (these are the queries in this case).
    
    Args:
        - ranges (list[list[int, int]]): Ranges for which encapsulated points need to be computed
    Returns:
        -points  (list[list[int, int]]): List of points, ecapsulated by ranges
    """
    
    ranges = [range(start, end+1) for start,end in ranges]
    points = np.array(list(product(*ranges)))
    return points


def define_model() -> tuple[tf.keras.Model, tf.keras.optimizers.Adam]:
    """Defines the neural network
    """
    inputs = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(256, activation="softplus")(inputs)
    x = tf.keras.layers.Dense(512, activation="softplus")(x)
    x = tf.keras.layers.Dense(1024, activation="softplus")(x)
    x = tf.keras.layers.Dense(64, activation="softplus")(x)
    x = tf.keras.layers.Dense(32, activation="softplus")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    opt = tf.keras.optimizers.Adam(LR)
    
    return model, opt

def create_qp_matrix(queries:np.ndarray, grid:np.ndarray) -> tf.Tensor:
    """Computes the qp_matrix by numpy broadcasting - much faster than using loops.
    1: Separate dimensions of the queries, in x1,x2 and y1,y2
    2: Same approach for the grid
    3: Create a mask for x, that is 1 if an x1 in a query is smaller than an x in the grid and the x2 is greater.
       Do the same for y. If both are True --> point is in query --> add a one to qp_matrix for the respective query and the respective point
    
    Args:
        queries (np.ndarray) : List of queries
        grid    (np.ndarray) : List of gridpoints
    
    Returns:
        qp_matrix (np.ndarray): Matrix mapping points to queries   
    """
    
    q_x1, q_x2 = np.expand_dims(queries[:,0,0], axis = 1),  np.expand_dims(queries[:,0,1], axis = 1)
    q_y1, q_y2 = np.expand_dims(queries[:,1,0], axis = 1), np.expand_dims(queries[:,1,1], axis = 1)
    g_x,  g_y = np.expand_dims(grid[:,0], axis = 0), np.expand_dims(grid[:,1], axis = 0)
    
    mask_x = (q_x1 <= g_x) & (g_x <= q_x2)
    mask_y = (q_y1 <= g_y) & (g_y <= q_y2)

    qp_matrix =  (mask_x & mask_y).astype(np.float32)
    
    return tf.constant(qp_matrix, dtype=tf.float32)

##########################################################
                         #Training
##########################################################
    
def soft_rank(model_out:tf.Tensor, epoche:int, soft_rank_temp:int=0.3) -> tf.Tensor:
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
    current_temp = soft_rank_temp * 0.95**epoche
    soft_rank_temp = current_temp if current_temp > 0.001 else 0.0001
    diff = tf.expand_dims(model_out, axis = 0) - tf.expand_dims(model_out, axis = 1)
    diff = tf.nn.sigmoid(diff/soft_rank_temp)
    ranks = 0.5 + tf.reduce_sum(diff, axis = 1)
    
    return ranks
    
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
        - (block_weight_matrix (tf.Tensor): Matrix mapping probabilities, that a point is in a block to queries. 
        
    Returns:
        - cluster_loss (tf.Tensor)
    """
    pos_diffs = tf.nn.relu(block_weight_matrix[:,1:] - block_weight_matrix[:,:-1])
    clusters = block_weight_matrix[:, 0] + tf.reduce_sum(pos_diffs, axis=1) 
    return tf.reduce_mean(clusters) 


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
    
    """ Computes the block_weight_matrix, that assignes the probabilities, that a point belongs to certain block, to the query, that contains this point.
    The block_weight_matrix, results from the center_dists_proba(bilities): NP x NB and the qp_matrix: NQ x NP
    NQ x NP * NP x NB --> NQ x NB
    
    """
    center_dists_proba = tf.nn.softmax(-(center_dists*center_dists)/temp, axis=1)  
    block_weight_matrix = tf.matmul(qp_matrix, center_dists_proba)
    block_weight_matrix = 1.0 - tf.exp(-block_weight_matrix) #Scales values to a range between 0 and 1
    
    return block_weight_matrix, center_dists_proba


grid = np.array(calc_grid_points_for_intervals([[0,N_POINTS_PER_AXIS-1], [0,N_POINTS_PER_AXIS-1]]), dtype=np.float32)
h_idxs = np.array(encode(grid, num_dims=N_DIMS, num_bits=N_BITS))

x_tf = tf.constant(grid, dtype=tf.float32)
model, optimizer = define_model()
queries = tf.constant(create_queries(N_QUERIES, N_DIMS, N_BITS), dtype=tf.float32)
qp_matrix = create_qp_matrix(queries, grid)
#blocks = tf.cast(tf.linspace(0,1,N_BLOCKS), tf.float32)
blocks = (tf.range(N_BLOCKS, dtype = tf.float32) + 0.5) / N_BLOCKS
loss_hist=[]
temp = TEMP
knn_idx, knn_dists = build_knn_graph(grid, k=4)
knn_idx_tf = tf.constant(knn_idx, dtype=tf.int32)
knn_dists = knn_dists.astype(np.float32)


soft_rank_values =[]
temp_values = []
scan_loss_values = []

for ep in range(EPOCHS):
    with tf.GradientTape() as tape:
        embeddings = tf.squeeze(model(x_tf), axis=1)
        ranks = soft_rank(embeddings, epoche=ep)
        ranks_min, ranks_max = tf.reduce_min(ranks), tf.reduce_max(ranks)
        ranks_normalized = (ranks - ranks_min)/(ranks_max - ranks_min + 1e-8)
        soft_rank_values.append(ranks_normalized.numpy())
        center_dists = tf.expand_dims(ranks_normalized, axis=1) - tf.expand_dims(blocks, axis=0)
        
        temp = temp * TEMP_DECAY if (temp * TEMP_DECAY) > TEMP_MIN else TEMP_MIN
        temp_values.append(temp)
        
        block_weight_matrix = compute_block_weight_matrix(center_dists, temp, qp_matrix)
        cluster_loss = CLUSTER_LOSS_WEIGHT*compute_cluster_loss(block_weight_matrix)
        knn_loss = compute_knn_loss(ranks_normalized, knn_idx_tf, knn_dists, alpha=0.1)
        scan_loss = tf.reduce_mean(tf.reduce_sum(block_weight_matrix, axis=1))
        scan_loss_values.append(scan_loss.numpy())
        
        loss = scan_loss + cluster_loss + knn_loss
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    compute_lipschitz_regularization(model)

    print(f"Epoch: {ep} Scan-Loss: {scan_loss:.3f} Cluster_Loss:{cluster_loss:.3f} KNN-Loss:{knn_loss:.3f} Total_Loss:{loss:.3f} Temperature: {temp:.3f}")

eval_model_vs_hilbert(model, x_tf, grid, qp_matrix)


side_length = int(len(example_distributions) / 2) if len(example_distributions) % 2 == 0 else int(len(example_distributions)//2+1)
fig, axs = plt.subplots(side_length, 2)
for i in range(0,2):
    for j in range(0, side_length):
        sns_plot = visualize_cluster_point_assignment(example_distributions[i+j], ax=axs[i,j])


plt.show()        
# axs[0, 0].plot(x, y)
# axs[0, 0].set_title('Axis [0, 0]')
# axs[0, 1].plot(x, y, 'tab:orange')
# axs[0, 1].set_title('Axis [0, 1]')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1, 0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1, 1]')

# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')

# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()



# distr = pd.DataFrame(example_distr_1)
# distr_long = distr.T.melt(var_name="Epoche", value_name="Probability") 
# sns.lineplot(data=distr_long, x=distr_long.groupby("Epoche").cumcount(),y="Probability", hue="Epoche")
# plt.xlabel("Block")
# plt.title("Probability that point 1 of the grid belongs to a block")
# plt.show()

# distr_2 = pd.DataFrame(example_distr_2)
# distr_long_2 = distr_2.T.melt(var_name="Epoche", value_name="Probability") 
# sns.lineplot(data=distr_long_2, x=distr_long_2.groupby("Epoche").cumcount(),y="Probability", hue="Epoche")
# plt.xlabel("Block")
# plt.title("Probability that point 2 of the grid belongs to a block")
# plt.show()


rank_distr = pd.DataFrame(soft_rank_values)   # shape: epochs x points
rank_distr["Epoch"] = rank_distr.index

ranks_long = rank_distr.melt(
    id_vars="Epoch", 
    var_name="Point", 
    value_name="Rank"
)
sns.lineplot(
    data=ranks_long,
    x="Point", y="Rank", hue="Epoch",
    estimator=None, lw=0.8, alpha=0.7
)

plt.xlabel("Point")
plt.title("Softranks")
plt.show()



fig, ax1 = plt.subplots()

ax1.plot(temp_values, color="red")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Temperature")

ax2 = ax1.twinx()
ax2.plot(scan_loss_values, color="blue", label="Epochs")
ax2.set_ylabel("Scan-Loss")


fig.tight_layout()
plt.show()

