import mlflow.tensorflow
import numpy as np
import random 
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MIN_QUERY_WIDTH = 2
MAX_QUERY_WIDTH = 5
N_DIMS = 2 
N_BITS = 4
N_POINTS_PER_AXIS = 2**N_BITS
N_BLOCKS=15
PERC_QUERY_CONTEXT = 0.1

QUERY_SIDE_LENGTHS = {"S":[1,3], "M":[4,7], "L":[8,12]}

N_VIS_QUERIES  = {"S":3, "M":3,"L":1}



def create_qp_matrix(queries:np.ndarray, grid:np.ndarray) -> np.ndarray:
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
    
    return qp_matrix

def load_digits_dataset():
    digits_data = np.loadtxt("./data/in/digits/digits.csv", delimiter=",")
    digits_data_scaled = digits_data * (N_POINTS_PER_AXIS-1)
    
    return digits_data_scaled


def create_queries(n_queries_dict:dict, queries_sizes:dict) -> list[list[int, int]]:
    """ Creates queries that are used for model training. Each query is randomly placed and has random width (within lower and upper border)
    A query is defines as [covered x-axis area, covered_y_axis_area] -> [[x,x],[y,y]]
    
    Args:
        - n_queries (dict)    : Number of queries for query type (S,M,L)
        - queries_sizes (dict): Span-width for each query type
    
    Returns: 
        -queries (list[list[int, int]]): List of queries.
    """
    queries = []
    for q_type in n_queries_dict:
        n_queries = n_queries_dict[q_type]
        for _ in range(n_queries):
            min_side_length, max_side_length = queries_sizes[q_type]
            
            query_width  = min_side_length + ((max_side_length-min_side_length)*random.random())
            query_height = min_side_length + ((max_side_length-min_side_length)*random.random())
            
            x_0 = random.random() * max(0, (N_POINTS_PER_AXIS-1-query_width))
            y_0 = random.random() * max(0, (N_POINTS_PER_AXIS-1-query_height))
            
            query = np.array([[x_0, query_width + x_0], [y_0, query_height + y_0]])
            
            if random.random() < 0.1:     #cancel selection in one dimension with 10% chance
                if random.random() < 0.5: #cancel selection on x axis
                    query[0] =[0, N_POINTS_PER_AXIS-1]
                else:                     #cancel selection on y axis
                    query[1] =[0, N_POINTS_PER_AXIS-1]
                    
            queries.append(query)
    
    return queries

def compute_query_context(queries:np.ndarray, points:np.ndarray) -> np.ndarray:
    query_context = []
    for query in queries:
        x_context = np.clip([query[0][0]- PERC_QUERY_CONTEXT * query[0][0], query[0][1]+ PERC_QUERY_CONTEXT * query[0][1]], 0, N_POINTS_PER_AXIS-1)
        y_context = np.clip([query[1][0]- PERC_QUERY_CONTEXT * query[1][0], query[1][1]+ PERC_QUERY_CONTEXT * query[1][1]], 0, N_POINTS_PER_AXIS-1)
        query_context.append([x_context,y_context])
    
    query_context = np.array(query_context) 

    context_qp = create_qp_matrix(query_context, points)
    context_idcs = [list(np.where(row == 1)[0]) for row in context_qp]
    context_points = [points[idxs] for idxs in context_idcs]

    return context_points

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
    blocks_for_points = np.minimum(blocks_for_points, N_BLOCKS - 1) #Avoid rounding errors that lead to values > N_BLOCKS-1
    return blocks_for_points.astype(np.int32)


def visualize_clusters_for_queries(blocks_for_queries:np.ndarray, points_for_queries:np.ndarray):
    blocks_for_queries_with_points =[ #zip together blocks and points, ordered by blocks
        sorted(zip(blocks, points), key=lambda x: x[0])
        for blocks, points in zip(blocks_for_queries, points_for_queries)
    ]

    sorted_blocks_for_queries = [np.array(query, dtype=object)[:,0] for query in blocks_for_queries_with_points]
    sorted_points_for_queries = [
        np.stack([p for _, p in query])  
        for query in blocks_for_queries_with_points
    ]

    #Find indices, where two subsequent points have block dists > 1 --> new cluster
    splits = [np.where(np.diff(blocks) > 1)[0] + 1 for blocks in sorted_blocks_for_queries]

    #Split list of points on those positions, where new clusters begin
    splitted_points_for_queries = [
        np.split(sorted_points_for_queries[i], splits[i])
        for i in range(N_QUERIES)
    ]

    for query in splitted_points_for_queries:
        for split in query:
            plt.plot(split[:,0], split[:,1], marker="o")
        plt.show()

def visualize_blocks_for_queries(queries, ranks_for_queries, blocks_for_queries, points_for_queries, block_rank_ranges, query_context, all_blocks):

    ranks_blocks_points_for_query =[ #zip together blocks and points, ordered by ranks
        sorted(zip(ranks,blocks, points), key=lambda x: x[0])
        for ranks,blocks, points in zip(ranks_for_queries, blocks_for_queries, points_for_queries)
    ]

    colors = plt.cm.tab20.colors

    #loop through queries - one plot per query
    for query_index, query in enumerate(ranks_blocks_points_for_query):
        _, blocks, points = zip(*query) #get blocks and points ordered by ranks
        splits =np.where(np.diff(blocks) > 0)[0]+1 #0 gives the split indices +1 for getting the actual position

        unique_blocks = np.unique(blocks) 

        #split into groups of points that belong to distinct blocks
        if len(splits) == 0:
            point_splits = np.array([points])
        else:
            point_splits = np.split(points, splits)

        fig, ax = plt.subplots(1, 2, figsize=(10,4))  
        query_rect = queries[query_index]
        query_rect[0][1] -= query_rect[0][0]
        query_rect[1][1] -= query_rect[1][0] 
        query_rect = query_rect.T #reformat queries from [x1,x2],[widht, height] -> needed by patch.Rectangle
        rect = patches.Rectangle((query_rect[0][0], query_rect[0][1]), query_rect[1][0], query_rect[1][1], linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)

        context = query_context[query_index] #plot context points around the query - just for optimal effect
        ax[0].scatter(context[:,0], context[:, 1], c="gray")
        for split_index, split in enumerate(point_splits): #iterate through splits (i.e. points for blocks, sorted by rank)and connect them to one line
            ax[0].scatter(split[:,0], split[:,1], color=colors[unique_blocks[split_index]], label =f"Block {unique_blocks[split_index]}")
            ax[0].plot(split[:,0], split[:,1],"--", c=colors[unique_blocks[split_index]], linewidth=0.9)

        ax[0].set_title(f"{len(points)} points of one query ordered by rank and colored by block")
        ax[0].set_xlabel("X")
        ax[0].set_ylabel("Y")
        ax[0].legend()
        plot_query_block_coverage(ax[1], query_index, block_rank_ranges, blocks_for_queries, ranks_for_queries, all_blocks)
        ax[1].set_title("Covered rank ranges in blocks by points in query")
        ax[1].set_xlabel("Blocks")
        ax[1].set_ylabel("Rank Range")
        plt.tight_layout()
        plt.show()

def plot_query_block_coverage(ax, query_idx, block_rank_ranges:np.ndarray, blocks_for_queries:np.ndarray, ranks_for_queries:np.ndarray, blocks:np.ndarray):
    blocks = list(range(len(blocks)))
    min_blocks_vals, max_block_vals = np.array([[min(rank_vals), max(rank_vals)] for rank_vals in block_rank_ranges]).T #min and max rank values for each block
    max_block_vals -= min_blocks_vals

    colors = plt.cm.tab20.colors
    
    blocks_q, ranks_for_blocks_q = compute_block_rank_ranges(blocks_for_queries[query_idx], ranks_for_queries[query_idx]) #rank ranges for a distinct query
    min_q_block_vals, max_q_block_vals = np.array([[[min(block_vals), max(block_vals)]] for block_vals in ranks_for_blocks_q]).T
    min_q_block_vals,max_q_block_vals = min_q_block_vals[0], max_q_block_vals[0] #min and max rank value for one block, touched by a query
    max_q_block_vals -= min_q_block_vals
    blocks_q = np.unique(blocks_q)

    for block_idx, block in enumerate(blocks): #loop through blocks and their corresponding min max rank, to plot the rank range being covered
        ax.bar(blocks[block_idx], max_block_vals[block_idx], bottom = min_blocks_vals[block_idx], color="gray")
        if block in blocks_q: #if a block is touched by a query, plot the min and max rank values of the points in this block, covered by the query
            idx_of_q_block = np.where(blocks_q==block)[0][0]
            ax.bar(blocks[block_idx], max_q_block_vals[idx_of_q_block], bottom=min_q_block_vals[idx_of_q_block], color=colors[block_idx])


def plot_block_distribution(blocks_for_points:np.ndarray) -> None:
    blocks,counts = np.unique(blocks_for_points, return_counts=True)
    plt.bar(blocks, counts)
    plt.show()


def plot_model_output_distribution(model_out):
    plt.plot(sorted(model_out))
    plt.show()

def sort_ranks_by_blocks(blocks_for_points, model_ranks):
    sorted_joined = sorted(list(zip(blocks_for_points, model_ranks)))
    sorted_blocks, sorted_ranks = zip(*sorted_joined)
    return sorted_blocks, sorted_ranks

def compute_block_rank_ranges(blocks_for_points:np.ndarray, model_ranks:np.ndarray) -> np.ndarray:
    sorted_blocks, sorted_ranks = sort_ranks_by_blocks(blocks_for_points, model_ranks)
    split_idcs = np.where(np.diff(sorted_blocks)>0)[0]+1
    ranks_in_blocks = np.split(sorted_ranks, split_idcs)

    return sorted_blocks, ranks_in_blocks

model = tf.keras.models.load_model("/home/paul-froehling/Dokumente/Code/neural_curve/mlartifacts/186531212038274695/models/m-98282a95db4645d6874fd98bd286d781/artifacts/data/model.keras")
N_QUERIES = 5
queries = np.array(create_queries(N_VIS_QUERIES, QUERY_SIDE_LENGTHS))

data = load_digits_dataset()
model_out = tf.squeeze(model(data), axis=1)
model_ranks = tf.argsort(tf.argsort(model_out)).numpy()
qp_matrix = create_qp_matrix(queries, data)
blocks = (np.arange(N_BLOCKS) + 0.5) / N_BLOCKS
blocks_for_points = assig_points_to_blocks(model_ranks)
_,block_rank_ranges = compute_block_rank_ranges(blocks_for_points, model_ranks)


point_idxs_for_query = [list(np.where(row==1)[0]) for row in qp_matrix]
points_for_queries = [data[idxs] for idxs in point_idxs_for_query]
ranks_for_queries = [model_ranks[idxs] for idxs in point_idxs_for_query]
blocks_for_queries = [blocks_for_points[idxs] for idxs in point_idxs_for_query]
query_context = compute_query_context(queries, data)
#plot_query_block_coverage(block_rank_ranges, blocks_for_queries, ranks_for_queries, blocks)
visualize_blocks_for_queries(queries, ranks_for_queries, blocks_for_queries, points_for_queries, block_rank_ranges, query_context, blocks)
#plot_block_distribution(blocks_for_points)
#plot_model_output_distribution(model_out)
#visualize_clusters_for_queries(blocks_for_queries, points_for_queries)

