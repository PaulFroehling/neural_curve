import os
import random 
import numpy as np
import tensorflow as tf 
import mlflow.tensorflow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data.data_loader import load_saddle_data
from curve_saddle.saddle_utils import create_qp_matrix, create_queries
from utils.general_utils import compute_block_rank_ranges, sort_ranks_by_blocks, assign_points_to_blocks
from curve_saddle.constants_saddle import QUERY_SIZES, N_BITS, N_POINTS_PER_AXIS

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


N_POINTS_PER_AXIS = 2**N_BITS
N_BLOCKS=15
PERC_QUERY_CONTEXT = 0.1
N_QUERIES = 5

QUERY_SIZES = {"S":[0.01*np.pi, 0.3*np.pi], "M":[0.3*np.pi, 1.0*np.pi], "L":[1.0*np.pi, 2*np.pi]} #Lower and upper value for lower and upper max_pi value
N_VIS_QUERIES  = {"S":2, "M":2,"L":2}



def compute_query_context(queries:((np.ndarray)), data_pi_and_rad:((np.ndarray)), coordinates:((np.ndarray))) -> ((np.ndarray)):
    '''
    Computes context points around an actual query for visualization purpose. Those context points can demonstrate
    where in the context of a dataset a certain query is located.

    Args:
        - queries (((np.ndarray))): List of queries of shape [min_π, max_π, min_radius, max_radius]
        - data_pi_and_rad (((np.ndarray))): π and radius values for a dataset with a circle base shape
        - coordinates (((np.ndarray))): Coordinates of the corresponding dataset 
    Returns:
        -context_points (((np.ndarray))): Context points for each query
    '''

    query_context = []
    for query in queries:
        context_query = np.array([query[0] - 0.1 * query[0], query[1] + 0.1 * query[1], query[2] - 0.1 * query[2], query[3]+0.1 * query[3]])
        context_query = np.clip(context_query, 0, N_POINTS_PER_AXIS-1)
        query_context.append(context_query)

    query_context = np.array(query_context) 
    context_qp = create_qp_matrix(query_context, data_pi_and_rad)
    context_idcs = [list(np.where(row == 1)[0]) for row in context_qp]
    context_points = [coordinates[idxs] for idxs in context_idcs]

    return context_points


def visualize_blocks_for_queries(   ranks_for_queries:np.ndarray,
                                    blocks_for_queries:np.ndarray,
                                    points_for_queries:np.ndarray,
                                    block_rank_ranges:np.ndarray,
                                    all_blocks:np.ndarray) -> None:
    '''
    Visualizes for one or a set of example queries: 
        - The curve by which the points with in the queries are ordered
        - The blocks to which those points belong (color of the each point corresponds to its block)
        - The rank span for each block + the part of this span, that is covered by a certain query

    Args:
        - ranks_for_queries:(np.ndarray): Ranks within each query
        - blocks_for_queries:(np.ndarray):Blocks for each point in a query
        - points_for_queries:(np.ndarray): Points contained by a query
        - block_rank_ranges:(np.ndarray): Rank ranges per block
        - all_blocks:(np.ndarray)): List of blocks
    Returns:
        - None
    '''
    ranks_blocks_points_for_query =[ #Zip together blocks and points, ordered by ranks. Like this, all lists are in the same order
        sorted(zip(ranks,blocks, points), key=lambda x: x[0])
        for ranks,blocks, points in zip(ranks_for_queries, blocks_for_queries, points_for_queries)
    ]

    colors = plt.cm.tab20.colors

    #Loop through queries - one plot per query
    for query_index, query in enumerate(ranks_blocks_points_for_query):
        _, blocks, points = zip(*query) #Get blocks and points ordered by ranks
        splits =np.where(np.diff(blocks) > 0)[0]+1 #Position 0 gives the split indices +1 for getting the actual position

        unique_blocks = np.unique(blocks) 

        #Split into groups of points that belong to distinct blocks. They should be connected and have the same color
        if len(splits) == 0: #Only one block
            point_splits = np.array([points])
        else:
            point_splits = np.split(points, splits)

        fig = plt.figure(figsize=(15,7))
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1 = fig.add_subplot(1, 2, 2)

        for split_index, split in enumerate(point_splits): #Iterate through splits (i.e. points for blocks, sorted by rank)and connect them to one line
            ax0.scatter(split[:,0], split[:,1],split[:,2], color=colors[unique_blocks[split_index]], label =f"Block {unique_blocks[split_index]}")
            ax0.plot(split[:,0], split[:,1],split[:,2],"--", c=colors[unique_blocks[split_index]], linewidth=0.9)

        ax0.set_title(f"{len(points)} points of one query ordered by rank and colored by block")
        ax0.set_xlabel("X")
        ax0.set_ylabel("Y")
        ax0.legend()
        plot_query_block_coverage(ax1, query_index, block_rank_ranges, blocks_for_queries, ranks_for_queries, all_blocks) #Block coverage for query
        ax1.set_title("Covered rank ranges in blocks by points in query")
        ax1.set_xlabel("Blocks")
        ax1.set_ylabel("Rank Range")
        plt.tight_layout()
        plt.show()


def plot_query_block_coverage(  ax:plt.Axes.axes,
                                query_idx:int,
                                block_rank_ranges:np.ndarray,
                                blocks_for_queries:np.ndarray,
                                ranks_for_queries:np.ndarray,
                                blocks:np.ndarray) -> None:
    '''
    Plots for every block the rank range it covers and adds the rank rank covered by a certain query, in the respective color of each block.

    Args:
        - ax (plt.Axes.axes): Axes, on which the plot will be drawn
        - query_idx (int): Index of the current query
        - block_rank_ranges (np.ndarray): Rank ranges of blocks
        - blocks_for_queries (np.ndarray): Blocks touched by queries
        - ranks_for_queries (np.ndarray): Ranks within each query
        - blocks (np.ndarray): List of blocks 
        
    Returns:
        - None
    '''

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



def create_query_visualizations():
    '''Create visualizations of queries for N_VIS_QUERY many numbers
    
    Args:
        -
    Returns:
        -
    '''
    coordinates, data_pi_and_rad = load_saddle_data(N_POINTS_PER_AXIS-1)
    min_radius, max_radius = min(data_pi_and_rad[:,1]), max(data_pi_and_rad[:,1])
    queries = np.array(create_queries(N_VIS_QUERIES, QUERY_SIZES, min_radius, max_radius))
    model = tf.keras.models.load_model("/home/paul-froehling/Dokumente/Code/neural_curve/mlartifacts/663913090262043014/models/m-7e8af0ba1e7c4c90ab3ff2ee6b8c6dec/artifacts/data/model.keras")


    model_out = tf.squeeze(model(coordinates), axis=1)
    model_ranks = tf.argsort(tf.argsort(model_out)).numpy()
    qp_matrix = create_qp_matrix(queries, data_pi_and_rad)
    blocks = (np.arange(N_BLOCKS) + 0.5) / N_BLOCKS
    blocks_for_points = assign_points_to_blocks(model_ranks, N_BLOCKS)
    _, block_rank_ranges = compute_block_rank_ranges(blocks_for_points, model_ranks)


    point_idxs_for_query = [list(np.where(row==1)[0]) for row in qp_matrix]
    points_for_queries = [coordinates[idxs] for idxs in point_idxs_for_query]
    ranks_for_queries = [model_ranks[idxs] for idxs in point_idxs_for_query]
    blocks_for_queries = [blocks_for_points[idxs] for idxs in point_idxs_for_query]
    visualize_blocks_for_queries(ranks_for_queries, blocks_for_queries, points_for_queries, block_rank_ranges, blocks)


create_query_visualizations()