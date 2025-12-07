import numpy as np

def sort_ranks_by_blocks(blocks_for_points, model_ranks):
    sorted_joined = sorted(list(zip(blocks_for_points, model_ranks)))
    sorted_blocks, sorted_ranks = zip(*sorted_joined)
    return sorted_blocks, sorted_ranks

def compute_block_rank_ranges(blocks_for_points:np.ndarray, model_ranks:np.ndarray) -> np.ndarray:
    sorted_blocks, sorted_ranks = sort_ranks_by_blocks(blocks_for_points, model_ranks)
    split_idcs = np.where(np.diff(sorted_blocks)>0)[0]+1
    ranks_in_blocks = np.split(sorted_ranks, split_idcs)

    return sorted_blocks, ranks_in_blocks


def assign_points_to_blocks(ranks:np.ndarray, n_blocks:int) -> np.ndarray:
    """
    For every rank of every point in the grid/dataset, compute the corresponding block 
    by dividing the ranks by block size.
    Args:
        - ranks (np.ndarray): List of ranks for points in the dataset
    Returns:
        - blocks_for_points (np.ndarray): List of blocks, to which points are assigned
    """
    N = len(ranks)
    blocks_for_points = (ranks * n_blocks) // N   
    blocks_for_points = np.minimum(blocks_for_points, n_blocks - 1) #Avoid rounding errors that lead to values > N_BLOCKS-1
    return blocks_for_points.astype(np.int32)