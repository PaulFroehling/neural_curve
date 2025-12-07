import numpy as np
import random

MIN_QUERY_WIDTH = 2
MAX_QUERY_WIDTH = 15
N_DIMS = 2 
N_BITS = 4
N_POINTS_PER_AXIS = 2**N_BITS


def create_queries(n_queries:int) -> list[list[int, int]]:
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

    for _ in range(n_queries):
        query_width = MIN_QUERY_WIDTH + int(((MAX_QUERY_WIDTH-MIN_QUERY_WIDTH)*random.random()))
        query = np.array([[0,query_width]]*N_DIMS)
        query_shifted = [dim+(random.random()*(N_POINTS_PER_AXIS-query_width)) for dim in query]
        queries.append(query_shifted)
        
    _, counts = np.unique(queries, return_counts=True)
    print(np.where(counts>1))
    
    return queries


print(create_queries(10000))