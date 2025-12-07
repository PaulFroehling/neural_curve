import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


def create_queries(n_queries:int, n_dims:int, n_bits:int) -> list[list[int, int]]:
    """ Creates queries that are used for model training. Each query is randomly placed and has random width (within lower and upper border)
    All queries are squared.
    A query is defines as [covered x-axis area, covered_y_axis_area] -> [[x,x],[y,y]]
    
    Args:
        - n_queries (int): Number of queries
        - n_dims    (int): Number of dimensions
        - n_bits    (int): Number of bits
    
    Returns: 
        -queries (list[list[int, int]]): List of queries.
    """
    queries = []
    for _ in range(n_queries):
        query_width = MIN_QUERY_SIDELENGTH + ((MAX_QUERY_SIDELENGTH-MIN_QUERY_SIDELENGTH)*random.random())
        query_height = MIN_QUERY_SIDELENGTH + ((MAX_QUERY_SIDELENGTH-MIN_QUERY_SIDELENGTH)*random.random())
        x_0 = random.random() * max(0, (N_POINTS_PER_AXIS-1-query_width))
        y_0 = random.random() * max(0, (N_POINTS_PER_AXIS-1-query_height))
        query = np.array([[x_0, query_width + x_0], [y_0, query_height + y_0]])
        
        if random.random() < 0.1:
            if random.random() < 0.5: #cancel selection on x axis
                query[0] =[0, N_POINTS_PER_AXIS-1]
            else:                     #cancel selection on y axis
                query[1] =[0, N_POINTS_PER_AXIS-1]
                
        queries.append(query)
    
    return queries

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


MIN_QUERY_SIDELENGTH = 1
MAX_QUERY_SIDELENGTH = 8
N_DIMS = 2 
N_BITS = 4
N_POINTS_PER_AXIS = 2**N_BITS
N_QUERIES = 20

datapoints = load_digits_dataset()
q = np.array(create_queries(N_QUERIES, N_DIMS, N_BITS))
qp = np.array(create_qp_matrix(q, datapoints))
print(qp.shape)

points_per_query = np.sum(qp, axis=1)
print(np.where(np.sum(qp, axis=1)<100))

# q_coords = q.swapaxes(-1, -2) 
# print(q_coords)
# q_coords_formatted = [[query[0], query[1]-query[0]] for query in q_coords]
# print(q_coords_formatted)
fig, ax = plt.subplots()

ax.scatter(datapoints[:,0],datapoints[:,1], linewidth=2)
for rect in q:
    ax.add_patch(patches.Rectangle((rect[0][0], rect[1][0]), rect[0][1]-rect[0][0], rect[1][1]-rect[1][0], linewidth=1.3, edgecolor='red', facecolor='none', linestyle="--"))

plt.show()

# plt.hist(points_per_query, bins=18)
# plt.show()