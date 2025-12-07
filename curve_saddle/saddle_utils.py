import tensorflow as tf
import numpy as np
import random

def create_qp_matrix(queries:tf.Tensor, selection_vals:np.ndarray) -> tf.Tensor:
    """Computes the qp_matrix by numpy broadcasting - much faster than using loops.
    1: Separate dimensions of the queries, in pi1,pi2, rad1,rad2
    2: Same approach for the grid
    3: Create a mask for pi, that is 1 if an p1 in a query is smaller than an pi in the data and the p2 is greater.
       Do the same for rad. If both are True --> point is in query --> add a one to qp_matrix for the respective query and the respective point
       REMARK: For some queries: pi2 is smaller than p1, because the query includes 2PI. For all pi2>2PI, I've subtracted 2PI
    Args:
        queries (np.ndarray)        : List of queries
        selection_vals (np.ndarray) : List of selection values for the queries on the saddle (defined by pi and radius value)
    
    Returns:
        qp_matrix (np.ndarray): Matrix mapping points to queries   
    """
    
    q_pi_1, q_pi_2 = np.expand_dims(queries[:,0], axis = 1),  np.expand_dims(queries[:,1], axis = 1)
    q_rad_1, q_rad_2 = np.expand_dims(queries[:,2], axis = 1), np.expand_dims(queries[:,3], axis = 1)
    pi,  rad = np.expand_dims(selection_vals[:,0], axis = 0), np.expand_dims(selection_vals[:,1], axis = 0)
    
    mask_pi_normal = (q_pi_1 <= pi) & (pi <= q_pi_2)
    mask_pi_wrapped = (q_pi_1 >= pi) | (pi <= q_pi_2)

    mask_pi = np.where(q_pi_1 > q_pi_2 , mask_pi_wrapped, mask_pi_normal)

    mask_rad = (q_rad_1 <= rad) & (rad <= q_rad_2)

    qp_matrix = (mask_pi & mask_rad).astype(np.float32)

    
    return tf.constant(qp_matrix, dtype=tf.float32)


def create_queries(n_queries_dict:dict, queries_sizes:dict, min_radius:int, max_radius:int) -> list[list[float, float, float, float]]:
    """ Creates queries based on values for pi and radius --> cutting out segments of the saddle with one whole
    
    Args:
        - n_queries (dict)    : Number of queries for query type (S,M,L)
        - queries_sizes (dict): PI-value span for each query type
    
    Returns: 
        -queries (list[list[min_pi,max_pi,min_radius, max_radius]]): List of queries.
    """
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