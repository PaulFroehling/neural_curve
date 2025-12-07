import numpy as np

def load_saddle_data(points_per_axis:int) -> tuple[np.ndarray, np.ndarray]:
    saddle_data = np.loadtxt("./data/in/saddle/saddle_3d.csv", delimiter=",") #pi, rad, x, y, z
    saddle_data_scaled = np.hstack((saddle_data[:, [0]], saddle_data[:, 1:] * points_per_axis))#scale everything except pi
    coordinates = saddle_data_scaled[:,2:]
    pi_and_rad = saddle_data_scaled[:,:2]
    print(f"Minima per axis: {coordinates.min(axis=0)}")
    print(f"Minima per axis: {coordinates.max(axis=0)}")
    
    return coordinates, pi_and_rad