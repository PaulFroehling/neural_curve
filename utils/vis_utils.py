import matplotlib.pyplot as plt
import numpy as np

def plot_block_distribution(blocks_for_points:np.ndarray) -> None:
    blocks,counts = np.unique(blocks_for_points, return_counts=True)
    plt.bar(blocks, counts)
    plt.show()


def plot_model_output_distribution(model_out):
    plt.plot(sorted(model_out))
    plt.show()