import numpy as np
import cupy as cp
import numba as nb
from numba import cuda
import math

@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
        an_array[x, y] += 1.0


def main():

    # Specify the dimensions of the array (assumed to be square)
    N = 256

    # Create a 2-D array on the host
    A = np.ones([N,N], dtype=np.float64)

    # Now convert this into a device array
    A_gpu = cp.asarray(A)
    #A_gpu = cuda.to_device(A)

    # Now launch the CUDA kernel on the device
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    increment_a_2D_array[blockspergrid, threadsperblock](A_gpu)

    # Sync devices
    cuda.synchronize()

    # Copy the data back to the host
    A_new = cp.asnumpy(A_gpu)
    #A_new = A_gpu.copy_to_host()

    # Print to the console
    print("A_new =", A_new, "\n")


if __name__ == "__main__":

    main()







