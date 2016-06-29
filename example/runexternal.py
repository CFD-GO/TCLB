from mpi4py import MPI
import numpy as np
def test(*args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print(rank)
    print(args[0])
    for i in range(3):
        print(np.asarray(args[i]).shape)

