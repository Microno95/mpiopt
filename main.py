import numpy as np
import mpiopt
from mpi4py import MPI


def MainProcess(rank, comm):
    idx = 0
    while True:
        req = []
        for rank_dest in range(1, comm.Get_size()):
            req.append(comm.isend(f"I am {rank} and this is the {idx} iteration", dest=rank_dest, tag=11))
        for r in req:
            r.wait()
        if idx == 2:
            break
        req = []
        data = []
        for rank_source in range(1, comm.Get_size()):
            req.append(comm.irecv(source=rank_source, tag=12))
        for r in req:
            data.append(r.wait())
        yield idx, data
        idx += 1


def WorkerProcess(rank, comm):
    while True:
        req = comm.irecv(source=0, tag=11)
        data = req.wait()
        yield data
        if data == "I am 0 and this is the 2 iteration":
            break
        req = comm.isend(f"I am {rank}", dest=0, tag=12)
        req.wait()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        for __sent, __received in MainProcess(rank, comm):
            print(f"[{rank}]: Sent: {__sent}, Received: {__received}")
    else:
        for __received in WorkerProcess(rank, comm):
            print(f"[{rank}]: Received: {__received}")
