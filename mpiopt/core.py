import os
import sys
import numpy as np
from mpi4py import MPI


MPI_REGULAR_SEND = 0x000002
MPI_CONTINUE = 0x000004
MPI_STOP = 0x000008


class MPIOrchestrator(object):
    def __init__(self, comm: MPI.Intracomm, rank: int = 0):
        self.comm = comm
        self.rank = rank
        self.__receive_buffer = None

    def scatter_values(self, values):
        values = np.asarray(values)
        assert(values.shape[0] == self.comm.Get_size())
        recvbuf = values.copy()
        self.comm.scatter([[values.shape[1:], values.dtype]]*self.comm.Get_size(), root=self.rank)
        self.comm.Scatter(values, recvbuf, root=self.rank)

    def __iter__(self):
        return self

    def __next__(self):
        self.comm.bcast(MPI_CONTINUE, root=self.rank)
        print("broadcast signal")
        if self.__receive_buffer is None:
            receive_data = []
            for mpi_idx in range(self.comm.Get_size()):
                if mpi_idx == self.rank:
                    continue
                else:
                    receive_data.append(self.comm.irecv(source=mpi_idx, tag=MPI_REGULAR_SEND))
            receive_data = MPI.Request.waitall(receive_data)
            self.__receive_buffer = np.stack(receive_data[:1] + receive_data, axis=0)
        else:
            data_to_send = self.__receive_buffer.copy()[0]
            self.comm.Gather(data_to_send, self.__receive_buffer, root=self.rank)
        self.comm.Barrier()
        return self.__receive_buffer

    def stop_children(self):
        self.comm.bcast(MPI_STOP, root=self.rank)
        return


class MPIChildren(object):
    def __init__(self, comm: MPI.Intracomm, rank: int, master_rank: int = 0):
        self.comm = comm
        self.rank = rank
        self.master_rank = master_rank
        self.__sent_before = False

    def forward(self, x):
        return np.atleast_1d(x)

    def __iter__(self):
        return self

    def __next__(self):
        shape, dtype = self.comm.scatter(None, root=self.master_rank)
        x = np.empty(shape, dtype=dtype)
        self.comm.Scatter(None, x, root=self.master_rank)
        data = self.comm.bcast(None, root=self.master_rank)
        print("broadcast data")
        if data == MPI_STOP:
            self.comm.Barrier()
            raise StopIteration("Received stop from orchestrator")
        else:
            if self.__sent_before:
                self.comm.Gather(self.forward(x), None, root=self.master_rank)
            else:
                self.comm.send(self.forward(x), dest=self.master_rank, tag=MPI_REGULAR_SEND)
                self.__sent_before = True
            self.comm.Barrier()
        return


def create_mpi_child(func):
    class FuncMPIChildren(MPIChildren):
        def forward(self, x):
            return func(x)

    return FuncMPIChildren
