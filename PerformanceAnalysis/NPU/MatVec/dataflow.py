#
# File: dataflow.py
# Project: NPU SIAM Acceleration
# Author: Victor Jimenez
# Description: Dataflow for NPU Matrix-Vector Multiplication.
#
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.scf import _for as range_

# Define how many columns we can use.
dev = AIEDevice.npu1

# Define the types used. Since dataflow is only used to move data,
# only the size of the type matters, not the underlying type itself --- uint16 ≡ bfloat16.
dtype_in = np.uint16
dtype_out = np.uint32

# Definition of the amount of cores that will be used.
n_aie_columns = 4
n_aie_compute_rows = 4
n_cores = n_aie_compute_rows * n_aie_columns

# Define the matrices' shapes. This has to match the values in the test.cpp file.
# A-Rows: M
# A-Columns: K
#
# B-Rows: K
# B-Columns: N
M = 17408
K = 17056
N = 1

# Define the submatrices' shapes. This submatrix is the batch of data that will be processed by a compute tile at once.
# These values have to match the values in the kernels.cc file.
# A-Rows: m
# A-Columns: k
#
# B-Rows: k
# B-Columns: n
m = 32
k = 32
n = 1

# Dataflow definition.
def dataflow():

    # Assertions for a sanity check.
    assert M % m == 0 and K % k == 0 and N % n == 0, "The matrix must be subtiled properly => M % m == 0 and K % k == 0 and N % n ==0."
    assert M % (m * n_cores) == 0, "Each core must perform the same amount of row iterations => M % (m * n_cores) == 0."

    assert n_aie_columns in range(1, 5), "There are 4 columns with ShimTile in NPU1."
    assert n_aie_compute_rows in range(1, 5), "There are 4 rows with ComputeTIle in NPU1."

    # How many iterations are needed for each submatrix accumulation.
    matrix_A_col_iterations_amount = K // k

    # How many times need A and B matrices to be resent.
    matrix_A_repeat = N // n
    matrix_B_repeat = M // (m * n_cores)

    # Define the device itself.
    @device(dev)
    def device_body():
        # Define the kernels accessed for computation. These have to match the definition in kernels.cc's extern block.
        zero_kernel = external_func("zero", inputs=[np.ndarray[(m * n,), np.dtype[dtype_out]]], link_with="kernels.o")
        mac_kernel = external_func("mac",   inputs=[np.ndarray[(m * k,), np.dtype[dtype_in]],
                                                    np.ndarray[(k * n,), np.dtype[dtype_in]],
                                                    np.ndarray[(m * n,), np.dtype[dtype_out]]], link_with="kernels.o")

        # Tile declarations. This will instantiate all rows for the accessed columns.
        ShimTiles = [tile(col, 0) for col in range(n_aie_columns)]
        MemTiles = [tile(col, 1) for col in range(n_aie_columns)]
        ComputeTiles = [[tile(col, row) for row in range(2, 2 + n_aie_compute_rows)] for col in range(n_aie_columns)]

        # Create list for the ObjectFifos from/to the ShimTiles.
        shim_in_matrix_A = []
        shim_in_matrix_B = []
        shim_out_matrix = []

        # Create list for the ObjectFifos from the MemTiles.
        mem_in_matrix_B = []
        mem_in_matrix_A = [[] for col in range(n_aie_columns)]
        mem_in_matrix_A_offsets = [row * m * k for row in range(n_aie_compute_rows)]

        # Create list for the ObjectFifos to the MemTiles.
        mem_out_matrix = [[] for col in range(n_aie_columns)]
        mem_out_matrix_offsets = [row * m * n for row in range(n_aie_compute_rows)]

        # Set up the actual ObjectFifos.
        # Each column is set up separately, since each NPU's columns' data is independent.
        # Inside the same NPU column, matrix A is divided between NPU rows from the Memtile, while matrix B is broadcasted.
        # The results from each NPU row are joined in the MemTile.
        for col in range(n_aie_columns):
            # Matrix A and B data from ShimTile to MemTile.
            shim_in_matrix_A.append(object_fifo(f"shim_in_matrix_A_{col}", ShimTiles[col], MemTiles[col], 2, np.ndarray[(m * k * n_aie_compute_rows,), np.dtype[np.uint16]],))
            shim_in_matrix_B.append(object_fifo(f"shim_in_matrix_B_{col}", ShimTiles[col], MemTiles[col], 2, np.ndarray[(k * n,), np.dtype[np.uint16]]))

            # Matrix C data from Memtile to ShimTile.
            shim_out_matrix.append(object_fifo(f"shim_out_matrix_{col}", MemTiles[col], ShimTiles[col], 2, np.ndarray[(m * n * n_aie_compute_rows,), np.dtype[np.uint32]]))

            # Matrix B broadcasting from Memtile to all rows in such column.
            mem_in_matrix_B.append(object_fifo(f"mem_in_matrix_B_{col}", MemTiles[col], [ComputeTiles[col][row] for row in range(n_aie_compute_rows)], 2, np.ndarray[(k * n,), np.dtype[np.uint16]]))

            for row in range(n_aie_compute_rows):
                # Matrix A split from Memtile to each row in such column.
                mem_in_matrix_A[col].append(object_fifo(f"mem_in_matrix_A_{col}_{row}", MemTiles[col], ComputeTiles[col][row], 2, np.ndarray[(m * k,), np.dtype[np.uint16]],))

                # Matrix C join from each row in such column to the Memtile.
                mem_out_matrix[col].append(object_fifo(f"mem_out_matrix_{col}_{row}", ComputeTiles[col][row], MemTiles[col], 2, np.ndarray[(m * n,), np.dtype[np.uint32]],
                                           [
                                               (m, 1),
                                               (n, m)
                                           ]))

            # Link:
            # Matrix A ShimTile -> Matrix A MemTile.
            object_fifo_link(shim_in_matrix_A[col], mem_in_matrix_A[col], [], mem_in_matrix_A_offsets)

            # Matrix C MemTile -> Matrix C ShimTile.
            object_fifo_link(mem_out_matrix[col], shim_out_matrix[col], mem_out_matrix_offsets, [])

            # Matrix B ShimTile -> Matrix B MemTile.
            object_fifo_link(shim_in_matrix_B[col], mem_in_matrix_B[col], [], [])

        # Instantiate the Compute Cores. Each compute core will acquire an output submatrix, iterate over all of A's columns and B's rows,
        # and release the output.
        for col in range(n_aie_columns):
            for row in range(n_aie_compute_rows):
                @core(ComputeTiles[col][row])
                def core_body():
                    # An "infinite" loop. This keeps the execution ready.
                    for _ in range_(sys.maxsize):
                        # Acquire an output submatrix and clear it.
                        elem_out = mem_out_matrix[col][row].acquire(ObjectFifoPort.Produce, 1)
                        zero_kernel(elem_out)

                        # For all of A's columns and B's rows, multiply and accumulate.
                        for _ in range_(matrix_A_col_iterations_amount):
                            elem_in_mat_A = mem_in_matrix_A[col][row].acquire(ObjectFifoPort.Consume, 1)
                            elem_in_mat_B = mem_in_matrix_B[col].acquire(ObjectFifoPort.Consume, 1)

                            mac_kernel(elem_in_mat_A, elem_in_mat_B, elem_out)

                            mem_in_matrix_B[col].release(ObjectFifoPort.Consume, 1)
                            mem_in_matrix_A[col][row].release(ObjectFifoPort.Consume, 1)

                        # Release the output submatrix.
                        mem_out_matrix[col][row].release(ObjectFifoPort.Produce, 1)

        # Define the runtime sequence that will communicate with the host.
        @runtime_sequence(
            np.ndarray[(matrix_A_repeat * M * K,), np.dtype[dtype_in]],
            np.ndarray[(matrix_B_repeat * K * N,), np.dtype[dtype_in]],
            np.ndarray[(M * N,), np.dtype[dtype_out]],
        )
        def sequence(A, B, C):
            bd_id = 0

            # Send and receive the data from each NPU column.
            for col in range(n_aie_columns):
                # Send each NPU column a set of A's rows. Matrix A has already been tiled to accommodate
                # the division into different columns.
                npu_dma_memcpy_nd(
                    metadata=shim_in_matrix_A[col],
                    bd_id=bd_id,
                    mem=A,
                    offsets=[0, 0, 0, matrix_A_repeat * (col * M * K) // n_aie_columns],
                    sizes=[1, 1, 1, matrix_A_repeat * M * K // n_aie_columns],
                    strides=[0, 0, 0, 1],
                )
                bd_id += 1

                # Send the entirety of B to each NPU column.
                npu_dma_memcpy_nd(
                    metadata=shim_in_matrix_B[col],
                    bd_id=bd_id,
                    mem=B,
                    offsets=[0, 0, 0, 0],
                    sizes=[1, 1, 1, matrix_B_repeat * K * N],
                    strides=[0, 0, 0, 1],
                )
                bd_id += 1

                # Receive the respective C submatrices from each NPU column.
                npu_dma_memcpy_nd(
                    metadata=shim_out_matrix[col],
                    bd_id=bd_id,
                    mem=C,
                    offsets=[0, 0, 0, col * M * N // n_aie_columns],
                    sizes=[1, 1, 1, M * N // n_aie_columns],
                    strides=[0, 0, 0, 1],
                )
                bd_id += 1

            dma_wait(*shim_out_matrix)

# Produce the dataflow file and verify its correctness.
with mlir_mod_ctx() as ctx:
    dataflow()
    verify = ctx.module.operation.verify()
    if not verify:
        print(verify)
    else:
        print(ctx.module)
