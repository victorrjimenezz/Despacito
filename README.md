# Despacito: Why Neural Processing Units Can Be Slower Than You Think

This is the source code for Despacito, where we compare the performance of a Neural Processing Unit (NPU) to that of a CPU and AVX-512 extensions with multithreading. We also characterize the performance of the NPU with respect to its architectural design.

## Repository Information

### Directory Structure

```
Despacito
├── include/            # Auxiliary Functions and Classes.
└── PerformanceAnalysis/
    ├── CPU/            # Code for performance analysis of a CPU naive single-threaded computation of matrix multiplication.
    ├── AVX/            # Code for performance analysis of a CPU AVX-512 Multithreaded computation of matrix multiplication.
    └── NPU/            # Code for performance analysis of an NPU computation of matrix multiplication.
        ├── MatMat/     # Code for performance analysis of an NPU computation of matrix-matrix multiplication.
        └── MatVec/     # Code for performance analysis of an NPU computation of matrix-vector multiplication.
```

### System Specifications 

This project was conducted on a 2023 HP Victus 16-s0075nr 16" Laptop with an AMD’s Ryzen 7 7840HS processor.

### NPU Execution

In order to test the NPU code, the user must first compile the NPU image using the Makefile located in the respective _MatMat_ or _MatVec_ folder. Then, the user can compile test.cpp using the _CMakeLists.txt_ file.

## Contact

Victor Jimenez | victor.jimenez@colorado.edu

## Acknowledgments

I would like to thank my advisors, Professor Tamara Silbergleit Lehman and Professor Eric Keller, for their support throughout this research project.
I would also like to thank Professor Alessandro Peri, for his guidance and motivation for this project.
