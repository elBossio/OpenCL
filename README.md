# Task 1 — Linear Algebra on GPU (OpenCL)

## Device info
- Platform name: NVIDIA CUDA  
- Device name: NVIDIA GeForce RTX 2060  

## Results

| Function             | OpenMP Time | OpenCL Total | OpenCL Copy-in | OpenCL Kernel | OpenCL Copy-out | OpenMP Bandwidth | OpenCL Bandwidth |
|---------------------|------------:|-------------:|---------------:|--------------:|----------------:|-----------------:|-----------------:|
| vector-times-vector | 16950 us    | 99689 us     | 74356 us       | 372 us        | 24960 us        | 22.2706 GB/s     | 1014.75 GB/s     |
| matrix-times-vector | 148101 us   | 356257 us    | 352741 us      | 69 us         | 3445 us         | 8.49783 GB/s     | 18239.7 GB/s     |
| matrix-times-matrix | 1884921 us  | 13961 us     | 7211 us        | 136 us        | 6613 us         | 0.0200267 GB/s   | 277.564 GB/s     |


# Task 2 — Reduce & Scan (OpenCL)

## Device info
- Platform name: NVIDIA CUDA  
- Device name: NVIDIA GeForce RTX 2060  

## Results

| Function        | OpenMP Time | OpenCL Total | OpenCL Copy-in | OpenCL Kernel | OpenCL Copy-out | OpenMP Bandwidth | OpenCL Bandwidth |
|----------------|------------:|-------------:|---------------:|--------------:|----------------:|-----------------:|-----------------:|
| reduce         | 13585 us    | 49232 us     | 44901 us       | 4206 us       | 123 us          | 9.26236 GB/s     | 29.9166 GB/s     |
| scan-inclusive | 17094 us    | 188411 us    | 51020 us       | 124279 us     | 13112 us        | 7.36101 GB/s     | 1.01247 GB/s     |
