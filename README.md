# Task 2 — Reduce & Scan (OpenCL)

## Device info
- Platform name: NVIDIA CUDA  
- Device name: NVIDIA GeForce RTX 2060  

## Results

| Function        | OpenMP Time | OpenCL Total | OpenCL Copy-in | OpenCL Kernel | OpenCL Copy-out | OpenMP Bandwidth | OpenCL Bandwidth |
|----------------|------------:|-------------:|---------------:|--------------:|----------------:|-----------------:|-----------------:|
| reduce         | 13585 us    | 49232 us     | 44901 us       | 4206 us       | 123 us          | 9.26236 GB/s     | 29.9166 GB/s     |
| scan-inclusive | 17094 us    | 188411 us    | 51020 us       | 124279 us     | 13112 us        | 7.36101 GB/s     | 1.01247 GB/s     |
