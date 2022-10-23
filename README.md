# cuda_for_eNeRF

## Installation
```shell
$ cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-<your cuda version>/bin/nvcc .
$ cd build
$ make -j16
```

## Test Speed
```shell
$ nsys nvprof ./testbed
```
