# cuda_for_eNeRF

## Installation
```shell
$ cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-<your cuda version>/bin/nvcc -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch  .
$ cd build
$ make -j16
```

## Check kernel
```shell
$ nsys nvprof ./testbed
```
