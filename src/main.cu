/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   main.cu
 *  @author Hangkun
 */

#include <cuda.h>
#include <filesystem/path.h>
#include <nerf-cuda/nerf_render.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include <Eigen/Dense>
#include <args/args.hxx>
#include <iostream>
#include <string>
#include <nerf-cuda/npy.hpp>
#include <nerf-cuda/half.hpp>
#include <vector>

using namespace args;
using namespace std;
using namespace ngp;
using namespace tcnn;
using half_float::half;
using namespace half_float::literal;
namespace fs = ::filesystem;

void printDeviceProp(const cudaDeviceProp& prop) {
  cout << "Device Name : " << prop.name << "\n";
  cout << "totalGlobalMem : " << prop.totalGlobalMem << "\n";
  cout << "sharedMemPerBlock " << prop.sharedMemPerBlock << "\n";
  cout << "regsPerBlock : " << prop.regsPerBlock << "\n";
  cout << "warpSize :" << prop.warpSize << "\n";
  cout << "memPitch : " << prop.memPitch << "\n";
  cout << "maxThreadsPerBlock " << prop.maxThreadsPerBlock << "\n";
  cout << "maxThreadsDim[0 - 2] : " << prop.maxThreadsDim[0] << " "
       << prop.maxThreadsDim[1] << " " << prop.maxThreadsDim[2] << "\n";
  cout << "maxGridSize[0 - 2] " << prop.maxGridSize[0] << " "
       << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << "\n";
  cout << "totalConstMem : " << prop.totalConstMem << "\n";
  cout << "major.minor : " << prop.major << "." << prop.minor << "\n";
  cout << "clockRate : " << prop.clockRate << "\n";
  cout << "textureAlignment :" << prop.textureAlignment << "\n";
  cout << "deviceOverlap : " << prop.deviceOverlap << "\n";
  cout << "multiProcessorCount : " << prop.multiProcessorCount << "\n";
}

__global__ void add_one(double* data, const int N = 5) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= N) {
    return;
  }
  data[index] += 1;
}

__global__ void matrix_add_one(MatrixView<double> data, const int M = 5,
                               const int N = 5) {
  const uint32_t encoded_index_x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t encoded_index_y = threadIdx.y + blockIdx.y * blockDim.y;
  if (encoded_index_x > M || encoded_index_y > N) {
    return;
  }
  data(encoded_index_x, encoded_index_y) += 1;
}

int main(int argc, char** argv) {

  cout << "Hello, EfficientNeRF!" << endl;
  cout << "Loading Model......" << endl;
  const std::string sigma_npy_path {"sigma_voxels_coarse_tv_new.npy"};
  const std::string index_npy_path {"index_voxels_coarse_tv_new.npy"};
  const std::string voxels_npy_path {"voxels_fine_tv_new.npy"};
  std::vector<float> sigma_voxels_coarse;
  std::vector<long> index_voxels_coarse;
  std::vector<float> voxels_fine;
  std::vector<uint64_t> coarse_shape {};
  std::vector<uint64_t> fine_shape {};
  std::vector<uint64_t> shape {};
  bool is_fortran;
  npy::LoadArrayFromNumpy(sigma_npy_path, coarse_shape, is_fortran, sigma_voxels_coarse);
  npy::LoadArrayFromNumpy(index_npy_path, shape, is_fortran, index_voxels_coarse);
  npy::LoadArrayFromNumpy(voxels_npy_path, fine_shape, is_fortran, voxels_fine);
  std::cout << "sigma size:" << sigma_voxels_coarse.size() << std::endl;
  std::cout << "index size:" << index_voxels_coarse.size() << std::endl;
  long* index_voxels_coarse_h = new long[index_voxels_coarse.size()];
  float* sigma_voxels_coarse_h = new float[sigma_voxels_coarse.size()];
  float* voxels_fine_h = new float[voxels_fine.size()];
  uint64_t* cg_s = new uint64_t[coarse_shape.size()];
  uint64_t* fg_s = new uint64_t[fine_shape.size()];
  if (!index_voxels_coarse.empty() && !sigma_voxels_coarse.empty() && !voxels_fine.empty())
  {
        memcpy(index_voxels_coarse_h, &index_voxels_coarse[0], index_voxels_coarse.size()*sizeof(long));
        memcpy(sigma_voxels_coarse_h, &sigma_voxels_coarse[0], sigma_voxels_coarse.size()*sizeof(float));
        memcpy(voxels_fine_h, &voxels_fine[0], voxels_fine.size()*sizeof(float));
        memcpy(cg_s, &coarse_shape[0], coarse_shape.size()*sizeof(uint64_t));
        memcpy(fg_s, &fine_shape[0], fine_shape.size()*sizeof(uint64_t));
  }

  NerfRender* render = new NerfRender();
  // render->HashTest(sigma_voxels_coarse_h, index_voxels_coarse_h, coarse_shape);
  render->load_nerf_tree(index_voxels_coarse_h, sigma_voxels_coarse_h, voxels_fine_h, cg_s, fg_s);
  render->Test(sigma_voxels_coarse_h, index_voxels_coarse_h, cg_s, fg_s);
  std::cout << "Hash completed." << std::endl;
  render->set_resolution(1200, 800);
  render->render_frame(1200, 800, 90., -30., 4.);
  // int deviceId;
  // cudaGetDevice(&deviceId);  // `deviceId` now points to the id of the currently
  //                            // active GPU.

  // cudaDeviceProp props;
  // cudaGetDeviceProperties(&props, deviceId);
  // printDeviceProp(props);

}
