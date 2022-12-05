/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerf_render.h
 *  @author Hangkun Xu
 */

#pragma once
#include <filesystem/path.h>
#include <nerf-cuda/common.h>
#include <nerf-cuda/nerf_network.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/random.h>

#include <json/json.hpp>
#include <nerf-cuda/common_device.cuh>

NGP_NAMESPACE_BEGIN

class NerfRender {
 public:
  NerfRender();
  ~NerfRender();

  // load network
  // need to do : load pretrained model !
  void load_nerf_tree(long* index_voxels_coarse_h,
                      float* sigma_voxels_coarse_h,
                      float* voxels_fine_h,
                      uint64_t* cg_s,
                      uint64_t* fg_s);

  // render !
  void render_frame(int w, int h, float theta, float phi, float radius);  // render an image according to camera outer parameters.

  void generate_rays(int w, int h, float focal, float* center,
                     Eigen::Matrix<float, 4, 4> c2w,
                     tcnn::GPUMatrixDynamic<float>& rays_o,
                     tcnn::GPUMatrixDynamic<float>& rays_d);

  void render_rays(int N_rays,
                   tcnn::GPUMatrixDynamic<float>& rgb_fine,
                   tcnn::GPUMatrixDynamic<float>& rays_o,
                   tcnn::GPUMatrixDynamic<float>& rays_d,
                   KeyValue* hashtable,
                   int N_samples, 
                   int N_importance, 
                   float perturb);
  
  void inference(int N_rays, int N_samples_, int N_importance,
                 tcnn::GPUMatrixDynamic<float>& rgb_fine,
                 tcnn::GPUMatrixDynamic<float>& xyz_,
                 tcnn::GPUMatrixDynamic<float>& dir_,
                 tcnn::GPUMemory<float>& weights_coarse,
                 KeyValue* hashtable);
  void load_snapshot(const std::string& filepath_string);

  void set_resolution(const int w, const int h);

  void create_hashtable(float* sigma_voxels_coarse,
                        long* index_voxels_coarse,
                        uint64_t* cg_s,
                        uint64_t* fg_s);

  void insert_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs);  

  void Test(float* sigma_voxels, long* index_voxels_coarse, uint64_t* cg_s, uint64_t* fg_s);  
 private:
  std::vector<float> m_aabb_v;
  // Scene parameters
  float m_bound = 1.;
  float m_scale = 1.;

  // Random Number
  uint32_t m_seed = 42;
  tcnn::pcg32 m_rng;

  // density grid parameter !
  Eigen::Vector3i m_cg_s;
  Eigen::Matrix<int,5,1> m_fg_s;

  tcnn::GPUMemory<long> m_index_voxels_coarse;
  tcnn::GPUMemory<float> m_sigma_voxels_coarse;
  tcnn::GPUMemory<float> m_voxels_fine;
  // CASCADE * H * H * H * size_of(float),
  // index calculation : cascade_level * H * H * H + nx * H * H + ny * H + nz

  // Cuda Stuff
  cudaStream_t m_inference_stream;

  // network variable
  filesystem::path m_network_config_path;
  nlohmann::json m_network_config;
  std::shared_ptr<NerfNetwork<precision_t>> m_nerf_network;

  // Middle variables
  tcnn::GPUMatrixDynamic<float> m_rays_o;
  tcnn::GPUMatrixDynamic<float> m_rays_d;
  tcnn::GPUMatrixDynamic<float> m_rgb_fine;
  KeyValue* m_nerf_voxels_index;

  // hashtable
  KeyValue* mhashtable;
  uint32_t mHashTableCapacity;
};

NGP_NAMESPACE_END
