#include <cuda.h>
#include <filesystem/directory.h>
#include <filesystem/path.h>
#include <nerf-cuda/common.h>
#include <nerf-cuda/nerf_network.h>
#include <nerf-cuda/nerf_render.h>
#include <nerf-cuda/render_utils.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image_write.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <json/json.hpp>
#include <nerf-cuda/common_device.cuh>
#include <set>
#include <typeinfo>
#include <vector>

#define maxThreadsPerBlock 512
#define PI acos(-1)
using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

NerfRender::NerfRender() {
  std::cout << "Hello, NerfRender!" << std::endl;
}

NerfRender::~NerfRender() {}

void NerfRender::set_resolution(const int w, const int h){
  int N = w * h;
  m_rays_o = tcnn::GPUMatrixDynamic<float>(N, 3, tcnn::RM);
  m_rays_d = tcnn::GPUMatrixDynamic<float>(N, 3, tcnn::RM);
  m_rgb_fine = tcnn::GPUMatrixDynamic<float>(N, 3, tcnn::RM);
}

void NerfRender::load_nerf_tree(long* index_voxels_coarse_h,
                                float* sigma_voxels_coarse_h,
                                float* voxels_fine_h,
                                uint64_t* cg_s,
                                uint64_t* fg_s) {
  std::cout << "load_nerf_tree" << std::endl;
  m_cg_s << cg_s[0], cg_s[1], cg_s[2];
  m_fg_s << fg_s[0], fg_s[1], fg_s[2], fg_s[3], fg_s[4];
  std::cout << m_cg_s << std::endl;
  std::cout << m_fg_s << std::endl;

  int coarse_grid_num = m_cg_s[0] * m_cg_s[1] * m_cg_s[2];
  std::cout << "coarse_grid_num: " << coarse_grid_num << std::endl;

  long long fine_grid_num = (long long)m_fg_s[0] * m_fg_s[1] * m_fg_s[2] * m_fg_s[3] * m_fg_s[4];
  std::cout << "fine_grid_num: " << fine_grid_num << std::endl;
  // std::cout << "real num: " << m_fg_s[0] * m_fg_s[1] * m_fg_s[2] * m_fg_s[3] * m_fg_s[4] << std::endl;
  // m_index_voxels_coarse.resize(coarse_grid_num);
  // m_index_voxels_coarse.copy_from_host(index_voxels_coarse_h);
  // m_sigma_voxels_coarse.resize(coarse_grid_num);
  // m_sigma_voxels_coarse.copy_from_host(sigma_voxels_coarse_h);
  m_voxels_fine.resize(fine_grid_num);
  m_voxels_fine.copy_from_host(voxels_fine_h);
  
  float host_data[3] = {0, 0, 0};
  // m_sigma_voxels_coarse.copy_to_host(host_data,3);
  // std::cout << "host_data[1]: " << host_data[1] << std::endl;
}


__device__ uint32_t hash(uint32_t k, uint32_t HashTableCapacity)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (HashTableCapacity-1);
}

// Insert the key/values in kvs into the hashtable
__global__ void gpu_hashtable_insert(KeyValue* hashtable, 
                                     float* sigma_voxels,
                                     long* index_voxels_coarse,
                                     uint64_t* fg_s,
                                     float threshold,
                                     uint32_t hashcapacity, 
                                     int numkvs)
{
    unsigned int threadid = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadid < numkvs)
    {
        if(sigma_voxels[threadid] > threshold)
        {
          uint32_t key = threadid;
          uint32_t value = index_voxels_coarse[threadid]*fg_s[1]*fg_s[2]*fg_s[3]*fg_s[4];  // coarse_index
          uint32_t slot = hash(key, hashcapacity);
          // hashtable[slot].value = value;
          while (true)
          {
            uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
            if (prev == kEmpty || prev == key)
            {
                hashtable[slot].value = value;
                break;
            }

            slot = (slot + 1) & (hashcapacity-1);
          }
       }
        else
          return;  
    }
}

__global__ void query_fine(MatrixView<float> rgb_final, 
                           float* sigma_voxels_coarse, 
                           MatrixView<float> rays_o, 
                           MatrixView<float> dir_, 
                           float weight_threashold, 
                           long* index_voxels_coarse,
                           float* voxels_fine,
                           KeyValue* hashtable,
                           Eigen::Vector3i cg_s,
                           Eigen::Matrix<int,5,1> fg_s,
                           int N_importance, int N_samples_fine, int N_rays) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  //const int j = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (i >= N_rays) {
    return;
  }
  
  const int N_samples_coarse = N_samples_fine / N_importance;
  const int grid_coarse = cg_s[0];

  float light_intensity = 1.;
  float weights_sum = 0.;
  for (int i_=0; i_<N_samples_coarse; i_++) {
    
    //int index_fine = i*N_samples_fine + j;
    //int index_coarse = i * N_samples_coarse + i_;

    float xyz_coarse[3];
    int ijk_coarse_[3];
    float vdir[] = {dir_(i, 0), dir_(i, 1), dir_(i, 2)};
    float rays_o_[] = {rays_o(i, 0), rays_o(i, 1), rays_o(i, 2)};
    _set_xyz(rays_o_, vdir, _z_vals(i_, N_samples_coarse), xyz_coarse);
    calc_index_(xyz_coarse, ijk_coarse_, grid_coarse);
    uint32_t hashcapacity = cg_s[0] * cg_s[1] * cg_s[2];
    uint32_t key = ijk_coarse_[0]*cg_s[1]*cg_s[2]+ijk_coarse_[1]*cg_s[2]+ijk_coarse_[2];
    uint32_t slot = hash(key, hashcapacity);
    if(hashtable[slot].key == kEmpty)
        continue;
    
    // eval_sh
    float basis_fn[9];
    precalc_basis(vdir, basis_fn);
    float stop_thresh = 1e-4;
    for (int j_=0; j_<N_importance; j_++) {
      int j = i_*N_importance + j_;
      // calc_index_coarse
      float coord_scope = 3.0;
      float xyz_min = -coord_scope;
      float xyz_max = coord_scope;
      float xyz_scope = xyz_max - xyz_min;

      int ijk_coarse[3];
      float xyz_[3];
      _set_xyz(rays_o_, vdir, _z_vals(j, N_samples_fine), xyz_);
      // query_coarse_index
      calc_index_(xyz_, ijk_coarse, grid_coarse);
      // int coarse_index = index_voxels_coarse[ijk_coarse[0]*cg_s[1]*cg_s[2]+ijk_coarse[1]*cg_s[2]+ijk_coarse[2]];
      key = ijk_coarse[0]*cg_s[1]*cg_s[2]+ijk_coarse[1]*cg_s[2]+ijk_coarse[2];
      slot = hash(key, hashcapacity);
      uint32_t coarse_index;
      // if(hashtable[slot].key == key)
      //     coarse_index = hashtable[slot].value;
      // else
      //     continue;
      while(true)
      {
          if(hashtable[slot].key == key)
          {
            coarse_index = hashtable[slot].value;
            break;
          }
          if(hashtable[slot].key == kEmpty)
          {
            break;
          }
          slot = (slot + 1) & (hashcapacity - 1);
      }
     
      if(hashtable[slot].key == kEmpty)
          continue;
      // int coarse_index = index_voxels_coarse[ijk_coarse[0]*cg_s[1]*cg_s[2]+ijk_coarse[1]*cg_s[2]+ijk_coarse[2]];
      // printf("coarse_index : %d\n", coarse_index);
  

      int grid_fine = 3;
      int res_fine = grid_coarse * grid_fine;

      int ijk_fine[3];

      ijk_fine[0] = int((xyz_[0] - xyz_min) / xyz_scope * res_fine) % grid_fine;
      ijk_fine[1] = int((xyz_[1] - xyz_min) / xyz_scope * res_fine) % grid_fine;
      ijk_fine[2] = int((xyz_[2] - xyz_min) / xyz_scope * res_fine) % grid_fine;
      // printf("ijk_fine[0] : %d\n", ijk_fine[0]);
      // line 195 @ efficient-nerf-render-demo/example-app/example-app.cpp
      // printf("voxels_fine: %f", voxels_fine[120]);
      uint32_t fine_index = coarse_index + ijk_fine[0]*fg_s[2]*fg_s[3]*fg_s[4] + ijk_fine[1]*fg_s[3]*fg_s[4] + ijk_fine[2]*fg_s[4];
      // printf("index of voxels_fine: %lld", fine_index);
      float sigma = (float)voxels_fine[fine_index];
      // printf("sigma is: %f\n", sigma);
      const int deg = 2;
      const int dim_sh = (deg + 1) * (deg + 1);
      float sh[3][dim_sh];

      for (int k=0; k<fg_s[4]-1; k++) {
        sh[k/dim_sh][k%dim_sh] = (float)voxels_fine[fine_index + k + 1];
      }

      // printf("basis: %f", basis_fn[0]);
      float delta_coarse;
      if(j < N_samples_fine-1) 
        delta_coarse = _z_vals(j+1, N_samples_fine) - _z_vals(j, N_samples_fine);
      if(j == N_samples_fine-1)
        delta_coarse = 1e5;
      float att = expf(-delta_coarse * _SOFTPLUS_M1(sigma));
      float weight = light_intensity * (1.f - att);
      weights_sum += weight;
      light_intensity *= att;

      float tmp =0.0;
      tmp = basis_fn[0] * sh[0][0] + basis_fn[1] * sh[0][1] + basis_fn[2] * sh[0][2] +
            basis_fn[3] * sh[0][3] + basis_fn[4] * sh[0][4] + basis_fn[5] * sh[0][5] +
            basis_fn[6] * sh[0][6] + basis_fn[7] * sh[0][7] + basis_fn[8] * sh[0][8];
      rgb_final(i, 0) += _SIGMOID(tmp) * weight;
      tmp = basis_fn[0] * sh[1][0] + basis_fn[1] * sh[1][1] + basis_fn[2] * sh[1][2] +
            basis_fn[3] * sh[1][3] + basis_fn[4] * sh[1][4] + basis_fn[5] * sh[1][5] +
            basis_fn[6] * sh[1][6] + basis_fn[7] * sh[1][7] + basis_fn[8] * sh[1][8];
      rgb_final(i, 1) += _SIGMOID(tmp) * weight;
      tmp = basis_fn[0] * sh[2][0] + basis_fn[1] * sh[2][1] + basis_fn[2] * sh[2][2] +
            basis_fn[3] * sh[2][3] + basis_fn[4] * sh[2][4] + basis_fn[5] * sh[2][5] +
            basis_fn[6] * sh[2][6] + basis_fn[7] * sh[2][7] + basis_fn[8] * sh[2][8];
      rgb_final(i, 2) += _SIGMOID(tmp) * weight;
      if (light_intensity <= stop_thresh) {
        break;
      }
    }
    if(light_intensity <= stop_thresh){
      break;
    }
  }
  rgb_final(i, 0) = rgb_final(i, 0) + 1 - weights_sum;
  rgb_final(i, 1) = rgb_final(i, 1) + 1 - weights_sum;
  rgb_final(i, 2) = rgb_final(i, 2) + 1 - weights_sum;
  // printf("rgb_final: %f\n", rgb_final(i, 0));
}

void NerfRender::inference(int N_rays, int N_samples_, int N_importance,
                           tcnn::GPUMatrixDynamic<float>& rgb_fine,
                           tcnn::GPUMatrixDynamic<float>& rays_o,
                           tcnn::GPUMatrixDynamic<float>& dir_,   
                           tcnn::GPUMemory<float>& sigma_voxels_coarse,
                           KeyValue* hashtable) {
  std::cout << "inference" << std::endl;
  // TODO
  // line 263-271 & 186-206 @ efficient-nerf-render-demo/example-app/example-app.cpp
  // use cuda to speed up

  float weight_threashold = 1e-4;

  dim3 threadsPerBlock(maxThreadsPerBlock/32, 32);
  dim3 numBlocks_coarse(div_round_up(N_rays, int(threadsPerBlock.x)), div_round_up(N_samples_, int(threadsPerBlock.y)));
  query_fine<<<div_round_up(N_rays, maxThreadsPerBlock), maxThreadsPerBlock>>>(rgb_fine.view(),
                                                    sigma_voxels_coarse.data(), 
                                                    rays_o.view(), 
                                                    dir_.view(), 
                                                    weight_threashold, 
                                                    m_index_voxels_coarse.data(),
                                                    m_voxels_fine.data(),
                                                    hashtable,
                                                    m_cg_s, m_fg_s,
                                                    N_importance, N_samples_, N_rays);
  
}

void NerfRender::render_rays(int N_rays,
                             tcnn::GPUMatrixDynamic<float>& rgb_fine,
                             tcnn::GPUMatrixDynamic<float>& rays_o,
                             tcnn::GPUMatrixDynamic<float>& rays_d,
                             KeyValue* hashtable,
                             int N_samples=128, 
                             int N_importance=5, 
                             float perturb=0.) {
  std::cout << "render_rays" << std::endl;
  int N_samples_fine = N_samples * N_importance;
  inference(N_rays, N_samples_fine, N_importance, rgb_fine, rays_o, rays_d, m_sigma_voxels_coarse, hashtable);
}

void NerfRender::generate_rays(int w,
                               int h,
                               float focal,
                               float* center,
                               Eigen::Matrix<float, 4, 4> c2w,
                               tcnn::GPUMatrixDynamic<float>& rays_o,
                               tcnn::GPUMatrixDynamic<float>& rays_d) {
  // TODO
  // line 287-292 @ efficient-nerf-render-demo/example-app/example-app.cpp
  // use cuda to speed up
  int N = w * h;
  float scale = 0.3;
  Eigen::Vector3f offset = Eigen::Vector3f(-0.85, 0, 0);
  auto new_pose = nerf_matrix_to_ngp(c2w, scale, offset);
  set_rays_o<<<div_round_up(N, maxThreadsPerBlock), maxThreadsPerBlock>>>(rays_o.view(), new_pose.block<3, 1>(0, 3), N);
  set_rays_d<<<div_round_up(N, maxThreadsPerBlock), maxThreadsPerBlock>>>(rays_d.view(), new_pose.block<3, 3>(0, 0), focal, center, w, h);
  tlog::info() << c2w;
}

__global__ void get_image(MatrixView<float> rgb_final, int N, float* rgbs){
  const int i = threadIdx.x + blockIdx.x * blockDim.x;  // N
  if(i >= N){
    return;
  }
  rgbs[i*3] = rgb_final(i, 0);
  rgbs[i*3+1] = rgb_final(i, 1);
  rgbs[i*3+2] = rgb_final(i, 2);
  // printf("rgbs: %f\n", rgbs[i*3]);
}

void NerfRender::render_frame(int w, int h, float theta, float phi, float radius) {
  auto c2w = pose_spherical(theta, phi, radius);
  c2w << -0.3164555916546028, -0.08060186108585553, 0.9451768080773424, 3.9187254837252734,
          0.9485892037892081, -0.033047126434636705, 0.31477993882868455, 1.4167708528373393,
          0.005863528577534578, 0.9961983875971426, 0.0869160031797175, 1.0845579674776806,
          0.0, 0.0, 0.0, 1.0;
  float focal = 0.5 * w / std::tan(0.5*1.4032185001629662);
  // float focal = 0.5 * w / std::tan(0.5*0.6911112070083618);
  float fl_x = 3550.114996400941;
  float fl_y = 3554.5152821087413;
  float c_x = 3010.450495927548;
  float c_y = 1996.026944099408;
  float img_w = 6000;
  float img_h = 4000;
  // float focal[2] = {fl_x * w / img_w, fl_y * h / img_h};
  float center[2] = {c_x * w / img_w, c_y * h / img_h};
  int N = w * h;  // number of pixels
 
  m_rays_o.initialize_constant(0.);
  m_rays_d.initialize_constant(0.);
  generate_rays(w, h, focal, center, c2w, m_rays_o, m_rays_d);
  std::cout << "Is it here?" << std::endl;
  m_rgb_fine.initialize_constant(0.);
  render_rays(N, m_rgb_fine, m_rays_o, m_rays_d, mhashtable, 128);

  // TODO
  // line 378-390 @ Nerf-Cuda/src/nerf_render.cu
  // save array as a picture
  float* rgbs_host = new float[N * 3];
  float* rgbs_dev; 
  cudaMalloc((void **)&rgbs_dev, sizeof(float)*N*3);

  get_image<<<div_round_up(N, maxThreadsPerBlock), maxThreadsPerBlock>>>(m_rgb_fine.view(), N, rgbs_dev);
  cudaMemcpy(rgbs_host, rgbs_dev, sizeof(float)*N*3, cudaMemcpyDeviceToHost);

  unsigned char us_image[N*3];

  for (int i = 0; i < N * 3; i++) {
    us_image[i] = (unsigned char) (255.0 * rgbs_host[i]);        
  }
  
  const char* filepath = "test.png";
  std::cout << "write an image" << std::endl;
  stbi_write_png(filepath, w, h, 3, us_image, w*3);
  FILE * fp;
  if((fp = fopen("rgb.txt","wb"))==NULL){
    printf("cant open the file");
    exit(0);
  }
  for(int i = 0; i < N; i++){
    fprintf(fp, "%f ", rgbs_host[i]);
  }
  fclose(fp);

  delete[] rgbs_host, rgbs_dev;
}

void NerfRender::create_hashtable(float* sigma_voxels,
                                  long* index_voxels_coarse,
                                  uint64_t* cg_s,
                                  uint64_t* fg_s){
    
    // allocate memory on GPU
    cudaMalloc(&mhashtable, sizeof(KeyValue) * mHashTableCapacity);
    std::cout << "Hash Size: " << mHashTableCapacity << std::endl;
    // Initialize hash table to empty
    static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
    cudaMemset(mhashtable, 0xff, sizeof(KeyValue) * mHashTableCapacity);
    float* sigma_voxels_dev;
    cudaMalloc(&sigma_voxels_dev, sizeof(float)*cg_s[0]*cg_s[1]*cg_s[2]);
    cudaMemcpy(sigma_voxels_dev, sigma_voxels, sizeof(float)*cg_s[0]*cg_s[1]*cg_s[2], cudaMemcpyHostToDevice);
    long* index_voxels_coarse_dev;
    cudaMalloc(&index_voxels_coarse_dev, sizeof(long)*cg_s[0]*cg_s[1]*cg_s[2]);
    cudaMemcpy(index_voxels_coarse_dev, index_voxels_coarse, sizeof(long)*cg_s[0]*cg_s[1]*cg_s[2], cudaMemcpyHostToDevice);
    uint64_t* fg_s_dev;
    cudaMalloc(&fg_s_dev, sizeof(uint64_t)*5);
    cudaMemcpy(fg_s_dev, fg_s, sizeof(uint64_t)*5, cudaMemcpyHostToDevice);

    float threshold = 1e-4;
    // Have CUDA calculate the thread block size
    // int mingridsize;
    // int threadblocksize;
    // cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);
    // int gridsize = ((uint32_t)mHashTableCapacity + threadblocksize - 1) / threadblocksize;
    int num_kvs = cg_s[0] * cg_s[1] *cg_s[2];
    // gpu_hashtable_insert<<<gridsize, threadblocksize>>>(mhashtable, sigma_voxels_dev, index_voxels_coarse_dev, fg_s_dev, threshold, (uint32_t)mHashTableCapacity, num_kvs);
    gpu_hashtable_insert<<<div_round_up(num_kvs, maxThreadsPerBlock), maxThreadsPerBlock>>>(mhashtable, sigma_voxels_dev, index_voxels_coarse_dev, fg_s_dev, threshold, (uint32_t)mHashTableCapacity, num_kvs);
    
    KeyValue* nerf_hash = new KeyValue[mHashTableCapacity];
    cudaMemcpy(nerf_hash, mhashtable, sizeof(KeyValue)*mHashTableCapacity, cudaMemcpyDeviceToHost);
    std::cout << "nerf_hash[0].key: " << nerf_hash[0].key << std::endl;
    std::cout << "nerf_hash[0].value: " << nerf_hash[0].value << std::endl;
    cudaFree(nerf_hash);
    cudaFree(index_voxels_coarse_dev);
    cudaFree(sigma_voxels_dev);
}

void NerfRender::Test(float* sigma_voxels,
                      long* index_voxels_coarse,
                      uint64_t* cg_s,
                      uint64_t* fg_s)
{
    //mHashTableCapacity = cg_s[0] * cg_s[1] * cg_s[2];
    mHashTableCapacity = fg_s[0] * 50;
    create_hashtable(sigma_voxels, index_voxels_coarse, cg_s, fg_s);
}

NGP_NAMESPACE_END