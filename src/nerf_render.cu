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

#define maxThreadsPerBlock 1024
#define PI acos(-1)

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

Eigen::Matrix<float, 4, 4> trans_t(float t){
  Eigen::Matrix<float, 4, 4> mat;
  mat << 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, t,
         0.0, 0.0, 0.0, 1.0;
  return mat;
}

Eigen::Matrix<float, 4, 4> rot_phi(float phi){
  Eigen::Matrix<float, 4, 4> mat;
  mat << 1.0, 0.0, 0.0, 0.0,
         0.0, cos(phi), -sin(phi), 0.0,
         0.0, sin(phi),  cos(phi), 0.0,
         0.0, 0.0, 0.0, 1.0;
  return mat;
}

Eigen::Matrix<float, 4, 4> rot_theta(float th){
  Eigen::Matrix<float, 4, 4> mat;
  mat << cos(th), 0.0, -sin(th), 0.0,
         0.0, 1.0, 0.0, 0.0,
         sin(th), 0.0,  cos(th), 0.0,
         0.0, 0.0, 0.0, 1.0;
  return mat;
}

NerfRender::NerfRender() {
  std::cout << "Hello, NerfRender!" << std::endl;
}

NerfRender::~NerfRender() {}

void NerfRender::load_nerf_tree(long* index_voxels_coarse_h,
                                float* sigma_voxels_coarse_h,
                                float* voxels_fine_h,
                                const int64_t* cg_s,
                                const int64_t* fg_s) {
  std::cout << "load_nerf_tree" << std::endl;
  m_cg_s << cg_s[0], cg_s[1], cg_s[2];
  m_fg_s << fg_s[0], fg_s[1], fg_s[2], fg_s[3], fg_s[4];
  std::cout << m_cg_s << std::endl;
  std::cout << m_fg_s << std::endl;

  int coarse_grid_num = m_cg_s[0] * m_cg_s[1] * m_cg_s[2];
  std::cout << "coarse_grid_num: " << coarse_grid_num << std::endl;

  int fine_grid_num = m_fg_s[0] * m_fg_s[1] * m_fg_s[2] * m_fg_s[3] * m_fg_s[4];
  std::cout << "coarse_grid_num: " << fine_grid_num << std::endl;

  m_index_voxels_coarse.resize(coarse_grid_num);
  m_index_voxels_coarse.copy_from_host(index_voxels_coarse_h);
  m_sigma_voxels_coarse.resize(coarse_grid_num);
  m_sigma_voxels_coarse.copy_from_host(sigma_voxels_coarse_h);
  m_voxels_fine.resize(fine_grid_num);
  m_voxels_fine.copy_from_host(voxels_fine_h);
  
  float host_data[3] = {0, 0, 0};
  m_sigma_voxels_coarse.copy_to_host(host_data,3);
  std::cout << "host_data[1]: " << host_data[1] << std::endl;
}

Eigen::Matrix<float, 4, 4> pose_spherical(float theta, float phi, float radius) {
  Eigen::Matrix<float, 4, 4> c2w;
  c2w = trans_t(radius);
  c2w = rot_phi(phi / 180. * (float)PI) * c2w;
  c2w = rot_theta(theta / 180. * (float)PI) * c2w;
  Eigen::Matrix<float, 4, 4> temp_mat;
  temp_mat << -1., 0., 0., 0.,
               0., 0., 1., 0.,
               0., 1., 0., 0.,
               0., 0., 0., 1.; 
  c2w = temp_mat * c2w; 
  return c2w;
}

// line 229-249 @ efficient-nerf-render-demo/example-app/example-app.cpp
__global__ void set_z_vals(float* z_vals, const int N) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= N) {
    return;
  }
  float near = 2.0; // val_dataset.near
  float far = 6.0; // val_dataset.far
  z_vals[index] = near + (far-near) / (N-1) * index;
}

// line 251-254 @ efficient-nerf-render-demo/example-app/example-app.cpp
__global__ void set_xyz(MatrixView<float> xyz, 
                        MatrixView<float> rays_o,
                        MatrixView<float> rays_d,
                        float* z_vals, 
                        int N_rays, 
                        int N_samples) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  if (i >= N_rays || j >= N_samples) {
    return;
  }
  xyz(i*N_samples+j, 0) = rays_o(i, 0) + z_vals[j] * rays_d(i, 0);
  xyz(i*N_samples+j, 1) = rays_o(i, 1) + z_vals[j] * rays_d(i, 1);
  xyz(i*N_samples+j, 2) = rays_o(i, 2) + z_vals[j] * rays_d(i, 2);
}

// line 128-137 @ efficient-nerf-render-demo/example-app/example-app.cpp
__global__ void calc_index_coarse(MatrixView<int> ijk_coarse, MatrixView<float> xyz, int grid_coarse, const int N) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= N) {
    return;
  }
  float coord_scope = 3.0;
  float xyz_min = -coord_scope;
  float xyz_max = coord_scope;
  float xyz_scope = xyz_max - xyz_min;

  for (int i=0; i<3; i++) {
    ijk_coarse(index, i) = int((xyz(index, i) - xyz_min) / xyz_scope * grid_coarse);
    ijk_coarse(index, i) = ijk_coarse(index, i) < 0? 0 : ijk_coarse(index, i);
    ijk_coarse(index, i) = ijk_coarse(index, i) > grid_coarse-1? grid_coarse-1 : ijk_coarse(index, i);
  }

}

__global__ void query_coarse_sigma(float* sigmas, float* sigma_voxels_coarse, MatrixView<int> ijk_coarse, Eigen::Vector3i cg_s, const int N) {
  const int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index >= N) {
    return;
  }
  int x = ijk_coarse(index, 0);
  int y = ijk_coarse(index, 1);
  int z = ijk_coarse(index, 2);
  sigmas[index] = sigma_voxels_coarse[x*cg_s[1]*cg_s[2] + y*cg_s[2] + z];
}

__global__ void set_dir(int w, int h, float focal, Eigen::Matrix<float, 4, 4> c2w, MatrixView<float> rays_o, MatrixView<float> rays_d){
  const int i = threadIdx.x + blockIdx.x * blockDim.x;   // h*w
  const int j = threadIdx.y + blockIdx.y * blockDim.y;   // w
  // tcnn::GPUMatrixDynamic<float> dirs(h * w, 3, tcnn::RM);
  if( i >= h*w || j >= w){
    return;
  }
  float dirs[3];
  dirs[0] = (float)((int)(i / w) - w/2) / focal;
  dirs[1] = (float)(j - h/2) / focal;
  dirs[2] = -1.;
  float sum = pow(dirs[0], 2) + pow(dirs[1], 2) + pow(dirs[2], 2);
  dirs[0] /= sum;
  dirs[1] /= sum;
  dirs[2] /= sum;
  // dirs((int)(i / w) * w + j, 0) = (float)((int)(i / w) - w/2) / focal;
  // dirs((int)(i / w) * w + j, 1) = (float)(j - h/2) / focal;
  // dirs((int)(i / w) * w + j, 2) = -1;
  // float sum =  dirs((int)(i / w) * w + j, 0) ^ 2 + dirs((int)(i / w) * w + j, 1) ^ 2 + dirs((int)(i / w) * w + j, 2) ^ 2;
  // dirs((int)(i / w) * w + j, 0) /= sum;
  // dirs((int)(i / w) * w + j, 1) /= sum;
  // dirs((int)(i / w) * w + j, 2) /= sum;

  // get_rays 
  // rays_d((int)(i / w) * w + j, 0) = c2w(0, 0) * dirs((int)(i / w) * w + j, 0) \
  //                                 + c2w(0, 1) * dirs((int)(i / w) * w + j, 1) \
  //                                 + c2w(0, 2) * dirs((int)(i / w) * w + j, 2);
  // rays_d((int)(i / w) * w + j, 1) = c2w(1, 0) * dirs((int)(i / w) * w + j, 0) \
  //                                 + c2w(1, 1) * dirs((int)(i / w) * w + j, 1) \
  //                                 + c2w(1, 2) * dirs((int)(i / w) * w + j, 2);
  // rays_d((int)(i / w) * w + j, 2) = c2w(2, 0) * dirs((int)(i / w) * w + j, 0) \
  //                                 + c2w(2, 1) * dirs((int)(i / w) * w + j, 1) \
  //                                 + c2w(2, 2) * dirs((int)(i / w) * w + j, 2);
  rays_d((int)(i / w) * w + j, 0) = c2w(0,0) * dirs[0] + c2w(0,1) * dirs[1] + c2w(0,2) * dirs[2];
  rays_d((int)(i / w) * w + j, 1) = c2w(1,0) * dirs[0] + c2w(1,1) * dirs[1] + c2w(1,2) * dirs[2];
  rays_d((int)(i / w) * w + j, 2) = c2w(2,0) * dirs[0] + c2w(2,1) * dirs[1] + c2w(2,2) * dirs[2];
  rays_o((int)(i / w) * w + j, 0) = c2w(0, 3);
  rays_o((int)(i / w) * w + j, 1) = c2w(1, 3);
  rays_o((int)(i / w) * w + j, 2) = c2w(2, 3);
}

__global__ void get_alphas(const int N_rays, const int N_samples, float* z_vals, float* sigmas, float* alphas){
  const int i = threadIdx.x + blockIdx.x * blockDim.x;   // N_rays
  const int j = threadIdx.y + blockIdx.y * blockDim.y;   // N_samples
  if(i >= N_rays){
    return;
  }
  if(j >= N_samples){
    return;
  }
  float delta_coarse;
  if(j < N_samples-1) 
    delta_coarse = z_vals[j+1] - z_vals[j];
  if(j == N_samples-1)
    delta_coarse = 1e5;
  float alpha = 1.0 - exp(-delta_coarse * log(1 + std::exp(sigmas[i * N_samples + j])));
  //alpha = 1.0 - alpha + 1e-10;
  alphas[i * N_samples + j] = alpha;
}

__global__ void get_cumprod(const int N_rays, const int N_samples, float* alphas, float* alphas_cumprod){
  const int j = threadIdx.x + blockIdx.x * blockDim.x;   // N_rays
  if(j >= N_rays){
    return;
  }
  alphas_cumprod[j*N_samples+0] = 1.0;
  //float cumprod = 1.0;
  for(int i=1; i<N_samples; i++){
    alphas_cumprod[j*N_samples+i] = alphas_cumprod[j*N_samples+i-1] * (1.0 - alphas[j*N_samples+i-1] + 1e-10);
    //alphas_cumprod[j*N_samples+i] = cumprod;
  }
}

__global__ void get_weights(const int N_rays, const int N_samples, float* alphas, float* alphas_cumprod, float* weights){
  const int i = threadIdx.x + blockIdx.x * blockDim.x;   // N_rays
  const int j = threadIdx.y + blockIdx.y * blockDim.y;   // N_samples
  if(i >= N_rays){
    return;
  }
  if(j >= N_samples){
    return;
  }
  weights[i*N_samples+j] = alphas[i*N_samples+j] * alphas_cumprod[i*N_samples+j];
}

void sigma2weights(tcnn::GPUMemory<float>& weights,    // N_rays*N_samples
                   tcnn::GPUMemory<float>& z_vals,     // N_samples
                   tcnn::GPUMemory<float>& sigmas) {   // N_rays*N_samples
  // TODO
  // line 258-261 @ efficient-nerf-render-demo/example-app/example-app.cpp
  // use cuda to speed up
  int N_samples = z_vals.size();
  int N_rays = weights.size() / N_samples;
  std::cout << "N_samples:" << N_samples << std::endl;
  std::cout << "N_rays:" << N_rays << std::endl;
  tcnn::GPUMemory<float> alphas(N_samples*N_rays);
  tcnn::GPUMemory<float> alphas_cumprod((N_samples + 1)*N_rays);
  
  dim3 threadsPerBlock(maxThreadsPerBlock/32, 32);
  dim3 numBlocks(div_round_up(N_rays, int(threadsPerBlock.x)), div_round_up(N_samples, int(threadsPerBlock.y)));
  get_alphas<<<numBlocks, threadsPerBlock>>>(N_rays, N_samples, z_vals.data(), sigmas.data(), alphas.data());

  std::cout << "get alphas" << std::endl;
  get_cumprod<<<div_round_up(N_rays, maxThreadsPerBlock), maxThreadsPerBlock>>>(N_rays, N_samples, alphas.data(), alphas_cumprod.data());
  get_weights<<<numBlocks, threadsPerBlock>>>(N_rays, N_samples, alphas.data(), alphas_cumprod.data(), weights.data());
  std::cout << "sigma2weights" << std::endl;
}

__global__ void sum_rgbs(MatrixView<float> rgb_final, MatrixView<float> rgbs, float* weights, int N_samples, const int N) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) {
    return;
  }
  float weights_sum = 0;
  rgb_final(i, 0) = 0;
  rgb_final(i, 1) = 0;
  rgb_final(i, 2) = 0;
  for (int j=0; j<N_samples; j++) {
    weights_sum += weights[i*N_samples+j];
    rgb_final(i, 0) += weights[i*N_samples+j] * rgbs(i*N_samples+j, 0);
    rgb_final(i, 1) += weights[i*N_samples+j] * rgbs(i*N_samples+j, 1);
    rgb_final(i, 2) += weights[i*N_samples+j] * rgbs(i*N_samples+j, 2);
  }
  rgb_final(i, 0) = rgb_final(i, 0) + 1 - weights_sum;
  rgb_final(i, 1) = rgb_final(i, 1) + 1 - weights_sum;
  rgb_final(i, 2) = rgb_final(i, 2) + 1 - weights_sum;
}

__global__ void get_weight_threasholds(float* weight_threasholds, float* weights, float weight_threashold, int N_samples, const int N) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= N) {
    return;
  }
  float max_item = 0;
  for (int j=0; j<N_samples; j++) {
    if (max_item < weights[i*N_samples+j]) {
      max_item = weights[i*N_samples+j];
    }
  }
  if (max_item < weight_threashold) {
    weight_threasholds[i] = max_item;
  }
  else {
    weight_threasholds[i] = weight_threashold;
  }
}

__global__ void query_fine(MatrixView<float> rgbs, 
                           float* sigmas,
                           float* weights_coarse, 
                           MatrixView<float> xyz_, 
                           MatrixView<float> dir_, 
                           float weight_threashold, 
                           long* index_voxels_coarse,
                           float* voxels_fine,
                           Eigen::Vector3i cg_s,
                           Eigen::Matrix<int,5,1> fg_s,
                           int N_importance, int N_samples_fine, int N_rays) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  //const int N_samples_coarse = N_samples_fine / N_importance;
  const int index_fine = i*N_samples_fine + j;
  const int index_coarse = index_fine / N_importance;
  if (i >= N_rays || j >= N_samples_fine) {
    return;
  }
  // line 264 @ efficient-nerf-render-demo/example-app/example-app.cpp
  if (weights_coarse[index_coarse] < weight_threashold) {
    float sigma_default = -20.0;
    sigmas[index_fine] = sigma_default;
    rgbs(index_fine, 0) = 1.0;
    rgbs(index_fine, 1) = 1.0;
    rgbs(index_fine, 2) = 1.0;
    return;
  }

  // calc_index_coarse
  
  int grid_coarse = cg_s[0];
  float coord_scope = 3.0;
  float xyz_min = -coord_scope;
  float xyz_max = coord_scope;
  float xyz_scope = xyz_max - xyz_min;

  int ijk_coarse[3];

  // query_coarse_index
  for (int i=0; i<3; i++) {
    ijk_coarse[i] = int((xyz_(index_fine, i) - xyz_min) / xyz_scope * grid_coarse);
    ijk_coarse[i] = ijk_coarse[i] < 0? 0 : ijk_coarse[i];
    ijk_coarse[i] = ijk_coarse[i] > grid_coarse-1? grid_coarse-1 : ijk_coarse[i];
  }
  int coarse_index = index_voxels_coarse[ijk_coarse[0]*cg_s[1]*cg_s[2]+ijk_coarse[1]*cg_s[2]+ijk_coarse[2]];

  // calc_index_fine
  
  int grid_fine = 3;
  int res_fine = grid_coarse * grid_fine;

  int ijk_fine[3];

  ijk_fine[0] = int((xyz_(index_fine, 0) - xyz_min) / xyz_scope * res_fine) % grid_fine;
  ijk_fine[1] = int((xyz_(index_fine, 1) - xyz_min) / xyz_scope * res_fine) % grid_fine;
  ijk_fine[2] = int((xyz_(index_fine, 2) - xyz_min) / xyz_scope * res_fine) % grid_fine;

  // line 195 @ efficient-nerf-render-demo/example-app/example-app.cpp
  sigmas[index_fine] = voxels_fine[coarse_index*fg_s[1]*fg_s[2]*fg_s[3]*fg_s[4] + ijk_fine[0]*fg_s[2]*fg_s[3]*fg_s[4] + ijk_fine[1]*fg_s[3]*fg_s[4] + ijk_fine[2]*fg_s[4]];

  const int deg = 2;
  const int dim_sh = (deg + 1) * (deg + 1);
  float sh[3][dim_sh];

  for (int k=0; k<fg_s[4]-1; k++) {
    sh[k/dim_sh][k%dim_sh] = voxels_fine[coarse_index*fg_s[1]*fg_s[2]*fg_s[3]*fg_s[4] + ijk_fine[0]*fg_s[2]*fg_s[3]*fg_s[4] + ijk_fine[1]*fg_s[3]*fg_s[4] + ijk_fine[2]*fg_s[4] + k+1];
  }

  // eval_sh
  float C0 = 0.28209479177387814;
  float C1 = 0.4886025119029199;
  float C2[5] = {1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396};

  float x = dir_(i, 0);
  float y = dir_(i, 1);
  float z = dir_(i, 2);

  float xx = x * x;
  float yy = y * y;
  float zz = z * z;
  float xy = x * y;
  float yz = y * z;
  float xz = x * z;
  
  for (int k=0; k<3; k++) {
    rgbs(index_fine, k) = C0 * sh[k][0];
    if (deg > 0) {
      rgbs(index_fine, k) = (rgbs(index_fine, k) -    \
                             C1 * y * sh[k][1] +      \
                             C1 * z * sh[k][2] -      \
                             C1 * x * sh[k][3]);
      if (deg > 1) {
        rgbs(index_fine, k) = (rgbs(index_fine, k) +                      \
                               C2[0] * xy * sh[k][4] +                    \
                               C2[1] * yz * sh[k][5] +                    \
                               C2[2] * (2.0 * zz - xx - yy) * sh[k][6] +  \
                               C2[3] * xz * sh[k][7] +                    \
                               C2[4] * (xx - yy) * sh[k][8]);
        }
    }
    // line 199 @ efficient-nerf-render-demo/example-app/example-app.cpp
    rgbs(index_fine, k) = 1 / (1 + exp(-rgbs(index_fine, k)));
    //rgbs(index_fine, k) = sh[k][0];
  }

}

void NerfRender::inference(int N_rays, int N_samples_, int N_importance,
                           tcnn::GPUMatrixDynamic<float>& rgb_fine,
                           tcnn::GPUMatrixDynamic<float>& xyz_,
                           tcnn::GPUMatrixDynamic<float>& dir_,
                           tcnn::GPUMemory<float>& z_vals,    
                           tcnn::GPUMemory<float>& weights_coarse) {
  std::cout << "inference" << std::endl;
  // TODO
  // line 263-271 & 186-206 @ efficient-nerf-render-demo/example-app/example-app.cpp
  // use cuda to speed up

  float weight_threashold = 1e-5;
  //tcnn::GPUMemory<float> weight_threasholds(N_rays);
  //get_weight_threasholds<<<div_round_up(N_rays, maxThreadsPerBlock), maxThreadsPerBlock>>> (weight_threasholds.data(), weights_coarse.data(), weight_threashold, N_samples_, N_rays);

  tcnn::GPUMatrixDynamic<float> rgbs(N_rays*N_samples_, 3, tcnn::RM);
  tcnn::GPUMemory<float> sigmas(N_rays*N_samples_);
  dim3 threadsPerBlock(maxThreadsPerBlock/32, 32);
  dim3 numBlocks_coarse(div_round_up(N_rays, int(threadsPerBlock.x)), div_round_up(N_samples_, int(threadsPerBlock.y)));
  query_fine<<<numBlocks_coarse, threadsPerBlock>>>(rgbs.view(), 
                                                    sigmas.data(), 
                                                    weights_coarse.data(), 
                                                    xyz_.view(), 
                                                    dir_.view(), 
                                                    weight_threashold, 
                                                    m_index_voxels_coarse.data(),
                                                    m_voxels_fine.data(),
                                                    m_cg_s, m_fg_s,
                                                    N_importance, N_samples_, N_rays);
  
  tcnn::GPUMemory<float> weights(N_rays*N_samples_);
  sigma2weights(weights, z_vals, sigmas);
  sum_rgbs<<<div_round_up(N_rays, maxThreadsPerBlock), maxThreadsPerBlock>>> (rgb_fine.view(), rgbs.view(), weights.data(), N_samples_, N_rays);

}

void NerfRender::render_rays(int N_rays,
                             tcnn::GPUMatrixDynamic<float>& rgb_fine,
                             tcnn::GPUMatrixDynamic<float>& rays_o,
                             tcnn::GPUMatrixDynamic<float>& rays_d,
                             int N_samples=128, 
                             int N_importance=5, 
                             float perturb=0.) {
  std::cout << "render_rays" << std::endl;
  int N_samples_coarse = N_samples;
  tcnn::GPUMemory<float> z_vals_coarse(N_samples_coarse);
  set_z_vals<<<div_round_up(N_samples_coarse, maxThreadsPerBlock), maxThreadsPerBlock>>> (z_vals_coarse.data(), N_samples_coarse);

  int N_samples_fine = N_samples * N_importance;
  tcnn::GPUMemory<float> z_vals_fine(N_samples_fine);
  set_z_vals<<<div_round_up(N_samples_fine, maxThreadsPerBlock), maxThreadsPerBlock>>> (z_vals_fine.data(), N_samples_fine);

  tcnn::GPUMatrixDynamic<float> xyz_coarse(N_rays*N_samples_coarse, 3, tcnn::RM);
  dim3 threadsPerBlock(maxThreadsPerBlock/32, 32);
  dim3 numBlocks_coarse(div_round_up(N_rays, int(threadsPerBlock.x)), div_round_up(N_samples_coarse, int(threadsPerBlock.y)));
  set_xyz<<<numBlocks_coarse, threadsPerBlock>>>(xyz_coarse.view(), rays_o.view(), rays_d.view(), z_vals_coarse.data(), N_rays, N_samples_coarse);


  tcnn::GPUMatrixDynamic<float> xyz_fine(N_rays*N_samples_fine, 3, tcnn::RM);
  dim3 numBlocks_fine(div_round_up(N_rays, int(threadsPerBlock.x)), div_round_up(N_samples_fine, int(threadsPerBlock.y)));
  set_xyz<<<numBlocks_fine, threadsPerBlock>>>(xyz_fine.view(), rays_o.view(), rays_d.view(), z_vals_fine.data(), N_rays, N_samples_fine);

  // line 155-161 @ efficient-nerf-render-demo/example-app/example-app.cpp
  tcnn::GPUMatrixDynamic<int> ijk_coarse(N_rays*N_samples_coarse, 3, tcnn::RM);
  calc_index_coarse<<<div_round_up(N_rays*N_samples_coarse, maxThreadsPerBlock), maxThreadsPerBlock>>> (ijk_coarse.view(), xyz_coarse.view(), m_cg_s[0], N_rays*N_samples_coarse);

  tcnn::GPUMemory<float> sigmas(N_rays*N_samples_coarse);
  query_coarse_sigma<<<div_round_up(N_rays*N_samples_coarse, maxThreadsPerBlock), maxThreadsPerBlock>>> (sigmas.data(), m_sigma_voxels_coarse.data(), ijk_coarse.view(), m_cg_s, N_rays*N_samples_coarse);

  /*
  int N = N_rays*N_samples_coarse;
  float* host_data = new float[N];
  sigmas.copy_to_host(host_data);
  for (int i=0; i<50; i++) {
    std::cout << host_data[i] << std::endl;
  }
  */
  
  /*
  int N = N_rays*N_samples_coarse;
  float* host_data = new float[N * 3];
  tcnn::MatrixView<float> view = sigmas.view();
  cudaMemcpy(host_data, &view(0,0), N*3 * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i=0; i<50; i++) {
    for (int j=0; j<3; j++) {
      std::cout << host_data[i*3+j] << "\t";
    }
    std::cout << std::endl;
  }
  */

  // line 261 @ efficient-nerf-render-demo/example-app/example-app.cpp
  tcnn::GPUMemory<float> weights_coarse(N_rays*N_samples_coarse);
  sigma2weights(weights_coarse, z_vals_coarse, sigmas);

  inference(N_rays, N_samples_fine, N_importance, rgb_fine, xyz_fine, rays_d, z_vals_fine, weights_coarse);

  
  //float host_data[N_samples_fine];
  //z_vals_fine.copy_to_host(host_data);
  //std::cout << host_data[0] << " " << host_data[1] << " " << host_data[2] << std::endl;
}

void NerfRender::generate_rays(int w,
                               int h,
                               float focal,
                               Eigen::Matrix<float, 4, 4> c2w,
                               tcnn::GPUMatrixDynamic<float>& rays_o,
                               tcnn::GPUMatrixDynamic<float>& rays_d) {
  // TODO
  // line 287-292 @ efficient-nerf-render-demo/example-app/example-app.cpp
  // use cuda to speed up
  int N = w * h;
  set_rays_o<<<div_round_up(N, maxThreadsPerBlock), maxThreadsPerBlock>>>(rays_o.view(), c2w.block<3, 1>(0, 3), N);
  set_rays_d<<<div_round_up(N, maxThreadsPerBlock), maxThreadsPerBlock>>>(rays_d.view(), c2w.block<3, 3>(0, 0), focal, w, h);
  //dim3 threadsPerBlock(maxThreadsPerBlock/32, 32);
  //dim3 numBlocks(div_round_up(w * h, int(threadsPerBlock.x)), div_round_up(w, int(threadsPerBlock.y)));
  //set_dir<<<numBlocks, threadsPerBlock>>>(w, h, focal, c2w, rays_o.view(), rays_d.view());
  tlog::info() << c2w;
}

__global__ void get_image(MatrixView<float> rgb_final, const int N, float* rgbs){
  const int i = threadIdx.x + blockIdx.x * blockDim.x;  // N
  if(i >= N){
    return;
  }
  rgbs[i * 3] = rgb_final(i, 0);
  rgbs[i * 3 + 1] = rgb_final(i, 1);
  rgbs[i * 3 + 2] = rgb_final(i, 2);
}

void NerfRender::render_frame(int w, int h, float theta, float phi, float radius) {
  auto c2w = pose_spherical(theta, phi, radius);
  float focal = 0.5 * w / std::tan(0.5*0.6911112070083618);
  int N = w * h;  // number of pixels
  // initial points corresponding to pixels, in world coordination
  tcnn::GPUMatrixDynamic<float> rays_o(N, 3, tcnn::RM);
  // direction corresponding to pixels,in world coordination
  tcnn::GPUMatrixDynamic<float> rays_d(N, 3, tcnn::RM);

  generate_rays(w, h, focal, c2w, rays_o, rays_d);
  

  tcnn::GPUMatrixDynamic<float> rgb_fine(N, 3, tcnn::RM);
  render_rays(N, rgb_fine, rays_o, rays_d, 128);
  // TODO
  // line 378-390 @ Nerf-Cuda/src/nerf_render.cu
  // save array as a picture
  
  float* rgbs_host = new float[N * 3];
  float* rgbs_dev; 
  cudaMalloc((void **)&rgbs_dev, sizeof(float) * N * 3);
  //cudaMemcpy(rgbs_dev, rgbs_host, sizeof(unsigned int) * N * 3, cudaMemcpyHostToDevice);

  get_image<<<div_round_up(N, maxThreadsPerBlock), maxThreadsPerBlock>>>(rgb_fine.view(), N, rgbs_dev);
  cudaMemcpy(rgbs_host, rgbs_dev, sizeof(unsigned int)*N*3, cudaMemcpyDeviceToHost);

  unsigned char us_image[N*3];

  for (int i = 0; i < N * 3; i++) {
    us_image[i] = (unsigned char) (255.0 * rgbs_host[i]);        
  }
  
  const char* filepath = "test.png";
  stbi_write_png(filepath, h, w, 3, us_image, w*3);
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

NGP_NAMESPACE_END