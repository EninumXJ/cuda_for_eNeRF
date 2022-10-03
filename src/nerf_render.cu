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

__device__ __constant__ const float C0 = 0.28209479177387814;
__device__ __constant__ const float C1 = 0.4886025119029199;
__device__ __constant__ const float C2[] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};

#define _SIGMOID(x) (1 / (1 + expf(-(x))))
#define _SOFTPLUS_M1(x) (logf(1 + expf((x) - 1)))


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

  int fine_grid_num = m_fg_s[0] * m_fg_s[1] * m_fg_s[2] * m_fg_s[3] * m_fg_s[4];
  std::cout << "coarse_grid_num: " << fine_grid_num << std::endl;

  m_index_voxels_coarse.resize(coarse_grid_num);
  m_index_voxels_coarse.copy_from_host(index_voxels_coarse_h);
  m_sigma_voxels_coarse.resize(coarse_grid_num);
  m_sigma_voxels_coarse.copy_from_host(sigma_voxels_coarse_h);
  m_voxels_fine.resize(fine_grid_num);
  m_voxels_fine.copy_from_host(voxels_fine_h);
  
  float host_data[3] = {0, 0, 0};
  // m_sigma_voxels_coarse.copy_to_host(host_data,3);
  // std::cout << "host_data[1]: " << host_data[1] << std::endl;
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


__device__ __inline__ void precalc_basis(const float* __restrict__ dir, float* __restrict__ out) {
  const float x = dir[0], y = dir[1], z = dir[2];
  const float xx = x * x, yy = y * y, zz = z * z;
  const float xy = x * y, yz = y * z, xz = x * z;
  out[0] = C0;
  out[1] = -C1 * y;
  out[2] = C1 * z;
  out[3] = -C1 * x;
  out[4] = C2[0] * xy;
  out[5] = C2[1] * yz;
  out[6] = C2[2] * (2.0 * zz - xx - yy);
  out[7] = C2[3] * xz;
  out[8] = C2[4] * (xx - yy);
}

__device__ __inline__ void calc_index_(const float* __restrict__ xyz, 
                                        int* __restrict__ ijk_, 
                                        int grid_coarse) {
  float coord_scope = 3.0;
  float xyz_min = -coord_scope;
  float xyz_max = coord_scope;
  float xyz_scope = xyz_max - xyz_min;

  for (int i=0; i<3; i++) {
    ijk_[i] = int((xyz[i] - xyz_min) / xyz_scope * grid_coarse);
    ijk_[i] = ijk_[i] < 0? 0 : ijk_[i];
    ijk_[i] = ijk_[i] > grid_coarse-1? grid_coarse-1 : ijk_[i];
  }
}

__device__ __inline__ void _set_xyz(const float* __restrict__ rays_o, 
                                    const float* __restrict__ dir, 
                                    const float z_val, 
                                    float* __restrict__ out) {

  out[0] = rays_o[0] + dir[0] * z_val;
  out[1] = rays_o[1] + dir[1] * z_val;
  out[2] = rays_o[2] + dir[2] * z_val;
}

__device__ __inline__ float _z_vals(const int index, const int N) {

  float near = 2.0; // val_dataset.near
  float far = 6.0; // val_dataset.far
  return near + (far-near) / (N-1) * index;
}

__global__ void query_fine(MatrixView<float> rgb_final, 
                           float* sigma_voxels_coarse, 
                           MatrixView<float> rays_o, 
                           MatrixView<float> dir_, 
                           float weight_threashold, 
                           long* index_voxels_coarse,
                           float* voxels_fine,
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

    float sigmas = sigma_voxels_coarse[ijk_coarse_[0]*cg_s[1]*cg_s[2] + ijk_coarse_[1]*cg_s[2] + ijk_coarse_[2]];
  
    // line 264 @ efficient-nerf-render-demo/example-app/example-app.cpp
    if (sigmas < weight_threashold) {
      //float sigma_default = -20.0;
      //sigmas[index_fine] = sigma_default;
      //weights[index_fine] = 0.0;
      //rgbs(index_fine, 0) = 1.0;
      //rgbs(index_fine, 1) = 1.0;
      //rgbs(index_fine, 2) = 1.0;
      continue;
    }

    for (int j_=0; j_<N_importance; j_++) {
      int j = i_*N_importance + j_;
      //int index_fine = i*N_samples_fine + index_coarse*N_importance + j;

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

      int coarse_index = index_voxels_coarse[ijk_coarse[0]*cg_s[1]*cg_s[2]+ijk_coarse[1]*cg_s[2]+ijk_coarse[2]];

      // calc_index_fine
  
      int grid_fine = 3;
      int res_fine = grid_coarse * grid_fine;

      int ijk_fine[3];

      ijk_fine[0] = int((xyz_[0] - xyz_min) / xyz_scope * res_fine) % grid_fine;
      ijk_fine[1] = int((xyz_[1] - xyz_min) / xyz_scope * res_fine) % grid_fine;
      ijk_fine[2] = int((xyz_[2] - xyz_min) / xyz_scope * res_fine) % grid_fine;

      // line 195 @ efficient-nerf-render-demo/example-app/example-app.cpp
      float sigma = (float)voxels_fine[coarse_index*fg_s[1]*fg_s[2]*fg_s[3]*fg_s[4] + ijk_fine[0]*fg_s[2]*fg_s[3]*fg_s[4] + ijk_fine[1]*fg_s[3]*fg_s[4] + ijk_fine[2]*fg_s[4]];

      const int deg = 2;
      const int dim_sh = (deg + 1) * (deg + 1);
      float sh[3][dim_sh];

      for (int k=0; k<fg_s[4]-1; k++) {
        sh[k/dim_sh][k%dim_sh] = (float)voxels_fine[coarse_index*fg_s[1]*fg_s[2]*fg_s[3]*fg_s[4] + ijk_fine[0]*fg_s[2]*fg_s[3]*fg_s[4] + ijk_fine[1]*fg_s[3]*fg_s[4] + ijk_fine[2]*fg_s[4] + k+1];
      }

      // eval_sh
      float basis_fn[9];
    
      precalc_basis(vdir, basis_fn);


      float delta_coarse;
      if(j < N_samples_fine-1) 
        delta_coarse = _z_vals(j+1, N_samples_fine) - _z_vals(j, N_samples_fine);
      if(j == N_samples_fine-1)
        delta_coarse = 1e5;
      float att = expf(-delta_coarse * _SOFTPLUS_M1(sigma));
      float weight = light_intensity * (1.f - att);
      weights_sum += weight;
      light_intensity *= att;

      for (int t=0; t<3; t++) {
        float tmp = 0.0;
        for (int k=0; k<9; k++) {
          tmp += basis_fn[k] * sh[t][k];
        }
        rgb_final(i, t) += _SIGMOID(tmp) * weight;
      }

      float stop_thresh = 1e-4;
      if (light_intensity <= stop_thresh) {
        break;
      }
    }

  }
  rgb_final(i, 0) = rgb_final(i, 0) + 1 - weights_sum;
  rgb_final(i, 1) = rgb_final(i, 1) + 1 - weights_sum;
  rgb_final(i, 2) = rgb_final(i, 2) + 1 - weights_sum;
  
}

void NerfRender::inference(int N_rays, int N_samples_, int N_importance,
                           tcnn::GPUMatrixDynamic<float>& rgb_fine,
                           tcnn::GPUMatrixDynamic<float>& rays_o,
                           tcnn::GPUMatrixDynamic<float>& dir_,   
                           tcnn::GPUMemory<float>& sigma_voxels_coarse) {
  std::cout << "inference" << std::endl;
  // TODO
  // line 263-271 & 186-206 @ efficient-nerf-render-demo/example-app/example-app.cpp
  // use cuda to speed up

  float weight_threashold = 1e-5;

  dim3 threadsPerBlock(maxThreadsPerBlock/32, 32);
  dim3 numBlocks_coarse(div_round_up(N_rays, int(threadsPerBlock.x)), div_round_up(N_samples_, int(threadsPerBlock.y)));
  query_fine<<<div_round_up(N_rays, maxThreadsPerBlock), maxThreadsPerBlock>>>(rgb_fine.view(),
                                                    sigma_voxels_coarse.data(), 
                                                    rays_o.view(), 
                                                    dir_.view(), 
                                                    weight_threashold, 
                                                    m_index_voxels_coarse.data(),
                                                    m_voxels_fine.data(),
                                                    m_cg_s, m_fg_s,
                                                    N_importance, N_samples_, N_rays);
  
}

void NerfRender::render_rays(int N_rays,
                             tcnn::GPUMatrixDynamic<float>& rgb_fine,
                             tcnn::GPUMatrixDynamic<float>& rays_o,
                             tcnn::GPUMatrixDynamic<float>& rays_d,
                             int N_samples=128, 
                             int N_importance=5, 
                             float perturb=0.) {
  std::cout << "render_rays" << std::endl;
  int N_samples_fine = N_samples * N_importance;
  inference(N_rays, N_samples_fine, N_importance, rgb_fine, rays_o, rays_d, m_sigma_voxels_coarse);
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
  tlog::info() << c2w;
}

__global__ void get_image(MatrixView<float> rgb_final, const int N, float* rgbs){
  const int i = threadIdx.x + blockIdx.x * blockDim.x;  // N
  if(i >= N){
    return;
  }
  rgbs[i*3] = rgb_final(i, 0);
  rgbs[i*3+1] = rgb_final(i, 1);
  rgbs[i*3+2] = rgb_final(i, 2);
}

void NerfRender::render_frame(int w, int h, float theta, float phi, float radius) {
  auto c2w = pose_spherical(theta, phi, radius);
  float focal = 0.5 * w / std::tan(0.5*0.6911112070083618);
  int N = w * h;  // number of pixels
 
  std::cout << "The bug is  here ?" << std::endl;
  m_rays_o.initialize_constant(0.);
  m_rays_d.initialize_constant(0.);
  generate_rays(w, h, focal, c2w, m_rays_o, m_rays_d);
  
  m_rgb_fine.initialize_constant(1.);
  render_rays(N, m_rgb_fine, m_rays_o, m_rays_d, 128);
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