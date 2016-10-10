/*!
 *  Copyright (c) 2016 by Contributors
 * \file rtc.h
 * \brief Wrapper for NVRTC
 */
#ifndef TINYFLOW_RTC_H_
#define TINYFLOW_RTC_H_
#include <tinyflow/base.h>
#include <nvrtc.h>
#include <cuda.h>

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>

#define CUDA_SAFE_CALL(x)                                               \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS) {                                       \
      const char *msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      dmlc::LogMessageFatal(__FILE__, __LINE__).stream()                \
          << "CudaError: " #x " failed with error: " << msg;            \
    }                                                                   \
  }

#define NVRTC_SAFE_CALL(x)                                              \
  {                                                                     \
    nvrtcResult result = x;                                             \
    if (result != NVRTC_SUCCESS) {                                      \
      dmlc::LogMessageFatal(__FILE__, __LINE__).stream()                \
          << "NvrtcError: " #x " failed with error: "                   \
          << nvrtcGetErrorString(result);                               \
    }                                                                   \
  }

namespace tinyflow {

typedef unsigned index_t;

/*!
 * \brief Runtime compile of cuda kernel code with NVRTC
 */
class Rtc {
 public:
  /*!
   * \brief Build a new kernel.
   *
   * If the same kernel has been compiled before it will be load from
   * cache instead of compile again.
   * \param name name of the kernel function.
   * \param input list of input ndarrays and their name.
   * \param output list of output ndarrays and their name.
   * \param kernel cuda code.
   */
  Rtc(const std::string& name, const std::string& kernel);
  /*!
   * \brief launch a kernel for flat tensor with the engine.
   * \param input list of input ndarray.
   * \param output list of output ndarray.
   * \param num_elements number of elements.
   */
  void Run(std::vector<TBlob> const& input,
           std::vector<TBlob> const& output,
           uint32_t num_elements);
  /*!
   * \brief launch a kernel with the engine.
   * \param input list of input ndarray.
   * \param output list of output ndarray.
   * \param grid_dim_X kernel grid dimensions.
   * \param grid_dim_Y kernel grid dimensions.
   * \param grid_dim_Z kernel grid dimensions.
   * \param block_dim_X kernel block dimensions.
   * \param block_dim_Y kernel block dimensions.
   * \param block_dim_Z kernel block dimensions.
   */
  void Run(std::vector<TBlob> const& input,
           std::vector<TBlob> const& output,
           uint32_t num_elements,
           uint32_t grid_dim_X,
           uint32_t grid_dim_Y,
           uint32_t grid_dim_Z,
           uint32_t block_dim_X,
           uint32_t block_dim_Y,
           uint32_t block_dim_Z);

 private:
  static const char str_type[];
  static std::unordered_map<std::string, char*> kernel_registry;

  std::string name_;
  index_t num_input_, num_output_;
  std::string code_;
  char* ptx_;
  std::unordered_map<int, CUmodule> module_;
  std::unordered_map<int, CUfunction> func_;

//  /*!
//   * \brief add supporting code to kernel.
//   */
//  std::string decorate(const std::string& name,
//                       std::vector<std::pair<std::string, TShape> > const& input,
//                       std::vector<std::pair<std::string, TShape> > const& output,
//                       const std::string kernel);
  /*!
   * \brief compile the kernel with nvrtc.
   */
  char* compile(const std::string& name, const std::string& code);
};

}  // namespace tinyflow

#endif  // TINYFLOW_RTC_H_
