/*!
 *  Copyright (c) 2015 by Contributors
 * \file rtc.cc
 * \brief wrapper for NVRTC
 */
#include <nnvm-rtc/rtc.h>
#include <cuda_runtime.h>
#include <iostream>

namespace nnvm {
namespace rtc {

const char RTC::str_type[] = "float";
std::unordered_map<std::string, char*> RTC::kernel_registry;


RTC::RTC(const std::string& name, const std::string& kernel) {
  name_ = name;
  code_ = kernel;
  if (RTC::kernel_registry.find(code_) != RTC::kernel_registry.end()) {
    ptx_ = RTC::kernel_registry[code_];
  } else {
    ptx_ = compile(name, code_);
  }
}


void RTC::Run(std::vector<void*> const& input,
              std::vector<void*> const& output,
              uint32_t num_elements) {
    const int kBaseThreadBits = 8;
    const int kBaseThreadNum  = 1 << kBaseThreadBits;
    const int kMaxGridNum     = 65535;
    // const int kBaseGridNum    = 1024;

    int num_block = (num_elements + kBaseThreadNum - 1) / kBaseThreadNum;
    if (num_block < kMaxGridNum) {
      return Run(input, output, num_elements, num_block, 1, 1, kBaseThreadNum, 1, 1);
    } else {
      // TODO(ziheng) for large kernel, repeat call kernel
      // int repeat = (num_block + kBaseGridNum - 1) / kBaseGridNum;
    }
}

void RTC::Run(std::vector<void*> const& input,
              std::vector<void*> const& output,
              uint32_t num_elements,
              uint32_t grid_dim_X,
              uint32_t grid_dim_Y,
              uint32_t grid_dim_Z,
              uint32_t block_dim_X,
              uint32_t block_dim_Y,
              uint32_t block_dim_Z) {
  // LOG(INFO) << "RTC: input_size = " << input.size();
  // LOG(INFO) << "RTC: output_size = " << output.size();
  CHECK(output.size());

  CUdevice cuDevice;
  CUcontext context;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));

  CUfunction func;
  // TODO (ziheng) dev_id, stream
  int dev_id = 0;
  if (func_.find(dev_id) != func_.end()) {
      func = func_[dev_id];
  } else {
    CUmodule module;
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx_, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&func, module, name_.c_str()));
    module_[dev_id] = module;
    func_[dev_id] = func;
  }

  std::vector<void*> args;
  for (auto& i : input)  args.push_back(i);
  for (auto& i : output) args.push_back(i);
  args.push_back(&num_elements);

  // LOG(INFO) << "Launch Kernel";
  CUDA_SAFE_CALL(cuLaunchKernel(func,
                                grid_dim_X, grid_dim_Y, grid_dim_Z,
                                block_dim_X, block_dim_Y, block_dim_Z,
                                0, NULL, args.data(), 0));
  CUDA_SAFE_CALL(cuCtxSynchronize());
}


// std::string RTC::decorate(const std::string& name,
//                          std::vector<std::pair<std::string, TShape> > const& input,
//                          std::vector<std::pair<std::string, TShape> > const& output,
//                          const std::string kernel) {
//     std::string source;
//     source = source + "\nextern \"C\" __global__ void " + name + "(";
//     for (auto &i : input) {
//         source = source + "const " + str_type + "* " + i.first + ",";
//     }
//     for (auto &i : output) {
//         source = source + str_type + "* " + i.first + ",";
//     }
//     source.pop_back();
//     source = source + ") {\n";
//     for (auto &i : input) {
//         source = source + "const int " + i.first + "_ndim = " +
//                   std::to_string(i.second.ndim()) + ";\n";
//         source = source + "const int " + i.first + "_dims[] = {";
//         for (index_t j = 0; j < i.second.ndim(); ++j) {
//             source = source + std::to_string(i.second[j]) + ",";
//         }
//         source.pop_back();
//         source = source + "};\n";
//     }
//     for (auto &i : output) {
//         source = source + "const int " + i.first + "_ndim = " +
//                   std::to_string(i.second.ndim()) + ";\n";
//         source = source + "const int " + i.first + "_dims[] = {";
//         for (index_t j = 0; j < i.second.ndim(); ++j) {
//             source = source + std::to_string(i.second[j]) + ",";
//         }
//         source.pop_back();
//         source = source + "};\n";
//     }
//     source = source + kernel + "\n}\n";
//     LOG(INFO) << source;
//     return source;
// }


char* RTC::compile(const std::string& name, const std::string& code) {
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,
                                       code.c_str(),
                                       (name+".cu").c_str(),
                                       0,
                                       NULL,
                                       NULL));
    nvrtcResult compile_res = nvrtcCompileProgram(prog, 0, NULL);
    size_t log_size;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));
    char *log = new char[log_size];
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
    CHECK_EQ(compile_res, NVRTC_SUCCESS) << log;

    size_t ptx_size;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptx_size));
    char *ptx = new char[ptx_size];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    // LOG(INFO) << "RTC Compile Successfully";
    return ptx;
}

}  // namespace rtc
}  // namespace nnvm
