/*!
 *  Copyright (c) 2015 by Contributors
 * \file rtc.cc
 * \brief Wrapper for NVRTC
 */
#include <tinyflow/rtc.h>
#include <cuda_runtime.h>
#include <iostream>
#if true
// #if ((TINYFLOW_USE_CUDA) && (TINYFLOW_USE_NVRTC))

namespace tinyflow {
const char Rtc::str_type[] = "float";
std::unordered_map<std::string, char*> Rtc::kernel_registry;

Rtc::Rtc(const std::string& name,
         std::vector<std::pair<std::string, TBlob> > const& input,
         std::vector<std::pair<std::string, TBlob> > const& output,
         const std::string& kernel) {
  name_ = name;
  num_input_ = input.size();
  num_output_ = output.size();
  code_ = decorate(name, input, output, kernel);
  if (Rtc::kernel_registry.find(code_) != Rtc::kernel_registry.end()) {
      ptx_ = Rtc::kernel_registry[code_];
  } else {
      ptx_ = compile(name, code_);
  }
}

void Rtc::Run(std::vector<TBlob> const& input,
                 std::vector<TBlob> const& output,
                 unsigned int grid_dim_X,
                 unsigned int grid_dim_Y,
                 unsigned int grid_dim_Z,
                 unsigned int block_dim_X,
                 unsigned int block_dim_Y,
                 unsigned int block_dim_Z) {
    CHECK_EQ(num_input_, input.size());
    CHECK_EQ(num_output_, output.size());
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
    for (auto& i : input)  args.push_back(i.data);
    for (auto& i : output) args.push_back(i.data);

    CUDA_SAFE_CALL(cuLaunchKernel(func,
                                  grid_dim_X, grid_dim_Y, grid_dim_Z,
                                  block_dim_X, block_dim_Y, block_dim_Z,
                                  0, NULL, args.data(), 0));
    CUDA_SAFE_CALL(cuCtxSynchronize());
}

std::string Rtc::decorate(const std::string& name,
                         std::vector<std::pair<std::string, TBlob> > const& input,
                         std::vector<std::pair<std::string, TBlob> > const& output,
                         const std::string kernel) {
    std::string source;
    source = source + "\nextern \"C\" __global__ void " + name + "(";
    for (auto &i : input) {
        source = source + "const " + str_type + "* " + i.first + ",";
    }
    for (auto &i : output) {
        source = source + str_type + "* " + i.first + ",";
    }
    source.pop_back();
    source = source + ") {\n";
    for (auto &i : input) {
        source = source + "const int " + i.first + "_ndim = " +
                  std::to_string(i.second.shape.ndim()) + ";\n";
        source = source + "const int " + i.first + "_dims[] = {";
        for (index_t j = 0; j < i.second.shape.ndim(); ++j) {
            source = source + std::to_string(i.second.shape[j]) + ",";
        }
        source.pop_back();
        source = source + "};\n";
    }
    for (auto &i : output) {
        source = source + "const int " + i.first + "_ndim = " +
                  std::to_string(i.second.shape.ndim()) + ";\n";
        source = source + "const int " + i.first + "_dims[] = {";
        for (index_t j = 0; j < i.second.shape.ndim(); ++j) {
            source = source + std::to_string(i.second.shape[j]) + ",";
        }
        source.pop_back();
        source = source + "};\n";
    }
    source = source + kernel + "\n}\n";
    LOG(INFO) << source;
    return source;
}

char* Rtc::compile(const std::string& name, const std::string& code) {
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
    LOG(INFO) << "RTC Compile Successfully";
    return ptx;
}

}  // namespace tinyflow

#endif  // ((TINYFLOW_USE_CUDA) && (TINYFLOW_USE_NVRTC))
