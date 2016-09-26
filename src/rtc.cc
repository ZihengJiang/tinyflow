/*!
 *  Copyright (c) 2015 by Contributors
 * \file rtc.cc
 * \brief Wrapper for NVRTC
 */
#include <tinyflow/rtc.h>
#include <cuda_runtime.h>
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
    cudaError_enum err;
    CUfunction func;
    int dev_id = 0; // TODO
    if (func_.find(dev_id) != func_.end()) {
        func = func_[dev_id];
    } else {
        CUmodule module;
        CHECK_EQ(err = cuModuleLoadDataEx(&module, ptx_, 0, 0, 0), CUDA_SUCCESS)
            << "CudaError: " << err;
        CHECK_EQ(err = cuModuleGetFunction(&func, module, name_.c_str()), CUDA_SUCCESS)
            << "CudaError: " << err;
        module_[dev_id] = module;
        func_[dev_id] = func;
    }

    cudaStream_t s;
    cudaStreamCreate(&s);
    std::vector<float*> float_args;
    for (auto& i : input) float_args.push_back(static_cast<float*>(i.data));
    for (auto& i : output) float_args.push_back(static_cast<float*>(i.data));
    std::vector<void*> args;
    for (auto& i : float_args) args.push_back(&i);
    CHECK_EQ(err = cuLaunchKernel(func,
                            grid_dim_X, grid_dim_Y, grid_dim_Z,
                            block_dim_X, block_dim_Y, block_dim_Z,
                            0, s,
                            args.data(), 0), CUDA_SUCCESS) << "CudaError: " << err;
    cudaError_t cuerr;
    CHECK_EQ(cuerr = cudaStreamSynchronize(s),
             cudaSuccess) << "CudaError: " << cuerr;
    cudaStreamDestroy(s);
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
    return source;
}

char* Rtc::compile(const std::string& name, const std::string& code) {
    nvrtcProgram prog;
    CHECK_EQ(nvrtcCreateProgram(&prog,
                                code.c_str(),
                                (name+".cu").c_str(),
                                0,
                                NULL,
                                NULL), NVRTC_SUCCESS);
    nvrtcResult compile_res = nvrtcCompileProgram(prog, 0, NULL);
    size_t log_size;
    CHECK_EQ(nvrtcGetProgramLogSize(prog, &log_size), NVRTC_SUCCESS);
    char *log = new char[log_size];
    CHECK_EQ(nvrtcGetProgramLog(prog, log), NVRTC_SUCCESS);
    CHECK_EQ(compile_res, NVRTC_SUCCESS) << log;

    size_t ptx_size;
    CHECK_EQ(nvrtcGetPTXSize(prog, &ptx_size), NVRTC_SUCCESS);
    char *ptx = new char[ptx_size];
    CHECK_EQ(nvrtcGetPTX(prog, ptx), NVRTC_SUCCESS);
    CHECK_EQ(nvrtcDestroyProgram(&prog), NVRTC_SUCCESS);
    return ptx;
}

}  // namespace tinyflow

#endif  // ((TINYFLOW_USE_CUDA) && (TINYFLOW_USE_NVRTC))
