#include <tinyflow/rtc.h>

using nnvm::TShape;
using tinyflow::TBlob;
using tinyflow::Rtc;
typedef unsigned uint;

int main() {
  const uint N = 10;
  const uint buffersize = N * sizeof(float);

  float *x_hptr = new float[N];
  for (uint i = 0; i < N; ++i) x_hptr[i] = 1;

  CUdevice cuDevice;
  CUcontext context;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));

  CUdeviceptr x_dptr;
  CUDA_SAFE_CALL(cuMemAlloc(&x_dptr, buffersize));
  CUDA_SAFE_CALL(cuMemcpyHtoD(x_dptr, x_hptr, buffersize));
  TBlob x;
  x.data = reinterpret_cast<void*>(&x_dptr);
  x.shape = TShape({N});
  x.dev_mask = tinyflow::kGPU;
  x.dtype = tinyflow::kFloat32;

  CUdeviceptr y_dptr;
  CUDA_SAFE_CALL(cuMemAlloc(&y_dptr, buffersize));
  TBlob y;
  y.data = reinterpret_cast<void*>(&y_dptr);
  y.shape = TShape({N});
  y.dev_mask = tinyflow::kGPU;
  y.dtype = tinyflow::kFloat32;

  std::vector<std::pair<std::string, TBlob> > input;
  std::vector<std::pair<std::string, TBlob> > output;
  input.push_back(std::pair<std::string, TBlob>("x", x));
  output.push_back(std::pair<std::string, TBlob>("y", y));
  char kernel[] ="                                     \n\
      __shared__ float s_rec[10];                      \n\
      s_rec[threadIdx.x] = x[threadIdx.x];             \n\
      y[threadIdx.x] = expf(s_rec[threadIdx.x]*5.0);";

  Rtc rtc("rtc_test", input, output, kernel);

  std::vector<TBlob> inputs, outputs;
  inputs.push_back(x);
  outputs.push_back(y);

  rtc.Run(inputs, outputs, 1, 1, 1, N, 1, 1);

  float *y_hptr = new float[N];
  CUDA_SAFE_CALL(cuMemcpyDtoH(y_hptr, y_dptr, buffersize));
  for (uint i = 0; i < N; ++i) std::cout << y_hptr[i] << " ";
  std::cout << std::endl;

  return 0;
}
