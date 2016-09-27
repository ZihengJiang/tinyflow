/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api.h
 * \brief C API to tiny flow
 */
#ifndef TINYFLOW_C_API_H_
#define TINYFLOW_C_API_H_

#include <nnvm/c_api.h>

typedef void* TBlobHandle;
typedef void* SessionHandle;
typedef void* RtcHandle;

NNVM_DLL int NNTBlobCreate(float* dptr,
                  const nn_uint* shape,
                  const nn_uint ndim,
                  const nn_uint dtype,
                  const nn_uint dev_mask,
                  TBlobHandle* out);

NNVM_DLL int NNSessionCreate(SessionHandle* handle);

NNVM_DLL int NNSessionClose(SessionHandle handle);

NNVM_DLL int NNSessionRun(SessionHandle handle,
                          SymbolHandle graph,
                          nn_uint num_feed,
                          const SymbolHandle* feed_placeholders,
                          const float** feed_dptr,
                          const nn_uint* feed_dtype,
                          const nn_uint* feed_shape_csr_ptr,
                          const nn_uint* feed_shape_data,
                          nn_uint* num_out,
                          const float*** out_dptr,
                          const nn_uint **out_shape_ndim,
                          const nn_uint ***out_shape_data);

NNVM_DLL int RtcCreate(char* name, nn_uint num_input, nn_uint num_output,
                       char** input_names, char** output_names,
                       TBlobHandle* inputs, TBlobHandle* outputs,
                       char* kernel, RtcHandle *out);

NNVM_DLL int RtcRun(RtcHandle handle, nn_uint num_input, nn_uint num_output,
                    TBlobHandle* inputs, TBlobHandle* outputs,
                    nn_uint gridDimX,
                    nn_uint gridDimY,
                    nn_uint gridDimZ,
                    nn_uint blockDimX,
                    nn_uint blockDimY,
                    nn_uint blockDimZ);

NNVM_DLL int RtcFree(RtcHandle handle);

#endif  // TINYFLOW_C_API_H_
