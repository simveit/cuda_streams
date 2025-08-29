/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * THIS VERSION IS ADAPTED FROM
 * https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

// Ref
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#using-cpu-timers
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#creation-and-destruction-of-streams
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#overlapping-behavior
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/#elapsed-time
// https://www.hpcadmintech.com/wp-content/uploads/2016/03/Carlo_Nardone_presentation.pdf
// https://github.com/NVIDIA-developer-blog/code-samples/blob/master/series/cuda-cpp/overlap-data-transfers/async.cu
// https://doku.lrz.de/files/10333586/10333589/9/1745835159657/NVVP-Streams-UM.pdf
// https://juser.fz-juelich.de/record/903617/files/06-CUDA-Streams-Events.pdf
// https://people.maths.ox.ac.uk/gilesm/cuda/lecs/lec6.pdf
// https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-steering-cuda-compilation

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const *func, char const *file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(char const *file, int const line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

float maxError(const float *a, const std::vector<float> b) {
  float absErr;
  float maxErr = 0;

  for (auto i = 0; i < b.size(); i++) {
    absErr = std::abs(a[i] - b[i]);
    maxErr = std::max(maxErr, absErr);
  }

  return maxErr;
}

__global__ void kernel(float *a, float *b, const int offset) {
  const auto i = offset + blockIdx.x * blockDim.x + threadIdx.x;
  const auto x = static_cast<float>(i);
  const auto s = sinf(x);
  const auto c = cosf(x);

  b[i] = a[i] + sqrtf(s * s + c * c);
}

int main() {

  // Hyperparams
  const auto N = 1 << 25;
  const auto threadsPerBlock = 1024;
  const auto blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  const auto numStreams = 4;
  const auto bytes = N * sizeof(float);
  const auto elementsPerStream = N / numStreams;
  const auto blocksPerGridStream =
      (elementsPerStream + threadsPerBlock - 1) / threadsPerBlock;
  const auto bytesPerStream = elementsPerStream * sizeof(float);
  const auto numWarmup = 100;
  const auto numIterations = 1000;

  std::cout << "N = " << N << "\nblocksPerGrid = " << blocksPerGrid
            << "\nthreadsPerBlock = " << threadsPerBlock << "\n";

  std::cout << "elementsPerStream = " << elementsPerStream
            << "\nblocksPerGrid = " << blocksPerGridStream
            << "\nthreadsPerBlock = " << threadsPerBlock << "\n";

  // Set device
  cudaDeviceProp prop;
  auto deviceId = 3;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::cout << "Device(running on device(" << deviceId << "): " << prop.name
            << "\n";
  CHECK_CUDA_ERROR(cudaSetDevice(deviceId));

  // Memory
  // std::vector<float> a_v;
  // std::vector<float> b_v;
  std::vector<float> r(N, 1);
  float *a;
  float *b;
  float *d_a;
  float *d_b;
  CHECK_CUDA_ERROR(cudaMallocHost((void **)&a, bytes));
  CHECK_CUDA_ERROR(cudaMallocHost((void **)&b, bytes));
  CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
  CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));

  float time_ms;

  cudaEvent_t startEvent, stopEvent;
  cudaStream_t stream[numStreams];
  CHECK_CUDA_ERROR(cudaEventCreate(&startEvent));
  CHECK_CUDA_ERROR(cudaEventCreate(&stopEvent));
  for (auto i = 0; i < numStreams; i++)
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream[i]));
  memset(a, 0, bytes);

  // Sequential
  for (auto i = 0; i < numWarmup; i++) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(b, d_b, bytes, cudaMemcpyDeviceToHost));
  }
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  CHECK_LAST_CUDA_ERROR();

  CHECK_CUDA_ERROR(cudaEventRecord(startEvent, 0));

  for (auto i = 0; i < numIterations; i++) {
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(b, d_b, bytes, cudaMemcpyDeviceToHost));
  }

  CHECK_CUDA_ERROR(cudaEventRecord(stopEvent, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stopEvent));
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, startEvent, stopEvent));
  std::cout << "Time for sequential transfer and execution in ms = "
            << time_ms / numIterations << "\n";
  std::cout << "Max error = " << maxError(b, r) << "\n";

  // Asynchron {copy, kernel, copy}
  for (auto i = 0; i < numWarmup; i++) {
    for (auto s = 0; s < numStreams; s++) {
      auto offset = s * elementsPerStream;
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a + offset, a + offset, bytesPerStream,
                                       cudaMemcpyHostToDevice, stream[s]));
      kernel<<<blocksPerGridStream, threadsPerBlock, 0, stream[s]>>>(d_a, d_b,
                                                                     offset);
      CHECK_CUDA_ERROR(cudaMemcpyAsync(b + offset, d_b + offset, bytesPerStream,
                                       cudaMemcpyDeviceToHost, stream[s]));
    }
  }
  for (auto s = 0; s < numStreams; s++) {
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream[s]));
  }
  CHECK_LAST_CUDA_ERROR();

  CHECK_CUDA_ERROR(cudaEventRecord(startEvent, 0));

  for (auto i = 0; i < numIterations; i++) {
    for (auto s = 0; s < numStreams; s++) {
      auto offset = s * elementsPerStream;
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a + offset, a + offset, bytesPerStream,
                                       cudaMemcpyHostToDevice, stream[s]));
      kernel<<<blocksPerGridStream, threadsPerBlock, 0, stream[s]>>>(d_a, d_b,
                                                                     offset);
      CHECK_CUDA_ERROR(cudaMemcpyAsync(b + offset, d_b + offset, bytesPerStream,
                                       cudaMemcpyDeviceToHost, stream[s]));
    }
  }

  CHECK_CUDA_ERROR(cudaEventRecord(stopEvent, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stopEvent));
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, startEvent, stopEvent));
  std::cout << "Time for asynchronous 1 transfer and execution in ms = "
            << time_ms / numIterations << "\n";
  std::cout << "Max error = " << maxError(b, r) << "\n";

  // Asynchron copy, Asynchron kernel, Asynchron copy

  // Asynchron {copy, kernel, copy}
  for (auto i = 0; i < numWarmup; i++) {
    for (auto s = 0; s < numStreams; s++) {
      auto offset = s * elementsPerStream;
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a + offset, a + offset, bytesPerStream,
                                       cudaMemcpyHostToDevice, stream[s]));
    }
    for (auto s = 0; s < numStreams; s++) {
      auto offset = s * elementsPerStream;
      kernel<<<blocksPerGridStream, threadsPerBlock, 0, stream[s]>>>(d_a, d_b,
                                                                     offset);
    }
    for (auto s = 0; s < numStreams; s++) {
      auto offset = s * elementsPerStream;
      CHECK_CUDA_ERROR(cudaMemcpyAsync(b + offset, d_b + offset, bytesPerStream,
                                       cudaMemcpyDeviceToHost, stream[s]));
    }
  }
  for (auto s = 0; s < numStreams; s++) {
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream[s]));
  }
  CHECK_LAST_CUDA_ERROR();

  CHECK_CUDA_ERROR(cudaEventRecord(startEvent, 0));

  for (auto i = 0; i < numIterations; i++) {
    for (auto s = 0; s < numStreams; s++) {
      auto offset = s * elementsPerStream;
      CHECK_CUDA_ERROR(cudaMemcpyAsync(d_a + offset, a + offset, bytesPerStream,
                                       cudaMemcpyHostToDevice, stream[s]));
    }
    for (auto s = 0; s < numStreams; s++) {
      auto offset = s * elementsPerStream;
      kernel<<<blocksPerGridStream, threadsPerBlock, 0, stream[s]>>>(d_a, d_b,
                                                                     offset);
    }
    for (auto s = 0; s < numStreams; s++) {
      auto offset = s * elementsPerStream;
      CHECK_CUDA_ERROR(cudaMemcpyAsync(b + offset, d_b + offset, bytesPerStream,
                                       cudaMemcpyDeviceToHost, stream[s]));
    }
  }

  CHECK_CUDA_ERROR(cudaEventRecord(stopEvent, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stopEvent));
  CHECK_LAST_CUDA_ERROR();
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, startEvent, stopEvent));
  std::cout << "Time for asynchronous 2 transfer and execution in ms = "
            << time_ms / numIterations << "\n";
  std::cout << "Max error = " << maxError(b, r) << "\n";

  // Cleanup
  CHECK_CUDA_ERROR(cudaEventDestroy(startEvent));
  CHECK_CUDA_ERROR(cudaEventDestroy(stopEvent));
  for (auto i = 0; i < numStreams; i++) {
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream[i]));
  }
  CHECK_CUDA_ERROR(cudaFreeHost(a));
  CHECK_CUDA_ERROR(cudaFreeHost(b));
  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));

  return 0;
}