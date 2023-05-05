#include "cuda_op/linear.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define THREAD_NUM 1024
#define SAMLL_SUM_NUM 32
__global__ void mat_add_impl(float* input1, float* input2,
    float* output,  int step,int length)
    {

        int threadId = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;  
        for(int i=threadId*step;i<step*(threadId+1)&&i<length;i++){
            output[i] = input1[i]+input2[i];
        }
    }
void mat_add(const TensorCUDA& input1,
    const TensorCUDA& input2,
    TensorCUDA& result,
    int block_x, int block_y, int thread_x){
        dim3 threadsPerBlock(thread_x);  
        dim3 numBlocks(block_x,  block_y);
        int step = (input1.get_size()-1)/( block_x*block_y*thread_x)+1;
        mat_add_impl<<<numBlocks,threadsPerBlock>>>(input1.get(), input2.get(), result.get(), step, input1.get_size());
            cudaError_t cudaStatus =  cudaDeviceSynchronize();
    if (cudaStatus !=  cudaSuccess) {
        printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    }

  void mat_add(const TensorCUDA& input1,
    const TensorCUDA& input2,
    TensorCUDA& result,
    int block_x, int block_y, int thread_x,int start,int end){
        dim3 threadsPerBlock(thread_x);  
        dim3 numBlocks(block_x,  block_y);
        int step = (end-start)/( block_x*block_y*thread_x)+1;

        mat_add_impl<<<numBlocks,threadsPerBlock>>>(input1.get()+start, input2.get()+start, result.get()+start, step, end-start);
    cudaError_t cudaStatus =  cudaDeviceSynchronize();
    if (cudaStatus !=  cudaSuccess) {
        printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }
    }

// length 长度(也就是需要做sum的数量)
// stride 实际为线程数
// size 数量 
// stride*size为实际需要合并的数量
__global__ void ReduceSumKernel(float *in, float *out, int length,  int size) {
  int offset = blockIdx.x * blockDim.x;
  int tid = offset + threadIdx.x;

  __shared__ float buffer[THREAD_NUM];
  // 在上一个版本中，共享内存数组中的每一个元素仅仅保存了一个全局内存数组的数据
  // 为了提高归约之前所做计算的比例，可以在归约之前将多个全局内存数组的数据累加到一个共享内存数组的一个元素中
  // 如果一个线程处理相邻的几个数据，会导致全局内存的非合并访问，所以必须让相邻的线程访问相邻的数据
  // 这就意味着同一个线程访问的数据之间有一个跨度，这里使用整个 grid 的大小作为跨度
  float t = 0;
  for(int index=0;index < length; index++){
      t = 0;
      for (int i = 0; i < size; i ++) {
          t += in[index * THREAD_NUM * size + i* THREAD_NUM + tid];
      }
      buffer[threadIdx.x] = t;

      __syncthreads();
      for (int i =THREAD_NUM>>1; i >= 32; i >>= 1) {
        if (threadIdx.x < i) {
          buffer[threadIdx.x] += buffer[threadIdx.x + i];
        }
        __syncthreads();
      }
 

      t = buffer[threadIdx.x];

      for (int i = 16; i >= 1; i >>= 1) {
        t += __shfl_down_sync(0xffffffff, t, i);
      }

      if (threadIdx.x == 0) {
        out[index] = t/(THREAD_NUM * size);
      }
       __syncthreads();
  }
}


void mat_2d_reduce_sum(const TensorCUDA& input1,
                TensorCUDA& result,
                int block_x,  int thread_x){
    ReduceSumKernel<<<block_x,thread_x>>>(input1.get(), result.get(), input1.get_shape()[0],  input1.get_shape()[1]/THREAD_NUM);
        cudaError_t cudaStatus =  cudaDeviceSynchronize();
    if (cudaStatus !=  cudaSuccess) {
        printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }

}

// length 长度(也就是需要做sum的数量)
// stride 实际为线程数
// size 数量 
// stride*size为实际需要合并的数量
__global__ void Reduce3dSumKernel(float *in, float *out, int length,  int size) {
  int offset = blockIdx.x * blockDim.x;
  int tid = offset + threadIdx.x;

  int step_size = (length-1)/(gridDim.x * blockDim.x)+1;
  // 在上一个版本中，共享内存数组中的每一个元素仅仅保存了一个全局内存数组的数据
  // 为了提高归约之前所做计算的比例，可以在归约之前将多个全局内存数组的数据累加到一个共享内存数组的一个元素中
  // 如果一个线程处理相邻的几个数据，会导致全局内存的非合并访问，所以必须让相邻的线程访问相邻的数据
  // 这就意味着同一个线程访问的数据之间有一个跨度，这里使用整个 grid 的大小作为跨度
  float t = 0;
  int index = tid*step_size*size;
  for(int i = tid*step_size;i<(tid+1)*step_size&&i<length;i++){
    t=0;
    for(int j=0;j<size;j++){
      t+= in[index];
      index+=1;
    }
    out[i]=t;
  }
}


void mat_3d_reduce_sum(const TensorCUDA& input1,
                TensorCUDA& result){
    std::vector<int> shape = input1.get_shape();
    int size =shape[shape.size()-1];
    Reduce3dSumKernel<<<1,1024>>>(input1.get(), result.get(), input1.get_size()/size,  size);
        cudaError_t cudaStatus =  cudaDeviceSynchronize();
    if (cudaStatus !=  cudaSuccess) {
        printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
        // 进行错误处理
    }

}


__global__ void batch_mat_divide_mat(float* in ,float* divided, int length, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_size = length/size;
    int step_size = (matrix_size-1)/(blockDim.x *blockDim.y)+1;
    for(int i=tid*step_size;i<(tid+1)*step_size&&i<matrix_size;i++){
        for(int j=0;j<size;j++){
          in[i*size+j]/=divided[i];
        }
    }

}

void batch_mat_divide_mat(TensorCUDA& input,
                TensorCUDA& mean){
  int size = input.get_shape()[2];
  batch_mat_divide_mat<<<1,1024>>>(input.get(), mean.get(), input.get_size(),  size);
    cudaError_t cudaStatus =  cudaDeviceSynchronize();
  if (cudaStatus !=  cudaSuccess) {
      printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
      // 进行错误处理
  }
}

__global__ void  mat_reduce_vector(float* input,  float* mean, float* output,  int step, int mean_shape)
    {

        int threadId = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;


        for(int i=0;i<step;i++){
            // input[threadId*step + i] -=output[(threadId*step + i)%mean_shape];
            output[threadId*step + i] = input[threadId*step + i]-mean[(threadId*step + i)/mean_shape];
            // input[threadId*step + i] = (threadId*step + i)/mean_shape;
        } 
    }

void mat_reduce_vector(TensorCUDA& input,
                TensorCUDA& mean, TensorCUDA& output,
                int block_x,  int thread_x){

            dim3 threadsPerBlock(thread_x);  
        dim3 numBlocks(block_x);
        int step = input.get_size()/( block_x*thread_x);
        int size = input.get_shape()[1]/( block_x*thread_x);
        mat_reduce_vector<<<block_x,thread_x>>>(input.get(), mean.get(),output.get(),  step, input.get_shape()[1]);
        cudaError_t cudaStatus =  cudaDeviceSynchronize();
        if (cudaStatus !=  cudaSuccess) {
            printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
            // 进行错误处理
        }
}

__global__ void  mat_layer_normlize_scale(float* input, float* var, float* scalar, int step, int mean_shape)
    {

        int threadId = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;

        int first_index = 0;
        int second_index = 0;
        int third_index = 0;
        for(int i=0;i<step;i++){
            first_index = threadId*step + i;
            second_index = (threadId*step + i)/mean_shape;
            third_index = (threadId*step + i)%mean_shape;
            input[first_index] =input[first_index]*scalar[third_index]/var[second_index];
            // input[first_index] =input[first_index]/var[second_index];
        } 
    }

void mat_layer_normlize_scale(const TensorCUDA& input1, const TensorCUDA& var, const TensorCUDA& scalar, int block_x,  int thread_x){
        int step = input1.get_size()/( block_x*thread_x);
        int size = input1.get_shape()[1]/( block_x*thread_x);
        mat_layer_normlize_scale<<<block_x,thread_x>>>(input1.get(), var.get(), scalar.get(), step, input1.get_shape()[1]);
        cudaError_t cudaStatus =  cudaDeviceSynchronize();
        if (cudaStatus !=  cudaSuccess) {
            printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
            // 进行错误处理
        }
}

__global__ void ReduceVarSumKernel(float *in, float *out, int length,  int size) {
  int offset = blockIdx.x * blockDim.x;
  int tid = offset + threadIdx.x;

  __shared__ float buffer[THREAD_NUM];
  // 在上一个版本中，共享内存数组中的每一个元素仅仅保存了一个全局内存数组的数据
  // 为了提高归约之前所做计算的比例，可以在归约之前将多个全局内存数组的数据累加到一个共享内存数组的一个元素中
  // 如果一个线程处理相邻的几个数据，会导致全局内存的非合并访问，所以必须让相邻的线程访问相邻的数据
  // 这就意味着同一个线程访问的数据之间有一个跨度，这里使用整个 grid 的大小作为跨度
  float t = 0;
  for(int index=0;index < length; index++){
      t = 0;
      for (int i = 0; i < size; i ++) {
          t += powf(in[index * THREAD_NUM * size + i* THREAD_NUM + tid], 2.0);
      }
      buffer[threadIdx.x] = t;

      __syncthreads();
      for (int i =THREAD_NUM>>1; i >= 32; i >>= 1) {
        if (threadIdx.x < i) {
          buffer[threadIdx.x] += buffer[threadIdx.x + i];
        }
        __syncthreads();
      }
 

      t = buffer[threadIdx.x];

      for (int i = 16; i >= 1; i >>= 1) {
        t += __shfl_down_sync(0xffffffff, t, i);
      }

      if (threadIdx.x == 0) {
        out[index] = sqrtf(t/(THREAD_NUM * size));
      }
       __syncthreads();
  }
}

void mat_2d_reduce_var(TensorCUDA& input1,
                TensorCUDA& result,
                int block_x,  int thread_x){
        int step = input1.get_size()/( block_x*thread_x);
        int size = input1.get_shape()[1]/( block_x*thread_x);
        ReduceVarSumKernel<<<block_x,thread_x>>>(input1.get(), result.get(), input1.get_shape()[0],  input1.get_shape()[1]/THREAD_NUM);
        cudaError_t cudaStatus =  cudaDeviceSynchronize();
        if (cudaStatus !=  cudaSuccess) {
            printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
            // 进行错误处理
        }
}


void matmul(const TensorCUDA& left,
            const TensorCUDA& right,
            TensorCUDA& result,
            cublasHandle_t& handle){

// 创建cublas库句柄
// cublasHandle_t handle;
// cublasStatus_t cublasStatus = cublasCreate(&handle);
// if (cublasStatus !=  CUBLAS_STATUS_SUCCESS) {
//     printf("failed to mat mul create handle: %d\n", cublasStatus);
//     // 进行错误处理
// }
float alpha =1.0;
float beta=0.0;
// 执行矩阵乘法操作
int M = left.get_shape()[0];
int K = left.get_shape()[1];
int N = right.get_shape()[1];

cublasStatus_t cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, right.get(), N,left.get(), K, &beta, result.get(), N);
if (cublasStatus !=  CUBLAS_STATUS_SUCCESS) {
    printf("failed to mat mul: %d\n", cublasStatus);
    // 进行错误处理
}
cudaDeviceSynchronize();

}
bool check_cuda_err() {
    cudaError_t err = cudaGetLastError();
    if(err == cudaSuccess) {
        return true;
    }
    else {
        printf("Cuda Error: %s \n",cudaGetErrorString(err));
        return false;
    }
}

void batch_matmul(const TensorCUDA& left,
            const TensorCUDA& right,
            TensorCUDA& result,
            cublasHandle_t& handle){
              // 创建cublas库句柄

  cublasStatus_t cublasStatus;
  float alpha =0.125;
  float beta=0.0;
  int L=left.get_shape()[1];
  int M=right.get_shape()[1];
  int K =right.get_shape()[2];
  int  N = right.get_shape()[0];

  BatchTensorCUDA left_batch(left);
  BatchTensorCUDA right_batch(right);
  BatchTensorCUDA result_batch(result);
   check_cuda_err();
  cublasStatus = cublasSgemmBatched(handle, CUBLAS_OP_T,CUBLAS_OP_N, 
                                    M, L, K, &alpha, 
                                    (const float**)right_batch.get(), K, 
                                    (const float**)left_batch.get(), K,    
                                    &beta, 
                                    result_batch.get(), M, N);
  // if(L!=1&&M!=1){
  //   printf("LM\n");
  //   left_batch.print(0,0,0);
  //   right_batch.print(0,0,0);
  //   result_batch.print(0,0,0);
  // }
   if (cublasStatus !=  CUBLAS_STATUS_SUCCESS) {
      printf("failed to batch mul: %d\n", cublasStatus);
      // 进行错误处理
  }
  check_cuda_err();
 
  cudaError_t cudaStatus =  cudaDeviceSynchronize();
  check_cuda_err();
  if (cudaStatus !=  cudaSuccess) {
      printf("failed to Synchronize %s\n", cudaGetErrorString(cudaStatus));
      // 进行错误处理
  }

}

void batch_matmul_without_transpose(const TensorCUDA& left,
            const TensorCUDA& right,
            TensorCUDA& result,
            cublasHandle_t& handle){
              // 创建cublas库句柄
  float alpha =1.0;
  float beta=0.0;
  int M = left.get_shape()[1];
  int K = left.get_shape()[2];
  int L = right.get_shape()[2];
  int  N = right.get_shape()[0];

  BatchTensorCUDA left_batch(left);// 32*3*3
  BatchTensorCUDA right_batch(right); // 32*3*64
  BatchTensorCUDA result_batch(result);


  cublasSgemmBatched(handle, CUBLAS_OP_N,CUBLAS_OP_N, 
                                    L, M, K, &alpha, 
                                    (const float**)right_batch.get(), L, 
                                    (const float**)left_batch.get(), K, 
    
                                    &beta, 
                                    result_batch.get(), L, N);
  // cublasSgemmBatched(handle, CUBLAS_OP_N,CUBLAS_OP_N, 
  //                                   L, M, K, &alpha, 
  //                                   (const float**)right_batch.get(), L, 
  //                                   (const float**)left_batch.get(), K, 
    
  //                                   &beta, 
  //                                   result_batch.get(), L, N);
  // cudaDeviceSynchronize();
  cudaError_t cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
    printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
    // 进行错误处理
}

}

// length 长度(也就是需要做sum的数量)
// stride 实际为线程数
// size 数量 
// stride*size为实际需要合并的数量
__global__ void max_index(float *in, int length, float *out,int *index,int start) {
  int offset = blockIdx.x * blockDim.x;
  int tid = offset + threadIdx.x;

  __shared__ float buffer[THREAD_NUM];
  __shared__ int buffer_index[THREAD_NUM];
  // 在上一个版本中，共享内存数组中的每一个元素仅仅保存了一个全局内存数组的数据
  // 为了提高归约之前所做计算的比例，可以在归约之前将多个全局内存数组的数据累加到一个共享内存数组的一个元素中
  // 如果一个线程处理相邻的几个数据，会导致全局内存的非合并访问，所以必须让相邻的线程访问相邻的数据
  // 这就意味着同一个线程访问的数据之间有一个跨度，这里使用整个 grid 的大小作为跨度
  float t = in[0];
  int t_index = 0;
  int size = (length-1)/THREAD_NUM +1;
  for (int i = 0; i < size; i ++) {
    if(i* THREAD_NUM + tid<length && in[i* THREAD_NUM + tid]>t){
      t = in[i* THREAD_NUM + tid];
      t_index = i * THREAD_NUM + tid;
    }
  }
    
  
  buffer[threadIdx.x] = t;
  buffer_index[threadIdx.x] = t_index;

  __syncthreads();
  for (int i =THREAD_NUM>>1; i >= 1; i >>= 1) {
    if (threadIdx.x < i && buffer[threadIdx.x]< buffer[threadIdx.x + i]) {
      buffer[threadIdx.x] =buffer[threadIdx.x + i];
      buffer_index[threadIdx.x] = buffer_index[threadIdx.x + i];
    }
    __syncthreads();
  }


  // t = buffer[threadIdx.x];
  // t_index = buffer_index[threadIdx.x];
  // for (int i = 16; i >= 1; i >>= 1) {
  //   if(__shfl_down_sync(0xffffffff, t, i)>t){
  //       t_index = __shfl_down_sync(0xffffffff, t_index, i);
  //   }else{
  //       __shfl_down_sync(0xffffffff, t_index, i);
  //   }
  // }

  if (threadIdx.x == 0) {
      out[0] = buffer[0] ;
      index[0] = buffer_index[0]+start;
  }
  __syncthreads();
}



void get_last_token(const TensorCUDA& left,
            const TensorCUDA& right,
            float* token_score,
            int* token_id,
            cublasHandle_t& handle,int start){
// left.save("word_embedding");
// right.save("hidden_out");

// 创建cublas库句柄
// cublasHandle_t handle;
// cublasCreate(&handle);
float alpha =1.0;
float beta=0.0;
// 执行矩阵乘法操作
int M = left.get_shape()[0];
int K = left.get_shape()[1];
float* result;


float all_score[M];

cudaError_t cudaStatus = cudaMalloc(&result, M* sizeof(float));
if (cudaStatus != cudaSuccess) {
    printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
    // 进行错误处理
}


cublasStatus_t cublasStatus = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, M, K,
            &alpha, right.get()+(right.get_size()-K), 1,left.get(), K, &beta, result, 1);
if (cublasStatus !=  CUBLAS_STATUS_SUCCESS) {
    printf("failed to mat mul: %d\n", cublasStatus);
    // 进行错误处理
}
cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
    printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
    // 进行错误处理
}
max_index<<<1,1024>>>(result, M, token_score, token_id,start);
cudaStatus = cudaDeviceSynchronize();
if (cudaStatus != cudaSuccess) {
    printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
    // 进行错误处理
}
cudaStatus = cudaMemcpy( all_score, result, M*sizeof(float), cudaMemcpyDeviceToHost);

if (cudaStatus != cudaSuccess) {
    printf("get_last_token cudaMemcpy failed: %s\n", cudaGetErrorString(cudaStatus));
    // 进行错误处理
}
cudaStatus = cudaFree(result);
if (cudaStatus != cudaSuccess) {
    printf("cudaFree matrix score failed: %s\n", cudaGetErrorString(cudaStatus));
    // 进行错误处理
}

}