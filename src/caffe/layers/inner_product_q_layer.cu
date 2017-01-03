#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_q_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void set_kernel_2(const int num_output, const int quant_index, Dtype* d, int* b, Dtype* output) {
	CUDA_KERNEL_LOOP(index, num_output) {
		output[index] = d[b[quant_index * num_output + index]];
	}
}

template <typename Dtype>
void InnerProductQLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
        const int K = this->layer_param_.inner_product_q_param().k();
        const int BITS = static_cast<int>(log2((float)K));
        const int TOTAL_BITS = 32;
        const int REST_BITS = TOTAL_BITS - BITS;
        const int M = this->layer_param_.inner_product_q_param().m();

        Dtype* bottom_data = bottom[0]->mutable_gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        Dtype* D = this->blobs_[0]->mutable_gpu_data();
        const Dtype* bias = this->blobs_[1]->gpu_data();
        int* B_data = (int*)(this->blobs_[2]->cpu_data());
        int Bsize = num_output * num_input / M * sizeof(int);
        int* B = new int[Bsize];
        for (int i = 0, total_bit_shift = 0; i < num_output * num_input / M; ++i, total_bit_shift += BITS) {
            int byte_shift = total_bit_shift / TOTAL_BITS;
            int bit_shift = total_bit_shift % TOTAL_BITS;
            int shift = REST_BITS - bit_shift;
            B[i] = (int)((shift < 0 ? B_data[byte_shift] << -shift | B_data[byte_shift + 1] >> (TOTAL_BITS + shift) :
                                      B_data[byte_shift] >> shift) & (K - 1));
        }
        int* Bgpu = 0;
        cudaMalloc((void**)&Bgpu, Bsize);
        cudaMemcpy(Bgpu, B, Bsize, cudaMemcpyHostToDevice);

        /*for (int i = 0; i < batch_size; ++i) {
            caffe_gpu_set<Dtype>(num_output, Dtype(0), top_data + i * num_output);
            for (int j = 0; j < num_input / M; ++j) {
                Dtype *S = bottom_data + i * num_input + j * M;
                Dtype *d = D + K * M * j;
                Dtype *output;
                Dtype *output2;
                cudaMalloc((void**)&output, K * sizeof(Dtype));
                cudaMalloc((void**)&output2, num_output * sizeof(Dtype));
                caffe_gpu_gemv<Dtype>(CblasTrans, M, K, (Dtype) 1., d, S, (Dtype) 0., output);
                set_kernel_2<Dtype><<<CAFFE_GET_BLOCKS(num_output), CAFFE_CUDA_NUM_THREADS>>>(num_output, j, output, Bgpu, output2);
                caffe_gpu_add<Dtype>(num_output, top_data + i * num_output, output2, top_data + i * num_output);
                cudaFree(output);
                cudaFree(output2);
            }
        }*/
        cudaFree(Bgpu);
        delete[] B;
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, num_output, 1, (Dtype)1.,
                              bias_multiplier_.gpu_data(), bias, (Dtype)1., top_data);
}

template <typename Dtype>
void InnerProductQLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductQLayer);

}
