#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_q_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductQLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
        const int K = this->layer_param_.inner_product_q_param().k();
        const int BITS = static_cast<int>(log2((float)K));
        const int TOTAL_BITS = 8;
        const int REST_BITS = TOTAL_BITS - BITS;
        const int M = this->layer_param_.inner_product_q_param().m();

        Dtype* bottom_data = bottom[0]->mutable_cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        Dtype* D = this->blobs_[0]->mutable_cpu_data();
        const Dtype* bias = this->blobs_[1]->cpu_data();
        unsigned char* B_data = (unsigned char*)(this->blobs_[2]->cpu_data());
        int* B = new int[num_output * num_input / M];
        for (int i = 0, total_bit_shift = 0; i < num_output * num_input / M; ++i, total_bit_shift += BITS) {
            int byte_shift = total_bit_shift / TOTAL_BITS;
            int bit_shift = total_bit_shift % TOTAL_BITS;
            int shift = REST_BITS - bit_shift;
            B[i] = static_cast<int>((shift < 0 ? B_data[byte_shift] << -shift | B_data[byte_shift + 1] >> (TOTAL_BITS + shift) :
                                              B_data[byte_shift] >> shift) & (K - 1));
        }
        for (int i = 0; i < batch_size; ++i) {
            caffe_set(num_output, Dtype(0), top_data + i * num_output);
            for (int j = 0; j < num_input / M; ++j) {
                Dtype *S = bottom_data + i * num_input + j * M;
                Dtype *d = D + K * M * j;
                Dtype *output = new Dtype[K];
                caffe_gpu_gemv(CblasTrans, K, M, (Dtype) 1., d, S, (Dtype) 1., output);
		LOG(ERROR) << d[0] << " " << d[1] << " " << d[2] << " " << d[3];
		LOG(ERROR) << S[0] << " " << S[1];
		LOG(ERROR) << output[0] << " " << output[1];
                for (int l = 0; l < num_output; ++l) {
                    top_data[i * num_output + l] += output[B[j * num_output + l]];
                }
                delete[] output;
            }
        }
        delete[] B;
        //caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_input, num_output, 1, (Dtype)1.,
        //                      bias_multiplier_.cpu_data(), bias, (Dtype)1., top_data);
}

template <typename Dtype>
void InnerProductQLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductQLayer);

}
