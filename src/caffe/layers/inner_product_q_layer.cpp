#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_q_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template <typename Dtype>
    void InnerProductQLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
        const int K = this->layer_param_.inner_product_q_param().k();
        const int M = this->layer_param_.inner_product_q_param().m();
        const int BITS = (int)log2(K);
        const int TOTAL_BITS = 8;
        num_output = this->layer_param_.inner_product_q_param().num_output();
        const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.inner_product_q_param().axis());
        num_input = bottom[0]->count(axis);
        // Check if we need to set up the weights
        if (this->blobs_.size() > 0) {
            LOG(INFO) << "Skipping parameter initialization";
        }
        else {
            this->blobs_.resize(3);
            vector<int> d_shape(2);
            d_shape[0] = num_input;
            d_shape[1] = K;
            this->blobs_[0].reset(new Blob<Dtype>(d_shape));
            vector<int> bias_shape(1, num_output);
            this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
            int b_shape_size = num_output * num_input / M;
            vector<int> b_shape(1, BITS * b_shape_size / (TOTAL_BITS * sizeof(Dtype)) + BITS * b_shape_size % (TOTAL_BITS * sizeof(Dtype)));
            this->blobs_[2].reset(new Blob<Dtype>(b_shape));
        }
        this->param_propagate_down_.resize(this->blobs_.size(), true);
    }

    template <typename Dtype>
    void InnerProductQLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
        // Figure out the dimensions
        const int axis = bottom[0]->CanonicalAxisIndex(
                this->layer_param_.inner_product_q_param().axis());
        const int new_K = bottom[0]->count(axis);
        CHECK_EQ(num_input, new_K)
            << "Input size incompatible with inner product parameters.";
        batch_size = bottom[0]->count(0, axis);
        vector<int> top_shape = bottom[0]->shape();
        top_shape.resize(2);
        top_shape[axis] = num_output;
        top[0]->Reshape(top_shape);

        vector<int> bias_shape(1, num_input);
        bias_multiplier_.Reshape(bias_shape);
        caffe_set(num_input, Dtype(1), bias_multiplier_.mutable_cpu_data());
    }

    template <typename Dtype>
    void InnerProductQLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
        // number of columns in the reduced weights matrix
        const int K = this->layer_param_.inner_product_q_param().k();
        // bits per number in B hash
        const int BITS = log2(K);
        const int TOTAL_BITS = 32;
        const int REST_BITS = TOTAL_BITS - BITS;
        // number of lines in the slices of the source matrix
        const int M = this->layer_param_.inner_product_q_param().m();

        // input image batch
        const Dtype* bottom_data = bottom[0]->cpu_data();
        // output image batch
        Dtype* top_data = top[0]->mutable_cpu_data();
        // reduced weights matrix
        const Dtype* D = this->blobs_[0]->cpu_data();
        // free member weights
        const Dtype* bias = this->blobs_[1]->cpu_data();
        // hash with indexes of D columns, each number uses log2(K) bits
        int* B_hash = (int*)(this->blobs_[2]->cpu_data());
        // D columns indexes unpacked from hash
        unsigned char* B = new unsigned char[num_output * num_input / M];
        for (int i = 0, total_bit_shift = 0; i < num_output * num_input / M; ++i, total_bit_shift += BITS) {
            int byte_shift = total_bit_shift / TOTAL_BITS;
            int bit_shift = total_bit_shift % TOTAL_BITS;
            int shift = REST_BITS - bit_shift;
            B[i] = (unsigned char)((shift < 0 ? B_hash[byte_shift] << -shift | B_hash[byte_shift + 1] >> (TOTAL_BITS + shift) :
                                      B_hash[byte_shift] >> shift) & (K - 1));
        }

        // result of the multiplication of a slice of the source matrix on a D slice
	
#define BATCHING
//#define INNER_CLOCK

#ifdef BATCHING
	Dtype *batched_input = new Dtype[batch_size * M];
	caffe_set(batch_size * M, Dtype(0), batched_input);
	Dtype *batched_output = new Dtype[batch_size * K];
	Dtype *new_d = new Dtype[M * K];
#ifdef INNER_CLOCK
	time_t sum_transp = 0;
	time_t sum_mult = 0;
	time_t sum_batch = 0;
	time_t sum_add = 0;
#endif

	volatile Dtype not_optimized = 0.0;
	for (int j = 0; j < num_input; j += M) {
#ifdef INNER_CLOCK
	    time_t cur_t = clock();
	    time_t prev_t;
#endif
	    for (int i = 0; i < batch_size; ++i) {
	        for (int l = 0; l < M; ++l) {
		    batched_input[i * M + l] = bottom_data[i * num_input + j + l];
		}
	    }
#ifdef INNER_CLOCK
	    prev_t = cur_t;
	    cur_t=clock();
	    sum_batch += cur_t - prev_t;
#endif
	    const Dtype *d = D + K * j;
	    for (int i = 0; i < K; ++i) {
	        for (int j = 0; j < M; ++j) {
		    new_d[i * M + j] = d[j * K + i];
	        }
	    }
#ifdef INNER_CLOCK
	    prev_t = cur_t;
	    cur_t=clock();
	    sum_transp += cur_t - prev_t;
#endif
	    caffe_cpu_gemm(CblasNoTrans, CblasTrans, 
			    batch_size, M, K,
			    (Dtype)1., batched_input,new_d, (Dtype)0., batched_output);
#ifdef INNER_CLOCK
	    prev_t = cur_t;
	    cur_t=clock();
	    sum_mult += cur_t - prev_t;
#endif
	    const int j_shift = j / M * num_output;
	    //not_optimized += B[j_shift];
	    for (int i = 0; i < batch_size; ++i) {
		const int i_shift = i * num_output;
		const int i_k_shift = K * i;
		not_optimized += batched_output[i_k_shift];
		for (int l = 0; l < num_output; ++l) {
		    top_data[i_shift + l] += batched_output[i_k_shift + B[j_shift + l]];
		}
	    }
#ifdef INNER_CLOCK
	    prev_t = cur_t;
	    cur_t=clock();
	    sum_add += cur_t - prev_t;
#endif
	}
#ifdef INNER_CLOCK
	LOG(ERROR) << "TRANSPOSE: " << 1000.0 * sum_transp / CLOCKS_PER_SEC; 
	LOG(ERROR) << " MULTIPLY: " << 1000.0 * sum_mult / CLOCKS_PER_SEC; 
	LOG(ERROR) << " BATCHING: " << 1000.0 * sum_batch / CLOCKS_PER_SEC; 
	LOG(ERROR) << " ADDITION: " << 1000.0 * sum_add / CLOCKS_PER_SEC << "\n";
#endif	
	delete[] batched_input;
	delete[] batched_output;
#else //BATCHING
        Dtype *output = new Dtype[K];
        caffe_set(K, Dtype(0), output);
        // the most time-consuming part of code
        for (int i = 0; i < batch_size; ++i) {
            caffe_set(num_output, Dtype(0), top_data + i * num_output);
            for (int j = 0; j < num_input / M; ++j) {
                // subvector of the source image vector
                const Dtype *s = bottom_data + i * num_input + j * M;
                // submatrix of the reduced weights matrix
                const Dtype *d = D + K * M * j;
                // multiplies vector on matrix
                caffe_cpu_gemv(CblasTrans, M, K, (Dtype)1., d, s, (Dtype)0., output);

                for (int l = 0; l < num_output; ++l) {
                    top_data[i * num_output + l] += output[B[j * num_output + l]];
                }
            }
        }
        delete[] output;
#endif //BATCHING
        delete[] B;
        // adds bias to the resulting matrix
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch_size, num_output, 1, (Dtype)1.,
                              bias_multiplier_.cpu_data(), bias, (Dtype)1., top_data);
    }

    template <typename Dtype>
    void InnerProductQLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down,
                                                 const vector<Blob<Dtype>*>& bottom) {
        /*if (this->param_propagate_down_[0]) {
            const Dtype* top_diff = top[0]->cpu_diff();
            const Dtype* bottom_data = bottom[0]->cpu_data();
            // Gradient with respect to weight
            if (transpose_) {
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                      K_, N_, M_,
                                      (Dtype)1., bottom_data, top_diff,
                                      (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
            } else {
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
                                      N_, K_, M_,
                                      (Dtype)1., top_diff, bottom_data,
                                      (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
            }
        }
        if (bias_term_ && this->param_propagate_down_[1]) {
            const Dtype* top_diff = top[0]->cpu_diff();
            // Gradient with respect to bias
            caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
                                  bias_multiplier_.cpu_data(), (Dtype)1.,
                                  this->blobs_[1]->mutable_cpu_diff());
        }
        if (propagate_down[0]) {
            const Dtype* top_diff = top[0]->cpu_diff();
            // Gradient with respect to bottom data
            if (transpose_) {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
                                      M_, K_, N_,
                                      (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
                                      (Dtype)0., bottom[0]->mutable_cpu_diff());
            } else {
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                                      M_, K_, N_,
                                      (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
                                      (Dtype)0., bottom[0]->mutable_cpu_diff());
            }
        }*/
    }

#ifdef CPU_ONLY
    STUB_GPU(InnerProductQLayer);
#endif

    INSTANTIATE_CLASS(InnerProductQLayer);
    REGISTER_LAYER_CLASS(InnerProductQ);

}  // namespace caffe
