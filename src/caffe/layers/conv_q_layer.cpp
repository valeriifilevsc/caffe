#include <vector>
#include <time.h>

#include "caffe/layers/conv_q_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void ConvolutionQLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		BaseConvolutionLayer<Dtype>::Reshape(bottom, top);

		conv_out_spatial_dim_ =  top[0]->count(this->channel_axis_ + 1);
		conv_in_spatial_dim_ = bottom[0]->count(this->channel_axis_ + 1);
		K = this->layer_param_.convolution_param().k();
		M = this->layer_param_.convolution_param().m();
		const int BITS = (int)log2(K);
		const int TOTAL_BITS = 32;
		const int REST_BITS = TOTAL_BITS - BITS;
		vector<int> cache_shape_(1, K * this->channels_ / M * conv_in_spatial_dim_);
		cache_.Reshape(cache_shape_);

		const int kernel_h = this->kernel_shape_.cpu_data()[0];
		const int kernel_w = this->kernel_shape_.cpu_data()[1];
		const int kernel_dim_ = kernel_h * kernel_w;
		const int b_shape_size = this->channels_ / M * kernel_dim_ * this->num_output_;
		if (this->blobs_.size() < 3) {
			this->blobs_.resize(3);
			vector<int> d_shape(2);
			d_shape[0] = K * this->channels_ / M;
			d_shape[1] = M;
			this->blobs_[0].reset(new Blob<Dtype>(d_shape));
			vector<int> b_binary_shape(1, BITS * b_shape_size / (8 * sizeof(Dtype)) + (BITS * b_shape_size % (8 * sizeof(Dtype)) ? 1 : 0));
			this->blobs_[2].reset(new Blob<Dtype>(b_binary_shape));
		}
		else {
			if (B_.count() == 0) {
				vector<int> b_shape(1, b_shape_size);
				B_.Reshape(b_shape);
			}
			int* B_hash = (int*)(this->blobs_[2]->cpu_data());
			int* B = B_.mutable_cpu_data();
			for (int i = 0, total_bit_shift = 0; i < b_shape_size; ++i, total_bit_shift += BITS) {
				int byte_shift = total_bit_shift / TOTAL_BITS;
				int bit_shift = total_bit_shift % TOTAL_BITS;
				int shift = REST_BITS - bit_shift;
				B[i] = (int)((shift < 0 ? B_hash[byte_shift] << -shift | B_hash[byte_shift + 1] >> (TOTAL_BITS + shift) :
										  B_hash[byte_shift] >> shift) & (K - 1));
			}
		}
	}

	template <typename Dtype>
	void ConvolutionQLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* D = this->blobs_[0]->cpu_data();
		// hash with indexes of D columns, each number uses log2(K) bits
		const int* B = B_.cpu_data();

		const int height = this->conv_input_shape_.cpu_data()[1];
		const int width = this->conv_input_shape_.cpu_data()[2];
		const int kernel_h = this->kernel_shape_.cpu_data()[0];
		const int kernel_w = this->kernel_shape_.cpu_data()[1];
		const int kernel_dim_ = kernel_h * kernel_w;
		const int pad_h = this->pad_.cpu_data()[0];
		const int pad_w = this->pad_.cpu_data()[1];
		const int stride_h = this->stride_.cpu_data()[0];
		const int stride_w = this->stride_.cpu_data()[1];
		const int dilation_h = this->dilation_.cpu_data()[0];
		const int dilation_w = this->dilation_.cpu_data()[1];
		const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int output_w = (width + 2 * pad_w -(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

		// TODO: support for multiple bottoms and tops
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();

			for (int n = 0; n < this->num_; ++n) {
				Dtype* top_data_image = &top_data[n * this->top_dim_];
				caffe_set(this->top_dim_, (Dtype)0., top_data_image);
				const Dtype* bottom_data_image = &bottom_data[n * this->bottom_dim_];

				for (int slice = 0; slice < this->channels_ / M; ++slice) {
					caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, K, conv_in_spatial_dim_, M, (Dtype)1.,
							D + slice * M * K, bottom_data_image + slice * M * conv_in_spatial_dim_, (Dtype)0.,
							this->cache_.mutable_cpu_data());
					const int* b_slice = &B[slice * this->num_output_ * kernel_dim_];

					for (int out_channel = 0; out_channel < this->num_output_; ++out_channel) {
						const int* b_out_channel = &b_slice[out_channel * kernel_dim_];
						Dtype* top_data_channel = &top_data_image[out_channel * this->conv_out_spatial_dim_];

						for (int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
							const int* b_kernel_row = &b_out_channel[kernel_row * kernel_w];

							for (int kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
								int b = b_kernel_row[kernel_col];
								int input_row = -pad_h + kernel_row * dilation_h;
								const Dtype* cache_row = &this->cache_.cpu_data()[b * conv_in_spatial_dim_];

								for (int output_row = 0; output_row < output_h; ++output_row) {
									if (static_cast<unsigned>(input_row) < static_cast<unsigned>(height)) {
										const Dtype* cache_row_row = &cache_row[input_row * width];
										Dtype* top_data_row = &top_data_channel[output_row * output_w];
										int input_col = -pad_w + kernel_col * dilation_w;
										for (int output_col = 0; output_col < output_w; ++output_col) {
											if (static_cast<unsigned>(input_col) < static_cast<unsigned>(width)) {
												top_data_row[output_col] += cache_row_row[input_col];
											}
											input_col += stride_w;
										}
									}
									input_row += stride_h;
								}
							}
						}
					}
				}

				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->cpu_data();
					this->forward_cpu_bias(top_data_image, bias);
				}
			}
		}
	}

    
//#ifdef CPU_ONLY
//STUB_GPU(ConvolutionQLayer);
//#endif

INSTANTIATE_CLASS(ConvolutionQLayer);

}  // namespace caffe
