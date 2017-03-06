#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/conv_q_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

#ifndef CPU_ONLY
	extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
	class ConvolutionQLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;
protected:
	ConvolutionQLayerTest() : blob_bottom_(new Blob<Dtype>(1, 4, 5, 5)), blob_top_(new Blob<Dtype>()),
		D_(new Blob<Dtype>(1, 1, 2, 4)){
		for (int i = 1; i <= 100; ++i) {
			blob_bottom_->mutable_cpu_data()[i - 1] = Dtype(i);
		}
		for (int i = 1; i <= 8; ++i) {
			D_->mutable_cpu_data()[i - 1] = Dtype(i);
		}
		blob_bottom_vec_.push_back(blob_bottom_);
		blob_top_vec_.push_back(blob_top_);
	}

	virtual ~ConvolutionQLayerTest() { }

	Blob<Dtype>* const blob_bottom_;
	Blob<Dtype>* const blob_top_;
	Blob<Dtype>* const D_;
	vector<Blob<Dtype>*> blob_bottom_vec_;
	vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConvolutionQLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConvolutionQLayerTest, TestForward) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
	vector<int> shape(1, 2);
	Blob<Dtype>* const B_ = new Blob<Dtype>(shape);
	int val1 = 88848439;
	int val2 = 1879048192;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
	B_->mutable_cpu_data()[0] = reinterpret_cast<Dtype&>(val1);
	B_->mutable_cpu_data()[1] = reinterpret_cast<Dtype&>(val2);
#pragma GCC diagnostic pop
	Blob<Dtype>* const bias_ = new Blob<Dtype>(shape);
	bias_->mutable_cpu_data()[0] = 0;
	bias_->mutable_cpu_data()[1] = -1;
#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif

	if (Caffe::mode() == Caffe::CPU || sizeof(Dtype) == 4 || IS_VALID_CUDA) {
		LayerParameter layer_param;
		ConvolutionParameter* convolution_param = layer_param.mutable_convolution_param();
		convolution_param->add_kernel_size(3);
		convolution_param->add_stride(2);
		convolution_param->add_pad(1);
		convolution_param->set_num_output(2);
		convolution_param->set_k(2);
		convolution_param->set_m(2);
		shared_ptr<ConvolutionQLayer<Dtype> > layer(new ConvolutionQLayer<Dtype>(layer_param));
		Blob<Dtype>* const blobs[] = {this->D_, bias_, B_};
		layer->blobs() = vector<shared_ptr<Blob<Dtype> > >(blobs, blobs + 3);
		layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		const Dtype *data = this->blob_top_->cpu_data();
		const int count = this->blob_top_->count();
		const Dtype correct[] = {3634, 5396, 3482, 6548, 9788, 6424, 4834, 7334, 4888,
								 4161, 6611, 4745, 6881, 11039, 7723, 4853, 7947, 5525};
		for (int i = 0; i < count; ++i) {
			EXPECT_EQ(data[i], correct[i]);
		}
	} else {
		LOG(ERROR) << "Skipping test due to old architecture.";
	}
}

}
