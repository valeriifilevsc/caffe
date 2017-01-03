#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_q_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class InnerProductQLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
protected:
    InnerProductQLayerTest()
    { 
    }
    virtual ~InnerProductQLayerTest() {
        /*delete[] blob_bottom_;
        delete[] blob_top_;
        delete[] D_;*/
    }

    Blob<Dtype>* blob_bottom_;
    Blob<Dtype>* blob_top_;
    Blob<Dtype>* D_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(InnerProductQLayerTest, TestDtypesAndDevices);


TYPED_TEST(InnerProductQLayerTest, TestForward) {     
    typedef typename TypeParam::Dtype Dtype;
    bool IS_VALID_CUDA = false;
    Blob<Dtype>* const B_ = new Blob<Dtype>(1, 1, 1, 1);
    int val = 147779584;
    
    this->blob_bottom_ = new Blob<Dtype>(1, 1, 4, 2);
    this->blob_top_ = new Blob<Dtype>();
    this->D_ = new Blob<Dtype>(1, 1, 8, 2);
    for (int i = 0; i < 8; i++) {
        this->blob_bottom_->mutable_cpu_data()[i] = Dtype(i / 2 + 1);
        this->D_->mutable_cpu_data()[2 * i] = Dtype(i / 2 + i % 2);
        this->D_->mutable_cpu_data()[2 * i + 1] = Dtype(i / 2 + i % 2 + 2);
    }
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
    B_->mutable_cpu_data()[0] = reinterpret_cast<Dtype&>(val);
#pragma GCC diagnostic pop
    Blob<Dtype>* const bias_ = new Blob<Dtype>(1, 1, 1, 5);
    caffe_set(5, Dtype(1), bias_->mutable_cpu_data());
#ifndef CPU_ONLY
    IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
    if (Caffe::mode() == Caffe::CPU || sizeof(Dtype) == 4 || IS_VALID_CUDA) {
        LayerParameter layer_param;
        InnerProductQParameter *inner_product_q_param = layer_param.mutable_inner_product_q_param();
        inner_product_q_param->set_num_output(5);
        inner_product_q_param->set_k(2);
        inner_product_q_param->set_m(2);
        shared_ptr<InnerProductQLayer<Dtype> > layer(new InnerProductQLayer<Dtype>(layer_param));
        Blob<Dtype>* const blobs[] = {this->D_, bias_, B_};
        layer->blobs() = vector<shared_ptr<Blob<Dtype> > >(blobs, blobs + 3);
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        const Dtype *data = this->blob_top_->cpu_data();
        const int count = this->blob_top_->count();
        const Dtype correct[] = {51, 67, 79, 87, 91};
        for (int i = 0; i < count; ++i) {
            EXPECT_EQ(data[i], correct[i]);
        }
    } else {
        LOG(ERROR) << "Skipping test due to old architecture.";
    }
}


TYPED_TEST(InnerProductQLayerTest, TestBigData) {  
    const int BATCH_SIZE = 47;	
    const int INPUT_LAYER = 10000;
    const int OUTPUT_LAYER = 512;
    const int TEST_NUM = 1;
    const int K = 32;
    const int M = 4;
    typedef typename TypeParam::Dtype Dtype;
    bool IS_VALID_CUDA = false;
    int b_size = OUTPUT_LAYER * INPUT_LAYER / M * int(log2(K) + 0.5) / 8 / sizeof(Dtype);
    Blob<Dtype>* const B_ = new Blob<Dtype>(1, 1, 1, b_size);
    //int val = 147779584;
    
    srand(138531);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
    for (int i = 0; i < b_size; ++i) {
	int tmp = (rand()<<16)|rand();
	B_->mutable_cpu_data()[i] = reinterpret_cast<Dtype&>(tmp);
    }
#pragma GCC diagnostic pop
    this->blob_bottom_ = new Blob<Dtype>(BATCH_SIZE, 1, INPUT_LAYER, 1);
    this->blob_top_ = new Blob<Dtype>();
    this->D_ = new Blob<Dtype>(INPUT_LAYER, 1, K, 1);
    for (int i = 0; i < BATCH_SIZE * INPUT_LAYER; i++) {
        this->blob_bottom_->mutable_cpu_data()[i] = Dtype(rand() & 255);
    }
    for (int i = 0; i < INPUT_LAYER * K; i++) {
        this->D_->mutable_cpu_data()[i] = Dtype(rand() & 255);
    }
    this->blob_bottom_vec_.push_back(this->blob_bottom_);
    this->blob_top_vec_.push_back(this->blob_top_);

    Blob<Dtype>* const bias_ = new Blob<Dtype>(1, 1, 1, OUTPUT_LAYER);
    caffe_set(OUTPUT_LAYER, Dtype(1), bias_->mutable_cpu_data());
#ifndef CPU_ONLY
    IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
    if (Caffe::mode() == Caffe::CPU || sizeof(Dtype) == 4 || IS_VALID_CUDA) {
        LayerParameter layer_param;
        InnerProductQParameter *inner_product_q_param = layer_param.mutable_inner_product_q_param();
        inner_product_q_param->set_num_output(OUTPUT_LAYER);
        inner_product_q_param->set_k(K);
        inner_product_q_param->set_m(M);
        shared_ptr<InnerProductQLayer<Dtype> > layer(new InnerProductQLayer<Dtype>(layer_param));
        Blob<Dtype>* const blobs[] = {this->D_, bias_, B_};
        layer->blobs() = vector<shared_ptr<Blob<Dtype> > >(blobs, blobs + 3);
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	time_t start_t = clock();
	for (int x = 0; x < TEST_NUM; ++x) {
            layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	}
	time_t cur_t = clock() - start_t;
	LOG(ERROR) << "MY OUTPUT "<< 1000.0 * cur_t / CLOCKS_PER_SEC;
    } else {
        LOG(ERROR) << "Skipping test due to old architecture.";
    }

    if (Caffe::mode() == Caffe::CPU ||
        sizeof(Dtype) == 4 || IS_VALID_CUDA) {
        LayerParameter layer_param;
        InnerProductParameter* inner_product_param =
            layer_param.mutable_inner_product_param();
        inner_product_param->set_num_output(OUTPUT_LAYER);
        inner_product_param->mutable_weight_filler()->set_type("uniform");
        inner_product_param->mutable_bias_filler()->set_type("uniform");
        inner_product_param->mutable_bias_filler()->set_min(1);
        inner_product_param->mutable_bias_filler()->set_max(2);
        shared_ptr<InnerProductLayer<Dtype> > layer(
            new InnerProductLayer<Dtype>(layer_param));
        layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	time_t start_t = clock();
	for (int x = 0; x < TEST_NUM; ++x) {
            layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	}
	time_t cur_t = clock() - start_t;
	LOG(ERROR) << "MY OUTPUT "<< 1000.0 * cur_t / CLOCKS_PER_SEC;
    } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
    }
}

}
