#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_q_layer.hpp"

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
            : blob_bottom_(new Blob<Dtype>(1, 1, 4, 2)),
              blob_top_(new Blob<Dtype>()),
              D_(new Blob<Dtype>(1, 1, 8, 2)){
        // fill the values
        for (int i = 0; i < 8; i++) {
            blob_bottom_->mutable_cpu_data()[i] = Dtype(i / 2 + 1);
            D_->mutable_cpu_data()[2 * i] = Dtype(i / 2 + i % 2);
            D_->mutable_cpu_data()[2 * i + 1] = Dtype(i / 2 + i % 2 + 2);
        }
        blob_bottom_vec_.push_back(blob_bottom_);
        blob_top_vec_.push_back(blob_top_);
    }
    virtual ~InnerProductQLayerTest() {
        /*delete[] blob_bottom_;
        delete[] blob_top_;
        delete[] D_;*/
    }

    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_;
    Blob<Dtype>* const D_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(InnerProductQLayerTest, TestDtypesAndDevices);

TYPED_TEST(InnerProductQLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    bool IS_VALID_CUDA = false;
    Blob<Dtype>* const B_ = new Blob<Dtype>(1, 1, 1, 1);
    int val = 147779584;
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

}
