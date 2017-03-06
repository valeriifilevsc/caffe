#ifndef CAFFE_CONV_Q_LAYER_HPP_
#define CAFFE_CONV_Q_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class ConvolutionQLayer : public ConvolutionLayer<Dtype> {
public:
	explicit ConvolutionQLayer(const LayerParameter& param) : ConvolutionLayer<Dtype>(param) {}
	//virtual inline const char* type() const { return "Convolution"; }

protected:
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	//virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	//virtual inline bool reverse_dimensions() { return false; }

	Blob<Dtype> cache_;
	Blob<int> B_;

private:
	int conv_out_spatial_dim_;
	int conv_in_spatial_dim_;
	int K;
	int M;
};

}

#endif  // CAFFE_CONV_Q_LAYER_HPP_
