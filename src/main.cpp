//
//  main.cpp
//  caffe
//
//  Created by Valerii Filev on 15/03/2017.
//
//

#include "main.hpp"
#include <caffe/caffe.hpp>

int main() {
    std::shared_ptr<caffe::Net<float>> net;
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
    net.reset(new caffe::Net<float>("/Users/valerii.filev/Looksery/caffe/models/GenderSqueezeNet.prototxt", caffe::TEST));
    net->CopyTrainedLayersFrom("/Users/valerii.filev/Looksery/caffe/models/GenderSqueezeNet.caffemodel");
    
    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net->num_outputs(), 1) << "Network should have exactly one output.";
    caffe::Blob<float>* input_layer = net->input_blobs()[0];
    CHECK_EQ(input_layer->channels(), 3) << "Input layer should have 3 channels.";
    
    float* input_data = input_layer->mutable_cpu_data();
    const float mean[] = {104.00699, 116.66877, 122.67892};
    for (int channel = 0; channel < 3; channel++) {
        for (int i = 0; i < 271; i++) {
            for (int j = 0; j < 191; j++) {
                *input_data = rand() % 256 - mean[channel];
                input_data++;
            }
        }
    }
    
    net->Forward();
    
    caffe::Blob<float>* output_layer = net->output_blobs()[0];
    CHECK_EQ(output_layer->channels(), 2) << "Output layer should have 2 channels.";
    std::cerr << output_layer->cpu_data()[0] << " " << output_layer->cpu_data()[1] << std::endl;
}
