#!/usr/bin/python2

import os
os.environ['GLOG_minloglevel'] = '1'
import caffe
import caffe.proto.caffe_pb2 as pb2
import google.protobuf.text_format as tf
import extract_weights as ew
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Convert conv and fc layers to quantized format')
parser.add_argument('-K', help='quantity of clusters', default=32, type=int)
parser.add_argument('-M', help='size of subvector', default=4, type=int)
parser.add_argument('-L', '--loadweights', help='load weights from D.npy and B.npy', action='store_true')
parser.add_argument('-W', '--newweights', help='file to save new weights')
parser.add_argument('-C', '--newmodel', help='file to save new prototxt')
parser.add_argument('model', help='model.prototxt')
parser.add_argument('weights', help='weights.caffemodel')
parser.add_argument('layer', help='conv or fc layer to convert')

args = parser.parse_args()

print('Start model converting')
net = caffe.Net(network_file=args.model, phase=caffe.TEST, weights=args.weights)
net_params = pb2.NetParameter()
with open(args.model) as f:
    tf.Merge(f.read(), net_params)
print('Model has been loaded')

if args.loadweights:
    D, B = np.load('D.npy'), np.load('B.npy')
    print('Weights have been loaded')
else:
    weights = net.params[args.layer][0].data
    # TODO: D calculation doesn't need to depend from fc or conv type
    is_fc = False
    for i in xrange(len(net._layer_names)):
        if net._layer_names[i] == args.layer:
            layer = net_params.layers[i]
            layer.ClearField('blobs_lr')
            layer.ClearField('weight_decay')
            if net.layers[i].type == 'Convolution':
                in_channels = weights.shape[1]
                weights = weights.transpose((0, 2, 3, 1)).reshape((-1, in_channels))

                param = layer.convolution_param
                param.engine = pb2.ConvolutionParameter.Engine.Value('QUANT')
                param.ClearField('bias_filler')
                param.ClearField('weight_filler')
            elif net.layers[i].type == 'InnerProduct':
                is_fc = True
                layer.type = pb2.V1LayerParameter.LayerType.Value('INNER_PRODUCT_Q')
                # TODO: type change is impossible
                # need to make engine like for conv layer or save-load-save .caffemodel
                # net.layers[i].type = 'InnerProductQ'
                param = layer.inner_product_q_param
                param.num_output = layer.inner_product_param.num_output
                layer.ClearField('inner_product_param')
            else:
                raise ValueError('%s layer %s type is unknown' % (args.layer, net.layers[i].type))
            param.k = args.K
            param.m = args.M
    print('Weights have been extracted')

    D, B = ew.calcBD(weights, args.K, args.M, is_fc)
    print('D and B have been calced')
    B = ew.codeB(B, args.K)
    print('B has been coded')
    np.save('D', D)
    np.save('B', B)

net.params[args.layer][0].reshape(*D.shape)
net.params[args.layer][0].data[...] = D
net.params[args.layer].add_blob()
net.params[args.layer][2].reshape(*B.shape)
net.params[args.layer][2].data[...] = B
print('D and B have been saved to layer %s' % args.layer)

new_weights = args.newweights if args.newweights else args.weights
new_model = args.newmodel if args.newmodel else args.model
with open(new_model, 'w') as f:
    f.write(tf.MessageToString(net_params))
print('Model has been saved to %s' % new_model)

net.save(new_weights)
print('Weights have been saved to %s' % new_weights)
