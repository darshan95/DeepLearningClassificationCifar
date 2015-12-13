import numpy as np
caffe_root = '/home/darshan/caffe/caffe'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Global Variable Paths
original_network = '/home/darshan/machine_learning/datasets/cifar-10/caffemodel-100epoch/deploy.prototxt'
original_model = '/home/darshan/machine_learning/datasets/cifar-10/caffemodel-100epoch/cifar10_full_iter_60000.caffemodel'
modified_network = '/home/darshan/machine_learning/datasets/cifar-100/deploy_100label.prototxt'
modified_model = '/home/darshan/machine_learning/datasets/cifar-100/modified_cifarNet_234400_100epoch.caffemodel'



# Load the original network.
print "[INFO] Loading original Net."
try:
	net = caffe.Net(original_network, original_model, caffe.TEST)
except:
	print "[ERROR] Loading original Net failed. Check caffemodel and network prototxt."
	raise "Model loading failed"

# Load model to be updated.
print "[INFO] Loading"
try:
	modified_net = caffe.Net(modified_network, caffe.TEST)
except:
	print "[ERROR] Loading modified Net failed. Check network prototxt."
	raise "Model loading failed"

# Replicate the model layers.
print "[INFO] Updating modified models softmax layer"
try:
    params = ['conv1', 'conv2', 'conv3']
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

    params_full_conv = ['conv1', 'conv2', 'conv3']
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]
except:
	print "[ERROR] Updating model layers failed."
	raise "Layer updation failed"

# Save the updated model.
print "[INFO] Saving the updated model."
try:
	modified_net.save(modified_model)
except:
	print "[ERROR] Saving updated model failed."
	raise "Error saving model"

print ""
print "DONE."
