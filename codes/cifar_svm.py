import numpy as np
caffe_root = '/home/darshan/caffe/caffe'  # this file is expected to be in {caffe_root}/examples
import sys
import os
sys.path.insert(0, caffe_root + 'python')

import caffe
from sklearn.svm import SVC
from sklearn import metrics

# Global Variable Paths
original_network = '/home/darshan/machine_learning/datasets/cifar-10/caffemodel-100epoch/deploy.prototxt'
original_model = '/home/darshan/machine_learning/datasets/cifar-10/caffemodel-100epoch/cifar10_full_iter_60000.caffemodel'
test_db = '/home/darshan/machine_learning/datasets/cifar-10/test'
train_db = '/home/darshan/machine_learning/datasets/cifar-10/train'
labels_file = '/home/darshan/machine_learning/datasets/cifar-10/train/labels.txt'

# Load the original network.
print "[INFO] Loading original Net."
try:
	net = caffe.Net(original_network, original_model, caffe.TEST)
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_transpose('data', (2,0,1))
	transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
	transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
	net.blobs['data'].reshape(1,3,32,32)
except:
	print "[ERROR] Loading original Net failed. Check caffemodel and network prototxt."
	raise "Model loading failed"

# Load Test and Train images.
input_features = []
input_labels = []

label_map = []
labels = []
print "[INFO] Loading training images"
try:
	print "[INFO] Loading labels"
	with open(labels_file, 'r') as l_file:
		for line in l_file.readlines():
			label_map.append(line.strip())
	labels = range(len(label_map))
except:
    print "[ERROR] Loading labels failed"
    raise "Loading label file failed"

print "[INFO] Loading Train Images"
try:
	data = np.load('training_features.npz')
	input_labels = data['input_labels']
	input_features = data['input_features']
except:
	categs = sorted(os.listdir(train_db))
	for categ in categs:
		try:
			cur_label = label_map.index(categ)
		except:
			continue
		print("[ %s%s ] Loading : %s" % (str(float(cur_label*100)/len(labels)), '%', categ))
		ctr = 0
		for img in os.listdir(train_db+'/'+categ):
			ctr += 1
			sys.stdout.write("\r")
			sys.stdout.write(" [ %s%s ] Complete" % (str(float(ctr*100)/len(os.listdir(train_db+'/'+categ))), "%"))
			try:
				img_path = str(train_db+'/'+categ+'/'+img)
				net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img_path))
				out = net.forward()
				img_feature = net.blobs['ip1'].data[0]
				input_features.append(img_feature)
				input_labels.append(cur_label)
			except:
				print "[ERROR] Loading images %s failed" % img_path
				raise "Loading image failed"
			sys.stdout.flush()
	print "Saving training features"
	try:
		np.savez('training_features.npz',input_features=np.array(input_features),input_labels=np.array(input_labels))
	except:
		print "[ERROR] Saving training features as numpy failed."
		raise "saving numpy failed"
print "Loading training images features complete."

	

print "[INFO] Normalizing features"
input_features = input_features/np.linalg.norm(input_features)

print "Training SVM on features"
clf = SVC()
clf.fit(np.array(input_features),np.array(input_labels))

print "[INFO] Loading Testing Images"
test_features = []
test_labels = []
try:
	data = np.load('testing_features.npz')
	test_labels = data['test_labels']
	test_features = data['test_features']
except:
	for categ in categs:
		try:
			cur_label = label_map.index(categ)
		except:
			continue
		print("[ %s%s ] Loading : %s" % (str(float(cur_label*100)/len(labels)), '%', categ))
		ctr = 0
		for img in os.listdir(test_db+'/'+categ):
			ctr += 1
			sys.stdout.write("\r")
			sys.stdout.write(" [ %s%s ] Complete" % (str(float(ctr*100)/len(os.listdir(test_db+'/'+categ))), "%"))
			try:
				img_path = str(test_db+'/'+categ+'/'+img)
				net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img_path))
				out = net.forward()
				img_feature = net.blobs['ip1'].data[0]
				test_features.append(img_feature)
				test_labels.append(cur_label)
			except:
				print "[ERROR] Loading images %s failed" % img_path
				raise "Loading image failed"
			sys.stdout.flush()
	try:
		print "Saving testing features"
		np.savez('testing_features.npz',test_features=np.array(test_features),test_labels=np.array(test_labels))
	except:
		print "[ERROR] Saving testing features as numpy failed."
		raise "saving numpy failed"

print "Loading testing images features complete."

print "[INFO] Normalizing Testing features"
test_features = test_features/np.linalg.norm(test_features)

print "Testing features on SVM"
predicted = clf.predict(test_features)
expected = test_labels
print "Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(expected, predicted))
print "Confusion matrix:\n", metrics.confusion_matrix(expected, predicted)

print ""
print "DONE."
