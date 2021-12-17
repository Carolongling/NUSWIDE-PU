import h5py
import numpy as np
import os, pickle, gzip

def loading_data(path):
	file = h5py.File(path)
	images = file['images'][:].transpose(0,3,2,1)
	tags = file['YAll'][:].transpose(1,0)
	labels = file['LAll'][:].transpose(1,0)
	tags=tags.transpose()
	labels=labels.transpose()
	file.close()
	return images, tags, labels

def save_object_to_zip(objects, filename):
	if not os.path.exists(filename):
		file_path = os.path.split(filename)[0]
		if file_path and not os.path.exists(file_path):  # 需要文件夹
			os.mkdir(os.path.split(filename)[0])  # 创建文件夹
		open(filename, 'a').close()   # 创建文件
	fil = gzip.open(filename, 'wb')
	pickle.dump(objects, fil)
	fil.close()

def load_object_from_zip(filename):
	fil = gzip.open(filename, 'rb')
	while True:
		try:
			return pickle.load(fil)
		except EOFError:
			break
	fil.close()

if __name__=='__main__':
	file_path = 'FLICKR-25K.mat'
	images, tags ,labels= loading_data(file_path)
	print(images.shape)
	print(tags.shape)
	print(labels.shape)
	save_object_to_zip(images,'images')
	save_object_to_zip(tags, 'tags')
	save_object_to_zip(labels, 'labels')
	# savefile_path = './'
	image_zip = load_object_from_zip('images')
	tags_zip = load_object_from_zip('tags')
	labels_zip = load_object_from_zip('labels')

	#print(images.shape)
	#print(tags.shape)