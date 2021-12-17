import collections
import numpy as np
from torch.nn.modules import module
from load_data import load_object_from_zip
from vgg import make_model, extract_feature
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from pu_learning import spies
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':
	images = load_object_from_zip('images')
	models = make_model()
	n = 100
	img = images[:n]
	img1 = img.transpose(0, 3, 2, 1)
	img2 = torch.from_numpy(img1).float()
	img_feature = extract_feature(models, img2)  # 出来的是二维 图片序号 另一个是具体的内容

	tags = load_object_from_zip('tags')
	# 取前n条文本特征
	tag_feature = tags[:n]
	tags_feature = torch.from_numpy(tag_feature).float()
	# ---------------------------------------------------------------------------------
	# 制作mixed dataset，选择前n条 用于predict的数据集
	mixed_dict = []
	for i in range(n):
		for j in range(n):
			mixed_dict.append([i, j])

	mixed_data = []  # list
	match_data = []
	y1 = []  # mixed里的真实标签 混有 n个secret positive的
	for i in mixed_dict:
		a = img_feature[i[0]].view(1, len(img_feature[i[0]]))  # [1, 100352]
		b = tags_feature[i[1]].view(1, len(tags_feature[i[1]]))  # [1, 1386]
		c = torch.cat((a, b), 1)
		d = c.view(101738,)
		mixed_data.append(d)  # 全部的 [[1, 101738],[]] list中的

		if i[0] == i[1]:
			y1.append(1)  # 如果字典的下标相等，打标签为1
		else:
			y1.append(0)
	mixed_feature = torch.stack(mixed_data)  # list中装着多个同纬度的tensor，想让这个list转为tensor

	# Unlabeled data的真实label
	y2 = np.array(y1)  # list转numpy 这个是mixed 的真实y
	mix_y_real = y2.astype(int)  # 转整 用于画出auc
	# Unlabeled data的构造label
	y3 = np.zeros(n*n,dtype=int)
	mix_y_false = y3.astype(int)  # 转整

	# ---------------------------------------------------------------------------------
	# Positive的dataset 选择n：2n条，将两个特征做拼接，对齐的交集数据
	img = images[n:500]
	img1 = img.transpose(0, 3, 2, 1)
	img2 = torch.from_numpy(img1).float()
	img_feature = extract_feature(models, img2)  # 出来的是二维 图片序号 另一个是具体的内容

	tags = load_object_from_zip('tags')
	# 取前n条文本特征
	tag_feature = tags[n:n*n+n]
	tags_feature = torch.from_numpy(tag_feature).float()
	P_dataset = torch.cat((img_feature, tags_feature), 1)

	# 打标签
	P_label = np.ones(n,dtype=int)
	# ---------------------------------------------------------------------------------


	# 制作train的数据集=100个正例 + 100*100个unlabeled包含100个secret positive
	trainAlldata = torch.cat((mixed_feature, P_dataset), 0)  # [110, 101738]
	traindata = np.array(trainAlldata)  # tensor转numpy
	MIX_Y_F = torch.from_numpy(mix_y_false)
	P_Y = torch.from_numpy(P_label)
	Y1 = torch.cat((MIX_Y_F,P_Y), 0)
	Y = np.array(Y1)
	# SPY算法
	s = spies(XGBClassifier(), XGBClassifier())
	# Trains models using spies method using training set (X, y)
	s.fit(X=traindata, y=Y)


	# Predicts classes for X. Uses second trained model from self
	out = s.predict(mixed_feature)

	# Predict class probabilities for X. Uses second trained model from self
	out_prob = s.predict_proba(mixed_feature)

	# 对predict出的正例和unlabeled计数
	data_count = collections.Counter(out)

	# 计算AUC 本质就是一个排序任务 y要是真实的y
	auc = roc_auc_score(mix_y_real, out_prob)

	# 输出结果
	print(data_count)
	print(out)
	print(out_prob)
	print(auc)


	# U_y1 = np.array(U_y)  # list转numpy
	# U_y2 = U_y1.astype(int)  # 转整
	# Unlabel_y = torch.from_numpy(U_y2)
	#
	# mixed = np.array(mixed_feature)   # tensor转numpy

	# print(tags_feature)
	# print(match)

	# mixed dataset n*n index来标记
	# index = [0, ... , 100]  # 样本序号前10个
	# index_tmp = list(range(0, n*n))
	# index = np.array(index_tmp)


# # -------------------------------------------------------------------------------------
# # 制作secret positive dataset
# se_img = images[10:20]
# se_img1 = se_img.transpose(0, 3, 2, 1)
# se_img2 = torch.from_numpy(se_img1).float()
# se_img_feature = extract_feature(models, se_img2)  # 出来的是二维 图片序号 另一个是具体的内容
# se_tags = load_object_from_zip('tags')
# # 取10:20条文本特征
# se_tag_feature = se_tags[10:20]
# se_tags_feature = torch.from_numpy(se_tag_feature).float()
# se_match = torch.cat((se_img_feature, se_tags_feature), 1)
# secret_match = np.array(se_match)  # tensor转numpy
# # -------------------------------------------------------------------------------------
# print(se_match)
# 制作secret positive dataset label原为1，打成0
# se_y = np.zeros(10, dtype=int)
# R_sey = np.ones(10, dtype=int)
# R_y = torch.from_numpy(R_sey)

	# 	a = img_feature[i[0]].view(1, len(img_feature[i[0]]))  # [1, 100352]
	# 	print(a.shape)
	# 	b = tags_feature[i[1]].view(1, len(tags_feature[i[1]]))  # [1, 1386]
	# 	print(b.shape)
	# 	c = torch.cat((a, b),1)
	# 	d = c.clone
	# 	print(c)   # [1, 101738]
	# 	print(d)
	#
	# print(c)
		# mixed_data.append([img_feature[i[0]], tags_feature[i[1]]])



	# torch.cat((image_feature[i[0]], tags_feature[i[1]]), 1)

		# 要加y，unlabel 设计

		# 设置超参数 30-40% ？ unlabel的多一些 取


		# mixed_data = torch.cat((image_feature[i[0]], tags_feature[i[1]]),1)
		# mix.concat(mixed_data)
	# print(mixed_data)
	# print(len(mixed_data))


	# # 给对齐交集打正例标签1
	# Y = np.ones(10)
	#
	# # mixed数据集打unlabeled data标签0
	# # Y = np.zeros(100)
	#
	# # SPY算法
	# s = spies(XGBClassifier(), XGBClassifier())
	# Y = Y.astype(int)
	# s.fit(X=match, y=Y)
	# # print(s.predict(uX))
	# out = s.predict(match)  # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
	# out_prob = s.predict_proba(match)
	# # out2 = s.predict(mixed_data)
	# data_count = collections.Counter(out)
	# print(data_count)
	# print(out)
	# print(out_prob)



	# data_count2 = collections.Counter(out2)
	# print(data_count2)
	# print(out2)

	# 正例集合=交集内元素。正例：mix=1：2
	# m_images,m_tags = torch.from_numpy(images[:50]).float(), torch.from_numpy(tags[:50]).float()
	# mixed_images, mixed_tags = torch.from_numpy(images[50:60]).float(), torch.from_numpy(tags[50:60]).float()
	#
	#
	# match_input_train = DataLoader(dataset=TensorDataset(m_images, m_tags), batch_size=4, shuffle=False)

	# 重新设计shuffle方法
	# mixed_match_input_train = DataLoader(dataset=TensorDataset(mixed_images, mixed_tags), batch_size=4, shuffle=True)
	#
	# # 序列化
	# print('loading and splitting data finish')
	#
	# # 给交集打正例标签1
	# Y = np.ones(50)
	# # 连接
	# Y = np.concatenate((Y, np.zeros(50)))
	#
	# model = CNN()
	# # 设置占位符
	# match_image_feature = torch.tensor([np.zeros(1000), np.zeros(1000)])
	# for epoch in range(1):
	# 	# 一批特征
	# 	print(epoch)
	# 	for batchidx, (x, y) in enumerate(match_input_train):
	# 		print(batchidx)
	# 		match_image_feature = torch.cat((match_image_feature, model(x)), 0)
	# 		print(match_image_feature.shape)
	# X = torch.cat((match_image_feature[2:], m_tags), 1)
	#
	# mixed_match_image_feature = torch.tensor([np.zeros(1000), np.zeros(1000)])
	# for epoch in range(1):
	# 	# 一批特征
	# 	print(epoch)
	# 	for batchidx, (x, y) in enumerate(mixed_match_input_train):
	# 		print(batchidx)
	# 		mixed_match_image_feature = torch.cat((mixed_match_image_feature, model(x)), 0)
	# 		print(mixed_match_image_feature.shape)
	# uX = torch.cat((mixed_match_image_feature[2:], mixed_tags), 1)
	#
	# X = X.detach().numpy()
	# uX = uX.detach().numpy()
	# XX = np.concatenate((X, uX))
	#
	#
