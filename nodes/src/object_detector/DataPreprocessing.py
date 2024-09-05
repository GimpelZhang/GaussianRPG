import os
import shutil
import xml.etree.ElementTree as ET
import json
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = 0
image_id = 20200000000
annotation_id = 0

def addCatItem(name):
	global category_item_id
	category_item = dict()
	category_item['supercategory'] = 'none'
	category_item_id += 1
	category_item['id'] = category_item_id
	category_item['name'] = name
	coco['categories'].append(category_item)
	category_set[name] = category_item_id
	return category_item_id


def addImgItem(file_name, size):
	global image_id
	if file_name is None:
		raise Exception('Could not find filename tag in xml file.')
	if size['width'] is None:
		raise Exception('Could not find width tag in xml file.')
	if size['height'] is None:
		raise Exception('Could not find height tag in xml file.')
	image_id += 1
	image_item = dict()
	image_item['id'] = image_id
	image_item['file_name'] = file_name
	image_item['width'] = size['width']
	image_item['height'] = size['height']
	coco['images'].append(image_item)
	image_set.add(file_name)
	return image_id


def addAnnoItem(object_name, image_id, category_id, bbox):
	global annotation_id
	annotation_item = dict()
	annotation_item['segmentation'] = []
	seg = []
	# bbox[] is x,y,w,h
	# left_top
	seg.append(bbox[0])
	seg.append(bbox[1])
	# left_bottom
	seg.append(bbox[0])
	seg.append(bbox[1] + bbox[3])
	# right_bottom
	seg.append(bbox[0] + bbox[2])
	seg.append(bbox[1] + bbox[3])
	# right_top
	seg.append(bbox[0] + bbox[2])
	seg.append(bbox[1])

	annotation_item['segmentation'].append(seg)

	annotation_item['area'] = bbox[2] * bbox[3]
	annotation_item['iscrowd'] = 0
	annotation_item['ignore'] = 0
	annotation_item['image_id'] = image_id
	annotation_item['bbox'] = bbox
	annotation_item['category_id'] = category_id
	annotation_id += 1
	annotation_item['id'] = annotation_id
	coco['annotations'].append(annotation_item)


def parseXmlFiles(xml_path):
	for f in os.listdir(xml_path):
		if not f.endswith('.xml'):
			continue

		bndbox = dict()
		size = dict()
		current_image_id = None
		current_category_id = None
		file_name = None
		size['width'] = None
		size['height'] = None
		size['depth'] = None

		xml_file = os.path.join(xml_path, f)

		tree = ET.parse(xml_file)
		root = tree.getroot()
		if root.tag != 'annotation':
			raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

		# elem is <folder>, <filename>, <size>, <object>
		for elem in root:
			current_parent = elem.tag
			current_sub = None
			object_name = None

			if elem.tag == 'folder':
				continue

			if elem.tag == 'filename':
				file_name = elem.text
				if file_name in category_set:
					raise Exception('file_name duplicated')

			# add img item only after parse <size> tag
			elif current_image_id is None and file_name is not None and size['width'] is not None:
				if file_name not in image_set:
					current_image_id = addImgItem(file_name, size)
					#print('add image with {} and {}'.format(file_name, size))
				else:
					raise Exception('duplicated image: {}'.format(file_name))
				# subelem is <width>, <height>, <depth>, <name>, <bndbox>
			for subelem in elem:
				bndbox['xmin'] = None
				bndbox['xmax'] = None
				bndbox['ymin'] = None
				bndbox['ymax'] = None

				current_sub = subelem.tag
				if current_parent == 'object' and subelem.tag == 'name':
					object_name = subelem.text
					if object_name not in category_set:
						current_category_id = addCatItem(object_name)
					else:
						current_category_id = category_set[object_name]

				elif current_parent == 'size':
					if size[subelem.tag] is not None:
						raise Exception('xml structure broken at size tag.')
					size[subelem.tag] = int(subelem.text)

				# option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
				for option in subelem:
					if current_sub == 'bndbox':
						if bndbox[option.tag] is not None:
							raise Exception('xml structure corrupted at bndbox tag.')
						bndbox[option.tag] = int(option.text)

				# only after parse the <object> tag
				if bndbox['xmin'] is not None:
					if object_name is None:
						raise Exception('xml structure broken at bndbox tag')
					if current_image_id is None:
						raise Exception('xml structure broken at bndbox tag')
					if current_category_id is None:
						raise Exception('xml structure broken at bndbox tag')
					bbox = []
					# x
					bbox.append(bndbox['xmin'])
					# y
					bbox.append(bndbox['ymin'])
					# w
					bbox.append(bndbox['xmax'] - bndbox['xmin'])
					# h
					bbox.append(bndbox['ymax'] - bndbox['ymin'])
					# print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
					addAnnoItem(object_name, current_image_id, current_category_id, bbox)

	print('完成xml到jsion格式转换！')
	print('目标的种类及编号为：', coco['categories'])



def del_images(image_path, ann_path):
	'''
	该函数用于删除未标注的图片
	:param xml_path: xml标注文件夹绝对路径
	:param image_path: 图片文件夹绝对路径
	:return: 删除的图片列表
	'''
	xmls = os.listdir(ann_path)
	images = os.listdir(image_path)
	del_images_list = []
	xml_index = []
	for xml in xmls:
		xml_index.append(xml.split('.')[0])
	for image in images:
		image_index = image.split('.')[0]
		if image_index not in xml_index and os.path.isfile(os.path.join(image_path, image)):
			os.remove(os.path.join(image_path, image))
			print('删除:',image)
			del_images_list.append(image)
	print('完成删除未标注图片{0}张'.format(len(del_images_list  )))
	print('删除的图片名列表：', del_images)
	xmls = os.listdir(ann_path)
	images = os.listdir(image_path)
	print('图片张数：',len(images))
	print('标注文件数：', len(xmls))
	if len(images) != len(xmls):
		print('错误：图片与标注文件不对应！')
	else:
		print('图片与标注文件数相等')

	return del_images_list

def rename(image_path, ann_path):
	'''
	rename函数根据图像名找到对应的标注文件，然后将两个文件改成相同的数字名
	:param image_path:
	:param ann_path:
	:return:
	'''

	image_names = os.listdir(image_path)
	ann_names = os.listdir(ann_path)
	print('图片数量：{0}，标注文件数量{1}'.format(len(image_names), len(ann_names)))
	if len(image_names) != len(ann_names):
		print('图片与标注文件数量相等')
	i = 0
	for image_name in image_names:
		image_oldname = os.path.join(image_path, image_name)
		index = image_name.split('.')[0]
		ann_name = index + ".xml"
		ann_oldname = os.path.join(ann_path, ann_name)
		las = image_name.split('.')[1]

		im_newname = str(i) + '.' + las
		an_newname = str(i) + '.xml'
		image_newname = os.path.join(image_path, im_newname)
		ann_newname = os.path.join(ann_path, an_newname)
		i += 1
		os.rename(image_oldname, image_newname)
		os.rename(ann_oldname, ann_newname)

	print('完成图像与对应标注文件的重命名')

def split_data(image_path, ann_path, save_split_path, rate = 0.8):
	'''
	按比例rate将数据集划分为训练集和验证集，并检查图片与标注文件的对应性
	:param image_path: 已标注图片路径
	:param ann_path: 标注文件路径
	:param save_split_path: 划分数据集保存的路径
	:param rate: 划分给训练集的比例
	:return: 返回训练集和测试集的图片与标注文件路径
	'''
	if not os.path.exists(save_split_path):
		os.mkdir(save_split_path)
	else:
		shutil.rmtree(save_split_path)
		os.mkdir(save_split_path)


		ann_train_path = os.path.join(save_split_path,'ann_train/')
		ann_val_path = os.path.join(save_split_path,'ann_val/')
		image_train_path = os.path.join(save_split_path,'images_train/')
		image_val_path = os.path.join(save_split_path,'images_val/')
		# 创建文件夹
		os.mkdir(ann_train_path)
		os.mkdir(ann_val_path)
		os.mkdir(image_train_path)
		os.mkdir(image_val_path)
		print('清空文件夹')


	images_names = os.listdir(image_path)  # 取图片的原始路径
	images_number = len(images_names)
	ann_names = os.listdir(ann_path)
	ann_number = len(ann_names)

	if images_number != ann_number:
		print('错误：图片数与标注文件数不相等')
	# 自定义抽取训练图片的比例，比方说100张抽10张，那就是0.1
	sample_number = int(images_number * rate)  # 按照rate比例从文件夹中取一定数量图片

	for name in images_names[0:sample_number]:
		shutil.copy(image_path + name, image_train_path + name)
	for name in ann_names[0:sample_number]:
		shutil.copy(ann_path + name, ann_train_path + name)

	for name in images_names[sample_number:images_number+1]:
		shutil.copy(image_path + name, image_val_path + name)
	for name in ann_names[sample_number:images_number+1]:
		shutil.copy(ann_path + name, ann_val_path + name)

	print('完成训练集({0})与测试集（{1}）划分'.format(round(rate,1),round((1-rate), 1)))
	print('图片总数为{0},标注文件总数为{1}'.format(images_number, ann_number))
	print('{0} 张图片用于训练，{1} 张图片用于验证'.format(sample_number, images_number - sample_number))

	# 检验图片与标注的匹配关系
	image_train_names = os.listdir(image_train_path)
	ann_train_names = os.listdir(ann_train_path)
	count = 0
	for i in range(len(image_train_names)):
		if image_train_names[i].split('.')[0] != ann_train_names[i].split('.')[0]:
			print('{0} 图片与{1}标注文件不匹配'.format(image_train_names[i][0]+image_train_names[i][1], ann_train_names[i][ann_train_names[i][1]]))
			count +=1
	if count == 0:
		print('训练集所有图片与标注文件一一对应')
	else:
		print('训练集图片与标注文件不匹配数目：',count)

	image_val_names = os.listdir(image_val_path)
	ann_val_names = os.listdir(ann_val_path)
	c = 0
	for i in range(len(image_val_names)):
		if image_val_names[i].split('.')[0] != ann_val_names[i].split('.')[0]:
			print('{0} 图片与{1}标注文件不匹配'.format(image_val_names[i][0]+image_val_names[i][1], ann_val_names[i][ann_val_names[i][1]]))
			c +=1
	if count == 0:
		print('验证集所有图片与标注文件一一对应')
	else:
		print('验证集图片与标注文件不匹配数目：', c)

	return image_train_path, image_val_path, ann_train_path, ann_val_path

def voc2coco_json(image_path, ann_path, save_split_path,save_coco_path):
	del_iammges = del_images(image_path, ann_path)
	if not os.path.exists(save_coco_path):
		os.mkdir(save_coco_path)
	else:
		shutil.rmtree(save_coco_path)
		os.mkdir(save_coco_path)
	annotations_path = os.path.join(save_coco_path,'annotations/')

	train2017_path = os.path.join(save_coco_path,'train2017/')
	val2017_path = os.path.join(save_coco_path,'val2017/')
	os.mkdir(annotations_path)
	os.mkdir(train2017_path)
	os.mkdir(val2017_path)

	image_train_path, image_val_path, ann_train_path, ann_val_path=split_data(image_path, ann_path, save_split_path, rate = 0.8)
	json_file = [os.path.join(annotations_path, 'instances_train2017.json'), os.path.join(annotations_path, 'instances_val2017.json')]
	ann_path = [ann_train_path, ann_val_path]
	for i in range(len(ann_path)):
		parseXmlFiles(ann_path[i])
		json.dump(coco, open(json_file[i], 'w'))

	images_train = os.listdir(image_train_path)
	images_val = os.listdir(image_val_path)

	for name in images_train:
		shutil.copy(image_train_path+name, train2017_path+name)
	for name in images_val:
		shutil.copy(image_val_path+name, val2017_path+name)


	print('完成数据清洗、拆分、xlm到json转换')

def data_augment(image_path, save_image_path = None):
	#images = os.listdir(image_path)
	image_string = tf.io.read_file(image_path)
	image = tf.image.decode_jpeg(image_string, channels = 3)
	# 翻转图像(垂直和水平)
	flipped_h = tf.image.flip_left_right(image)
	flipped_v = tf.image.flip_up_down(image)

	bright_0 = tf.image.adjust_brightness(image, 0.2)
	bright_5 = tf.image.adjust_brightness(image, 0.5)
	bright_8 = tf.image.adjust_brightness(image, 0.6)
	bright_10 = tf.image.adjust_brightness(image, 0.8)

	grayscaled = tf.image.rgb_to_grayscale(image)

	saturated_3 = tf.image.adjust_saturation(image, 3)
	saturated_8 = tf.image.adjust_saturation(image, 8)

	#visualize(image, bright_0)
	#visualize(image, flipped_h)
	#visualize(image, flipped_v)
	# visualize(image, tf.squeeze(grayscaled))

	visualize(image, saturated_3)

def visualize(original, augmented):
	plt.figure(figsize = (20, 10))
	plt.subplot(1, 2, 1)
	plt.title("Original Picture", fontsize=50, fontweight='bold')
	# plt.axis("off")  # 关闭坐标轴显示
	#plt.imshow(original)

	plt.subplot(1, 2, 2)
	plt.title("saturation 3", fontsize=50, fontweight='bold')
	# plt.axis("off")  # 关闭坐标轴显示
	#plt.imshow(augmented)
	plt.xticks(fontsize = 30)
	plt.yticks(fontsize = 30)
	plt.tight_layout()
	plt.savefig('./saturation3.png')
	#plt.show()


if __name__ == '__main__':

	image_path = "D:/PythonFile/aicar/aicar标志物数据集/JPEGImages/"
	ann_path = "D:/PythonFile/aicar/aicar标志物数据集/Annotations/"
	save_coco_path ="D:/PythonFile/aicar/coco2017/"
	save_split_path = "D:/PythonFile/aicar/splitdata/"
	#voc2coco_json(image_path, ann_path, save_split_path, save_coco_path)
	data_augment("D:\PythonFile\shangqi\ObjectDistance\dog.png")

