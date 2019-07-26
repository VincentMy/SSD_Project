import os
import sys
import random
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
from pascalvoc_common import VOC_LABELS
from dataset_utils import int64_feature, float_feature, bytes_feature

DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

def process_image(directory,name):
    """
    将图片数据存储为bytes,
    directory:voc文件夹
    name:图片名
    return:需要写入tfr的数据
    """
    #Read the image file
    #DIRECTORY_IMAGES = 'JPEGImages'
    filename = directory+DIRECTORY_IMAGES+name+'.jpg'
    """
    下面的函数表示对图片进行读取
    tf.gfile.GFile(filename,mode)表示获取文本操作句柄，类似于python提供的文本操作open()函数，filename是要打开的文件名，mode是以何种方式读写，将会返回一个
    文本操作句柄。
    tf.gfile.FastGFile(filename,mode)函数与GFile的差别仅仅在于“无阻塞”，即该函数会无阻塞以较快的方式获取文本操作句柄
    """
    image_data = tf.gfile.GFile(filename,'rb').read()
    #Read the XML annotation file
    filename = os.path.join(directory,DIRECTORY_ANNOTATIONS,name+'.xml')
    #获取.xml文件的树形结构
    tree = ET.parse(filename)
    root = tree.getroot()
    #Image shape,获取.xml中图片的长宽和通道数
    size = root.find('size')
    #图片的长宽和通道数
    shape = [int(size.find('height').text),
    int(size.find('width').text),
    int(size.find('depth').text)]
    #Find annotations
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    #获取root下面的所有object
    for obj in root.findall('object'):
        label = obj.find('name').text
        #获取该类别对应的数值
        labels.append(int(VOC_LABELS[label][0]))
        #[b'dog', b'person']
        labels_text.append(label.encode('ascii'))
        #difficult表示目标是否难以识别(0表示容易识别)
        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        #truncated 表示是否被截断(0表示完整)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)
        bbox = obj.find('bndbox')
        #获取gt的位置，并进行归一化
        bboxes.append((float(bbox.find('ymin').text)/shape[0],
        float(bbox.find('xmin').text)/shape[1],
        float(bbox.find('ymax').text)/shape[0],
        float(bbox.find('xmax').text)/shape[1]))
        #返回读取的图片、图片大小、每个Object对应的--》(图片的gt，图片的数字化标签，图片的文字化标签，是否难以读取，是否被截断)
        return image_data,shape,bboxes,labels,labels_text,difficult,truncated     

def convert_to_example(image_data,labels,labels_text,bboxes,shape,difficult,truncated):
    """
    Build an Example proto for an image example

    Args:
    image_data:string,JPEG encoding of RGB image;
    labels:list of integers ,identifier for the ground truth
    labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
     Example proto
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        [l.append(point) for l,point in zip([ymin,xmin,ymax,xmax],b)]
    image_format = b'JPEG'
    #Example protocol对象就是对图片加上bbox等位置信息的一种封装
    #tf.train.Example主要是将数据处理成二进制方面，为了提升IO效率和方便管理数据
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height':int64_feature(shape[0]),
        'image/width':int64_feature(shape[1]),
        'image/channels':int64_feature(shape[2]),
        'image/shape':int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),  # 图像编码格式
        'image/encoded': bytes_feature(image_data)}))  # 二进制图像数据
    return example


def add_to_tfrecord(dataset_dir,name,tfrecord_writer):
    """
    Loads data from image and annotations files and add them to a TFRecord
    Args:
    dataset_dir:Dataset directory;
    name:Image name to add to the TFRecord;
    """
    #由文件名读取数据
    image_data,shape,bboxes,labels,labels_text,difficult,truncated = process_image(dataset_dir,name)
    example = convert_to_example(image_data,labels,labels_text,bboxes,shape,difficult,truncated)

    tfrecord_writer.writer(example.SerializeToString())



def get_output_filename(output_dir,name,idx):
    return '%s%s_%03d.tfrecord'%(output_dir,name,idx)

# dataset_dir:D:/Document/data/VOC/VOC_data/VOCdevkit/VOC2007
def run(dataset_dir,output_dir,name='voc_train',shuffling=False):
    """
    Runs the conversion operation
    Args:
    dataset_dir: The dataset directory where the dataset is stored
    output_dir:Output directory
    """
    #tf.gfile.Exists()表示判断目录或文件是否存在
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)
    # Dataset filenames， and shuffling
    #annotations的路径
    path = os.path.join(dataset_dir,DIRECTORY_ANNOTATIONS)
    #listdir(path)列出path下的所有文件,filenames表示所有文件的名字,list类型
    filenames = sorted(os.listdir(path))
    print(type(filenames))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)
    #process dataset files
    i = 0
    fidx = 0
    while i < len(filenames): #循环文件名
        #open new TFRecord file,例如： ./TFOutput/voc_train_000.tfrecord
        tf_filename = get_output_filename(output_dir,name,fidx) #获取输出文件名
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            #一个文件200张图片
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                #可以看成是print，做输出操作
                sys.stdout.write('\r>>Converting image %d/%d' %(i+1,len(filenames)))
                sys.stdout.flush()
                filename = filenames[i]
                img_name =filename[:,-4]
                add_to_tfrecord(dataset_dir,img_name,tfrecord_writer)
                i += 1
                j += 1
                fidx += 1
    print('\nFinished converting the Pascal VOC dataset!')


    
if __name__ == "__main__":
    dataset_dir = 'D:/Document/data/VOC/VOC_data/VOCdevkit/VOC2007/'
    output_dir = './TFOutput/'
    run(dataset_dir,output_dir)
   # process_image(dataset_dir,'000001')
