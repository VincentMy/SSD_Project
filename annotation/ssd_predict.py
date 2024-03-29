import tensorflow as tf
import numpy as np
from cv2 import cv2

class ssd(object):
    def __init__(self):
        #这个表示的是原始图像为300时，经过卷积后的6个特征图的尺度
        self.feature_map_size = [[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]]
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]
        self.feature_layers = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
        self.img_size = (300,300)
        self.num_classes = 21
        #此处表示的是6个特征图中，每个特征图所生成的anchor的数量
        self.boxes_len = [4,6,6,6,4,4]
        #只有block4层需要进行正则化，因为数值比较大
        self.isL2norm = [True,False,False,False,False,False]
        #表示每一层中anchor的最小和最大尺寸
        self.anchor_sizes = [[21., 45.], [45., 99.], [99., 153.],[153., 207.],[207., 261.], [261., 315.]]
        #表示每一层中生成长方形的anchor框，所需要的比值数据
        self.anchor_ratios = [[2, .5], [2, .5, 3, 1. / 3], [2, .5, 3, 1. / 3],
                         [2, .5, 3, 1. / 3], [2, .5], [2, .5]]
        #表示300缩放到每一层后的缩放倍数
        self.anchor_steps = [8, 16, 32, 64, 100, 300]
        self.prior_scaling = [0.1, 0.1, 0.2, 0.2]#特征图先验框缩放比例
        #每层总共所生成的anchor的数量
        self.n_boxes = [5776,2166,600,150,36,4]
        self.threshold = 0.2
    def l2norm(self,x,trainable=True,scope='L2Normalization'):
        n_channels = x.get_shape().as_list()[-1] #通道数
        l2_norm = tf.nn.l2_normalize(x,axis=[3],epsilon=1e-12) #只对每个像素点在channels上做归一化
        with tf.variable_scope(scope):
            gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                    trainable=trainable)
        return l2_norm * gamma
    #dilation 表示的是，是否有扩展，有点类似于上采样的情况，好比把3x3的特征图扩展成5x5的
    def conv2d(self,x,filter,k_size,stride=[1,1],padding='same',dilation=[1,1],activation=tf.nn.relu,scope='conv2d'):
        return tf.layers.conv2d(inputs=x,filters=filter,kernel_size=k_size,strides=stride,dilation_rate=dilation,padding=padding,name=scope,activation=activation)
    def max_pool2d(self,x,pool_size,stride,scope='max_pool2d'):
        return tf.layers.max_pooling2d(inputs=x,pool_size=pool_size,strides=stride,name=scope,padding='same')
    def pad2d(self,x,pad):
        return tf.pad(x,paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])
    def dropout(self,x,d_rate=0.5):
        return tf.layers.dropout(inputs=x,rate=d_rate)
    #根据给定的特征图输出相应的预测值
    def ssd_prediction(self,x,num_classes,box_num,isL2norm,scope='multibox'):
        # [-1] + [38,38] = [-1,38,38]
        reshape = [-1] +x.get_shape().as_list()[1:-1]#去除第一个和最后一个得到shape
        #print("reshape:",reshape.shape)
        with tf.variable_scope(scope):
            if isL2norm:
                x = self.l2norm(x)
            # 预测位置 -- 》 坐标和大小 回归 padding=same所以维度不变
            location_pred = self.conv2d(x,filter=box_num * 4,k_size=[3,3],activation=None,scope='conv_loc')
            #[-1,38,38,4,4]
            location_pred = tf.reshape(location_pred,reshape + [box_num,4])
            #预测类别 -- 》 分类softmax
            class_pred = self.conv2d(x,filter=box_num * num_classes,k_size=[3,3],activation=None,scope='conv_cls')
            #[-1,38,38,4,21]
            class_pred = tf.reshape(class_pred,reshape + [box_num,num_classes])
            return location_pred,class_pred
    def set_net(self):
        check_points = {}
        predictions = []
        locations = []
        x = tf.placeholder(dtype=tf.float32,shape=[None,300,300,3])
        with tf.variable_scope('ssd_300_vgg'):
            #b1 因为重复执行了两个相同的卷积，此处可以使用slim.repeat()操作
            net = self.conv2d(x,filter=64,k_size=[3,3],scope='conv1_1')
            net = self.conv2d(net,64,[3,3],scope='conv1_2')
            net = self.max_pool2d(net,pool_size=[2,2],stride=[2,2],scope='pool1')
            #b2
            net = self.conv2d(net,filter=128,k_size=[3,3],scope='conv2_1')
            net = self.conv2d(net,128,[3,3],scope='conv2_2')
            net = self.max_pool2d(net,pool_size=[2,2],stride=[2,2],scope='pool2')
            #b3
            net = self.conv2d(net,filter=256,k_size=[3,3],scope='conv3_1')
            net = self.conv2d(net,256,[3,3],scope='conv3_2')
            net = self.conv2d(net,256,[3,3],scope='conv3_3')
            net = self.max_pool2d(net,pool_size=[2,2],stride=[2,2],scope='pool3')
            #b4
            net = self.conv2d(net,filter=512,k_size=[3,3],scope='conv4_1')
            net = self.conv2d(net, 512, [3, 3], scope='conv4_2')
            net = self.conv2d(net, 512, [3, 3], scope='conv4_3')
            check_points['block4'] = net
            net = self.max_pool2d(net, pool_size=[2, 2], stride=[2, 2], scope='pool4')
            #b5
            net = self.conv2d(net, filter=512, k_size=[3, 3], scope='conv5_1')
            net = self.conv2d(net, 512, [3, 3], scope='conv5_2')
            net = self.conv2d(net, 512, [3, 3], scope='conv5_3')
            net = self.max_pool2d(net, pool_size=[3, 3], stride=[1, 1], scope='pool5')
            #b6
            net = self.conv2d(net,1024,[3,3],dilation=[6,6],scope='conv6')
            #b7
            net = self.conv2d(net,1024,[1,1],scope='conv7')
            check_points['block7'] = net
            #b8
            net = self.conv2d(net,256,[1,1],scope='conv8_1x1')
            net = self.conv2d(self.pad2d(net,1),512,[3,3],[2,2],scope='conv8_3x3',padding='valid')
            check_points['block8'] = net
            #b9
            net = self.conv2d(net, 128, [1, 1], scope='conv9_1x1')
            net = self.conv2d(self.pad2d(net,1), 256, [3, 3], [2, 2], scope='conv9_3x3', padding='valid')
            check_points['block9'] = net
            #b10
            net = self.conv2d(net, 128, [1, 1], scope='conv10_1x1')
            net = self.conv2d(net, 256, [3, 3], scope='conv10_3x3', padding='valid')
            check_points['block10'] = net
            #b11
            net = self.conv2d(net, 128, [1, 1], scope='conv11_1x1')
            net = self.conv2d(net, 256, [3, 3], scope='conv11_3x3', padding='valid')
            check_points['block11'] = net
            for i,j in enumerate(self.feature_layers):
                loc,cls = self.ssd_prediction(x = check_points[j],num_classes=self.num_classes,box_num = self.boxes_len[i],isL2norm = self.isL2norm[i],scope=j+'_box')
                #分类结果需要进行softmax
                predictions.append(tf.nn.softmax(cls))
                locations.append(loc)
            return locations,predictions,x
    #生成先验框
    #输出的是特征图每个坐标中心点(x,y)和4个或6个长宽
    def ssd_anchor_layer(self,img_size,feature_map_size,anchor_size,anchor_ratio,anchor_step,box_num,offset=0.5):
        #y-->[[0,0,0],[1,1,1],[2,2,2]]
        #x-->[[0,1,2],[0,1,2],[0,1,2]]
        #x,y表示的是特征图中的坐标点
        y,x = np.mgrid[0:feature_map_size[0],0:feature_map_size[1]]
        #其中加0.5表示的是移动到坐标的中心点，乘以anchor_step表示的是通过缩放比回归到原始图中，然后除以img_size的长或宽，进行归一化
        y = (y.astype(np.float32) + offset) * anchor_step /img_size[0]
        x = (x.astype(np.float32) + offset) * anchor_step /img_size[1]
        y = np.expand_dims(y,axis=-1)
        x = np.expand_dims(x,axis=-1)
        #计算两个长宽比为1的h和w
        h = np.zeros((box_num,),np.float32)
        w = np.zeros((box_num,),np.float32)
        #第一个正方形，长宽都为min_size
        h[0] = anchor_size[0]/img_size[0]
        w[0] = anchor_size[0]/img_size[0]
        #第二个正方形，长为(min_size*max_size)^(-2)
        h[1] = (anchor_size[0] * anchor_size[1]) ** 0.5/img_size[0]
        w[1] = (anchor_size[0] * anchor_size[1]) ** 0.5/img_size[1]
        #其余的长方形，根据ratio参数
        for i, j in enumerate(anchor_ratio):
            h[i+2] = anchor_size[0]/img_size[0]/(j ** 0.5)
            w[i+2] = anchor_size[0]/img_size[1] * (j ** 0.5)
        return y,x,h,w
    #解码网络,cx，cy为真是值，location为预测值
    #解码所得到的是真是框的坐标
    def ssd_decode(self,location,box,prior_scaling):
        y_a,x_a,h_a,w_a = box
        cx = location[:,:,:,:,0]*w_a*prior_scaling[0] + x_a
        cy = location[:,:,:,:,1]*h_a*prior_scaling[1] + y_a
        w = w_a * tf.exp(location[:, :, :, :, 2] * prior_scaling[2])
        h = h_a * tf.exp(location[:, :, :, :, 3] * prior_scaling[3])
        #合并最后一维,并把坐标转化为左上角和右下角的坐标
        bboxes = tf.stack([cy - h / 2.0, cx - w / 2.0, cy + h / 2.0, cx + w / 2.0], axis=-1)
        return bboxes
    #先验框筛选,只返回置信度score大于阈值的筛选数据
    #classes:每个框所对应的最大预测值的下标，即分类
    #scores：每个框所对应的最大预测值
    #anchor_box：筛选后的先验框，即最大置信度大于阈值的
    def choose_anchor_boxes(self,predictions,anchor_box,n_box):
        #原来anchor_box: (?, 38, 38, 4, 4)
        #把先验框转化为[n_box,4]的形式
        anchor_box = tf.reshape(anchor_box,[n_box,4])
        prediction = tf.reshape(predictions, [n_box, 21])
        #第一个预测值是背景的预测
        prediction = prediction[:,1:]
        #tf.argmax返回在第axis维度下的最大值坐标，+1是因为第一个是背景的预测值，已去掉
        classes = tf.argmax(prediction, axis=1) + 1
        #表示返回的是预测值的最大值，上面的为索引，这个为具体的数值
        scores = tf.reduce_max(prediction,axis=1)
        filter_mask = scores > self.threshold
        #tf.boolean_mask(tensor,mask) 表示的是根据mask中boolean值返回tensor中过的值，值返回为true的具体的数值，不是下标
        classes = tf.boolean_mask(classes, filter_mask)
        scores = tf.boolean_mask(scores, filter_mask)
        anchor_box = tf.boolean_mask(anchor_box, filter_mask)
        return classes,scores,anchor_box
    #返回前top_k的最大值
    def bboxes_sort(self,classes,scores,bboxes,top_k=400):
        #np.argsort(s)表似输出s中从小到大的数值对应的索引
        idxes = np.argsort(-scores)
        #输出前top_k的最大值
        classes = classes[idxes][:top_k]
        scores = scores[idxes][:top_k]
        bboxes = bboxes[idxes][:top_k]
        return classes, scores, bboxes
        # 计算IOU
    def bboxes_iou(self,bboxes1, bboxes2):
        bboxes1 = np.transpose(bboxes1)
        bboxes2 = np.transpose(bboxes2)

        # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
        int_ymin = np.maximum(bboxes1[0], bboxes2[0])
        int_xmin = np.maximum(bboxes1[1], bboxes2[1])
        int_ymax = np.minimum(bboxes1[2], bboxes2[2])
        int_xmax = np.minimum(bboxes1[3], bboxes2[3])

        # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
        int_h = np.maximum(int_ymax - int_ymin, 0.)
        int_w = np.maximum(int_xmax - int_xmin, 0.)

        # 计算IOU
        int_vol = int_h * int_w  # 交集面积
        vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])  # bboxes1面积
        vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])  # bboxes2面积
        iou = int_vol / (vol1 + vol2 - int_vol)  # IOU=交集/并集
        return iou
    # NMS
    def bboxes_nms(self,classes, scores, bboxes, nms_threshold=0.5):
        #生成一个scores.shape大小的矩阵，其中的值都为True
        keep_bboxes = np.ones(scores.shape, dtype=np.bool)
        for i in range(scores.size - 1):
            if keep_bboxes[i]:
                overlap = self.bboxes_iou(bboxes[i], bboxes[(i + 1):])
                keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i + 1):] != classes[i])
                keep_bboxes[(i + 1):] = np.logical_and(keep_bboxes[(i + 1):], keep_overlap)
        idxes = np.where(keep_bboxes)
        return classes[idxes], scores[idxes], bboxes[idxes]
    def handle_img(self,img_path):
        means = np.array((123.,117.,104.))
        self.img = cv2.imread(img_path)
        img = np.expand_dims(cv2.resize(cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB) - means,self.img_size),axis=0)
        return img
    def draw_rectangle(self,img,classes,scores,bboxes,colors,thickness=2):
        shape = img.shape
        #遍历所有的框，即最后预测完图片中有几个类，就有几个框
        for i in range(bboxes.shape[0]):
            bbox=bboxes[i]
            #左上角坐标
            p1 = (int(bbox[0] * shape[0]),int(bbox[1] * shape[1]))
            #右下角坐标
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            #p1[::-1]表示倒叙输出,thickness表示所画的线的宽度
            cv2.rectangle(img,p1[::-1],p2[::-1],colors[0],thickness)
            s = '%s/%.3f' % (self.classes[classes[i] - 1],scores[i])
            p1 = (p1[0] -5, p1[1])
            #添加文字，参数依次为：图片、添加的文字、左上角坐标、字体、字体大小、颜色、字体粗细
            cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, colors[1], 1)
        cv2.namedWindow("img",0)
        cv2.resizeWindow("img", 640, 480)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_this(self,locations,predictions):

        layers_anchors = []
        classes_list = []
        scores_list = []
        bboxes_list = []
        for i, s in enumerate(self.feature_map_size):
            anchor_bboxes = self.ssd_anchor_layer(self.img_size, s,
                                                  self.anchor_sizes[i],
                                                  self.anchor_ratios[i],
                                                  self.anchor_steps[i],
                                                  self.boxes_len[i])
            layers_anchors.append(anchor_bboxes)
        for i in range(len(predictions)):
            d_box = self.ssd_decode(locations[i], layers_anchors[i], self.prior_scaling)
            cls, sco, box = self.choose_anchor_boxes(predictions[i], d_box, self.n_boxes[i])
            classes_list.append(cls)
            scores_list.append(sco)
            bboxes_list.append(box)
        classes = tf.concat(classes_list, axis=0)
        scores = tf.concat(scores_list, axis=0)
        bboxes = tf.concat(bboxes_list, axis=0)
        return classes,scores,bboxes
if __name__ == '__main__':
    sd = ssd()
    locations, predictions, x = sd.set_net()
    classes, scores, bboxes = sd.run_this(locations, predictions)
    sess = tf.Session()
    ckpt_filename = 'ssd_vgg_300_weights.ckpt'
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)
    img = sd.handle_img('person.jpg')

    rclasses, rscores, rbboxes = sess.run([classes, scores, bboxes], feed_dict={x: img})
    rclasses, rscores, rbboxes = sd.bboxes_sort(rclasses, rscores, rbboxes)

    rclasses, rscores, rbboxes = sd.bboxes_nms(rclasses, rscores, rbboxes)

    sd.draw_rectangle(sd.img,rclasses,rscores,rbboxes,[[0,0,255],[255,0,0]])


















        