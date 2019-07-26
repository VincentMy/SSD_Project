import numpy as np
import tensorflow as tf
#import tf_extended as tfe
import math
slim = tf.contrib.slim
#对每一层的feature map 的预测框进行处理，去掉不满足要求的预测框(即设为0)，同时对满足要求的预测框找出与真实框的对应关系
#bbox表示的是真实框的位置，anchors_layers表示的是某一层中所有的anchors框的坐标信息
def tf_ssd_bboxes_encode_layer(labels,bboxes,anchors_layer,num_classes,no_annotation_label,ignore_threshold=0.5,prior_scaling=[0.1,0.1,0.2,0.2],dtype=tf.float32):
    yref,xref,href,wref = anchors_layer  #固定生成的anchor的中心坐标及w,h等
    #print(yref.shape,xref.shape,href.shape,wref.shape)#(38, 38, 1) (38, 38, 1) (4,) (4,)
    #计算anchor的左上角和右下角坐标，最左上角坐标位(0,0),最右下角坐标位(n,n)
    ymin = yref - href/2  
    xmin = xref - wref/2
    ymax = yref + href/2
    xmax = xref + wref/2
    vol_anchors = (xmax-xmin) * (ymax -ymin) #anchor的面积,表示的是所有anchor的面积,shape为(38,38,4)
    #print("vol_anchors:",vol_anchors.shape)
    shape = (yref.shape[0],yref.shape[1],href.size) #对于第一个特征图，shape=(38,38,4)；第二个特征图的shape=(19,19,6)……
    #初始化每个特征图上的点对应的各个box所属标签维度,如：38x38x4
    feat_labels = tf.zeros(shape,dtype=tf.int64)
    #初始化每个特征图上的点对应的各个box所属目标的得分维度，如：38x38x4
    feat_scores = tf.zeros(shape,dtype=dtype)
    #初始化每个预测框四个点坐标(存放历史数据)
    feat_ymin = tf.zeros(shape,dtype=dtype)
    feat_xmin = tf.zeros(shape,dtype=dtype)
    feat_ymax = tf.ones(shape,dtype=dtype)
    feat_xmax = tf.ones(shape,dtype=dtype)
    #计算预测框与真实框的IOU，box为真实框坐标
    def jaccard_with_anchors(bbox):
        int_ymin = tf.maximum(ymin,bbox[0])
        int_xmin = tf.maximum(xmin,bbox[1])
        int_ymax = tf.minimum(ymax,bbox[2])
        int_xmax = tf.minimum(xmax,bbox[3])
        h = tf.maximum(int_ymax - int_ymin,0)
        w = tf.maximum(int_xmax - int_xmin,0)
        #计算交集的面积
        inter_vol = h * w
        #计算并集的面积
        union_vol = vol_anchors -inter_vol + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol,union_vol)
        return jaccard
    #计算某个参考框包含真实框的得分情况
    def intersection_with_anchors(bbox):
        int_ymin = tf.maximum(ymin,bbox[0])
        int_xmin = tf.maximum(xmin,bbox[1])
        int_ymax = tf.minimum(ymax,bbox[2])
        int_xmax = tf.minimum(xmax,bbox[3])
        h = tf.maximum(int_ymax - int_ymin,0)
        w = tf.maximum(int_xmax - int_xmin,0)
        #计算交集的面积
        inter_vol = h * w
        #将重叠区域面积除以参考框面积作为该参考框得分
        scores = tf.div(inter_vol,vol_anchors)
        return scores
    #下面是个循环，condition是循环条件，满足该条件进行循环，body表示的是循环体
    def condition(i,feat_labels,feat_scores,feat_ymin,feat_xmin,feat_ymax,feat_xmax):
        #当i的值小于labels大小时，条件成立，返回True
        r = tf.less(i,tf.shape(labels))
        return r[0]
    def body(i,feat_labels,feat_scores,feat_ymin,feat_xmin,feat_ymax,feat_xmax):
        #计算bbox与anchor的IOU值
        label = labels[i]
        bbox = bboxes[i]
        #计算所有的anchor与该bbox的IOU的值
        jaccard = jaccard_with_anchors(bbox)
        #当jaccard大于feat_scores时返回True,否则返回False,返回的个数例如：(38,38,4)
        mask = tf.greater(jaccard,feat_scores)
        # 逻辑与
        mask = tf.logical_and(mask,feat_scores > -0.5)
        mask = tf.logical_and(mask,label < num_classes) ##label 满足<21
        imask = tf.cast(mask,tf.int64)
        fmask = tf.cast(mask,dtype)
        # 更新那些满足要求的预测框，使他们类别，四个点的坐标位置和置信度分别为真实框的值，否则为0
        #当imask为1，即符合当前标签，定义为当前label，当imask为0，即不符合当前标签时，保持历史lable状态，即feat_labels
        feat_labels = imask * label + (1-imask) * feat_labels
        #mask为true的部分，用jaccard中对应位置的元素替换，为false的部分，用feat_scores中对应位置的元素替换
        feat_scores = tf.where(mask,jaccard,feat_scores)
        #对于交并比大于阈值的，其坐标点就为gt的坐标点，交并比小于阈值的，坐标点就为历史坐标点,如果某个框所有类别都不属于，这样其坐标值都为0
        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax
        return [i+1,feat_labels,feat_scores,feat_ymin,feat_xmin,feat_ymax,feat_xmax]

    i = 0
    [i,feat_labels,feat_scores,feat_ymin,feat_xmin,feat_ymax,feat_xmax] = tf.while_loop(condition,body,[i,feat_labels,feat_scores,feat_ymin,feat_xmin,feat_ymax,feat_xmax])
    #转换成中心点和长宽,此处的feat相关值，是对应的某一特征图所有anchor所对应的坐标值，如果某一个anchor属于某个label,则其feat坐标即为该label所对应的gt坐标
    #每个anchor所对应的gt的坐标
    feat_cy = (feat_ymax + feat_ymin) / 2
    feat_cx = (feat_xmax + feat_xmin) / 2
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    #对坐标进行编码
    #lcx=(bcx−dcx)/dw, lcy=(bcy−dcy)/dh
    #lw=log(bw/dw), lh=log(bh/dh)
    #其中feat_?表示的是gt的坐标，?ref表示的是anchor的坐标
    feat_cy = (feat_cy - yref)/href/prior_scaling[0]
    feat_cx = (feat_cx - xref)/wref/prior_scaling[1]
    feat_h = tf.log(feat_h/href)/prior_scaling[2]
    feat_w = tf.log(feat_w/wref)/prior_scaling[3]
    #tf.stack(?,axis=-1)表示在最后维中把数据合并
    feat_localizations = tf.stack([feat_cx,feat_cy,feat_w,feat_h],axis=-1)
    #返回每个anchor对饮的类别标签，坐标位置，交并比得分
    return feat_labels,feat_localizations,feat_scores

#logits :(list of) predictions logits Tensors ;每一层logits输出未经过softmax,类别预测值
#localisations:预测的框
#前缀为g的表示gt,gscores是每个先验框与gt的IOU的值
def ssd_losses(logits,localisations,gclasses,glocalisations,gscores,match_threshold=0.5,negative_ratio=3.,alpha=1,label_smoothing=0.,scope=None):
    with tf.name_scope(scope,'ssd_losses'):
        #提取类别数和batch_size,liguts的第一维应该是batch
        lshape = tf.shape(logits[0],5) #tensor_shape 函数可以取代
        num_classes = lshape[-1]
        batch_size = lshape[0]
        #Flatten out all vectors
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        #按照ssd特征层循环,把特征图中所有的先验框的相关信息进行reshape
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i],[-1,num_classes]))
            fgclasses.append(tf.reshape(gclasses[i],[-1]))
            fgscores.append(tf.reshape(gscores[i],[-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        #其中axis=0表示按照第0维合并，即按照batch合并
        logits = tf.concat(flogits,axis=0)      # 全部的搜索框，表示全部搜索框的21个类别的输出，
        gclasses = tf.concat(fgclasses,axis=0)   # 全部的搜索框，真实的类别数字
        gscores = tf.concat(fgscores, axis=0)    # 全部的搜索框，和真实框的IOU
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        """[<tf.Tensor 'ssd_losses/concat:0' shape=(279424, 21) dtype=float32>,
            <tf.Tensor 'ssd_losses/concat_1:0' shape=(279424,) dtype=int64>,
            <tf.Tensor 'ssd_losses/concat_2:0' shape=(279424,) dtype=float32>,
            <tf.Tensor 'ssd_losses/concat_3:0' shape=(279424, 4) dtype=float32>,
            <tf.Tensor 'ssd_losses/concat_4:0' shape=(279424, 4) dtype=float32>]
        """
        dtype = logits.dtype
        pmask = gscores > match_threshold #(全部搜索框数目，21)，类别搜索框和真实框IOU大于阈值
        fpmask = tf.cast(pmask,dtype) #浮点型前景掩码(前景假定为含有对象的IOU足够的搜索框标号)
        n_positives = tf.reduce_sum(fpmask) #前景总数

        #Hard negative mining ....
        no_classes = tf.cast(pmask,tf.int32)
        #此时每一行的21个数转化为概率
        predictions = slim.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask),gscores > -0.5) #IOU达不到阈值的类别搜索框位置记1
        fnmask = tf.cast(nmask,dtype)
        #nmask为Ture时表示背景，取predictions的第0个概率值，当nmask为False时，表示前景有值，取值0
        nvalues = tf.where(nmask,predictions[:,0],1- fnmask)
        #[-1]表示转换成1维，具体维度不确定，所以填-1
        nvalues_flat = tf.reshape(nvalues,[-1])
        #在nmask中剔除n_neg个最不可能背景点（对应class0概率最低）
        #计算标签为背景色的总数，并转化为int32值
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask),tf.int32)
        #3 x 前景掩码数量 + batch_size
        n_neg = tf.cast(negative_ratio * n_positives,tf.int32) +batch_size
        #选两者中最小的
        n_neg = tf.minimum(n_neg,max_neg_entries)
        #最不可能为背景的n_neg个点
        val,idxes = tf.nn.top_k(-nvalues_flat,k=n_neg)
        max_hard_pred = -val[-1]
        #不是前景，又最不像背景的n_neg个点
        nmask = tf.logical_and(nmask,nvalues < max_hard_pred)
        fnmask = tf.cast(nmask,dtype)
    with tf.name_scope('cross_entropy_pos'):
        loss = tf.nn.sparse_softmax_cross_entropy_wwith_logits(logits=logits,labels,gclasses) #gclasses为0-20
        loss = tf.div(tf.reduce_sum(loss * fpmask,batch_size,name='value'))
        tf.losses.add_loss(loss)
    with tf.name_scope('cross_entropy_neg'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=no_classes)  # {0,1}
        loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
        tf.losses.add_loss(loss)
    # Add localization loss: smooth L1, L2, ...
    with tf.name_scope('localization'):
        # Weights Tensor: positive mask + random negative.
        weights = tf.expand_dims(alpha * fpmask, axis=-1)
        loss = custom_layers.abs_smooth(localisations - glocalisations)
        loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
        tf.losses.add_loss(loss)
    







    








    
if __name__ == "__main__":
    anchors_layer=[]
    anchor_xy = np.random.rand(38,38,2)
    anchor_wh = np.random.rand(4,2)
    yref = anchor_xy[:,:,0].reshape(38,38,1)
    xref = anchor_xy[:,:,1].reshape(38,38,1)
    href = anchor_wh[:,0] + 1
    wref = anchor_wh[:,1] + 1
    anchors_layer.append(yref)
    anchors_layer.append(xref)
    anchors_layer.append(href)
    anchors_layer.append(wref)
    #print(anchors_layer)
    tf_ssd_bboxes_encode_layer(None,None,anchors_layer,None,None,ignore_threshold=0.5,prior_scaling=[0.1,0.1,0.2,0.2],dtype=tf.float32)
   
    