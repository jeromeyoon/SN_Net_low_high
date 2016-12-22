import os,time,pdb,argparse,threading
from glob import glob
import numpy as np
from numpy import inf
import tensorflow as tf
from ops import *
from utils import *
from random import shuffle
from network import networks
from normal import norm_
import scipy.ndimage
class DCGAN(object):
    def __init__(self, sess, image_size=108, is_train=True,is_crop=True,\
                 batch_size=32,num_block=1,ir_image_shape=[256, 256,1], normal_image_shape=[256, 256, 3],\
	         light_shape=[64,64,3],df_dim=64,dataset_name='default',checkpoint_dir=None):


        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.normal_image_shape = normal_image_shape
        self.ir_image_shape = ir_image_shape
        self.df_dim = df_dim
        self.dataset_name = dataset_name
	self.num_block = num_block
        self.checkpoint_dir = checkpoint_dir
	self.low_ir_image_shape=[64,64,1]
	self.high_ir_image_shape=[256,256,1]
	self.low_normal_image_shape=[64,64,3]
	self.high_normal_image_shape=[256,256,3]
	self.use_queue = True
	self.mean_nir = -0.3313 #-1~1
	self.dropout =0.7
	self.loss ='L2'
	self.pair = True
	self.build_model()
	
    def build_model(self):
	
	if not self.use_queue:

        	self.low_ir_images = tf.placeholder(tf.float32, [self.batch_size] + self.low_ir_image_shape,
                                    name='low_ir_images')
        	self.low_normal_images = tf.placeholder(tf.float32, [self.batch_size] + self.low_normal_image_shape,
                                    name='low_normal_images')
	else:
		print ' using queue loading'
		self.low_ir_image_single = tf.placeholder(tf.float32,shape=self.low_ir_image_shape)
		self.high_ir_image_single = tf.placeholder(tf.float32,shape=self.high_ir_image_shape)
		self.low_normal_image_single = tf.placeholder(tf.float32,shape=self.low_normal_image_shape)
		self.high_normal_image_single = tf.placeholder(tf.float32,shape=self.high_normal_image_shape)
	
		q = tf.FIFOQueue(4000,[tf.float32, tf.float32,tf.float32,tf.float32],[[self.low_ir_image_shape[0],self.low_ir_image_shape[1],1],[self.high_ir_image_shape[0],self.high_ir_image_shape[1],1], [self.low_normal_image_shape[0],self.low_normal_image_shape[1],3],[self.high_normal_image_shape[0],self.high_normal_image_shape[1],3]])
		self.enqueue_op = q.enqueue([self.low_ir_image_single, self.high_ir_image_single,self.low_normal_image_single,self.high_normal_image_single])
		self.low_ir_images,self.high_ir_images,self.low_normal_images,self.high_normal_images = q.dequeue_many(self.batch_size)

	self.keep_prob = tf.placeholder(tf.float32)
        #self.low_normal_images = tf.placeholder(tf.float32, [self.batch_size] + self.low_normal_image_shape,name='low_ir_images')
	net  = networks(self.num_block,self.batch_size,self.df_dim)
	self.low_G = net.generator_low(self.low_ir_images)
        self.resized = tf.image.resize_bicubic(self.low_G,[256,256])
	self.inputs = tf.concat(3,[self.resized,self.high_ir_images])
	self.concat = tf.concat(3,[self.inputs,self.high_normal_images])
	self.cropped = tf.random_crop(self.concat ,[self.batch_size,64,64,7])
	self.cropped_input = tf.slice(self.cropped,[0,0,0,0],[self.batch_size,64,64,4])
	self.high_normal_images = tf.slice(self.cropped,[0,0,0,4],[self.batch_size,64,64,3])
	self.high_G = net.generator_high(self.cropped_input)
	
	################ Discriminator Loss ######################
	if self.pair:
	    self.low_D = net.discriminator_low(tf.concat(3,[self.low_normal_images,self.low_normal_images]),self.keep_prob)
	    self.low_D_  = net.discriminator_low(tf.concat(3,[self.low_G,self.low_normal_images]),self.keep_prob,reuse=True)
	    self.high_D = net.discriminator_high(tf.concat(3,[self.high_normal_images,self.high_normal_images]),self.keep_prob)
	    self.high_D_  = net.discriminator_high(tf.concat(3,[self.high_G,self.high_normal_images]),self.keep_prob,reuse=True)

	else:
	    self.low_D = net.discriminator_low(self.low_normal_images,self.keep_prob)
	    self.low_D_  = net.discriminator_low(self.low_G,self.keep_prob,reuse=True)
	    self.high_D = net.discriminator_high(self.high_normal_images,self.keep_prob)
	    self.high_D_  = net.discriminator_high(self.high_G,self.keep_prob,reuse=True)

	# generated surface normal
        #self.d_loss_real = binary_cross_entropy_with_logits(tf.pack(np.random.uniform(0.7,1.2,size=(self.batch_size,1)).astype(np.float32)), self.D)
        #self.d_loss_fake = binary_cross_entropy_with_logits(tf.pack(np.random.uniform(0.0,0.3,size=(self.batch_size,1)).astype(np.float32)), self.D_)
	#### Low resolution ####
	self.low_d_loss_real = binary_cross_entropy_with_logits(tf.random_uniform(self.low_D.get_shape(),minval=0.7,maxval=1.2,dtype=tf.float32,seed=0), self.low_D)
        self.low_d_loss_fake = binary_cross_entropy_with_logits(tf.random_uniform(self.low_D.get_shape(),minval=0.0,maxval=0.3,dtype=tf.float32,seed=0), self.low_D_)
        self.low_d_loss = self.low_d_loss_real + self.low_d_loss_fake

	#### High resolution ####
	self.high_d_loss_real = binary_cross_entropy_with_logits(tf.random_uniform(self.high_D.get_shape(),minval=0.7,maxval=1.2,dtype=tf.float32,seed=0), self.high_D)
        self.high_d_loss_fake = binary_cross_entropy_with_logits(tf.random_uniform(self.high_D.get_shape(),minval=0.0,maxval=0.3,dtype=tf.float32,seed=0), self.high_D_)
        self.high_d_loss = self.high_d_loss_real + self.high_d_loss_fake

	############# Generative loss #################
	if self.loss == 'L1':
            self.low_L_loss = tf.div(tf.reduce_sum(tf.abs(tf.sub(self.low_G,self.low_normal_images))),self.low_ir_image_shape[1]*self.low_ir_image_shape[2]*3)
            self.high_L_loss = tf.div(tf.reduce_sum(tf.abs(tf.sub(self.high_G,self.high_normal_images))),self.high_ir_image_shape[1]*self.high_ir_image_shape[2]*3)
	else:
            self.low_L_loss = tf.div(tf.reduce_sum(tf.square(tf.sub(self.low_G,self.low_normal_images))),self.low_ir_image_shape[1]*self.low_ir_image_shape[2]*3)
            self.high_L_loss = tf.div(tf.reduce_sum(tf.square(tf.sub(self.high_G,self.high_normal_images))),self.high_ir_image_shape[1]*self.high_ir_image_shape[2]*3)
	
        self.low_g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.low_D_), self.low_D_)
        self.low_gen_loss = self.low_g_loss + self.low_L_loss 

        self.high_g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.high_D_), self.high_D_)
	self.ang_loss = norm_(self.high_G,self.high_normal_images)
        self.high_gen_loss = self.high_g_loss + self.high_L_loss + self.ang_loss
 
	self.saver = tf.train.Saver(max_to_keep=10)
	t_vars = tf.trainable_variables()
	self.low_d_vars =[var for var in t_vars if 'low_d_' in var.name]
	self.low_g_vars =[var for var in t_vars if 'low_g_' in var.name]
	self.high_d_vars =[var for var in t_vars if 'high_d_' in var.name]
	self.high_g_vars =[var for var in t_vars if 'high_g_' in var.name]
    def train(self, config):
        #####Train DCGAN####

        global_step1 = tf.Variable(0,name='global_step1',trainable=False)
        global_step2 = tf.Variable(0,name='global_step2',trainable=False)
        global_step3 = tf.Variable(0,name='global_step3',trainable=False)
        global_step4 = tf.Variable(0,name='global_step4',trainable=False)

	g_lr = tf.train.exponential_decay(config.g_learning_rate,global_step1,1000,0.5,staircase=True)
	
	low_d_optim = tf.train.AdamOptimizer(config.d_learning_rate,beta1=config.beta1) \
                          .minimize(self.low_d_loss, global_step=global_step1,var_list=self.low_d_vars)
        low_g_optim = tf.train.AdamOptimizer(g_lr,beta1=config.beta1) \
                          .minimize(self.low_gen_loss, global_step=global_step2,var_list=self.low_g_vars)
	
	high_d_optim = tf.train.AdamOptimizer(config.d_learning_rate,beta1=config.beta1) \
                          .minimize(self.high_d_loss, global_step=global_step3,var_list=self.high_d_vars)
        high_g_optim = tf.train.AdamOptimizer(g_lr,beta1=config.beta1) \
                          .minimize(self.high_gen_loss, global_step=global_step4,var_list=self.high_g_vars)

	tf.initialize_all_variables().run()
	
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loda training and validation dataset path
        data_low = json.load(open("/research2/ECCV_journal/deconv/low_high/scale/low_traininput.json"))
        data_low_label = json.load(open("/research2/ECCV_journal/deconv/low_high/scale/low_traingt.json"))
        data_high = json.load(open("/research2/ECCV_journal/deconv/low_high/scale/high_traininput.json"))
        data_high_label = json.load(open("/research2/ECCV_journal/deconv/low_high/scale/high_traingt.json"))
        low_datalist =[''.join(data_low[idx]) for idx in xrange(0,len(data_low))]
        low_labellist =[''.join(data_low_label[idx]) for idx in xrange(0,len(data_low_label))]
        high_datalist =[''.join(data_high[idx]) for idx in xrange(0,len(data_high))]
        high_labellist =[''.join(data_high_label[idx]) for idx in xrange(0,len(data_high_label))]
	shuf = range(len(low_datalist))
	random.shuffle(shuf)
        list_val = [11,16,21,22,33,36,38,53,59,92]


	if self.use_queue:
	    # creat thread
	    coord = tf.train.Coordinator()
            num_thread =4
            for i in range(num_thread):
 	        t = threading.Thread(target=self.load_and_enqueue,args=(coord,low_datalist,low_labellist,high_datalist,high_labellist,shuf,i,num_thread))
	 	t.start()

	if self.use_queue:
	    for epoch in xrange(config.epoch):
	        #shuffle = np.random.permutation(range(len(data)))
	        batch_idxs = min(len(low_datalist), config.train_size)/config.batch_size
		sum_low_L = 0.0
		sum_high_L = 0.0
		sum_low_g =0.0
		sum_high_g =0.0
		sum_ang =0.0
		sum_low_d_real =0.0
		sum_low_d_fake =0.0
		if epoch ==0:
		    train_log = open(os.path.join("logs",'train_%s.log' %config.dataset),'w')
		else:
	    	    train_log = open(os.path.join("logs",'train_%s.log' %config.dataset),'aw')

		for idx in xrange(0,batch_idxs):
        	     start_time = time.time()
		     _,low_d_loss =self.sess.run([low_d_optim,self.low_d_loss],feed_dict={self.keep_prob:self.dropout})
		     _,low_g_loss,low_L_loss =self.sess.run([low_g_optim,self.low_g_loss,self.low_L_loss],feed_dict={self.keep_prob:self.dropout})
	
		     _,high_d_loss =self.sess.run([high_d_optim,self.high_d_loss],feed_dict={self.keep_prob:self.dropout})
		     _,high_g_loss,high_L_loss,ang_loss = self.sess.run([high_g_optim,self.high_g_loss,self.high_L_loss,self.ang_loss])
		     print("Epoch: [%2d] [%4d/%4d] time: %4.4f low_g_loss: %.6f low_L: %.6f low_d_loss:%.4f high_g_loss:%.6f high_L_loss: %.6f, ang_loss:%.6f" \
		     % (epoch, idx, batch_idxs,time.time() - start_time,low_g_loss,low_L_loss,low_d_loss,high_g_loss,high_L_loss,ang_loss))

		     sum_low_L += low_L_loss 	
		     sum_low_g += low_g_loss
		     sum_high_L += high_L_loss 	
		     sum_high_g += high_g_loss
		     sum_ang += ang_loss
		train_log.write('epoch %06d mean_low_g %.6f mean_low_L %.6f mean_high_g: %.6f mean_high_L: %.6f mean_ang_loss:%.6f\n' %(epoch,sum_low_g/(batch_idxs), sum_low_L/(batch_idxs),sum_high_g/batch_idxs,sum_high_L/batch_idxs,sum_ang/batch_idxs))
		train_log.close()
	        self.save(config.checkpoint_dir,global_step1)

	else:
	    for epoch in xrange(config.epoch):
	         # loda training and validation dataset path
	         shuffle_ = np.random.permutation(range(len(data)))
	         batch_idxs = min(len(data), config.train_size)/config.batch_size
		    
	         for idx in xrange(0, batch_idxs):
        	     start_time = time.time()
		     batch_files = shuffle_[idx*config.batch_size:(idx+1)*config.batch_size]
    		     batches = [get_image(datalist[batch_file],labellist[batch_file],self.image_size,np.random.randint(64,224-64),\
					np.random.randint(64,224-64), is_crop=self.is_crop) for batch_file in batch_files]

		     batches = np.array(batches).astype(np.float32)
		     batch_images = np.reshape(batches[:,:,:,0],[config.batch_size,64,64,1])
		     batchlabel_images = np.reshape(batches[:,:,:,1:],[config.batch_size,64,64,3])
		     
		     # Update Normal D network
		     _= self.sess.run([d_optim], feed_dict={self.ir_images: batch_images,self.normal_images:batchlabel_images })
		     self.writer.add_summary(summary_str, global_step.eval())

		     # Update NIR G network
		     _,g_loss,L1_loss = self.sess.run([g_optim,self.g_loss,self.L1_loss], feed_dict={ self.ir_images: batch_images,self.normal_images:batchlabel_images})
		     print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f L1_loss:%.4f" \
		     % (epoch, idx, batch_idxs,time.time() - start_time,low_g_loss,low_L1_loss,d_loss))
	         self.save(config.checkpoint_dir,global_step)
    
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name,self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

	    
    def load_and_enqueue(self,coord,low_file_list, low_label_list,high_file_list,high_label_list,shuf,idx=0,num_thread=1):
	count =0;
	length = len(low_file_list)
	rot=[0,90,180,270]
	while not coord.should_stop():
	    i = (count*num_thread + idx) % length;
	    r = random.randint(0,3)
            low_input_img = scipy.misc.imread(low_file_list[shuf[i]]).reshape([64,64,1]).astype(np.float32)
	    low_gt_img = scipy.misc.imread(low_label_list[shuf[i]]).reshape([64,64,3]).astype(np.float32)
            high_input_img = scipy.misc.imread(high_file_list[shuf[i]]).reshape([256,256,1]).astype(np.float32)
	    high_gt_img = scipy.misc.imread(high_label_list[shuf[i]]).reshape([256,256,3]).astype(np.float32)

	    low_input_img = low_input_img/127.5 -1.
	    low_gt_img = low_gt_img/127.5 -1.
	    high_input_img = high_input_img/127.5 -1.
	    high_gt_img = high_gt_img/127.5 -1.

	    low_input_img = scipy.ndimage.rotate(low_input_img,rot[r])
	    low_gt_img = scipy.ndimage.rotate(low_gt_img,rot[r])
	    high_input_img = scipy.ndimage.rotate(high_input_img,rot[r])
	    high_gt_img = scipy.ndimage.rotate(high_gt_img,rot[r])
            self.sess.run(self.enqueue_op,feed_dict={self.low_ir_image_single:low_input_img, self.high_ir_image_single:high_input_img,self.low_normal_image_single:low_gt_img,self.high_normal_image_single:high_gt_img})
	    count +=1
		
