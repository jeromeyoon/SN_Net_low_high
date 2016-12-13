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
                 batch_size=32,num_block=1,ir_image_shape=[64, 64,1], normal_image_shape=[64, 64, 3],\
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
	self.use_queue = True
	self.mean_nir = -0.3313 #-1~1
	self.dropout =0.7
	self.loss ='L2'
	self.pair = False
	self.build_model()
    def build_model(self):
	
	if not self.use_queue:

        	self.ir_images = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='ir_images')
        	self.normal_images = tf.placeholder(tf.float32, [self.batch_size] + self.normal_image_shape,
                                    name='normal_images')
	else:
		print ' using queue loading'
		self.ir_image_single = tf.placeholder(tf.float32,shape=self.ir_image_shape)
		self.normal_image_single = tf.placeholder(tf.float32,shape=self.normal_image_shape)
		q = tf.FIFOQueue(4000,[tf.float32,tf.float32],[[self.ir_image_shape[0],self.ir_image_shape[1],1],[self.normal_image_shape[0],self.normal_image_shape[1],3]])
		self.enqueue_op = q.enqueue([self.ir_image_single,self.normal_image_single])
		self.ir_images, self.normal_images = q.dequeue_many(self.batch_size)

        #self.ir_test = tf.placeholder(tf.float32, [1,600,800,1],name='ir_test')
	self.keep_prob = tf.placeholder(tf.float32)
	net  = networks(self.num_block,self.batch_size,self.df_dim)
	self.G = net.generator(self.ir_images)
	#self.sample = net.sampler(self.ir_test)
	if self.pair:
	    self.D = net.discriminator(tf.concat(3,[self.normal_images,self.ir_images]),self.keep_prob)
	    self.D_  = net.discriminator(tf.concat(3,[self.G,self.ir_images]),self.keep_prob,reuse=True)
	else:
	    self.D = net.discriminator(self.normal_images,self.keep_prob)
	    self.D_  = net.discriminator(self.G,self.keep_prob,reuse=True)

	# generated surface normal
        #self.d_loss_real = binary_cross_entropy_with_logits(tf.pack(np.random.uniform(0.7,1.2,size=(self.batch_size,1)).astype(np.float32)), self.D)
        #self.d_loss_fake = binary_cross_entropy_with_logits(tf.pack(np.random.uniform(0.0,0.3,size=(self.batch_size,1)).astype(np.float32)), self.D_)
        self.d_loss_real = binary_cross_entropy_with_logits(tf.random_uniform(self.D.get_shape(),minval=0.7,maxval=1.2,dtype=tf.float32,seed=0), self.D)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.random_uniform(self.D.get_shape(),minval=0.0,maxval=0.3,dtype=tf.float32,seed=0), self.D_)
        self.d_loss = self.d_loss_real + self.d_loss_fake
	if self.loss == 'L1':
            self.L_loss = tf.div(tf.reduce_sum(tf.abs(tf.sub(self.G,self.normal_images))),self.ir_image_shape[1]*self.ir_image_shape[2]*3)
	else:
            self.L_loss = tf.div(tf.reduce_sum(tf.square(tf.sub(self.G,self.normal_images))),self.ir_image_shape[1]*self.ir_image_shape[2]*3)
        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)

	self.ang_loss = norm_(self.G,self.normal_images)
        self.gen_loss = self.g_loss + self.L_loss *100 + self.ang_loss 

	self.saver = tf.train.Saver(max_to_keep=10)
	t_vars = tf.trainable_variables()
	self.d_vars =[var for var in t_vars if 'd_' in var.name]
	self.g_vars =[var for var in t_vars if 'g_' in var.name]
	

    def train(self, config):
        #####Train DCGAN####

        global_step = tf.Variable(0,name='global_step',trainable=False)
        global_step1 = tf.Variable(0,name='global_step1',trainable=False)
	
	d_optim = tf.train.AdamOptimizer(config.d_learning_rate,beta1=config.beta1) \
                          .minimize(self.d_loss, global_step=global_step,var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.g_learning_rate,beta1=config.beta1) \
                          .minimize(self.gen_loss, global_step=global_step1,var_list=self.g_vars)
	tf.initialize_all_variables().run()
	
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loda training and validation dataset path
        data = json.load(open("/research2/ECCV_journal/with_light/json/traininput.json"))
        data_label = json.load(open("/research2/ECCV_journal/with_light/json/traingt.json"))
        datalist =[data[idx] for idx in xrange(0,len(data))]
        labellist =[data_label[idx] for idx in xrange(0,len(data))]
	shuf = range(len(data))
        list_val = [11,16,21,22,33,36,38,53,59,92]


	if self.use_queue:
	    # creat thread
	    coord = tf.train.Coordinator()
            num_thread =16
            for i in range(num_thread):
 	        t = threading.Thread(target=self.load_and_enqueue,args=(coord,datalist,labellist,shuf,i,num_thread))
	 	t.start()

	if self.use_queue:
	    for epoch in xrange(config.epoch):
	        #shuffle = np.random.permutation(range(len(data)))
	        batch_idxs = min(len(data), config.train_size)/config.batch_size
		sum_L = 0.0
		sum_g =0.0
		sum_ang =0.0
		sum_d_real =0.0
		sum_d_fake =0.0
		if epoch ==0:
		    train_log = open(os.path.join("logs",'train_%s.log' %config.dataset),'w')
		    val_log = open(os.path.join("logs",'val_%s.log' %config.dataset),'w')
		else:
	    	    train_log = open(os.path.join("logs",'train_%s.log' %config.dataset),'aw')
		    val_log = open(os.path.join("logs",'val_%s.log' %config.dataset),'w')

		for idx in xrange(0,batch_idxs):
        	     start_time = time.time()
		     _,d_loss_real,d_loss_fake =self.sess.run([d_optim,self.d_loss_real,self.d_loss_fake],feed_dict={self.keep_prob:self.dropout})
		     _,g_loss,L_loss,ang_loss =self.sess.run([g_optim,self.g_loss,self.ang_loss,self.L_loss],feed_dict={self.keep_prob:self.dropout})
		     print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f L_loss:%.4f ang_loss: %.6f d_loss_real:%.4f d_loss_fake:%.4f" \
		     % (epoch, idx, batch_idxs,time.time() - start_time,g_loss,L_loss,ang_loss,d_loss_real,d_loss_fake))
		     sum_L += L_loss 	
		     sum_g += g_loss
		     sum_ang += ang_loss
		     sum_d_real += d_loss_real
	  	     sum_d_fake += d_loss_fake	
		train_log.write('epoch %06d mean_g %.6f  mean_L %.6f mean_ang %.6f d_real %.6f d_fake %.6f\n' %(epoch,sum_g/(batch_idxs),sum_L/(batch_idxs),sum_ang/batch_idxs,sum_d_real/(batch_idxs),sum_d_fake/batch_idxs))
		train_log.close()
	        self.save(config.checkpoint_dir,global_step)
		"""
		####### Validation #########
		for idx2 in xrange(0,len(list_val)):
		    for tilt in range(1,10):	
		        print("Epoch: [%2d] [%4d/%4d] " % (epoch, idx2, len(list_val)))
		        img = '/research2/IR_normal_small/save%03d/%d' % (list_val[idx2],tilt)
			light = random.randint(1,12)
			input_ = scipy.misc.imread(img+'/%d3.bmp' %light).astype(float)
			input_ = scipy.misc.imresize(input_,[600,800])
			input_ = input_/127.5 - 1.0
			input_ = np.reshape(input_,[1,600,800,1])
			gt_ = scipy.misc.imread(img+'/12_Normal.bmp').astype(float)
			gt_ = gt_/127.5 -1.0
 		        sample = self.sess.run([self.sample],feed_dict={self.ir_test: input_})
			L1_loss = tf.reduce_mean(tf.square(tf.sub(sample,gt_)))
			sum_L1 += L1_loss
			
		val_log.write('epoch %06d mean_L1 %.6f \n' %(epoch,sum_L1/(len(range(1,10)*len(list_val)))))
		val_log.close()
		"""

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
		     #mask_mean = batch_mask * self.mean_nir
		     #batch_images = batch_images- mask_mean
		     # Update Normal D network
		     _= self.sess.run([d_optim], feed_dict={self.ir_images: batch_images,self.normal_images:batchlabel_images })
		     self.writer.add_summary(summary_str, global_step.eval())

		     # Update NIR G network
		     _,g_loss,L1_loss = self.sess.run([g_optim,self.g_loss,self.L1_loss], feed_dict={ self.ir_images: batch_images,self.normal_images:batchlabel_images})
		     print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.6f L1_loss:%.4f" \
		     % (epoch, idx, batch_idxs,time.time() - start_time,g_loss,L1_loss,d_loss))
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

	    
    def load_and_enqueue(self,coord,file_list,label_list,shuf,idx=0,num_thread=1):
	count =0;
	length = len(file_list)
	rot=[0,90,180,270]
	while not coord.should_stop():
	    i = (count*num_thread + idx) % length;
	    r = random.randint(0,2)
            input_img = scipy.misc.imread(file_list[shuf[i]]).reshape([224,224,1]).astype(np.float32)
	    gt_img = scipy.misc.imread(label_list[shuf[i]]).reshape([224,224,3]).astype(np.float32)
	    input_img = input_img/127.5 -1.
	    gt_img = gt_img/127.5 -1.
	    rand_x = np.random.randint(64,224-64)
	    rand_y = np.random.randint(64,224-64)
	    input_img = scipy.ndimage.rotate(input_img,rot[r])
	    gt_img = scipy.ndimage.rotate(gt_img,rot[r])
            self.sess.run(self.enqueue_op,feed_dict={self.ir_image_single:input_img[rand_y:rand_y+64,rand_x:rand_x+64],self.normal_image_single:gt_img[rand_y:rand_y+64,rand_x:rand_x+64]})
	    count +=1
		
