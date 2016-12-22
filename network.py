from ops import *
import tensorflow as tf

class networks(object):
    def __init__(self,num_block,batch_size,df_dim):
	self.num_block = num_block
	self.batch_size = batch_size
	self.df_dim = df_dim
  
    def generator_low(self,nir):
	#g_bn0 = batch_norm(self.batch_size,name='g_bn0')
        g_nir0 =lrelu(conv2d(nir,self.df_dim*2,k_h=3,k_w=3,name='low_g_nir0'))
	g_bn1 = batch_norm(self.batch_size,name='low_g_bn1')
        g_nir1 =lrelu(g_bn1(conv2d(g_nir0,self.df_dim*4,k_h=3,k_w=3,name='low_g_nir1')))
	g_bn2 = batch_norm(self.batch_size,name='low_g_bn2')
        g_nir2 =lrelu(g_bn2(conv2d(g_nir1,self.df_dim*4,k_h=3,k_w=3,name='low_g_nir2')))
	g_bn3 = batch_norm(self.batch_size,name='low_g_bn3')
        g_nir3_1 =conv2d(g_nir2,self.df_dim*2,k_h=3,k_w=3,name='low_g_nir3')
        g_nir3_2 =lrelu(g_bn3(g_nir3_1))
        g_nir4 =conv2d(g_nir3_2,3,k_h=3,k_w=3,name='low_g_nir4')
	return tf.tanh(g_nir3_1), tf.tanh(g_nir4)


    def generator_high(self,nir):
	#g_bn0 = batch_norm(self.batch_size,name='g_bn0')
        g_nir0 =lrelu(conv2d(nir,self.df_dim*2,k_h=3,k_w=3,name='high_g_nir0'))
	g_bn1 = batch_norm(self.batch_size,name='high_g_bn1')
        g_nir1 =lrelu(g_bn1(conv2d(g_nir0,self.df_dim*4,k_h=3,k_w=3,name='high_g_nir1')))
	g_bn2 = batch_norm(self.batch_size,name='high_g_bn2')
        g_nir2 =lrelu(g_bn2(conv2d(g_nir1,self.df_dim*4,k_h=3,k_w=3,name='high_g_nir2')))
	g_bn3 = batch_norm(self.batch_size,name='high_g_bn3')
        g_nir3 =lrelu(g_bn3(conv2d(g_nir2,self.df_dim*2,k_h=3,k_w=3,name='high_g_nir3')))
        g_nir4 =conv2d(g_nir3,3,k_h=3,k_w=3,name='high_g_nir4')
	return tf.tanh(g_nir4)


    def discriminator_low(self, image,keep_prob, reuse=False): # input 64 x 64
	with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
            h0 = lrelu(conv2d(image,self.df_dim,d_h=2,d_w=2,name='low_d_h0_conv')) #output size: 32x32
	    d_bn1 = batch_norm(self.batch_size,name='low_d_bn1')
            h1 = lrelu(d_bn1(conv2d(h0, self.df_dim*2, d_h=2,d_w=2,name='low_d_h1_conv'))) #output size: 16 x16 
	    d_bn2 = batch_norm(self.batch_size,name='low_d_bn2')
            h2 = lrelu(d_bn2(conv2d(h1, self.df_dim*4, d_h=2,d_w=2,name='low_d_h2_conv'))) #output size: 8 x 8
	    d_bn3 = batch_norm(self.batch_size,name='low_d_bn3')
            h3 = conv2d(h2, 1, d_h=2,d_w=2,name='low_d_h3_conv') #output size: 4x4
            #h4 = conv2d(h3,1, k_h=1,k_w=1,name='d_h4_conv') #output size: 4x4
            #h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'low_d_h4_lin') #output 512
	    #h4 = tf.nn.dropout(h4,keep_prob)
	    #h5 = linear(h4,1,'d_h5_lin')
	    #h5 = linear(h4, 1, 'd_h5_lin')
            return tf.nn.sigmoid(h3)


    def discriminator_high(self, image,keep_prob, reuse=False):
	with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
            h0 = lrelu(conv2d(image,self.df_dim,d_h=2,d_w=2,name='high_d_h0_conv')) #output size: 32x32
	    d_bn1 = batch_norm(self.batch_size,name='high_d_bn1')
            h1 = lrelu(d_bn1(conv2d(h0, self.df_dim*2, d_h=2,d_w=2,name='high_d_h1_conv'))) #output size: 16 x16 
	    d_bn2 = batch_norm(self.batch_size,name='high_d_bn2')
            h2 = lrelu(d_bn2(conv2d(h1, self.df_dim*4, d_h=2,d_w=2,name='high_d_h2_conv'))) #output size: 8 x 8
	    #d_bn3 = batch_norm(self.batch_size,name='high_d_bn3')
            h3 = conv2d(h2, 1, d_h=2,d_w=2,name='high_d_h3_conv') #output size: 4x4
            #h4 = conv2d(h3,1, k_h=1,k_w=1,name='d_h4_conv') #output size: 4x4
            #h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'high_d_h4_lin') #output 512
	    #h4 = tf.nn.dropout(h4,keep_prob)
	    #h5 = linear(h4,1,'d_h5_lin')
	    #h5 = linear(h4, 1, 'd_h5_lin')
            return tf.nn.sigmoid(h3)

    """
    def sampler_low(self,nir):
	tf.get_variable_scope().reuse_variables()
	g_nir0 =lrelu(conv2d(nir,self.df_dim*2,k_h=3,k_w=3,name='low_g_nir0'))
	g_bn1 = batch_norm(self.batch_size,name='low_g_bn1')
        g_nir1 =lrelu(g_bn1(conv2d(g_nir0,self.df_dim*4,k_h=3,k_w=3,name='low_g_nir1')))
	g_bn2 = batch_norm(self.batch_size,name='low_g_bn2')
        g_nir2 =lrelu(g_bn2(conv2d(g_nir1,self.df_dim*4,k_h=3,k_w=3,name='low_g_nir2')))
	g_bn3 = batch_norm(self.batch_size,name='low_g_bn3')
        g_nir3 =lrelu(g_bn3(conv2d(g_nir2,self.df_dim*2,k_h=3,k_w=3,name='low_g_nir3')))
        g_nir4 =conv2d(g_nir3,3,k_h=3,k_w=3,name='low_g_nir4')
	return tf.tanh(g_nir4)

    def sampler_high(self,nir):
	tf.get_variable_scope().reuse_variables()
	g_nir0 =lrelu(conv2d(nir,self.df_dim*2,k_h=3,k_w=3,name='high_nir0'))
	g_bn1 = batch_norm(self.batch_size,name='high_g_bn1')
        g_nir1 =lrelu(g_bn1(conv2d(g_nir0,self.df_dim*4,k_h=3,k_w=3,name='high_g_nir1')))
	g_bn2 = batch_norm(self.batch_size,name='high_g_bn2')
        g_nir2 =lrelu(g_bn2(conv2d(g_nir1,self.df_dim*4,k_h=3,k_w=3,name='high_g_nir2')))
	g_bn3 = batch_norm(self.batch_size,name='high_g_bn3')
        g_nir3 =lrelu(g_bn3(conv2d(g_nir2,self.df_dim*2,k_h=3,k_w=3,name='high_g_nir3')))
        g_nir4 =conv2d(g_nir3,3,k_h=3,k_w=3,name='high_g_nir4')
	return tf.tanh(g_nir4)
    """
