import tensorflow as tf 
import numpy as np
import cv2
import os
import SR_data_load

tf.app.flags.DEFINE_float('lr', 0.0001, 'learninig rate')
tf.app.flags.DEFINE_integer('step', 1000, 'step')
tf.app.flags.DEFINE_integer('image_size', 64, 'image_patch_size')
tf.app.flags.DEFINE_boolean('restore_model', True, '')
tf.app.flags.DEFINE_string('image_path','/your/data/set_path/*.jpg','Where is  Data Imageset')
flag= tf.app.flags.FLAGS



os.environ['CUDA_VISIBLE_DEVICES']='0'

def preprocess(images):
    pp_images= images/255.0
    pp_images = pp_images*2.0 - 1.0
    return pp_images

def postprocess(images):
    pp_images=((images+1.0)/2.0)*255.0
    return pp_images

def save_image(image,feat_name):
    image=postprocess(image)
    _,w,h,c = image.shape
    image=np.reshape(image,(w,h,c))
    for i in range(c):
        cv2.imwrite(str('./feature_map/'+str(i)+'/'+feat_name+'_'+str(i)+'.png'),np.reshape(image[:,:,i],(w,h)))

def phaseShift(features, scale, shape_1, shape_2):
    X = tf.reshape(features, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])
    return tf.reshape(X, shape_2)
    
def pixelShuffler(features, scale=2):
    size = tf.shape(features)
    batch_size = size[0]
    h,w,c = size[1],size[2],features.get_shape().as_list()[-1]# 64
    channel_target = c // (scale * scale) # 16 
    channel_factor = c // channel_target  #4
    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]#[n,h,w,2,2]
    shape_2 = [batch_size, h * scale, w * scale, 1]#[n,h*2,w*2,1]
    input_split = tf.split(axis=3, num_or_size_splits=channel_target, value=features) #features, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)
    return output

##############################
#Input and Output placeholder#
##############################
input_image= tf.placeholder(tf.float32,shape=[None,flag.image_size,flag.image_size,3],name="Inputimage")
output_image= tf.placeholder(tf.float32,shape=[None,flag.image_size,flag.image_size,3],name="Ouputimage")

input_valid_image = tf.placeholder(tf.float32,shape=[1,None,None,3],name="Input_Valid_Image")
#######################
# Network architecture#
###################################################
#input - conv - relu - conv - relu - conv - output#
###################################################

###############
#Declare layer#
###############
#SRCNN consists of 3-layer
# 1st layer : feature extraction and representation:
#   kernel size : 9 , input channel : 3(RGB), output channel : 64
# 2nd layer : non-linear mapping 
#   kernel size : 1 , input channel : 64    , output channel : 32
# 3rd layer : reconstruction layer
#   kernel size : 5 , input channel : 32    , output channel : 3(RGB)
#####################################################################
first_conv = tf.keras.layers.Conv2D(kernel_size=9,filters=128,padding='SAME')
first_conv_relu = tf.keras.layers.ReLU()

second_conv =tf.keras.layers.Conv2D(kernel_size=1,filters=64,padding='SAME')
second_conv_relu = tf.keras.layers.ReLU()

third_conv =tf.keras.layers.Conv2D(kernel_size=5,filters=3,padding='SAME')

##################################
#Design srcnn network using layer#
##################################
#Preprocess  [0, 255] -> [-1, 1]
pre_pro_feature = preprocess(input_image)

#1st layer : feature extraction and representation
first_feature  = first_conv(pre_pro_feature)
first_feature_relu = first_conv_relu(first_feature)

#2nd layer : non-linear mapping
second_feature = second_conv(first_feature_relu)
second_feature_relu = second_conv_relu(second_feature)

#3nd layer : reconstruction 
third_feature = third_conv(second_feature_relu)
reconstructed_image = postprocess(third_feature)


####################
#Validation Network#
####################
pre_pro_valid = preprocess(input_valid_image)
valid_first  = first_conv(pre_pro_valid)
valid_first_relu = first_conv_relu(valid_first)
valid_second =  second_conv(valid_first_relu)
valid_second_relu = second_conv_relu(valid_second)
valid_third = third_conv(valid_second_relu)
valid_output = postprocess(valid_third)


##############################
#Evaluation and loss function#
##############################
psnr = tf.reduce_mean(tf.image.psnr(output_image,reconstructed_image,max_val=255))
loss = tf.reduce_mean(tf.abs(output_image-reconstructed_image))


#################################################
#Opimizer, calcuate gradinet and apply gradient #
#################################################
optimizer = tf.train.AdamOptimizer(0.0001)
conv_var = tf.trainable_variables()
conv_gradients=optimizer.compute_gradients(loss,var_list=conv_var)
conv_apply_gradients = optimizer.apply_gradients(conv_gradients)

######To save model####
saver = tf.train.Saver()

#############
#Run Session#
#############
sess = tf.Session()
sess.run(tf.global_variables_initializer())

data_generator = SR_data_load.get_batch(image_path=flag.image_path,num_workers=8,batch_size=16,image_size=flag.image_size)

if flag.restore_model==True:
    saver.restore(sess,'./model.ckpt-simple-srcnn')

for i in range(flag.step):
    #Get mini-batch 
    data = next(data_generator)
    hr_image = np.asarray(data[0])
    lr_image = np.asarray(data[1])
    
    #define feed_dict
    feed_dict = {input_image:lr_image,output_image:hr_image}
    
    #Step that compute gradient and then apply 
    _=sess.run(conv_apply_gradients,feed_dict=feed_dict)
    
    if i%100==0:
        p,l=sess.run([psnr,loss],feed_dict=feed_dict)
        print(i,"step, ",p,"db,  loss:",l)


saver.save(sess,'./model.ckpt-simple-srcnn')

#Test 
test_image = cv2.imread('./lr_image.png')
tis  = test_image.shape
test_image_reshape = np.reshape(test_image,[1,tis[0],tis[1],tis[2]])
out_image=sess.run(valid_output,feed_dict={input_valid_image:test_image_reshape})
ois= out_image.shape
out_image_reshape = np.reshape(out_image,(ois[1],ois[2],ois[3]))
cv2.imwrite('./predicted_output.png',out_image_reshape)



save_image(sess.run(valid_first,feed_dict={input_valid_image:test_image_reshape}),"1st_conv")
save_image(sess.run(valid_first_relu,feed_dict={input_valid_image:test_image_reshape}),"1st_conv_relu")
save_image(sess.run(valid_second,feed_dict={input_valid_image:test_image_reshape}),"2nd_conv")
save_image(sess.run(valid_second_relu,feed_dict={input_valid_image:test_image_reshape}),"2nd_conv_relu")
save_image(sess.run(valid_third,feed_dict={input_valid_image:test_image_reshape}),"3rd_conv")
