import tensorflow as tf 
import numpy as np
import cv2
import os
import SR_data_load

tf.app.flags.DEFINE_float('lr', 0.0001, 'learninig rate')
tf.app.flags.DEFINE_integer('step', 10000, 'step')
tf.app.flags.DEFINE_integer('image_size', 64, 'image_patch_size')
tf.app.flags.DEFINE_boolean('restore_model', True, '')
tf.app.flags.DEFINE_string('image_path','/your/data/set_path/*.jpg','Where is  Data Imageset')
flag= tf.app.flags.FLAGS



os.environ['CUDA_VISIBLE_DEVICES']='1'

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
        cv2.imwrite(str('./'+str(i)+'/'+feat_name+'_'+str(i)+'.png'),np.reshape(image[:,:,i],(w,h)))

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
first_conv = tf.keras.layers.Conv2D(kernel_size=9,filters=64,padding='SAME')
first_conv_relu = tf.keras.layers.ReLU()

second_conv =tf.keras.layers.Conv2D(kernel_size=1,filters=32,padding='SAME')
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
third_conv = third_conv(second_feature_relu)
reconstructed_image = postprocess(third_conv)

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
'''
out_image=sess.run(net_output,feed_dict={input_image:blur_image})
ois= out_image.shape
out_image = np.reshape(out_image,(ois[1],ois[2]))

cv2.imwrite('./predicted_output.png',out_image)
print('value of output',np.max(out_image),",",np.mean(out_image),",",np.min(out_image))

'''
'''
save_image(sess.run(feature_sf,feed_dict={input_image:blur_image}),"feat1_sf")

save_image(sess.run(feat_rb1_c1,feed_dict={input_image:blur_image}),"feat2_rb1_c1")
save_image(sess.run(feat_rb1_r1,feed_dict={input_image:blur_image}),"feat3_rb1_r1")
save_image(sess.run(feat_rb1_c2,feed_dict={input_image:blur_image}),"feat4_rb1_c2")
save_image(sess.run(feat_rb1_r2,feed_dict={input_image:blur_image}),"feat5_rb1_r2")
save_image(sess.run(feat_rb1_c3,feed_dict={input_image:blur_image}),"feat6_rb1_c3")

save_image(sess.run(feat_rb1_out,feed_dict={input_image:blur_image}),"feat7_rb1_out")

save_image(sess.run(feat_rb2_c1,feed_dict={input_image:blur_image}),"feat8_rb2_c1")
save_image(sess.run(feat_rb2_r1,feed_dict={input_image:blur_image}),"feat9_rb2_r1")
save_image(sess.run(feat_rb2_c2,feed_dict={input_image:blur_image}),"feat10_rb2_c2")
save_image(sess.run(feat_rb2_r2,feed_dict={input_image:blur_image}),"feat11_rb2_r2")
save_image(sess.run(feat_rb2_c3,feed_dict={input_image:blur_image}),"feat12_rb2_c3")

save_image(sess.run(feat_rb2_out,feed_dict={input_image:blur_image}),"feat13_rb2_out")

save_image(sess.run(feat_rb3_c1,feed_dict={input_image:blur_image}),"feat14_rb3_c1")
save_image(sess.run(feat_rb3_r1,feed_dict={input_image:blur_image}),"feat15_rb3_r1")
save_image(sess.run(feat_rb3_c2,feed_dict={input_image:blur_image}),"feat16_rb3_c2")
save_image(sess.run(feat_rb3_r2,feed_dict={input_image:blur_image}),"feat17_rb3_r2")
save_image(sess.run(feat_rb3_c3,feed_dict={input_image:blur_image}),"feat18_rb3_c3")

save_image(sess.run(feat_rb3_out,feed_dict={input_image:blur_image}),"feat19_rb3_out")

save_image(sess.run(feat_rb4_c1,feed_dict={input_image:blur_image}),"feat20_rb4_c1")
save_image(sess.run(feat_rb4_r1,feed_dict={input_image:blur_image}),"feat21_rb4_r1")
save_image(sess.run(feat_rb4_c2,feed_dict={input_image:blur_image}),"feat22_rb4_c2")
save_image(sess.run(feat_rb4_r2,feed_dict={input_image:blur_image}),"feat23_rb4_r2")
save_image(sess.run(feat_rb4_c3,feed_dict={input_image:blur_image}),"feat24_rb4_c3")

save_image(sess.run(feat_rb4_out,feed_dict={input_image:blur_image}),"feat25_rb4_out")

save_image(sess.run(feat_rb5_c1,feed_dict={input_image:blur_image}),"feat26_rb5_c1")
save_image(sess.run(feat_rb5_r1,feed_dict={input_image:blur_image}),"feat27_rb5_r1")
save_image(sess.run(feat_rb5_c2,feed_dict={input_image:blur_image}),"feat28_rb5_c2")
save_image(sess.run(feat_rb5_r2,feed_dict={input_image:blur_image}),"feat29_rb5_r2")
save_image(sess.run(feat_rb5_c3,feed_dict={input_image:blur_image}),"feat30_rb5_c3")

save_image(sess.run(feat_rb5_out,feed_dict={input_image:blur_image}),"feat31_rb5_out")

save_image(sess.run(feat_final,feed_dict={input_image:blur_image}),"feat32_final")

save_image(sess.run(feat_recon,feed_dict={input_image:blur_image}),"feat33_recon")


'''



'''
feature_sf = sess.run(feature_sf,feed_dict={input_image:blur_image})
save_image(feature_sf,"feat1_sf")

feat_rb1_c3 = sess.run(feat_rb1_c3,feed_dict={input_image:blur_image})
save_image(feat_rb1_c3,"feat2_rb1_c3")

feat_rb1_out = sess.run(feat_rb1_out,feed_dict={input_image:blur_image})
save_image(feat_rb1_out,"feat3_rb1_out")

feat_rb2_c3 = sess.run(feat_rb2_c3,feed_dict={input_image:blur_image})
save_image(feat_rb2_c3,"feat4_rb2_c3")

feat_rb2_out = sess.run(feat_rb2_out,feed_dict={input_image:blur_image})
save_image(feat_rb2_out,"feat5_rb2_out")

feat_rb3_c3 = sess.run(feat_rb3_c3,feed_dict={input_image:blur_image})
save_image(feat_rb3_c3,"feat6_rb3_c3")

feat_rb3_out = sess.run(feat_rb3_out,feed_dict={input_image:blur_image})
save_image(feat_rb3_out,"feat7_rb3_out")

feat_rb4_c3 = sess.run(feat_rb4_c3,feed_dict={input_image:blur_image})
save_image(feat_rb4_c3,"feat8_rb4_c3")

feat_rb4_out = sess.run(feat_rb4_out,feed_dict={input_image:blur_image})
save_image(feat_rb4_out,"feat9_rb4_out")

feat_rb5_c3 = sess.run(feat_rb5_c3,feed_dict={input_image:blur_image})
save_image(feat_rb5_c3,"feat10_rb5_c3")

feat_rb5_out = sess.run(feat_rb5_out,feed_dict={input_image:blur_image})
save_image(feat_rb5_out,"feat11_rb5_out")

feat_final = sess.run(feat_final,feed_dict={input_image:blur_image})
save_image(feat_final,"feat12_final")

feat_recon = sess.run(feat_recon,feed_dict={input_image:blur_image})
save_image(feat_recon,"feat13_recon")

feat_up = sess.run(feat_up,feed_dict={input_image:blur_image})
save_image(feat_up,"feature14_sf")
'''

