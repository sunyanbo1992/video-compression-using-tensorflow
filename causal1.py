import numpy as np
import tensorflow as tf
import cv2
import math
import tensorflow.contrib.layers as lays
import os
import merge_block
import split_block
import reconstruct
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim
from ssim import *

training_set = []
validation_set = []
image_training = []
image_target = []
image_test = []

image_all = []
jpeg_all = []
image_validation = []

target_path = "target1/"

#validation_path = "validation"

train_path = "train_25/"

train_image = os.listdir(train_path)

target_image = os.listdir(target_path)

test_path = "test_300/"
test_image = os.listdir(test_path)

train_image.sort(key=lambda x:int(x[:-4]))

target_image.sort(key=lambda x:int(x[:-4]))

test_image.sort(key=lambda x:int(x[:-4]))

def u_mse(prediction,Y):
    mse= 0.0
    prediction = tf.convert_to_tensor(prediction,dtype=tf.float32)
    Y = tf.convert_to_tensor(Y,dtype=tf.float32)
    #n = tf.convert_to_tensor(n,dtype=tf.float32)
    n = 49.0
    mse = tf.reduce_sum(tf.pow(prediction - Y,2.0))/(2.0*n) # in your case n = 1024
    mse = tf.convert_to_tensor(mse,dtype=tf.float32)
    return mse

blocks_target = np.zeros([8,8,8,8])
blocks_training = np.zeros([8,8,8,8])

for img in target_image:
    num = img.split('.')[0]
    image1 = target_path + num + ".png"
    image2 = train_path + num + ".jpg"
    input_matrix_target = cv2.imread(image1, 0) / 255.0
    input_matrix_training = cv2.imread(image2, 0) / 255.0
    
    for i in xrange(8):
        for j in xrange(8):
            blocks_target[i, j] = input_matrix_target[i*8:(i+1)*8, j*8:(j+1)*8]
    
    for i in xrange(8):
        for j in xrange(8):
            blocks_training[i, j] = input_matrix_training[i*8:(i+1)*8, j*8:(j+1)*8]
    
    batch = []
    batch_target = []
    
    for i in xrange(7):
        for j in xrange(7):
            all_rows_concatenated = []
            for x in xrange(2):
                all_rows_concatenated.append(np.concatenate([blocks_training[i + x, j], blocks_training[i + x, j + 1]], axis = 1))
            combined_block_training = np.concatenate(all_rows_concatenated, axis = 0)
            batch.append(combined_block_training)
            batch_target.append(blocks_target[i + 1, j + 1])
    batch = np.asarray(batch)
    batch_target = np.asarray(batch_target)
    image_training.append(batch)
    image_target.append(batch_target)
    
image_training = np.asarray(image_training)
image_target = np.asarray(image_target)

blocks_test = np.zeros([8,8,8,8])
    
for img in test_image:
    image = os.path.join(test_path, img)
    input_matrix = cv2.imread(image, 0) / 255.0
    
    batch = []
    for i in xrange(8):
        for j in xrange(8):
            blocks_test[i, j] = input_matrix[i*8:(i+1)*8, j*8:(j+1)*8]
    
    for i in range(7):
        
        for j in xrange(7):
            all_rows_concatenated = []
            for x in xrange(2):
                all_rows_concatenated.append(np.concatenate([blocks_test[i + x, j], blocks_test[i + x, j + 1]], axis = 1))
            combined_block = np.concatenate(all_rows_concatenated, axis = 0)
            batch.append(combined_block)
    batch = np.asarray(batch)
    
    image_test.append(batch)
    
n_visible = 64
n_hidden = 9216

X = tf.placeholder("float", [None, 256], name='X')
Z_ = tf.placeholder("float", [None, 64], name = "Z")
keep_prob = tf.placeholder("float")

initializer = tf.contrib.layers.xavier_initializer()

W = tf.Variable(initializer([256, n_hidden]))

b = tf.Variable(initializer([n_hidden]))

W_prime = tf.Variable(initializer([n_hidden, n_visible]))

b_prime = tf.Variable(initializer([n_visible]))

W_2 = tf.Variable(initializer([n_hidden, n_hidden]))

W_3 = tf.Variable(initializer([n_hidden, n_hidden]))

b_2 = tf.Variable(initializer([n_hidden]))

b_3 = tf.Variable(initializer([n_hidden]))

W_4 = tf.Variable(initializer([n_hidden, n_hidden]))

b_4 = tf.Variable(initializer([n_hidden]))

def model(X, W, b):
    Y = tf.nn.relu6(tf.matmul(X, W) + b)
    
    Y_1 = tf.nn.relu6(tf.matmul(Y, W_2) + b_2)
    
    Y_2 = tf.nn.relu6(tf.matmul(Y_1, W_3) + b_3)

    Y_3 = tf.nn.relu6(tf.matmul(Y_2, W_4) + b_4)
    
    Y_drop = tf.nn.dropout(Y_1, keep_prob)

    Z = tf.nn.relu6(tf.matmul(Y_drop, W_prime) + b_prime)  # reconstructed input
    
    return Z

Z = model(X, W, b)
Z_ = tf.to_float(Z_)
Z = tf.to_float(Z)

cost = u_mse(Z, Z_)

train_op = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost)  # construct an optimizer

mse_figure1 = []
mse_figure2 = []
mse_all = 0.0
psnr = 0.0
ssim_all = 0.0
msssim_all = 0.0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        print "-" * 20
        ran_train = range(len(image_training))
        np.random.shuffle(ran_train)

        target = image_target[ran_train]
        training = image_training[ran_train]
        #while k < len(image_training):
        for j in range(len(image_training)):
            #for n in xrange(len(image_training[j])):
            batch = training[j]
            batch_1 = target[j]

            batch = batch.reshape(-1, 256)
            batch_1 = batch_1.reshape(-1, 64)

            sess.run(train_op, feed_dict={X: batch, Z_: batch_1, keep_prob: 0.8})
            mse_training = sess.run(cost, feed_dict={X: batch, Z_: batch_1, keep_prob: 1.0})
                #mse_figure1.append(mse_training)
            if j % 1000 == 0:
                print i, mse_training
            #print i, mse_valid
    
    index = 0
    decoded_blocks = []
    for j in xrange(len(image_test)):
        for n in xrange(len(image_test[j])):
            quant_matrix = image_test[j][n]
        
            reconstruct_matrix = quant_matrix.reshape(-1, 256)
        
            pred_matrix = sess.run(Z, feed_dict={X: reconstruct_matrix, keep_prob: 1.0})
            pred_matrix = pred_matrix.reshape(8, 8)
            decoded_blocks.append(pred_matrix)
            
            if len(decoded_blocks) == 49:
                width = 56
                height = 56
                width_padded, height_padded = merge_block.calc_new_size(width, height)
                columns, rows = width_padded / 8, height_padded / 8

                final_matrix = merge_block.merge_blocks(decoded_blocks, rows, columns)

                img_array = np.zeros(shape = [height, width])

                for i in range(height):
                    for j in range(width):
                        if final_matrix[i,j] < 0:
                            img_array[i,j] = 16.0
                        elif final_matrix[i,j] > 255:
                            img_array[i,j] = 239
                        else:
                            img_array[i,j] = final_matrix[i, j]

                img_array = img_array.astype(np.uint8, copy=False)

               # cv2.imwrite("casual1/%d.jpg" % index, img_array)
		#original_matrix = cv2.imread("casual_test_300/%d.png" % index, 0)

		mse_all += mean_squared_error(original_matrix, img_array)
                psnr += cv2.PSNR(original_matrix, img_array)
		(score, diff) = compare_ssim(original_matrix, img_array, full=True)
		
		ssim_all += score
		msssim_all += msssim(original_matrix, img_array)
                decoded_blocks  = []
                index += 1

print "mse result:", mse_all / 300
print "psnr result:", psnr / 300
print "ssim result:", ssim_all / 300
print "msssim result:", msssim_all / 300
