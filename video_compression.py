import numpy as np
import cv2
import sys
import math
import pickle
import tensorflow as tf
def GetVideoProperties(filePath):
    inputVideo = cv2.VideoCapture(filePath);
    fileName = filePath
    if inputVideo.isOpened() == None:
        return False;
    frameCount = 0;
    while inputVideo.isOpened():
        ret, frame = inputVideo.read()
        if ret == False:
            break;
        if frame is None:
            return False;
        if frame.shape[2] != 3:
            return False;
        frameSize = (int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frameCount += 1;
    fps = inputVideo.get(cv2.CAP_PROP_FPS)
    return [fps, frameSize, frameCount]

def NumToNNValuesSimple(num, maxValue):
    res = np.zeros((maxValue, 1))
    res[num, 0]= 1.0
    res = res.flatten()
    return res

def ImageToNNValues(img):
    #print img
    result = img.flatten()
    result = result / 256.0
    return result

def NNValuesToImage(values, frameSize):
    values *= 256.0
    result = values.reshape(frameSize[0], frameSize[1], 3)
    return result

def ReconstructFrame(nnoutput, frameSize, frameNum, frameCount, NumToNNValuesSimple):
    frame = NNValuesToImage(nnoutput, frameSize)
    return frame

def ReconstructMovie(nnoutput, frameSize, frameCount, fps, filePath, NumToNNValuesSimple):
    
    print "Loading neural network"
    if nnoutput is None:
        return False
    fourcc = cv2.VideoWriter_fourcc(*'XVID');
    outputVideo = cv2.VideoWriter(filePath, fourcc, fps, frameSize);
    if outputVideo.isOpened() == None:
        print "Could not open the output video for write: ", filePath
        return False;
    for frameNum in range(frameCount):
        if  frameNum == 0 or frameNum == frameCount - 1 or frameNum % 100 == 0:
            print "Reconstructing frame ", frameNum + 1, " of ", frameCount
        frame = ReconstructFrame(nnoutput[frameNum], frameSize, frameNum, frameCount, NumToNNValuesSimple)
        #norm = np.linalg.norm(frame)
        #frame = frame / norm
        #print frame
        frame = frame * 255.0
        frame = frame.astype('u1')
        
        #print len(frame), len(frame[0]), len(frame[0][0])
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        outputVideo.write(frame);
    print "Saved ", filePath
    return True

if __name__ == '__main__':

    inFilePath = "/home/lenovo3/3/box1.avi"

    nnFilePath = inFilePath + ".nncv";
    outFilePath = inFilePath + ".n_n.avi";
    maxIters = 1000;
    epsilon = 0.00000000001;

    [fps, frameSize, frameCount] = GetVideoProperties(inFilePath)
    
    print inFilePath, " - fps: ", fps, " - frameSize: ", frameSize, " - frameCount: ", frameCount

    layerSizes = [];
    
    inputLayerSize = NumToNNValuesSimple(0, frameCount).shape[0];
    outputLayerSize = frameSize[0] * frameSize[1] * 3;
    hiddenLayerSize = int(math.sqrt(frameCount)) + 1;
    layerSizes.append(inputLayerSize );
    layerSizes.append(hiddenLayerSize );
    layerSizes.append(hiddenLayerSize );
    layerSizes.append(outputLayerSize );
    layerSizes = np.array(layerSizes)
    
    nnPtr = cv2.ml.ANN_MLP_create();
    nnPtr.setBackpropMomentumScale(0.0)
    nnPtr.setBackpropWeightScale(0.001)
    nnPtr.setLayerSizes( layerSizes );
    nnPtr.setActivationFunction(cv2.ml.ANN_MLP_GAUSSIAN, 2, 1);
    nnPtr.setTrainMethod( cv2.ml.ANN_MLP_RPROP, 0.1, sys.float_info.epsilon)
    nnPtr.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, maxIters, epsilon));

    samples = np.empty((frameCount, inputLayerSize))
    responses = np.empty((frameCount, outputLayerSize))

    inputVideo = cv2.VideoCapture(inFilePath) ;
    fileName = inFilePath;

    for frameNum in range(frameCount):
        if frameNum == 0 or frameNum == frameCount - 1 or frameNum % 100 == 0:
            print "Loading frame ", frameNum + 1, " of ", frameCount
        ret, frame = inputVideo.read()
        if ret == False:
            break;
        if frame is None:
            print 1;
        imageNNValues = ImageToNNValues(frame);
        frameNumNNValues = NumToNNValuesSimple(frameNum, frameCount);
        #print len(samples), len(samples[0]), len(responses), len(responses[0])
        np.copyto(samples[frameNum], frameNumNNValues);
        np.copyto(responses[frameNum], imageNNValues);
    print samples.shape, responses.shape
    def weight_variable(shape):  
      initial = tf.truncated_normal(shape, stddev=0.1)  
      return tf.Variable(initial)  
  
    def bias_variable(shape):  
      initial = tf.constant(0.1, shape=shape)  
      return tf.Variable(initial)  
  
    def conv2d(x, W):  
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  
  
    def max_pool_2x2(x):  
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", shape=[None, frameCount])
    y_ = tf.placeholder("float", shape=[None, frameSize[0] * frameSize[1] * 3])
    W_conv1 = weight_variable([10, 10, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, frameCount, frameCount, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([10, 10, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([frameCount / 16 * 64, 4096])
    b_fc1 = bias_variable([4096])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, frameCount / 16 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder('float')
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([4096, frameSize[0] * frameSize[1] * 3])
    b_fc2 = bias_variable([frameSize[0] * frameSize[1] * 3])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    
    for i in range(2000):
        batch = samples, responses
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            print "step %d, training accuracy %g"%(i, train_accuracy)
        train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
    nnoutput = sess.run(y_conv, feed_dict={x: batch[0], keep_prob: 0.5})
    #print "Training neural network"
    #nnPtr.train(samples, cv2.ml.ROW_SAMPLE, responses);
    #print len(samples)
    #print "Saving neural network"
    #nnPtr.save(nnFilePath);
    #print nnPtr
    ReconstructMovie(nnoutput, frameSize, frameCount, fps, outFilePath, NumToNNValuesSimple);
    inputVideo.release()
    outputVideo.release()
    cv2.destroyAllWindows()
