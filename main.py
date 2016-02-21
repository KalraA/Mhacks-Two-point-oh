# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:37:33 2016

@author: louis
"""
#Basic includes
import cPickle
import gzip
import os
import sys
import timeit
import math

#Computer Vision Includes
import cv2
import argparse

#Machine learning includes
import numpy as np
import theano
import theano.tensor as T
import csv
import numpy
import six.moves.cPickle as pickle


def _shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class LogisticRegression:
    def __init__(self, input, W,b):
        self.W = W
        self.b = b
        self.input = input

        self.y_pred_x = T.nnet.softmax(T.dot(input,self.W) + self.b)
        self.y_pred = T.argmax(self.y_pred_x, axis=1)
        self.params = [self.W, self.b]
    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.y_pred_x)[T.arange(y.shape[0]),y])

"""
    def __init__(self, input, input_num,output_num):
        self.W = theano.shared(value = numpy.zeros((input_num,output_num),dtype=theano.config.floatX), name = "W",borrow=True)
        self.b = theano.shared(value= numpy.zeros((output_num,),dtype= theano.config.floatX),name = "b", borrow=True)

        self.y_pred_x = T.nnet.softmax(T.dot(input,self.W) + self.b)
        self.y_pred = T.argmax(self.y_pred_x,axis=1)
        self.params = [self.W, self.b]
        self.input = input
"""


class MLP(object):
    def __init__(self, params):
        self.input = T.matrix("x")
        print params[0].shape
        print params[1].shape

        print params[1][0].shape
        print params[1][1].shape

        print params[2][0].shape
        print params[2][1].shape
        self.hiddenLayer= HiddenLayer(
                rng = 1,
                n_in = 1,
                n_out = 1,
                input = self.input,
                W = params[0],
                b = params[1])
        self.hiddenTwo = HiddenLayer(
                rng = 1,
                n_in = 1,
                n_out = 1,
                input = self.hiddenLayer.output,
                W = params[2],
                b = params[3])
        self.logRegressionLayer = LogisticRegression(
                input = self.hiddenTwo.output,
                W = params[4],
                b = params[5])
        
        self.params = self.hiddenLayer.params + self.hiddenTwo.params + self.logRegressionLayer.params
        self.pred = self.logRegressionLayer.y_pred
"""
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        self.hiddenTwo = HiddenLayer(
            rng = numpy.random.RandomState(121591),
            input = self.hiddenLayer.output,
            n_in = n_hidden,
            n_out = n_hidden,
            activation = T.tanh
        )
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenTwo.output,
            input_num=n_hidden,
            output_num=n_out
        )
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.params = self.hiddenLayer.params + self.hiddenTwo.params + self.logRegressionLayer.params
        self.pred = self.logRegressionLayer.y_pred
        self.input = input
"""
def train(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=19, n_hidden=500):
    csvfile=open("fer2013.csv")

    # allocate symbolic variables for the data
    index = T.iscalar()  # index to a [mini]batch   
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=48 * 48,
        n_hidden=n_hidden,
        n_out=7
    )

    cost = (
        classifier.negative_log_likelihood(y)
        
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    reader = csv.DictReader(csvfile)

    x_train =numpy.zeros(shape=(28709, 48 * 48),dtype=theano.config.floatX)
    y_train =numpy.zeros(shape=(28709,),dtype=theano.config.floatX)
   
    cur_index=0
    for row in reader:
        x_train[cur_index] =map(float,row["pixels"].split(" "))
        y_train[cur_index] = row["emotion"]    
    train = [x_train, y_train]
    train_a =_shared_dataset(train)

    train_model = theano.function(
            inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_a[0][index * batch_size: (index + 1) * batch_size],
            y: train_a[1][index * batch_size: (index + 1) * batch_size]
        }
    )

    # f = file("EmotionalLearning.save","wb")
    # for itt in range (0,5):
    #     for row in range(0,(28709/batch_size)):
    #         train_model(row)
    #         print((5 * (28709 / batch_size)) - (row + itt * (28709/batch_size)), " remaining operations")

#     cPickle.dump(classifier,f, protocol=cPickle.HIGHEST_PROTOCOL)
#     a.close()
# def loadmodel():
#     f = open("EmotionalLearning.save","rb")
#     model = MLP(cPickle.load(f))
#     f.close()
#     return theano.function(
#             inputs=[model.input],
#             outputs=model.pred)

# def predict(image,model):
#     return model(image)

def rots(e1, e2, ordis=1):
    ydif = e2[1]-e1[1]
    xdif = e2[0]-e1[0]
    yrot = math.atan(ydif/xdif)*(180/3.14159)
    dis = (xdif**2 + ydif**2)**.5
    if dis < 0.9*ordis:
        return[yrot,1]

    return [yrot, 0]

class cvHelper:
    def __init__(self):
        self.haarFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.haarEyes = cv2.CascadeClassifier("haarcascade_eye.xml")
        self.haarMouth = cv2.CascadeClassifier("Mouth.xml")
    def detectFace(self,image):
        detectedFaces = self.haarFace.detectMultiScale(image,1.3,5)
        return detectedFaces
    def detectEye(self,image):
        detectedEyes = self.haarEyes.detectMultiScale(image,1.3,5)
        return detectedEyes
    def detectMouth(self,image):
        detectedMouth = self.haarMouth.detectMultiScale(image,1.3,5)
        return detectedMouth

def mkfrm(yrot, zrot, blink, emo, mouth, aladdin):
    if aladdin:
        e = 'h';
        m = 'c';
        z = 'f';
        b = 'o'
        if zrot == 1:
            z = 's';
        if blink == 0:
            b = 'b'
        if mouth == 0:
            m = 'o'
        elif mouth == 2:
            m = 'w'
        elif mouth == 3:
            m = 'a'
        if emo == 0:
            e = 'a'
        elif emo == 1:
            e = 's'
        elif emo == 2:
            e = 'n'
        else:
            e = 'h'
        strr = e + z + b + m + ".png";
    else:
        e = 'Hap';
        m = '2';
        z = 'F';
        b = 'O'
        if zrot == 1:
            z = 'S';
        if blink == 0:
            b = 'C'
        if mouth == 0:
            m = '3'
        elif mouth == 2:
            m = '4'
        elif mouth == 3:
            m = '1'
        if emo == 0:
            e = 'Ang'
        elif emo == 1:
            e = 'Sad'
        elif emo == 2:
            e = 'Neu'
        else:
            e = 'Hap'
        strr = z + e + b + m + ".png";
    image = cv2.imread(strr, -1);
    (h, w) = image.shape[:2]
    if aladdin:
        center = (267, 270);
    else:
        center = (180, 550); 
    M = cv2.getRotationMatrix2D(center, yrot, 1.0)
    if aladdin:
        rotated = cv2.warpAffine(image, M, (w, h));
    else:
        rotated = cv2.warpAffine(image, M, (300, h))
    return rotated;

if __name__ == '__main__':    
    """
    #Training Code
    """            
    #Runtime Code
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help = "path to the (optional) video file")
    args = vars(ap.parse_args())
    
    
    # if the video path was not supplied, grab the reference to the
    # camera
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)

    # otherwise, load the video
    else:
        camera = cv2.VideoCapture(args["video"])
    img = cv2.imread("dickbutt.png")
    x_offset = 0
    y_offset = 0
    alad = True;
    if alad:
        y_off = 200;
        x_off = 0;
    else:
        y_off = 250;
        x_off = 100;
    helper = cvHelper()
    a = True;
    counter = 0; 
    rcounter = 0;
    yrot = 0;
    zrot = 0;
    ordis = 1;
    emo = 0;
    mouth = 0;
    zrotf = 0;
    yrotf = 0;
    orMouth = 0;
    b = True;
    yrotp = 0;
    z = False;
    bck = cv2.imread('Background.png')
    img2 = cv2.imread('Background.png')
    sadbol = True;
    hapbol = True;
    angbol = True;
    neubol = True;
    # shape = img.shape
    # b_channel, g_channel, r_channel = cv2.split(img)
    # alpha_channel = np.ones((shape[0], shape[1])) * 50 #creating a dummy alpha channel image.
    # b_channel = np.float64(b_channel)
    # g_channel = np.float64(g_channel)
    # r_channel = np.float64(r_channel)
    # alpha_channel = np.float64(alpha_channel)
    # bck = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    #emotionalModel =loadmodel()
    while True:
        if counter < 1000:
            counter += 1;
        # grab the current frame
        (grabbed, frame) = camera.read()
        # check to see if we have reached the end of the
        # video
        #Text being stored:




        if not grabbed:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detectedFaces = helper.detectFace(gray)
        if len(detectedFaces) == 1:
            gray = gray[detectedFaces[0][1]:
                    detectedFaces[0][1] + detectedFaces[0][3]+50,
                    detectedFaces[0][0]: 
                    detectedFaces[0][0] + detectedFaces[0][2]+50]
            facePic = cv2.resize(gray, (48, 48))
            if counter > 100 and angbol:
                print 'angry'
                aface = cv2.resize(gray, (48, 48));
                angbol = False;
            elif counter > 80 and sadbol:
                print 'sad'
                sface = cv2.resize(gray, (48, 48));
                sadbol = False;
            elif counter > 60 and hapbol:
                print 'happy'
                hface = cv2.resize(gray, (48, 48));
                hapbol = False;
            elif counter > 40 and neubol:
                print 'neutral'
                nface = cv2.resize(gray, (48, 48));
                neubol = False;
            facePic = cv2.equalizeHist(facePic)
            arr = facePic.flatten().reshape(1,48*48)
            #print(predict(arr,emotionalModel))
        detectedEyes = helper.detectEye(gray)
        detectedMouths = helper.detectMouth(gray)
        if detectedFaces is not None:
            for face in detectedFaces:
                   cv2.rectangle(frame,(face[0],face[1]),
                                          (face[0]+face[2],face[1]+face[3]),
                                                         (155, 255, 25),2)
        if detectedEyes is not None:
            for eye in detectedEyes:
                    cv2.rectangle(frame, (eye[0],eye[1]),
                            (eye[0]+eye[2],eye[1]+eye[3]),
                            (155,55,200),2)

        if detectedMouths is not None:
            if not len(detectedMouths) == 0:
                lowestMouth = detectedMouths[0]
                for mouth in detectedMouths:
                    if lowestMouth[1]+lowestMouth[3] <  mouth[1]+mouth[3]:
                        lowestMouth = mouth
                cv2.rectangle(frame,(lowestMouth[0],lowestMouth[1]), (lowestMouth[0]+lowestMouth[2],lowestMouth[1]+lowestMouth[3]),(255,55,155),2)
            if b and counter > 10:
                orMouth = lowestMouth[2];
                b = False;
            if orMouth*0.8 > lowestMouth[2]:
                if z:
                    mouth = 3;
                else:
                    mouth = 2;
                    z = True;
            elif orMouth*0.8 <= lowestMouth[2] < 1.1*orMouth:
                mouth = 1;
                z = False;
            elif orMouth*1.1 <= lowestMouth[2] < 1.3*orMouth:
                z = True;
                mouth = 2;
            else:
                mouth = 0;
                z = False;
        if len(detectedEyes) == 0:
            blink = 0
        else:
            blink = 1
        if len(detectedEyes) >= 2:
            vec1 = (2* detectedEyes[0][0] + 0.5 * detectedEyes[0][2],2* detectedEyes[0][1] + 0.5 * detectedEyes[0][3])
            vec2 = (2* detectedEyes[1][0] + 0.5 * detectedEyes[1][2],2* detectedEyes[1][1] + 0.5 * detectedEyes[1][3])
            if a and counter > 10:
                 print "to"
                 ordis = ((vec1[0]-vec2[0])**2 + (vec1[1]-vec2[1])**2)**0.5
                 a = False;
            rotations = rots(vec1,vec2, ordis)
            yrot += rotations[0];
            zrot += rotations[1];
            rcounter += 1;
            if rcounter > 0:
                 yrot = yrot/1;
                 yrotf = yrot;
                 if zrot > 0:
                    zrotf = 1;
                 else:
                    zrotf = 0;
                 yrot = 0;
                 zrot = 0;
                 rcounter = 0
        # show the frame and record if the user presses a key
            #frame[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
        if abs(yrotf - yrotp) > 2 and yrotf > yrotp:
            print 'hi'
            yrotf = yrotp + 2;
        elif abs(yrotf - yrotp) > 2 and yrotf < yrotp:
            print 'hi'
            yrotf = yrotp - 2;
        if counter > 100:
            d1 = np.sum((facePic.astype("float") - aface.astype("float")) ** 2)
            d1 /= float(facePic.shape[0] * aface.shape[1])
            d2 = np.sum((facePic.astype("float") - sface.astype("float")) ** 2)
            d2 /= float(facePic.shape[0] * sface.shape[1])
            d3 = np.sum((facePic.astype("float") - nface.astype("float")) ** 2)
            d3 /= float(facePic.shape[0] * nface.shape[1])
            d4 = np.sum((facePic.astype("float") - hface.astype("float")) ** 2)
            d4 /= float(facePic.shape[0] * hface.shape[1])
            ma = [d1, d2, d3, d4];
            if min(ma) == d1:
                emo = 0;
            elif min(ma) == d2:
                emo = 1;
            elif min(ma) == d3:
                emo = 2;
            else:
                emo = 3;
        print yrotf
        print yrotp
        bob = mkfrm(yrotf, zrotf, blink, emo, mouth, alad);
        currframe = bck[:,:,:];
        sh = bob.shape;
        for c in range(0, 3):
            currframe[y_off:y_off + sh[0], x_off:x_off + sh[1], c] = bob[:, :, c]*(bob[:, :, 3]/255.0) + currframe[y_off:y_off + sh[0], x_off:x_off + sh[1], c] * (1.0 - bob[:, :, 3]/255.0)
        # b_channel, g_channel, r_channel, alpha_channel = cv2.split(bob)
        # b_channel = np.float64(b_channel)
        # g_channel = np.float64(g_channel)
        # r_channel = np.float64(r_channel)
        # alpha_channel = np.float64(alpha_channel)
        # bob = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        # sh = bob.shape;
        # currframe = bck
        # print currframe[y_off:y_off + sh[0], x_off:x_off + sh[1]].shape
        # lol = currframe[y_off:y_off + sh[0], x_off:x_off + sh[1]]
        # print lol.shape
        # print bob.shape
        # currframe[y_off:y_off + sh[0], x_off:x_off + sh[1]] = cv2.merge((lol, bob))
        yrotp = yrotf;
        if counter < 40:
            currframe = cv2.imread('textone.png');
        elif counter < 60:
            currframe = cv2.imread('texttwo.png');
        elif counter < 80:
            currframe = cv2.imread('textthree.png');
        elif counter < 100:
            currframe = cv2.imread('textfour.png');
        elif counter < 120:
           currframe = cv2.imread('textfive.png');
        cv2.imshow("Frame", currframe)
        currframe[:, :, :] = img2[:, :, :]
        key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break
        if key == ord("a"):
            alad = not alad;
            if alad:
                y_off = 200;
                x_off = 0;
            else:
                y_off = 250;
                x_off = 150;

        # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
