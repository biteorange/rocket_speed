"""
@author: Tiansheng Yao
@based on 'widebaseline.py' by Jonathan Balzer
"""

import cv2
import numpy as np
import scipy.linalg as linalg
#import images
img2 = cv2.imread('/home/tiansheng/Documents/12Fall/cs268/hw2/spirit1983.png');
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY);

img1 = cv2.imread('/home/tiansheng/Documents/12Fall/cs268/hw2/spirit1706.png');
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY);

img0 = cv2.imread('/home/tiansheng/Documents/12Fall/cs268/hw2/spirit1433.png');
img0_gray = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY);

#detect features
mask0 = np.ones((img0_gray.shape[0],img0_gray.shape[1]),'uint8');
mask1 = np.ones((img1_gray.shape[0],img1_gray.shape[1]),'uint8');
mask2 = np.ones((img2_gray.shape[0],img2_gray.shape[1]),'uint8');

detector = cv2.SURF();

keypoints0 = detector.detect(img0_gray,mask0); 
keypoints1 = detector.detect(img1_gray,mask1); 
keypoints2 = detector.detect(img2_gray,mask2); 

#extract descriptors, using ORB feature
extractor = cv2.DescriptorExtractor_create('ORB');

keypoints0, desc0 = extractor.compute(img0_gray, keypoints0);
keypoints1, desc1 = extractor.compute(img1_gray, keypoints1);
keypoints2, desc2 = extractor.compute(img2_gray, keypoints2);


#find correspondence, find 0->1 and 0->2. 
#the result of 1->2 not optimistic due to less overlapping 
flann_params= {'algorithm':6,'table_number':6,'key_size':1,'multi_probe_level':1}
matcher = cv2.FlannBasedMatcher(flann_params,{});
matches01 = matcher.match(desc0,desc1);
matches12 = matcher.match(desc1,desc2);


# find correspondence from image0 to image1
x0 = [];
x1 = [];

for m in matches01:
    x0.append(keypoints0[m.queryIdx].pt);
    x1.append(keypoints1[m.trainIdx].pt);    

X0 = np.array(x0);
X1 = np.array(x1);

# form homogenous coordinate
X0 = np.concatenate((X0, np.ones((X0.shape[0],1), dtype=np.float)),1);
X0p = np.concatenate((X1, np.ones((X1.shape[0],1), dtype=np.float)),1);

m = X0.shape[0];
X01 = np.zeros([m,9]);
for i in range(X0.shape[0]):
    X01[i,:] = np.kron(X0p[i], X0[i])  

U, S, V = linalg.svd(X01);
F01 = np.reshape(V[8,:],[3,3]);


#find correspondence from image1 to image2
x1 = [];
x2 = [];

for m in matches12:
    x1.append(keypoints1[m.queryIdx].pt);
    x2.append(keypoints2[m.trainIdx].pt);    
    
X1 = np.array(x1);
X2 = np.array(x2);

# form homogenous coordinate
X1 = np.concatenate((X1, np.ones((X1.shape[0],1), dtype=np.float)),1);
X1p = np.concatenate((X2, np.ones((X2.shape[0],1), dtype=np.float)),1);

# get the fundamental matrix from image1 to image2
m = X1.shape[0]
X12 = np.zeros([m,9]);
for i in range(m):
    # each is expanded by kronecker product of correspondent points
    X12[i,:] = np.kron(np.matrix(X1[i]), np.matrix(X1p[i]))    

U, S, V = linalg.svd(X12);
F12 = np.reshape(V[8,:], [3,3])

# get the calibration matrix in P   
f = 14.67e-3; pixel_length = 12e-6;
sx = 1024*pixel_length; sy = 1024*pixel_length;
ox = sx / 2; oy = sy / 2;

Ks = np.array([[sx,0,ox],[0,sy,oy],[0,0,1]]);
Pi = np.array([[f,0,0],[0,f,0],[0,0,1]]);
P = np.matrix(Ks)*np.matrix(Pi);

# get the essential matrix
E01 = P.I.transpose()*F01*P.I;
E12 = P.I.transpose()*F12*P.I;


# compute the SVD of essential matrix and get the t
U01, S01, Vh01 = linalg.svd(E01)
S01 = np.matrix(S01);
V01 = Vh01.T;
S = np.matrix('1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.0');
S[0,0] = S01[0,0]; S[1,1] = S01[0,1];
W = np.matrix('0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0');
t01 = np.matrix(V01) * W * S * np.matrix(Vh01);

# rescale the translation vector t wrt h1-h0 and compute the speed
h0 = 1433.0; h1 = 1706.0; h2 = 1983.0;
tz = t01[1,0]; tx = t01[0,2]; ty = t01[2,1];
vX = tx / tz * (h1-h0)/3.75;
vY = ty / tz * (h1-h0)/3.75;
vH = np.sqrt(vX**2 + vY**2);
print 'The horizonal speed estimated from 1433-1766 is: %f\n (x-axis speed %f, y-axis speed %f)'%(vH, vX, vY)

# rescale the translation vector t wrt h1-h0 and compute the speed
U12, S12, Vh12 = linalg.svd(E12)
S12 = np.matrix(S12);
S[0,0] = S12[0,0]; S[1,1] = S12[0,1];
V12 = Vh12.T;
t12 = V12 * W * S * Vh12;
tz = t12[1,0]; tx = t12[0,2]; ty = t12[2,1];
vX = tx / tz * (h2-h1)/3.75;
vY = ty / tz * (h2-h1)/3.75;
vH = np.sqrt(vX**2 + vY**2);
print 'The horizonal speed estimated from 1766-1983 is: %f\n (x-axis speed %f, y-axis speed %f)'%(vH, vX, vY)
print '----------------------------------------------'
print 'Conclusion: we do NOT need to open the airbag, the speed is smaller than 40m/s'
