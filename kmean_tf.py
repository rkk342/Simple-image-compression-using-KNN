import numpy as np
import cv2
from numpy.core.fromnumeric import argmin, transpose
import time 
import tensorflow as tf

path = r'c:\Users\Lord\Programming\Python\DONE\KMeans\peppers-large.tiff'
img = cv2.imread(path, cv2.IMREAD_COLOR)

cluster = 20

clusterspos = []
for i in range(cluster):
    clusterspos.append([])
clusterscol = []
for i in range(cluster):
    clusterscol.append([])

img_temp = img.copy()
img_temp = img_temp.transpose()

# Random cluster points
initial_0 = np.random.randint(np.amin(img_temp[0]), np.amax(img_temp[0]), cluster)
initial_1 = np.random.randint(np.amin(img_temp[1]), np.amax(img_temp[1]), cluster)
initial_2 = np.random.randint(np.amin(img_temp[2]), np.amax(img_temp[2]), cluster)

initial = np.concatenate((initial_0, initial_1, initial_2))
initial = initial.reshape(cluster,3)
# end

change = 6
ok = 21
time0 = time.time()
niters = 1

#tensorflow


while change > 5:

    img1 = img.reshape(img.shape[0]*img.shape[1],3)
    img1 = tf.convert_to_tensor(img1)
    img2 = tf.repeat(img1, repeats=cluster, axis=0)
    initial_ = tf.tile(initial, (img1.shape[0],1))
    img2 = tf.cast(img2, tf.int32)
    final = tf.math.subtract(img2, initial_)
    final = tf.square(final)
    final = tf.matmul(final, tf.ones([3,1], dtype=tf.dtypes.int32))
    final = tf.reshape(final, [img1.shape[0],cluster])
    ok_old = ok
    initial_old = initial.copy()
    final = tf.argmin(final, axis=1)

    for i in range(cluster):
        temp = tf.where(final==i)
        clusterspos[i] = temp
        initial[i] = tf.reduce_mean(tf.gather(img1, temp), axis=0)
    
    ok = np.absolute(initial_old-initial)
    ok = np.sum(ok)
    change = abs(ok_old-ok)
    print(niters)
    niters = niters + 1

time1 = time.time()
print('time for iteration',time1-time0)

img1 = img1.numpy()

for k in range(cluster):
    img1[np.array(clusterspos[k])] = initial[k]

img1 = img1.reshape(img.shape[0], img.shape[1], 3)

cv2.namedWindow('voilla', cv2.WINDOW_FREERATIO)
cv2.imshow('voilla', img1)
cv2.waitKey(0)
cv2.destroyAllWindows

print('end')
