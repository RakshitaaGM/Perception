import cv2
import numpy as np
import math
# 1-1)
img = cv2.imread('Lenna.jpg')
grayImage = np.zeros(img.shape[0])
R = np.array(img[:, :, 0])
G = np.array(img[:, :, 1])
B = np.array(img[:, :, 2])
#RGB channels are separated
R = (R * 0.299)
G = (G * 0.587)
B = (B * 0.114)
Avg = (R+G+B)
grayImage = img[:,:,0] #to fix the datatype of the grayImage
grayImage[:,:] = Avg
cv2.imshow('Gray_image',grayImage)
cv2.waitKey(0)

# 1-2)
# Downsampling - Every fourth pixel is stored in a separate array
k=l=0
resized = grayImage[:64,:64]
for i in range(grayImage.shape[0]):
    for j in range(grayImage.shape[0]):
        if(i%4==0 and j%4==0):
           # print(l)
           resized[k,l] = grayImage[i,j]
           # print((i, j), (k, l), grayImage[i, j])
           l = l + 1

    l=0
    if(i%4==0):
        k=k+1
#
cv2.imshow('resized_image',resized)
cv2.waitKey(0)

# 1-3)
# Convolution with sobel kernel
sobel_x = np.zeros(grayImage.shape)
sobel_y = np.zeros(grayImage.shape)

sobel   = np.zeros(grayImage.shape)
ROWS = grayImage.shape[0]
COLS = grayImage.shape[1]
image= grayImage
print('grayImage shape',grayImage.shape)
for i in range(1,ROWS-1):
    for j in range(1,COLS-1):
        # if(i-1 > 0 and j-1 > 0 and i+1 < ROWS and j+1 < COLS):
            # print(grayImage[r-1,c-1],grayImage[r,(c - 1)],grayImage[(r + 1) , (c - 1)])
        sobel_x[i-1,j-1]  = (-1 * image[i-1][j-1]) + (0 * image[i-1][j]) + (1 * image[i-1][j+1]) + \
                            (-2 * image[i][j-1])   + (0 * image[i][j])   + (2 * image[i][j+1])   + \
                            (-1 * image[i+1][j-1]) + (0 * image[i+1][j]) + (1 * image[i+1][j+1])
        sobel_y[i-1, j-1] = (-1 * image[i-1][j-1]) + (-2 * image[i-1][j]) + (-1 * image[i-1][j+1]) + \
                            (0 * image[i][j-1])    + (0 * image[i][j])    + (0 * image[i][j+1]) + \
                            (1 * image[i+1][j-1]) +  (2 * image[i+1][j])  + (1 * image[i+1][j+1])

        sobel[i-1 , j-1] = math.sqrt((sobel_x[i-1,j-1])*(sobel_x[i-1,j-1]) + (sobel_y[i-1 , j-1])*(sobel_y[i-1 , j-1]))

maximum = np.amax(sobel)
minimum = np.amin(sobel)
sobel_disp = np.zeros(grayImage.shape, dtype = np.uint8)
for r in range(1,ROWS-1):
    for c in range(1,COLS-1):
        sobel_disp[r,c] = (sobel[r , c] - minimum) * (255-0) / (maximum - minimum)

cv2.imshow('Sobel_operated',sobel_disp)
cv2.waitKey(0)
#2-1)
#Plotting histogram
SIZE = ROWS*COLS
import matplotlib.pyplot as plt
# hist = np.zeros((SIZE,1))
#
# for r in range(grayImage.shape[0]):
#     for c in range(grayImage.shape[1]):
#         hist[r*COLS+c] = grayImage[r,c]

#Histogram analysis and plottinh histogram distribution
#For Grayscale Image
SIZE = ROWS*COLS
import matplotlib.pyplot as plt
# hist = np.zeros((SIZE,1))
hist = np.zeros(255)
for r in range(grayImage.shape[0]):
    for c in range(grayImage.shape[1]):
        hist[grayImage[r,c]] +=1
# print(hist)
x = np.arange(255)
for i in range(0,255):
    x[i]=i
plt.bar(x,hist)
plt.title('Histogram plot')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()
# 2-2)
cumulative = np.zeros(255)
for i in range(255):
    if(i==0):
        cumulative[i]=hist[i]
    else:
        cumulative[i] = hist[i]+cumulative[i-1]
# print(cumulative)
plt.bar(x,cumulative)
plt.title('Accumulative Histogram Distribution')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()
#2-3) Histogram equalization
# Normalize

for i in range(255):
    cumulative[i]=(cumulative[i]-cumulative.min())*255/(cumulative.max()-cumulative.min())
cumulative = cumulative.astype('uint8')
plt.bar(x,cumulative)
plt.show()
def equalize(cumulative):
    img_new = cumulative[grayImage.ravel()]
    print(cumulative.shape,img_new.shape)
    img_new = np.reshape(img_new,grayImage.shape)
    return(img_new)

img_new = equalize(cumulative)
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)
fig.add_subplot(1,2,1)
plt.imshow(grayImage, cmap='gray')

# display the new image
fig.add_subplot(1,2,2)
plt.imshow(img_new, cmap='gray')
plt.show(block=True)
# print(img_new.dtype)
hist_new = np.zeros(256)
print(hist_new.shape)
print(img_new.max(),grayImage.max())
#For flattening the image
# # print(img_new)
for i in range(img_new.shape[0]):
    for j in range(img_new.shape[1]):
        # print(hist_new[img_new[i,j]])
        hist_new[img_new[i,j]] +=1
x = np.arange(256)
plt.bar(x,hist_new)
plt.title('Histogram plot - Equalized')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.show()
#Reference:https://medium.com/hackernoon/histogram-equalization-in-python-from-scratch-ebb9c8aa3f23







