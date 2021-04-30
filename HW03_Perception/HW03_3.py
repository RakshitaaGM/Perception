import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

image = cv2.imread('ParkingLot.jpg')
img_cpy =  np.copy(image)
img_cpy_1 =  np.copy(image)


# 3-1)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.hist(gray.ravel(),255)
plt.title("Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")
plt.show()
#Selected threshold is 165
ret,thresh1 = cv2.threshold(gray,165,255,cv2.THRESH_BINARY)
cv2.imshow('Thresholded Binary image',thresh1)
cv2.waitKey(0)
blur = cv2.GaussianBlur(gray,(5, 5), 0)
# 3-2)
canny = cv2.Canny(blur, 50, 100)
lines = cv2.HoughLinesP(canny, 2, np.pi/180, 100,np.array([]), minLineLength= 40, maxLineGap=5)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(image,(x1, y1),(x2,y2),(255, 0, 0),3)
    # plt.plot((x1, y1),(x2,y2))
cv2.imshow("Lines in Image Space", image)
cv2.waitKey(0)
# plt.show()

perpendicular = []
parallel = []
intersection = []
blue = []
green = []
# 3-3)
for line in lines:
    x1, y1, x2, y2 = line[0]
    m = (y2 - y1) / (x2 - x1)
    distance = math.sqrt((x2 - x1)^2 +(y2 - y2)^2)
    if(distance > 10 and distance < 11):
        # print('green')
        # print((x1, y1, x2, y2))
        # print(m)
        green.append((x1, y1, x2, y2))
        cv2.line(img_cpy_1,(x1, y1),(x2,y2),(0, 255, 0),3)
    if(distance >= 11 and distance < 13):
        if(m>0):
            blue.append((x1, y1, x2, y2))

        # print('blue')
        # print(x1, y1, x2, y2)
        cv2.line(img_cpy_1, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # print(m)
    if (distance >= 13 and distance < 15):
        # print('red')
        # print(x1, y1, x2, y2)
        cv2.line(img_cpy_1, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # perpendicular.append((x1, y1, x2, y2))
        # print(m)

    if(m<0):
        perpendicular.append((x1, y1, x2, y2))
    elif(m>0):
        parallel.append((x1, y1, x2, y2))
cv2.imshow("Colored lines", img_cpy_1)
cv2.waitKey(0)
midpoints = []
for line in blue:
    x1,y1,x2,y2 = line
    mp_x, mp_y = ((x1+x2)/2),((y1+y2)/2)
    midpoints.append([mp_x, mp_y, (x1,y1,x2,y2)])
    # print(mp_x,mp_y)
for line in green:
    x1,y1,x2,y2 = line
    mp_x, mp_y = ((x1+x2)/2),((y1+y2)/2)
    midpoints.append([mp_x, mp_y,(x1,y1,x2,y2)])
    # print(mp_x,mp_y)

midpoints.sort()
for i in range(len(midpoints)-1):
    mp_x,mp_y,(x1,y1,x2,y2) = midpoints[i]
    mp_x_1, mp_y_1, (x3, y3, x4, y4) = midpoints[i+1]
    polygon = np.array([[(mp_x, mp_y), (mp_x_1, mp_y_1), (x4, y4), (x2,y2)]])
    cv2.fillPoly(img_cpy, np.array(polygon, 'int32'), color=(i*10, i*10, i*5))
    polygon = np.array([[(mp_x, mp_y), (mp_x_1, mp_y_1), (x3, y3), (x1, y1)]])
    cv2.fillPoly(img_cpy, np.array(polygon, 'int32'), color=(i * 5, i * 10, i * 10))


cv2.imshow("Polygon on parking spaces", img_cpy)
cv2.waitKey(0)
cv2.destroyAllWindows()




