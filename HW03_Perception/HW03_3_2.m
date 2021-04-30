%3-2)
image = imread('ParkingLot.jpg');
imshow(image);
title('Original gray scale image')
figure;
imhist(image);
BW = imbinarize(image,0.65);
imshow(BW)
title('Binarized image')
[H,theta,rho] = hough(BW,'RhoResolution',0.5,'Theta',-90:0.5:89);
subplot(2,1,1);
imshow(image);
title('Original Image.png');
subplot(2,1,2);
imshow(imadjust(rescale(H)),'XData',theta,'YData',rho,...
      'InitialMagnification','fit');
title('Hough transform of parking lot.png');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
colormap(gca,hot);

