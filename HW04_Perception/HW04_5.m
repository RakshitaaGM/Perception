fileID = fopen('002_00000000.bin');
A = fread(fileID,'single');
x = [];y=[];z=[];intensity=[];
for i=1:4:length(A)
     x = [x,A(i)];
 end
for j=2:4:length(A)
    y = [y,A(j)];
 end

for k=3:4:length(A)
    z = [z,A(k)];
end 
for l=4:4:length(A)
   intensity = [intensity,A(l)];
end
%%
%%5-1
[theta,rho,phi] = cart2sph(x,y,z);
figure(1);
polarplot = polarscatter(theta,phi,0.4,rho);
%%5-2
figure(2);
polarplot = polarscatter(theta,phi,0.4,intensity);
title('Polar Coordinate to 2D Depth image visulaization of Lidar data');


