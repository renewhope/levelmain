clc
clear all
close all
I = imread('image_56.jpg');
I2=I;
I=imresize(I,[600 600]);
Imgg=imresize(I,[256 256]);
imshow(I)
title('Original Image')

% Create mask and specify seed location. You can also use roipoly to create the mask interactively.
mask = false(size(I)); 
mask(170,70) = true;

% Compute the weight array based on grayscale intensity differences.
W = graydiffweight(I, mask, 'GrayDifferenceCutoff', 25);

% Segment the image using the weights.
thresh = 0.01;
[BW, D] = imsegfmm(W, mask, thresh);
dd=D(:,:,1)>0.1;

st=strel('disk',18);

d1=imerode(dd,st);

mul=immultiply(d1,I(:,:,1));

Img1 = imresize(mul,[256 256]);
Img=double(Img1(:,:,1));   
G=fspecial('gaussian',5);
Img_smooth=conv2(Img,G,'same');  
[Ix,Iy]=gradient(Img_smooth);
f=Ix.^2+Iy.^2;
g=1./(1+f);    
equldis=2; weight=6;   
width = 256;
height = 256;
radius = 10;
centerW = width/3.3;
centerH = height/2.3;
[W,H] = meshgrid(1:width,1:height);
mask = ((W-centerW).^2 + (H-centerH).^2) < radius^2;


%  mask=roipoly(Img1)
if  mean2(I2)>50
mask=imread('mask1.jpg');
else
mask=imread('mask.jpg');
end

BW = double(im2bw(mask)); 
% BW=mask;
[nrow, ncol]=size(Img1);
c0=4; 
initialLSF= -c0*2*(0.5-BW); 
u=initialLSF;
u=initialLSF;
evolution=230;
% move evolution
for n=1:evolution
    u=levelset(u, g ,equldis, weight);    
    if mod(n,20)==0
         pause(1);
        figure(4),imshow(Imgg, [0, 255]);colormap(gray);hold on;
        [c,h] = contour(u,[0 0],'r');        
        title('level set');
        hold off;
    end
end

u=imfill(u,'holes');

% u=immultiply(u,u1);
st=strel('disk',2);
u2=imdilate(u,st);
u1=double(imclearborder(im2bw(u)));
imwrite(u1,'seg.jpg')
st1=strel('disk',1);
aa=double(imread('seg.jpg'));
aa=imerode(aa,st1);

figure,
imshow(Imgg, [0, 255]);colormap(gray);hold on;
[c,h] = contour(1-aa,[0 0],'r');

segg=immultiply(u1,double(rgb2gray(Imgg)));


figure,
subplot(2,2,1)
imshow(I)
title('input')

subplot(2,2,2)
imshow(u1)
title('binary')

subplot(2,2,4)
imshow(uint8(segg))
title('segment')

subplot(2,2,3)
imshow(Imgg, [0, 255]);colormap(gray);hold on;
[c,h] = contour(1-aa,[0 0],'r');
title('boundary')





