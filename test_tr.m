clc;clear all;close all
[fle,pth]=uigetfile('*.jpg','choose image');
I=imread([pth,fle]);
I2=CustomFcn([pth,fle]);
load network
ca=classify(net,I2);
imshow(I)
title(string(ca))