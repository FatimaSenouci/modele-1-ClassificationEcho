clc;clear;close all
outputFolder =fullfile('Data')
rootFolder = fullfile(outputFolder,'dataecho'); % load data 
categories = {'endometreSAIN','endometreMalade'}; %we are choosing 2 categories 
  imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames','readfcn',@CustomFcn); %create image data stror to manage data so the images in categories are now in imds 
  %count the number of images in categories 
  tbl = countEachLabel(imds)
  %wish categorie has least number 
  minSetCount = min(tbl{:,2})
  %reduce each categorie  so we update our imds 
  imds =splitEachLabel(imds,minSetCount,'randomize');
  countEachLabel(imds);
endometreSAIN = find(imds.Labels == ' endometreSAIN' ,1);
  endometreMalade = find(imds.Labels == 'endometreMalade' ,1);
 
 
  figure
  subplot(2,2,1);
  imshow(readimage(imds,endometreSAIN));
  subplot(2,2,2);
  imshow(readimage(imds,endometreMalade));
  
  
  [tr,tst]=splitEachLabel(imds,.70,'randomized');

%Define Network Architecture
layers = [
    imageInputLayer([100 100 3])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
  
 %Specify Training Options
options = trainingOptions('sgdm', ...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-4, ...
    'Verbose',true, ...
    'verbosefrequency',1,...
    'Plots','training-progress');


% title('first convolution layer weights');
% %Train Network Using Training Data
% net = trainNetwork(imdsTrain,layers,options);
% %Classify Validation Images and Compute Accuracy
% YPred = classify(net,imdsValidation);
% YValidation = imdsValidation.Labels;
% Lbls=imdsValidation.Labels;
% plotconfusion(Lbls,YPred)
% save network.mat net
% 
% accuracy = sum(YPred == YValidation)/numel(YValidation)
net = trainNetwork(tr,layers,options);

YPredicted = classify(net,tst);
Lbls=tst.Labels;
plotconfusion(Lbls,YPredicted)
save network.mat net
