netWidth = 32;
layers = [
    imageInputLayer([224 224 3],'Name','input')
    convolution2dLayer(3,netWidth,'Padding','same','Name','convInp')
    batchNormalizationLayer('Name','BNInp')
    reluLayer('Name','reluInp')
    
   convolution2dLayer(5,netWidth,'Padding','same','Stride',2,'Name','ConvQ1')
    batchNormalizationLayer('Name','BNQ1')
    reluLayer('Name','reluQ1')
    
    %part1
     convolution2dLayer(1,netWidth,'Padding','same','Stride',1,'Name','ConvQ111')
    batchNormalizationLayer('Name','BNQ111')
    reluLayer('Name','reluQ111')
    
   depthConcatenationLayer(5,'Name','concat_1')
   batchNormalizationLayer('Name','BNQ113')
   %part2
   convolution2dLayer(1,2*netWidth,'Padding','same','Stride',2,'Name','ConvQ1111')
    batchNormalizationLayer('Name','BNQ1111')
    reluLayer('Name','reluQ1111')
    
   depthConcatenationLayer(4,'Name','concat_11')
   batchNormalizationLayer('Name','BNQ117')
   %part3
    convolution2dLayer(1,2*netWidth,'Padding','same','Stride',1,'Name','ConvQ11111')
    batchNormalizationLayer('Name','BNQ11111')
    reluLayer('Name','reluQ11111')
    
   depthConcatenationLayer(5,'Name','concat_111')
   batchNormalizationLayer('Name','BNQ1177')
  %part4 
    convolution2dLayer(1,4*netWidth,'Padding','same','Stride',2,'Name','ConvQ111111')
    batchNormalizationLayer('Name','BNQ111111')
    reluLayer('Name','reluQ111111')
    
   depthConcatenationLayer(5,'Name','concat_1111')
   batchNormalizationLayer('Name','BNQ11777')
 %part5  
   convolution2dLayer(1,8*netWidth,'Padding','same','Stride',1,'Name','ConvQ1111111')
    batchNormalizationLayer('Name','BNQ1111111')
    reluLayer('Name','reluQ1111111')
 
   depthConcatenationLayer(6,'Name','concat_11111')
   batchNormalizationLayer('Name','BNQ117777')
    
    %part6  
   convolution2dLayer(1,12*netWidth,'Padding','same','Stride',2,'Name','ConvQ111111e0')
    batchNormalizationLayer('Name','BNQ1111111e0')
    reluLayer('Name','reluQ1111111e0')
 
   depthConcatenationLayer(8,'Name','concat_11111e')
   batchNormalizationLayer('Name','BNQ117777e')
    
   
    averagePooling2dLayer(14,'Stride',1,'Name','globalPool')
    fullyConnectedLayer(1200,'Name','fc1')
     dropoutLayer('Name','drop1')
    fullyConnectedLayer(4,'Name','fc2')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];
    
%%
lgraph = layerGraph(layers);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);

plot(lgraph);
%% part1
Skip1= [
    convolution2dLayer(3,netWidth,'Padding','same','Stride',1,'Name','skipConvQ1')
    batchNormalizationLayer('Name','skipBNQ1')
    reluLayer('Name','skipreluQ1')]
   
lgraph = addLayers(lgraph,Skip1);
lgraph = connectLayers(lgraph,'reluQ1','skipConvQ1');
lgraph = connectLayers(lgraph,'skipreluQ1','concat_1/in2');

%%
skip2 = [
    convolution2dLayer(5,netWidth,'Padding','same','Stride',1,'Name','skipConvU1')
    batchNormalizationLayer('Name','skipBNU1')
    reluLayer('Name','skipreluU1')]

lgraph = addLayers(lgraph,skip2);
lgraph = connectLayers(lgraph,'reluQ1','skipConvU1');
lgraph = connectLayers(lgraph,'skipreluU1','concat_1/in3');

%%

skip3 = [
    convolution2dLayer(7,netWidth,'Padding','same','Stride',1,'Name','skipConvT1')
    batchNormalizationLayer('Name','skipBNT1')
    reluLayer('Name','skipreluT1')]
   

lgraph = addLayers(lgraph,skip3);
lgraph = connectLayers(lgraph,'reluQ1','skipConvT1');
lgraph = connectLayers(lgraph,'skipreluT1','concat_1/in4');

lgraph = connectLayers(lgraph,'reluQ1','concat_1/in5');
%% part2  ////////////////////////////////////////////////////////////////
Skip11= [
    convolution2dLayer(3,2*netWidth,'Padding','same','Stride',2,'Name','skipConvQ11')
    batchNormalizationLayer('Name','skipBNQ11')
    reluLayer('Name','skipreluQ11')]
   
lgraph = addLayers(lgraph,Skip11);
lgraph = connectLayers(lgraph,'BNQ113','skipConvQ11');
lgraph = connectLayers(lgraph,'skipreluQ11','concat_11/in2');

%%
skip22 = [
    convolution2dLayer(5,2*netWidth,'Padding','same','Stride',2,'Name','skipConvU11')
    batchNormalizationLayer('Name','skipBNU11')
    reluLayer('Name','skipreluU11')]

lgraph = addLayers(lgraph,skip22);
lgraph = connectLayers(lgraph,'BNQ113','skipConvU11');
lgraph = connectLayers(lgraph,'skipreluU11','concat_11/in3');

%%

skip33 = [
    convolution2dLayer(7,2*netWidth,'Padding','same','Stride',2,'Name','skipConvT11')
    batchNormalizationLayer('Name','skipBNT11')
    reluLayer('Name','skipreluT11')]
   

lgraph = addLayers(lgraph,skip33);
lgraph = connectLayers(lgraph,'BNQ113','skipConvT11');
lgraph = connectLayers(lgraph,'skipreluT11','concat_11/in4');

%% part 3 //////////////////////////////////////////////////
Skip111= [
    convolution2dLayer(3,2*netWidth,'Padding','same','Stride',1,'Name','skipConvQ111')
    batchNormalizationLayer('Name','skipBNQ111')
    reluLayer('Name','skipreluQ111')]
   
lgraph = addLayers(lgraph,Skip111);
lgraph = connectLayers(lgraph,'BNQ117','skipConvQ111');
lgraph = connectLayers(lgraph,'skipreluQ111','concat_111/in2');

%%
skip222 = [
    convolution2dLayer(5,2*netWidth,'Padding','same','Stride',1,'Name','skipConvU111')
    batchNormalizationLayer('Name','skipBNU111')
    reluLayer('Name','skipreluU111')]

lgraph = addLayers(lgraph,skip222);
lgraph = connectLayers(lgraph,'BNQ117','skipConvU111');
lgraph = connectLayers(lgraph,'skipreluU111','concat_111/in3');

%%

skip333 = [
    convolution2dLayer(7,2*netWidth,'Padding','same','Stride',1,'Name','skipConvT111')
    batchNormalizationLayer('Name','skipBNT111')
    reluLayer('Name','skipreluT111')]
   

lgraph = addLayers(lgraph,skip333);
lgraph = connectLayers(lgraph,'BNQ117','skipConvT111');
lgraph = connectLayers(lgraph,'skipreluT111','concat_111/in4');

lgraph = connectLayers(lgraph,'BNQ117','concat_111/in5');
%% part4/////////////////////////
Skip1111= [
    convolution2dLayer(3,4*netWidth,'Padding','same','Stride',2,'Name','skipConvQ1111')
    batchNormalizationLayer('Name','skipBNQ1111')
    reluLayer('Name','skipreluQ1111')]
   
lgraph = addLayers(lgraph,Skip1111);
lgraph = connectLayers(lgraph,'BNQ1177','skipConvQ1111');
lgraph = connectLayers(lgraph,'skipreluQ1111','concat_1111/in2');

%%
skip2222 = [
    convolution2dLayer(5,4*netWidth,'Padding','same','Stride',2,'Name','skipConvU1111')
    batchNormalizationLayer('Name','skipBNU1111')
    reluLayer('Name','skipreluU1111')]

lgraph = addLayers(lgraph,skip2222);
lgraph = connectLayers(lgraph,'BNQ1177','skipConvU1111');
lgraph = connectLayers(lgraph,'skipreluU1111','concat_1111/in3');

%%

skip3333 = [
    convolution2dLayer(7,4*netWidth,'Padding','same','Stride',2,'Name','skipConvT1111')
    batchNormalizationLayer('Name','skipBNT1111')
    reluLayer('Name','skipreluT1111')]
   

lgraph = addLayers(lgraph,skip3333);
lgraph = connectLayers(lgraph,'BNQ1177','skipConvT1111');
lgraph = connectLayers(lgraph,'skipreluT1111','concat_1111/in4');



%% part5
Skip11111= [
    convolution2dLayer(3,8*netWidth,'Padding','same','Stride',1,'Name','skipConvQ11111')
    batchNormalizationLayer('Name','skipBNQ11111')
    reluLayer('Name','skipreluQ11111')]
   
lgraph = addLayers(lgraph,Skip11111);
lgraph = connectLayers(lgraph,'BNQ11777','skipConvQ11111');
lgraph = connectLayers(lgraph,'skipreluQ11111','concat_11111/in2');

%%
skip22222 = [
    convolution2dLayer(5,8*netWidth,'Padding','same','Stride',1,'Name','skipConvU11111')
    batchNormalizationLayer('Name','skipBNU11111')
    reluLayer('Name','skipreluU11111')]

lgraph = addLayers(lgraph,skip22222);
lgraph = connectLayers(lgraph,'BNQ11777','skipConvU11111');
lgraph = connectLayers(lgraph,'skipreluU11111','concat_11111/in3');

%%

skip33333 = [
    convolution2dLayer(7,8*netWidth,'Padding','same','Stride',1,'Name','skipConvT11111')
    batchNormalizationLayer('Name','skipBNT11111')
    reluLayer('Name','skipreluT11111')]
   

lgraph = addLayers(lgraph,skip33333);
lgraph = connectLayers(lgraph,'BNQ11777','skipConvT11111');
lgraph = connectLayers(lgraph,'skipreluT11111','concat_11111/in4');

lgraph = connectLayers(lgraph,'BNQ11777','concat_11111/in5');


%% %% part6
Skip111111= [
    convolution2dLayer(3,12*netWidth,'Padding','same','Stride',2,'Name','skipConvQ11111e1')
    batchNormalizationLayer('Name','skipBNQ11111e1')
    reluLayer('Name','skipreluQ11111e1')];
   
lgraph = addLayers(lgraph,Skip111111);
lgraph = connectLayers(lgraph,'BNQ117777','skipConvQ11111e1');
lgraph = connectLayers(lgraph,'skipreluQ11111e1','concat_11111e/in2');

%%
skip222222 = [
    convolution2dLayer(5,12*netWidth,'Padding','same','Stride',2,'Name','skipConvU11111e2')
    batchNormalizationLayer('Name','skipBNU11111e2')
    reluLayer('Name','skipreluU11111e2')];

lgraph = addLayers(lgraph,skip222222);
lgraph = connectLayers(lgraph,'BNQ117777','skipConvU11111e2');
lgraph = connectLayers(lgraph,'skipreluU11111e2','concat_11111e/in3');

%%

skip333333 = [
    convolution2dLayer(7,12*netWidth,'Padding','same','Stride',2,'Name','skipConvT11111e3')
    batchNormalizationLayer('Name','skipBNT11111e3')
    reluLayer('Name','skipreluT11111e3')];
   

lgraph = addLayers(lgraph,skip333333);
lgraph = connectLayers(lgraph,'BNQ117777','skipConvT11111e3');
lgraph = connectLayers(lgraph,'skipreluT11111e3','concat_11111e/in4');



%%
%Extra

SkipE1= [
    convolution2dLayer(3,netWidth,'Padding','same','Stride',4,'Name','skipConvQEE1')
    batchNormalizationLayer('Name','skipBNQEE1')
    reluLayer('Name','skipreluQ1EE')];
   
lgraph = addLayers(lgraph,SkipE1);
lgraph = connectLayers(lgraph,'BNQ113','skipConvQEE1');
lgraph = connectLayers(lgraph,'skipreluQ1EE','concat_11111/in6');
%%
SkipE11= [
    convolution2dLayer(3,netWidth,'Padding','same','Stride',2,'Name','skipConvQEEE1')
    batchNormalizationLayer('Name','skipBNQEEE1')
    reluLayer('Name','skipreluQ1EEE')];
   
lgraph = addLayers(lgraph,SkipE11);
lgraph = connectLayers(lgraph,'BNQ117','skipConvQEEE1');
lgraph = connectLayers(lgraph,'skipreluQ1EEE','concat_1111/in5');
%%
SkipE111= [
    convolution2dLayer(3,netWidth,'Padding','same','Stride',8,'Name','skipConvQEE1w')
    batchNormalizationLayer('Name','skipBNQEE1w')
    reluLayer('Name','skipreluQ1EEw')];
   
lgraph = addLayers(lgraph,SkipE111);
lgraph = connectLayers(lgraph,'reluQ1','skipConvQEE1w');
lgraph = connectLayers(lgraph,'skipreluQ1EEw','concat_11111e/in5');
%%
SkipE1111= [
    convolution2dLayer(3,netWidth,'Padding','same','Stride',8,'Name','skipConvQEE1g')
    batchNormalizationLayer('Name','skipBNQEE1g')
    reluLayer('Name','skipreluQ1EEg')];
   
lgraph = addLayers(lgraph,SkipE1111);
lgraph = connectLayers(lgraph,'BNQ113','skipConvQEE1g');
lgraph = connectLayers(lgraph,'skipreluQ1EEg','concat_11111e/in6');
%%
%%
SkipE11111= [
    convolution2dLayer(3,netWidth,'Padding','same','Stride',4,'Name','skipConvQEEE1v')
    batchNormalizationLayer('Name','skipBNQEEE1v')
    reluLayer('Name','skipreluQ1EEEv')];
   
lgraph = addLayers(lgraph,SkipE11111);
lgraph = connectLayers(lgraph,'BNQ117','skipConvQEEE1v');
lgraph = connectLayers(lgraph,'skipreluQ1EEEv','concat_11111e/in7');
%%
%%
SkipE111111= [
    convolution2dLayer(3,netWidth,'Padding','same','Stride',2,'Name','skipConvQEEE1v1')
    batchNormalizationLayer('Name','skipBNQEEE1v1')
    reluLayer('Name','skipreluQ1EEEv1')];
   
lgraph = addLayers(lgraph,SkipE111111);
lgraph = connectLayers(lgraph,'BNQ11777','skipConvQEEE1v1');
lgraph = connectLayers(lgraph,'skipreluQ1EEEv1','concat_11111e/in8');
%%
analyzeNetwork(lgraph)

%% Set up our training data
allImages = imageDatastore('DFU-UK', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainingImages, testImages] = splitEachLabel(allImages, 0.80, 'randomize');
%% Re-train the Network
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 100, 'MiniBatchSize', 64,'Plots','training-progress');
myNet = trainNetwork(trainingImages, lgraph, opts);
%% Measure network accuracy
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)

%% https://www.mathworks.com/help/deeplearning/ref/predict.html
YPred = predict(myNet,testImages)
YPred(1:2,:)
testImages.Labels
%%
save SkinNetwork myNet