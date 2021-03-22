
%%
net = myNet;
lgraph = layerGraph(net);
net.Layers

analyzeNetwork(lgraph)
%%
lgraph = removeLayers(lgraph,{'fc2','softmax','classoutput'});

numClasses = numel(categories(trainingImages.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc21','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','fc2w_softmax1')
    classificationLayer('Name','ClassificationLayer_1')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'drop1','fc21');
analyzeNetwork(lgraph)

%
%% Set up our training data
allImages = imageDatastore('skinData', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
allImages.ReadFcn = @customReadDatastoreImage;
[trainingImages, testImages] = splitEachLabel(allImages, 0.80, 'randomize');
%% Re-train the Network
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 100, 'MiniBatchSize', 8,'Plots','training-progress');
myNet1 = trainNetwork(trainingImages, lgraph, opts);
%% Measure network accuracy
predictedLabels = classify(myNet1, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)
%%
save skinAfterTransferLearning myNet1
%%
function data = customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[500 375]);
end

