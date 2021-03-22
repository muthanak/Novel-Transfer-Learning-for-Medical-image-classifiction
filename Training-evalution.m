%% Set up our training data
allImages = imageDatastore('Cells', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[trainingImages, testImages] = splitEachLabel(allImages, 0.10, 'randomize');
%% Re-train the Network
opts = trainingOptions('sgdm', 'InitialLearnRate', 0.001, 'MaxEpochs', 70, 'MiniBatchSize', 64,'Plots','training-progress');
myNet = trainNetwork(trainingImages, lgraph, opts);
%% Measure network accuracy
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)
%% confusion metrix 
RE= testImages.Labels;
cm = confusionmat(RE, predictedLabels);
[cm,order] = confusionmat(RE, predictedLabels,'Order',{'Abnormal(Ulcer)','Normal(Healthy skin)'}) 
cm1= bsxfun (@rdivide, cm, sum(cm,2))
mean(diag(cm1))

confusionchart(cm1)
%%
tp_m = diag(cm);

 for i = 1:3 
    TP = tp_m(i);
    FP = sum(cm(:, i), 1) - TP;
    FN = sum(cm(i, :), 2) - TP;
    TN = sum(cm(:)) - TP - FP - FN;

    Accuracy = (TP+TN)./(TP+FP+TN+FN);

    TPR = TP./(TP + FN);%tp/actual positive  RECALL SENSITIVITY
    if isnan(TPR)
        TPR = 0;
    end
    PPV = TP./ (TP + FP); % tp / predicted positive PRECISION
    if isnan(PPV)
        PPV = 0;
    end
    TNR = TN./ (TN+FP); %tn/ actual negative  SPECIFICITY
    if isnan(TNR)
        TNR = 0;
    end
    FPR = FP./ (TN+FP);
    if isnan(FPR)
        FPR = 0;
    end
    FScore = (2*(PPV * TPR)) / (PPV+TPR);

    if isnan(FScore)
        FScore = 0;
    end
end
%% https://www.mathworks.com/help/deeplearning/ref/predict.html
YPred = predict(myNet,testImages)
YPred(1:2,:)

%%
save SkinDFUpaper3 myNet