%% 
% calibrate for MX TGB

pathToRecordingsFolder = fullfile('MX_TGB_finetune');

location = pathToRecordingsFolder;
ads = audioDatastore(location);

%%
load mx_tgb_finetune.mat
lab = cell2mat(mxtgb_finetune(:,2));

ads.Labels = categorical(lab,[0,1,2],{'normal','alert','alarm'});
disp('mx_tgb_finetune dataset count:')
summary(ads.Labels) % labelTable = countEachLabel(ads)

%%
rng default;
ads = shuffle(ads);
[adsTrainValidation, adsTest] = splitEachLabel(ads,0.9);
[adsTrain, adsValidation] = splitEachLabel(adsTrainValidation,0.8);

%transTrain = transform(adsTrain,@(x,info)helperReadSPData(x,info),'IncludeInfo',true);
%transVal = transform(adsValidation,@(x,info)helperReadSPData(x,info),'IncludeInfo',true);
%transTest = transform(adsTest,@(x,info)helperReadSPData(x,info),'IncludeInfo',true);
disp('Training count:')
countEachLabel(adsTrain)

%% 
disp('Validation count:')
countEachLabel(adsValidation)

disp('Test count:')
countEachLabel(adsTest)

%%

overlapPercentage = 75;

trainFeatures = [];
trainLabels = [];
while hasdata(adsTrain)
    [audioIn,fileInfo] = read(adsTrain);
    features = vggishPreprocess(audioIn,fileInfo.SampleRate,OverlapPercentage=overlapPercentage);
    numSpectrograms = size(features,4);
    trainFeatures = cat(4,trainFeatures,features);
    trainLabels = cat(2,trainLabels,repelem(fileInfo.Label,numSpectrograms));
end

%% VAL Extract spectrograms from the validation set and replicate the labels.

validationFeatures = [];
validationLabels = [];
val_segmentsPerFile = zeros(numel(adsValidation.Files), 1);
idx = 1;
while hasdata(adsValidation)
    [audioIn,fileInfo] = read(adsValidation);
    features = vggishPreprocess(audioIn,fileInfo.SampleRate,OverlapPercentage=overlapPercentage);
    numSpectrograms = size(features,4);
    validationFeatures = cat(4,validationFeatures,features);
    validationLabels = cat(2,validationLabels,repelem(fileInfo.Label,numSpectrograms));

    val_segmentsPerFile(idx) = numSpectrograms;
    idx = idx + 1;
end

%% TEST Extract spectrograms from the TEST set and replicate the labels.

testFeatures = [];
testLabels = [];
test_segmentsPerFile = zeros(numel(adsTest.Files), 1);
idx = 1;
while hasdata(adsTest)
    [audioIn,fileInfo] = read(adsTest);
    features = vggishPreprocess(audioIn,fileInfo.SampleRate,OverlapPercentage=overlapPercentage);
    numSpectrograms = size(features,4);
    testFeatures = cat(4,testFeatures,features);
    testLabels = cat(2,testLabels,repelem(fileInfo.Label,numSpectrograms));

    test_segmentsPerFile(idx) = numSpectrograms;
    idx = idx + 1;
end


%%
load trainedNet
numClasses = 3; % MX and TGB trained on its two classes %height(labelTable);
net = trainedNet;

lgraph = layerGraph(net.Layers);
lgraph = removeLayers(lgraph,"FCFinal");
lgraph = removeLayers(lgraph,"softmax");
lgraph = removeLayers(lgraph,"classOut");
lgraph.Layers(end) % inspect last

lgraph = addLayers(lgraph,[ ...
    fullyConnectedLayer(numClasses,Name="FC_FineTune",WeightLearnRateFactor=10,BiasLearnRateFactor=10)
    softmaxLayer(Name="softmax")
    classificationLayer(Name="classOut")]);

lgraph = connectLayers(lgraph,"EmbeddingBatch","FC_FineTune");

%%

miniBatchSize = 16;
options = trainingOptions("adam", ...
    MaxEpochs=10, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData={validationFeatures,validationLabels}, ...
    ValidationFrequency=32, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.5, ...
    LearnRateDropPeriod=2, ...
    OutputNetwork="best-validation-loss", ...
    Verbose=false, ...
    Plots="training-progress");

%% TRAINING

[finetunedNet, finetunednetInfo] = trainNetwork(trainFeatures,trainLabels,lgraph,options);

%% VALIDATION

validationPredictions = classify(finetunedNet,validationFeatures);

idx = 1;
validationPredictionsPerFile = categorical;
for ii = 1:numel(adsValidation.Files)
    validationPredictionsPerFile(ii,1) = mode(validationPredictions(idx:idx+val_segmentsPerFile(ii)-1));
    idx = idx + val_segmentsPerFile(ii);
end

%%

validationPredictions = classify(finetunedNet,validationFeatures);

cnnValAccuracy = sum(validationPredictions==adsValidation.Labels)/numel(validationPredictions)*100

figure(Units="normalized",Position=[0.2 0.2 0.5 0.5]);
confusionchart(adsValidation.Labels,validationPredictionsPerFile, ...
    Title=sprintf("Confusion Matrix for Validation Data \nAccuracy = %0.2f %%",mean(validationPredictionsPerFile==adsValidation.Labels)*100), ...
    ColumnSummary="column-normalized", ...
    RowSummary="row-normalized")

%% 

%%  TEST

testPredictions = classify(finetunedNet,testFeatures);

idx = 1;
testPredictionsPerFile = categorical;
for ii = 1:numel(adsTest.Files)
    testPredictionsPerFile(ii,1) = mode(testPredictions(idx:idx+test_segmentsPerFile(ii)-1));
    idx = idx + test_segmentsPerFile(ii);
end

%%

testPredictions = classify(finetunedNet,testFeatures);

cnnTestAccuracy = sum(testPredictions==adsTest.Labels)/numel(testPredictions)*100

figure(Units="normalized",Position=[0.2 0.2 0.5 0.5]);
confusionchart(adsTest.Labels,testPredictionsPerFile, ...
    Title=sprintf("Confusion Matrix for test Data \nAccuracy = %0.2f %%",mean(testPredictionsPerFile==adsTest.Labels)*100), ...
    ColumnSummary="column-normalized", ...
    RowSummary="row-normalized")
