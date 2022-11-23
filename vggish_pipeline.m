% get pretrainet network

VGGishLocation = tempdir;
addpath(fullfile(VGGishLocation,'vggish'))

pathToRecordingsFolder = fullfile('ims_wav_3class_aug');
location = pathToRecordingsFolder;
ads = audioDatastore(location);

%%
load ims_dataset_aug.mat
lab = cell2mat(ims_dataset_aug(:,2));
ads.Labels = categorical(lab,[0,1,2],{'normal','alert','alarm'});
summary(ads.Labels) % labelTable = countEachLabel(ads)

%%
numClasses = height(labelTable);

%%
rng default;
ads = shuffle(ads);
[adsTrain, adsValidation] = splitEachLabel(ads,0.8);

countEachLabel(adsTrain)

%% 

countEachLabel(adsValidation)

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

%% Extract spectrograms from the validation set and replicate the labels.

validationFeatures = [];
validationLabels = [];
segmentsPerFile = zeros(numel(adsValidation.Files), 1);
idx = 1;
while hasdata(adsValidation)
    [audioIn,fileInfo] = read(adsValidation);
    features = vggishPreprocess(audioIn,fileInfo.SampleRate,OverlapPercentage=overlapPercentage);
    numSpectrograms = size(features,4);
    validationFeatures = cat(4,validationFeatures,features);
    validationLabels = cat(2,validationLabels,repelem(fileInfo.Label,numSpectrograms));

    segmentsPerFile(idx) = numSpectrograms;
    idx = idx + 1;
end


%%

net = vggish;

lgraph = layerGraph(net.Layers);

lgraph = removeLayers(lgraph,"regressionoutput");
lgraph.Layers(end) % inspect last

lgraph = addLayers(lgraph,[ ...
    fullyConnectedLayer(numClasses,Name="FCFinal",WeightLearnRateFactor=10,BiasLearnRateFactor=10)
    softmaxLayer(Name="softmax")
    classificationLayer(Name="classOut")]);

lgraph = connectLayers(lgraph,"EmbeddingBatch","FCFinal");

%%

miniBatchSize = 128;
options = trainingOptions("adam", ...
    MaxEpochs=5, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    ValidationData={validationFeatures,validationLabels}, ...
    ValidationFrequency=50, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.5, ...
    LearnRateDropPeriod=2, ...
    OutputNetwork="best-validation-loss", ...
    Verbose=false, ...
    Plots="training-progress");

%%

[trainedNet, netInfo] = trainNetwork(trainFeatures,trainLabels,lgraph,options);

%%

validationPredictions = classify(trainedNet,validationFeatures);

idx = 1;
validationPredictionsPerFile = categorical;
for ii = 1:numel(adsValidation.Files)
    validationPredictionsPerFile(ii,1) = mode(validationPredictions(idx:idx+segmentsPerFile(ii)-1));
    idx = idx + segmentsPerFile(ii);
end

%%

figure(Units="normalized",Position=[0.2 0.2 0.5 0.5]);
confusionchart(adsValidation.Labels,validationPredictionsPerFile, ...
    Title=sprintf("Confusion Matrix for Validation Data \nAccuracy = %0.2f %%",mean(validationPredictionsPerFile==adsValidation.Labels)*100), ...
    ColumnSummary="column-normalized", ...
    RowSummary="row-normalized")
%%

