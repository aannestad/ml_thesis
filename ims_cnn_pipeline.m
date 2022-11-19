pathToRecordingsFolder = fullfile('ims_wav');
location = pathToRecordingsFolder;
ads = audioDatastore(location);

%%

lab = cell2mat(ims_dataset(:,2));
ads.Labels = categorical(lab,[0,1],{'normal','alert'});
summary(ads.Labels)

%%

adsSample = subset(ads,[89,58,4152,4051]);
SampleRate = 20000;
for i = 1:4
    [audioSamples,info] = read(adsSample); 
    subplot(2,2,i)
    stft(audioSamples,SampleRate,'FrequencyRange','onesided');
    title('Label: '+string(info.Label))
end

%%

rng default;
ads = shuffle(ads);
[adsTrain,adsTest] = splitEachLabel(ads,0.8);

%%

disp(countEachLabel(adsTrain))

%%

disp(countEachLabel(adsTest))

%%

transTrain = transform(adsTrain,@(x,info)helperReadSPData(x,info),'IncludeInfo',true);
transTest = transform(adsTest,@(x,info)helperReadSPData(x,info),'IncludeInfo',true);

%%

sigLength = 20480;
dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer([sigLength 1])
    
    logSpectrogramLayer(sigLength,'Window',hamming(1280),'FFTLength',1280,...
        'OverlapLength',900)
    
    convolution2dLayer(5,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2)

    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numel(categories(ads.Labels)))
    softmaxLayer
    classificationLayer('Classes',categories(ads.Labels));
    ];


%%

% Set the hyperparameters to use in training the network. Use a mini-batch size 
% of |50| and a learning rate of |1e-4|. Specify Adam optimization. Set |UsePrefetch| 
% to |true| to enable asynchronous prefetch and queuing of data to optimize training 
% performance. Background dispatching of data and using a GPU to train the network 
% requires Parallel Computing Toolbox™.

UsePrefetch = true;
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',5, ...
    'MiniBatchSize',100, ...
    'Shuffle','every-epoch', ...
    'DispatchInBackground',UsePrefetch,...
    'Plots','training-progress',...
    "Verbose",false);

%% 
% Train the network.

[trainedNet,trainInfo] = trainNetwork(transTrain,layers,options);

%% 
% Use the trained network to predict the digit labels for the test set. Compute 
% the prediction accuracy.

[YPred,probs] = classify(trainedNet,transTest);
cnnAccuracy = sum(YPred==adsTest.Labels)/numel(YPred)*100

%% 
% Summarize the performance of the trained network on the test set with a confusion 
% chart. Display the precision and recall for each class by using column and row 
% summaries. The table at the bottom of the confusion chart shows the precision 
% values. The table to the right of the confusion chart shows the recall values.

figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
ccDCNN = confusionchart(adsTest.Labels,YPred);
ccDCNN.Title = 'Confusion Chart for DCNN';
ccDCNN.ColumnSummary = 'column-normalized';
ccDCNN.RowSummary = 'row-normalized';

%%

function [out,info] = helperReadSPData(x,info)

N = numel(x);
if N > 20480
    x = x(1:20480);
elseif N < 20480
    pad = 20480-N;
    prepad = floor(pad/2);
    postpad = ceil(pad/2);
    x = [zeros(prepad,1) ; x ; zeros(postpad,1)];
end
x = x./max(abs(x));
out = {x./max(abs(x)),info.Label};
end

