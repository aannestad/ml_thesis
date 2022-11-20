%% Augment class S2: alarm

load('set123_labels.mat')

augmenter = audioDataAugmenter( ...
    "AugmentationMode","independent", ...
    "AugmentationParameterSource","specify", ...
    "ApplyTimeShift",true, ...
    "ApplyPitchShift",true, ...
    "ApplyTimeStretch",false,...
    "PitchShiftProbability",1,...
    "VolumeGain",0, ...
    "SemitoneShiftRange",[-0.5 0.5], ...
    "AddNoiseProbability",1, ...
    "SemitoneShift",0.5,...
    "SNR",0.01, ...
    "TimeShiftProbability",1,...
    "TimeShift",0.001);

labels = cell2mat(set123_labels(:,2));
n_least_freq_class = sum(labels==1);
n_aug = n_least_freq_class - sum(labels==2);

alarm_idx = find(labels==2);

alarm_data = set123_labels(labels==2,:);
n_alarms = length(alarm_data);

for i = 1:n_aug
    
    idx = randi(n_alarms);                            % random select alarm
    
    %[y,fs] = audioread(['ims_alarm\',alarm_data{idx,1}]);
    [y,fs] = audioread(['./ims_alarm/',alarm_data{idx,1}]);

    data = augment(augmenter,y,fs);
    y_aug = data.Audio{1};

    audiowrite(['./ims_alarm_aug/',alarm_data{idx,1}(1:end-4),'_aug',num2str(i),'.wav'],y_aug,20000);

end

%%
figure
melSpectrogram(y,fs)

figure
melSpectrogram(y_aug,fs)