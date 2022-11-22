pathToRecordingsFolder = fullfile('ims_wav_3class_aug');

location = pathToRecordingsFolder;
ads = audioDatastore(location);

load ims_dataset_aug.mat

labels = cell2mat(ims_dataset_aug(:,2));
ads.Labels = categorical(labels,[0,1,2],{'normal','alert','alarm'});
summary(ads.Labels)

% draw random 3 of each class

%idx = randperm(length(normal_data),n_least_freq_class)';
%normal_select = normal_data(idx,:);

normal_idx = 0+randperm(sum(labels==0),3);

alert_idx = sum(labels==0)+randperm(sum(labels==1),3);
alarm_idx = sum(labels==0)+sum(labels==1)+randperm(sum(labels==2),3);

adsSample = subset(ads,[normal_idx,alert_idx,alarm_idx]);
adsSample2 = subset(ads,[normal_idx,alert_idx,alarm_idx]);

SampleRate = 20000;
for i = 1:9
    [audioSamples,info] = read(adsSample); 
    subplot(3,3,i)
    stft(audioSamples,SampleRate,'FrequencyRange','onesided');
    title('Label: '+string(info.Label))
end
hold on
figure

for i = 1:9
    [audioSamples,info] = read(adsSample2); 
    features = vggishPreprocess(audioSamples,20000);
    subplot(3,3,i)
    surf(features,EdgeColor="none")
    view(90,-90)
    title('Label: '+string(info.Label))
end