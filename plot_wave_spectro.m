pathToRecordingsFolder = fullfile('ims_wav_3class');

location = pathToRecordingsFolder;
ads = audioDatastore(location);

load ims_dataset_3class.mat

labels = cell2mat(ims_dataset_3class(:,2));
ads.Labels = categorical(labels,[0,1,2],{'normal','alert','alarm'});
summary(ads.Labels)

% draw random 1 of each class

n_samples = 3;

%idx = randperm(length(normal_data),n_least_freq_class)';
%normal_select = normal_data(idx,:);

normal_idx = 0+randperm(sum(labels==0),1);

alert_idx = sum(labels==0)+randperm(sum(labels==1),1);
alarm_idx = sum(labels==0)+sum(labels==1)+randperm(sum(labels==2),1);

adsSample = subset(ads,[normal_idx,alert_idx,alarm_idx]);

SampleRate = 20000;
plot_data = cell(3,2);
for i = 1:n_samples
    [audioSample,info] = read(adsSample); 
    plot_data{i,1} = audioSample;
    plot_data{i,2} = info;
end

k = 1;
for i = 1:6
    %i_mod = mod(i-1,n_samples)+1;

    audioSamples = plot_data{k,1};
    info = plot_data{k,2};

    if mod(i,2) == 1

        subplot(3,2,i)
        plot(audioSamples)
        title('Condition: '+string(info.Label))
    else
        subplot(3,2,i)
        stft(audioSamples,SampleRate,'FrequencyRange','onesided');
        title('Condition: '+string(info.Label))

        k = k + 1;
    end

end
hold on
% figure
% 
% for i = 1:4
%     [audioSamples,info] = read(adsSample2); 
%     features = vggishPreprocess(audioSamples,20000);
%     subplot(3,2,i)
%     surf(features,EdgeColor="none")
%     axis tight
%     view(90,-90)
%     title('Condition: '+string(info.Label))
% end


% imf_cond = emd(audioSamples,MaxNumIMF=5,Display=1);
% hht(imf_cond(:,1),20000,'FrequencyLimits',[0,10000]);
%% 

pathToRecordingsFolder = fullfile('ims_wav_3class');

location = pathToRecordingsFolder;
ads = audioDatastore(location);

load ims_dataset_3class.mat

labels = cell2mat(ims_dataset_3class(:,2));
ads.Labels = categorical(labels,[0,1,2],{'normal','alert','alarm'});
summary(ads.Labels)

% draw random 1 of each class

n_samples = 3;

%idx = randperm(length(normal_data),n_least_freq_class)';
%normal_select = normal_data(idx,:);

normal_idx = 51; % 0+randperm(sum(labels==0),1);

alert_idx = 3357; % sum(labels==0)+randperm(sum(labels==1),1);

alarm_idx = 4051; %5519; sum(labels==0)+sum(labels==1)+randperm(sum(labels==2),1);

adsSample = subset(ads,[normal_idx,alert_idx,alarm_idx]);

SampleRate = 20000;
plot_data = cell(3,2);
for i = 1:n_samples
    [audioSample,info] = read(adsSample); 
    plot_data{i,1} = audioSample;
    plot_data{i,2} = info;
end

k = 1;
for i = 1:6
    %i_mod = mod(i-1,n_samples)+1;

    audioSamples = plot_data{k,1};
    info = plot_data{k,2};

    if mod(i,2) == 1

        subplot(3,2,i)
        plot(audioSamples)
        title('Condition: '+string(info.Label))
    else
        subplot(3,2,i)
        stft(audioSamples,SampleRate,'FrequencyRange','onesided');
        title('Condition: '+string(info.Label))

        k = k + 1;
    end

end
hold on



