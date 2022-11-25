% load: one pipeline for MX, one for TGB
%% -------------------- Build MX labeled dataset --------------------------
% load each folder: 2000 RPM, channel 2
% hardcode label pr folder

folders = {'nonfault/2000RPM','fault1/2000RPM','fault2/2000RPM','fault3/2000RPM'};
n_folders = length(folders);

labels = [0,1,1,1];

fs_out = 20000;
fs_out_length = fs_out + 480;

% save mx_dataset and tgb_dataset file with filenames and label

mx_labels = cell(250*3*4,2);

k = 1;

for fld = 1:n_folders

    h = waitbar(0, 'Starting');  % initiate waitbar display
    s = datetime("now");

    folder_name = folders{fld};
    list_id = {dir(folder_name).name}'; 
    list_id = list_id(3:end,:);
 
    n_ids = length(list_id);

    tmp_label = labels(fld);

    if strcmp(folder_name(1:8),'nonfault')  % for folder non-fault, not for blind non-fault
        Accel_Comp = 5;
    else
        Accel_Comp = 10;
    end

    for i = 1:n_ids

        [data,fs_in] = audioread([folder_name,'/',list_id{i}]);
        acc= data(:,2);
        y = acc * Accel_Comp;
     
        [P,Q] = rat(fs_out/fs_in);  % Konverter forholdet til brøk (% abs(P/Q*Fs-Fs_out) % sjekk avrundingsdifferanse)

        y = resample(y,P,Q);   % downsample  IDEA: test spectrograms for ORIGINAL: better results?

        y_m = length(y);

        n_samples = floor(y_m/fs_out_length);
        m = n_samples * fs_out_length;

        l = 1;
        
        for j = 1:fs_out_length:m                  

            y_j = y(j:j+fs_out_length-1);                  % split into 1 sec. samples

            mx_labels{k,1} = [num2str(fld-1),'_',list_id{i}(1:end-4),'_',num2str(l),'.wav'];
            mx_labels{k,2} = tmp_label;

            audiowrite(['MX\',mx_labels{k,1}],y_j,fs_out);

            k = k + 1;
            l = l + 1;
            
        end

       is = datetime("now")-datetime(s);
       esttime = is * (n_ids/i);
       h = waitbar(i/n_ids,h,[['Done: ',num2str(i),'/',num2str(n_ids)],'. Remaining time = ',char(esttime-(datetime("now")-datetime(s)))]);

    end
    delete(h)
end

mx_labels = mx_labels(1:k-1,:);

save mx_labels mx_labels


%% -------------------- Build TBG labeled dataset -------------------------
% load each folder: 2000 RPM, channel 2
% hardcode label pr folder

folders = {'nonfault_TGB_220901/2000RPM','F1_F9_220905/2000RPM','F8_220902/2000RPM'};
n_folders = length(folders);

labels = [0,2,2];

fs_out = 20000;
fs_out_length = fs_out + 480;

% save tgb_finetune

tgb_labels = cell(250*3*3,2);

k = 1;

for fld = 1:n_folders

    h = waitbar(0, 'Starting');  % initiate waitbar display
    s = datetime("now");

    folder_name = folders{fld};
    list_id = {dir(folder_name).name}'; 
    list_id = list_id(3:end,:);
 
    n_ids = length(list_id);

    tmp_label = labels(fld);

    if strcmp(folder_name(1:8),'nonfault')
        Accel_Comp = 10;
    else
        Accel_Comp = 25;
    end

    for i = 1:n_ids

        [data,fs_in] = audioread([folder_name,'/',list_id{i}]);
        acc= data(:,2);
        y = acc * Accel_Comp;
     
        [P,Q] = rat(fs_out/fs_in);  % Konverter forholdet til brøk (% abs(P/Q*Fs-Fs_out) % sjekk avrundingsdifferanse)

        y = resample(y,P,Q);   % downsample  

        y_m = length(y);

        n_samples = floor(y_m/fs_out_length);
        m = n_samples * fs_out_length;

        l = 1;
        
        for j = 1:fs_out_length:m                  

            y_j = y(j:j+fs_out_length-1);                  % split into 1 sec. samples

            tgb_labels{k,1} = [num2str(fld-1),'_',list_id{i}(1:end-4),'_',num2str(l),'.wav'];
            tgb_labels{k,2} = tmp_label;

            audiowrite(['TGB\',tgb_labels{k,1}],y_j,fs_out);

            k = k + 1;
            l = l + 1;
            
        end

       is = datetime("now")-datetime(s);
       esttime = is * (n_ids/i);
       h = waitbar(i/n_ids,h,[['Done: ',num2str(i),'/',num2str(n_ids)],'. Remaining time = ',char(esttime-(datetime("now")-datetime(s)))]);

    end
    delete(h)
end

tgb_labels = tgb_labels(1:k-1,:);

save tgb_labels tgb_labels

%% MX_finetune: Save balanced 3 CLASS (normal, alert) labels in wave format for audiostore
clear
load mx_labels

labels = cell2mat(mx_labels(:,2));
n_least_freq_class = sum(labels==0);  % use label=0=normal as fewest

normal_data = mx_labels(labels==0,:);
alert_data = mx_labels(labels==1,:);
%alarm_data = mx_labels(labels==2,:);  % no alarms in MX faults

idx = randperm(length(alert_data),n_least_freq_class)';
alert_data = alert_data(idx,:);

mx_finetune = [normal_data; alert_data]; %...                        % file name
   
n = length(mx_finetune);

h = waitbar(0, 'Starting');  % initiate waitbar display
s = datetime("now");

for i = 1:n

    file_name = mx_finetune{i,1};

    [y,fs] = audioread(['MX/',mx_finetune{i}]);
 
    audiowrite(['MX_finetune\',mx_finetune{i,1}],y,fs); % idea: all label to filename?

    is = datetime("now")-datetime(s);
    esttime = is * (n/i);
    h = waitbar(i/n,h,[['Done: ',num2str(i),'/',num2str(n)],'. Remaining time = ',char(esttime-(datetime("now")-datetime(s)))]);

end
delete(h)
save mx_finetune mx_finetune

%% TGB_finetune: Save balanced CLASS (normal, alarm) labels in wave format for audiostore
clear
load tgb_labels

labels = cell2mat(tgb_labels(:,2));
n_least_freq_class = sum(labels==0);  % use label=0=normal as fewest

normal_data = tgb_labels(labels==0,:);
%alert_data = tgb_labels(labels==1,:);   % no alerts in TGB faults
alarm_data = tgb_labels(labels==2,:);

idx = randperm(length(alarm_data),n_least_freq_class)';
alarm_select = alarm_data(idx,:);

tgb_finetune = [normal_data; alarm_select]; %...                        % file name
   
n = length(tgb_finetune);

h = waitbar(0, 'Starting');  % initiate waitbar display
s = datetime("now");

for i = 1:n

    file_name = tgb_finetune{i,1};
    channel = str2double(tgb_finetune{i,1}(9));

    [y,fs] = audioread(['TGB/',tgb_finetune{i,1}]);
 
    audiowrite(['TGB_finetune\',tgb_finetune{i,1}],y,fs); % idea: all label to filename?

    is = datetime("now")-datetime(s);
    esttime = is * (n/i);
    h = waitbar(i/n,h,[['Done: ',num2str(i),'/',num2str(n)],'. Remaining time = ',char(esttime-(datetime("now")-datetime(s)))]);

end
delete(h)

save tgb_finetune tgb_finetune