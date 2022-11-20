% Prepare IMS data

folders = {'ims/1','ims/2','ims/3'};

n_folders = length(folders);

% ------ Hardcode start/finish dates and ch failures for each folder ------

set1start = datetime(2003,10,22,12,06,24);            % test (set) 1
set1finish = datetime(2003,11,25,23,39,56);
set1ch_failure = [5,6,7,8];

set2start = datetime(2004,02,12,10,32,39);            % test (set) 2
set2finish = datetime(2004,02,19,06,22,39);
set2ch_failure = 1;

set3start = datetime(2004,03,04,09,27,46);            % test (set) 3
set3finish = datetime(2004,04,18,02,42,55);
set3ch_failure = 3;

set123start_finish = [set1start,set1finish;
                      set2start,set2finish;
                      set3start,set3finish];

set123ch_failure = {set1ch_failure;
                    set2ch_failure;
                    set3ch_failure};

% ------------------------------ Main loop --------------------------------
% one list: all folder build a list of filenames and labels
set123_labels = cell(4*2156+984+6324,2); % (set, label)   (1;2;3,0,1,2)

k = 1;

for fld = 1:n_folders

    h = waitbar(0, 'Starting');  % initiate waitbar display
    s = datetime("now");

    start_time = set123start_finish(fld,1);
    finish_time = set123start_finish(fld,2);

    lifetime = duration(finish_time - start_time);

    % alert = 10% of remaining lifetime
    alert_state = finish_time - lifetime*0.1;

    % alarm = 1% of remaining lifetime
    alarm_state = finish_time - lifetime*0.01;

    ch_failure = set123ch_failure{fld};

    folder_name = folders{fld};
    list_id = {dir(folder_name).name}'; 
    list_id = list_id(3:end,:);
 
    n_ids = length(list_id);

    tmp_labels = zeros(n_ids,1);

    for i = 1:n_ids
        data = table2array(readtable([folder_name,'/',list_id{i}], 'FileType','text'));
        m = size(data,2);

        current_datetime = datetime(list_id{i},'InputFormat','yyyy.MM.dd.HH.mm.ss');

        if current_datetime > alert_state              % below 10% -> alert
            %if current_datetime < alarm_state      
            tmp_label = 1;
            %else
            %    tmp_label = 2;                          % below 1% -> alarm
            %end
        else
            tmp_label = 0;                            % above 10% -> normal
        end

        for j = 1:m
            if ismember(j, ch_failure)
               
                    y = data(:,j); % rescale(data(:,j));
                    set123_labels{k,1} = ['test',num2str(fld),'_ch',num2str(j),'_',list_id{i},'.wav'];
                    set123_labels{k,2} = tmp_label;

                    k = k + 1;
            end
        end

        is = datetime("now")-datetime(s);
        esttime = is * (n_ids/i);
        h = waitbar(i/n_ids,h,[['Done: ',num2str(i),'/',num2str(n_ids)],'. Remaining time = ',char(esttime-(datetime("now")-datetime(s)))]);

    end
    delete(h)
end


%% Save balanced labels in wave format for audiostore
labels = cell2mat(set123_labels(:,2));
n_least_freq_class = sum(labels==1);

normal_data = set123_labels(labels==0,:);
alert_data = set123_labels(labels==1,:);

idx = randperm(length(normal_data),n_least_freq_class)';
normal_select = normal_data(idx,:);

ims_dataset = [normal_select;alert_data, ...                        % file name
               zeros(n_least_freq_class);ones(n_least_freq_class)];  % label

n = length(ims_dataset);

h = waitbar(0, 'Starting');  % initiate waitbar display
s = datetime("now");

for i = 1:n

    folder_name = ims_dataset{i,1}(5);
    file_name = ims_dataset{i,1}(11:end);
    channel = str2double(ims_dataset{i,1}(9));

    data = table2array(readtable(['ims/',folder_name,'/',file_name(1:end-4)], 'FileType','text'));
    y = data(:,channel);

    audiowrite(['ims_wav\',ims_dataset{i,1}],y,20000);

    is = datetime("now")-datetime(s);
    esttime = is * (n/i);
    h = waitbar(i/n,h,[['Done: ',num2str(i),'/',num2str(n)],'. Remaining time = ',char(esttime-(datetime("now")-datetime(s)))]);

end


%%
