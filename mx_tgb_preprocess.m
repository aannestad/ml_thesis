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

        y = resample(y,P,Q);   % downsample  

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

save mx_labels