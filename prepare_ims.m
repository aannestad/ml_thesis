% Convert IMS to single channel wav

folders = {'ims/1','ims/2','ims/3'};
n_folders = length(folders);

for fld = 1:n_folders
    
    folder_name = folders{fld};
    list_id = {dir(folder_name).name}'; 
    list_id = list_id(3:end,:);

    n_ids = length(list_id);

    h = waitbar(0, 'Starting');  % initiate waitbar display
    s = datetime("now");

    for i = 1:n_ids
        data = table2array(readtable([folder_name,'/',list_id{i}], 'FileType','text'));
        m = size(data,2);
        for j = 1:m
           y = data(:,j); % rescale(data(:,j));
           var_name = ['ch',num2str(j),'_',list_id{i},'.wav'];  
           audiowrite([folder_name,'wav\',var_name],y,20000);
        end
                                                                          % Update waitbar display
        is = datetime("now")-datetime(s);
        esttime = is * (n_ids/i);
        h = waitbar(i/n_ids,h,[['Done: ',num2str(i),'/',num2str(n_ids)],'. Remaining time = ',char(esttime-(datetime("now")-datetime(s)))]);

    end

end