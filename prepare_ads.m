file_name_1 = "ch1_2003.10.22.12.06.24.wav";
file_name_2 = "ch1_2003.11.25.23.39.56.wav";

start_time = datetime(2003,10,22,12,06,24);
finish_time = datetime(2003,11,25,23,39,56);

%%


pathToRecordingsFolder = fullfile('1wav');
location = pathToRecordingsFolder;
ads = audioDatastore(location);


ads.Labels = helpergenLabels(ads,start_time, finish_time);

summary(ads.Labels)

function Labels = helpergenLabels(ads, start_time, finish_time)
% This function is only for use in the "Spoken Digit Recognition with
% Custom Log Spectrogram Layer and Deep Learning" example. It may change or
% be removed in a future release.

lifetime = duration(finish_time - start_time);

% alert = 10% of remaining lifetime
alert_state = finish_time - lifetime*0.1;

% alarm = 1% of remaining lifetime
alarm_state = finish_time - lifetime*0.01;


n = numel(ads.Files);
tmp_labels = zeros(n,1);

h = waitbar(0, 'Starting');  % initiate waitbar display
s = datetime("now");

for i = 1:n

    current_file = ads.Files{i};
    current_datetime = current_file(end-22:end-4);

    current_datetime = datetime(current_datetime,'InputFormat','yyyy.MM.dd.HH.mm.ss');

    if current_datetime > alert_state        % below 10%
        if current_datetime < alarm_state       
            tmp_labels(i) = 1;
        else
            tmp_labels(i) = 2;               % below 1%
        end
    end
                                                                              % Update waitbar display
    is = datetime("now")-datetime(s);
    esttime = is * (n/i);
    h = waitbar(i/n,h,[['Done: ',num2str(i),'/',num2str(n)],'. Remaining time = ',char(esttime-(datetime("now")-datetime(s)))]);
end

Labels = categorical(tmp_labels);

end

function [out,info] = helperReadSPData(x,info)
% This function is only for use in the "Spoken Digit Recognition with
% Custom Log Spectrogram Layer and Deep Learning" example. It may change or
% be removed in a future release.

N = numel(x);
if N > 8192
    x = x(1:8192);
elseif N < 8192
    pad = 8192-N;
    prepad = floor(pad/2);
    postpad = ceil(pad/2);
    x = [zeros(prepad,1) ; x ; zeros(postpad,1)];
end
x = x./max(abs(x));
out = {x./max(abs(x)),info.Label};
end

%%





