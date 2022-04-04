% Gets the list of files
files = dir(['data/commands/five/*.wav']);
n_files = 1000
count = 0;
k = 1;
% Loops over each file
while count<n_files
    % Gets the file name
    file = files(k).name;
    % Create hash to save
    hash = strrep(string(file),'.wav','.csv')
    % Reads the file
    [d,r] = audioread(fullfile('commands/five', file), 'double');
    h = 20;
    try
        % Fits the sinwaves to get the formant tracks
        [F,M] = swsmodel(d,r,h);
        % Save formant tracks
        A = [F;M];
        csvwrite(fullfile('waves/800hz/five', hash), A)
        %save(fullfile('waves/four', hash),'F','M','-ascii')
        count = count + 1;
    catch
        disp(hash,' not processed')
    end
    k = k + 1;
end


% Gets the list of files
files = dir(['commands/six/*.wav']);
n_files = 1000
count = 0;
k = 1;
% Loops over each file
while count<n_files
    % Gets the file name
    file = files(k).name;
    % Create hash to save
    hash = strrep(string(file),'.wav','.csv')
    % Reads the file
    [d,r] = audioread(fullfile('commands/six', file), 'double');
    h = 20;
    try
        % Fits the sinwaves to get the formant tracks
        [F,M] = swsmodel(d,r,h);
        % Save formant tracks
        A = [F;M];
        csvwrite(fullfile('waves/800hz/six', hash), A)
        %save(fullfile('waves/four', hash),'F','M','-ascii')
        count = count + 1;
    catch
        disp(hash,' not processed')
    end
    k = k + 1;
end

% Gets the list of files
files = dir(['commands/seven/*.wav']);
n_files = 1000
count = 0;
k = 1;
% Loops over each file
while count<n_files
    % Gets the file name
    file = files(k).name;
    % Create hash to save
    hash = strrep(string(file),'.wav','.csv')
    % Reads the file
    [d,r] = audioread(fullfile('commands/seven', file), 'double');
    h = 20;
    try
        % Fits the sinwaves to get the formant tracks
        [F,M] = swsmodel(d,r,h);
        % Save formant tracks
        A = [F;M];
        csvwrite(fullfile('waves/800hz/seven', hash), A)
        %save(fullfile('waves/four', hash),'F','M','-ascii')
        count = count + 1;
    catch
        disp(hash,' not processed')
    end
    k = k + 1;
end

% Gets the list of files
files = dir(['commands/eight/*.wav']);
n_files = 1000
count = 0;
k = 1;
% Loops over each file
while count<n_files
    % Gets the file name
    file = files(k).name;
    % Create hash to save
    hash = strrep(string(file),'.wav','.csv')
    % Reads the file
    [d,r] = audioread(fullfile('commands/eight', file), 'double');
    h = 20;
    try
        % Fits the sinwaves to get the formant tracks
        [F,M] = swsmodel(d,r,h);
        % Save formant tracks
        A = [F;M];
        csvwrite(fullfile('waves/800hz/eight', hash), A)
        %save(fullfile('waves/four', hash),'F','M','-ascii')
        count = count + 1;
    catch
        disp(hash,' not processed')
    end
    k = k + 1;
end

% Gets the list of files
files = dir(['commands/nine/*.wav']);
n_files = 1000
count = 0;
k = 1;
% Loops over each file
while count<n_files
    % Gets the file name
    file = files(k).name;
    % Create hash to save
    hash = strrep(string(file),'.wav','.csv')
    % Reads the file
    [d,r] = audioread(fullfile('commands/nine', file), 'double');
    h = 20;
    try
        % Fits the sinwaves to get the formant tracks
        [F,M] = swsmodel(d,r,h);
        % Save formant tracks
        A = [F;M];
        csvwrite(fullfile('waves/800hz/nine', hash), A)
        %save(fullfile('waves/four', hash),'F','M','-ascii')
        count = count + 1;
    catch
        disp(hash,' not processed')
    end
    k = k + 1;
end


% Gets the list of files
files = dir(['commands/zero/*.wav']);
n_files = 1000
count = 0;
k = 1;
% Loops over each file
while count<n_files
    % Gets the file name
    file = files(k).name;
    % Create hash to save
    hash = strrep(string(file),'.wav','.csv')
    % Reads the file
    [d,r] = audioread(fullfile('commands/zero', file), 'double');
    h = 20;
    try
        % Fits the sinwaves to get the formant tracks
        [F,M] = swsmodel(d,r,h);
        % Save formant tracks
        A = [F;M];
        csvwrite(fullfile('waves/800hz/zero', hash), A)
        %save(fullfile('waves/four', hash),'F','M','-ascii')
        count = count + 1;
    catch
        disp(hash,' not processed')
    end
    k = k + 1;
end