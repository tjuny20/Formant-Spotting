% Gets the list of files
files = dir(['commands/one/*.wav'])
% Loops over each file
for k = 2:2
    % Gets the file name
    file = files(k).name;
    % Create hash to save
    hash = strrep(string(file),'.wav','.csv')
    % Reads the file
    [d,r] = audioread(fullfile('commands/one', file), 'double');
    h = 20;
    try
        % Fits the sinwaves to get the formant tracks
        [F,M] = swsmodel(d,r,h);
        % Save formant tracks
        %save(fullfile('waves/800hz/two', hash),'F','M','-ascii')
    catch
        disp(hash,'not processed')
    end
    plot(F'); % show all the frequencies
    dr = synthtrax(F,M,r);
    % Listen to it
    %sound(dr,r)
    % Compare to noise-excited reconstruction of LPC analysis
    [a,g] = lpcfit(d);
    dl = lpcsynth(a,g);
    %sound(dl,r);
    % The LPC reconstruction is based on more or less the same information 
    % as the sinewave replica, but it sounds less 'weird'
    % Compare the spectrograms
    subplot(211)
    specgram(d,256,r);
    title('Original');
    subplot(212)
    specgram(dr,256,r);
    title('Sine wave replica');
    %subplot(313)
    %specgram(dl,256,r);
    %title('Noise-excited LPC reconstruction');
end

