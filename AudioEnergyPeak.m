clear;
clc;

%% window segmentation using buffer and energy
% https://kr.mathworks.com/help/signal/ref/buffer.html
% https://kr.mathworks.com/matlabcentral/answers/885654-find-energy-for-each-second-of-audio-file
% 를 참조하였음.

%% 전체 pass에 대한 sound 읽기
% path = 'dataset/Fighting/attack/Artlist/Vadi Sound - Wrestling Ring - Slapping Punch.wav';
path = './dataset/Fighting';
labelArray = dir(path);
for labelIndex=3:length(labelArray)
    label = labelArray(labelIndex).name;
    fprintf("%s \n", label);
    
    labelPath = strcat(path, '/', label);
    SourceArray = dir(labelPath);
    for sourceIndex = 3:length(SourceArray)
        source = SourceArray(sourceIndex).name;
        soundPath = strcat(labelPath, '/', source);
        soundArray = dir(soundPath);
        for soundIndex = 3:length(soundArray)
            fileName = soundArray(soundIndex).name;
            soundPath = strcat(soundPath, '/', fileName);
            [sound, fs] = audioread(soundPath);

            fprintf('label: %s, source: %s, sound: %s \n', label, source, soundPath);

            dataSize = 44100;
            startPoint = 1;
            endPoint = startPoint + dataSize - 1;       
            windowSize = 20 * dataSize / 1000;
            hopSize = 10 * dataSize / 1000;

            sound = resample(sound, fs, dataSize);
                
            segment = buffer(mean(sound(startPoint:endPoint, :), 2), windowSize, hopSize);
            energy = sum(segment.^2);
            energyPeak = find(energy == max(energy));
            peakIndex = energyPeak(1);
            index = floor(peakIndex / length(energy) * dataSize);
            startPointCut = startPoint + index;
            targetSound = mean(sound(startPointCut-windowSize:startPointCut+windowSize-1, :), 2);
            startPoint = startPointCut + 1;
            endPoint = startPoint + dataSize - 1;
        end
    end 
end

