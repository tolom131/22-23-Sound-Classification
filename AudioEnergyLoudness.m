soundPath = 'dataset\Fighting\attack\Artlist\Alberto Sueri - RPG Game - Energetic Punch .wav';
samplingRate = 44100;
type = 1;

[sound, fs] = audioread(soundPath);
samplingRate = double(samplingRate);
fprintf("%f \n", samplingRate);
resampledSound = resample(sound, samplingRate, fs); 
windowSize = 20*samplingRate/1000; % 20ms
hopSize = 10*samplingRate/1000; % 10ms

if (type==1)
    [peakArray, originalPeakArray, energyArray, loundnessArray] = FindOnePeak(resampledSound, samplingRate, windowSize, hopSize);
else
    [peakArray, originalPeakArray, energyArray, loundnessArray] = FindManyPeak(resampledSound, samplingRate, windowSize, hopSize);
end

dict = {peakArray, originalPeakArray, energyArray, loundnessArray};

function [peakArray, originalPeakArray, energyArray, loundnessArray] = FindOnePeak(sound, fs, windowSize, hopSize)
    peakArray = zeros(1, 2); % 1: peak index, 2: energy/loudness
    originalPeakArray = zeros(1, 2);
    dataSize = length(sound);
    startPoint = 1; % from the start 
    endPoint = startPoint + dataSize - 1; % to the end

    %% Energy Calculation
    segment = buffer(mean(sound(startPoint:endPoint, :), 2), windowSize, hopSize);
    energy = sum(segment.^2);
    energyPeaks = find(energy == max(energy));
    energyPeakIndex = energyPeaks(1);
    peakArray(1, 1) = floor(energyPeakIndex / length(energy) * dataSize);
    originalPeakArray(1, 1) = energyPeakIndex;

    %% Loudness Calculation
    [loudness, ~] = acousticLoudness(mean(sound(startPoint:endPoint, :), 2), fs, 'TimeVarying', true, 'TimeResolution', 'high');
    loudnessPeaks = find(loudness == max(loudness));
    loudnessPeakIndex = loudnessPeaks(1);
    peakArray(1, 2) = floor(loudnessPeakIndex / length(loudness) * dataSize);
    originalPeakArray(1, 2) = loudnessPeakIndex;

    energyArray = energy;
    loundnessArray = transpose(loudness);
end

function [peakArray, originalPeakArray, energyArray, loundnessArray] = FindManyPeak(sound, fs, windowSize, hopSize)
    peakArray = zeros(1000, 2); % 1000은 임의의 숫자
    originalPeakArray = zeros(100, 2);
    energyArray = zeros(1000, 100);
    loundnessArray = zeros(1000, 2000);

    dataSize = fs;
    startPoint = 1;
    endPoint = startPoint + dataSize - 1;

    count = 0;
    while(endPoint < size(sound,1))
        count = count+1;
        
        %% Energy Calculation
        segment = buffer(mean(sound(startPoint:endPoint, :), 2), windowSize, hopSize);
        energy = sum(segment.^2);
        energyPeaks = find(energy == max(energy));
        energyPeakIndex = energyPeaks(1);
        peakArray(count, 1) = floor(energyPeakIndex / length(energy) * dataSize);
        originalPeakArray(count, 1) = energyPeakIndex;

        %% Loudness Calculation
        [loudness, ~] = acousticLoudness(mean(sound(startPoint:endPoint, :), 2), fs, 'TimeVarying', true, 'TimeResolution', 'high');
        loudnessPeaks = find(loudness == max(loudness));
        loudnessPeakIndex = loudnessPeaks(1);
        peakArray(count, 2) = floor(loudnessPeakIndex / length(loudness) * dataSize);
        originalPeakArray(count, 2) = loudnessPeakIndex;

        energyArray(count, :) = transpose(energy);
        loundnessArray(count, :) = loudness;

        startPoint = endPoint + 1;
        endPoint = startPoint + dataSize - 1;

        % fprintf("loudness size %d, %d \n", size(loudness, 1), size(loudness, 2));
    end

    peakArray = peakArray(1:count, :);
    originalPeakArray = originalPeakArray(1:count, :);
    energyArray = energyArray(1:count, :);
    loundnessArray = loundnessArray(1:count, :);
end
