clear;

%% Parameter Setting
maxAugmentCount = 9;
fs = 44100;
sound_path = 'data/Fighting/matlab/original_sound.mat';
sound_dict = load(sound_path);
aug = audioDataAugmenter("NumAugmentations", maxAugmentCount,...
                        "TimeStretchProbability", 0, ...
                        "PitchShiftProbability", 0, "SemitoneShiftRange", [-2, 2], ...
                        "VolumeControlProbability", 0.8, "VolumeGainRange", [-5, 5], ...
                        "AddNoiseProbability", 0.8, "SNRRange", [15, 25], ...
                        "TimeShiftProbability", 0);

%% x_train augmentation
fprintf("x_train augmentation \n");
x_train = sound_dict.x_train;
x_train_length = size(x_train, 1);
n_features = size(x_train, 2);
aug_x_train = zeros(x_train_length*10, n_features);
aug_y_train = zeros(x_train_length*10, 1);
for soundIndex = 1:x_train_length
    if rem(soundIndex, 1000) == 0
        fprintf("\tsoundIndex: %d \n", soundIndex);
    end
    sound = x_train(soundIndex, :);
    newData = augment(aug, sound, fs);
    for augIndex = 1:maxAugmentCount + 1
        if augIndex == maxAugmentCount + 1   
            augmentedData = sound;
        else
            augmentedData = newData.Audio(augIndex, :);
        end
        aug_x_train((soundIndex-1)*10 + augIndex, :) = augmentedData;
        aug_y_train((soundIndex-1)*10 + augIndex) = sound_dict.y_train(soundIndex);
    end
end

%% Save
fprintf("make segmentation struct\n");
% aug_dict = {aug_x_train, aug_y_train, aug_x_val, aug_y_val, aug_x_test, aug_y_test};
aug_dict = {aug_x_train, aug_y_train};
save('data/Fighting/matlab/augmented_x_train.mat', 'aug_dict');
fprintf("done \n");