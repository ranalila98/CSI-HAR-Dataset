

clc;
clear;
close all;

% Define folders
data_folder = 'data';
train_folder = 'Train';
test_folder = 'Test';

% List of activities
activities = ["lie down", "fall", "bend", "run", "sitdown", "standup", "walk"];

% Create Train and Test folders
for i = 1:length(activities)
    activity = activities(i);
    mkdir(fullfile(train_folder, activity));
    mkdir(fullfile(test_folder, activity));
end

% Process each activity
for i = 1:length(activities)
    activity = activities(i);
    activity_path = fullfile(data_folder, activity);

    % Check if folder exists
    if ~isfolder(activity_path)
        fprintf('Skipping %s, folder not found.\n', activity_path);
        continue;
    end

    % Get list of CSV files, excluding annotation files
    csv_files = dir(fullfile(activity_path, 'user_*.csv'));

    image_paths = [];

    for j = 1:length(csv_files)
        file_name = csv_files(j).name;
        file_path = fullfile(activity_path, file_name);

        try
            % Read CSV data (skip text columns if present)
            data_table = readtable(file_path);
            numeric_data = table2array(data_table(:, varfun(@isnumeric, data_table, 'OutputFormat', 'uniform')));

            % Generate image
            figure('Visible', 'off');
            pcolor(numeric_data);
            shading flat;
            axis off;

            % Save image
            img_filename = strrep(file_name, '.csv', '.jpg');
            img_path = fullfile('generated_imagesss', activity, img_filename);
            mkdir(fileparts(img_path));
            saveas(gcf, img_path);
            close(gcf);

            % Store image path for train-test split
            image_paths = [image_paths; {img_path}];

        catch ME
            fprintf('Error processing %s: %s\n', file_path, ME.message);
        end
    end

    % Train-Test Split (80% Train, 20% Test)
    num_images = length(image_paths);
    num_train = round(0.8 * num_images);
    rand_indices = randperm(num_images);
    
    train_imgs = image_paths(rand_indices(1:num_train));
    test_imgs = image_paths(rand_indices(num_train+1:end));

    % Move images to respective Train and Test folders
    for k = 1:length(train_imgs)
        movefile(train_imgs{k}, fullfile(train_folder, activity, extractAfter(train_imgs{k}, 'generated_imagesss\')));
    end
    for k = 1:length(test_imgs)
        movefile(test_imgs{k}, fullfile(test_folder, activity, extractAfter(test_imgs{k}, 'generated_imagesss\')));
    end

    fprintf('Processed %s: %d train, %d test images.\n', activity, num_train, num_images - num_train);
end

fprintf('Dataset preparation complete.\n');
