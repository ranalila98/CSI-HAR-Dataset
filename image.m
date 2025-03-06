% Define folders
data_folder = 'data';
train_folder = 'generated_images/Train';
test_folder = 'generated_images/Test';

% List of activities
activities = {'lie down', 'fall', 'bend', 'run', 'sitdown', 'standup', 'walk'};

% Ensure train and test folders exist
for i = 1:length(activities)
    mkdir(fullfile(train_folder, activities{i}));
    mkdir(fullfile(test_folder, activities{i}));
end

% Process each activity folder
for i = 1:length(activities)
    activity = activities{i};
    activity_path = fullfile(data_folder, activity);
    
    if ~isfolder(activity_path)
        fprintf('Skipping %s, folder not found.\n', activity_path);
        continue;
    end

    image_paths = {};

    % Loop over files in the activity folder
    files = dir(fullfile(activity_path, '*.csv'));
    for j = 1:length(files)
        filename = files(j).name;
        
        % Skip annotation files
        if startsWith(filename, 'Annotation_')
            continue;
        end
        
        file_path = fullfile(activity_path, filename);

        try
            % Load CSV using readtable (for easier handling of mixed data types)
            data = readtable(file_path);

            % Drop non-numeric columns
            numeric_data = table2array(data(:, varfun(@isnumeric, data, 'OutputFormat', 'uniform')));

            % Generate image using pcolor
            figure('Visible', 'off'); % Hide figure window
            pcolor(numeric_data);  % pcolor is used here
            shading flat;          % Apply flat shading
            axis off;              % Hide axes

            % Save image as PNG
            img_filename = strrep(filename, '.csv', '.png');
            img_path = fullfile('generated_images', activity, img_filename);
            saveas(gcf, img_path);
            close;

            % Store image path
            image_paths{end + 1} = img_path;

        catch exception
            fprintf('Error processing %s: %s\n', file_path, exception.message);
        end
    end

    % Train-Test Split (80% Train, 20% Test)
    num_images = length(image_paths);
    num_train = floor(0.8 * num_images);
    train_imgs = image_paths(1:num_train);
    test_imgs = image_paths(num_train + 1:end);

    % Move images to respective Train and Test folders
    for j = 1:length(train_imgs)
        [~, img_name, ext] = fileparts(train_imgs{j});
        movefile(train_imgs{j}, fullfile(train_folder, activity, [img_name ext]));
    end

    for j = 1:length(test_imgs)
        [~, img_name, ext] = fileparts(test_imgs{j});
        movefile(test_imgs{j}, fullfile(test_folder, activity, [img_name ext]));
    end

    fprintf('Processed %s: %d train, %d test images.\n', activity, length(train_imgs), length(test_imgs));
end

disp('Dataset preparation complete.');
