%% generate synthetic data for testing the AutoEncoder. 
% Data are generated in a biomimetic way; however, the purpose of this script is mainly to show the code running.
clear all
close all


%% Biphasic Gaussian generator. --> TEST dataset
% Approximately, in fast reaching movements, there is a
% biphasic EMG activation pattern, related to acceleration and deceleration. 
% This is modelled with two Gaussian EMG bursts.
% for the test, I want to simulate 9 directions, 100 samples each, 16
% muscles (as in Scano et al., 2019, P2P movements).
n_dir = 9; n_samples = 100; n_muscles = 16; n_bursts = 2;
temp = zeros(100,1);  
n_planes = 5; %f=frontal; r=right; l=left; h=horizontal; u=up;

for p=1:n_planes
    synthetic_test{p} = zeros(900,16);
    for i = 1: n_muscles
        for j = 1: n_dir
            for k =1: n_bursts
    %             x_width = 30;        %burst width (in samples) interval 20-35
    %             starting_point = 10; %burst begin interval 1st Gaussian 5-20  2nd Gaussian 50-65
    %             sigma = 6;           %std interval 2-8
    %             gain = 1.5;          %Gaussian amplitude scaling 0-1
                x_width =  round(25 + (40-25) *rand(1)) ;        %burst width (in samples)
                gain =  0 + (1-0) *rand(1) ;                     %Gaussian amplitude scaling
                sigma = round(2 + (8-2) *rand(1));               %std
                if k==1
                     starting_point = round(10 + (25-10) *rand(1)) ;  %burst begin
                else if k==2
                        starting_point = round(45 + (60-45) *rand(1)) ;  %burst begin
                    end
                end
                mean  = starting_point+x_width/2; 
                x = starting_point : 1 :starting_point+x_width;
                y = gaussmf(x,[sigma mean])*gain;
                temp(x) = y;
                synthetic_test{p}(j*n_samples-99:j*n_samples,i) = temp;
            end
            temp = zeros(100,1);
        end
    end
end
figure,
for i = 1:min(size(synthetic_test{1}))
    subplot(16,1,i)
    area(synthetic_test{1}(:,i))
end

% assign test data
test_export_f(1,:,:) = synthetic_test{1};
test_export_r(1,:,:) = synthetic_test{2};
test_export_l(1,:,:) = synthetic_test{3};
test_export_h(1,:,:) = synthetic_test{4};
test_export_u(1,:,:) = synthetic_test{5};

%% generate the training set
n_trials_training = 9; % each set of movements is repeated 10 times; 9 used for training
noise_gain = 0.2;
for p=1:n_planes
    synthetic_training{p} = zeros (9,900,16);
    for t=1:n_trials_training
        synthetic_training{p}(t,:,:) = synthetic_test{p};
        for s=1:n_dir*n_samples
            for i=1:n_muscles
                synthetic_training{p}(t,s,i) = synthetic_training{p}(t,s,i) + noise_gain *rand(1);
            end
        end
    end
end
% shows an example of noisy data
figure,
for i = 1:min(size(synthetic_test{1}))
    subplot(16,1,i)
    b = squeeze(synthetic_training{1}(1,:,i));
    area(b)
end    

% assign training data
train_export_f = synthetic_training{1};
train_export_r = synthetic_training{2};
train_export_l = synthetic_training{3};
train_export_h = synthetic_training{4};
train_export_u = synthetic_training{5};

%% save

save('S00_input.mat', 'test_export_f', 'test_export_h', 'test_export_l', 'test_export_r', 'test_export_u', 'train_export_f', 'train_export_h', 'train_export_l', 'train_export_r', 'train_export_u');

