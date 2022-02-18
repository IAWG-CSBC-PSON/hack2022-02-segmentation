%%
clc
clear
if(~isdeployed)
    cd(fileparts(which('trainData.m')));
end
%% Store data
% Note this only grabs the first channel right now
% TODO: Add extra channel
% imds1 = imageDatastore("../images/LPTCdapilamin64/artefactStores/channel1/");
% imds2 = imageDatastore("../images/LPTCdapilamin64/artefactStores/channel1/");

imds = imageDatastore({'../images/LPTCdapilamin64/artefactStores/channel1/', '../images/LPTCdapilamin64/artefactStores/channel2/'});
% classNames = ["background", "blobs", "lint", "tile artefacts", "background"];
% labelID = [0,1,2,3,4];

classNames = ["background", "artefact"];
labelID = {0, 1};

pxds = pixelLabelDatastore({'../images/LPTCdapilamin64/artefactStores/masks/', '../images/LPTCdapilamin64/artefactStores/masks/'}, classNames, labelID);
%% Gather data
clf
im = imread(imds.Files{1});

C = read(pxds);
I = read(imds);


% Show an image with annotation
cmap = makeCmap(length(classNames));
B = labeloverlay(I,C{1}, 'Colormap', cmap);

figure(1)
subplot(121)
imshow(B)
pixelLabelColorbar(cmap,classNames);
subplot(122)
imshow(I)
%%
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionStores(imds,pxds,0.6,0.2);
%% Building the network
% Specify the network image size. This is typically the same as the traing image sizes.
imageSize = [256 256];

% Specify the number of classes.
numClasses = numel(classNames);

% Create DeepLab v3+.
% lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");
layer = dropoutLayer(0.1);
lgraph = unetLayers(imageSize,numClasses,'EncoderDepth',5);

%% Training options
dsVal = combine(imdsVal,pxdsVal);
dsTrain = combine(imdsTrain, pxdsTrain);

% Define training options. 
% options = trainingOptions('sgdm', ...
%     'LearnRateSchedule','piecewise',...
%     'LearnRateDropPeriod',10,...
%     'LearnRateDropFactor',0.3,...
%     'Momentum',0.9, ...
%     'InitialLearnRate',1e-3, ...
%     'L2Regularization',0.005, ...
%     'ValidationData',dsVal,...
%     'MaxEpochs',5, ...  
%     'MiniBatchSize',8, ...
%     'Shuffle','every-epoch', ...
%     'CheckpointPath', tempdir, ...
%     'VerboseFrequency',2,...
%     'Plots','training-progress',...
%     'ValidationPatience', 4);

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs',30, ...
    'VerboseFrequency',10,...
    'MiniBatchSize',10,...
    'Momentum',0.9,...
    'Plots','training-progress',...
    'ValidationData', dsVal,...
    'L2Regularization', 10^-2);
%% Augmentation (Optional)

%% Training
tic
doTraining = true;
if doTraining    
    [net, info] = trainNetwork(dsTrain,lgraph,options);
else
    data = load(pretrainedNetwork); 
    net = data.net;
end
toc
%% Get results
pxdsResults = semanticseg(imdsTest,net, ...
    'MiniBatchSize',4, ...
    'WriteLocation',tempdir, ...
    'Verbose',false);

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);
%%

%%
I = readimage(imdsTest,7);
C = semanticseg(I, net);
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
imshow(B)
pixelLabelColorbar(cmap, classNames);
%% Functions
function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.

colormap(gca,cmap)

% Add colorbar to current figure.
c = colorbar('peer', gca);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end

function cmap = makeCmap(n)
% Define the colormap used by CamVid dataset.

j = jet();
cmap = j(round(linspace(1,length(j),n)),:);
% Normalize between [0 1].
% cmap = cmap ./ 255;
end

function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionStores(imds,pxds,trainProp,valProp)
% Partition CamVid data by randomly selecting 60% of the data for training. The
% rest is used for testing.

% Set initial random state for example reproducibility.
% rng(1234);
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use 60% of the images for training.
numTrain = round(trainProp * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use 20% of the images for validation
numVal = round(valProp * numFiles);
valIdx = shuffledIndices(numTrain+1:numTrain+numVal);

% Use the rest for testing.
testIdx = shuffledIndices(numTrain+numVal+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
valImages = imds.Files(valIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(valImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = 0:1;

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
valLabels = pxds.Files(valIdx);
testLabels = pxds.Files(testIdx);

pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(valLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end
