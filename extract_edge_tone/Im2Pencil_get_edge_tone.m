% Demo for Structured Edge Detector (please see readme.txt first).
addpath('./toolbox/');
%% set opts for training (see edgesTrain.m)
opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=0;                 % set to true to enable nms

%% evaluate edge detector on BSDS500 (see edgesEval.m)
if(0), edgesEval(model, 'show',1, 'name','' ); end

%% detect edge and visualize results

path = './data/input/';
files = dir([path, '*.jpg']);
Imgnum = size(files, 1);
result_path = './data/edge_tone/';
mkdir(result_path);
for i=1:Imgnum
    I = imread(strcat(path, files(i,1).name));
    if size(I, 3) < 3
        I = repmat(I, [1,1,3]);
    end
    %%
    E=edgesDetect(I,model);
    imwrite(1-E, strcat(result_path, files(i,1).name(1:end-4),'_edge.jpg'));
    %%
    I=rgb2gray(I);
    nhoodSize = [11, 11];
    smoothValue=0.5*diff(getrangefromclass(I)).^2;
    I=imguidedfilter(I,I,'NeighborhoodSize', nhoodSize, 'DegreeOfSmoothing',smoothValue);
    imwrite(I,strcat(result_path,files(i,1).name(1:end-4),'_gf.jpg'));
end
