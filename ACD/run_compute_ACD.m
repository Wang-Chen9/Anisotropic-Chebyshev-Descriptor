clear;
close all;
clc;

% add path of this folder and all its subfolders
addpath(genpath('./tools/'));
addpath(genpath('./curvature/'));
addpath(genpath('./reorient /'));

% -----editable parameters--------
% important
n_angles=8; % default 8
num_angles=n_angles;
num_eigs=70; % default 70
nscales=6; % default 6
num_order=nscales;

% keep it the origin is ok.
alpha=4; % default 4;
curv_smooth=10; % default 10 is ok for all shapes

%
param.curv_smooth=curv_smooth;
param.alpha=alpha;
angles=linspace(0,180,n_angles+1);
param.angles=angles(1:end-1);
param.n_eigen=num_eigs;
param.order=nscales;

% ------------------------------------------------------------------------
DATA_ROOT_DIR       =fullfile('../CSMCNN/datasets/FAUST/');
SHAPE_DIR           =fullfile(DATA_ROOT_DIR,'shapes');
DESC_DIR            =fullfile(DATA_ROOT_DIR,'ACD');

warning off;
mkdir(DESC_DIR);
warning on;

% dispay infos
fprintf('[i] compute ACD descriptors:\n');

SHAPES = dir(fullfile(SHAPE_DIR, '*.mat'));
SHAPES = natsort({SHAPES.name}');

for s = 1:length(SHAPES)
    shapename = SHAPES{s};
    fprintf(1, '  %-30s \t', shapename);
    time_start = tic;
    % Load shape
    load(fullfile(SHAPE_DIR, shapename),'shape');
    area=shape.area;
    
    
    desc=compute_ACD(shape,param);
    
    % elasped time
    elapsed_time = toc(time_start);
    fprintf('%3.2fs\n',elapsed_time);
    save(fullfile(DESC_DIR,shapename),'desc','num_angles','num_eigs','num_order','alpha');    
    
end