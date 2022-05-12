%读取shape，计算LBO矩阵及特征值和特征向量，并保存
addpath(genpath('./gptoolbox-master/'));
addpath(genpath('./tools/'));
DATA_ROOT_DIR=fullfile('../CSMCNN/datasets/FAUST/');
SHAPES_DIR=fullfile(DATA_ROOT_DIR,'shapes');
EVECS_DIR=fullfile(DATA_ROOT_DIR,'evecs');
mkdir(EVECS_DIR);

%
SHAPES=dir(fullfile(SHAPES_DIR,'*.mat'));
SHAPES={SHAPES.name}';

num_eigs=120;

for s=1:numel(SHAPES)
    shapename=SHAPES{s};
    fprintf(1,'%s processing....\n',shapename);
    %load shape
    load(fullfile(SHAPES_DIR,shapename));
    V=[shape.X,shape.Y,shape.Z];
    F=shape.TRIV;
    
    W=cotmatrix(V,F);
    W=-W;
    A=massmatrix(V,F,'barycentric');
    
    %特征分解
    [evecs,evals]=eigs(W,A,num_eigs,1e-5);
%     [evecs,evals]=eigs(W,num_eigs,'sm');
    evals=diag(evals);
    [evals,idx]=sort(evals);
    evecs=evecs(:,idx);
    
    shape.W=W;
    shape.A=A;
    shape.evecs=evecs;
    shape.evals=evals;
    
    shape.A=full(diag(shape.A));%
    
    save(fullfile(EVECS_DIR,shapename),'shape');
    clear shape;
end