function desc=compute_ACSD(shape,param)
% shape: struct with fields, shape.X,shape.Y,shape.Z,shape.TRIV
% param: struct with fields and the default values can be seen in file
%        run_compute_ACD.m

options.curv_smooth = param.curv_smooth;
options.n_eigen = param.n_eigen;
options.alpha = param.alpha;
options.angle = 0;
order=param.order;

desc=cell(1,order);

for an=1:numel(param.angles)
    options.angle=pi*param.angles(an)/180;
    [evecs,evals,~,~]=calc_ALB([shape.X,shape.Y,shape.Z],shape.TRIV,options);
    desc{an}=compute_ACSD_single(evecs,evals,order);
end
desc=cell2mat(desc);
desc=normalize(desc,'L2',2);
end


function desc=compute_ACSD_single(evecs,evals,order)
[~,K]=size(evecs);
cheby_filters=zeros(K,order);

x=2*evals./evals(end)-1; % shift evals to [-1,1]

cheby_filters(:,1)=ones(K,1);
cheby_filters(:,2)=x;

for k=3:order
    cheby_filters(:,k)=2*x.*cheby_filters(:,k-1)-cheby_filters(:,k-2);
end

desc=evecs.^2*cheby_filters;
end










