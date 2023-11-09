% clear
% clc
dim = size(fea,2);
if dim <400
    FeaNumCandi =10:10:100;
else
    FeaNumCandi =50:50:300;
end

RSFS_goodpara = RSFS20test(fea,gnd,FeaNumCandi);
RSFS_result = RSFS20(fea,gnd,FeaNumCandi,RSFS_goodpara);


save("F:\Users\cnnyl\Desktop\2dealt\RSFS\Isolet.mat",'RSFS_goodpara','RSFS_result')
%     path = strcat(".\mytest\newpara\test\RSFS20\",'RSFS20',num2str(tryb),'.mat');
%     save(path,'accscore')
% end