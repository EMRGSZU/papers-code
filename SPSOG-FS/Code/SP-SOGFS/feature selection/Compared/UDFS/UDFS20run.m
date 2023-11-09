dim = size(fea,2);
if dim <400
    FeaNumCandi =10:10:100;
else
    FeaNumCandi =50:50:300;
end


UDFS_result = UDFS20(fea,gnd,FeaNumCandi);

save("F:\Users\cnnyl\Desktop\2dealt\UDFS\Isolet.mat",'UDFS_result')
