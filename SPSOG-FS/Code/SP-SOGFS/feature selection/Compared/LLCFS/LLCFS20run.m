dim = size(fea,2);
if dim <400
    FeaNumCandi =10:10:100;
else
    FeaNumCandi =50:50:300;
end
        
LLCFS_result = LLCFS20(fea,gnd,FeaNumCandi);


save("F:\Users\cnnyl\Desktop\2dealt\LLCFS\Isolet.mat",'LLCFS_result')
