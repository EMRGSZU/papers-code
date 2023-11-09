legendArr = {'SOGFS-SP','SOGFS','USFS','URAFS','RSFS','UDFS','LLCFS'};



NewFeaNum = 50:50:300; 
% NewFeaNum = 10:10:100;
% sogfs = accscoreCopy(4).output;
% bestAcc = accscoreCopy1(4).output;
% RSFS_result = accscoreCopy2(4).output;
% UDFS_result = accscoreCopy3(4).output;
% LLCFS_result =accscore(4).output;

    figure(1)
%     title('Compared BA ACC','Color','b')
    xlabel('Num of selected features'),ylabel('Acc')
    line(NewFeaNum,SPSOGFS_result,'Color','r','Marker','o','linewidth',1.1)
%     line(NewFeaNum,a1,'Color','k','Marker','v','LineStyle','--','linewidth',1.1)
    line(NewFeaNum,SOGFS_result,'Color','b','Marker','*','linewidth',1.1)
    line(NewFeaNum,USFS_result,'Color',[0.39,0.83,0.07],'Marker','s','LineStyle','--','linewidth',1.1)
    line(NewFeaNum,URAFS_result,'Color','m','Marker','.','LineStyle',':','linewidth',1.1)
    line(NewFeaNum,RSFS_result,'Color','k','Marker','+','LineStyle',':','linewidth',1.1)
    line(NewFeaNum,UDFS_result,'Color',[0.93,0.69,0.13],'Marker','x','LineStyle','--','linewidth',1.1)
    line(NewFeaNum,LLCFS_result,'Color',[0.49,0.18,0.56],'Marker','d','linewidth',1.1)
   
    legend(legendArr,'Location','southwest')
%     
% 
%     subplot(2,2,2)
%     subtitle('Compared SRBCT ACC','Color','b')
%     xlabel('Num of selected features'),ylabel('Acc')
%     line(NewFeaNum,bestAcc2,'Color','r','Marker','o','linewidth',1.1)
%     line(NewFeaNum,SRBCT,'Color','k','Marker','v','LineStyle','--','linewidth',1.1)
%     legend(legendArr,'Location','southwest')
%    
% NewFeaNum = 10:10:100;
% 
%     figure(1)
%     title('Compared BA ACC','Color','b')
%     xlabel('Num of selected features'),ylabel('Acc')
%     line(NewFeaNum,bestAcc,'Color','r','Marker','o','linewidth',1.1)
%     line(NewFeaNum,a1,'Color','k','Marker','v','LineStyle','--','linewidth',1.1)
%     line(NewFeaNum,bestAccCopy,'Color','b','Marker','*','linewidth',1.1)
%     legend(legendArr,'Location','southwest')
    
%     subplot(2,2,4)
%     subtitle('Compared BA ACC','Color','b')
%     xlabel('Num of selected features'),ylabel('Acc')
%     line(NewFeaNum,bestAcc4,'Color','r','Marker','o','linewidth',1.1)
%     line(NewFeaNum,BA,'Color','k','Marker','v','LineStyle','--','linewidth',1.1)
%     legend(legendArr,'Location','southwest')