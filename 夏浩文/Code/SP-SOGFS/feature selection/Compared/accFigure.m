legendArr = {'Now','Before','SOGFS'};



NewFeaNum = 50:50:300;
    

    set(gca, 'GridAlpha',0)
    
    subplot(2,2,1)
    subtitle('Compared JAFFE ACC','Color','b')
    xlabel('Num of selected features'),ylabel('Acc')
    line(NewFeaNum,bestAcc1,'Color','r','Marker','o','linewidth',1.1)
    line(NewFeaNum,JAFFE,'Color','k','Marker','v','LineStyle','--','linewidth',1.1)
    legend(legendArr,'Location','southwest')
    

    subplot(2,2,2)
    subtitle('Compared SRBCT ACC','Color','b')
    xlabel('Num of selected features'),ylabel('Acc')
    line(NewFeaNum,bestAcc2,'Color','r','Marker','o','linewidth',1.1)
    line(NewFeaNum,SRBCT,'Color','k','Marker','v','LineStyle','--','linewidth',1.1)
    legend(legendArr,'Location','southwest')
   
NewFeaNum = 10:10:100;

    subplot(2,2,3)
    subtitle('Compared USPS ACC','Color','b')
    xlabel('Num of selected features'),ylabel('Acc')
    line(NewFeaNum,bestAcc3,'Color','r','Marker','o','linewidth',1.1)
    line(NewFeaNum,USPS,'Color','k','Marker','v','LineStyle','--','linewidth',1.1)
    legend(legendArr,'Location','southwest')
    
    subplot(2,2,4)
    subtitle('Compared BA ACC','Color','b')
    xlabel('Num of selected features'),ylabel('Acc')
    line(NewFeaNum,bestAcc4,'Color','r','Marker','o','linewidth',1.1)
    line(NewFeaNum,BA,'Color','k','Marker','v','LineStyle','--','linewidth',1.1)
    legend(legendArr,'Location','southwest')