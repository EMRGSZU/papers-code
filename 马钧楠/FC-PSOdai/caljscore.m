function [jscore,jrate]= caljscore(pop,flag,featurescore)
        n=size(pop,1);
        popscore=zeros(1,n); 
        
        for j=1:n
            tscore=featurescore.*pop(j).Position;
            tscore=sum(tscore);
            popscore(j)=tscore;
            poprate(j)=pop(j).Cost(2);
        end
        
        if flag==1
            jscore=mean(popscore);
            jrate=mean(poprate);
        elseif flag==2
            popscore=popscore*(-1);
            jscore=geomean(popscore)*(-1);
            jrate=geomean(poprate);
        else
            popscore=popscore*(-1);
            jscore=harmmean(popscore)*(-1);
            jrate=harmmean(poprate);
        end
end