function [Solution,Iter_time,gbestacc,Fitness_time,Update_time,S_rate]=PotentialPso(M,data,cut)
% Solution=PotentialPso(200,traind);
Iter=70;       %最大迭代次数  1
% load('SRBCT.mat');
D=size(data,2)-1;             %搜索空间维数  1
% 
% M=150;             %初始化群体个体数目
% AddM=50;

% data=[data(:,2:end) data(:,1)];
% cut=disc_MDLP(data);
% maxp=size(cut,2)-1;
n_feature=D;


Solution=0;
Iter_time=zeros(1,Iter);
Fitness_time=zeros(1,Iter);
Update_time=zeros(1,Iter);
S_rate=zeros(1,Iter);

featurelib=zeros(1,n_feature);
featurescore=zeros(1,n_feature);
subfeaturescore=zeros(1,n_feature);
featureacc=zeros(1,n_feature);
popscore=zeros(1,M);
elite=0;
srate=0;
mscore=0;
gbestacc=0;
alltime1=clock;
parfor i=1:M   %初始化粒子位置 根据潜在切点表
    for j=1:D
        if cut(j,1)==0      
            x(i,j)=0;
        else         
       % x(i,j)=round(rand(1)*cut(j,1));
        x(i,j)=randi([0 cut(j,1)]);
        end
    end
end


%x(i,j)为切点坐标

%p(1:M,1)=-inf;
flag=0;
score_flag=0;
fitnum=zeros(M,1);
gbestvalue=zeros(Iter,1);
gbestfit=0;
for it=1:Iter
    iter_time1 = clock;
    

    if it>Iter/7 
        if srate==0 & score_flag~=0
            if score_flag==3
                score_flag=2;
            elseif score_flag==2 
                score_flag=1;
            else
                score_flag=0;
            end
        else
            if score_flag==1 & srate>0.7
               score_flag=3;
            end
        end
    end
            
%             if score_flag==0
%                 flag_0=flag_0+1;
%             end
%             
%             if flag_0>=3
%                 score_flag=3;
%                 flag_0=0;
%             end
%    end
        
%          if it>=n_iter*6/7
%              score_flag=1;
%          end
    if it==Iter/7
        featurescore=scoresystem2(featurelib,featureacc,elite,featurescore);
        avg1=mean(fitnum);
        featurelib=zeros(1,n_feature);
        featureacc=zeros(1,n_feature);
        elite=0;
        score_flag=3;
    end
    
    if it>Iter/7
            if avg1-avg2>=0.1 
                %subfeaturescore=scoresystem1(featurelib,featureacc,n_featurelib,elite,n_feature,featurescore);
                subfeaturescore=scoresystem2(featurelib,featureacc,elite,subfeaturescore);
                %featurescore=subfeaturescore;
                avg1=avg2;
                %mscore=mean(subfeaturescore);
%                 pos=[];
%                 z1=1;
                for sub=1:n_feature
                    if featurescore(sub)~=0
                        if subfeaturescore(sub)<featurescore(sub) &featurescore(sub)~=0
                            featurescore(sub)=0;
%                              pos(z1)=sub;
                        else
                            featurescore(sub)=subfeaturescore(sub);
                        end
                    end
                end
                elite=0;
            end
    end
    
   if it>=Iter/7
       n=size(x,1);
        popscore=zeros(1,n);
        
        for j=1:n
            tscore=featurescore.*x(j,:);
            tscore=sum(tscore);
            popscore(j)=tscore;
        end
        if score_flag==1
            mscore=mean(popscore);
        elseif score_flag==2
            popscore=popscore*(-1);
            mscore=geomean(popscore)*(-1);
            popscore=popscore*(-1);
        else
            popscore=popscore*(-1);
            mscore=harmmean(popscore)*(-1);
            popscore=popscore*(-1);
        end
   end
    
    

    
    
    
    srate=0;
    fit_time1=clock;
    parfor j=1:M
         tmpdata=data;
         sindex=[]; %select feature index
         nindex=[]; %n-select feature index 
        
         
         for k=1:D   %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度  
            if x(j,k)>0 
                index1=find(data(:,k)>=cut(k,x(j,k)+1));
                index2=find(data(:,k)<cut(k,x(j,k)+1));
                tmpdata(index1,k)=1;
                tmpdata(index2,k)=0;
                sindex=[sindex k];            
            else
                nindex=[nindex k];
            end
        end
        tmpdata(:,nindex)=[];
        
        
        if score_flag~=0
            if popscore(j)>mscore
                [balance_acc(j,1),distance(j,1),tmpfitnum(j,1)]=fitness(tmpdata);
                srate=srate+1;
                if tmpfitnum(j,1)>fitnum(j,1)
                    pbest(j,:)=x(j,:);   %更新每个粒子的最佳位置
                    fitnum(j,1)=tmpfitnum(j,1);
                end
            end
            
        else
            [balance_acc(j,1),distance(j,1),tmpfitnum(j,1)]=fitness(tmpdata);
            if tmpfitnum(j,1)>fitnum(j,1)
                pbest(j,:)=x(j,:);   %更新每个粒子的最佳位置
                fitnum(j,1)=tmpfitnum(j,1);
            end
        end
                
        
    end
    fit_time2=clock;
    Fitness_time(it)=etime(fit_time2,fit_time1);

    [gvalue,index]=max(fitnum);
    if gbestfit<gvalue
        gbest=x(index,:);  %更新最优种群位置
        gbestfit=gvalue;
        gbestacc=balance_acc(index,1);
    end
    gbestvalue(it,1)=gbestfit;
   
   
    %缩放和停止准则
    if  it>=11&&gbestvalue(it,1)>=1.01*gbestvalue(it-10,1)&&gbestvalue(it-1,1)==gbestvalue(it-10,1)
        M=M+50;
        flag=1;
        break;
    elseif it>=11 &&gbestvalue(it-1,1)==gbestvalue(it-10,1)
        break;
    end
    
    %更新粒子位置
    up_time1=clock;
   parfor j=1:M
       if score_flag==0 | popscore(j)>mscore
           up_flag=1;
       else
           if rand>0.8
               up_flag=1;
           else
               up_flag=0;
           end
       end
       if up_flag
        mean_1=(pbest(j,:)+gbest)/2;
        std=abs(pbest(j,:)-gbest);
       for k=1:D   %根据每个粒子转换数据
            if  rand()<0.5
                updatep=round(normrnd(mean_1(:,k),std(:,k))); 
                if updatep>cut(k,1)||updatep<0
                    x(j,k)=0;
                else
                    x(j,k)=updatep;
                end
                
             else
                 x(j,k)=pbest(j,k);
             end                  
       end 
       end
      
    end
    up_time2=clock;
    Update_time(it)=etime(up_time2,up_time1);
    
    
    
    
    
    for j=1:size(x,1)
        c=x(j,:);
        featurelib=featurelib+c;
        featureacc=featureacc+c.*balance_acc(j);
    end
    elite=elite+j;
    avg2=mean(fitnum);
    srate=srate/M;
    S_rate(it)=srate;
    iter_time2 = clock;
    iter_time=etime(iter_time2, iter_time1);
    Iter_time(it)=iter_time;
    logger(['# iteration = ', num2str(it), '/', num2str(Iter), ', time = ', num2str(iter_time), 's',', fit_time = ',num2str(Fitness_time(it)),', up_time = ',num2str(Update_time(it))]);
    m_fit=avg2;
    logger(['# avg fitness = ',num2str(m_fit),' best num features = ',num2str(sum(gbest)),' srate = ',num2str(srate)]);
end


alltime2=clock;
alltime=etime(alltime2, alltime1);
itertime=alltime/it;
% m_fittime=mean(Fitness_time);
% m_uptime=mean(Update_time);
% m_rate=mean(S_rate);
logger(['alltime = ', num2str(alltime), 's','  itertime = ', num2str(itertime), 's']);

if flag==1
    [Solution,Iter_time,gbestacc,Fitness_time,Update_time,S_rate]=PotentialPso(M,data,cut);
end

if flag==0   
% disp('---------------------END--------------------------');
% if it<70
%         Iter_time(1,it:end)=0;
%         Fitness_time(1,it:end)=0;
%         Update_time(1,it:end)=0;
%         S_rate(1,it:end)=0;
% end  
Solution=gbest;
save SRBCTtestOne
% disp('--------------------------------------------------');
end
end








