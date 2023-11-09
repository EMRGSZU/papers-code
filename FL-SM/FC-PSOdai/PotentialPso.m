 function [EP,pbest,Iter_time,Fitness_time,Update_time,S_rate,Do_time]=PotentialPso(M,data,cut)
% M=150;
% load('data/SRBCT.mat');
% data=[data(:,2:end) data(:,1)];
% cut=disc_MDLP(data);


Iter=50;       %最大迭代次数
D=size(data,2)-1;             %搜索空间维数
datalabel=data(:,end);
dlength=size(data,1);
ssindex=cell(1,dlength);

ddindex=cell(1,dlength);

n_feature=D;
Iter_time=zeros(1,Iter);
Fitness_time=zeros(1,Iter);
Update_time=zeros(1,Iter);
Do_time=zeros(1,Iter);
S_rate=zeros(1,Iter);

featurelib=zeros(1,n_feature);
featurescore=zeros(1,n_feature);
subfeaturescore=zeros(1,n_feature);
featureacc=zeros(1,n_feature);
popscore=zeros(1,M);
elite=0;
srate=0;
jrate=0;
jscore=0;
gbestacc=0;
flag_0=0;
flag=0;

for i=1:dlength
    testlabel=datalabel(i);
    ddindex{i}=find(testlabel~=datalabel);  %不同类别的
    temps=find(testlabel==datalabel); %相同类别的
    ssindex{i}=temps(temps~=i); %相同类别的,排除自己  
end 

EP=[];


T=ceil(M/15); %邻居个数
nObj=3;
%创建子问题
sp=CreateSubProblems(nObj,M,T);
%空个体
empty_individual.Position=zeros(1,D);
empty_individual.Cost=[];
empty_individual.g=[];
empty_individual.IsDominated=[];
empty_individual.Gbest=[];
empty_individual.select=0;
h=[0,0,1/D];
pop=repmat(empty_individual,M,1);
pbest=repmat(empty_individual,M,1);

parfor i=1:M   
    for j=1:D
        if cut(j,1)~=0     
        pop(i).Position(j)=randi([0 cut(j,1)]);
        end
    end
end

fitg=1./zeros(M,1);

parfor j=1:M
%parfor j=1:M
         tmpdata=data;
%          sindex=[]; %select feature index
         nindex=[]; %n-select feature index 
        for k=1:D   %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度  
            if pop(j).Position(:,k)~=0
                for z=1:pop(j).Position(:,k)+1
                   if z==1
                       tmpdata(data(:,k)<=cut(k,z+1),k)=z;
                   elseif z==pop(j).Position(:,k)+1
                       tmpdata(data(:,k)>cut(k,z),k)=z;
                   else
                       tmpdata(data(:,k)>cut(k,z) & data(:,k)<=cut(k,z+1),k)=z;
                   end
                end
       
            else
                nindex=[nindex k];
            end
        end
        tmpdata(:,nindex)=[];
%          pop(j).Cost=fitness(tmpdata,D)';
%         [balance_acc(j,1),distance(j,1),pf(j,1)]=fitness(tmpdata,D);
%         pop(j).Cost=[balance_acc(j,1) distance(j,1) pf(j,1)];
       
        
        pop(j).Cost=fitness(tmpdata,D,ssindex,ddindex)';
        pop(j).g=DecomposedCost(pop(j),h,sp(j).lambda);
%         z=min(z,pop(j).Cost);
            %如果粒子当前位置支配的话
        if pop(j).g<=fitg(j,:)
        pbest(j)=pop(j);   %更新每个粒子的最佳位置
        fitg(j,:)=pop(j).g;
        end
        
end


 pop=DetermineDomination(pop,flag);
 %创造额外解
 ndpop=pop(~[pop.IsDominated]);
 EP=[ndpop]; %#ok
   % EP=EP(~[EP.IsDominated]);
 [~,deld,~]=unique(cat(1,EP.Position),'rows');
 EP=EP(deld);
tmpeps=size(EP,1);
for it=1:Iter
iter_time1 = clock;

%求出gbest
    GT=numel(EP);
    EI=randsample(GT,1);
    gbest=EP(EI).Position;
    
    
    if it<Iter/5
        flag=0;
    end
    
    if it>Iter/5 & it<Iter*4/5
            if srate==0 & flag~=0
                if flag==3
                    flag=2;
                elseif flag==2 
                    flag=1;
                else
                    flag=0;
                end
            else
                if flag==1 & srate>0.7
                    flag=3;
                end
            end
            
            if flag==0
                flag_0=flag_0+1;
            end
            
            if flag_0>=3
                flag=3;
                flag_0=0;
            end
    end
    
    
    if it==Iter/5 
       featurescore=scoresystem2(featurelib,featureacc,elite,featurescore);
       a=transpose({pop.Cost});
       a=cell2mat(a);
       avg1=mean(a);
       featurelib=zeros(1,n_feature);
       featureacc=zeros(1,n_feature);
       elite=0;
       flag=3;
    end
    
    if it>Iter/5
        if avg1(1)-avg2(1)>=0.1 |avg1(3)-avg2(3)>=0.05
            subfeaturescore=scoresystem2(featurelib,featureacc,elite,subfeaturescore);
            avg1(1)=avg2(1);
            avg1(3)=avg2(3);
            pos=[];
            for sub=1:n_feature
                if featurescore(sub)~=0
                    if subfeaturescore(sub)<featurescore(sub) &featurescore(sub)~=0
                        featurescore(sub)=0;
                    else
                        featurescore(sub)=subfeaturescore(sub);
                    end
                end
            end
            elite=0;
     end
            
  end
    
   if flag~=0
       [jscore,jrate]=caljscore(pop,flag,featurescore);
   end
srate=0; 

fittime1=clock;

 parfor i=1:M %parfor
  
   %1g 内部
   
       meanb=(pbest(i).Position+gbest)/2;
       std=abs(pbest(i).Position-gbest);
        for k=1:D   %根据每个粒子转换数据
            smethod=rand();
            if  smethod<=0.9
                if rand()<0.5
                    updatep=round(normrnd(meanb(:,k),std(:,k))); %更新粒子
                    if updatep>cut(k,1)||updatep<0 
                        pop(i).Position(k)=0;
                    else
                          pop(i).Position(k)=updatep;
                    end
                else
                      pop(i).Position(k)=pbest(i).Position(k);
                end
            else
               %if rand()<0.2-tmpeps/M   %粒子突变 
                if rand()<0.2-(it/Iter)*0.2
                     pop(i).Position(k)=0;
               end                      
            end           
        end      
        tmpdata=data;
%       sindex=[]; %select feature index
        nindex=[]; %n-select feature index 
        if size(find(pop(i).Position),2)==0
            pop(i).Position=initparticle(cut,D);
        end
        for k=1:D   %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度 
            if pop(i).Position(k)~=0
                for z=1:pop(i).Position(k)+1
                   if z==1
                       tmpdata(data(:,k)<=cut(k,z+1),k)=z;
                   elseif z==pop(i).Position(k)+1
                       tmpdata(data(:,k)>cut(k,z),k)=z;
                   else
                       tmpdata(data(:,k)>cut(k,z) & data(:,k)<=cut(k,z+1),k)=z;
        
                   end
                end
       
            else
                nindex=[nindex k];
            end
        end
           
        tmpdata(:,nindex)=[];
        if flag~=0
            if surrogate(featurescore,pop(i).Position,jscore,jrate)
                pop(i).Cost= fitness(tmpdata,D,ssindex,ddindex)';   
                pop(i).g=DecomposedCost(pop(i),h,sp(i).lambda);
                pop(i).select=1;
                srate=srate+1;
            else
                pop(i).select=0;
            end
        else
            pop(i).Cost= fitness(tmpdata,D,ssindex,ddindex)';   
            pop(i).g=DecomposedCost(pop(i),h,sp(i).lambda);
        end
       


 end
  fittime2=clock;
 Fitness_time(it)=etime(fittime2,fittime1);
 uptime1=clock;
 
 parfor i=1:M
    if pop(i).g<=fitg(i,:)
        pbest(i,:)=pop(i);   %更新每个粒子的最佳位置
        fitg(i,:)=pop(i).g;
    end
end 
 
 for i=1:M
      for j=sp(i).Neighbors
            pop(i).g=DecomposedCost(pop(i),h,sp(j).lambda);
            if pop(i).g<=pop(j).g
                pop(j).g=pop(i).g;
                pop(j).Position=pop(i).Position;
            end
      end 
      if pop(i).g<=fitg(i,:)
        pbest(i,:)=pop(i);   %更新每个粒子的最佳位置
        fitg(i,:)=pop(i).g;
      end
 end
 uptime2=clock;
 Updata_time(it)=etime(uptime2,uptime1);
 
 dom_time1=clock;
    pop=DetermineDomination(pop,flag);
    
    ndpop=pop(~[pop.IsDominated]);
    
    EP=[EP;ndpop]; %#ok
    
    EP=DetermineDomination(EP,flag);
    EP=EP(~[EP.IsDominated]);
    [~,deld,~]=unique(cat(1,EP.Position),'rows');
    EP=EP(deld);
dom_time2=clock;   
 Do_time(it)=etime(dom_time2,dom_time1);

 for j=1:M
      featurelib=featurelib+pop(j).Position;
      featureacc=featureacc+pop(j).Position*(1-pop(j).Cost(1));
 end
 elite=elite+j;
        
        
        
    iter_time2=clock;
    Iter_time(it)=etime(iter_time2,iter_time1);
    S_rate(it)=srate/M;
    logger(['# iteration = ', num2str(it), '/', num2str(Iter), ', time = ', num2str(Iter_time(it)), 's',', fittime = ', num2str(Fitness_time(it)),', uptime = ', num2str(Updata_time(it)),', domtime = ', num2str( Do_time(it))]);
    b=transpose({pop.Cost});
    a=cell2mat(b);
    avgacc = mean(a);
    avg2=avgacc;
    logger(['# avg fitness = ', num2str(avgacc(1)), ', ', num2str(avgacc(2)), ', ', num2str(avgacc(3)),', srate =',num2str(S_rate(it))]);
  
end
end
    




