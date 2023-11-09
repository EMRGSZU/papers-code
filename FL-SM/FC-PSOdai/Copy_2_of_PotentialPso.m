function [EP,pbest,EP3,PB3,EP4,PB4,EP5,PB5,EP6,PB6]=PotentialPso(M,data,cut)
% Solution=PotentialPso(150,traind,cut);
Iter=50;       %最大迭代次数
D=size(data,2)-1;             %搜索空间维数

% balance_acc=zeros(M,1);
% distance=zeros(M,1);
% pf=zeros(M,1);
  %暂时无用，后续需要修改
% x=zeros(M,D);
EP=[];
EP3=[];
PB3=[];
EP4=[];
PB4=[];
EP5=[];
PB5=[];
EP6=[];
PB6=[];

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
h=[0,0,1/D];
pop=repmat(empty_individual,M,1);
pbest=repmat(empty_individual,M,1);
%pbest=[];  %预分配内存  

% for i=1:M   %初始化粒子位置 根据潜在切点表x为粒子位置
%     for j=1:D
%         if cut(j,1)==0
%             pop(i).Position(j)=0;
%         else         
%         pop(i).Position(j)=randi([0 cut(j,1)]);
%         end
%     end
% end

parfor i=1:M   %初始化粒子位置 根据潜在切点表x为粒子位置
    for j=1:D
        if cut(j,1)~=0     
        %pop(i).Position(j)=randi([0 cut(j,1)]);
        if rand()<0.5
            pop(i).Position(j)=0
        else
             pop(i).Position(j)=randi(cut(j,1))
        end
        end
    end
end
%  nozero=(cut(:,1)~=0);
%  a=find(nozero);
% for i=1:M   %初始化粒子位置 根据潜在切点表x为粒子位置
%     
%     for j=a   
% %         pop(i).Position(:,j)=randi([0 cut(j,1)]);
% pop(i).Position(:,j)=randi([0 1]);
%     end
% end
fitg=1./zeros(M,1);
%fitcost=ones(M,3);
%for j=1:M
parfor j=1:M
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
       
        
        pop(j).Cost=fitness(tmpdata,D)';
        pop(j).g=DecomposedCost(pop(j),h,sp(j).lambda);
%         z=min(z,pop(j).Cost);
            %如果粒子当前位置支配的话
        if pop(j).g<=fitg(j,:)
        pbest(j)=pop(j);   %更新每个粒子的最佳位置
        fitg(j,:)=pop(j).g;
        end
        
end


 pop=DetermineDomination(pop);
 %创造额外解
 ndpop=pop(~[pop.IsDominated]);
 EP=[ndpop]; %#ok
   % EP=EP(~[EP.IsDominated]);
 [~,deld,~]=unique(cat(1,EP.Position),'rows');
 EP=EP(deld);
tmpeps=size(EP,1);
for it=1:Iter
    %disp(it);
    %是否支配 
%     for i=1:M
%        [~,kk]=min(cat(1,pop(sp(i).Neighbors).g));
%        pop(i).Gbest=pop(kk).Position;1
%     end
    %交叉开始
%     if it==Iter
%         break;
%     end  

%求出gbest
    GT=numel(EP);
    EI=randsample(GT,1);
    gbest=EP(EI).Position;

 parfor i=1:M
  
   %1g 内部
   
       mean=(pbest(i).Position+gbest)/2;
       std=abs(pbest(i).Position-gbest);
        for k=1:D   %根据每个粒子转换数据
            smethod=rand();
            if  smethod<=0.9
                if rand()<0.5
                    updatep=round(normrnd(mean(:,k),std(:,k))); %更新粒子
                    if updatep>cut(k,1)||updatep<0 
                        pop(i).Position(k)=0;
                    else
                          pop(i).Position(k)=updatep;
                    end
                else
                      pop(i).Position(k)=pbest(i).Position(k);
                end
            else
               if rand()<0.15-(it/Iter)*tmpeps/M   %粒子突变 
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
       
         pop(i).Cost= fitness(tmpdata,D)';   
%         for j=sp(i).Neighbors
%             pop(i).g=DecomposedCost(pop(i),h,sp(j).lambda);
%             if pop(i).g<=pop(j).g 
%                  pop(j).g=pop(i).g;
%                  pop(j).Position=pop(i).Position;
%             end
%         end 
        pop(i).g=DecomposedCost(pop(i),h,sp(i).lambda);
%         z=min(z,pop(j).Cost);
            %如果粒子当前位置支配的话
%         if pop(i).g<=fitg(i,:)
%         pbest(i,:)=pop(i).Position;   %更新每个粒子的最佳位置
%         fitg(i,:)=pop(i).g;
%         end
 end
 
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
    pop=DetermineDomination(pop);
    
    ndpop=pop(~[pop.IsDominated]);
    
    EP=[EP;ndpop]; %#ok
    
    EP=DetermineDomination(EP);
    EP=EP(~[EP.IsDominated]);
    [~,deld,~]=unique(cat(1,EP.Position),'rows');
    EP=EP(deld);
    if it==30
        EP3=EP;
        PB3=pbest;
    end
    if it==40
        EP4=EP;
        PB4=pbest;
    end
    
    
    if it==50
        EP5=EP;
        PB5=pbest;
    end
    if it==60
        EP6=EP;
        PB6=pbest; 
    end
    
    
%     if numel(EP)>nArchive
%         Extra=numel(EP)-nArchive;
%         ToBeDeleted=randsample(numel(EP),Extra);
%         EP(ToBeDeleted)=[];
%     end
end
    

disp('---------------------END--------------------------');
disp('--------------------------------------------------');



