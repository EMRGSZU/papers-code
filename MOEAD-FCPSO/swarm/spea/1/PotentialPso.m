function [EP,pbest]=PotentialPso(M,data,cut)
% Solution=PotentialPso(150,traind,cut);
Iter=50;       %最大迭代次数
D=size(data,2)-1;             %搜索空间维数
traind=data;
%pbest=zeros(M,D);  %预分配内存  
% balance_acc=zeros(M,1);
% distance=zeros(M,1);
% pf=zeros(M,1);
datalabel=data(:,end);
dlength=size(data,1);
ssindex=cell(1,dlength);

ddindex=cell(1,dlength);

for i=1:dlength
    testlabel=datalabel(i);
    ddindex{i}=find(testlabel~=datalabel);  %不同类别的
    temps=find(testlabel==datalabel); %相同类别的
    ssindex{i}=temps(temps~=i); %相同类别的,排除自己  
end 

% tmpfitnum=zeros(M,1);   %暂时无用，后续需要修改
% x=zeros(M,D);
% 
% 
% nArchive=15;
nObj=3;
K=round(sqrt(2*M));

%空个体

empty_individual.Position=zeros(1,D);
empty_individual.Cost=[];
empty_individual.S=[];
empty_individual.R=[];
empty_individual.sigma=[];
empty_individual.sigmaK=[];
empty_individual.D=[];
empty_individual.F=[];

% Union=[];
pop=repmat(empty_individual,M,1);

empty_individual.Position=zeros(1,D);
empty_individual.Cost=[1;1;1];
empty_individual.S=[];
empty_individual.R=[];
empty_individual.sigma=[];
empty_individual.sigmaK=[];
empty_individual.D=[];
empty_individual.F=[];
pbest=repmat(empty_individual,M,1);

parfor i=1:M   
    for j=1:D
        if cut(j,1)~=0     
        pop(i).Position(j)=randi([0 cut(j,1)]);
%         if rand()<0.5
%             pop(i).Position(j)=0
%         else
%              pop(i).Position(j)=randi(cut(j,1))
%         end
        end
    end
end

%pbest=pop;

% 
% for i=1:M   %初始化粒子位置 根据潜在切点表x为粒子位置
%     for j=1:D
%         if cut(j,1)==0
%             pop(i).Position(j)=0;
%         else         
%         pop(i).Position(j)=randi([0 cut(j,1)]);
%         end
%     end
% end
flag=0;
%fitnum=1./zeros(nObj,300);
Swarm=[];
for it=1:Iter
   % disp(it);
    Union=[Swarm;pop];
    Swarm=[];
    parfor j=1:numel(Union)
         tmpdata=data;
%          sindex=[]; %select feature index
         nindex=[]; %n-select feature index 
        for k=1:D   %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度  
            if Union(j).Position(:,k)~=0
                for z=1:Union(j).Position(:,k)+1
                   if z==1
                       tmpdata(data(:,k)<=cut(k,z+1),k)=z;
                   elseif z==Union(j).Position(:,k)+1
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
%         [balance_acc(j,1),distance(j,1),pf(j,1)]=fitness(tmpdata,D);
%         Union(j).Cost=[balance_acc(j,1) distance(j,1) pf(j,1)]';
          Union(j).Cost=fitness(tmpdata,D,ssindex,ddindex);
%         z=min(z,pop(j).Cost);
            %如果粒子当前位置支配的话
%         if all(Union(j).Cost<=pbest(j).Cost) && (any(Union(j).Cost<pbest(j).Cost)) 
%         pbest(j)=Union(j);   %更新每个粒子的最佳位置
%        % fitnum(:,j)=Union(j).Cost;
%         end
        
    end
    
    nUn=numel(Union);
    dom=false(nUn,nUn);
   for i=1:nUn
       Union(i).S=0;
   end
    for i=1:nUn
        for j=i+1:nUn
            
            if Dominates(Union(i),Union(j))
                Union(i).S=Union(i).S+1;   %s为支配的个数
                dom(i,j)=true;
                
            elseif Dominates(Union(j),Union(i))
                Union(j).S=Union(j).S+1;
                dom(j,i)=true;
                
            end
            
        end
    end
    
    
    S=[Union.S];
    parfor i=1:nUn
        Union(i).R=sum(S(dom(:,i)));  %所有支配该点的支配个数的和
    end
    
    Z=[Union.Cost]';
    SIGMA=pdist2(Z,Z,'seuclidean');  %SIGMA表示距离，行代表本身与其他距离，列代表其他与本身距离
    SIGMA=sort(SIGMA);
    parfor i=1:nUn
        Union(i).sigma=SIGMA(:,i);
        Union(i).sigmaK=Union(i).sigma(K);
        Union(i).D=1/(Union(i).sigmaK+2);
        Union(i).F=Union(i).R+Union(i).D;
    end
   
 
   
    gbestp=BinaryTournamentSelection(Union,[Union.F]);
    gbest=gbestp.Position;
    nND=sum([Union.R]==0);
    

    if nND<=M
           F=[Union.F];
%            [F, SO]=sort(F);
           [~, SO]=sort(F);
           Union=Union(SO);
           Swarm=Union(1:min(M,nUn));
    else
        SIGMA=SIGMA(:,[Union.R]==0); %找到不被支配的的Q 
        Swarm=Union([Union.R]==0);
        
        k=2;
        while numel(Swarm)>M
            while min(SIGMA(k,:))==max(SIGMA(k,:)) && k<size(SIGMA,1)
                k=k+1;
            end
            
            [~, j]=min(SIGMA(k,:));
            
            Swarm(j)=[];
            SIGMA(:,j)=[];
        end
           
    end
    
    parfor kk=1:M
      if all(Swarm(kk).Cost<=pbest(kk).Cost) && (any(Swarm(kk).Cost<pbest(kk).Cost)) 
        pbest(kk)=Swarm(kk);   %更新每个粒子的最佳位置
       % fitnum(:,j)=Union(j).Cost;
      end
    end
    
    pop=Swarm;
    EP=Union([Union.R]==0);
    %tmpeps=size(EP,1);
    if it==Iter
        EP=Union([Union.R]==0);
        break;
    end
    
    %更新粒子位置
    parfor j=1:M
       mean=(pbest(j).Position+gbest)/2;
       std=abs(pbest(j).Position-gbest);
        for k=1:D   %根据每个粒子转换数据
            smethod=rand();
            if  smethod<=0.9
                if rand()<0.5
                    updatep=round(normrnd(mean(:,k),std(:,k))); %更新粒子
                    if updatep>cut(k,1)||updatep<0 
                        pop(j).Position(k)=0;
                    else
                          pop(j).Position(k)=updatep;
                    end
                else
                      pop(j).Position(k)=pbest(j).Position(k);
                end
            else
             %  if rand()<0.2-(tmpeps/M)    %粒子突变 
                if rand()<0.2-(it/Iter)*0.2
                     pop(j).Position(k)=0;
               end                      
            end           
        end 
    end
    %pbest2=pop;
end
    

if flag==0   
disp('---------------------END--------------------------');
disp('--------------------------------------------------');
end


