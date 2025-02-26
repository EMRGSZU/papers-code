function [EP,pbest]=PotentialPso(M,data,cut)
% Solution=PotentialPso(150,traind,cut);
Iter=50;       %最大迭代次数
D=size(data,2)-1;             %搜索空间维数
traind=data;

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
%pbest=zeros(M,D);  %预分配内存  
% balance_acc=zeros(M,1);
% distance=zeros(M,1);
% pf=zeros(M,1);
%Zr = GenerateReferencePoints(3, 12);
% tmpfitnum=zeros(M,1);   %暂时无用，后续需要修改
% x=zeros(M,D);
% 
% 
% nArchive=15;
nObj=3;
Zr = GenerateReferencePoints(3,12); %权重初始化也叫做参考点
%Zr(Zr==0)=1e-6; %0变为1e-6

%空个体

empty_individual.Position=zeros(1,D);
empty_individual.Cost=[];
empty_individual.Rank=[];
empty_individual.DominationSet=[];
empty_individual.DominatedCount=[];
empty_individual.NormalizedCost = [];
empty_individual.AssociatedRef = [];
empty_individual.DistanceToAssociatedRef = [];

%empty_individual.CrowdingDistance=[];

params.nPop = M;
params.Zr = Zr;
params.nZr = size(Zr,2);
params.zmin = [];
params.zmax = [];
params.smin = [];


% Union=[];
pop=repmat(empty_individual,M,1);

empty_individual.Position=zeros(1,D);
empty_individual.Cost=[1;1;1];
empty_individual.Rank=[];
empty_individual.DominationSet=[];
empty_individual.DominatedCount=[];
empty_individual.NormalizedCost = [];
empty_individual.AssociatedRef = [];
empty_individual.DistanceToAssociatedRef = [];

pbest=repmat(empty_individual,M,1);
parfor i=1:M   %初始化粒子位置 根据潜在切点表x为粒子位置
    for j=1:D
        if cut(j,1)~=0     
            pop(i).Position(j)=randi([0 cut(j,1)]);
        end
    end
end

%fitnum=1./zeros(nObj,M);
Swarm=[];
for it=1:Iter
    %disp(it);
   % Union=[Swarm;pop];
   Union=pop;
   % Swarm=[];
   %for j=1:M
   parfor j=1:M
        
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
        if all(Union(j).Cost<=pbest(j).Cost) && (any(Union(j).Cost<pbest(j).Cost))  
        pbest(j)=Union(j);   %更新每个粒子的最佳位置
        end    
        
        
        
        
        
    end
    
    Union=[Swarm;Union];
    Swarm=[];
    [Union, F]=NonDominatedSorting(Union);
    [Union, params] = NormalizePopulation(Union, params);
    [Union, d, rho] = AssociateToReferencePoint(Union, params);
    %Union=CalcCrowdingDistance(Union,F);
    %[Union, F]=SortPopulation(Union);   
    g=1;
    while numel(Swarm)<M
        if (numel(Swarm)+numel(F{g}))<=M
            Swarm=[Swarm;Union([F{g}])];
            g=g+1;
        elseif (numel(Swarm)+numel(F{g}))>M
           % u=M-numel(Swarm);
            LastFront = F{g};
          %  Swarm=[Swarm;Union(F{g}(:,1:u))];  不在通过拥挤度选择粒子，而是通过参考点来进行
         %   [Union, params] = NormalizePopulation(Union, params);
            %对粒子进行标准化以及进行选择
         %   [Union, d, rho] = AssociateToReferencePoint(Union, params);
          while true
                [~, j] = min(rho);
                AssocitedFromLastFront = [];
                for i = LastFront
                    if Union(i).AssociatedRef == j
                         AssocitedFromLastFront = [AssocitedFromLastFront i]; %#ok
                    end
                end
                if isempty(AssocitedFromLastFront)
                    rho(j) = inf;
                    continue;
                end
            if rho(j) == 0
                ddj = d(AssocitedFromLastFront, j);
                [~, new_member_ind] = min(ddj);
            else
                new_member_ind = randi(numel(AssocitedFromLastFront));
            end
            MemberToAdd = AssocitedFromLastFront(new_member_ind);
            LastFront(LastFront == MemberToAdd) = [];
            Swarm = [Swarm; Union(MemberToAdd)]; %#ok
            rho(j) = rho(j) + 1;
            if numel(Swarm) >= M
                break;
            end     
          end         
        end
    end
    
    [Swarm, F] = NonDominatedSorting(Swarm);
    gbestp=BinaryTournamentSelection(Swarm,F);
    gbest=gbestp.Position;
    
    
    
    pop=Swarm;
    if it==Iter
        EP=Swarm([F{1}]);
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
              % if rand()<0.2   %粒子突变 
               if rand()<0.2-(it/Iter)*0.2
                     pop(j).Position(k)=0;
               end                      
            end           
        end 
    end
%     Swarm=pop;

end
    

disp('---------------------END--------------------------');
disp('--------------------------------------------------');



