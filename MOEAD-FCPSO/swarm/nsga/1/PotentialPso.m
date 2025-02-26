function [EP,pbest]=PotentialPso(M,data,cut)
% Solution=PotentialPso(150,traind,cut);
Iter=50;       %����������
D=size(data,2)-1;             %�����ռ�ά��
traind=data;

datalabel=data(:,end);
dlength=size(data,1);
ssindex=cell(1,dlength);

ddindex=cell(1,dlength);

for i=1:dlength
    testlabel=datalabel(i);
    ddindex{i}=find(testlabel~=datalabel);  %��ͬ����
    temps=find(testlabel==datalabel); %��ͬ����
    ssindex{i}=temps(temps~=i); %��ͬ����,�ų��Լ�  
end 
%pbest=zeros(M,D);  %Ԥ�����ڴ�  
% balance_acc=zeros(M,1);
% distance=zeros(M,1);
% pf=zeros(M,1);
%Zr = GenerateReferencePoints(3, 12);
% tmpfitnum=zeros(M,1);   %��ʱ���ã�������Ҫ�޸�
% x=zeros(M,D);
% 
% 
% nArchive=15;
nObj=3;
Zr = GenerateReferencePoints(3,12); %Ȩ�س�ʼ��Ҳ�����ο���
%Zr(Zr==0)=1e-6; %0��Ϊ1e-6

%�ո���

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
parfor i=1:M   %��ʼ������λ�� ����Ǳ���е��xΪ����λ��
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
        for k=1:D   %����ÿ������ ~ά ��k�� ת������
            %  j�����ӣ��ĵ�kά��  
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
            %������ӵ�ǰλ��֧��Ļ�
        if all(Union(j).Cost<=pbest(j).Cost) && (any(Union(j).Cost<pbest(j).Cost))  
        pbest(j)=Union(j);   %����ÿ�����ӵ����λ��
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
          %  Swarm=[Swarm;Union(F{g}(:,1:u))];  ����ͨ��ӵ����ѡ�����ӣ�����ͨ���ο���������
         %   [Union, params] = NormalizePopulation(Union, params);
            %�����ӽ��б�׼���Լ�����ѡ��
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
    
    %��������λ��
    parfor j=1:M
       mean=(pbest(j).Position+gbest)/2;
       std=abs(pbest(j).Position-gbest);
        for k=1:D   %����ÿ������ת������
            smethod=rand();
            if  smethod<=0.9
                if rand()<0.5
                    updatep=round(normrnd(mean(:,k),std(:,k))); %��������
                    if updatep>cut(k,1)||updatep<0 
                        pop(j).Position(k)=0;
                    else
                          pop(j).Position(k)=updatep;
                    end
                else
                      pop(j).Position(k)=pbest(j).Position(k);
                end
            else
              % if rand()<0.2   %����ͻ�� 
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



