function [EP,pbest]=PotentialPso(M,data,cut)
% Solution=PotentialPso(150,traind,cut);
Iter=50;       %����������
D=size(data,2)-1;             %�����ռ�ά��
traind=data;
%pbest=zeros(M,D);  %Ԥ�����ڴ�  
% balance_acc=zeros(M,1);
% distance=zeros(M,1);
% pf=zeros(M,1);
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

% tmpfitnum=zeros(M,1);   %��ʱ���ã�������Ҫ�޸�
% x=zeros(M,D);
% 
% 
% nArchive=15;
nObj=3;
K=round(sqrt(2*M));

%�ո���

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
% for i=1:M   %��ʼ������λ�� ����Ǳ���е��xΪ����λ��
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
%         if all(Union(j).Cost<=pbest(j).Cost) && (any(Union(j).Cost<pbest(j).Cost)) 
%         pbest(j)=Union(j);   %����ÿ�����ӵ����λ��
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
                Union(i).S=Union(i).S+1;   %sΪ֧��ĸ���
                dom(i,j)=true;
                
            elseif Dominates(Union(j),Union(i))
                Union(j).S=Union(j).S+1;
                dom(j,i)=true;
                
            end
            
        end
    end
    
    
    S=[Union.S];
    parfor i=1:nUn
        Union(i).R=sum(S(dom(:,i)));  %����֧��õ��֧������ĺ�
    end
    
    Z=[Union.Cost]';
    SIGMA=pdist2(Z,Z,'seuclidean');  %SIGMA��ʾ���룬�д��������������룬�д��������뱾�����
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
        SIGMA=SIGMA(:,[Union.R]==0); %�ҵ�����֧��ĵ�Q 
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
        pbest(kk)=Swarm(kk);   %����ÿ�����ӵ����λ��
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
             %  if rand()<0.2-(tmpeps/M)    %����ͻ�� 
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


