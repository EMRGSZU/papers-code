function EP=Pso(M,data)
% Solution=PotentialPso(150,traind,cut);
Iter=70;       %��������
D=size(data,2)-1;             %�����ռ�ά��
%traind=data;
cdata=data;
[~,weights] = relieff(data(:,1:end-1),data(:,end),1);
weight=mapminmax(weights,0,1);
fazhi=0.52;
yuxianindex=find(weight<=fazhi);
EP=[];
T=ceil(M/30); %�ھӸ���
nObj=3;
%����������
sp=CreateSubProblems(nObj,M,T);
%�ո���
empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.err=[];
empty_individual.dis=[];
empty_individual.pf=[];
empty_individual.g=[];
empty_individual.IsDominated=[];
empty_individual.Gbest=[];
h=[0,0,1/D];
pop=repmat(empty_individual,M,1);

% for i=1:M   %��ʼ������λ�� ���Ǳ���е��xΪ����λ��
%     for j=1:D
%         if cut(j,1)==0
%             pop(i).Position(j)=0;
%         else         
%         pop(i).Position(j)=randi([0 cut(j,1)]);
%         end
%     end
% end
parfor i=1:M
    tmprand=rand(1,D);
    pop(i).Position= tmprand > 0.6; 
    if rand()>0.35
    pop(i).Position(:,yuxianindex)=0;
    end
end

%fitg=1./zeros(M,1);

parfor j=1:M
    [pop(j).Cost,pop(j).err,pop(j).dis]=fitness(cdata,pop(j).Position,D);
    pop(j).g=DecomposedCost(pop(j),h,sp(j).lambda);
end

fitg=[pop.g]';

pbest=pop;
pop=DetermineDomination(pop);
    %��������
ndpop=pop(~[pop.IsDominated]);
EP=[ndpop];
[~,deld,~]=unique(cat(1,EP.Position),'rows');
lastEP=EP(deld);
gstay=0;
GT=numel(lastEP);
EI=randsample(GT,1);
gbest=lastEP(EI);

for it=1:Iter
    disp(it);
    if gstay>=3 
        for j=1:M
        %parfor j=1:M
             mean=(pbest(j).Position+zeros(1,D))/2;
             std=abs(pbest(j).Position-zeros(1,D));
             test=rand(1,D);
             otheri=find(test>0.5);
             updated=normrnd(mean,std);
             updated((updated>0.5))=1;
             updated((updated<=0.5))=0;   
             pop(j).Position=updated;
             pop(j).Position(:,otheri)=pbest(j).Position(:,otheri);
            % [~,pop(j).Cost(1),pop(j).Cost(2)]=fitness(cdata,pop(j).Position); %��ȷ�� 
             [pop(j).Cost,pop(j).err,pop(j).dis]=fitness(cdata,pop(j).Position,D);
               pop(j).g=DecomposedCost(pop(j),h,sp(j).lambda);
            if all(pop(j).Cost<=pbest(j).Cost) && (any(pop(j).Cost<pbest(j).Cost)) 
            % if pop(j).Cost<pbest(j).Cost
                 pbest(j)=pop(j);
            end
          
        end     
    else
      % for j=1:M
    parfor j=1:M
             mean=(pbest(j).Position+gbest.Position)/2;
             std=abs(pbest(j).Position-gbest.Position);
             test=rand(1,D);
             otheri=find(test>0.5);
             updated=normrnd(mean,std);
             updated((updated>0.5))=1;
             updated((updated<=0.5))=0;   
             pop(j).Position=updated;
             pop(j).Position(:,otheri)=pbest(j).Position(:,otheri);
             [pop(j).Cost,pop(j).err,pop(j).dis]=fitness(cdata,pop(j).Position,D);
             pop(j).g=DecomposedCost(pop(j),h,sp(j).lambda);
            if all(pop(j).Cost<=pbest(j).Cost) && (any(pop(j).Cost<pbest(j).Cost)) 
            % if pop(j).Cost<pbest(j).Cost
                 pbest(j)=pop(j);
            end
        end
    end
  
     
 for i=1:M
      for j=sp(i).Neighbors
            %pop(i).g=DecomposedCost(pop(i),h,sp(j).lambda);
           tmpg=DecomposedCost(pbest(i),h,sp(j).lambda);
            if tmpg<pbest(j).g
                pbest(j)=pbest(i);
                pbest(j).g=tmpg;
               % pop(j).Position=pop(i).Position;
            end
      end 
 end
    %�Ƿ�֧��
    pop=DetermineDomination(pop);
    %��������
    ndpop=pop(~[pop.IsDominated]);
    EP=[EP;ndpop]; %#ok
    EP=DetermineDomination(EP);
    EP=EP(~[EP.IsDominated]);
    [~,deld,~]=unique(cat(1,EP.Position),'rows');
    EP=EP(deld);
    b=EPstay(EP,lastEP);
    if b==0
        gstay=gstay+1;
    else
        gstay=0;
    end
    lastEP=EP;
    GT=numel(lastEP);
    EI=randsample(GT,1);
    gbest=lastEP(EI);
end
    
  
disp('---------------------END--------------------------');
disp('--------------------------------------------------');


