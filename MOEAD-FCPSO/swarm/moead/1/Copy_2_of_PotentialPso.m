function [EP,pbest,EP3,PB3,EP4,PB4,EP5,PB5,EP6,PB6]=PotentialPso(M,data,cut)
% Solution=PotentialPso(150,traind,cut);
Iter=50;       %����������
D=size(data,2)-1;             %�����ռ�ά��

% balance_acc=zeros(M,1);
% distance=zeros(M,1);
% pf=zeros(M,1);
  %��ʱ���ã�������Ҫ�޸�
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

T=ceil(M/15); %�ھӸ���
nObj=3;
%����������
sp=CreateSubProblems(nObj,M,T);
%�ո���
empty_individual.Position=zeros(1,D);
empty_individual.Cost=[];
empty_individual.g=[];
empty_individual.IsDominated=[];
empty_individual.Gbest=[];
h=[0,0,1/D];
pop=repmat(empty_individual,M,1);
pbest=repmat(empty_individual,M,1);
%pbest=[];  %Ԥ�����ڴ�  

% for i=1:M   %��ʼ������λ�� ����Ǳ���е��xΪ����λ��
%     for j=1:D
%         if cut(j,1)==0
%             pop(i).Position(j)=0;
%         else         
%         pop(i).Position(j)=randi([0 cut(j,1)]);
%         end
%     end
% end

parfor i=1:M   %��ʼ������λ�� ����Ǳ���е��xΪ����λ��
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
% for i=1:M   %��ʼ������λ�� ����Ǳ���е��xΪ����λ��
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
        for k=1:D   %����ÿ������ ~ά ��k�� ת������
            %  j�����ӣ��ĵ�kά��  
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
            %������ӵ�ǰλ��֧��Ļ�
        if pop(j).g<=fitg(j,:)
        pbest(j)=pop(j);   %����ÿ�����ӵ����λ��
        fitg(j,:)=pop(j).g;
        end
        
end


 pop=DetermineDomination(pop);
 %��������
 ndpop=pop(~[pop.IsDominated]);
 EP=[ndpop]; %#ok
   % EP=EP(~[EP.IsDominated]);
 [~,deld,~]=unique(cat(1,EP.Position),'rows');
 EP=EP(deld);
tmpeps=size(EP,1);
for it=1:Iter
    %disp(it);
    %�Ƿ�֧�� 
%     for i=1:M
%        [~,kk]=min(cat(1,pop(sp(i).Neighbors).g));
%        pop(i).Gbest=pop(kk).Position;1
%     end
    %���濪ʼ
%     if it==Iter
%         break;
%     end  

%���gbest
    GT=numel(EP);
    EI=randsample(GT,1);
    gbest=EP(EI).Position;

 parfor i=1:M
  
   %1g �ڲ�
   
       mean=(pbest(i).Position+gbest)/2;
       std=abs(pbest(i).Position-gbest);
        for k=1:D   %����ÿ������ת������
            smethod=rand();
            if  smethod<=0.9
                if rand()<0.5
                    updatep=round(normrnd(mean(:,k),std(:,k))); %��������
                    if updatep>cut(k,1)||updatep<0 
                        pop(i).Position(k)=0;
                    else
                          pop(i).Position(k)=updatep;
                    end
                else
                      pop(i).Position(k)=pbest(i).Position(k);
                end
            else
               if rand()<0.15-(it/Iter)*tmpeps/M   %����ͻ�� 
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
        for k=1:D   %����ÿ������ ~ά ��k�� ת������
            %  j�����ӣ��ĵ�kά�� 
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
            %������ӵ�ǰλ��֧��Ļ�
%         if pop(i).g<=fitg(i,:)
%         pbest(i,:)=pop(i).Position;   %����ÿ�����ӵ����λ��
%         fitg(i,:)=pop(i).g;
%         end
 end
 
 parfor i=1:M
    if pop(i).g<=fitg(i,:)
        pbest(i,:)=pop(i);   %����ÿ�����ӵ����λ��
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
        pbest(i,:)=pop(i);   %����ÿ�����ӵ����λ��
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



