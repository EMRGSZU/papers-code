function Solution=PotentialPso(M,data,cut)
% Solution=PotentialPso(150,traind,cut);
Iter=70;       %����������
D=size(data,2)-1;             %�����ռ�ά��
traind=data;
Solution=0;
pbest=zeros(M,D);  %Ԥ�����ڴ�  
balance_acc=zeros(M,1);
distance=zeros(M,1);
pf=zeros(M,1);

tmpfitnum=zeros(M,1);   %��ʱ���ã�������Ҫ�޸�
x=zeros(M,D);


T=ceil(M/15); %�ھӸ���
nArchive=15;
nObj=3;


%����������
sp=CreateSubProblems(nObj,M,T);
%�ո���
empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.g=[];
empty_individual.IsDominated=[];

h=zeros(1,nObj);
pop=repmat(empty_individual,M,1);

for i=1:M   %��ʼ������λ�� ����Ǳ���е��xΪ����λ��
    for j=1:D
        if cut(j,1)==0
            x(i,j)=0;
        else         
        x(i,j)=randi([0 cut(j,1)]);
        end
    end
    pop(i).Position=x(i,:);
end


flag=0;

fitnum=1./zeros(M,nObj);
gbestvalue=zeros(Iter,1);
gbestfit=0;



for it=1:Iter
    disp(it);
   
    for j=1:M
         tmpdata=data;
%          sindex=[]; %select feature index
         nindex=[]; %n-select feature index 
        for k=1:D   %����ÿ������ ~ά ��k�� ת������
            %  j�����ӣ��ĵ�kά��  
            if pop(j).Position(:,k)~=0
                for z=1:pop(j).Position(:,k)+1
                   if z==1
                       tmpdata(find(data(:,k)<=cut(k,z+1)),k)=z;
                   elseif z==pop(j).Position(:,k)+1
                       tmpdata(find(data(:,k)>cut(k,z)),k)=z;
                   else
                       tmpdata(find(data(:,k)>cut(k,z) & data(:,k)<=cut(k,z+1)),k)=z;
                   end
                end
       
            else
                nindex=[nindex k];
            end
        end
        tmpdata(:,nindex)=[];
        [balance_acc(j,1),distance(j,1),pf(j,1),tmpfitnum(j,1)]=fitness(tmpdata,D);
        pop(j).Cost=[balance_acc(j,1) distance(j,1) pf(j,1)];
%         z=min(z,pop(j).Cost);
            %������ӵ�ǰλ��֧��Ļ�
        if all(pop(j).Cost<=fitnum(j,:)) && (any(pop(j).Cost<fitnum(j,:))) 
        pbest(j,:)=x(j,:);   %����ÿ�����ӵ����λ��
        fitnum(j,:)=pop(j).Cost;
        end
        
    end
    
    %��ʼ�ֽ�֧�俪ʼ�ֽ�
    for i=1:M
        pop(i).g=DecomposedCost(pop(i),h,sp(i).lambda);
    end
    %�Ƿ�֧��
    pop=DetermineDomination(pop);
    %��������
    EP=pop(~[pop.IsDominated]);
    %���濪ʼ
    
   
    
    
    GT=numel(EP);
    EI=randsample(GT,1);
    gbest=EP(EI).Position;
    
    %����������Ⱥλ��
%     [gvalue,index]=max(fitnum);
%     if gbestfit<gvalue
%         gbest=x(index,:);  
%         gbestfit=gvalue;
%     end
%     gbestvalue(it,1)=gbestfit;
   

 for i=1:M
     sk=randsample(T,2);
     
    j1=sp(i).Neighbors(sk(1));
    p1=pop(j1);
    
    j2=sp(i).Neighbors(sk(2));
    p2=pop(j2);
    
    y=empty_individual;
    y.Position=Crossover(p1.Position,p2.Position,D);     
    
      tmpdata=data;
%          sindex=[]; %select feature index
        nindex=[]; %n-select feature index 
        for k=1:D   %����ÿ������ ~ά ��k�� ת������
            %  j�����ӣ��ĵ�kά�� 
            if y.Position(k)~=0
                for z=1:y.Position(k)+1
                   if z==1
                       tmpdata(find(data(:,k)<=cut(k,z+1)),k)=z;
                   elseif z==y.Position(k)+1
                       tmpdata(find(data(:,k)>cut(k,z)),k)=z;
                   else
                       tmpdata(find(data(:,k)>cut(k,z) & data(:,k)<=cut(k,z+1)),k)=z;
        
                   end
                end
       
            else
                nindex=[nindex k];
            end
        end
        
        
        
        tmpdata(:,nindex)=[];
        [crossban,crosdistance,crosspf,crosstmpfitnum]=fitness(tmpdata,D);
        
     
        y.Cost=[crossban crosdistance crosspf];
        
        for j=sp(i).Neighbors
            y.g=DecomposedCost(y,h,sp(j).lambda);
            if y.g<=pop(j).g
                pop(j)=y;
            end
        end    
 end
   pop=DetermineDomination(pop);
    
    ndpop=pop(~[pop.IsDominated]);
    
    EP=[EP;ndpop]; %#ok
    
    EP=DetermineDomination(EP);
    EP=EP(~[EP.IsDominated]);
    
    %
    
    if numel(EP)>nArchive
        Extra=numel(EP)-nArchive;
        ToBeDeleted=randsample(numel(EP),Extra);
        EP(ToBeDeleted)=[];
    end
%     ���ź�ֹͣ׼��
%     if  it>=16&&gbestvalue(it,1)>=1.01-*gbestvalue(it-15,1)&&gbestvalue(it-1,1)==gbestvalue(it-15,1)
%         M=M+50;
%         flag=1;
%         break;
%     elseif it>=16&&gbestvalue(it-1,1)==gbestvalue(it-15,1)
%         break;
%     end
    
    %��������λ��
%     for j=1:M
%        mean=(pbest(j,:)+gbest)/2;
%        std=abs(pbest(j,:)-gbest);
%         for k=1:D   %����ÿ������ת������
%             smethod=rand();
%             if  smethod<=0.9
%                 if rand()<0.5
%                     updatep=round(normrnd(mean(:,k),std(:,k))); %��������
%                     if updatep>cut(k,1)||updatep<0 
%                         x(j,k)=0;
%                     else
%                         x(j,k)=updatep;
%                     end
%                 else
%                      x(j,k)=pbest(j,k);
%                 end
%             else
%                if rand()<0.2   %����ͻ�� 
%                    x(j,k)=0;
%                end                      
%             end           
%             
%         end 
%         pop(j).Position=x(j,:);
%     end

end
    
if flag==1
     Solution=PotentialPso(M,traind,cut);
end

if flag==0   
disp('---------------------END--------------------------');
Solution=gbest;
disp('--------------------------------------------------');
end


