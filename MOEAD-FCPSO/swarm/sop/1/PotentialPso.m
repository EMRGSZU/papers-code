function [solution2,Solution]=PotentialPso(M,data,cut)
% Solution=PotentialPso(150,traind,cut);
Iter=60;       %��������
D=size(data,2)-1;             %�����ռ�ά��
traind=data;
Solution=0;
%pbest=zeros(M,D);  %Ԥ�����ڴ�  
%balance_acc=zeros(M,1);
%distance=zeros(M,1);
%pf=zeros(M,1);
%tmpfitnum=Inf(M,1);
%x=zeros(D,1); 

empty_individual.Position=zeros(1,D);
empty_individual.Cost=[];
empty_individual.err=[];
empty_individual.dis=[];
empty_individual.pf=[];
pop=repmat(empty_individual,M,1);
empty_individual.Cost=Inf;
pbest=repmat(empty_individual,M,1);
gbest=repmat(empty_individual,1,1);



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



parfor i=1:M   %��ʼ������λ�� ���Ǳ���е��xΪ����λ��
    for j=1:D 
         if cut(j,1)~=0          
              pop(i).Position(j)=randi([0 cut(j,1)]);
         end
    end
end


flag=0;
%fitnum=Inf(M,1);
gbestvalue=Inf(Iter,1);
%gbestfit=Inf;
for it=1:Iter
    disp(it);
    %for j=1:M
    parfor j=1:M
         tmpdata=data;
%          sindex=[]; %select feature index
         nindex=[]; %n-select feature index 
        for k=1:D   %���ÿ������? ~ά ��k�� ת�����?
            %  j�����ӣ��ĵ�kά��  
            if pop(j).Position(k)~=0
                for z=1:pop(j).Position(k)+1
                   if z==1
                       tmpdata(data(:,k)<=cut(k,z+1),k)=z;
                   elseif z==pop(j).Position(k)+1
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
        [pop(j).Cost,pop(j).err,pop(j).dis,pop(j).pf]=fitness(tmpdata,D,ssindex,ddindex);
        if pop(j).Cost<pbest(j).Cost
        pbest(j)=pop(j);   %����ÿ�����ӵ����λ��?
        end
    end
    [gvalue,index]=min([pbest(:).Cost]);
    if gvalue<gbest.Cost
        gbest=pbest(index);
    end
    gbestvalue(it,1)=gbest.Cost;
   
   
    %���ź�ֹͣ׼��
%     if  it>=16&&gbestvalue(it,1)>=1.01*gbestvalue(it-15,1)&&gbestvalue(it-1,1)==gbestvalue(it-15,1)
%         M=M+50;
%         flag=1;
%         break;
%     elseif it>=16&&gbestvalue(it-1,1)==gbestvalue(it-15,1)
%         break;
%     end
    
    %��������λ��
    parfor j=1:M
       mean=(pbest(j).Position+gbest.Position)/2;
       std=abs(pbest(j).Position-gbest.Position);
        for k=1:D   %���ÿ������ת�����
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
               %if rand()<0.2   %����ͻ�� 
               if rand()<0.2-(it/Iter)*0.2
                   pop(j).Position(k)=0;
               end                      
            end           
            
        end 
    end

end
    
if flag==1
     [solution2,Solution]=PotentialPso(M,traind,cut);
end

if flag==0   
disp('---------------------END--------------------------');
Solution=gbest;
solution2=pbest;
disp('--------------------------------------------------');
end


