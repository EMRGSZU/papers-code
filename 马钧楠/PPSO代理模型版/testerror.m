function [length,acc] = testerror(x,traind,testd,cut)
% x=x(i,:);
% length=size(find(x));
length=sum(x);



tta=traind;  %�����е���ɢ��ѵ������
ttal=tta(:,end); %ѵ��������ǩ
 sindex=[]; %select feature index
         nindex=[];
        for k=1:size(tta,2)-1   %����ÿ������ ~ά ��k�� ת������
            %  j�����ӣ��ĵ�kά��  
            if x(:,k)>0
                index1=tta(:,k)>=cut(k,x(1,k)+1);
                index2=tta(:,k)<cut(k,x(1,k)+1);
                tta(index1,k)=1;
                tta(index2,k)=0;
                sindex=[sindex k]; 
                      
            else
                nindex=[nindex k];
            end          
        end
        tta(:,nindex)=[];
        tta=[tta ttal];
        

tte=testd; %�����е���ɢ�����Լ���
ttel=tte(:,end); %ѵ��������ǩ
 sindex=[]; %select feature index
         nindex=[];
      for k=1:size(tte,2)-1   %����ÿ������ ~ά ��k�� ת������
            %  j�����ӣ��ĵ�kά��  
            if x(:,k)>0
                 index1=tte(:,k)>=cut(k,x(1,k)+1);
                index2=tte(:,k)<cut(k,x(1,k)+1);
                tte(index1,k)=1;
                tte(index2,k)=0;
                sindex=[sindex k]; 
                   
            else
                nindex=[nindex k];
            end          
        end
        tte(:,nindex)=[];
        tte=[tte ttel];

  cp = classperf(tte(:,end));
  ctype=size(unique(tte(:,end)),1);
 Mdl=fitcknn(tta(:,1:end-1),tta(:,end));
 pclass=predict(Mdl,tte(:,1:end-1));
 classperf(cp,pclass);
 a=cp.ErrorDistributionByClass;
b=cp.SampleDistributionByClass;
result=(b-a)./b;
result=sum(result);
acc=result/ctype;       
 
        
        
 %Mdl=fitcknn(tta(:,1:end-1),tta(:,end));
 %pclass=predict(Mdl,tte(:,1:end-1));
%result=(pclass==tte(:,end));
%error=size(find(result),1)/size(result,1); %�鿴���ķ�����ȷ��errorӦΪaccuarcy


end

