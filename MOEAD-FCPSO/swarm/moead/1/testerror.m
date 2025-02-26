function [length,error] = testerror(x,traind,testd,cut)
% x=x(i,:);
length=size(find(x),2);


tta=traind;  %根据切点离散化训练集；
nindex=[];
        for k=1:size(tta,2)-1   %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度  
            if x(:,k)~=0
                for z=1:x(:,k)+1
                   if z==1
                       tta(traind(:,k)<=cut(k,z+1),k)=z;
                   elseif z==x(:,k)+1
                       tta(traind(:,k)>cut(k,z),k)=z;
                   else
                       tta(traind(:,k)>cut(k,z) & traind(:,k)<=cut(k,z+1),k)=z;
                   end
                end         
            else
                nindex=[nindex k];
            end          
        end
        tta(:,nindex)=[];
        

tte=testd; %根据切点离散化测试集合
nindex=[];
      for k=1:size(tte,2)-1   %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度  
            if x(:,k)~=0
                for z=1:x(:,k)+1
                   if z==1
                       tte(testd(:,k)<=cut(k,z+1),k)=z;
                   elseif z==x(:,k)+1
                       tte(testd(:,k)>cut(k,z),k)=z;
                   else
                       tte(testd(:,k)>cut(k,z) & testd(:,k)<=cut(k,z+1),k)=z;
                   end
                end         
            else
                nindex=[nindex k];
           end          
       end
        tte(:,nindex)=[];
        
  cp = classperf(tte(:,end));
        ctype=size(unique(tte(:,end)),1);
 Mdl=fitcknn(tta(:,1:end-1),tta(:,end));
 pclass=predict(Mdl,tte(:,1:end-1));
 classperf(cp,pclass);
 a=cp.ErrorDistributionByClass;
b=cp.SampleDistributionByClass;
result=(b-a)./b;
result=sum(result);
error=result/ctype;
%result=(pclass==tte(:,end));
%error=size(find(result),1)/size(result,1); %查看最后的分类正确率error应为accuarcy


end

