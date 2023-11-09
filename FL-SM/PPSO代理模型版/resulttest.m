
x=x(1,:);
size(find(x))
tta=traind;
sindex=[];
nindex=[];
 for k=1:size(tta,2)-1  %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度  
            if x(:,k)>0 
                index1=find(tta(:,k)>=cut(k,x(:,k)+1));
                index2=find(tta(:,k)<cut(k,x(:,k)+1));
                tta(index1,k)=1;
                tta(index2,k)=0;
                sindex=[sindex k];            
            else
                nindex=[nindex k];
            end
        end
        tta(:,nindex)=[];
%         
%         
tte=testd;
sindex=[];
nindex=[];
 for k=1:size(tte,2)-1  %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度  
            if x(:,k)>0 
                index1=find(tte(:,k)>=cut(k,x(:,k)+1));
                index2=find(tte(:,k)<cut(k,x(:,k)+1));
                tte(index1,k)=1;
                tte(index2,k)=0;
                sindex=[sindex k];            
            else
                nindex=[nindex k];
            end
        end
        tte(:,nindex)=[];
 Mdl=fitcknn(tta(:,1:end-1),tta(:,end));
 pclass=predict(Mdl,tte(:,1:end-1));
result=(pclass==tte(:,end))