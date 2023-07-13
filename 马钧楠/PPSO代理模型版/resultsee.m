x=Solution;
size(find(x))


tta=traind;
nindex=[];
        for k=1:size(tta,2)-1   %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度  
            if x(:,k)~=0
                for z=1:x(:,k)+1
                   if z==1
                       tta(find(traind(:,k)<=cut(k,z+1)),k)=z;
                   elseif z==x(:,k)+1
                       tta(find(traind(:,k)>cut(k,z)),k)=z;
                   else
                       tta(find(traind(:,k)>cut(k,z) & traind(:,k)<=cut(k,z+1)),k)=z;
                   end
                end         
            else
                nindex=[nindex k];
            end          
        end
        tta(:,nindex)=[];
        

tte=testd;
nindex=[];
      for k=1:size(tte,2)-1   %根据每个粒子 ~维 度k· 转换数据
            %  j个粒子，的第k维度  
            if x(:,k)~=0
                for z=1:x(:,k)+1
                   if z==1
                       tte(find(testd(:,k)<=cut(k,z+1)),k)=z;
                   elseif z==x(:,k)+1
                       tte(find(testd(:,k)>cut(k,z)),k)=z;
                   else
                       tte(find(testd(:,k)>cut(k,z) & testd(:,k)<=cut(k,z+1)),k)=z;
                   end
                end         
            else
                nindex=[nindex k];
            end          
        end
        tte(:,nindex)=[];
        
        
        
 Mdl=fitcknn(tta(:,1:end-1),tta(:,end));
 pclass=predict(Mdl,tte(:,1:end-1));
result=(pclass==tte(:,end))