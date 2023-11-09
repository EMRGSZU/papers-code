

function y=Crossover(x1,x2,D,gbest,cut)
%      x3=zeros(1,D);
%     s=rand(1,D);
%     sel1=find(s>=0.5);
%     sel2=find(s<=0.1);
%     k1=find(x1);
%     s1=szie(k1,2);
%     k2=find(x2);
%     s2=size(k2,2);
%     [k3,a,b]=intersect(k1,k2);
%     k1(:,a')=[];
%     k2(:,b')=[];
%     
%     x3(:,k3)=x2(:,k3);
%     iu
%     x3(:,k3)=x2(:,k3);
%     x3=x2(:,k3);
%     x3(:,sel1)=x1(:,sel1);
%     x3(:,sel2)=0;
%     c=size(find(x1==x2));
%  更新粒子位置
%        mean=(x1+x2)/2;
%        std=abs(x1-x2);
%         for k=1:D   %根据每个粒子转换数据
%             smethod=rand();
%             if  smethod<=0.9
%                 if rand()<0.5
%                     updatep=round(normrnd(mean(:,k),std(:,k))); %更新粒子
%                     if updatep>cut(k,1)||updatep<0 
%                         x3(:,k)=0;
%                     else
%                         x3(:,k)=updatep;
%                     end
%                 else
%                      x3(:,k)=gbest(:,k);
%                 end
%             else
%                if rand()<0.2   %粒子突变 
%                    x3(:,k)=0;
%                end                      
%             end           
%             
%         end 
%     y=x3;

   x3=zeros(1,D);
%  更新粒子位置
       mean=(x1+x2)/2;
       std=abs(x1-x2);
        for k=1:D   %根据每个粒子转换数据
                    updatep=round(normrnd(mean(:,k),std(:,k))); %更新粒子
                    if updatep>cut(k,1)||updatep<0 
                        x3(:,k)=0;
                    else
                        x3(:,k)=updatep;
                    end     
  
        end 
    y=x3;
end