function [ output_args ] = updateparticle(pop,pbest,gbest)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
        mean=(pbest+gbest)/2;
       std=abs(pbest-gbest);
        parfor k=1:D   %根据每个粒子转换数据
            smethod=rand();
            if  smethod<=0.9
                if rand()<0.5
                    updatep=round(normrnd(mean(:,k),std(:,k))); %更新粒子
                    if updatep>cut(k,1)||updatep<0 
                        pop.Position(k)=0;
                    else
                          pop.Position(k)=updatep;
                    end
                else
                      pop.Position(k)=pbest(:,k);
                end
            else
               if rand()<0.2   %粒子突变 
                     pop.Position(k)=0;
               end                      
            end           
        end   

end

