function [ output_args ] = updateparticle(pop,pbest,gbest)
%UNTITLED2 �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
        mean=(pbest+gbest)/2;
       std=abs(pbest-gbest);
        parfor k=1:D   %����ÿ������ת������
            smethod=rand();
            if  smethod<=0.9
                if rand()<0.5
                    updatep=round(normrnd(mean(:,k),std(:,k))); %��������
                    if updatep>cut(k,1)||updatep<0 
                        pop.Position(k)=0;
                    else
                          pop.Position(k)=updatep;
                    end
                else
                      pop.Position(k)=pbest(:,k);
                end
            else
               if rand()<0.2   %����ͻ�� 
                     pop.Position(k)=0;
               end                      
            end           
        end   

end

