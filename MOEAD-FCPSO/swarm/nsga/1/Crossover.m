

function y=Crossover(x1,x2,D,gbest,cut)
    s=rand(1,D);
    sel1=find(s>=0.5);
    x3=x2;
    x3(:,sel1)=x1(:,sel1);

 %��������λ��
       mean=(x3+gbest)/2;
       std=abs(x3-gbest);
        for k=1:D   %����ÿ������ת������
            smethod=rand();
            if  smethod<=0.9
                if rand()<0.5
                    updatep=round(normrnd(mean(:,k),std(:,k))); %��������
                    if updatep>cut(k,1)||updatep<0 
                        x3(:,k)=0;
                    else
                        x3(:,k)=updatep;
                    end
                else
                     x3(:,k)=gbest(:,k);
                end
            else
               if rand()<0.2   %����ͻ�� 
                   x3(:,k)=0;
               end                      
            end           
            
        end 
    y=x3;
end