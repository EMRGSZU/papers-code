function eff=evaluation(pop,k,X)

eff=pop;
% for(i=1:1:size(pop,1))
%     for(j=1:1:k)
%         eff(i,j)=pop(i,j);
%     end
% end

for(i=1:size(pop,1))
    eff(i,k+1)=sum(pop(i,1:k));%%Ŀ��һ����ѡ��������ĸ���������Сֵ��
    if eff(i,k+1)==0
        eff(i,k+2)=1000;
        eff(i,k+1)=k+2;
    else
        pos=pop(i,:);
        Y=X(:,1);
        eff(i,k+2)=fitness(X,Y,pos);
%         eff(i,k+2)=kmean_res(pop(i,:),X,k);%%Ŀ������������ķ�������ʣ��á�1-���������ȡ�������Сֵ��
    end
end


function ery=kmean_res(par,X,k)
par;
nx=size(X,1);

sel=[];
for(i=2:k)%%�ڲ���ʱ�е����ݼ���һ��Ϊ����У��������һ��Ϊ���У���Ҫע��
    if par(i)==1
        sel=[sel,i];
    end
end
Y=X(:,sel);
cn=size(sel,2);
class=[];
for(i=1:nx)
    nearest=10e100;
    for(j=2:nx)
        if(i~=j)
            dist=0;
            for(jj=1:cn)
                if(Y(i,jj)~=inf&Y(j,jj)~=inf)
                    dist=dist+abs(Y(i,jj)-Y(j,jj));
                end
            end
            if dist<nearest
                class(i)=X(j,1);%%ѡ�����ڵ���
                nearest=dist;
            end
        end
    end
end
correct=0;
for i=1:nx
    if class(i)==X(i,1)
        correct=correct+1;
    end
end
ery=1-(correct/nx);

            
                

