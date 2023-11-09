function eff=evaluation(pop,k,X)

eff=pop;
% for(i=1:1:size(pop,1))
%     for(j=1:1:k)
%         eff(i,j)=pop(i,j);
%     end
% end

for(i=1:size(pop,1))
    eff(i,k+1)=sum(pop(i,1:k));%%目标一：被选择的特征的个数（求最小值）
    if eff(i,k+1)==0
        eff(i,k+2)=1000;
        eff(i,k+1)=k+2;
    else
        pos=pop(i,:);
        Y=X(:,1);
        eff(i,k+2)=fitness(X,Y,pos);
%         eff(i,k+2)=kmean_res(pop(i,:),X,k);%%目标二：分类器的分类错误率（用【1-分类器精度】，求最小值）
    end
end


function ery=kmean_res(par,X,k)
par;
nx=size(X,1);

sel=[];
for(i=2:k)%%在测试时有的数据集第一列为标号列，或者最后一列为类列，需要注意
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
                class(i)=X(j,1);%%选择所在的类
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

            
                

