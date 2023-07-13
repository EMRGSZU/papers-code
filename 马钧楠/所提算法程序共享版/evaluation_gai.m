function eff=evaluation_gai(pop,k,X,flag,featurescore)

mscore=0;
mrate=0;
if flag==1
    for i=1:size(pop,1)
        pos=pop(i,1:k);
        mrate=mrate+sum(pos);
        mscore=mscore+sum(featurescore.*pos);
    end
    mrate=mrate/size(pop,1)
    mscore=mscore/size(pop,1)
end
eff=pop;

if flag==0 
    for(i=1:size(pop,1))
        eff(i,k+1)=sum(pop(i,1:k));%%目标一：被选择的特征的个数（求最小值）
        if eff(i,k+1)==0
            eff(i,k+2)=1000;
            eff(i,k+1)=k+2;
        else
            pos=pop(i,:);
            Y=X(:,1);
            eff(i,k+2)=fitness(X,Y,pos);
        end
    end
else
    for(i=1:size(pop,1))
        eff(i,k+1)=sum(pop(i,1:k));%%目标一：被选择的特征的个数（求最小值）
        if eff(i,k+1)==0 || surrogate(featurescore,pop(i,1:k),mscore,mrate,eff(i,k+1))==0
            eff(i,k+2)=1000;
            eff(i,k+1)=k+2;
        else
            pos=pop(i,:);
            Y=X(:,1);
            eff(i,k+2)=fitness(X,Y,pos);
        end
    end
end





            
                

