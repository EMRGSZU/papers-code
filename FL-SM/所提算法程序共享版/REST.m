function rest=REST(par,good,k)
np=size(par,1);
nb=size(good,1);
for j=1:nb
    for i=1:np
        if par(i,k+2)==good(j,k+2)& par(i,k+1)==good(j,k+1)
            par(i,1:k+2)=0;
           break;
        end
    end
end
for i=1:np
    c=sum(par,2);
    i = find(c==0); %find the row indexes which are all zeros
    par(i,:)=[];
end
rest=par;
             