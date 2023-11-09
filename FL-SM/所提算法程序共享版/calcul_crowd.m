
function crowd_value=calcul_crowd2(new_AC,M,k)


s2=size(new_AC,1);
if(s2>=2)
    for(i=1:M)
        LIM_f(i,2)=max(new_AC(:,k+i));
        LIM_f(i,1)=min(new_AC(:,k+i));
    end
    DD=[];
    crowd_value=[];
    for(i=1:M)
        [val,ind]=sort(new_AC(:,k+i));
        for(j=1:s2)
            if(j==1)
                DD(ind(j),i)=4*(new_AC(ind(j+1),k+i)-new_AC(ind(j),k+i))/((LIM_f(i,2)-LIM_f(i,1)));
            elseif(j==s2)
                DD(ind(j),i)=4*(new_AC(ind(j),k+i)-new_AC(ind(j-1),k+i))/((LIM_f(i,2)-LIM_f(i,1)));
            else
                DD(ind(j),i)=(new_AC(ind(j+1),k+i)-new_AC(ind(j-1),k+i))/((LIM_f(i,2)-LIM_f(i,1)));
            end
        end
    end
    for(jj=1:s2)
        crowd_value(jj)=sum(DD(jj,:));
    end
else
    crowd_value(1)=1;
end










