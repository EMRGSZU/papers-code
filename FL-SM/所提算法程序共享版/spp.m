function sp=spp(X,M,k)

D=[];
 for(i=1:1:size(X,1))
     d=0;dd=[];
     for(j=1:1:size(X,1))
         if(j~=i)
             ddd=0;
             for(jj=1:M)
               ddd=ddd+abs(X(j,k+jj)-X(i,k+jj));
             end
            dd=[dd;ddd];
         end
     end
    d=min(dd);
    D=[D;d];
end
avd=mean(D);
sumd=0;
for(i=1:1:size(X,1))
    sumd=sumd+(avd-D(i))^2;
end
sp=sqrt(sumd/(size(X,1)-1));
    