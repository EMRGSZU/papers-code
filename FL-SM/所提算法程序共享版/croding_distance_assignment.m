function FF_dis=croding_distance_assignment(par,M,k)
npar=size(par,1);

for i=1:npar
    par_dis(i)=0;
    for j=1:M
        par=sort(par,j);
        for ii=2:(npar-1)
            par_dis(ii)=par_dis(i)+(par(ii+1,k+j)-par(ii-1,k+j));
        end
    end
    par_dis(1)=inf;
    par_dis(npar)=inf;
end
FF_dis=par_dis;