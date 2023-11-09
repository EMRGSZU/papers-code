function [id] = SOGFS_sp2(X,distX1,idx,k,gamma,c,num,dim)
A = zeros(num);     %权重矩阵A
rr = zeros(num,1);
for i = 1:num
    di = distX1(i,2:k+2);   %取每个点最近的K个点
    rr(i) = 0.5*(k*di(k+1)-sum(di(1:k)));
    id = idx(i,2:k+2);      %取每个点最近的K个点的下标
    A(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;
r = mean(rr);
lambda = r;

A0 = (A+A')/2;
D0 = diag(sum(A0));
L0 = D0 - A0;
[F, ~, evs]=eig1(L0, c, 0);

[W] = InterationW_sp(L0,X',gamma,dim);
NITER = 50;
for iter = 1:NITER
    distf = L2_distance_1(F',F');
    distx = L2_distance_1(W'*X,W'*X);
    if iter>5
        [~, idx] = sort(distx,2);
    end;
    A = zeros(num);
    for i=1:num
        idxa0 = idx(i,2:k+1);
        dfi = distf(i,idxa0);
        dxi = distx(i,idxa0);
        ad = -(dxi+lambda*dfi)/(2*r);
        A(i,idxa0) = EProjSimplex_new(ad);
    end;
    
    A = (A+A')/2;
    D = diag(sum(A));
    L = D-A;
    
    [W] = InterationW_sp(L,X',gamma,dim);
    F_old = F;
    [F, ~, ev]=eig1(L, c, 0);
    evs(:,iter+1) = ev;
    
    fn1 = sum(ev(1:c));
    fn2 = sum(ev(1:c+1));
    if fn1 > 0.000000001
        lambda = 2*lambda;
    elseif fn2 < 0.00000000001
        lambda = lambda/2;  F = F_old;
    elseif iter>1
        break;
    end;
end;

sqW = (W.^2);
sumW = sum(sqW,2);
[~,id] = sort(sumW,'descend');
