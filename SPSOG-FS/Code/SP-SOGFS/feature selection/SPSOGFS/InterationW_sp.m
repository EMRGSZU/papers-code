function [W] = InterationW_sp(L,X,gamma,dim)
% X 为 n*dim 矩阵
% W 为 dim*dim矩阵


INTER_W = 100;
Q = eye(dim);
xlx = X'*L*X;
xx = X'*X;
p = 1;

for i = 1:INTER_W
    W = inv(xlx+xx+gamma*Q)*xx;
    tempQ = 0.5*p * (sqrt(sum(W.^2,2)+eps)).^(p-2);
    Q = diag(tempQ);
    
    w1(i) = sum(sum((X-X*W).^2));     %X-XW的范数
    w2(i) = gamma*sum(sqrt(sum(W.^2,2)));
    w3(i) = trace(W'*X'*L*X*W);
    WResult(i) = w1(i)+w2(i)+w3(i);
%     if i > 1 && abs(WResult(i-1)-WResult(i)) < 0.000001
    if i > 1 && abs(WResult(i-1)-WResult(i)) < 0.001
        break;
    end;
end

