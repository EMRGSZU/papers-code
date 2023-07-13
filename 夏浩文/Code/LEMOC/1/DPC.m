function [cl] = DPC(data,weight,percent,cluster_num,isKernal)
%DPC 此处显示有关此函数的摘要
%   此处显示详细说明
N = size(data,1);
c=zeros(N,N);
for i = 1:N-1
    ix = data(i,:);
    for  j=i+1:N
        jx = data(j,:);
        c(i,j) = sqrt(sum(weight.*(ix-jx).^2));
        c(j,i) =  c(i,j);
    end
end
NX = N*(N-1) / 2;
ND = size(c, 1);
position = floor(NX*percent+1);  %% round 是一个四舍五入函数
sda = sort(squareform(c));
dc = sda(position);

dist = c;
if isKernal
    rho = sum((exp(-(dist/dc).^2)),2) - 1;
else
    rho = sum((c-dc)<0, 2) - 1;
end


%rho=rho';
%% 先求矩阵列最大值，再求最大值，最后得到所有距离值中的最大值
if isKernal == true

    maxd=max(max(dist));
else
    maxd=max(max(c));
end
%% 将 rho 按降序排列，ordrho 保持序
[~, ordrho]=sort(rho,'descend');

%% 处理 rho 值最大的数据点
delta(ordrho(1))=-1.;
nneigh(ordrho(1))=0;

%% 生成 delta 和 nneigh 数组
if isKernal == true
    for ii=2:ND
        delta(ordrho(ii))=maxd;
        for jj=1:ii-1
            if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
                delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
                nneigh(ordrho(ii))=ordrho(jj);
            end
        end
    end
else
    for ii=2:ND
        delta(ordrho(ii))=maxd;
        for jj=1:ii-1
            if(c(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
                delta(ordrho(ii))=c(ordrho(ii),ordrho(jj));
                nneigh(ordrho(ii))=ordrho(jj);
            end
        end
    end
end
%% 生成 rho 值最大数据点的 delta 值
delta(ordrho(1))=max(delta(:));
%% ind 在后面并没有用到
for i=1:ND
    ind(i)=i;
    gamma(i)=rho(i)*delta(i);
end

NCLUST = cluster_num;
for i=1:ND
    cl(i)=-1;
end
[~, Index] = sort(gamma, 'descend');
% cl是每个数据点的所属类别号
% icl是所有聚类中心的序号
icl = Index(1:NCLUST);
cl(Index(1:NCLUST)) = 1:NCLUST;
%% 在矩形区域内统计数据点（即聚类中心）的个数
% Calculate Distance Matrix
for i=1:ND
    if (cl(ordrho(i))==-1)
        cl(ordrho(i))=cl(nneigh(ordrho(i)));
    end
end
