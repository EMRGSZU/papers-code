function [b, a]=GH_bisec...
    (M,N,patterns,targets,fstar,sigma)

% optimization problem is :
% min a(i,:)'T
% max b(i,:)'T
% st.1=<1'T=<Fm
% ti are 0 or 1

b=zeros(N,M);
a=zeros(N,M);
weight=zeros(N,N);

parfor i=1:N    
    cls1=targets(1,i);
    Temp_Tr=patterns;
    Temp_Tr(:,i)=[];
    Temp_class=targets;
    Temp_class(:,i)=[];
    x=patterns(:,i);
    Ncls1=0;
    for j=1:size(Temp_class,2)
        if Temp_class(1,j)==cls1
            Ncls1=Ncls1+1;
        end
    end
    Ncls2=size(Temp_class,2)-Ncls1;
    A=zeros(M,Ncls1);
    B=zeros(M,Ncls2);
    k=1;m=1;
    f_cls1_temp=zeros(M,1);
    f_cls2_temp=zeros(M,1); 
    f_Sparsity=zeros(M,1);
    for j=1:size(Temp_Tr,2)
        if Temp_class(1,j)==cls1
            A(:,k)=Temp_Tr(:,j); 
            k=k+1;       
        else 
            B(:,m)=Temp_Tr(:,j);
            m=m+1;
        end
    end
    %%%
    C=repmat(x,1,Ncls1)-A;
    D=repmat(x,1,Ncls2)-B;
    ro=C.^2;
    theta=D.^2;
    w=zeros(N,N);
    V=1:N;
    V(i)=[];
    %%%
    for p=1:N
        Dist_ro=sqrt(sum(repmat(fstar(:,p),[1,Ncls1]).*ro,1));
        Dist_alpha=sqrt(sum(repmat(fstar(:,p),[1,Ncls2]).*theta,1));
        w11=exp(-1*(Dist_ro-min(Dist_ro)).^2/sigma);
        w22=exp(-1*(Dist_alpha-min(Dist_alpha)).^2/sigma);
        w(p,V)=[w11(1,:),w22(1,:)];
    end
    w_ave=mean(w);
    w_ave(i)=[];    
    w1=w_ave(1,1:Ncls1);
    w2=w_ave(1,Ncls1+1:end);
    w3=0;
    normalized_w1=w1/sum(w1);
    normalized_w2=w2/sum(w2);
    wandering=0;
    w1=(1-wandering)*normalized_w1+wandering*rand(1,size(w1,2));
    w2=(1-wandering)*normalized_w2+wandering*rand(1,size(w2,2));    
    normalized_w1=w1/sum(w1);
    normalized_w2=w2/sum(w2);    
    weight(:,i)=[normalized_w1';normalized_w2';w3];
    w=weight(:,i);
    for n=1:Ncls1
        f_cls1_temp=f_cls1_temp+w(n).*ro(:,n);
    end
    for n=1:Ncls2
        f_cls2_temp=f_cls2_temp+w(n+Ncls1).*theta(:,n);
    end
    f_Sparsity=w(end).*ones(M,1);
    f_temp=f_Sparsity+f_cls2_temp;
    b(i,:)=f_temp'./Ncls2;
    a(i,:)=f_cls1_temp'./Ncls1;

end
end




