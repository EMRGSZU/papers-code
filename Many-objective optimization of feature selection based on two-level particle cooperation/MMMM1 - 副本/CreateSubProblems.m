function sp=CreateSubProblems(nObj,nPop,T)
% 创造个子问题个数
    empty_sp.lambda=[];
    empty_sp.Neighbors=[];

    sp=repmat(empty_sp,nPop,1);
    
    %theta=linspace(0,pi/2,nPop);
     [W,~] = UniformPoint(300,3);
     %W=W((1:end-3),:);
%      W=W((4:end),:);
     %W=1./W;
    for i=1:nPop
        sp(i).lambda=W(i,:);          %sp为列
        
        %sp(i).lambda=[cos(theta(i))
        %              sin(theta(i))];
    end

    LAMBDA=[sp.lambda];
    LAMBDA=reshape(LAMBDA,3,300)';
    D=pdist2(LAMBDA,LAMBDA);

    for i=1:nPop
        [~, SO]=sort(D(i,:));
        sp(i).Neighbors=SO(2:T+1);
    end

end