function [TT]=MOBPSO_NSGA3(a,b,M,coatalpha,FWeights,rp)
%% 问题定义

nVar=M;

VelMax=6;

%% MOPSO 设置
% nPop=max(min(round(M/20),300),100);   % 粒子群大小
nPop=100;
nRep=100;   % 外部存档大小

MaxIt=100;  % 最大迭代次数

phi1=2.05;%学习因子
phi2=2.05;
phi=phi1+phi2;
chi=2/(phi-2+sqrt(phi^2-4*phi));

w=chi;              % 惯性权重
c1=chi*phi1;        % 个人学习权重
c2=chi*phi2;        % 全局学习权重

rnd=0;      % 变异几率

 % nsga3算法参数
 n_obj=3;
 n_division=27;
 zr = reference_points(n_obj, n_division);
 optimun = ones(1,n_obj);
 param.n_pop = nPop;
 param.zr = zr;
 param.zmin = [];
 param.zmax = [];
 param.smin = [];

%% 初始化

suma=sum(a);
sumb=sum(b);

particle=CreateEmptyParticle(nPop);
idxOfGF=find(FWeights>0.52); %index of Features which weight is over 0.52 given by relieff 
for i=1:nPop 
    particle(i).Velocity=zeros(1,nVar);
    particle(i).Position=zeros(1,nVar); 
    if mod(i,3)==0
        randomnum=randperm(numel(idxOfGF),min(coatalpha,numel(idxOfGF)));
        particle(i).Position(idxOfGF(randomnum))=1;
    else
        randomnum=randperm(nVar,coatalpha); % 测试初始化值
        particle(i).Position(randomnum)=1; % 初始化自变量X的位置
    end 
    particle(i).Cost=MyCost(particle(i).Position,a,b,M,coatalpha,suma,sumb,rp); % 求解目标函数
    particle(i).Best=particle(i);
end

[rep, param] = select_pop(particle, param);

pre_h= CreateEmptyParticle(1);
pre_h.Position=zeros(1,nVar);
k=0;
K=3;

% hv = zeros(1,MaxIt);

%% MOPSO 主循环

for it=1:MaxIt
    for i=1:nPop
        rep_h=SelectLeader(rep);
        if ~all(rep_h.Position==pre_h.Position)
            pre_h=rep_h;
            k=0;
        elseif k==K
           rep_h.Position=zeros(1,nVar);
        else
            k=k+1;
        end
        particle(i).Velocity=w*particle(i).Velocity ...   
                             +(0.7-0.4*it/MaxIt)*c1*rand(1,nVar).*(particle(i).Best.Position - particle(i).Position) ...
                             +(0.3+0.4*it/MaxIt)*c2*rand(1,nVar).*(rep_h.Position - particle(i).Position);
                         
        particle(i).Velocity=min(max(particle(i).Velocity,-VelMax),+VelMax);
        
        switchIndicator=2./(1+exp(-abs(particle(i).Velocity)))-1; 
        tmpRand=rand(1,nVar);
        particle(i).Position(tmpRand<=switchIndicator & particle(i).Velocity>0)=1;
        particle(i).Position(tmpRand<=switchIndicator & particle(i).Velocity<0)=0;

        particle(i).Cost=MyCost(particle(i).Position,a,b,M,coatalpha,suma,sumb,rp);

        % mutate
%         rnd = it/MaxIt;
        if rand<rnd
            pmutate=mutate(particle(i));
            pmutate.Cost=MyCost(pmutate.Position,a,b,M,coatalpha,suma,sumb,rp);
            tmp=[particle(i);particle(i).Best;pmutate];
        else
            tmp=[particle(i);particle(i).Best];
        end
        % update pbest
        tmp=DetermineDomination(tmp);
        tmp=GetNonDominatedParticles(tmp);
        particle(i).Best=tmp(randperm(numel(tmp),1));

%         particle(i).Best=select(select(particle(i),particle(i).Best),pmutate);

    end
    
    j1=1;
    temp_particle=particle;
    for j=1:numel(particle)
        if (1<=sum(particle(j).Position)&&sum(particle(j).Position)<=coatalpha)
            temp_particle(j1)=particle(j);
            j1=j1+1;
        end
    end
    temp_particle=temp_particle(1:j1-1);
    
    rep=[rep; temp_particle];
    [rep, param] = select_pop(rep, param);
    
%     hv(it)=HV(rep, [1, 1, 1]);
    
end
%% 结果
% save('CEC_RES\param_hv\bear_bpso_nsga.mat','hv');
TT=rep;

end

function object=mutate(p)
    mu=0.1;
    D=numel(p.Position);
    no_of_1=sum(p.Position);
    no_of_0=D-no_of_1;
    p0=min(no_of_0,no_of_1)/no_of_0*mu;
    p1=min(no_of_0,no_of_1)/no_of_1*mu;
    object=p;
    if rand<0.5
        object.Position(and(p.Position==0,rand(1,D)<p0))=1;
    else
        object.Position(and(p.Position==1,rand(1,D)<p1))=0;
    end
end

function rep_h=SelectLeader(rep)
zr = [rep.GridIndex];
nzr = unique(zr);
rho = zeros(size(nzr));
for i = 1:numel(nzr)
    rho(i) = sum(zr == nzr(i));
end
min_n = min(rho);
idx = [];
for i = 1:numel(nzr)
    if rho(i) == min_n
        idx = [idx find(zr==nzr(i))];
    end
end
rep_h = rep(idx(randi(numel(idx))));
end

function z=MyCost(x,a,b,M,alpha,suma,sumb,rp)
    z1=a*x'/suma;
    z2=1-b*x'/sumb;
    z3 = rp(logical(x));
    z3 = z3(1);
    z=[z1; z2; z3];
end

function pop=DetermineDomination(pop)

    npop=numel(pop);
    
    for i=1:npop
        pop(i).Dominated=false;
        for j=1:i-1
            if all(pop(i).Position==pop(j).Position)
                pop(i).Dominated=true;
                break;
            end
            if ~pop(j).Dominated
                if Dominates(pop(i),pop(j))
                    pop(j).Dominated=true;
                elseif Dominates(pop(j),pop(i))
                    pop(i).Dominated=true;
                    break;
                end
            end
        end
    end

end

function dom=Dominates(x,y)

    if isstruct(x)
        if numel(find(x.Position~=y.Position))==0
            dom = 1;
            return
        end
    end

    if isstruct(x)
        x=x.Cost;
    end

    if isstruct(y)
        y=y.Cost;
    end
    
    dom=all(x<=y) && any(x<y);

end