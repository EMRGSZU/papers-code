clear ; close all; clc
fprintf('Loading Data\n');

addpath("../data")
addpath("..")

CR = 0.2;
tic

folder = "../data/";
fileInfo = dir(folder+"**/*.mat");
fileName = {fileInfo.name};
fileNum = length(fileName);

for num= 5:6
    name = fileName{num};
    load(name);
    
    [negM,strM,medM] = isLinear(fea,[0.1,0.7],"z");

%     data = zscore(fea);
    data =fea;
    cluster_num = length(unique(gnd));
    NumD=size(data,2);
    weight1 = ismember((1:NumD)',negM); windx1 = find(weight1);
    weight2 = ismember((1:NumD)',medM); windx2 = find(weight2);
    weight3 = ismember((1:NumD)',strM); windx3 = find(weight3);

%     best1 =[];
%     best2 =[];
%     best3 =[];
%     best4 =[];
%     best5 =[];

    times= 2;
%     rand('state', 0);
    for runs=1:times
        disp(runs)

        A = 1;
        time1 = datetime;
        isKernal = false;

        %% data 第一个维度的长度，相当于样本数
        N = size(data,1);
        %% 初始化为零
        c=zeros(N,N);
        ObjectiveNum=2; % 选择目标个数
        H = [99 13  7  5  4  0  3  0  2];
        H = H(ObjectiveNum-1);
        [popsize,W] = F_weight(H,ObjectiveNum);
        W(W==0) = 0.000001;
        T = 2;
        %邻居判断
        BX = zeros(popsize);
        for i = 1 : popsize
            for j = i : popsize
                BX(i,j) = norm(W(i,:)-W(j,:));
                BX(j,i) = BX(i,j);
            end
        end
        [~,BX] = sort(BX,2);
        BX = BX(:,1:T);
        Population=rand(popsize,NumD+1);
%         Sum_pop=sum(Population(:,1:end-1),2);
%         Population1=Population(:,1:end-1)./repmat(Sum_pop,1,NumD);
%         [outMat,outInd]  = getOutlier(data,3,name);
%         Population=rand(popsize,NumD+1);
        Population1=Population(:,1:NumD);
        Population1(:,windx1') = Population1(:,windx1')*10;
        Population1(:,windx3') = Population1(:,windx3')*0.1;
        Sum_pop=sum(Population1,2);
        Population1=Population1./repmat(Sum_pop,1,NumD);
        %% 每个子问题
        for qkt=1:popsize
            weight = Population1(qkt,:);

            percent = Population(qkt,end);
            cl = DPC(data,weight,percent,cluster_num,isKernal);
            dtype=1;
            [DB,CH,Dunn,KL,Han,~] = valid_internal_deviation(data,cl,dtype);
%             [distFromEu,dmax]= similarity_euclid(data);
%             S = ind2cluster(cl);
%             [~,Sep,~] = valid_internal_intra(distFromEu,S,dtype,dmax);
            cp = valid_compactness(data, cl);
            Objective_Value3=0;
            %Objective_Value4=Edge(cl,data,c);
            FunctionValue(qkt,:)=[cp -CH];
%             FunctionValue(qkt,:)=[cp -CH DB -Dunn -Sep];
            label_FunctionValue(qkt,:)=cl;
        end

        Boundary=[1;0];
        Coding='Real';
        Z = min(FunctionValue);
        generation= 20 * popsize/ (popsize);
        for tt=1:1:generation
            time2 = datetime;
            numberofupdate = 0;
            for ip = 1 : popsize
                P = 1:popsize;
                kx = randperm(length(P));
                %产生子代
                Offspring = F_generator(Population(ip,:),Population(P(kx(1)),:),Population(P(kx(2)),:),Population(P(kx(3)),:),Population(P(kx(4)),:),Boundary,CR);
                Offspring1=Offspring(1:end-1);
                Offspring1(windx1') = Offspring1(windx1')*10;
                Offspring1(windx3') = Offspring1(windx3')*0.1;
                Offspring1=Offspring1./sum(Offspring1);
%                 Offspring1=Offspring(:,1:NumD);
%                 Sum_off=sum(Offspring1,2);
%                 Offspring1=Offspring1./repmat(Sum_off,1,NumD);
                percent = Offspring(end);
                cl = DPC(data,Offspring1,percent,cluster_num,isKernal);
                label_OffFunValue=cl;
                [DB,CH,Dunn,KL,Han,~] = valid_internal_deviation(data,cl,dtype);
%                 [distFromEu,dmax]= similarity_euclid(data);
%                 S = ind2cluster(cl);
%                 [~,Sep,~] = valid_internal_intra(distFromEu,S,dtype,dmax);
                cp = valid_compactness(data, cl);
                OffFunValue=[cp -CH];
%                 OffFunValue=[cp -CH DB -Dunn -Sep];

                %更新最优理想点
                Z = min(Z,OffFunValue);
                    
                %更新邻居个体
                for j = 1 : T
                    if A == 1
                        g_old = max(abs(FunctionValue(BX(ip,j),:)-Z).*W(BX(ip,j),:));
                        g_new = max(abs(OffFunValue-Z).*W(BX(ip,j),:));
                    elseif A == 2
                        d1 = abs(sum((FunctionValue(BX(ip,j),:)-Z).*W(BX(ip,j),:)))/norm(W(BX(ip,j),:));
                        g_old = d1+5*norm(FunctionValue(BX(ip,j),:)-(Z+d1*W(BX(ip,j),:)/norm(W(BX(ip,j),:))));
                        d1 = abs(sum((OffFunValue-Z).*W(BX(ip,j),:)))/norm(W(BX(ip,j),:));
                        g_new = d1+5*norm(OffFunValue-(Z+d1*W(BX(ip,j),:)/norm(W(BX(ip,j),:))));
                    end
                    if g_new < g_old
                        %更新当前向量的个体
                        numberofupdate = numberofupdate + 1;
                        Population(ip,:) = Offspring;
                        FunctionValue(ip,:) = OffFunValue;
                        label_FunctionValue(ip,:)=label_OffFunValue;
                        %                 rrs(BX(ip,j),:)=rrx;
                    end
                end
                %FunctionValue = FunctionValue.*repmat(Fmax-Fmin,popsize,1)+repmat(Fmin,popsize,1);
            end
            label_FunctionValue;
            %
            for ij=1:1:popsize
                [~, ~, Rn, NMI] = exMeasure(label_FunctionValue(ij,:)',gnd);
                rr(ij,:)=[ NMI Rn];
            end
            loc = 40*(runs-1)+ 2*(tt-1);
            resNMIR(:,loc+1:loc+2) = rr;

            if ismember(tt,[5,10,15,20])
                [rrmin2,Indd2]=max(rr(:,1));
                ttstart = (tt/5 - 1)*3;
                resMat(runs,ttstart+1:ttstart+3) = [rr(Indd2,:),seconds(datetime-time2)];
            end
%             best1 = [best1, Population];
%             best2 = [best2, FunctionValue];
%             best3 = [best3, label_FunctionValue];
%             [rrmin,Indd]=max(rr(:,1));
%             best4 = [best4,rr(Indd,:)'];
%             best5 = [best5,Indd];
        end
        [rrmin,Indd]=max(rr(:,1));
        dur = seconds(datetime-time1);

        disp("duration is"+ num2str(dur)+" over")
    end
    saveName = "result/result_" + fileName{num} ;
    save(saveName,"resMat","resNMIR")
    disp(fileName{num}+" over")

    clearvars -except num fileName CR fileName1
end
toc
%     dlmwrite(['data' num2str(CR) '.csv' ],finalresult, 'precision', 4, 'newline', 'pc');
% end
