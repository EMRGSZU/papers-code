function [fitness_val,cross_point] = fitness(X_index, X_all, Y, position)
    gamma = 0.2;
    cross_point = [];
    if sum(position) == 0
        fitness_val = [1; 1];
        return;
    end

    %第一个目标特征比例
    n_sample = size(X_all, 1);
    effective_sample = ones(n_sample,1);
    effective_sample = find(effective_sample == 1); %for distance
    num_feature = sum(position);  % num_feature => selected feature
    feature_rate = num_feature / size(position, 2);
    
    %第二个目标距离
    %dw = 0;   %相同类别
    X = X_all(:, position);
    
    D = pdist(X, 'euclidean');
    D2 = pdist(X, 'cityblock');
    distances1 = squareform(D);
    distances2 =  squareform(D2) / num_feature;

    cur_X = X(X_index,:);  %取X值
    cur_Y = Y(X_index,:);  %取Y值
    diff_class = sum((Y ~= cur_Y));  %统计X值
    same_class = sum((Y == cur_Y));  %统计Y值相同样本
    
%    Dist_Rep = abs(sqrt(sum((X - repmat(cur_X, n_sample, 1)).^2,2))); 
    Dist_Rep = distances1(X_index,effective_sample);
    [EE_Rep , ~]=sort(Dist_Rep);
%    diff_index = (Y ~= cur_Y);
%    same_index = (Y == cur_Y);
%    same_w = exp(-1 .* distances2(X_index, same_index));
%    diff_w = exp(-1 .* distances2(X_index, diff_index));
%    same_w = same_w ./ sum(same_w);
%    diff_w = diff_w ./ sum(diff_w);
%    distance = sum(same_w .*distances2(X_index, same_index)) / (sum(diff_w .* distances2(X_index, diff_index))+ 1e-5);

    Next=1;
    k=0;
    DR=0; %用于统计相邻样本满足相同类别的数目
    FAR = 0;
    diff_index = (Y ~= cur_Y);
    db = min(distances2(X_index, effective_sample(diff_index)));
    acc = 1;
    if same_class > 1
        while Next == 1
             k=k+1;
             UNQ=unique(EE_Rep);
             r=UNQ(k);
             F2=(Dist_Rep<=r);
             numb_same=sum(Y(F2) == cur_Y);    %相同类别总数
             numb_diff=sum(Y(F2) ~= cur_Y);    %不相类别总数
             
             if gamma * ((numb_same - 1)/(same_class - 1))<(numb_diff/diff_class)
                 %meet_circle = find(F2 == 1); %落在超球体内部
                 %same_index = (Y(F2) == cur_Y); %相同类别的index
                 %dw = max(distances2(X_index, meet_circle(same_index)));
                 %diff_index = Y(F2) ~= cur_Y;
                 %same_index = Y(F2) == cur_Y;
                 %same_w = exp(-1 .* distances1(X_index, meet_circle(same_index)));
                 %diff_w = exp(-1 .* distances1(X_index, meet_circle(diff_index)));
                 %same_w = same_w ./ sum(same_w);
                 %diff_w = diff_w ./ sum(diff_w);
                 %distance = sum(same_w .*distances2(X_index, same_index)) / (sum(diff_w .* distances2(X_index, diff_index))+ 1e-6);
                 Next = 0;
                 if (k-1)==0
                    r=UNQ(k);
                 else
                    r=0.5*(UNQ(k-1)+UNQ(k));
                 end
                 if r==0
                    r=1e-6;
                 end   
                 F2=(Dist_Rep<=r);  
                 [Q]=find(F2==1); %求超球体内的样本
                 for u = 1:size(Q,2)
                     if Q(u) ~= X_index   %我们暂时统计一下相同类别的样本，当然我们也可以将不同类别加进来(但是为了保证都是正样本)
                         %quasiTest_P=X(Q(u),:);
                         %Dist_quasiTest=abs(sqrt(sum((X-repmat(quasiTest_P,n_sample,1)).^2,2))); %其余训练点离该样本点的距离
                         Dist_quasiTest = distances1(Q(u),effective_sample);
                         [small , ~]=sort(Dist_quasiTest); %对其距离做一个排序
                         min_Uniq=unique(small);
                         m=0;
                        %下面这段代码找到最接近的几个样本点(取决于多少个重复，如果没重复就只是2个)
                         No_nereser=0;
                         while No_nereser< 2  %除去本身外最临近的那个值
                             m=m+1;
                             a1=min_Uniq(m);
                             NN=Dist_quasiTest<=a1;
                             No_nereser=sum(NN);
                         end
                        %判断最接近样本点到底label是什么
                         No_NN_C1 = sum(Y(NN) == cur_Y);
                         No_NN_C2 = No_nereser - No_NN_C1;             
                         %if (No_NN_C1 - 1)> No_NN_C2
                         %第三个目标，相邻样本之间的误差
                         %     if Y(Q(u)) == cur_Y
                         %       cross_point =[cross_point , Q(u)];
                         %       DR=DR+1;    %在选取该特征的情况下，满足球体内的样本在该特征下临近的几个样本也和自己是同类别
                         %     else 
                         %       BR=BR-1;
                         %     end
                         %end
                        if Y(Q(u)) == cur_Y && (No_NN_C1-1)>No_NN_C2
                            DR=DR+1;    %在选取该特征的情况下，满足球体内的样本在该特征下临近的几个样本也和自己是同类别
                            cross_point =[cross_point , Q(u)];
                        end
                        if Y(Q(u)) ~= cur_Y && No_NN_C1>(No_NN_C2-1)
                            FAR=FAR+1;  %%在选取该特征的情况下，满足球体内的样本在该特征下临近的几个样本也和自己是不同类别
                        end
                     end
                 end
                 acc = (1 - DR/same_class + FAR/diff_class) / 2;
             end
        end
    else
        dw = 1;
    end
    %distance = 1 / (1 + exp(-5 * (dw - db)));   
    fitness_val = [acc; feature_rate];
end

