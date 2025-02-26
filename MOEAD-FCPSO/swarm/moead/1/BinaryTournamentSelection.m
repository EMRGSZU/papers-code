%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MATLAB Code for                                              %
%                                                               %
%  Non-dominated Sorting Genetic Algorithm II (NSGA-II)         %
%  Version 1.0 - April 2010                                     %
%                                                               %
%  Programmed By: S. Mostapha Kalami Heris                      %
%                                                               %
%         e-Mail: sm.kalami@gmail.com                           %
%                 kalami@ee.kntu.ac.ir                          %
%                                                               %
%       Homepage: http://www.kalami.ir                          %
%                                                               %
%  BinaryTournamentSelection.m : implelemnts binary tournament  %
%                                selection实现二进制比赛选择     %
%                                选择操作                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p=BinaryTournamentSelection(pop)
%计算拥挤度之后的pop
%p：要么是1要么是2 
% 在锦标赛选择过程中，随机选择n个人，其中n等于| tour_size |。 从这些个体中只选择
% 一个，并将其添加到交配池中，交配池的大小为| pool_size |。 根据两个标准进行选择
% 。 首先是解决方案所在的等级或前沿。 选择排名较低的个人。 其次，如果两个人的排名
% 相同，则比较拥挤距离。 拥挤距离较大的人选择。
    npop=numel(pop);
    
    i=randi([1 2],[1 npop]);    %生成均匀分布的伪随机整数，区间[1 2],1*npop矩阵
    
    p1=pop(i(1));    %要么是1，要么是2
    p2=pop(i(2));    %要么是1，要么是2
%谁的非支配排序小用谁    
    if p1.Rank < p2.Rank
        p=p1;
    elseif p1.Rank > p2.Rank
        p=p2;
    else
        if p1.CrowdingDistance>p2.CrowdingDistance
            p=p1;
        else
            p=p2;
        end
    end

end