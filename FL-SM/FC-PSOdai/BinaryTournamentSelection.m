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
%                                selectionʵ�ֶ����Ʊ���ѡ��     %
%                                ѡ�����                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function p=BinaryTournamentSelection(pop)
%����ӵ����֮���pop
%p��Ҫô��1Ҫô��2 
% �ڽ�����ѡ������У����ѡ��n���ˣ�����n����| tour_size |�� ����Щ������ֻѡ��
% һ������������ӵ�������У�����صĴ�СΪ| pool_size |�� ����������׼����ѡ��
% �� �����ǽ���������ڵĵȼ���ǰ�ء� ѡ�������ϵ͵ĸ��ˡ� ��Σ���������˵�����
% ��ͬ����Ƚ�ӵ�����롣 ӵ������ϴ����ѡ��
    npop=numel(pop);
    
    i=randi([1 2],[1 npop]);    %���ɾ��ȷֲ���α�������������[1 2],1*npop����
    
    p1=pop(i(1));    %Ҫô��1��Ҫô��2
    p2=pop(i(2));    %Ҫô��1��Ҫô��2
%˭�ķ�֧������С��˭    
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