    function [x,cumpr]=ECDF(data,paras)
    % generate empirical distribution function
    % input:
    % data is a vector
    % output:
    % x is sample observation vector
    % cumpr is cumulative probability vector
    if min(size(data))~=1
        error('data must be a vector')
    end
    n=length(data);
    data=reshape(data,n,1);%transposed to a column vector
    data=sort(data);
    [x,a,~]=unique(data);
    diffa = diff(a);
    frequency=[diffa;n-a(length(x))+1];
    cumpr=cumsum(frequency)/n;
    frq = frequency/n;
    
    y1 = cumpr;
%   y2 = frq;
 
    plot(x,y1,'LineWidth',3)
    xlabel('x','FontSize',20,'Fontname','Times newman')
    ylabel('y','Rotation',0,'FontSize',20,'Fontname','Times newman')
    xlim([-inf,inf])
    ylim([0 1])
    title('Diagram','FontSize',15.0)
    legend('经验分布')
