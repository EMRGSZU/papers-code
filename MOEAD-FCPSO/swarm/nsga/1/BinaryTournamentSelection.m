function p=BinaryTournamentSelection(pop,F)
%����ӵ����֮���pop

   ii=randperm(numel( pop));
   %if numel(ii)==1
      % p=pop(F{1}(ii));
   %else
   p1=pop(ii(1));    %Ҫô��1��Ҫô��2
   p2=pop(ii(2));    %Ҫô��1��Ҫô��2
%˭�ķ�֧������С��˭    
    if p1.Rank <p2.Rank
        p=p1;
    elseif p1.Rank > p2.Rank
        p=p2;
    else
        if p1.DistanceToAssociatedRef>=p2.DistanceToAssociatedRef
            p=p2;
        else
            p=p1;
        end
    end

   % end