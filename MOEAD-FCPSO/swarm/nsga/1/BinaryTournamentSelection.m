function p=BinaryTournamentSelection(pop,F)
%计算拥挤度之后的pop

   ii=randperm(numel( pop));
   %if numel(ii)==1
      % p=pop(F{1}(ii));
   %else
   p1=pop(ii(1));    %要么是1，要么是2
   p2=pop(ii(2));    %要么是1，要么是2
%谁的非支配排序小用谁    
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