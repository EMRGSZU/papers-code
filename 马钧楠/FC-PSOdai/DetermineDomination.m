%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA124
% Project Title: Implementation of MOEA/D
% Muti-Objective Evolutionary Algorithm based on Decomposition
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function pop=DetermineDomination(pop,flag)

%      flag=0;
    nPop=numel(pop);

    for i=1:nPop
        pop(i).IsDominated=false;
    end
    
    if flag==0
        for i=1:nPop
            for j=i+1:nPop
                if Dominates(pop(i),pop(j))
                    pop(j).IsDominated=true;
                
                elseif Dominates(pop(j),pop(i))
                    pop(i).IsDominated=true;
                
                end
            end
        end
    else
        for i=1:nPop
            if pop(i).select==1
                for j=i+1:nPop
                    if pop(j).select==0
                        pop(i).IsDominated=false;
                    elseif Dominates(pop(i),pop(j))
                        pop(j).IsDominated=true;               
                    elseif Dominates(pop(j),pop(i))
                        pop(i).IsDominated=true;
                
                    end
                end
            else
                pop(i).IsDominated=true;
        end
    end
            
    
    
            
    


end