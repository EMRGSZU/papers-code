function EP = Update(x,EP)
nEP=numel(EP);
if DominateC2(x,EP)
    EP(nEP+1)=x;
end

