function pop=initialize(popsize,bounds)
numVars=size(bounds,1);
range=(bounds(:,2)-bounds(:,1))';
pop=zeros(popsize,numVars);
pop(:,1:numVars)=(ones(popsize,1)*bounds(:,1)')+(ones(popsize,1)*range).*(rand(popsize,numVars));
pop=round(pop);