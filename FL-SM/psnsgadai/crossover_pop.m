function popc = crossover_pop(pop, n_crossover, X, Y,flag,featurescore)
% 交叉
    individual.position = [];
    individual.cost = [];
    popc = repmat(individual, n_crossover, 1);
    allscore=0;
    allrate=0;
    n=size(pop,1);
    for j=1:n
            tscore=featurescore.*pop(j).position;
            allscore=allscore+sum(tscore);
            allrate=allrate+pop(j).cost(2);
    end
    mscore=allscore/n;
    mrate=allrate/n;
    parfor k = 1:n_crossover
%         while true
%             parents = randsample(pop, 2, 1);
%             r = unifrnd(0, 1);
%             if r > 0.1
%                 position = simple_crossover(parents(1).position, parents(2).position);
%             else
%                 position = combine_crossover(parents(1).position, parents(2).position);
%             end
%             if ~inpop(position, pop)
%                 break;
%             end
%         end
        parents = randsample(pop, 2, 1);
        r = unifrnd(0, 1);
        if r > 0.1
            position = simple_crossover(parents(1).position, parents(2).position);
        else
            position = combine_crossover(parents(1).position, parents(2).position);
        end
        popc(k).position = position;
        if flag==0
            if surrogate(featurescore,position,mscore,mrate)
                popc(k).cost = fitness(X, Y, position);
            else
                popc(k).cost=[1;1;1];
            end
        else 
            popc(k).cost = fitness(X, Y, position);   
        end
%         if mod(k, 100) == 0
%             logger(['crossover ', num2str(k), '/', num2str(n_crossover), ', cost = ', num2str(pop(k).cost')]);
%         end
    end
end

function new_position = simple_crossover(position1, position2)
    new_position = repmat(position1, 1);
    index = randsample(numel(position2), round(numel(position2) / 2));
    new_position(index) = position2(index);
end

function new_position = combine_crossover(position1, position2)
    new_position = repmat(position1, 1);
    new_position(position2) = true;
end

function b = inpop(position, pop)
    b = false;
    for i = 1:numel(pop)
        if all(pop(i).position == position)
            b = true;
            break;
        end
    end
end
