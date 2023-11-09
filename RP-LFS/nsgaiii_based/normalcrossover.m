function [popc,fittime] = normalcrossover(pop, num, n_crossover, X, Y)
% 交叉 ---- parent1 随机挑选， parent2根据不相交特征的概率值来挑选
    individual.position = [];
    individual.cost = [];
    individual.crossPoint = [];
    popc = repmat(individual, n_crossover, 1);
    fittime=0;
    parfor k = 1:n_crossover
        parents = randsample(pop, 2, 1);
        position = simple_crossover(parents(1).position, parents(2).position);
        fitness_time1=clock;
        popc(k).position = position;
        [popc(k).cost,popc(k).crossPoint] = fitness(num, X, Y, position);%fitness(X, Y, position);
        fitness_time2=clock;
        fitness_time=etime(fitness_time2,fitness_time1);
        fittime=fittime+fitness_time;
    end
end

function new_position = simple_crossover(position1, position2)
    new_position = repmat(position1, 1);
    index = randsample(numel(position2), round(numel(position2) / 2));
    new_position(index) = position2(index);
end

