function [popm,fittime] = normalmutation(pop, num, n_mutation, X, Y)
% 变异
    individual.position = [];
    individual.cost = [];
    individual.crossPoint = [];
    popm = repmat(individual, n_mutation, 1);
    fittime=0;
    parfor k = 1:n_mutation

        p = pop(randi(numel(pop)));
        position = normal_bit_mutate(p.position);
        popm(k).position = position;
        fitness_time1=clock;
        [popm(k).cost,popm(k).crossPoint] =fitness(num, X, Y, position);%fitness(X, Y, position);
        fitness_time2=clock;
        fitness_time=etime(fitness_time2,fitness_time1);
        fittime=fittime+fitness_time;
    end
end

%function new_position = quick_bit_mutate(position)
%    mu = 0.1;
%    r = rand;

%    index_one = find(position == true);
%    index_zero = find(position == false);
%    n_one = numel(index_one);
%    n_zero = numel(index_zero);%

%    n_mu = ceil(min(n_one, n_zero) * mu);
%    new_position = repmat(position, 1);
%    if r > 0.5
%        index = randsample(index_one, n_mu);
%        new_position(index) = false;
%    else
%        index = randsample(index_zero, n_mu);
%        new_position(index) = true;
%    end
%end
function new_position = normal_bit_mutate(position)
    mu = 0.1;
    r = rand;

    index_one = find(position == true);
    index_zero = find(position == false);
    n_one = numel(index_one);
    n_zero = numel(index_zero);

    n_mu_one = ceil(n_one * mu);
    n_mu_zero = ceil(n_zero * mu);
    new_position = repmat(position, 1);
    if r > 0.5
        index = randsample(index_one, n_mu_one);
        new_position(index) = false;
    else
        index = randsample(index_zero, n_mu_zero);
        new_position(index) = true;
    end
end
