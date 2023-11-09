function [popm,systemselect,fittime] = normalmutation(pop,Fl, n_mutation, X, Y,flag,featurescore,pos)
% 变异
    individual.position = [];
    individual.cost = [];
    popm = repmat(individual, n_mutation, 1);
    %popm_lable=zeros(n_mutation,2);
    systemselect=0;
    jscore=0;
    jrate=0;
    fittime=0;
    
     if flag~=0
        
        %n=size(Fl,1);
        %Flscore=zeros(1,n);
        n=size(pop,1);
        popscore=zeros(1,n);
        
        for j=1:n
            %tscore=featurescore.*Fl(j).position;
            tscore=featurescore.*pop(j).position;
            tscore=sum(tscore);
            %Flscore(j)=tscore;
            %Flrate(j)=Fl(j).cost(2);
            popscore(j)=tscore;
            poprate(j)=pop(j).cost(2);
        end
        
        if flag==1
            %jscore=mean(Flscore);
            %jrate=mean(Flrate);
            jscore=mean(popscore);
            jrate=mean(poprate);
        elseif flag==2
            %Flscore=Flscore*(-1);
            %jscore=geomean(Flscore)*(-1);
            %jrate=geomean(Flrate);
            popscore=popscore*(-1);
            jscore=geomean(popscore)*(-1);
            jrate=geomean(poprate);
        else
            %Flscore=Flscore*(-1);
            %jscore=harmmean(Flscore)*(-1);
            %jrate=harmmean(Flrate);
            popscore=popscore*(-1);
            jscore=harmmean(popscore)*(-1);
            jrate=harmmean(poprate);
        end
    end
    
    parfor k = 1:n_mutation

        p = pop(randi(numel(pop)));
        position = quick_bit_mutate(p.position);
        fitness_time=0;
        
        if flag~=0
                for z=1:size(pos,2)
                    position(pos(z))=0;
                end
                
                
                if surrogate(featurescore,position,jscore,jrate)
                    fitness_time1=clock;
                    cost =fitness(X, Y, position);%fitness(X, Y, position);
                    fitness_time2=clock;
                    fitness_time=etime(fitness_time2,fitness_time1);
                    systemselect=systemselect+1;
                else
                    cost=[1;1;1];
                    fitness_time=0;
                end
        else 
                fitness_time1=clock;
                cost =fitness(X, Y, position);%fitness(X, Y, position);
                fitness_time2=clock;
                fitness_time=etime(fitness_time2,fitness_time1);
                  
        end
        
        popm(k).position=position;
        popm(k).cost = cost;
        fittime=fittime+fitness_time;
    end
    %if flag~=0
    %    [popm_lable,index]=sortrows(popm_lable,[-1,2]);
    %    for k=1:n_mutation
    %        i=index(k);
    %        if k<round(n_mutation*flag/10)
    %            if popm_lable([i,1])>jscore & popm_lable([i,2])<jrate
    %                popm(i).cost=fitness(X, Y, popm(i).position);
    %                systemselect=systemselect+1;
    %            else
    %                popm(i).cost=[1;1;1];
    %            end
    %        else
    %            popm(i).cost=[1;1;1];
    %        end
    %    end
    %end
    
end

function new_position = quick_bit_mutate(position)
    mu = 0.1;
    r = rand;

    index_one = find(position == true);
    index_zero = find(position == false);
    n_one = numel(index_one);
    n_zero = numel(index_zero);

    n_mu = ceil(min(n_one, n_zero) * mu);
    new_position = repmat(position, 1);
    if r > 0.5
        index = randsample(index_one, n_mu);
        new_position(index) = false;
    else
        index = randsample(index_zero, n_mu);
        new_position(index) = true;
    end
end


