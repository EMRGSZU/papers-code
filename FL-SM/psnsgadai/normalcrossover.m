function [popc,systemselect,fittime] = normalcrossover(pop,Fl, n_crossover, X, Y,flag,featurescore,pos)
% 交叉
    individual.position = [];
    individual.cost = [];
    popc = repmat(individual, n_crossover, 1);
    %popc_lable=zeros(n_crossover,2);
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
    
    parfor k = 1:n_crossover
        parents = randsample(pop, 2, 1);
        position = simple_crossover(parents(1).position, parents(2).position);
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
        popc(k).position = position;
        popc(k).cost=cost;
        
        fittime=fittime+fitness_time;
        %if flag==0
        %    popc(k).cost = fitness(X, Y, position);
        %else
        %    popc_lable(k)=scoreandrate(featurescore,position);
        %end     
    end
    
    
    
    %if flag~=0
    %    [popc_lable,index]=sortrows(popc_lable,[-1,2]);
    %    for k=1:n_crossover
    %        i=index(k);
    %        if k<round(n_crossover*flag/10)
    %            if popc_lable([i,1])>jscore & popc_lable([i,2])<jrate
    %                popc(i).cost=fitness(X, Y, popc(i).position);
    %                systemselect=systemselect+1;
    %            else
    %                popc(i).cost=[1;1;1];
    %            end
    %        else
    %            popc(i).cost=[1;1;1];
    %        end
    %    end
    %end
   
    
end

function new_position = simple_crossover(position1, position2)
    new_position = repmat(position1, 1);
    index = randsample(numel(position2), round(numel(position2) / 2));
    new_position(index) = position2(index);
end

