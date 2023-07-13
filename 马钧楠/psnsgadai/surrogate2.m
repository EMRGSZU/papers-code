function [pop,cselect]=surrogate2(pop,pop_lable,index,flag,mscore,mrate,n_pop,X,Y)
    cselect=0;
    [popc_lable,index]=sortrows(popc_lable,[-1,2]);
        parfor k=1:round(n_crossover*flag)
            if pop_lable([index(k),1])>mscore & pop_lable([index(k),2])<mrate
                pop(index(k)).cost=fitness(X, Y, position);
            else
                pop(index(k)).cost=[1;1;1];
            end
        end
            
end