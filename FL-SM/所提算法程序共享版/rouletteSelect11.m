function sel_dim=rouletteSelect11(dim,deep,kk)

sum=0;
ss=size(deep,2);
fitness=deep';
for i=1:ss
    sum=sum+fitness(i,1);
end
for i=1:ss
    rfitness(i,1)=fitness(i,1)/sum;%relative fitness
end
for i=2:ss
    rfitness(i,1)=rfitness(i-1,1)+rfitness(i,1);%cumulative fitness :reflect roulette select
end
for i=1:kk %selection operation
    p=rand;
    j=1;
    while p>rfitness(j,1)
        j=j+1;
    end
    sel_dim(i)=dim(j);
end

