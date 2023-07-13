function ppop = non_domination_sort_mod1(PP,M,k,maxP)
ppop=[];
s1=size(PP,1);
if s1>maxP
    ppop=sel_pareto(PP,M,k);
    PP=REST(PP,ppop,k);
    while size(ppop,1)<maxP &     size(PP,1)~=0
        A1=sel_pareto(PP,M,k);%%16列
        PP=REST(PP,A1,k);%%注意！REST函数要定义一下！正确~~~~
        ppop=[ppop;A1];%%ppop里面存放的是前三层的所有个体
    end
    crowd_value=calcul_crowd(A1,M,k);
    [FFF,I]=sort(crowd_value);%%从小往大排序，记录对FF_dis集合中的元素排序，FF_dis包含各个点的聚集距离,I里面存放的是数据原来在的位置
    FF=[];%%用来存放聚集距离比较大的前面的几个点
    nsel=size(ppop,1)-maxP;%%在第三层挑出需要被剪掉的个体数
    for i=1:nsel
        FF=[FF;A1(I(i),1:k+2)];%%选择最外层边界的部分个体待删去
    end
    ppop=REST(ppop,FF,k);

    
%     [val,ind1]=sort(A1(:,k+1));
%     AA1=A1;
%     AA1(:,k+1)=
%     s2=size(A1,1);%%第三层的个体数
%     %%下面程序表示e.g.前两层中的个体数填不满种群，前三层的又超出了种群个数，所以需要从第三层中选择部分个体以填满种群
%     for i=2:(s2-1)
%         FF_dis(ind1(i))=0;
%         for j=1:M
%             FF_dis(ind1(i))=FF_dis(ind1(i))+A1(ind1(i+1),k+j)-A1(ind1(i-1),k+j);
%         end
%     end
%     FF_dis(ind1(1))=inf;%%保留边界上的端点
%     FF_dis(ind1(s2))=inf;
%     [FFF,I]=sort(FF_dis);%%从小往大排序，记录对FF_dis集合中的元素排序，FF_dis包含各个点的聚集距离,I里面存放的是数据原来在的位置
%     FF=[];%%用来存放聚集距离比较大的前面的几个点
%     nsel=size(ppop,1)-maxP;%%在第三层挑出需要被剪掉的个体数
%     for i=1:nsel
%         FF=[FF;A1(I(i),1:k+2)];%%选择最外层边界的部分个体待删去
%     end
%     ppop=REST(ppop,FF,k);
else
    ppop=PP;
end




function crowd_value=calcul_crowd(new_AC,M,k)


s2=size(new_AC,1);
if(s2>=2)
for(i=1:M)
      LIM_f(i,2)=max(new_AC(:,k+i));
      LIM_f(i,1)=min(new_AC(:,k+i));
end 
DD=[];
crowd_value=[];
    for(i=1:M)
        [val,ind]=sort(new_AC(:,k+i));
        for(j=1:s2)
            if(j==1 )
                DD(ind(j),i)=inf;
            elseif(j==s2)
                DD(ind(j),i)=inf;
            else                
                DD(ind(j),i)=(new_AC(ind(j+1),k+i)-new_AC(ind(j-1),k+i))/(2*(LIM_f(i,2)-LIM_f(i,1)));  
            end
        end
    end
    for(jj=1:s2)
       crowd_value(jj)=sum(DD(jj,:));
    end
else
    crowd_value(1)=1;
end










