load('mid_end.mat')
axxx=[];
for i=1:30
    axxx(1,i)=mean(midacc(i,:));
    axxx(2,i)=mean(endacc(i,:));
end
axxx(1,31)=mean(axxx(1,:));
axxx(2,31)=mean(axxx(2,:));