function [av_d,st_d]=hymet1(AC_SET,k)
AC_SET2=AC_SET;
for i=1:size(AC_SET2,1)
    new_AC=AC_SET2{i};
    new_AC(:,k+1)=AC_SET2{i}(:,k+1)/(k-1);
    hym(i)=sum(hypeIndicatorSampled(new_AC(:,k+1:k+2)));
end

av_d=mean(hym)
st_d=std(hym)
