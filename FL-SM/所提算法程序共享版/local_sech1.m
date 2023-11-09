
function  [BEST,TEM]=local_sech1(AC,M,k,X,TEM)


gg=max(sum(AC(:,1:k)'));

s1=size(AC,1);
MM=sum(AC(:,1:k));
deep=MM/s1+1/k;
dim=1:k;
BEST=[];
for i=1:gg
    sel_dim1=rouletteSelect11(dim,deep,i);
    sel_dim2=rouletteSelect11(dim,deep,i);
    BASED1=zeros(1,k); BASED2=zeros(1,k);
    BASED1(sel_dim1)=1; BASED2(sel_dim2)=1;
    B1EFF=evaluation(BASED1,k,X);
    B2EFF=evaluation(BASED2,k,X);
    if B1EFF(k+2)<=B2EFF(k+2)
        BEST=[BEST;B1EFF];
    else
        BEST=[BEST;B2EFF];
    end
end
% 
% NPEFF=[];
% s1=size(AC,1);
% for jj=1:s1
%     refs=AC(jj,1:k);
% %     addr=find(refs==1);
% %     s3=size(addr,1);
% %     if s3>2
%         s4=ceil(rand*k);
%         s5=ceil(rand*k);
%         while s5==s4
%             s5=ceil(rand*k);
%         end
%         
%         NEW1=refs;NEW2=refs;
%         NEW1(s4)=1-NEW1(s4);
%         NEW2(s5)=1-NEW2(s5);
%         N1EFF=evaluation(NEW1,k,X);
%         N2EFF=evaluation(NEW2,k,X);
%         if abs(N1EFF(k+2)-AC(jj,k+2))>abs(N2EFF(k+2)-AC(jj,k+2))
%             flg1=s4;flg2=s5;
%         else
%             flg1=s5;flg2=s4;
%         end
%         
%         s6=ceil(rand*s1);
%         while s6==jj & s1>2
%            s6=ceil(rand*s1);
%         end
%         
%         cand=AC(s6,1:k);
%         if cand(flg1)==1 & cand(flg2)==1
%             cand(flg2)=0;
%         elseif cand(flg1)==0 & cand(flg2)==0
%             cand(flg1)=1;
%         elseif  cand(flg1)==1 & cand(flg2)==0
%             cand(flg1)=0;
%         else
%             cand(flg2)=0;
%         end
%         
%         CEFF=evaluation(cand,k,X);
%         NPEFF=[NPEFF;N1EFF;N2EFF;CEFF];
% %     end
% end
% 
% 
% 
% 
% 
%         