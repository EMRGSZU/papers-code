k=max(err,[],2);
for i=1:10   
    [n(i) d ]=min(leng(i,find(err(i,:)==k(i))));
    a=find(err(i,:)==k(i));
    nix(i)=a(1,d);
    
end
n=n';
nix=nix';
kang=[n k nix];