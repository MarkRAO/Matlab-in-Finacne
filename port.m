close all; clc; clear;
averageReturn=zeros(374,5);
etfsReturn1=etfsReturn;

for i=1:5
    for j=1:374
        averageReturn(j,i)=mean(etfsReturn1(j:j+52,i));   
       
    end
end



covMatrix=cell(374,1);

for k=1:374
    covMatrix{k}=cov(etfsReturn1(k:k+52,1:5));
end


weight=cell(374,1);

for l=1:374
    weight{l}=inv(covMatrix{l})*averageReturn(l,:)';
    for i=1:length(weight{l})
        if weight{l}(i)<0
            weight{l}(i)=0;
        end
    weight{l}=weight{l}/sum(weight{l});
    end
end


weightedReturn=cell(374,1);
for m=53:425
    weightedReturn{m-52}=etfsReturn1(m,:).*weight{m-52}';
end

weightedReturn=cell2mat(weightedReturn);

portReturn=sum(weightedReturn,2);

figure;
plot(portReturn);
