
%% Close all figures, clear the workspace, and clear the command window
close all; clc; clear;
%% Question 1

%% Part a: Determine the parameter
y1=30; y2=15;
d1=0.1; d2=0.16;
rho=0.5;
% We could use CDF of exponential ditribution to determine lamda
L1=-log(1-0.1)/12;
L2=-log(1-0.16)/12;
%% Part b: Use the Gaussian Copula Method
m=2;
N_sample=1e6;
sigma = rho*ones(m)+(1-rho)*eye(m);
X= mvnrnd(zeros(1,m), sigma, N_sample);
U=normcdf( X );
Y = [-log(1-U(:,1))/L1,-log(1-U(:,2))/L2];
Y1=Y(:,1); Y2=Y(:,2);
Y1(Y1>30*12)=30*12;
Y2(Y2>15*12)=15*12;
Y1=Y1/12;
Y2=Y2/12;

figure;
histogram(Y1,'BinWidth',3/12,'Normalization','probability')
title('Mortgage 1 Histogram')
xlabel('Years');
ylabel('Probability');
figure;
histogram(Y2,'BinWidth',3/12,'Normalization','probability')
title('Mortgage 2 Histogram')
xlabel('Years');
ylabel('Probability');
%% Part c: 
%It is more likely for mortgage 2 to survived until maturity. Mortgage 2 is
%more likely to default at any given quater at first.This change at
%appoximately the 10th year.
%% Part d: Compute average time and confidence interval.
AverageTime1=mean(Y1);
AverageTime2=mean(Y2);
fprintf('Average time of mortgage 1:%5.2f\n',AverageTime1)
fprintf('Average time of mortgage 2:%5.2f\n',AverageTime2)

alpha=0.1;
%95% confidence interval for mortgage 1

CI1=[AverageTime1-norminv(1-alpha/2,0,1)*std(Y1)/sqrt(N_sample)...
    AverageTime1+norminv(1-alpha/2,0,1)*std(Y1)/sqrt(N_sample)];

%95% confidence interval for mortgage 2
CI2=[AverageTime2-norminv(1-alpha/2,0,1)*std(Y2)/sqrt(N_sample)...
    AverageTime2+norminv(1-alpha/2,0,1)*std(Y2)/sqrt(N_sample)];

fprintf('The 95%% confidence interval average time of mortgage 1 is [%5.2f, %5.2f]\n',...
    CI1(1),CI1(2))
fprintf('The 95%% confidence interval average time of mortgage 2 is[%5.2f, %5.2f]\n',...
    CI2(1),CI2(2))
%% Part e: estimate the conditional expectationof final paymenttimes

% i Mortgage 1¡¯s expected final payment time if mortgage 2 defaults
Mean1_2Default=mean(Y1(Y2<15));
% ii Mortgage 1¡¯s expected final payment time if mortgage 2 survives
Mean1_2Survive=mean(Y1(Y2==15));  
% iii Mortgage 2¡¯s expected final payment time if mortgage 1 defaults
Mean2_1Default=mean(Y2(Y1<30));
% iv Mortgage 2¡¯s expected final payment time if mortgage 1 survives
Mean2_1Survive=mean(Y2(Y1==30));

fprintf('Mortgage 1¡¯s expected final payment time if mortgage 2 defaults is %.2f\n',Mean1_2Default)
fprintf('Mortgage 1¡¯s expected final payment time if mortgage 2 survives is %.2f\n',Mean1_2Survive)
fprintf('Mortgage 2¡¯s expected final payment time if mortgage 1 defaults is %.2f\n',Mean2_1Default)
fprintf('Mortgage 2¡¯s expected final payment time if mortgage 1 survives is %.2f\n',Mean2_1Survive)
%% Part f: Discussion of correlation.

%The correlation coefficient is positive which reveal the positive
%correlation of the two mortgage default. From part e, when mortgage 2
%defaults, mortgage 1¡¯s expected final payment is smaller and it is
%morelikely for mortgage 1 to default.When mortgage 1
%defaults, mortgage 2¡¯s expected final payment is smaller and it is
%morelikely for mortgage 2 to default. This reflect the possitive
%correlation between two mortgages.
