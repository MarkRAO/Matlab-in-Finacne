
%% Close all figures, clear the workspace, and clear the command window
close all; clc; clear;
%% Question 1

%% Part a: Construct and plot one path of Amazon's stock
 
%Since we can take advantage of no-arbitrage and use risk-neutral pricing,
%the value for mu is risk free interest rate.
mu = 0.0174; 

vol = 0.5797; % volatility
dt = 1; %each stock price is simulated at the end of the year in one step
t = 0:dt:20;
s = zeros( length(t),1); %stock prices
s(1) = 1762.71;  % stock price at time zero
for step_num = 2 : length(s)
    s(step_num) = s(step_num-1)*exp((mu-vol^2/2)*dt + vol*sqrt(dt)*randn);
end

figure;
plot(t, s);
title('Simulated Stock Price Path');
xlabel('Time (years)');
ylabel('Price (USD)');
%% Part b: Construct 1 million stock price paths

N_sample=1e6;

StockAllPath= ones(N_sample,21);

for sim_num = 1 : N_sample
    StockOnePath=ones(1,20);
    StockOnePath(1)=1762.71;
    UpperTri=triu(exp((mu-vol.^2./2).*dt + vol.*sqrt(dt).*randn(20,20)));
    StockOnePath=StockOnePath*UpperTri;
    StockOnePath=[1762.71,StockOnePath];
    StockAllPath(sim_num,:)=StockOnePath;
end
Mean20D=mean(StockAllPath(21,:));
fprintf('The mean DAY 20 stock price is %5.2f\n',Mean20D);

SE=std(StockAllPath(21,:))/sqrt(N_sample);
fprintf('The Standard error of DAY 20 stock price mean is %5.2f\n',SE);

alpha=0.05;
CI=[Mean20D-norminv(1-alpha/2,0,1)*SE...
    Mean20D+norminv(1-alpha/2,0,1)*SE];
fprintf('The 95%% confidence interval is [%5.2f, %5.2f]\n',...
    CI(1),CI(2))

%Histogram of the day 20 cumulative return

Return20=(StockAllPath(:,21)-1762.71)./1762.71*100;
expected_stock_price20 = mean(Return20);
figure; 
hold on
h=histogram(Return20);
h.EdgeColor='none';
plot([expected_stock_price20 expected_stock_price20], ...
	[0,max(h.Values)], 'r--','LineWidth',3);
legend({'Histogram' 'Mean Price'});
hold off;
xlabel('Stock Return (%)');
ylabel('Frequency');
title('Day 20 Stock Return Histogram');

%Histogram of the day 10 cumulative return

Return10=(StockAllPath(:,11)-1762.71)./1762.71*100;
expected_stock_price10 = mean(Return10);
figure; 
hold on
h1=histogram(Return10);
h1.EdgeColor='none';
plot([expected_stock_price10 expected_stock_price10], ...
	[0,max(h1.Values)], 'r--','LineWidth',3);
legend({'Histogram' 'Mean Price'});
hold off;
xlabel('Stock Return (%)');
ylabel('Frequency');
title('Day 10 Stock Return Histogram');

%% Part c: 

%Two histograms are similar because we both assume log-normal distribution
%for return and the drift is small. 
fprintf('The mean DAY 20 stock price is %5.2f\n',expected_stock_price20);
fprintf('The mean DAY 10 stock price is %5.2f\n',expected_stock_price10);
%According to the mean for return, Day 20 have higher return mean which 
%means greater deviation in stock price, because it have 10 more days to
%deviate and the drift for the process is not zero.
%% Part d: Compute and report the mean payoff of a regular European call option

call_payoff=max(StockAllPath(:,21)-1800,zeros(N_sample,1));
fprintf('The mean payoff for European call option is %5.2f\n',mean(call_payoff));


%% Part e: Discount the payoff
r=0.0174; % risk-free rate
T=20; % 20 periods in total
discounted_payoff = mean(call_payoff) * exp(-r*T);
fprintf('The price for European call option is %5.2f\n',discounted_payoff);

%% Part f:  Price the knock-out option

logicalVector=call_payoff>(2300-1800);
knock_call_payoff=call_payoff;
knock_call_payoff(logicalVector)=0;
discounted_knock_payoff = mean(knock_call_payoff) * exp(-r*T);
fprintf('The price for knock-out call option is %5.2f\n',discounted_knock_payoff);

%% Question 2
close all; clc; clear;
%% Part a.Obtain Data
% Returns of stocks
R_MSFT=0.128;
R_BABA=0.1204;
R_V=0.0677;
R_JPM=0.1186;

% Define current stock prices
s0_B = 177.53;
s0_M = 144.12;
s0_V = 178.95;
s0_J = 125.73;

% How many paths to generate
N_monteCarlo = 1e6;

% Which VaR we want
N_days = 7;
X = 0.99;

%Correlation matrix
Sigma=[1 0.57 0.81 0.47;
      0.57 1 0.47 0.53;
      0.81 0.47 1 0.41;
      0.47 0.53 0.41 1];
  
% Define annual volatilities
annVol_M = 0.1211;
annVol_B = 0.2613;
annVol_V = 0.1771;
annVol_J = 0.1943;

%% Part b: Simulate 1 million values of this equally-weighted portfolio

% Construct a vector of current stock prices
s0 = [s0_M; s0_B; s0_V; s0_J];

% Define means
mu = zeros(4,1);

% Define number of trading days in a year
TRADING_DAYS_PER_YEAR = 252;
dt = N_days / TRADING_DAYS_PER_YEAR;

% Construct a vector of volatilities
vol = [annVol_M; annVol_B; annVol_V;annVol_J];

% Compute drift and diffusion
drift = (mu - vol.^2/2 ) * dt;
diffusion = vol * sqrt(dt);

% Perform Cholesky decomposition
L = chol(Sigma,'lower');

% Sample uncorrelated standard normals
Z = randn( length(mu), N_monteCarlo );

% Construct correlated noise
noise = L * Z;

% Construct stock prices after N_days
sT = s0 .* exp( drift + diffusion .* noise );

% Construct weight for each stock
W_M=(1/s0_M)/sum(1./s0);
W_B=(1/s0_B)/sum(1./s0);
W_V=(1/s0_V)/sum(1./s0);
W_J=(1/s0_J)/sum(1./s0);

weight=[W_M W_B W_V W_J];

% Portfolio value at time T
PortfolioT=weight*sT;

% Histogram for portfolio value at time T
figure; 
hold on
h2=histogram(PortfolioT);
h2.EdgeColor='none';
expected_value=mean(PortfolioT);
plot([expected_value expected_value], ...
	[0,max(h2.Values)], 'r--','LineWidth',3);
legend({'Histogram' 'Mean Price'});
hold off;
xlabel('Stock Return (%)');
ylabel('Frequency');
title('Portfolio Value at time T Histogram');
%% Part c: Compute VaR

% Profit
profit=PortfolioT-s0'*weight';

% Compute VaR
VaR = -prctile( profit, 100 * (1-X) );

fprintf('%d-Day %.0f%% Value-At-Risk: $%.2f\n',N_days,X*100,VaR);

%% Part d

%There is a 0.01 probability that the portfolio will fall in value by more
%than 8.8262 over a seven-day period.

%% Question 3
close all; clc; clear;

%% First Pass at the data
data=readtable('topstocks_1996.csv');

disp(data.Properties.VariableNames');
disp(data.date(1:3));
data.date = datetime(data.date,'ConvertFrom','yyyymmdd');
disp(data.date(1:3));

data=table2timetable(data);
data=sortrows(data,'date');
disp(data(1:10,:));

prices=unstack(data,'PRC','TICKER');
disp(prices(1:5,:));

figure;
plot(prices.date,prices{:,:});
legend(prices.Properties.VariableNames{:});
ylabel('Prices');
title('Stock Prices for Largest 5 Companies in 1996')

%% Manipulating Data

prices = unstack(data,'PRC','PERMNO');
disp(prices.Properties.VariableNames');
data = sortrows(data,{'date','PERMNO'});
tickers=data.TICKER(data.date==data.date(1));

prices = unstack(data,'PRC','PERMNO','NewDataVariableNames',tickers);
disp(prices.Properties.VariableNames');

comNames = data.COMNAM( data.date == data.date(1));

figure;
plot(prices.date,prices{:,:});
legend(comNames);
ylabel('Prices');
title('Stock Prices for Largest 5 Companies in 1996');

%% Computing Cumulative Adjusted Return

returns = unstack(data,'RET','PERMNO','NewDataVariableNames',tickers);
cumreturns = returns;
cumreturns{:,:} = cumprod((1+returns{:,:}))-1;
cumreturns{:,:} = exp(cumsum(log(1+returns{:,:})))-1;

figure;
plot(prices.date,100*cumreturns{:,:});
leg=legend(comNames,'location','northwest');

title('Cumulative Return for Largest 5 Companies in 1996');
ylabel('Cumulative Return(%)');
set(gca,'Ygrid','on');

%% Computing Return on an Equal-Weighted Portfolio
data.group = repmat('portfolio',size(data,1),1);
ewreturns = unstack(data,'RET','group','AggregationFunction',@mean);
disp(ewreturns(1:3,:));

ewreturns.cumPortfolioReturn = cumprod(1+ewreturns.portfolio)-1;

hold on;
plot(ewreturns.date,100*ewreturns.cumPortfolioReturn,'r','LineWidth',3)
leg.String{end}='Equal-Weighted Return';


%% Compare to the Performance of the Overall Market

mktreturns=unstack(data,'vwretd','group','AggregationFunction',@mean);
mktreturns.cumReturn = cumprod( 1+ mktreturns.portfolio)-1;

plot(mktreturns.date,100*mktreturns.cumReturn,'k','LineWidth',3);
leg.String{end}='Market Return';

%% Merging with Other Data

file_name='F-F_Research_Data_Factors';
opts=detectImportOptions(file_name);
opts.DataLines(end)=1123;
kfdata=readtable(file_name,opts);
disp(kfdata(1:3,:));

%% Formatting Dates

kfdata.Properties.VariableNames{1} = 'kfdate';
fprintf('%d\n',kfdata.kfdate(1:3));

year=floor( kfdata.kfdate/100);
month=kfdata.kfdate-year*100;

kfdata.date = datetime(year,month+1,1)-1;
kfdata.date.Format = 'defaultdate';
disp(kfdata.date(1:3,:));
kfdata.kfdate=[];

kfdata=table2timetable(kfdata);
figure;
plot(kfdata.date,kfdata.RF)
ylabel('Risk-Free Rate');
title('1-month Risk-Free Rate from Ken French''s data');

kfdata{:,:}=kfdata{:,:}/100;

%% Merging

combineData=innerjoin(returns,kfdata,'Keys','date');

%% Computing Stocks Betas

excessReturns=combineData{2:end,1:5} - combineData.RF(1:end-1);
mktExcessReturn=combineData.Mkt_RF(2:end);
covarianceMatrix=cov([excessReturns , mktExcessReturn]);
beta = covarianceMatrix(end,1:end-1)/covarianceMatrix(end,end);

fprintf('Betas of 5 Largest Companies in 1996\n');
datestrings=datestr(combineData.date,'mmmmyyyy');
fprintf('Estimated Period:%s - %s\n',datestrings(1,:),datestrings(end,:));
fprintf('%20s %10s\n','Name','Beta');

for ii=1:length(beta)
    fprintf(['%20s %10.3f\n'],comNames{ii},beta(ii));
end

%% Security Market Line

meanExcessReturns=mean(excessReturns);
figure;
plot(beta,meanExcessReturns*100,'o');
xlabel('Risk (Beta)');
ylabel('Expected Excess Return(%)')
title('Testing CAPM using Individual Stock Returns');

text(beta,meanExcessReturns*100,comNames);

meanRm=mean(mktExcessReturn);
hold on;
plot([0,beta],[0,beta*meanRm*100],'r');
legend('Mean Stock Returns','SML','location','northwest');

% Coca Cola has the lowest beta and Intel has the highest beta.

% Because the number of companies we use to compute SML is small. We can
% include more major companies in the market to make this better.
