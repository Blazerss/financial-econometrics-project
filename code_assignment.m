%% IMPORTING AND CLEANING DATA
clc
clear 
close

rng(0349063);
idx=randsample(30,1);
Euro50=readtable("EuroStoxx50.xlsx");
stock=Euro50{idx,1};
asset=readtable(append(stock{1},".csv"),"Range","1307:6244");
asset.Properties.VariableNames([1 5])={'Dates' 'Close'};

p1=asset.Close;
figure(1)
dates=asset.Dates(2:end);
plot(asset.Dates,p1)
title("BMW price")
p1=rmmissing(p1);
r=diff(log(p1))*100;
figure(2),
plot(dates,r)
title("BMW returns")

%% CHARACTERISTICS OF RETURNS
figure(3),
autocorr(r)
title("Returns Autocorrelation")
subplot(2,1,1),autocorr(abs(r))
title("Absolute Returns Autocorrelation")
subplot(2,1,2),autocorr(r.^2)
title("Squared Returns Autocorrelation")

n=length(r);
m1= mean(r);
z=r-m1;
m2= mean(z.^2);
z=normalize(r);
m3= mean(z.^3);
m4= mean(z.^4);
JB = n * (m3^2+(m4-3)^2/4)/6 ;

figure(4),histfit(r)

%% COCHRANE VARIANCE RATIO TEST
y=log(p1);
n=length(y);
maxK = floor(n^(1/3));
kvalues = 2:maxK;
K = length(kvalues);
[~, pValue, stat, ~, VR] = vratiotest(y,'period',kvalues,'IID', false(1,K));

figure(5),plot(kvalues, pValue, 'r*')
title("P-value Cochrane ratio test for different k")
xlabel("k")
ylabel("p-value_{k}","Rotation",0)
yline(0.05,"b--","0.05")


%% CHOICE OF THE ORDER p
P=20;
logL=zeros(P,1);
AIC=zeros(P,1);
for i=1:P
     Mdl = garch(0,i);
     [~,~,logL(i)] = estimate(Mdl,r);
    AIC(i)=2*(i+1)-2*logL(i);
end
[minAIC,idx]=min(AIC);
numP= idx;
figure(6),plot(1:P,AIC,"r*")
xlabel("p")
ylabel("AIC","Rotation",0)
title("AIC for different ARCH(p)")


%% FIT ARCH(20)

Mdl = garch(0,numP);
EstMdl = estimate(Mdl,r);
cond_var_arch=infer(EstMdl,r);
cond_var_arch_for=forecast(EstMdl,22,r);

figure(7),
subplot(2,1,1),plot(dates,cond_var_arch)
title("Inferred Conditional Variance ARCH(20)")
hold on
subplot(2,1,2),plot(dates,r.^2)
title("Squared returns")
hold off


%% DIAGNOSTIC CHECKING OF ARCH(20)
standRes=r./sqrt(cond_var_arch);
figure(8),
plot(dates,standRes)
title("Standardized Residuals")
figure(9),
subplot(2,1,1),
autocorr(standRes)
title("Sample Autocorrelation Standardized Returns")
subplot(2,1,2),
autocorr(standRes.^2)
title("Sample Autocorrelation Standardized Squared Returns")
figure(10),
qqplot(standRes)

n_e=length(standRes);
m1_e= mean(standRes);
z_e=standRes-m1_e;
m2_e= mean(z_e.^2);
z_e=normalize(standRes);
m3_e= mean(z_e.^3);
m4_e= mean(z_e.^4);
JB_e = n_e * (m3_e^2+(m4_e-3)^2/4)/6 ;


%% FIT GARCH(1,1)

Mdl_asset1 = garch(1,1);
[EstMdl_2,~,logL_garch] = estimate(Mdl_asset1,r);
cond_var_garch= infer(EstMdl_2,r);
cond_var_garch_for=forecast(EstMdl_2,22,r);
AIC_garch=2*3-2*logL_garch;

figure(11),
subplot(2,1,1),plot(dates,cond_var_garch)
title("Inferred Conditional Variance GARCH(1,1)")
hold on
subplot(2,1,2),plot(dates,r.^2)
title("Squared returns")
hold off

%% RESIDUALS ANALYSIS GARCH(1,1)
standRes_garch=r./sqrt(cond_var_garch);
figure(12),
subplot(2,1,1),autocorr(standRes_garch)
title("Sample Autocorrelation Standardized Returns")
subplot(2,1,2),autocorr(standRes_garch.^2)
figure(13),
qqplot(standRes_garch)



%% COMPARING CONDITIONAL VARIANCE FORECASTS
dates_for=[datetime(2024,06,3:7) datetime(2024,06,10:14) datetime(2024,06,17:21) datetime(2024,06,24:28) datetime(2024,07,1:2)];
figure(14),
subplot(2,1,1),plot(dates,cond_var_arch,"-b")
hold on
plot(dates_for,cond_var_arch_for,"r")
xlim([datetime(2024,01,01) datetime(2024,07,01)])
title("Conditional Variance Forecast ARCH(20)")
hold off
subplot(2,1,2),plot(dates,cond_var_garch,"-b")
hold on
plot(dates_for,cond_var_garch_for,"r")
xlim([datetime(2024,01,01) datetime(2024,07,01)])
title("Conditional Variance Forecast GARCH(1,1)")
hold off

%% LEVERAGE EVIDENCE
fit= smooth(r(1:end-1), r(2:end).^2, 0.95, 'loess');

figure(15),
crosscorr(r, r.^2)
title("Sample cross-correlation between r_{t}^2 and r_{t}")

figure(16),
plot(r(1:end-1), r(2:end).^2, 'k.', r(1:end-1), fit, 'r.')
title("Volatility Smile")

%% FIT MODELS FOR LEVERAGE
EstMdl_gjr_n = estimate(gjr(1,1), r);
ht1=infer(EstMdl_gjr_n,r);

EstMdl_egarch_n = estimate(egarch(1,1), r);
ht2=infer(EstMdl_egarch_n,r);

Mdl_gjr = gjr(1,1);
Mdl_gjr.Distribution = "t";
EstMdl_gjr = estimate(Mdl_gjr, r);
ht3=infer(EstMdl_gjr,r);

Mdl_Egar = egarch(1,1);
Mdl_Egar.Distribution = "t";
EstMdl_egarch = estimate(Mdl_Egar, r);
ht4=infer(EstMdl_egarch,r);

%% DIAGNOSTIC CHECKING LEVERAGE MODELS
figure(17),
subplot(2,2,1),
autocorr((r./sqrt(ht1)).^2)
title("Gaussian GJR-GARCH(1,1)")
subplot(2,2,2),
autocorr((r./sqrt(ht2)).^2)
title("Gaussian EGARCH(1,1)")
subplot(2,2,3),
autocorr((r./sqrt(ht3)).^2)
title("Student's t EGARCH(1,1)")
subplot(2,2,4),
autocorr((r./sqrt(ht4)).^2)
title("Student's t EGARCH(1,1)")

figure(18),
subplot(2,2,1),
qqplot(r./sqrt(ht1))
title("Gaussian GJR-GARCH(1,1)")
subplot(2,2,2),
qqplot(r./sqrt(ht2))
title("Gaussian EGARCH(1,1)")
subplot(2,2,3),
pd=makedist("tLocationScale","mu",0,"sigma",1,"nu",6);
qqplot(r./sqrt(ht3),pd)
title("Student's t GJR-GARCH(1,1)")
xlabel(["Quantiles of standardized Student's t" ;" with \nu = 6"])
subplot(2,2,4),
qqplot(r./sqrt(ht4),pd)
title("Student's t EGARCH(1,1)")
xlabel(["Quantiles of standardized Student's t" ;" with \nu = 6"])

%% CONDITIONAL STANDARD DEVIATION COMPARISON
figure(19),
subplot(2,1,1),plot(dates,sqrt(cond_var_garch),"-b")
title("Conditional Standard deviation GARCH(1,1) normal distribution")

figure(20),
subplot(2,1,1),plot(dates,sqrt(ht1),"-b")
title("Conditional Standard deviation GJR(1,1) normal distribution")

subplot(2,1,2),plot(dates,sqrt(ht2),"-r")
title("Conditional Standard deviation EGARCH(1,1) normal distribution")

figure(21),
subplot(2,1,1),plot(dates,sqrt(ht3),"-k")
title("Conditional Standard deviation GJR(1,1) student's t distribution")

subplot(2,1,2),plot(dates,sqrt(ht4),"-g")
title("Conditional Standard Deviation EGARCH(1,1) student's t distribution")

%% IMPORTING AND CLEANING THE DATA

rng(0349063);
idx_2=randsample(30,3);
stock_2=Euro50{idx_2,1};
asset_1=readtable(append(stock_2{1},".csv"),"Range","1307:6244");
asset_2=readtable(append(stock_2{2},".csv"),"Range","1307:6244");
asset_3=readtable(append(stock_2{3},".csv"),"Range","1307:6244");

r_1=diff(log(asset_1.Var5))*100;
r_2=diff(log(asset_2.Var5))*100;
r_3=diff(log(asset_3.Var5))*100;

r_t=[r_1 r_2 r_3];
r_t=rmmissing(r_t);
figure(22),
plot(dates,r_t)
legend(["BMW" "INTESA SP" "ENI"],"location","southwest")

%% CHARACTERISTICS OF THE RETURNS
m1= mean(r_t);
z=r_t-m1;
m2= mean(z.^2);
z=normalize(r_t);
m3= mean(z.^3);
m4= mean(z.^4);

%% RISKMETRICS APPROACH
lambda=0.06;
mh_ii   = NaN(length(r_t),3);
mh_ij   = [];
mrho_ij   = [];
for i = 1:3
    r_i =  r_t(:,i);
    h_ii = filter(1, [1 -(1-lambda)], lambda * r_i.^2);
    mh_ii(:,i) = h_ii;
    for j = i+1:3
         r_j =  r_t(:,j);
         h_jj = filter(1, [1 -(1-lambda)], lambda * r_j.^2);
         h_ij = filter(1, [1 -(1-lambda)], lambda * r_i.*r_j);
         rho_ij =  h_ij ./(sqrt(h_ii.*h_jj));
         mh_ij = [mh_ij, h_ij] ; 
         mrho_ij = [mrho_ij, rho_ij];
         
    end
end
figure(23),
plot(dates,mh_ii)
title("Conditional Variances Riskmetrics")
legend(["h_{BMW}" "h_{ISP}" "h_{ENI}"],"location","best")
ylabel("h_{i}","rotation",0)

figure(24),
plot(dates,mh_ij)
title("Conditional Covariances Riskmetrics")
legend(["h_{BMW-ISP}" "h_{BMW-ENI}" "h_{ISP-ENI}"],"location","best")
ylabel("h_{ij}","rotation",0)

figure(25),
plot(dates,mrho_ij)
title("Conditional Correlation Riskmetrics")
legend(["\rho_{BMW-ISP}" "\rho_{BMW-ENI}" "\rho_{ISP-ENI}"],"location","best")
ylabel("\rho_{ij}","rotation",0)

%% Engle's DCC APPROACH
cond_var_asset=zeros(length(r_t(:,1)),3);
stand_r_t=zeros(length(r_t(:,1)),3);
for i=1:3
Mdl_asset1 = garch(1,1);
EstMdl_asset(:,i) = estimate(Mdl_asset1,r_t(:,i));
cond_var_asset(:,i)= infer(EstMdl_asset(:,1),r_t(:,i));
stand_r_t(:,i)= r_t(:,i)./sqrt(cond_var_asset(:,i));
end

% MLE of a and b
a = 0.06; b =  0.94; 
mQbar = cov(stand_r_t,1); 
vPsi0 = [log(a/(1-a)); log(b/(1-b)) ]; 

f  = @(vPsi)fDCC_LogLikelihood(stand_r_t, mQbar, vPsi);

opts = optimset('Display','iter','TolX',1e-4,'TolFun',1e-4,...
                'Diagnostics','off', 'MaxIter',1000, 'MaxFunEvals', 1000,...
                'LargeScale', 'off', 'PlotFcns', @optimplotfval);

[vPsi, fval, exitflag, output] = fminunc(f, vPsi0, opts);
a = exp(vPsi(1))/(1+exp(vPsi(1)));    
b = (1-a)*exp(vPsi(2))/(1+exp(vPsi(2)));

Q=zeros(3,3);
P_engle=zeros(3,3,length(stand_r_t(:,1)));
Ht_engle=zeros(3,3,length(stand_r_t(:,1)));
for i = 1:length(stand_r_t(:,1))
    if i ==1
        Q =  mQbar;
    else            
        Q = (1 - a - b) * mQbar + a * ( stand_r_t(i-1,:)' * stand_r_t(i-1,:) ) + b * Q ;
    end
    mQnsqrt = diag(1 ./ sqrt(diag(Q)));
    P_engle(:,:,i) = mQnsqrt * Q * mQnsqrt;
    Ht_engle(:,:,i) = diag(sqrt(cond_var_asset(i,:))) * P_engle(:,:,i) * diag(sqrt(cond_var_asset(i,:)));
end

figure(26),
plot(dates,reshape(Ht_engle(1,1,:),[],1))
hold on
plot(dates,reshape(Ht_engle(2,2,:),[],1))
plot(dates,reshape(Ht_engle(3,3,:),[],1))
title("Conditional Variances Engle's DCC")
legend(["h_{BMW}" "h_{ISP}" "h_{ENI}"],"location","best")
ylabel("h_{i}","rotation",0)
hold off

figure(27),
plot(dates,reshape(Ht_engle(1,2,:),[],1))
hold on
plot(dates,reshape(Ht_engle(1,3,:),[],1))
plot(dates,reshape(Ht_engle(2,3,:),[],1))
title("Conditional Covariances Engle's DCC")
legend(["h_{BMW-ISP}" "h_{BMW-ENI}" "h_{ISP-ENI}"],"location","best")
ylabel("h_{ij}","rotation",0)
hold off

figure(28),
plot(dates,reshape(P_engle(1,2,:),[],1))
hold on
plot(dates,reshape(P_engle(1,3,:),[],1))
plot(dates,reshape(P_engle(2,3,:),[],1))
title("Conditional Correlations Engle's DCC")
legend(["\rho_{BMW-ISP}" "\rho_{BMW-ENI}" "\rho_{ISP-ENI}"],"location","best")
ylabel("\rho_{ij}","rotation",0)
hold off

%% O-GARCH APPROACH

%pca analysis
V=pca(r_t);
f_t=r_t*V;

%fit GARCH(1,1) for each factor
cond_var_fac=zeros(length(r_t(:,1)),3);
stand_r_t=zeros(length(r_t(:,1)),3);
for i=1:3
    Mdl_ft = garch(1,1);
    EstMdl_fac = estimate(Mdl_ft,f_t(:,i));
    cond_var_fac(:,i)= infer(EstMdl_fac,f_t(:,i));
end
H_ft=zeros(3,3,length(cond_var_fac(:,1)));
%construct Ht for the factor
for i=1:length(cond_var_fac(:,1))
    H_ft(:,:,i)= diag(cond_var_fac(i,:));
end


Ht_ogar=zeros(3,3,length(cond_var_fac(:,1)));
Pt_ogar=zeros(3,3,length(cond_var_fac(:,1)));
% recover Ht for the returns
for i=1:length(cond_var_fac(:,1))
    Ht_ogar(:,:,i)= V*H_ft(:,:,i)*V';
    hinv=diag(1./sqrt(diag(Ht_ogar(:,:,i))));
    Pt_ogar(:,:,i)=hinv*Ht_ogar(:,:,i)*hinv;
end

figure(29),
plot(dates,reshape(Ht_ogar(1,1,:),[],1))
hold on
plot(dates,reshape(Ht_ogar(2,2,:),[],1))
plot(dates,reshape(Ht_ogar(3,3,:),[],1))
title("Conditional Variances O-GARCH")
legend(["h_{BMW}" "h_{ISP}" "h_{ENI}"],"location","best")
ylabel("h_{i}","rotation",0)
hold off

figure(30),
plot(dates,reshape(Ht_ogar(1,2,:),[],1))
hold on
plot(dates,reshape(Ht_ogar(1,3,:),[],1))
plot(dates,reshape(Ht_ogar(2,3,:),[],1))
title("Conditional Covariances O-GARCH")
legend(["h_{BMW-ISP}" "h_{BMW-ENI}" "h_{ISP-ENI}"],"location","best")
ylabel("h_{ij}","rotation",0)
hold off

figure(31),
plot(dates,reshape(Pt_ogar(1,2,:),[],1))
hold on
plot(dates,reshape(Pt_ogar(1,3,:),[],1))
plot(dates,reshape(Pt_ogar(2,3,:),[],1))
title("Conditional Correlations O-GARCH")
legend(["\rho_{BMW-ISP}" "\rho_{BMW-ENI}" "\rho_{ISP-ENI}"],"location","best")
ylabel("\rho_{ij}","rotation",0)
hold off

%% COPULA
%estimate garch-t model for the asset
stds_residuals=NaN(length(r_t),3);
cond_stdev=NaN(length(r_t),3);
du=NaN(3,1);
mrho=[];
dnu=[];
u=NaN(length(r_t),3);

for i=1:3
[coeff, stds_residuals(:,i),cond_stdev(:,i)] = fgarch11t_fit(r_t(:,i));
du(i) = coeff.Distribution.DoF;
u(:,i)   = tcdf(stds_residuals(:,i)*sqrt(du(i)/(du(i)-2)), du(i));

end
for i=1:3
    for j=i+1:3
     [rho,nu] = copulafit('t',[u(:,i) u(:,j)],'Method','ApproximateML');
     mrho=[mrho rho(2,1)];
     dnu=[dnu nu];
    end
    
end

w_1 = -sqrt((dnu(1)+1)*sqrt(1-mrho(1))/sqrt(1+mrho(1)));
td_12 = 2*tpdf(w_1,dnu(1)+1); % tail dependence

w_2 = -sqrt((dnu(2)+1)*sqrt(1-mrho(2))/sqrt(1+mrho(2)));
td_13 = 2*tpdf(w_2,dnu(2)+1); % tail dependence

w_3 = -sqrt((dnu(3)+1)*sqrt(1-mrho(3))/sqrt(1+mrho(3)));
td_23 = 2*tpdf(w_3,dnu(3)+1); % tail dependence

rho_matrix=[1 mrho(1) mrho(2); mrho(1) 1 mrho(3); mrho(2) mrho(3) 1];
disp(rho_matrix)

td_matrix=[1 td_12 td_13; td_12 1 td_23; td_13 td_23 1];
disp(td_matrix)

%% VAR ESTIMATION

T=3500;
P=0.05;
w=22;
n=length(r_t(:,1))-T;
delta=floor(n/w);

est_var_1=NaN(n,3);
est_var_2=NaN(n,3);
est_var_3=NaN(n,3);
est_var_4=NaN(n,3);

cond_var_1=NaN(n,3);
cond_var_2=NaN(n,3);
cond_var_3=NaN(n,3);
cond_var_4=NaN(n,3);

garch_n = garch(1,1);
garch_n.Offset = NaN;

egarch_n = egarch(1,1);
egarch_n.Offset = NaN;

garch_t = garch(1,1);
garch_t.Distribution = "t";
garch_t.Offset = NaN;

gjr_t = gjr(1,1);
gjr_t.Distribution = "t";
gjr_t.Offset = NaN;


for i=1:3

    for k=0:w:(delta*w)
        
        Est_garch_n = estimate(garch_n, r_t(k+1:T+k,i));

        Est_egarch_n = estimate(egarch_n,  r_t(k+1:T+k,i));

        Estgarch_t = estimate(garch_t,  r_t(k+1:T+k,i));
        dof_1=Estgarch_t.Distribution.DoF;

        Est_gjr_t = estimate(gjr_t,  r_t(k+1:T+k,i));
        dof_2=Est_gjr_t.Distribution.DoF;

        for j=0:w-1 
            if k==delta*w && j>=(n-delta*w)
                break
            end
            cond_var_1(j+k+1,i)=forecast(Est_garch_n,1,r_t(j+k+1:T+j+k,i));
            est_var_1(j+k+1,i)=-(Est_garch_n.Offset+norminv(P)*sqrt(cond_var_1(j+k+1,i)));

            cond_var_2(j+k+1,i)=forecast(Est_egarch_n,1,r_t(j+k+1:T+j+k,i));
            est_var_2(j+k+1,i)=-(Est_egarch_n.Offset+norminv(P)*sqrt(cond_var_2(j+k+1,i)));

            cond_var_3(j+k+1,i)=forecast(Estgarch_t,1,r_t(j+k+1:T+j+k,i));
            est_var_3(j+k+1,i)=-(Estgarch_t.Offset+(tinv(P,dof_1)/sqrt(dof_1/(dof_1-2)))*sqrt(cond_var_3(j+k+1,i)));

            cond_var_4(j+k+1,i)=forecast(Est_gjr_t,1,r_t(j+k+1:T+j+k,i));
            est_var_4(j+k+1,i)=-(Est_gjr_t.Offset+(tinv(P,dof_2)/sqrt(dof_2/(dof_2-2)))*sqrt(cond_var_4(j+k+1,i)));

        end
    end
end


%% EVALUATION OF VAR PREDICTION backtesting
m=length(est_var_1);
exp_violations=floor(P*m);

violations1 = r_t(T+1:end,:) < -est_var_1;
violations2 = r_t(T+1:end,:) < -est_var_2;
violations3 = r_t(T+1:end,:) < -est_var_3;
violations4 = r_t(T+1:end,:) < -est_var_4;

backtest1 = sum(violations1);
backtest2 = sum(violations2);
backtest3 = sum(violations3);
backtest4 = sum(violations4);

%% EVALUATION OF VAR PREDICTION total loss incurred (check loss function)

check_function1 = (P- violations1).*(r_t(T+1:end,:) + est_var_1);
check_function2 = (P- violations2).*(r_t(T+1:end,:) + est_var_2);
check_function3 = (P- violations3).*(r_t(T+1:end,:) + est_var_3);
check_function4 = (P- violations4).*(r_t(T+1:end,:) + est_var_4);

total_loss_var1 = sum(check_function1);
total_loss_var2 = sum(check_function2);
total_loss_var3 = sum(check_function3);
total_loss_var4 = sum(check_function4);


%% EVALUATION OF VAR PREDICTION Diebold-Mariano test statistic

lambda=0.06;

h_rm= filter(1, [1 -(1-lambda)], lambda * r_t.^2);
cond_var_rm= lambda*(r_t(T:end-1,:).^2)+(1-lambda)*h_rm(T:end-1,:);
var_rm= -sqrt(cond_var_rm)*norminv(P);
violation_rm= r_t(T+1:end,:) < -var_rm;
backtest_rm= sum(violation_rm);
check_function_rm = (P - violation_rm).*(r_t(T+1:end,:) + var_rm);
total_loss_rm= sum(check_function_rm);


q = floor(m^(1/3));

d_1 = check_function1-check_function_rm;
d_2 = check_function2-check_function_rm;
d_3 = check_function3-check_function_rm;
d_4 = check_function4-check_function_rm;

gamma_v01=var(d_1);
gamma_v02=var(d_2);
gamma_v03=var(d_3);
gamma_v04=var(d_4);

dbar1 = (1/m)*sum(d_1);
dbar2 = (1/m)*sum(d_2);
dbar3 = (1/m)*sum(d_3);
dbar4 = (1/m)*sum(d_4);

for i=1:3
    [acf_d1(:,i),lags] = autocorr(d_1(:,i));
    acf_d2(:,i) = autocorr(d_2(:,i));
    acf_d3(:,i) = autocorr(d_3(:,i));
    acf_d4(:,i) = autocorr(d_4(:,i));
end

sigma_LRV1d = gamma_v01+2*sum((1-lags(2:q)/q).*(acf_d1(2:q,:).*gamma_v01));
sigma_LRV2d = gamma_v02+2*sum((1-lags(2:q)/q).*(acf_d2(2:q,:).*gamma_v02));
sigma_LRV3d = gamma_v03+2*sum((1-lags(2:q)/q).*(acf_d3(2:q,:).*gamma_v03));
sigma_LRV4d = gamma_v04+2*sum((1-lags(2:q)/q).*(acf_d4(2:q,:).*gamma_v04));

DM_test1d = dbar1./sqrt(sigma_LRV1d/m);
DM_test2d = dbar2./sqrt(sigma_LRV2d/m);
DM_test3d = dbar3./sqrt(sigma_LRV3d/m);
DM_test4d = dbar4./sqrt(sigma_LRV4d/m);

pvalue1d = 2*( 1-normcdf(abs(DM_test1d)));
pvalue2d = 2*( 1-normcdf(abs(DM_test2d)));
pvalue3d = 2*( 1-normcdf(abs(DM_test3d)));
pvalue4d = 2*( 1-normcdf(abs(DM_test4d)));

MSFE_1  = mean((r_t(T+1:end,:).^2 - cond_var_1).^2);
MSFE_2  = mean((r_t(T+1:end,:).^2 - cond_var_2).^2);
MSFE_3  = mean((r_t(T+1:end,:).^2 - cond_var_3).^2);
MSFE_4  = mean((r_t(T+1:end,:).^2 - cond_var_4).^2);
MSFE_rm = mean((r_t(T+1:end,:).^2 - cond_var_rm).^2);

%% MSFE for asset 1 USING Gaussian GJR-GARCH

cond_var_5=NaN(n,1);
for k=0:w:(delta*w)
    Est_gjr_n = estimate(gjr(1,1),r_t(k+1:T+k,1));
    for j=0:w-1
        if k==delta*w && j>=(n-delta*w)
            break
        end
        cond_var_5(j+k+1,1)=forecast(Est_garch_n,1,r_t(j+k+1:T+j+k,1));
    end
end

MSFE_5  = mean((r_t(T+1:end,1).^2 - cond_var_5).^2);

%% COVID and UK-RU war analysis 
covid=dates(dates>= "2020-02-01" & dates<="2020-09-01");
war=dates(dates>= "2022-02-01" & dates<="2022-09-01");
figure(32),
plot(dates(T+1:end),r_t(T+1:end,:))
xline(covid(1),"g--")
xline(covid(end),"g--","COVID","LabelOrientation","horizontal","LabelHorizontalAlignment","RIGHT")
xline(war(1),"K--")
xline(war(end),"K--","UK-RU WAR","LabelOrientation","horizontal","LabelHorizontalAlignment","right")
legend(["BMW " "INTESA S.P." "ENI"],"location","southeast")

m_covid=length(covid);
k= dates>= "2020-02-01" & dates<="2020-09-01";
u=k(dates>= "2020-02-01" & dates<="2020-09-01");

backtest_var1_covid = sum(violations1(u,:));
backtest_var2_covid = sum(violations2(u,:));
backtest_var3_covid = sum(violations3(u,:));
backtest_var4_covid = sum(violations4(u,:));
backtest_rm_covid   = sum(violation_rm(u,:));

total_loss_var1_covid = sum(check_function1(u,:));
total_loss_var2_covid = sum(check_function2(u,:));
total_loss_var3_covid = sum(check_function3(u,:));
total_loss_var4_covid = sum(check_function4(u,:));
total_loss_rm_covid   = sum(check_function_rm(u,:));

q_covid = floor(m_covid^(1/3));

d_1_covid = check_function1(u,:)-check_function_rm(u,:);
d_2_covid = check_function2(u,:)-check_function_rm(u,:);
d_3_covid = check_function3(u,:)-check_function_rm(u,:);
d_4_covid = check_function4(u,:)-check_function_rm(u,:);

gamma_v01_covid=var(d_1_covid);
gamma_v02_covid=var(d_2_covid);
gamma_v03_covid=var(d_3_covid);
gamma_v04_covid=var(d_4_covid);

dbar1_covid = (1/m)*sum(d_1_covid);
dbar2_covid = (1/m)*sum(d_2_covid);
dbar3_covid = (1/m)*sum(d_3_covid);
dbar4_covid = (1/m)*sum(d_4_covid);

for i=1:3
    [acf_d1_covid(:,i),lags] = autocorr(d_1_covid(:,i));
    acf_d2_covid(:,i) = autocorr(d_2_covid(:,i));
    acf_d3_covid(:,i) = autocorr(d_3_covid(:,i));
    acf_d4_covid(:,i) = autocorr(d_4_covid(:,i));
end

sigma_LRV1d_covid = gamma_v01_covid+2*sum((1-lags(2:q_covid)/q_covid).*(acf_d1_covid(2:q_covid,:).*gamma_v01_covid));
sigma_LRV2d_covid = gamma_v02_covid+2*sum((1-lags(2:q_covid)/q_covid).*(acf_d2_covid(2:q_covid,:).*gamma_v02_covid));
sigma_LRV3d_covid = gamma_v03_covid+2*sum((1-lags(2:q_covid)/q_covid).*(acf_d3_covid(2:q_covid,:).*gamma_v03_covid));
sigma_LRV4d_covid = gamma_v04_covid+2*sum((1-lags(2:q_covid)/q_covid).*(acf_d4_covid(2:q_covid,:).*gamma_v04_covid));

DM_test1d_covid = dbar1_covid./sqrt(sigma_LRV1d_covid/m);
DM_test2d_covid = dbar2_covid./sqrt(sigma_LRV2d_covid/m);
DM_test3d_covid = dbar3_covid./sqrt(sigma_LRV3d_covid/m);
DM_test4d_covid = dbar4_covid./sqrt(sigma_LRV4d_covid/m);

pvalue1d_covid = 2*( 1-normcdf(abs(DM_test1d_covid)));
pvalue2d_covid = 2*( 1-normcdf(abs(DM_test2d_covid)));
pvalue3d_covid = 2*( 1-normcdf(abs(DM_test3d_covid)));
pvalue4d_covid = 2*( 1-normcdf(abs(DM_test4d_covid)));

MSFE_1_covid  = mean((r_t(k,:).^2 - cond_var_1(u,:)).^2);
MSFE_2_covid  = mean((r_t(k,:).^2 - cond_var_2(u,:)).^2);
MSFE_3_covid  = mean((r_t(k,:).^2 - cond_var_3(u,:)).^2);
MSFE_4_covid  = mean((r_t(k,:).^2 - cond_var_4(u,:)).^2);
MSFE_rm_covid  = mean((r_t(k,:).^2 - cond_var_rm(u,:)).^2);

m_war=length(war);
h= dates>= "2022-02-01" & dates<="2022-09-01";
g=h(dates>= "2022-02-01" & dates<="2022-09-01");

backtest_var1_war = sum(violations1(g,:));
backtest_var2_war = sum(violations2(g,:));
backtest_var3_war = sum(violations3(g,:));
backtest_var4_war = sum(violations4(g,:));
backtest_rm_war   = sum(violation_rm(g,:));

total_loss_var1_war = sum(check_function1(g,:));
total_loss_var2_war = sum(check_function2(g,:));
total_loss_var3_war = sum(check_function3(g,:));
total_loss_var4_war = sum(check_function4(g,:));
total_loss_rm_war   = sum(check_function_rm(g,:));

q_war = floor(m_war^(1/3));

d_1_war = check_function1(g,:)-check_function_rm(g,:);
d_2_war = check_function2(g,:)-check_function_rm(g,:);
d_3_war = check_function3(g,:)-check_function_rm(g,:);
d_4_war = check_function4(g,:)-check_function_rm(g,:);

gamma_v01_war=var(d_1_war);
gamma_v02_war=var(d_2_war);
gamma_v03_war=var(d_3_war);
gamma_v04_war=var(d_4_war);

dbar1_war = (1/m)*sum(d_1_war);
dbar2_war = (1/m)*sum(d_2_war);
dbar3_war = (1/m)*sum(d_3_war);
dbar4_war = (1/m)*sum(d_4_war);

for i=1:3
    [acf_d1_war(:,i),lags] = autocorr(d_1_war(:,i));
    acf_d2_war(:,i) = autocorr(d_2_war(:,i));
    acf_d3_war(:,i) = autocorr(d_3_war(:,i));
    acf_d4_war(:,i) = autocorr(d_4_war(:,i));
end

sigma_LRV1d_war = gamma_v01_war+2*sum((1-lags(2:q_war)/q_war).*(acf_d1_war(2:q_war,:).*gamma_v01_war));
sigma_LRV2d_war = gamma_v02_war+2*sum((1-lags(2:q_war)/q_war).*(acf_d2_war(2:q_war,:).*gamma_v02_war));
sigma_LRV3d_war = gamma_v03_war+2*sum((1-lags(2:q_war)/q_war).*(acf_d3_war(2:q_war,:).*gamma_v03_war));
sigma_LRV4d_war = gamma_v04_war+2*sum((1-lags(2:q_war)/q_war).*(acf_d4_war(2:q_war,:).*gamma_v04_war));

DM_test1d_war = dbar1_war./sqrt(sigma_LRV1d_war/m);
DM_test2d_war = dbar2_war./sqrt(sigma_LRV2d_war/m);
DM_test3d_war = dbar3_war./sqrt(sigma_LRV3d_war/m);
DM_test4d_war = dbar4_war./sqrt(sigma_LRV4d_war/m);

pvalue1d_war = 2*( 1-normcdf(abs(DM_test1d_war)));
pvalue2d_war = 2*( 1-normcdf(abs(DM_test2d_war)));
pvalue3d_war = 2*( 1-normcdf(abs(DM_test3d_war)));
pvalue4d_war = 2*( 1-normcdf(abs(DM_test4d_war)));

MSFE_1_war  = mean((r_t(h,:).^2 - cond_var_1(g,:)).^2);
MSFE_2_war  = mean((r_t(h,:).^2 - cond_var_2(g,:)).^2);
MSFE_3_war  = mean((r_t(h,:).^2 - cond_var_3(g,:)).^2);
MSFE_4_war  = mean((r_t(h,:).^2 - cond_var_4(g,:)).^2);
MSFE_rm_war = mean((r_t(h,:).^2 - cond_var_rm(g,:)).^2);