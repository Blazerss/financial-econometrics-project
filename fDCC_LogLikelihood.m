function [dmloglik, da, db] = fDCC_LogLikelihood(my_star, mQbar, vPsi)
% input: mystar, devolatilised series and parameter values n times N
% matrix
% vPsi vector of parameters of dimension 2+ N*(N+1)/2
[n N] = size(my_star); % dimensions n and N
da = exp(vPsi(1))/(1+exp(vPsi(1)));    % a is between 0 and 1 
db = (1-da) * exp(vPsi(2))/(1+exp(vPsi(2)));% b is between 0 and (1-a) 
dmloglik = 0;   % minus log likelihood (initialisation)
for i = 1:n
    if i ==1       % first observation
        mQ =  mQbar;      
    else            
        mQ = (1 - da - db) * mQbar + da * ( my_star(i-1,:)' * my_star(i-1,:) ) + db * mQ ;
    end
    mQnsqrt = diag(1 ./ sqrt(diag(mQ)));
    mP = mQnsqrt * mQ * mQnsqrt;
    dmloglik = dmloglik + 0.5 * ( log(det(mP)) + my_star(i,:) * inv(mP) * my_star(i,:)'); 
    %disp(Q); disp(P);
end
end
 

