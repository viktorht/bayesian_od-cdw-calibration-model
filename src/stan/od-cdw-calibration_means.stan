functions {
#include custom_functions.stan
}
data {
   int<lower=1> N;
   int<lower=1> N_train;
   int<lower=1> N_test;
   vector[N] x;
   vector[N] y;
   array[N_train] int<lower=1,upper=N> ix_train;
   array[N_test] int<lower=1,upper=N> ix_test;
   int<lower=0,upper=1> likelihood;
}
transformed data {
   // Hard coded parameters
   real<lower=0> sigma = 1; // model error
   real<lower=0> sigma_x = 5; // OD measurement error
   real<lower=0> sigma_y = 0.77 / 2; // CDW measurement error
   #vector[N] x_std = standardise_vector(x, mean(x), sd(x));
}
parameters {
   vector<lower=0>[N] x_true; // true OD value
   real beta1; // intercept
   real<lower=0> beta2; // slope
   //vector[N] eps; // A model error for each measurement pair
}
transformed parameters {
   vector<lower=0>[N] y_true; // true CDW value
   y_true = beta1 + beta2 * x_true;
}
model {
    // Priors
    x_true ~ normal(60, 60);
    y_true ~ normal(7, 7);
    beta1 ~ normal(0, 1);
    beta2 ~ normal(0.5, 1);
    //eps ~ normal(0, sigma);

   // likelihood
   if (likelihood){
      x ~ normal(x_true, sigma_x); // x is a N long 
      y ~ normal(y_true, sigma_y);
   }
}
generated quantities {
   // transform parameters back into natural space
   // beta1 = mean(x) + beta1 * 2 * sd(x); 
   // beta2 = mean(x) + beta2 * 2 * sd(x); 
   
   vector[N_test] yrep;
   vector[N_test] llik;
   for (n in 1:N_test){
      yrep[n] = normal_rng(beta1 + x_true[ix_test[n]] * beta2, sigma_y);
      llik[n] = normal_lpdf(y[ix_test[n]] | beta1 + x_true[ix_test[n]] * beta2, sigma_y);
   }
}
