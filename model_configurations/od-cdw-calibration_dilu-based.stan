data {
   int<lower=0> N; // number of measurements
   vector[N] x_meas; // measurement explanatory (OD)
   vector[N] y_meas; // measurement response (CDW)
   int D; // Number of dilutions
   array[N] int<lower=1, upper=D> dilution_x; // array for mapping measurements of x to dilutions 
   array[N] int<lower=1, upper=D> dilution_y; // array for mapping measurements of y to dilutions
   real<lower=0> sigma_x; // OD measurement error
   real<lower=0> sigma_y; // CDW measurement error
   real<lower=0> prior_mean_sigma_eps; // Prior for model error
   real<lower=0> prior_std_sigma_eps; // Prior for model error
}
transformed data {
   // Hard coded parameters

}
parameters {
   vector<lower=0>[D] x_true; // true OD value
   real beta1; // intercept
   real<lower=0> beta2; // slope
   vector<lower=0>[D] y_true; // true CDW value
   real<lower=0> sigma_eps;
   //vector[N] eps; // A model error for each measurement pair
}
transformed parameters {
   real<lower=0> sigma;
   sigma = sqrt(sigma_y^2 + sigma_eps^2); // combined model error and measurement error 
}
model {
    // Priors
    x_true ~ normal(60, 60); // needs to cover all reasonable values of x independent of cultures/dilutions
    y_true ~ normal(7, 7); // needs to cover all reasonable values of y independent of cultures/dilutions
    beta1 ~ normal(0, 1);
    beta2 ~ normal(0.5, 1);
    sigma_eps ~ normal(prior_mean_sigma_eps, prior_std_sigma_eps); // model error

   // measurement model
   x_meas ~ normal(x_true[dilution_x], sigma_x); // x_meas is a N long 
   y_meas ~ normal(y_true[dilution_y], sigma_y);

   // likelihood
   y_true ~ normal(beta2 * x_true, sigma);

}
generated quantities {
  real ypred[D];
  real log_lik[D];
  
  ypred = normal_rng(y_true, sigma);
  
  // calculate log likelihood
  for (i in 1:D){
    log_lik[i] = normal_lpdf(y_meas[i] | y_true[i], sigma); 
  }
}
