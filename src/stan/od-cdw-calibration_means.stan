data {
   int<lower=0> N; // number of measurements
   vector[N] x_meas; // measurement explanatory (OD)
   vector[N] y_meas; // measurement response (CDW)
}
transformed data {
   // Hard coded parameters
   real<lower=0> sigma = 1; // model error
   real<lower=0> sigma_x = 5; // OD measurement error
   real<lower=0> sigma_y = 0.77 / 2; // CDW measurement error
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

   // measurement model
   x_meas ~ normal(x_true, sigma_x); // x_meas is a N long 
   y_meas ~ normal(y_true, sigma_y);

   // likelihood
}
