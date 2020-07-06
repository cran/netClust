#include <RcppArmadillo.h>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::depends("RcppArmadillo")]]


void set_seed(double seed) {
  Rcpp::Environment base_env("package:base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(std::floor(std::fabs(seed)));
}


Rcpp::NumericMatrix multinom_r_cpp_call(Rcpp::NumericVector p, int n, int K){
  
  // Obtain environment containing function
  Rcpp::Environment base("package:stats"); 
  
  // Make function callable from C++
  Rcpp::Function multinom_r = base["rmultinom"];    
  
  // Call the function and receive its list output
  Rcpp::NumericMatrix res = multinom_r(Rcpp::_["n"] = n, Rcpp::_["size"] = K,
                                       Rcpp::_["prob"]  = p); // example of additional param
  
  // Return test object in list structure
  return res;
}


double absDbl(double x){
  if (x < 0){
    return -x;
  }else{
    return x;
  }
}


arma::vec arma_setdiff(arma::uvec& x, arma::uvec& y){
  
  x = arma::unique(x);
  y = arma::unique(y);
  
  for (size_t j = 0; j < y.n_elem; j++) {
    arma::uvec q1 = arma::find(x == y[j]);
    if (!q1.empty()) {
      x.shed_row(q1(0));
    }
  }
  
  Rcpp::NumericVector x2 = Rcpp::wrap(x);
  x2.attr("dim") = R_NilValue;
  return x2;
}


double EuclDistVec(arma::vec X, arma::vec Y){
  int n = X.size();
  double dist = 0;
  for (int i=0; i < n; i++) {
    dist = dist + pow((X(i)-Y(i)),2);
  }
  dist = sqrt(dist);
  return dist;
}


arma::vec MahalanobisVec(arma::mat x, arma::rowvec center, arma::mat cov){
  int n = x.n_rows;
  arma::mat x_cen;
  x_cen.copy_size(x);
  for (int i=0; i < n; i++) {
    x_cen.row(i) = x.row(i) - center;
  }
  return sum((x_cen * cov.i()) % x_cen, 1);    
}


arma::vec dmvnormVec ( arma::mat x,  arma::mat mean,  arma::mat sigma, bool log){ 
  
  arma::vec distval = MahalanobisVec(x,  mean, sigma);
  
  double logdet = sum(arma::log(arma::eig_sym(sigma)));
  double log2pi = 1.8378770664093454835606594728112352797227949472755668;
  arma::vec logretval = -( (x.n_cols * log2pi + logdet + distval)/2  ) ;
  
  if(log){ 
    return(logretval);
    
  }else { 
    return(exp(logretval));
  }
}


double Mahalanobis(arma::rowvec x, arma::rowvec center, arma::mat cov){
  arma::rowvec x_cen;
  double dist;
  x_cen = x - center;
  dist = as_scalar(x_cen * cov.i() * x_cen.t());
  return dist;    
}



double dmvnorm (arma::rowvec x, arma::rowvec mean,  arma::mat sigma, bool log){
  
  double distval = Mahalanobis(x,  mean, sigma);
  
  double logdet = sum(arma::log(arma::eig_sym(sigma)));
  double log2pi = 1.8378770664093454835606594728112352797227949472755668;
  double logretval = -( (x.n_cols * log2pi + logdet + distval)/2  ) ;
  
  if(log){
    return(logretval);
    
  }else {
    return(exp(logretval));
  }
}


List ll_seq(arma::cube x, arma::vec tau, arma::cube Mu, arma::field<arma::mat> Sigma, arma::vec seq, int n){
  int K = tau.size();
  double ll2 = 0;
  seq = seq - 1;
  for (int i = 0; i < (n-1); i++){
    for (int j = (i+1); j < n; j++){
      arma::rowvec xVec (2);
      arma::rowvec MuVec (2);
      for (int l=0; l < 2; l++){
        xVec(l) = x(i,j,l);
        MuVec(l) = Mu(seq(i),seq(j),l);
      } 
      arma::mat SigmaMat = Sigma(seq(i),seq(j));
      double dens = dmvnorm(xVec, MuVec, SigmaMat, TRUE);
      ll2 = ll2 + dens;
      for (int k = 0; k < K; k++){
        for (int m = 0; m < K; m++){
          if ((seq(i) == k) && (seq(j) == m)){
            ll2 = ll2 + log(tau(k)) + log(tau(m));
          }
        }
      }
    }
  }
  
  double likelihood = exp(ll2);
  
  List ret;
  ret["loglikelihood"] = ll2;
  ret["likelihood"] = likelihood;
  
  return ret;
}


double rho_fn(double rho, arma::vec Pi, double sigma2_k, arma::cube x, arma::vec Mu_kk, arma::mat unq_seq_mat, int k, int n){
  double rho_eqn = 0;
  int Pi_len = Pi.size();
  arma::mat A = {{0,1},{1,0}};
  for (int l = 0; l < Pi_len; l++){
    double rSum = 0;
    for (int i = 0; i < (n-1); i++){
      for (int j = (i+1); j < n; j++){
        arma::vec xVec (2);
        arma::vec yVec (2);
        for (int l=0; l < 2; l++){
          xVec(l) = x(i,j,l);
        } 
        yVec = xVec - Mu_kk;
        if ((unq_seq_mat(l,i) == k) && (unq_seq_mat(l,j) == k)){
          rSum = rSum + as_scalar(sigma2_k*pow(rho,3) - 0.5*pow(rho,2)*yVec.t()*A*yVec + rho*(yVec.t()*yVec - sigma2_k) - 0.5*yVec.t()*A*yVec);
        }
      }
    }
    rho_eqn = rho_eqn + Pi(l)*rSum;
  }
  return rho_eqn;
}


double unirootC2(arma::vec Pi, double sigma2_k, arma::cube x, arma::vec Mu_kk, arma::mat unq_seq_mat, int k, int n, double ll, double ul, double eps){
  double diff;
  double md;
  int ind = 0;
  arma::mat fn_ll (1,1);
  arma::mat fn_ul (1,1);
  arma::mat fn_md (1,1);
  int itr=0;
  while (ind == 0){
    md = (ll+ul)/2;
    diff = ul - ll;
    if (diff < eps) {
      ind = 1;
    }
    fn_ll = rho_fn(ll, Pi, sigma2_k, x, Mu_kk, unq_seq_mat, k, n);
    fn_ul = rho_fn(ul, Pi, sigma2_k, x, Mu_kk, unq_seq_mat, k, n);
    fn_md = rho_fn(md, Pi, sigma2_k, x, Mu_kk, unq_seq_mat, k, n);
    arma::mat s_ll = sign(fn_ll);
    arma::mat s_ul = sign(fn_ul);
    arma::mat s_md = sign(fn_md);
    if (s_ll(0,0) == s_ul(0,0)) {
      ind = 1;
    } else {
      if (s_ll(0,0) == s_md(0,0)) {
        ll = md;
      }else {
        ul = md;
      }
    }
    itr++;
  }
  return md;
}


List EStep(arma::cube x, arma::vec tau, arma::cube Mu, arma::field<arma::mat> Sigma, arma::vec seq, int burn, int itr, int n){
  int K = tau.size();
  arma::vec lbl (K);
  for (int k = 0; k < K; k++){
    lbl(k) = k;
  }
  seq = seq - 1;
  arma::mat seq_mat (itr, n);
  seq_mat.row(0) = seq.t();
  arma::mat count (itr, 1);
  count.ones();
  arma::mat unq_seq_mat;
  unq_seq_mat.insert_rows(0, seq_mat.row(0));
  arma::mat unq_seq_mat_count;
  unq_seq_mat_count.insert_rows(0, count.row(0));
  
  int b = 0;
  for (int ib = 0; ib < burn; ib++){
    int pos = as_scalar(arma::randi<arma::rowvec>(1, arma::distr_param(0,(n-1))));
    arma::uvec num_posVec = find(seq == seq(pos));
    int num_pos = num_posVec.size();
    if (num_pos > 3){
      arma::uword q = arma::conv_to<arma::uword>::from(arma::find(lbl == seq(pos)));
      arma::vec choice = lbl;
      choice.shed_row(q);
      arma::vec WSS (K);
      WSS.zeros();
      for (int k = 0; k < K; k++){
        arma::vec alt_seq = seq;
        alt_seq(pos) = k;
        for (int i = 0; i < (n-1); i++){
          for (int j = (i+1); j < n; j++){
            arma::vec xVec (2);
            arma::vec MuVec (2);
            for (int l=0; l < 2; l++){
              xVec(l) = x(i,j,l);
              MuVec(l) = Mu(alt_seq(i),alt_seq(j),l);
            } 
            WSS(k) = WSS(k) + EuclDistVec(xVec, MuVec);   
          }
        }
      }
      arma::vec WSS_candidate = WSS;
      arma::vec WSS_wo_choice = WSS;
      WSS_candidate.shed_row(q);
      arma::vec WSS_candidate_inv = 1/WSS_candidate;
      arma::vec alt_seq = seq;
      arma::vec prob_choice (K-1);
      arma::vec prob_choice_cum (K-1);
      double u_choice = arma::randu<double>();
      int choice_pos =-1;
      for (int k = 0; k < (K-1); k++){
        prob_choice(k) = (1.0/WSS_candidate(k))/accu(WSS_candidate_inv);
        if (k == 0){
          prob_choice_cum(k) = prob_choice(k);
          if (u_choice <= prob_choice(k)){
            alt_seq(pos) = choice(k);
            choice_pos = k;
          }
        }else {
          prob_choice_cum(k) = prob_choice_cum(k-1)+prob_choice(k);
          if (u_choice > prob_choice_cum(k-1) && u_choice <= prob_choice_cum(k)){
            alt_seq(pos) = choice(k);
            choice_pos = k;
          }
        }
      }
      // Rcout << "The value of choice : " << choice << "\n";
      // Rcout << "The value of prob_choice : " << prob_choice << "\n";
      // Rcout << "The value of u_choice : " << u_choice << "\n";
      // Rcout << "The value of choice : " << alt_seq(pos) << "\n";
      // Rcout << "The value of choice_pos : " << choice_pos << "\n";
      // alt_seq(pos) = choice(index_min(WSS_candidate));
      // WSS_wo_choice.shed_row(choice(index_min(WSS_candidate)));
      WSS_wo_choice.shed_row(choice(choice_pos));
      arma::vec WSS_wo_choice_inv = 1/WSS_wo_choice;
      // double trans_prob_alt_input = (1.0/min(WSS_candidate))/accu(WSS_candidate_inv);
      // double trans_prob_input_alt = (1.0/min(WSS_wo_choice))/accu(WSS_wo_choice_inv);
      double trans_prob_alt_input = (1.0/WSS_candidate(choice_pos))/accu(WSS_candidate_inv);
      double trans_prob_input_alt = (1.0/WSS_wo_choice(choice_pos))/accu(WSS_wo_choice_inv);
      List l1 = ll_seq(x, tau, Mu, Sigma, alt_seq+1, n);
      List l2 = ll_seq(x, tau, Mu, Sigma, seq+1, n);
      double ll1 = l1["loglikelihood"];
      double ll2 = l2["loglikelihood"];
      double alpha = (trans_prob_alt_input/trans_prob_input_alt)*exp(ll1-ll2);
      if (alpha > 1){
        alpha = 1;
      }
      double u = arma::randu<double>();
      // int accept = 0;
      // Rcout << "The value of seq : " << seq(pos) << "\n";
      if (u < alpha){
        // accept = 1;
        seq = alt_seq;
      }
    }
  }
  
  int cum_accept = 1;
  
  for (int it = 0; it < (itr-1); it++){
    int pos = as_scalar(arma::randi<arma::rowvec>(1, arma::distr_param(0,(n-1))));
    arma::uvec num_posVec = find(seq == seq(pos));
    int num_pos = num_posVec.size();
    if (num_pos > 3){
      arma::uword q = arma::conv_to<arma::uword>::from(arma::find(lbl == seq(pos)));
      arma::vec choice = lbl;
      choice.shed_row(q);
      arma::vec WSS (K);
      WSS.zeros();
      for (int k = 0; k < K; k++){
        arma::vec alt_seq = seq;
        alt_seq(pos) = k;
        for (int i = 0; i < (n-1); i++){
          for (int j = (i+1); j < n; j++){
            arma::vec xVec (2);
            arma::vec MuVec (2);
            for (int l=0; l < 2; l++){
              xVec(l) = x(i,j,l);
              MuVec(l) = Mu(alt_seq(i),alt_seq(j),l);
            } 
            WSS(k) = WSS(k) + EuclDistVec(xVec, MuVec);   
          }
        }
      }
      arma::vec WSS_candidate = WSS;
      arma::vec WSS_wo_choice = WSS;
      WSS_candidate.shed_row(q);
      arma::vec WSS_candidate_inv = 1/WSS_candidate;
      arma::vec alt_seq = seq;
      // alt_seq(pos) = choice(index_min(WSS_candidate));
      // WSS_wo_choice.shed_row(choice(index_min(WSS_candidate)));
      // arma::vec WSS_candidate_inv = 1/WSS_candidate;
      // arma::vec WSS_wo_choice_inv = 1/WSS_wo_choice;
      // double trans_prob_alt_input = (1.0/min(WSS_candidate))/accu(WSS_candidate_inv);
      // double trans_prob_input_alt = (1.0/min(WSS_wo_choice))/accu(WSS_wo_choice_inv);
      
      arma::vec prob_choice (K-1);
      arma::vec prob_choice_cum (K-1);
      double u_choice = arma::randu<double>();
      int choice_pos =-1;
      for (int k = 0; k < (K-1); k++){
        prob_choice(k) = (1.0/WSS_candidate(k))/accu(WSS_candidate_inv);
        if (k == 0){
          prob_choice_cum(k) = prob_choice(k);
          if (u_choice <= prob_choice(k)){
            alt_seq(pos) = choice(k);
            choice_pos = k;
          }
        }else {
          prob_choice_cum(k) = prob_choice_cum(k-1)+prob_choice(k);
          if (u_choice > prob_choice_cum(k-1) && u_choice <= prob_choice_cum(k)){
            alt_seq(pos) = choice(k);
            choice_pos = k;
          }
        }
      }
      WSS_wo_choice.shed_row(choice(choice_pos));
      arma::vec WSS_wo_choice_inv = 1/WSS_wo_choice;
      double trans_prob_alt_input = (1.0/WSS_candidate(choice_pos))/accu(WSS_candidate_inv);
      double trans_prob_input_alt = (1.0/WSS_wo_choice(choice_pos))/accu(WSS_wo_choice_inv);
      
      List l1 = ll_seq(x, tau, Mu, Sigma, alt_seq+1, n);
      List l2 = ll_seq(x, tau, Mu, Sigma, seq+1, n);
      double ll1 = l1["loglikelihood"];
      double ll2 = l2["loglikelihood"];
      double alpha = (trans_prob_alt_input/trans_prob_input_alt)*exp(ll1-ll2);
      if (alpha > 1){
        alpha = 1;
      }
      double u = arma::randu<double>();
      // int accept = 0;
      if (u < alpha){
        // accept = 1;
        cum_accept = cum_accept + 1;
        seq = alt_seq;
      }
      
      seq_mat.row(it+1) = seq.t();
      
      arma::vec match (b+1);
      match.zeros();
      for (int r = 0; r <= b; r++){
        if (min(seq_mat.row(it+1) == unq_seq_mat.row(r)) == 1){
          match(r) = 1;
          unq_seq_mat_count(r,0) = unq_seq_mat_count(r,0)+1; 
        }
      }
      if (accu(match) == 0){
        b = b+1;
        unq_seq_mat.insert_rows(b, seq.t());
        arma::mat tempC (1, 1);
        tempC.ones();
        unq_seq_mat_count.insert_rows(b, tempC.row(0));
      }
    }else{
      unq_seq_mat_count(b,0) = unq_seq_mat_count(b,0)+1; 
    }
  }
  int num_unq_seq_mat = unq_seq_mat.n_rows;
  arma::vec Pi (num_unq_seq_mat);
  for (int l = 0; l < num_unq_seq_mat; l++){
    Pi(l) = unq_seq_mat_count(l,0)/itr;
  }
  
  // Rcout << "The value of itr : " << itr << "\n";
  // Rcout << "The value of unq_seq_mat_count : " << accu(unq_seq_mat_count) << "\n";
  // Rcout << "The value of seq_mat num row : " << seq_mat.n_rows << "\n";
  // Rcout << "The value of cum_accept : " << cum_accept << "\n";
  
  List ret;
  ret["Pi"] = Pi;
  ret["unq_seq_mat"] = unq_seq_mat+1;
  
  return ret;  
  
}


List MStep(arma::cube x, arma::vec Pi, arma::vec rho, arma::vec Sigma_inter, arma::mat unq_seq_mat, int n){
  int K = rho.size();
  arma::mat A = {{0,1},{1,0}};
  arma::mat I2(2,2);
  arma::vec O2 (2);
  arma::vec tau (K);
  tau.zeros();
  arma::cube Mu (K,K,2);
  Mu.zeros();
  arma::cube Sigma_inv_k (2,2,K);
  arma::field<arma::mat> Sigma(K, K);
  arma::mat SigmaMat (2,2);
  SigmaMat.zeros();
  unq_seq_mat = unq_seq_mat - 1;
  int num_unq_seq_mat = unq_seq_mat.n_rows;
  
  for (int k = 0; k < K; k++){
    for (int l = 0; l < num_unq_seq_mat; l++){
      int ctr = 0;
      for (int i = 0; i < (n-1); i++){
        for (int j = (i+1); j < n; j++){
          for (int m = 0; m < K; m++){
            if ((unq_seq_mat(l,i) == k) && (unq_seq_mat(l,j) == m)){
              ctr = ctr+1;
            }
            if ((unq_seq_mat(l,i) == m) && (unq_seq_mat(l,j) == k)){
              ctr = ctr+1;
            }
          }
        }
      }
      tau(k) = tau(k) + Pi(l)*ctr; 
    }
    tau(k) = tau(k)/(n*(n-1));
  } 
  
  // Rcout << "The value of tau : " << tau << "\n";
  
  
  for (int k = 0; k < K; k++){
    Sigma_inv_k.slice(k) = (I2.eye() - rho(k)*A)/(Sigma_inter(k)*(1-pow(rho(k),2)));
  }
  
  // Rcout << "The value of Sigma_inv_k : " << Sigma_inv_k << "\n";
  
  for (int k = 0; k < K; k++){
    for (int m = k; m < K; m++){
      double den = 0;
      if (k == m){
        double num = 0;
        for (int t = 0; t < num_unq_seq_mat; t++){
          int ctr = 0;
          // arma::vec xSum (2);
          // xSum.zeros();
          double xSum = 0;
          for (int i = 0; i < (n-1); i++){
            for (int j = (i+1); j < n; j++){
              arma::vec xVec (2);
              for (int l=0; l < 2; l++){
                xVec(l) = x(i,j,l);
              } 
              if ((unq_seq_mat(t,i) == k) && (unq_seq_mat(t,j) == m)){
                ctr = ctr+1;
                xSum = xSum + as_scalar(trans(O2.ones())*Sigma_inv_k.slice(k)*xVec);
              }
            }
          }
          num = num + Pi(t)*xSum;
          den = den + Pi(t)*ctr;
        }
        den = den*as_scalar(trans(O2.ones())*Sigma_inv_k.slice(k)*O2.ones());
        double mu_kk = num/den;
        //double mu_kk = as_scalar((trans(O2.ones())*Sigma_inv_k.slice(k)*num)/(trans(O2.ones())*Sigma_inv_k.slice(k)*O2.ones()*den));
        Mu(k,m,0) = mu_kk;
        Mu(k,m,1) = mu_kk;
      } else{
        arma::vec num (2);
        num.zeros();
        for (int t = 0; t < num_unq_seq_mat; t++){
          int ctr = 0;
          arma::vec xSum (2);
          xSum.zeros();
          for (int i = 0; i < (n-1); i++){
            for (int j = (i+1); j < n; j++){
              arma::vec xVec (2);
              for (int l=0; l < 2; l++){
                xVec(l) = x(i,j,l);
              } 
              if ((unq_seq_mat(t,i) == k) && (unq_seq_mat(t,j) == m)){
                ctr = ctr+1;
                xSum = xSum + xVec;
              }
              if ((unq_seq_mat(t,i) == m) && (unq_seq_mat(t,j) == k)){
                ctr = ctr+1;
                xSum = xSum + A*xVec;
              }
            }
          }
          num = num + Pi(t)*xSum;
          den = den + Pi(t)*ctr;
        }
        arma::vec Mu_AA = num/den;
        Mu(k,m,0) = Mu_AA(0);
        Mu(k,m,1) = Mu_AA(1);
      }
    }
  }
  
  for (int k = 1; k < K; k++){
    for (int m = 0; m < k; m++){
      arma::vec MuVec = {Mu(m,k,0), Mu(m,k,1)};
      arma::vec Mu_AA = A*MuVec;
      Mu(k,m,0) = Mu_AA(0);
      Mu(k,m,1) = Mu_AA(1);
    }
  }
  
  // Rcout << "The value of Mu : " << Mu << "\n";
  
  for (int k = 0; k < K; k++){
    arma::vec Mu_kk = {Mu(k,k,0), Mu(k,k,1)};
    rho(k) = unirootC2(Pi, Sigma_inter(k), x, Mu_kk, unq_seq_mat, k, n, -0.9999, 0.9999, 0.0001);
  }
  
  // Rcout << "The value of rho : " << rho << "\n";
  
  for (int k = 0; k < K; k++){
    for (int m = k; m < K; m++){
      arma::mat num (2,2);
      num.zeros();
      double den = 0;
      double S_kk_num = 0;
      if (k == m){
        for (int t = 0; t < num_unq_seq_mat; t++){
          int ctr = 0;
          double xSum = 0;
          for (int i = 0; i < (n-1); i++){
            for (int j = (i+1); j < n; j++){
              arma::vec xVec (2);
              arma::vec MuVec (2);
              for (int l=0; l < 2; l++){
                xVec(l) = x(i,j,l);
                MuVec(l) = Mu(k,m,l);
              } 
              if ((unq_seq_mat(t,i) == k) && (unq_seq_mat(t,j) == m)){
                ctr = ctr+1;
                xSum = xSum + as_scalar(trans(xVec - MuVec)*(I2.eye() - rho(k)*A)*(xVec - MuVec));
              }
            }
          }
          S_kk_num = S_kk_num + Pi(t)*xSum;
          den = den + Pi(t)*ctr;
        }
        Sigma_inter(k) = S_kk_num/(2*(1 - pow(rho(k), 2))*den);
        SigmaMat = Sigma_inter(k)*(I2.eye() + rho(k)*A);
      } else{
        for (int t = 0; t < num_unq_seq_mat; t++){
          int ctr = 0;
          arma::mat xSum (2,2);
          xSum.zeros();
          for (int i = 0; i < (n-1); i++){
            for (int j = (i+1); j < n; j++){
              arma::vec xVec (2);
              arma::vec MuVec (2);
              for (int l=0; l < 2; l++){
                xVec(l) = x(i,j,l);
                MuVec(l) = Mu(k,m,l);
              } 
              if ((unq_seq_mat(t,i) == k) && (unq_seq_mat(t,j) == m)){
                ctr = ctr+1;
                xSum = xSum + (xVec - MuVec)*trans(xVec - MuVec);
              }
              if ((unq_seq_mat(t,i) == m) && (unq_seq_mat(t,j) == k)){
                ctr = ctr+1;
                xSum = xSum + (A*xVec - MuVec)*trans(A*xVec - MuVec);
              }
            }
          }
          num = num + Pi(t)*xSum;
          den = den + Pi(t)*ctr;
        }
        SigmaMat = num/den;
      }
      Sigma(k,m) = SigmaMat;
    }
  }
  
  for (int k = 1; k < K; k++){
    for (int m = 0; m < k; m++){
      arma::mat SigmaMat = Sigma(m,k);
      Sigma(k,m) = A*SigmaMat*trans(A);
    }
  }
  
  // Rcout << "The value of Sigma : " << Sigma << "\n";
  
  List ret;
  ret["tau"] = tau;
  ret["Sigma_inter"] = Sigma_inter;
  ret["rho"] = rho;
  ret["Mu"] = Mu;
  ret["Sigma"] = Sigma;
  
  return ret;
}


double logLSeq(arma::cube x, arma::vec tau, arma::cube Mu, arma::field<arma::mat> Sigma, arma::mat unq_seq_mat, int n){
  int K = tau.size();
  int num_unq_seq_mat = unq_seq_mat.n_rows;
  double ll1 = 0;
  arma::vec ll2 (num_unq_seq_mat);
  ll2.zeros();
  unq_seq_mat = unq_seq_mat - 1;
  
  for (int t = 0; t < num_unq_seq_mat ; t++){
    for (int i = 0; i < (n-1); i++){
      for (int j = (i+1); j < n; j++){
        arma::rowvec xVec (2);
        arma::rowvec MuVec (2);
        for (int l=0; l < 2; l++){
          xVec(l) = x(i,j,l);
          MuVec(l) = Mu(unq_seq_mat(t,i),unq_seq_mat(t,j),l);
        } 
        arma::mat SigmaMat = Sigma(unq_seq_mat(t,i), unq_seq_mat(t,j));
        double dens = dmvnorm(xVec, MuVec, SigmaMat, TRUE);
        ll2(t) = ll2(t) + dens;
        for (int k = 0; k < K; k++){
          for (int m = 0; m < K; m++){
            if ((unq_seq_mat(t,i) == k) && (unq_seq_mat(t,j) == m)){
              ll2(t) = ll2(t) + log(tau(k)) + log(tau(m));
            }
          }
        }
      }
    }
  }
  
  double add_const = (-1)*max(ll2);
  for (int t = 0; t < num_unq_seq_mat ; t++){
    ll1 = ll1 + exp(ll2(t) + add_const);
  }
  double ll = log(ll1) - add_const;
  return ll;
}


List EMC(arma::cube x, arma::vec tau, arma::cube Mu, arma::field<arma::mat> Sigma, double eps, int burn, int itr, arma::vec seq, int max_itr, int n){
  int K = tau.size();
  arma::vec Pi ;
  arma::vec Sigma_inter (K);
  arma::vec rho (K);
  for (int k = 0; k < K; k++){
    arma::mat SigmaMat = Sigma(k,k);
    Sigma_inter(k) = SigmaMat(0,0);
    rho(k) = SigmaMat(0,1)/SigmaMat(0,0);
  }
  
  arma::mat unq_seq_mat (1,n);
  unq_seq_mat.row(0) = seq.t();
  
  int b = 0;
  double ll_old = arma::datum::inf;
  ll_old = (-1)*ll_old;
  double ll = logLSeq(x, tau, Mu, Sigma, unq_seq_mat, n);
  // Rcout << "***** ****************** ***** \n";
  // Rcout << "***** Sequence EM begins ***** \n";
  // Rcout << "***** ****************** ***** \n";
  //Rcout << "The value of b : " << b << "\n";
  // Rcout << "The value of tau : " << tau << "\n";
  // Rcout << "The value of Sumtau : " << accu(tau) << "\n";
  // Rcout << "The value of Mu : " << Mu << "\n";
  // Rcout << "The value of Sigma : " << Sigma << "\n";
  //Rcout << "The value of ll : " << ll << "\n";
  int stop = 0;
  
  while (stop == 0){
    b++ ;
    if (b > max_itr) {
      stop = 1;
      break;
    };
    double ll_diff = ll - ll_old;
    double ll_ratio = absDbl(ll_diff) / absDbl(ll);
    if (ll_ratio < eps) {
      stop = 1;
      break;
    };
    ll_old = ll;
    
    if (b == 2){
      burn = burn/5;
    }
    
    List E = EStep(x, tau, Mu, Sigma, seq, burn, itr, n);
    arma::vec Pi = E["Pi"];
    arma::mat unq_seq_mat = E["unq_seq_mat"];
    // int num_distinct_seq = unq_seq_mat.n_rows;
    
    // NumericVector Pi_temp (num_distinct_seq);
    // for (int l = 0; l < num_distinct_seq; l++){
    //   Pi_temp(l) = Pi(l);
    // }
    // NumericMatrix multinom_seq = multinom_r_cpp_call(Pi_temp, 1, 1);
    // for (int l = 0; l < num_distinct_seq; l++){
    //   if (multinom_seq(l,0) == 1){
    //     seq = trans(unq_seq_mat.row(l));
    //   }
    // }
    
    seq = trans(unq_seq_mat.row(index_max(Pi)));
    // Rcout << "The value of SumPi : " << accu(Pi) << "\n";
    
    List M = MStep(x, Pi, rho, Sigma_inter, unq_seq_mat, n);
    arma::vec Mtau = M["tau"];
    arma::cube MMu = M["Mu"];
    arma::field<arma::mat> MS = M["Sigma"];
    arma::field<arma::mat> MSigma(K, K);
    for (int k = 0; k < K; k++){
      for (int m = 0; m < K; m++){
        MSigma(k,m) =  MS(k+m*K,0);
      }
    }
    arma::vec MSigma_inter = M["Sigma_inter"];
    arma::vec Mrho = M["rho"];
    
    tau = Mtau;
    Mu = MMu;
    Sigma = MSigma;
    rho = Mrho;
    Sigma_inter = MSigma_inter;
    // Rcout << "The value of tau : " << tau << "\n";
    // Rcout << "The value of rho : " << rho << "\n";
    // Rcout << "The value of Sigma_inter : " << Sigma_inter << "\n";
    // Rcout << "The value of Sumtau : " << accu(tau) << "\n";
    // Rcout << "The value of Mu : " << Mu << "\n";
    // Rcout << "The value of Sigma : " << Sigma << "\n";
    
    // double ll_indep = logLindep(x, tau, Mu, Sigma, n);
    ll = logLSeq(x, tau, Mu, Sigma, unq_seq_mat, n);
    //Rcout << "The value of b : " << b << "\n";
    //Rcout << "The value of ll_indep : " << ll_indep << "\n";
    //Rcout << "The value of ll : " << ll << "\n";
  }
  
  int M = (K - 1) + K*K + 2*K + 3*K*(K-1)/2;
  double BIC = -2 * ll + M * log(n*(n-1)/2);
  double AIC = -2 * ll + M * 2;
  
  List ret;
  ret["Pi"] = Pi;
  ret["ll"] = ll;
  ret["tau"] = tau;
  ret["Mu"] = Mu;
  ret["Sigma"] = Sigma;
  ret["seq"] = seq;
  ret["BIC"] = BIC;
  ret["AIC"] = AIC;
  ret["id"] = seq;
  
  return ret;
}


List EM_initiate(arma::cube x, int K, double sigma_mult, int n, int sid){
  
  set_seed(sid);
  arma::mat A = {{0,1},{1,0}};
  arma::vec ini_tau (K);
  NumericVector ini_tau1 (K);
  arma::cube ini_Mu (K,K,2);
  ini_Mu.zeros();
  arma::field<arma::mat> ini_Sigma(K, K);
  arma::mat unq_seq_mat (1,n);
  
  for (int k = 0; k < K; k++){
    ini_tau(k) = 1/double(K);
    ini_tau1(k) = 1/double(K);
  }
  
  // Rcout << "ini_tau : " << ini_tau << "\n";
  
  NumericMatrix multinom_seq = multinom_r_cpp_call(ini_tau1, n, 1);
  arma::vec input_seq (n);
  for (int i = 0; i < n; i++){
    for (int k = 0; k < K; k++){
      if (multinom_seq(k,i) == 1){
        input_seq(i) = k+1;
      }
    }
  }
  unq_seq_mat.row(0) = input_seq.t();
  
  // Rcout << "input_seq : " << input_seq.t() << "\n";
  
  for (int k = 0; k < K; k++){
    for (int m = 0; m < K; m++){
      int ctr = 0;
      for (int i = 0; i < (n-1); i++){
        if (input_seq(i) == (k+1)){
          for (int j = (i+1); j < n; j++){
            if (input_seq(j) == (m+1)){
              ctr = ctr+1;
              if (k == m){
                ini_Mu(k,m,0) = ini_Mu(k,m,0) + x(i,j,0) + x(i,j,1);
                ini_Mu(k,m,1) = ini_Mu(k,m,1) + x(i,j,0) + x(i,j,1);
              }else{
                ini_Mu(k,m,0) = ini_Mu(k,m,0) + x(i,j,0);
                ini_Mu(k,m,1) = ini_Mu(k,m,1) + x(i,j,1);
              }
            }
          }
        }
      }
      if (k == m){
        ini_Mu(k,m,0) = ini_Mu(k,m,0)/(2*ctr);
        ini_Mu(k,m,1) = ini_Mu(k,m,1)/(2*ctr);
      }else{
        ini_Mu(k,m,0) = ini_Mu(k,m,0)/ctr;
        ini_Mu(k,m,1) = ini_Mu(k,m,1)/ctr;
      }
    }
  }
  
  for (int k = 0; k < K; k++){
    for (int m = k; m < K; m++){
      ini_Sigma(k,m) = sigma_mult*arma::eye(2,2);
    }
  }
  
  for (int k = 1; k < K; k++){
    for (int m = 0; m < k; m++){
      arma::vec MuVec = {ini_Mu(m,k,0), ini_Mu(m,k,1)};
      arma::vec Mu_AA = A*MuVec;
      ini_Mu(k,m,0) = Mu_AA(0);
      ini_Mu(k,m,1) = Mu_AA(1);
      ini_Sigma(k,m) = sigma_mult*arma::eye(2,2);
    }
  }
  
  // Rcout << "ini_Mu : " << ini_Mu << "\n";
  // Rcout << "ini_Sigma : " << ini_Sigma << "\n";
  // Rcout << "ini_Psi : " << ini_Psi << "\n";
  
  double logl = logLSeq(x, ini_tau, ini_Mu, ini_Sigma, unq_seq_mat, n);
  
  // Rcout << "logl : " << logl << "\n";
  
  List ret;
  ret["input_seq"] = input_seq;
  ret["ll"] = logl;
  ret["tau"] = ini_tau;
  ret["Mu"] = ini_Mu;
  ret["Sigma"] = ini_Sigma;
  
  return ret;
}

List netEM_uni(arma::cube x, int K, double eps, int num_rand_start, int num_run_smallEM, int max_itr_smallEM, int burn, int MCMC_itr, double sigma_mult, int alpha){
  int n = x.n_rows;
  arma::vec rand_st_seed_no (num_rand_start);
  arma::vec rand_st_LL (num_rand_start);
  arma::vec small_EM_LL (num_run_smallEM);
  arma::vec seed_no (num_run_smallEM);
  arma::imat small_EM_id(num_run_smallEM, K);
  
  for (int it = 0; it < num_rand_start; it++){
    rand_st_seed_no(it) = it+alpha;
    int sid = it+alpha;
    //Rcout << "sid : " << sid << "\n";
    List EM_ini = EM_initiate(x, K, sigma_mult, n, sid);
    rand_st_LL(it) =  EM_ini["ll"];
    //Rcout << "rand_st_LL : " << rand_st_LL(it) << "\n";
  }
  
  // change non-finite elements to zero
  //A.elem( find_nonfinite(A) ).zeros();
  rand_st_LL.replace(arma::datum::nan, -1000000000);
  //Rcout << "rand_st_LL : " << rand_st_LL << "\n";
  
  arma::uvec LL_sort_ind = sort_index(rand_st_LL, "descend");
  
  for (int it = 0; it < num_run_smallEM; it++){
    
    seed_no(it) = LL_sort_ind(it)+alpha;
    
    int sid = LL_sort_ind(it)+alpha;
    // Rcout << "sid : " << sid << "\n";
    List EM_ini = EM_initiate(x, K, sigma_mult, n, sid);
    
    arma::vec ini_tau = EM_ini["tau"];
    arma::vec input_seq = EM_ini["input_seq"];
    arma::cube ini_Mu = EM_ini["Mu"];
    arma::field<arma::mat> ini_S = EM_ini["Sigma"];
    arma::field<arma::mat> ini_Sigma(K, K);
    for (int k = 0; k < K; k++){
      for (int m = 0; m < K; m++){
        ini_Sigma(k,m) =  ini_S(k+m*K,0);
      }
    }
    // Rcout << "***** *************** ***** \n";
    // Rcout << "***** Short EM begins ***** \n";
    // Rcout << "***** Iteration # : " << it << "\n";
    // Rcout << "***** *************** ***** \n";
    
    int max_itr = max_itr_smallEM;
    List EMR = EMC(x, ini_tau, ini_Mu, ini_Sigma, eps, burn, MCMC_itr, input_seq, max_itr, n);
    
    arma::vec Mtau = EMR["tau"];
    arma::cube MMu = EMR["Mu"];
    arma::field<arma::mat> MS = EMR["Sigma"];
    arma::field<arma::mat> MSigma(K, K);
    for (int k = 0; k < K; k++){
      for (int m = 0; m < K; m++){
        MSigma(k,m) =  MS(k+m*K,0);
      }
    }
    
    arma::ivec Mid = EMR["id"];
    arma::ivec Mid_count (K);
    Mid_count.zeros();
    //Rcout << "id : " << trans(Mid) << "\n";
    for (int i = 0; i < n; i++){
      for (int k = 0; k < K; k++){
        if(Mid(i) == (k+1)){
          Mid_count(k) = Mid_count(k) + 1;
        }
      }
    }
    //Rcout << "ID distribution : " << trans(Mid_count) << "\n";
    small_EM_id.row(it) = trans(Mid_count);
    small_EM_LL(it) = EMR["ll"];
  }
  
  int max_ind = index_max(small_EM_LL);
  // double max_ll = max(small_EM_LL);
  arma::imat small_EM_best_id = small_EM_id.row(max_ind);
  
  // Rcout << "Small EM LL : " << trans(small_EM_LL) << "\n";
  // Rcout << "Max LL : " << max_ll << "\n";
  // Rcout << "Selected Iteration # : " << max_ind << "\n";
  // Rcout << "small_EM_best_id : " << trans(small_EM_best_id) << "\n";
  
  int sid = seed_no(max_ind);
  // Rcout << "sid : " << sid << "\n";
  List EM_ini = EM_initiate(x, K, sigma_mult, n, sid);
  
  arma::vec ini_tau = EM_ini["tau"];
  arma::vec input_seq = EM_ini["input_seq"];
  arma::cube ini_Mu = EM_ini["Mu"];
  arma::field<arma::mat> ini_S = EM_ini["Sigma"];
  arma::field<arma::mat> ini_Sigma(K, K);
  for (int k = 0; k < K; k++){
    for (int m = 0; m < K; m++){
      ini_Sigma(k,m) =  ini_S(k+m*K,0);
    }
  }
  
  // Rcout << "***** ************** ***** \n";
  // Rcout << "***** Long EM begins ***** \n";
  // Rcout << "***** ************** ***** \n";
  int max_itr = 50;
  List EMR = EMC(x, ini_tau, ini_Mu, ini_Sigma, eps, burn, MCMC_itr, input_seq, max_itr, n);
  
  arma::vec Mtau = EMR["tau"];
  arma::cube MMu = EMR["Mu"];
  arma::field<arma::mat> MS = EMR["Sigma"];
  arma::field<arma::mat> MSigma(K, K);
  for (int k = 0; k < K; k++){
    for (int m = 0; m < K; m++){
      MSigma(k,m) =  MS(k+m*K,0);
    }
  }
  double ll = EMR["ll"];
  
  double M = (K - 1) + K*K + 2*K + K*(K-1) + 0.5*K*(K-1);
  double BIC = -2 * ll  + M * log(n*(n-1)/2);
  
  List ret;
  ret["Pi"] = EMR["Pi"];
  ret["ll"] = EMR["ll"];
  ret["tau"] = EMR["tau"];
  ret["Mu"] = EMR["Mu"];
  ret["Sigma"] = EMR["Sigma"];
  ret["BIC"] = EMR["BIC"];
  ret["id"] = EMR["id"];
  ret["BIC"] = BIC;
  
  return ret;
  
}  




//' Returns the EM object for unilayer network
//'
//' @param x multiple network
//' @param K number of clusters
//' @param eps epsilon for convergence
//' @param num_rand_start number of random starts
//' @param num_run_smallEM number of runs for small EM
//' @param max_itr_smallEM maximum number of runs for small EM
//' @param burn number of runs for burn for Metropolis Hastings
//' @param MCMC_itr number of runs for Metropolis Hastings iterations
//' @param sigma_mult scaling multiplier for Sigma matrix
//' @param alpha seed provided by the user
//' @return EM object
//' @export
// [[Rcpp::export]]
List netEM_unilayer(arma::cube x, int K, double eps, int num_rand_start, int num_run_smallEM, int max_itr_smallEM, int burn, int MCMC_itr, double sigma_mult, int alpha){
  
  int stop = 0;
  List ret;
  
  while (stop == 0){
    
    if (K < 1){
      Rcout << "Wrong number of mixture components ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (eps <= 0){
      Rcout << "Wrong value of eps ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (num_rand_start < 1){
      Rcout << "Wrong number of random restarts ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (num_run_smallEM < 1){
      Rcout << "Wrong number of small EM ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (max_itr_smallEM < 1){
      Rcout << "Wrong number of iterations for small EM ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (burn < 1){
      Rcout << "Wrong number of burns ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (MCMC_itr < 1){
      Rcout << "Wrong number of MCMC iterations ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (sigma_mult <= 0){
      Rcout << "Wrong value for Sigma scale multiplier ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (alpha < 0){
      Rcout << "Wrong value for seed ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    
    int n = x.n_rows;
    for (int i = 0; i < n; i++){
      for (int j = 0; j < (i+1); j++){
        for (int l=0; l < 2; l++){
          if (x(i,j,l) != 0){
            stop = 1;
            break;
          }
        } 
      }
    }
    
    if (stop == 1){
      Rcout << "Wrong entry in network data ...\n";
      ret["Status"] = "Incorrect data";
      break;
    }
    
    ret = netEM_uni(x, K, eps, num_rand_start, num_run_smallEM, max_itr_smallEM, burn, MCMC_itr, sigma_mult, alpha);
    stop = 1;
  }
  
  
  return ret;
}


double logLindep_mult(arma::field<arma::mat> x, arma::vec tau, arma::field<arma::mat> Mu, arma::field<arma::mat> Sigma, arma::field<arma::mat> Psi, int n, int p) {
  int K = tau.size();
  double ll = 0;
  double ll2 = 0;
  double dens = 0;
  
  for (int i = 0; i < (n-1); i++){
    for (int j = (i+1); j < n; j++){
      ll2 = 0;
      arma::mat xMat = x(i,j);
      arma::rowvec xVec (2*p);
      for (int l=0; l < 2*p; l++){
        int q = floor(l/2);
        int r = l - 2*q;
        xVec(l) = xMat(r,q);
      } 
      for (int k = 0; k < K; k++){
        for (int m = 0; m < K; m++){
          arma::mat MuMat = Mu(k,m);
          arma::rowvec MuVec (2*p);
          for (int l=0; l < 2*p; l++){
            int q = floor(l/2);
            int r = l - 2*q;
            MuVec(l) = MuMat(r,q);
          }
          arma::mat SigmaMat = Sigma(k,m);
          arma::mat PsiMat = Psi(k,m);
          arma::mat PsiSigmaKron = kron(PsiMat, SigmaMat);
          dens = dmvnorm(xVec, MuVec, PsiSigmaKron, FALSE);
          ll2 = ll2 + tau(k)*tau(m)*dens;
        }
      }
      ll = ll + log(ll2);
    }
  }
  return ll;
}


List ll_seq_mult(arma::field<arma::mat> x, arma::vec tau, arma::field<arma::mat> Mu, arma::field<arma::mat> Sigma, arma::field<arma::mat> Psi, arma::vec seq, int n, int p){
  int K = tau.size();
  double ll2 = 0;
  seq = seq - 1;
  for (int i = 0; i < (n-1); i++){
    for (int j = (i+1); j < n; j++){
      arma::mat xMat = x(i,j);
      arma::mat MuMat = Mu(seq(i),seq(j));
      arma::rowvec xVec (2*p);
      arma::rowvec MuVec (2*p);
      for (int l=0; l < 2*p; l++){
        int q = floor(l/2);
        int r = l - 2*q;
        xVec(l) = xMat(r,q);
        MuVec(l) = MuMat(r,q);
      } 
      arma::mat SigmaMat = Sigma(seq(i),seq(j));
      arma::mat PsiMat = Psi(seq(i),seq(j));
      arma::mat PsiSigmaKron = kron(PsiMat, SigmaMat);
      double dens = dmvnorm(xVec, MuVec, PsiSigmaKron, TRUE);
      ll2 = ll2 + dens;
      for (int k = 0; k < K; k++){
        for (int m = 0; m < K; m++){
          if ((seq(i) == k) && (seq(j) == m)){
            ll2 = ll2 + log(tau(k)) + log(tau(m));
          }
        }
      }
    }
  }
  
  double likelihood = exp(ll2);
  
  List ret;
  ret["loglikelihood"] = ll2;
  ret["likelihood"] = likelihood;
  
  return ret;
}


List EStep_mult(arma::field<arma::mat> x, arma::vec tau, arma::field<arma::mat> Mu, arma::field<arma::mat> Sigma, arma::field<arma::mat> Psi, arma::vec seq, int burn, int itr, int n, int p){
  int K = tau.size();
  arma::vec lbl (K);
  for (int k = 0; k < K; k++){
    lbl(k) = k;
  }
  seq = seq - 1;
  arma::mat seq_mat (itr, n);
  seq_mat.row(0) = seq.t();
  arma::mat count (itr, 1);
  count.ones();
  arma::mat unq_seq_mat;
  unq_seq_mat.insert_rows(0, seq_mat.row(0));
  arma::mat unq_seq_mat_count;
  unq_seq_mat_count.insert_rows(0, count.row(0));
  
  int b = 0;
  for (int ib = 0; ib < burn; ib++){
    int pos = as_scalar(arma::randi<arma::rowvec>(1, arma::distr_param(0,(n-1))));
    arma::uvec num_posVec = find(seq == seq(pos));
    int num_pos = num_posVec.size();
    if (num_pos > 3){
      arma::uword q = arma::conv_to<arma::uword>::from(arma::find(lbl == seq(pos)));
      arma::vec choice = lbl;
      choice.shed_row(q);
      arma::vec WSS (K);
      WSS.zeros();
      for (int k = 0; k < K; k++){
        arma::vec alt_seq = seq;
        alt_seq(pos) = k;
        for (int i = 0; i < (n-1); i++){
          for (int j = (i+1); j < n; j++){
            arma::mat xMat = x(i,j);
            arma::mat MuMat = Mu(seq(i),seq(j));
            arma::vec xVec (2*p);
            arma::vec MuVec (2*p);
            for (int l=0; l < 2*p; l++){
              int q = floor(l/2);
              int r = l - 2*q;
              xVec(l) = xMat(r,q);
              MuVec(l) = MuMat(r,q);
            } 
            WSS(k) = WSS(k) + EuclDistVec(xVec, MuVec);   
            
          }
        }
      }
      arma::vec WSS_candidate = WSS;
      arma::vec WSS_wo_choice = WSS;
      WSS_candidate.shed_row(q);
      arma::vec alt_seq = seq;
      alt_seq(pos) = choice(index_min(WSS_candidate));
      WSS_wo_choice.shed_row(choice(index_min(WSS_candidate)));
      arma::vec WSS_candidate_inv = 1/WSS_candidate;
      arma::vec WSS_wo_choice_inv = 1/WSS_wo_choice;
      double trans_prob_alt_input = (1.0/min(WSS_candidate))/accu(WSS_candidate_inv);
      double trans_prob_input_alt = (1.0/min(WSS_wo_choice))/accu(WSS_wo_choice_inv);
      List l1 = ll_seq_mult(x, tau, Mu, Sigma, Psi, alt_seq+1, n, p);
      List l2 = ll_seq_mult(x, tau, Mu, Sigma, Psi, seq+1, n, p);
      double ll1 = l1["loglikelihood"];
      double ll2 = l2["loglikelihood"];
      double alpha = (trans_prob_alt_input/trans_prob_input_alt)*exp(ll1-ll2);
      if (alpha > 1){
        alpha = 1;
      }
      double u = arma::randu<double>();
      // int accept = 0;
      // Rcout << "The value of seq : " << seq(pos) << "\n";
      if (u < alpha){
        // accept = 1;
        seq = alt_seq;
      }
    }
  }
  
  int cum_accept = 1;
  
  for (int it = 0; it < (itr-1); it++){
    int pos = as_scalar(arma::randi<arma::rowvec>(1, arma::distr_param(0,(n-1))));
    arma::uvec num_posVec = find(seq == seq(pos));
    int num_pos = num_posVec.size();
    if (num_pos > 3){
      arma::uword q = arma::conv_to<arma::uword>::from(arma::find(lbl == seq(pos)));
      arma::vec choice = lbl;
      choice.shed_row(q);
      arma::vec WSS (K);
      WSS.zeros();
      for (int k = 0; k < K; k++){
        arma::vec alt_seq = seq;
        alt_seq(pos) = k;
        for (int i = 0; i < (n-1); i++){
          for (int j = (i+1); j < n; j++){
            arma::mat xMat = x(i,j);
            arma::mat MuMat = Mu(seq(i),seq(j));
            arma::vec xVec (2*p);
            arma::vec MuVec (2*p);
            for (int l=0; l < 2*p; l++){
              int q = floor(l/2);
              int r = l - 2*q;
              xVec(l) = xMat(r,q);
              MuVec(l) = MuMat(r,q);
            }  
            WSS(k) = WSS(k) + EuclDistVec(xVec, MuVec);   
          }
        }
      }
      arma::vec WSS_candidate = WSS;
      arma::vec WSS_wo_choice = WSS;
      WSS_candidate.shed_row(q);
      arma::vec alt_seq = seq;
      alt_seq(pos) = choice(index_min(WSS_candidate));
      WSS_wo_choice.shed_row(choice(index_min(WSS_candidate)));
      arma::vec WSS_candidate_inv = 1/WSS_candidate;
      arma::vec WSS_wo_choice_inv = 1/WSS_wo_choice;
      double trans_prob_alt_input = (1.0/min(WSS_candidate))/accu(WSS_candidate_inv);
      double trans_prob_input_alt = (1.0/min(WSS_wo_choice))/accu(WSS_wo_choice_inv);
      List l1 = ll_seq_mult(x, tau, Mu, Sigma, Psi, alt_seq+1, n, p);
      List l2 = ll_seq_mult(x, tau, Mu, Sigma, Psi, seq+1, n, p);
      double ll1 = l1["loglikelihood"];
      double ll2 = l2["loglikelihood"];
      double alpha = (trans_prob_alt_input/trans_prob_input_alt)*exp(ll1-ll2);
      if (alpha > 1){
        alpha = 1;
      }
      double u = arma::randu<double>();
      // int accept = 0;
      if (u < alpha){
        // accept = 1;
        cum_accept = cum_accept + 1;
        seq = alt_seq;
      }
      
      seq_mat.row(it+1) = seq.t();
      
      arma::vec match (b+1);
      match.zeros();
      for (int r = 0; r <= b; r++){
        if (min(seq_mat.row(it+1) == unq_seq_mat.row(r)) == 1){
          match(r) = 1;
          unq_seq_mat_count(r,0) = unq_seq_mat_count(r,0)+1; 
        }
      }
      if (accu(match) == 0){
        b = b+1;
        unq_seq_mat.insert_rows(b, seq.t());
        arma::mat tempC (1, 1);
        tempC.ones();
        unq_seq_mat_count.insert_rows(b, tempC.row(0));
      }
    }else{
      unq_seq_mat_count(b,0) = unq_seq_mat_count(b,0)+1; 
    }
  }
  int num_unq_seq_mat = unq_seq_mat.n_rows;
  arma::vec Pi (num_unq_seq_mat);
  for (int l = 0; l < num_unq_seq_mat; l++){
    Pi(l) = unq_seq_mat_count(l,0)/itr;
  }
  
  // Rcout << "The value of itr : " << itr << "\n";
  // Rcout << "The value of unq_seq_mat_count : " << accu(unq_seq_mat_count) << "\n";
  // Rcout << "The value of seq_mat num row : " << seq_mat.n_rows << "\n";
  // Rcout << "The value of cum_accept : " << cum_accept << "\n";
  
  List ret;
  ret["Pi"] = Pi;
  ret["unq_seq_mat"] = unq_seq_mat+1;
  
  return ret;  
  
}


List MStep_mult(arma::field<arma::mat> x, arma::vec Pi, arma::field<arma::mat> Sigma, arma::field<arma::mat> Psi, arma::mat unq_seq_mat, int n, int K, int p){
  // int K = rho.size();
  arma::mat A = {{0,1},{1,0}};
  arma::mat I2(2,2);
  arma::vec O2 (2);
  arma::vec tau (K);
  tau.zeros();
  arma::field<arma::mat> Mu (K,K);
  arma::mat MuMat (2,p);
  MuMat.zeros();
  arma::vec rho (K);
  for (int k = 0; k < K; k++){
    arma::mat SigmaMat = Sigma(k,k);
    rho(k) = SigmaMat(0,1)/SigmaMat(0,0);
  }
  unq_seq_mat = unq_seq_mat - 1;
  int num_unq_seq_mat = unq_seq_mat.n_rows;
  
  for (int k = 0; k < K; k++){
    for (int l = 0; l < num_unq_seq_mat; l++){
      int ctr = 0;
      for (int i = 0; i < (n-1); i++){
        for (int j = (i+1); j < n; j++){
          for (int m = 0; m < K; m++){
            if ((unq_seq_mat(l,i) == k) && (unq_seq_mat(l,j) == m)){
              ctr = ctr+1;
            }
            if ((unq_seq_mat(l,i) == m) && (unq_seq_mat(l,j) == k)){
              ctr = ctr+1;
            }
          }
        }
      }
      tau(k) = tau(k) + Pi(l)*ctr; 
    }
    tau(k) = tau(k)/(n*(n-1));
  } 
  
  for (int k = 0; k < K; k++){
    for (int m = k; m < K; m++){
      double den = 0;
      if (k == m){
        arma::vec num (p);
        num.zeros();
        for (int t = 0; t < num_unq_seq_mat; t++){
          arma::vec Mu_A (p);
          Mu_A.zeros();
          double Mu_B = 0;
          for (int i = 0; i < (n-1); i++){
            for (int j = (i+1); j < n; j++){
              arma::mat xMat = x(i,j); 
              if ((unq_seq_mat(t,i) == k) && (unq_seq_mat(t,j) == m)){
                Mu_A = Mu_A + xMat.t()*(I2.eye() - rho(k)*A)*O2.ones();
                Mu_B = Mu_B + 2*(1-rho(k));
              }
            }
          }
          num = num + Pi(t)*Mu_A;
          den = den + Pi(t)*Mu_B;
        }
        arma::vec mu_kk = num/den;
        Mu(k,m) = O2.ones()*mu_kk.t();
      } else{
        arma::mat num (2,p);
        num.zeros();
        for (int t = 0; t < num_unq_seq_mat; t++){
          int ctr = 0;
          arma::mat Mu_A (2,p);
          Mu_A.zeros();
          for (int i = 0; i < (n-1); i++){
            for (int j = (i+1); j < n; j++){
              arma::mat xMat = x(i,j); 
              if ((unq_seq_mat(t,i) == k) && (unq_seq_mat(t,j) == m)){
                ctr = ctr+1;
                Mu_A = Mu_A + xMat;
              }
              if ((unq_seq_mat(t,i) == m) && (unq_seq_mat(t,j) == k)){
                ctr = ctr+1;
                Mu_A = Mu_A + A*xMat;
              }
            }
          }
          num = num + Pi(t)*Mu_A;
          den = den + Pi(t)*ctr;
        }
        Mu(k,m) = num/den;
        Mu(m,k) = A*Mu(k,m);
      }
    }
  }
  
  // Rcout << "The value of Mu : " << Mu << "\n";
  
  for (int k = 0; k < K; k++){
    for (int m = k; m < K; m++){
      arma::mat MuMat = Mu(k,m);
      arma::mat SigmaMat (2,2);
      arma::mat PsiMat = Psi(k,m);
      arma::mat inv_PsiMat = PsiMat.i();
      arma::mat num (2,2);
      num.zeros();
      if (k == m){
        double rho_k_num = 0;
        double rho_k_den = 0;
        for (int t = 0; t < num_unq_seq_mat; t++){
          double xSum = 0;
          double xxSum = 0;
          for (int i = 0; i < (n-1); i++){
            for (int j = (i+1); j < n; j++){
              arma::mat xMat = x(i,j);
              if ((unq_seq_mat(t,i) == k) && (unq_seq_mat(t,j) == m)){
                arma::mat temp_mat = (xMat - MuMat)*inv_PsiMat*trans(xMat - MuMat);
                xSum = xSum + trace(A*temp_mat);
                xxSum = xxSum + trace(temp_mat);
              }
            }
          }
          rho_k_num = rho_k_num + Pi(t)*xSum;
          rho_k_den = rho_k_den + Pi(t)*xxSum;
        }
        rho(k) = rho_k_num/rho_k_den;
        SigmaMat = (I2.eye() + rho(k)*A);
        Sigma(k,m) = SigmaMat;
      } else{
        for (int t = 0; t < num_unq_seq_mat; t++){
          arma::mat xSum (2,2);
          xSum.zeros();
          for (int i = 0; i < (n-1); i++){
            for (int j = (i+1); j < n; j++){
              arma::mat xMat = x(i,j);
              if ((unq_seq_mat(t,i) == k) && (unq_seq_mat(t,j) == m)){
                xSum = xSum + (xMat - MuMat)*inv_PsiMat*trans(xMat - MuMat);
              }
              if ((unq_seq_mat(t,i) == m) && (unq_seq_mat(t,j) == k)){
                xSum = xSum + (A*xMat - MuMat)*inv_PsiMat*trans(A*xMat - MuMat);
              }
            }
          }
          num = num + Pi(t)*xSum;
        }
        double detnum = det(num);
        SigmaMat = num/pow(detnum, 1/2);
        Sigma(k,m) = SigmaMat;
        Sigma(m,k) = A*SigmaMat*trans(A);
      }
    }
  }
  
  // Rcout << "The value of rho : " << rho << "\n";
  // Rcout << "The value of Sigma : " << Sigma << "\n";
  
  for (int k = 0; k < K; k++){
    for (int m = k; m < K; m++){
      arma::mat MuMat = Mu(k,m);
      arma::mat SigmaMat = Sigma(k,m);
      arma::mat inv_SigmaMat = SigmaMat.i();
      arma::mat num (p,p);
      num.zeros();
      double den = 0;
      for (int t = 0; t < num_unq_seq_mat; t++){
        int ctr = 0;
        arma::mat xSum (p,p);
        xSum.zeros();
        for (int i = 0; i < (n-1); i++){
          for (int j = (i+1); j < n; j++){
            arma::mat xMat = x(i,j);
            if ((unq_seq_mat(t,i) == k) && (unq_seq_mat(t,j) == m)){
              ctr = ctr+1;
              xSum = xSum + trans(xMat - MuMat)*inv_SigmaMat*(xMat - MuMat);
            }
            if ((unq_seq_mat(t,i) == m) && (unq_seq_mat(t,j) == k)){
              ctr = ctr+1;
              xSum = xSum + trans(A*xMat - MuMat)*inv_SigmaMat*(A*xMat - MuMat);
            }
          }
        }
        num = num + Pi(t)*xSum;
        den = den + Pi(t)*ctr;
      }
      arma::mat PsiMat = num/(2*den);
      Psi(k,m) = PsiMat;
      Psi(m,k) = PsiMat;
    }
  }
  // Rcout << "The value of Psi : " << Psi << "\n";
  
  List ret;
  ret["tau"] = tau;
  ret["rho"] = rho;
  ret["Mu"] = Mu;
  ret["Sigma"] = Sigma;
  ret["Psi"] = Psi;
  
  return ret;
}


double logLSeq_mult(arma::field<arma::mat> x, arma::vec tau, arma::field<arma::mat> Mu, arma::field<arma::mat> Sigma, arma::field<arma::mat> Psi, arma::mat unq_seq_mat, int n, int p){
  int K = tau.size();
  int num_unq_seq_mat = unq_seq_mat.n_rows;
  double ll1 = 0;
  arma::vec ll2 (num_unq_seq_mat);
  ll2.zeros();
  unq_seq_mat = unq_seq_mat - 1;
  
  for (int t = 0; t < num_unq_seq_mat ; t++){
    for (int i = 0; i < (n-1); i++){
      for (int j = (i+1); j < n; j++){
        arma::mat xMat = x(i,j);
        arma::mat MuMat = Mu(unq_seq_mat(t,i),unq_seq_mat(t,j));
        arma::rowvec xVec (2*p);
        arma::rowvec MuVec (2*p);
        for (int l=0; l < 2*p; l++){
          int q = floor(l/2);
          int r = l - 2*q;
          xVec(l) = xMat(r,q);
          MuVec(l) = MuMat(r,q);
        } 
        arma::mat SigmaMat = Sigma(unq_seq_mat(t,i),unq_seq_mat(t,j));
        arma::mat PsiMat = Psi(unq_seq_mat(t,i),unq_seq_mat(t,j));
        arma::mat PsiSigmaKron = kron(PsiMat, SigmaMat);
        double dens = dmvnorm(xVec, MuVec, PsiSigmaKron, TRUE);
        ll2(t) = ll2(t) + dens;
        for (int k = 0; k < K; k++){
          for (int m = 0; m < K; m++){
            if ((unq_seq_mat(t,i) == k) && (unq_seq_mat(t,j) == m)){
              ll2(t) = ll2(t) + log(tau(k)) + log(tau(m));
            }
          }
        }
      }
    }
  }
  
  double add_const = (-1)*max(ll2);
  for (int t = 0; t < num_unq_seq_mat ; t++){
    ll1 = ll1 + exp(ll2(t) + add_const);
  }
  double ll = log(ll1) - add_const;
  return ll;
}


List EM_initiate_mult(arma::field<arma::mat> x, int K, double sigma_mult, double psi_mult, int n, int p, int sid){
  
  set_seed(sid);
  arma::mat A = {{0,1},{1,0}};
  arma::vec ini_tau (K);
  NumericVector ini_tau1 (K);
  arma::field<arma::mat> ini_Mu(K, K);
  arma::field<arma::mat> ini_Sigma(K, K);
  arma::field<arma::mat> ini_Psi(K, K);
  arma::mat unq_seq_mat (1,n);
  
  arma::mat MuMat (2,p);
  
  for (int k = 0; k < K; k++){
    ini_tau(k) = 1/double(K);
    ini_tau1(k) = 1/double(K);
  }
  
  // Rcout << "ini_tau : " << ini_tau << "\n";
  
  NumericMatrix multinom_seq = multinom_r_cpp_call(ini_tau1, n, 1);
  arma::vec input_seq (n);
  for (int i = 0; i < n; i++){
    for (int k = 0; k < K; k++){
      if (multinom_seq(k,i) == 1){
        input_seq(i) = k+1;
      }
    }
  }
  unq_seq_mat.row(0) = input_seq.t();
  
  // Rcout << "input_seq : " << input_seq.t() << "\n";
  
  for (int k = 0; k < K; k++){
    for (int m = 0; m < K; m++){
      MuMat.zeros();
      int ctr = 0;
      for (int i = 0; i < (n-1); i++){
        if (input_seq(i) == (k+1)){
          for (int j = (i+1); j < n; j++){
            if (input_seq(j) == (m+1)){
              arma::mat xMat = x(i,j);
              ctr = ctr+1;
              for (int l=0; l < p; l++){
                if (k == m){
                  MuMat(0,l) = MuMat(0,l) + xMat(0,l) + xMat(1,l);
                  MuMat(1,l) = MuMat(1,l) + xMat(0,l) + xMat(1,l);
                }else{
                  MuMat(0,l) = MuMat(0,l) + xMat(0,l);
                  MuMat(1,l) = MuMat(1,l) + xMat(1,l);
                }
              }
            }
          }
        }
      }
      if (k == m){
        ini_Mu(k,m) = MuMat/(2*ctr);
      }else{
        ini_Mu(k,m) = MuMat/ctr;
      }
    }
  }
  
  for (int k = 0; k < K; k++){
    for (int m = k; m < K; m++){
      ini_Sigma(k,m) = sigma_mult*arma::eye(2,2);
      ini_Psi(k,m) = psi_mult*arma::eye(p,p);
    }
  }
  
  for (int k = 1; k < K; k++){
    for (int m = 0; m < k; m++){
      arma::mat Mu_A = ini_Mu(m,k);
      arma::mat Mu_AA = A*Mu_A;
      ini_Mu(k,m) = Mu_AA;
      ini_Sigma(k,m) = sigma_mult*arma::eye(2,2);
      ini_Psi(k,m) = psi_mult*arma::eye(p,p);
    }
  }
  
  // Rcout << "ini_Mu : " << ini_Mu << "\n";
  // Rcout << "ini_Sigma : " << ini_Sigma << "\n";
  // Rcout << "ini_Psi : " << ini_Psi << "\n";
  
  double logl = logLSeq_mult(x, ini_tau, ini_Mu, ini_Sigma, ini_Psi, unq_seq_mat, n, p);
  
  // Rcout << "logl : " << logl << "\n";
  
  List ret;
  ret["input_seq"] = input_seq;
  ret["ll"] = logl;
  ret["tau"] = ini_tau;
  ret["Mu"] = ini_Mu;
  ret["Sigma"] = ini_Sigma;
  ret["Psi"] = ini_Psi;
  
  return ret;
}


List EMC_mult(arma::field<arma::mat> x, arma::vec tau, arma::field<arma::mat> Mu, arma::field<arma::mat> Sigma, arma::field<arma::mat> Psi, double eps, int burn, int itr, arma::vec seq, int max_itr, int n, int p){
  int K = tau.size();
  arma::vec Pi ;
  arma::vec rho (K);
  for (int k = 0; k < K; k++){
    arma::mat SigmaMat = Sigma(k,k);
    rho(k) = SigmaMat(0,1)/SigmaMat(0,0);
  }
  
  arma::mat unq_seq_mat (1,n);
  unq_seq_mat.row(0) = seq.t();
  
  int b = 0;
  double ll_old = arma::datum::inf;
  ll_old = (-1)*ll_old;
  double ll = logLSeq_mult(x, tau, Mu, Sigma, Psi, unq_seq_mat, n, p);
  // Rcout << "***** ****************** ***** \n";
  // Rcout << "***** Sequence EM begins ***** \n";
  // Rcout << "***** ****************** ***** \n";
  // Rcout << "The value of b : " << b << "\n";
  // Rcout << "The value of tau : " << tau << "\n";
  // Rcout << "The value of Sumtau : " << accu(tau) << "\n";
  // Rcout << "The value of Mu : " << Mu << "\n";
  // Rcout << "The value of Sigma : " << Sigma << "\n";
  // Rcout << "The value of ll : " << ll << "\n";
  int stop = 0;
  
  while (stop == 0){
    b++ ;
    if (b > max_itr) {
      stop = 1;
      break;
    };
    double ll_diff = ll - ll_old;
    double ll_ratio = absDbl(ll_diff) / absDbl(ll);
    if (ll_ratio < eps) {
      stop = 1;
      break;
    };
    ll_old = ll;
    
    if (b == 2){
      burn = burn/5;
    }
    
    List E = EStep_mult(x, tau, Mu, Sigma, Psi, seq, burn, itr, n, p);
    
    arma::vec Pi = E["Pi"];
    arma::mat unq_seq_mat = E["unq_seq_mat"];
    // int num_distinct_seq = unq_seq_mat.n_rows;
    
    // NumericVector Pi_temp (num_distinct_seq);
    // for (int l = 0; l < num_distinct_seq; l++){
    //   Pi_temp(l) = Pi(l);
    // }
    // NumericMatrix multinom_seq = multinom_r_cpp_call(Pi_temp, 1, 1);
    // for (int l = 0; l < num_distinct_seq; l++){
    //   if (multinom_seq(l,0) == 1){
    //     seq = trans(unq_seq_mat.row(l));
    //   }
    // }
    
    seq = trans(unq_seq_mat.row(index_max(Pi)));
    // Rcout << "The value of SumPi : " << accu(Pi) << "\n";
    
    List M = MStep_mult(x, Pi, Sigma, Psi, unq_seq_mat, n, K, p);
    
    arma::vec Mtau = M["tau"];
    arma::field<arma::mat> MM = M["Mu"];
    arma::field<arma::mat> MMu(K, K);
    arma::field<arma::mat> MS = M["Sigma"];
    arma::field<arma::mat> MSigma(K, K);
    arma::field<arma::mat> MP = M["Psi"];
    arma::field<arma::mat> MPsi(K, K);
    for (int k = 0; k < K; k++){
      for (int m = 0; m < K; m++){
        MMu(k,m) =  MM(k+m*K,0);
        MSigma(k,m) =  MS(k+m*K,0);
        MPsi(k,m) =  MP(k+m*K,0);
      }
    }
    
    arma::vec Mrho = M["rho"];
    
    tau = Mtau;
    Mu = MMu;
    Sigma = MSigma;
    Psi = MPsi;
    rho = Mrho;
    // Rcout << "The value of tau : " << tau << "\n";
    // Rcout << "The value of rho : " << rho << "\n";
    // Rcout << "The value of Sumtau : " << accu(tau) << "\n";
    // Rcout << "The value of Mu : " << Mu << "\n";
    // Rcout << "The value of Sigma : " << Sigma << "\n";
    
    // double ll_indep = logLindep_mult(x, tau, Mu, Sigma, Psi, n, p);
    ll = logLSeq_mult(x, tau, Mu, Sigma, Psi, unq_seq_mat, n, p);
    // Rcout << "The value of b : " << b << "\n";
    // Rcout << "The value of ll_indep : " << ll_indep << "\n";
    // Rcout << "The value of ll : " << ll << "\n";
  }
  
  int M = (K - 1) + K*K + 2*K + 3*K*(K-1)/2 + K*p*(p-1)/2;
  double BIC = -2 * ll + M * log(n*(n-1)/2);
  double AIC = -2 * ll + M * 2;
  
  List ret;
  ret["Pi"] = Pi;
  ret["ll"] = ll;
  ret["tau"] = tau;
  ret["Mu"] = Mu;
  ret["Sigma"] = Sigma;
  ret["Psi"] = Psi;
  ret["seq"] = seq;
  ret["BIC"] = BIC;
  ret["AIC"] = AIC;
  ret["id"] = seq;
  
  return ret;
}


List netEM_mult(NumericVector y, int K, int p, double eps, int num_rand_start, int num_run_smallEM, int max_itr_smallEM, int burn, int MCMC_itr, double sigma_mult, double psi_mult, int n, int alpha){
  
  arma::field<arma::mat> x(n, n);
  
  for (int i = 0; i < (n-1); i++){
    for (int j = (i+1); j < n; j++){
      arma::mat xMat (2,p);
      int ctr = 0;
      for (int l = 0; l < p; l++){
        for (int k = 0; k < 2; k++){
          int d = ctr*pow(n,2) + j*n + i;
          xMat(k,l) = y(d);
          ctr = ctr+1;
        }
      }
      x(i,j) = xMat;
    }
  }
  
  arma::vec rand_st_seed_no (num_rand_start);
  arma::vec rand_st_LL (num_rand_start);
  arma::vec small_EM_LL (num_run_smallEM);
  arma::vec seed_no (num_run_smallEM);
  arma::imat small_EM_id(num_run_smallEM, K);
  
  for (int it = 0; it < num_rand_start; it++){
    rand_st_seed_no(it) = it+alpha;
    int sid = it+alpha;
    // Rcout << "sid : " << sid << "\n";
    List EM_ini = EM_initiate_mult(x, K, sigma_mult, psi_mult, n, p, sid);
    rand_st_LL(it) =  EM_ini["ll"];
    // Rcout << "rand_st_LL : " << rand_st_LL(it) << "\n";
  }
  rand_st_LL.replace(arma::datum::nan, -1000000000);
  arma::uvec LL_sort_ind = sort_index(rand_st_LL, "descend");
  
  for (int it = 0; it < num_run_smallEM; it++){
    
    seed_no(it) = LL_sort_ind(it)+alpha;
    
    int sid = LL_sort_ind(it)+alpha;
    // Rcout << "sid : " << sid << "\n";
    List EM_ini = EM_initiate_mult(x, K, sigma_mult, psi_mult, n, p, sid);
    
    arma::vec ini_tau = EM_ini["tau"];
    arma::vec input_seq = EM_ini["input_seq"];
    // Rcout << "ini_tau : " << ini_tau << "\n";
    arma::field<arma::mat> EMM = EM_ini["Mu"];
    arma::field<arma::mat> ini_Mu(K, K);
    arma::field<arma::mat> EMS = EM_ini["Sigma"];
    arma::field<arma::mat> ini_Sigma(K, K);
    arma::field<arma::mat> EMP = EM_ini["Psi"];
    arma::field<arma::mat> ini_Psi(K, K);
    for (int k = 0; k < K; k++){
      for (int m = 0; m < K; m++){
        ini_Mu(k,m) =  EMM(k+m*K,0);
        ini_Sigma(k,m) =  EMS(k+m*K,0);
        ini_Psi(k,m) =  EMP(k+m*K,0);
      }
    }
    
    // Rcout << "***** *************** ***** \n";
    // Rcout << "***** Short EM begins ***** \n";
    // Rcout << "***** Iteration # : " << it << "\n";
    // Rcout << "***** *************** ***** \n";
    
    int max_itr = max_itr_smallEM;
    List EMR = EMC_mult(x, ini_tau, ini_Mu, ini_Sigma, ini_Psi, eps, burn, MCMC_itr, input_seq, max_itr, n, p);
    
    arma::vec Mtau = EMR["tau"];
    arma::field<arma::mat> MM = EMR["Mu"];
    arma::field<arma::mat> MMu(K, K);
    arma::field<arma::mat> MS = EMR["Sigma"];
    arma::field<arma::mat> MSigma(K, K);
    arma::field<arma::mat> MP = EMR["Psi"];
    arma::field<arma::mat> MPsi(K, K);
    for (int k = 0; k < K; k++){
      for (int m = 0; m < K; m++){
        MMu(k,m) =  MM(k+m*K,0);
        MSigma(k,m) =  MS(k+m*K,0);
        MPsi(k,m) =  MP(k+m*K,0);
      }
    }
    
    arma::ivec Mid = EMR["id"];
    arma::ivec Mid_count (K);
    Mid_count.zeros();
    // Rcout << "id : " << trans(Mid) << "\n";
    for (int i = 0; i < n; i++){
      for (int k = 0; k < K; k++){
        if(Mid(i) == (k+1)){
          Mid_count(k) = Mid_count(k) + 1;
        }
      }
    }
    // Rcout << "ID distribution : " << trans(Mid_count) << "\n";
    small_EM_id.row(it) = trans(Mid_count);
    small_EM_LL(it) = EMR["ll"];
  }
  
  int max_ind = index_max(small_EM_LL);
  // double max_ll = max(small_EM_LL);
  arma::imat small_EM_best_id = small_EM_id.row(max_ind);
  
  // Rcout << "Small EM LL : " << trans(small_EM_LL) << "\n";
  // Rcout << "Max LL : " << max_ll << "\n";
  // Rcout << "Selected Iteration # : " << max_ind << "\n";
  // Rcout << "small_EM_best_id : " << trans(small_EM_best_id) << "\n";
  
  int sid = seed_no(max_ind);
  // Rcout << "sid : " << sid << "\n";
  List EM_ini = EM_initiate_mult(x, K, sigma_mult, psi_mult, n, p, sid);
  
  arma::vec ini_tau = EM_ini["tau"];
  arma::vec input_seq = EM_ini["input_seq"];
  // Rcout << "ini_tau : " << ini_tau << "\n";
  arma::field<arma::mat> EMM = EM_ini["Mu"];
  arma::field<arma::mat> ini_Mu(K, K);
  arma::field<arma::mat> EMS = EM_ini["Sigma"];
  arma::field<arma::mat> ini_Sigma(K, K);
  arma::field<arma::mat> EMP = EM_ini["Psi"];
  arma::field<arma::mat> ini_Psi(K, K);
  for (int k = 0; k < K; k++){
    for (int m = 0; m < K; m++){
      ini_Mu(k,m) =  EMM(k+m*K,0);
      ini_Sigma(k,m) =  EMS(k+m*K,0);
      ini_Psi(k,m) =  EMP(k+m*K,0);
    }
  }
  
  // Rcout << "***** ************** ***** \n";
  // Rcout << "***** Long EM begins ***** \n";
  // Rcout << "***** ************** ***** \n";
  int max_itr = 50;
  List EMR = EMC_mult(x, ini_tau, ini_Mu, ini_Sigma, ini_Psi, eps, burn, MCMC_itr, input_seq, max_itr, n, p);
  
  arma::vec Mtau = EMR["tau"];
  arma::field<arma::mat> MM = EMR["Mu"];
  arma::field<arma::mat> MMu(K, K);
  arma::field<arma::mat> MS = EMR["Sigma"];
  arma::field<arma::mat> MSigma(K, K);
  arma::field<arma::mat> MP = EMR["Psi"];
  arma::field<arma::mat> MPsi(K, K);
  for (int k = 0; k < K; k++){
    for (int m = 0; m < K; m++){
      MMu(k,m) =  MM(k+m*K,0);
      MSigma(k,m) =  MS(k+m*K,0);
      MPsi(k,m) =  MP(k+m*K,0);
    }
  }
  
  double ll_indep = logLindep_mult(x, Mtau, MMu, MSigma, MPsi, n, p);
  double ll = EMR["ll"];
  
  int M = (K - 1) + K*K + 2*K + K*(K-1) + 0.5*K*(K-1) + K*p*(p-1)/2;
  double BIC = -2 * ll  + M * log(n*(n-1)/2);
  
  List ret;
  ret["Pi"] = EMR["Pi"];
  ret["ll"] = EMR["ll"];
  ret["tau"] = EMR["tau"];
  ret["Mu"] = EMR["Mu"];
  ret["Sigma"] = EMR["Sigma"];
  ret["Psi"] = EMR["Psi"];
  ret["BIC"] = EMR["BIC"];
  ret["id"] = EMR["id"];
  ret["ll_indep"] = ll_indep;
  ret["BIC"] = BIC;
  
  return ret;
}


//' Returns the EM object for multilayer network
//'
//' @param y multiple network
//' @param K number of clusters
//' @param p number of layers
//' @param eps epsilon for convergence
//' @param num_rand_start number of random starts
//' @param num_run_smallEM number of runs for small EM
//' @param max_itr_smallEM maximum number of runs for small EM
//' @param burn number of runs for burn for Metropolis Hastings
//' @param MCMC_itr number of runs for Metropolis Hastings iterations
//' @param sigma_mult scaling multiplier for Sigma matrix
//' @param psi_mult scaling multiplier for Psi matrix
//' @param n number of nodes of the network
//' @param alpha seed provided by the user
//' @return EM object
//' @export
// [[Rcpp::export]]
List netEM_multilayer(NumericVector y, int K, int p, double eps, int num_rand_start, int num_run_smallEM, int max_itr_smallEM, int burn, int MCMC_itr, double sigma_mult, double psi_mult, int n, int alpha){
  
  int stop = 0;
  List ret;
  
  while (stop == 0){
    
    if (K < 1){
      Rcout << "Wrong number of mixture components ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (p < 1){
      Rcout << "Wrong number of layers ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (eps <= 0){
      Rcout << "Wrong value of eps ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (num_rand_start < 1){
      Rcout << "Wrong number of random restarts ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (num_run_smallEM < 1){
      Rcout << "Wrong number of small EM ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (max_itr_smallEM < 1){
      Rcout << "Wrong number of iterations for small EM ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (burn < 1){
      Rcout << "Wrong number of burns ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (MCMC_itr < 1){
      Rcout << "Wrong number of MCMC iterations ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (sigma_mult <= 0){
      Rcout << "Wrong value for Sigma scale multiplier ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (psi_mult <= 0){
      Rcout << "Wrong value for Psi scale multiplier ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    if (alpha < 0){
      Rcout << "Wrong value for seed ...\n";
      stop = 1;
      ret["Status"] = "Incorrect parameter";
      break;
    }
    
    
    arma::field<arma::mat> x(n, n);
    
    for (int i = 0; i < n; i++){
      for (int j = 0; j < (i+1); j++){
        arma::mat xMat (2,p);
        int ctr = 0;
        for (int l = 0; l < p; l++){
          for (int k = 0; k < 2; k++){
            int d = ctr*pow(n,2) + j*n + i;
            xMat(k,l) = y(d);
            if (xMat(k,l) != 0){
              stop = 1;
              break;
            }
            ctr = ctr+1;
          }
        }
        x(i,j) = xMat;
      }
    }
    
    if (stop == 1){
      Rcout << "Wrong entry in network data ...\n";
      ret["Status"] = "Incorrect data";
      break;
    }
    
    ret = netEM_mult(y, K, p, eps, num_rand_start, num_run_smallEM, max_itr_smallEM, burn, MCMC_itr, sigma_mult, psi_mult, n, alpha);
    stop = 1;
  }
  
  
  return ret;
}
