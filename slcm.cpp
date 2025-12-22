#include <RcppEigen.h>
using namespace Rcpp;
using namespace Eigen;
#include <omp.h> // OpenMP library




// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]


// [[Rcpp::export]]
List slcm_rcpp_vec_optimized(const Eigen::MatrixXd &Y, 
                             int nclass = 2, 
                             int maxiter = 100, 
                             double tol = 1e-6, 
                             bool verbose = false) {
  
  const int N = Y.rows();
  const int nitem = Y.cols();
  
  // Y complement
  const MatrixXd Y_c = MatrixXd::Ones(N, nitem) - Y;
  
  // 初始化參數
  // VectorXd eta = VectorXd::Constant(nclass, 1.0 / nclass);
  
  VectorXd eta = (VectorXd::Random(nclass).array() + 1.0) * 0.5;
  eta.array()/=eta.sum();
  // MatrixXd pikj = (MatrixXd::Random(nclass, nitem).array() * 0.3 + 0.5)
  //                                                 .max(0.2).min(0.8);
  MatrixXd pikj = (MatrixXd::Random(nclass, nitem).array() + 1.0) * 0.5;
  
  double loglik_old = R_NegInf;
  double loglik;
  
  MatrixXd posterior(N, nclass);
  MatrixXd loglik_mat(N, nclass);
  
  // 預分配記憶體
  MatrixXd logpik(nclass, nitem);
  MatrixXd log1mpik(nclass, nitem);
  VectorXd Nk(nclass);
  VectorXd log_eta = eta.array().log();
  
  // 用於向量化計算的臨時變數
  VectorXd max_log_vec(N);
  VectorXd sum_exp_vec(N);
  int iter;
  
  // --- 預計算對數概率 ---
  logpik = pikj.array().log();
  log1mpik = (1.0 - pikj.array()).log();
  
  // --- E-step: 計算後驗概率 ---
  // 計算對數似然: Y * log(pikj)^T + Y_c * log(1-pikj)^T
  loglik_mat = Y * logpik.transpose() + Y_c * log1mpik.transpose();
  
  // 加上類別先驗 log(eta)
  loglik_mat.rowwise() += log_eta.transpose();
  
  // 安全計算後驗概率 (log-sum-exp)
  // for (int i = 0; i < N; ++i) {
  //   double max_log = loglik_mat.row(i).maxCoeff();
  //   posterior.row(i) = (loglik_mat.row(i).array() - max_log).exp();
  //   double sum_exp = posterior.row(i).sum();
  //   posterior.row(i) /= sum_exp;
  // }
  
  // 1. 計算每行的最大值
  max_log_vec = loglik_mat.rowwise().maxCoeff();
  
  // 2. 計算 exp(loglik_mat - max_log)
  posterior = (loglik_mat.array().colwise() - max_log_vec.array()).exp();
  
  // 3. 計算每行的和
  sum_exp_vec = posterior.rowwise().sum();
  
  // // 4. 標準化
  // posterior.array().colwise() /= sum_exp_vec.array();
  
  for (iter = 0; iter < maxiter; ++iter) {
    // // --- 預計算對數概率 ---
    // logpik = pikj.array().log();
    // log1mpik = (1.0 - pikj.array()).log();
    // 
    // // --- E-step: 計算後驗概率 ---
    // // 計算對數似然: Y * log(pikj)^T + Y_c * log(1-pikj)^T
    // loglik_mat = Y * logpik.transpose() + Y_c * log1mpik.transpose();
    // 
    // // 加上類別先驗 log(eta)
    // loglik_mat.rowwise() += log_eta.transpose();
    // 
    // // 安全計算後驗概率 (log-sum-exp)
    // // for (int i = 0; i < N; ++i) {
    // //   double max_log = loglik_mat.row(i).maxCoeff();
    // //   posterior.row(i) = (loglik_mat.row(i).array() - max_log).exp();
    // //   double sum_exp = posterior.row(i).sum();
    // //   posterior.row(i) /= sum_exp;
    // // }
    // 
    // // 1. 計算每行的最大值
    // max_log_vec = loglik_mat.rowwise().maxCoeff();
    // 
    // // 2. 計算 exp(loglik_mat - max_log)
    // posterior = (loglik_mat.array().colwise() - max_log_vec.array()).exp();
    // 
    // // 3. 計算每行的和
    // sum_exp_vec = posterior.rowwise().sum();
    // 
    // 4. 標準化
    posterior.array().colwise() /= sum_exp_vec.array();
    
    // --- M-step: 更新參數 ---
    // 更新 eta
    Nk = posterior.colwise().sum();
    eta = Nk / N;
    log_eta = eta.array().log();
    
    // 更新 pikj
    pikj = posterior.transpose() * Y;
    // for (int k = 0; k < nclass; ++k) {
    //   if (Nk(k) > 1e-10) {  // 避免除以零
    //     pikj.row(k) /= Nk(k);
    //   }
    // }
    VectorXd inv_Nk = (Nk.array() > 1e-15).select(Nk.array().inverse(), 0.0);
    pikj.array().colwise() *= inv_Nk.array();
    
    // 數值邊界保護
    pikj = pikj.array().max(1e-15).min(1.0 - 1e-15);
    
    // --- 計算對數似然 ---
    // double loglik = 0.0;
    // for (int i = 0; i < N; ++i) {
    //   double max_log = loglik_mat.row(i).maxCoeff();
    //   loglik += max_log + std::log((loglik_mat.row(i).array() - max_log).exp().sum());
    // }
    
    // 同樣使用向量化計算對數似然
    
    
    // --- 預計算對數概率 ---
    logpik = pikj.array().log();
    log1mpik = (1.0 - pikj.array()).log();
    
    // --- E-step: 計算後驗概率 ---
    // 計算對數似然: Y * log(pikj)^T + Y_c * log(1-pikj)^T
    loglik_mat = Y * logpik.transpose() + Y_c * log1mpik.transpose();
    // 加上類別先驗 log(eta)
    loglik_mat.rowwise() += log_eta.transpose();
    
    // 1. 計算每行的最大值
    max_log_vec = loglik_mat.rowwise().maxCoeff();
    // 2. 計算 exp(loglik_mat - max_log)
    posterior = (loglik_mat.array().colwise() - max_log_vec.array()).exp();
    // 3. 計算每行的和
    sum_exp_vec = posterior.rowwise().sum();
    // 4. 標準化
    // posterior.array().colwise() /= sum_exp_vec.array();
    
    
    
    VectorXd loglik_terms = max_log_vec.array() + sum_exp_vec.array().log();
    // double loglik = loglik_terms.sum();
    loglik = loglik_terms.sum();
    
    // if (verbose) {
    //   Rcout << "Iter " << iter + 1 << " logLik: " << loglik 
    //         << " Change: " << std::abs(loglik - loglik_old) << std::endl;
    // }
    
    // 收斂檢查
    if (std::abs(loglik - loglik_old) < tol) {
      // if (verbose) Rcout << "Converged at iteration " << iter + 1 << std::endl;
      break;
    }
    loglik_old = loglik;
  }
  
  // if (verbose) {
  //   Rcout << "eta " << eta << " logLik: " << loglik 
  //         << " pikj: " << pikj << std::endl;
  // }
  
  return List::create(
    _["eta"] = eta,
    _["pikj"] = pikj,
    _["posterior"] = posterior,
    _["loglik"] = loglik,
    _["nclass"] = nclass,
    _["niter"] = (iter < maxiter) ? iter + 1 : maxiter
  );
}


// [[Rcpp::export]]
List slcm_rcpp_omp(const Eigen::MatrixXd &Y, 
                   const int nclass, 
                   const int maxiter, 
                   const double tol, 
                   const int nrun,
                   const int ncores) {
  
  const int N = Y.rows();
  const int nitem = Y.cols();
  
  // Y complement
  const MatrixXd Y_c = MatrixXd::Ones(N, nitem) - Y;
  // 為每個 run 準備輸出
  // 保存每次 run 的結果
  std::vector<VectorXd> eta_list(nrun);
  std::vector<MatrixXd> pikj_list(nrun);
  std::vector<MatrixXd> posterior_list(nrun);
  std::vector<double> loglik_list(nrun);
  std::vector<int> niter_list(nrun);
  
#pragma omp parallel for num_threads(ncores)
  for (int r = 0; r < nrun; ++r) {
    VectorXd eta = (VectorXd::Random(nclass).array() + 1.0) * 0.5;
    eta.array()/=eta.sum();
    // MatrixXd pikj = (MatrixXd::Random(nclass, nitem).array() * 0.3 + 0.5)
    //                                                 .max(0.2).min(0.8);
    MatrixXd pikj = (MatrixXd::Random(nclass, nitem).array() + 1.0) * 0.5;
    
    double loglik_old = R_NegInf;
    double loglik;
    
    MatrixXd posterior(N, nclass);
    MatrixXd loglik_mat(N, nclass);
    
    // 預分配記憶體
    MatrixXd logpik(nclass, nitem);
    MatrixXd log1mpik(nclass, nitem);
    VectorXd Nk(nclass);
    VectorXd log_eta = eta.array().log();
    
    // 用於向量化計算的臨時變數
    VectorXd max_log_vec(N);
    VectorXd sum_exp_vec(N);
    int iter;
    
    
    // --- 預計算對數概率 ---
    logpik = pikj.array().log();
    log1mpik = (1.0 - pikj.array()).log();
    
    // --- E-step: 計算後驗概率 ---
    loglik_mat = Y * logpik.transpose() + Y_c * log1mpik.transpose();
    
    loglik_mat.rowwise() += log_eta.transpose();
    
    max_log_vec = loglik_mat.rowwise().maxCoeff();
    
    
    posterior = (loglik_mat.array().colwise() - max_log_vec.array()).exp();
    
    
    sum_exp_vec = posterior.rowwise().sum();
    
    for (iter = 0; iter < maxiter; ++iter) {
      // // --- 預計算對數概率 ---
      // logpik = pikj.array().log();
      // log1mpik = (1.0 - pikj.array()).log();
      // 
      // // --- E-step: 計算後驗概率 ---
      // loglik_mat = Y * logpik.transpose() + Y_c * log1mpik.transpose();
      // 
      // loglik_mat.rowwise() += log_eta.transpose();
      // 
      // max_log_vec = loglik_mat.rowwise().maxCoeff();
      // 
      // 
      // posterior = (loglik_mat.array().colwise() - max_log_vec.array()).exp();
      // 
      // 
      // sum_exp_vec = posterior.rowwise().sum();

      posterior.array().colwise() /= sum_exp_vec.array();
      
      // --- M-step: 更新參數 ---
      // 更新 eta
      Nk = posterior.colwise().sum();
      eta = Nk / N;
      log_eta = eta.array().log();
      
      // 更新 pikj
      pikj = posterior.transpose() * Y;

      VectorXd inv_Nk = (Nk.array() > 1e-15).select(Nk.array().inverse(), 0.0);
      pikj.array().colwise() *= inv_Nk.array();
      
      // 數值邊界保護
      pikj = pikj.array().max(1e-15).min(1.0 - 1e-15);
      
      // 同樣使用向量化計算對數似然
      // --- 預計算對數概率 ---
      logpik = pikj.array().log();
      log1mpik = (1.0 - pikj.array()).log();
      
      // --- E-step: 計算後驗概率 ---
      loglik_mat = Y * logpik.transpose() + Y_c * log1mpik.transpose();
      
      loglik_mat.rowwise() += log_eta.transpose();
      
      max_log_vec = loglik_mat.rowwise().maxCoeff();
      
      
      posterior = (loglik_mat.array().colwise() - max_log_vec.array()).exp();
      
      
      sum_exp_vec = posterior.rowwise().sum();
      
      VectorXd loglik_terms = max_log_vec.array() + sum_exp_vec.array().log();
      // double loglik = loglik_terms.sum();
      loglik = loglik_terms.sum();
      
      // if (verbose) {
      //   Rcout << "Iter " << iter + 1 << " logLik: " << loglik 
      //         << " Change: " << std::abs(loglik - loglik_old) << std::endl;
      // }
      
      // 收斂檢查
      if (std::abs(loglik - loglik_old) < tol) {
        // if (verbose) Rcout << "Converged at iteration " << iter + 1 << std::endl;
        break;
      }
      loglik_old = loglik;
      
      
    }
    eta_list[r] = eta;
    pikj_list[r] = pikj;
    posterior_list[r] = posterior;
    loglik_list[r] = loglik;
    niter_list[r]=(iter < maxiter) ? iter + 1 : maxiter;
    
  }
  
  
  // // 找出 log-likelihood 最大的 run
  // int best_idx = 0;
  // double max_loglik = loglik_list[0];
  // for (int r = 1; r < nrun; ++r) {
  //   if (loglik_list[r] > max_loglik) {
  //     max_loglik = loglik_list[r];
  //     best_idx = r;
  //   }
  // }
  
  
  return List::create(
    _["eta"] = eta_list,
    _["pikj"] = pikj_list,
    _["posterior"] = posterior_list,
    _["loglik"] = loglik_list,
    _["niter"] = niter_list,
    _["nclass"] = nclass);
  
}

