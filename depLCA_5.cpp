#include <Rcpp.h>
#include <RcppEigen.h>
#include <unsupported/Eigen/KroneckerProduct>
#include <omp.h> // OpenMP library


using namespace Rcpp;
using namespace Eigen;


// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]




// [[Rcpp::export]]
MatrixXd eta_cpp(const MatrixXd& beta,
                 const MatrixXd& Xprev,
                 const int& nxprev,
                 const int& nclass) {
  
  // 直接預分配 betamatrix 並填充數據
  MatrixXd betamatrix(nxprev + 1, nclass);
  betamatrix.leftCols(nclass - 1) = beta;
  betamatrix.col(nclass - 1).setZero();  // 最後一列設為 0
  
  
  MatrixXd eta = Xprev * betamatrix;  
  
  // 就地計算 softmax
  eta.array().colwise() -= eta.rowwise().maxCoeff().array();  // 減去每行最大值
  eta = eta.array().exp();                                   // 指數化
  eta.array().colwise() /= eta.rowwise().sum().array();      // 歸一化
  
  return eta; 
}






// [[Rcpp::export]]
MatrixXd h_p_cpp(const MatrixXd& eta,
                 const MatrixXd& cond_prob) {
  
  // const MatrixXd Xprev
  // const int nxprev
  // const int nclass
  // MatrixXd eta = eta_cpp(beta,Xprev,nxprev,nclass);
  
  // MatrixXd h_p=cond_prob.array()*eta.array();
  MatrixXd h_p=eta.cwiseProduct(cond_prob);
  
  // // 计算每行的和
  // VectorXd rowSumsTemp = h_p_temp.rowwise().sum();
  // // 按列进行除法
  // MatrixXd h_p = h_p_temp.array().colwise() / rowSumsTemp.array();
  
  h_p.array().colwise() /= h_p.rowwise().sum().array();      // 歸一化
  
  return h_p;
}





// [[Rcpp::export]]
MatrixXd Hess1_cpp(const MatrixXd& eta_p,
                   const MatrixXd& Xprev,
                   const int& npeop,
                   const int& nxprev,
                   const int& nclass) {
  
  const int nclass_minus_1 = nclass - 1;
  
  MatrixXd hess = MatrixXd::Zero((nxprev+1)*nclass_minus_1, 
                                 (nxprev+1)*nclass_minus_1);
  // MatrixXd term1 = MatrixXd::Zero(nclass_minus_1, nclass_minus_1);
  MatrixXd term1(nclass_minus_1, nclass_minus_1);
  VectorXd eta_i(nclass_minus_1);
  MatrixXd X_i(1, nxprev+1);
  // MatrixXd XtX(nxprev+1, nxprev+1);
  
  for (int i = 0; i < npeop; ++i) {
    
    // MatrixXd eta_i = eta_p.row(i).segment(0, nclass - 1);
    
    eta_i.noalias() = eta_p.row(i).head(nclass_minus_1);
    X_i.noalias() = Xprev.row(i);
    // MatrixXd Xprev_i = Xprev.row(i);
    
    
    // MatrixXd eta_diag = eta_p.row(i).segment(0, nclass - 1).asDiagonal();
    // MatrixXd eta_outer = eta_i * eta_i.transpose();
    
    // MatrixXd term1 = eta_i.asDiagonal();
    
    term1 = eta_i.asDiagonal();
    
    // MatrixXd term3 = eta_i.transpose() * eta_i;
    
    // term1.diagonal() = eta_i;
    term1.noalias()-=eta_i * eta_i.transpose();
    
    // MatrixXd term4 =term1-term3;
    
    // MatrixXd term2 = Xprev_i.transpose() * Xprev_i;
    // MatrixXd term2 = Xprev.row(i).transpose() * Xprev.row(i);
    
    hess -= KroneckerProduct(term1,X_i.transpose() * X_i);
    // hess -= KroneckerProduct( term1,
    //                           Xprev.row(i).transpose() * Xprev.row(i) );
    // hess -= KroneckerProduct(term1-term3,term2);
  }
  
  return hess;
}





// [[Rcpp::export]]
// 用backtracking algorithm
List max_beta_6_no_covariate_cpp(MatrixXd& beta,
                                 const MatrixXd& Xprev,
                                 MatrixXd& eta_p,
                                 const MatrixXd& h_p,
                                 const int& npeop,
                                 const int& nxprev,
                                 const int& nclass,
                                 const int& maxiter,
                                 const double& step_length,
                                 const double& tol,
                                 const int& it) {
  
  const int nclass_minus_1 = nclass - 1;
  int count=0;
  MatrixXd eta=eta_p;
  
  
  // MatrixXd eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
  // VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
  VectorXd deri =(h_p.leftCols(nclass_minus_1)-eta_p.leftCols(nclass_minus_1)).colwise().sum();
  // MatrixXd eta_i_diag= MatrixXd::Zero(nclass - 1,nclass - 1) ;
  
  // VectorXd eta_i = eta_p.row(0).segment(0, nclass - 1);
  
  // MatrixXd eta_i_diag(nclass - 1,nclass - 1);
  // eta_i_diag.setZero();
  // eta_i_diag.diagonal() = eta_i; // 更新值而不是重新分配
  // 
  // MatrixXd hess=-1*npeop*(eta_i_diag-eta_i * eta_i.transpose());
  VectorXd eta_i = eta_p.row(0).head(nclass_minus_1);
  MatrixXd hess = eta_i.asDiagonal();  // 先构造对角矩阵
  hess.noalias() -= eta_i * eta_i.transpose();  // 减去外积
  hess *= -npeop;  // 最后缩放
  
  // VectorXd eta_i =eta_p.row(0).head(nclass - 1);
  // MatrixXd hess = MatrixXd::Zero(nclass - 1,nclass - 1);
  // hess.diagonal() = eta_i; 
  // hess-=eta_i * eta_i.transpose();
  // // hess=-1*npeop*hess;
  // hess *= -1 * npeop;
  // MatrixXd hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
  
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
  // MatrixXd ginv_hess = cod.pseudoInverse();
  // // MatrixXd ginv_hess = hess.inverse();
  //
  //
  // MatrixXd direction = -1*ginv_hess * deri;
  MatrixXd direction = -cod.pseudoInverse() * deri;
  
  // MatrixXd direction = -hess.ldlt().solve(deri);
  // // 使用 CompleteOrthogonalDecomposition 來求解
  // CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_decomp(hess);
  // MatrixXd direction = -cod_decomp.solve(deri);
  
  // direction.resize(nxprev+1 ,nclass_minus_1);
  
  
  // MatrixXd eta=eta_cpp(beta,Xprev,nxprev,nclass);
  
  
  // MatrixXd result_temp=h_p.array()*eta_p.array().log();
  //
  // // 計算結果總和
  // double llik=result_temp.sum();
  
  
  double llik=(h_p.array()*eta_p.array().log()).sum();
  
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  
  double llik_p;
  
  MatrixXd beta_p(nxprev+1 ,nclass_minus_1);
  double a=1.0;
  
  int iter = 0;
  for(iter = 0; iter < maxiter; ++iter){
    
    
    beta_p.noalias() =beta+a*direction.transpose();
    eta_p.noalias() =eta_cpp(beta_p,Xprev,nxprev,nclass);
    llik_p=(h_p.array()*eta_p.array().log()).sum();
    
    // llik_p= beta_likeli_cpp(beta_p,h_p,Xprev,nxprev,nclass);
    
    
    // max_beta_score_bisection_1_cpp(beta,Xprev,eta_p,h_p,
    //                                     npeop,nxprev,nclass,err);
    
    
    // double max_diff = (beta_p - beta).cwiseAbs().maxCoeff();
    // Rcpp::Rcout << "Iteration " << iter << ", Max Difference: " << max_diff << std::endl;
    // beta=beta_p;
    
    
    
    if(llik_p>llik){
      // Rcpp::Rcout << "Iteration " << iter << ", llik: " << llik_p << std::endl;
      ++count;
      llik= llik_p;
      beta.noalias() =beta_p;
      eta.noalias() =eta_p;
      if(a*direction.cwiseAbs().maxCoeff()<tol|count==it) break;
      // if (((a * direction).cwiseAbs().array() < tol).all()) break;
      
      
      // eta_p重複計算了，在llik_p已經有計算過
      // eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
      // deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
      // hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
      
      
      deri.noalias() =(h_p.leftCols(nclass_minus_1)-eta_p.leftCols(nclass_minus_1)).colwise().sum();
      
      // eta_i_diag= MatrixXd::Zero(nclass - 1,nclass - 1) ;
      
      // eta_i = eta_p.row(0).segment(0, nclass - 1);
      // eta_i_diag.setZero();
      // eta_i_diag.diagonal() = eta_i; // 更新值而不是重新分配
      // hess=-1*npeop*(eta_i_diag-eta_i * eta_i.transpose());
      
      eta_i.noalias() =eta_p.row(0).head(nclass_minus_1);
      hess = eta_i.asDiagonal();  // 先构造对角矩阵
      hess.noalias() -= eta_i * eta_i.transpose();  // 减去外积
      hess *= -npeop;  // 最后缩放
      
      // eta_i.noalias() =eta_p.row(0).head(nclass - 1);
      // hess.setZero();
      // hess.diagonal() = eta_i; 
      // hess-=eta_i * eta_i.transpose();
      // // hess=-1*npeop*hess;
      // hess *= -1 * npeop;
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
      // ginv_hess = cod.pseudoInverse();
      // // MatrixXd ginv_hess = hess.inverse();
      //
      //
      // direction = -1*ginv_hess * deri;
      direction.noalias() = -cod.pseudoInverse() * deri;
      // direction = -hess.ldlt().solve(deri);
      // // 使用 CompleteOrthogonalDecomposition 來求解
      // CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_decomp(hess);
      // MatrixXd direction = -cod_decomp.solve(deri);
      
      // direction.resize(nxprev+1 ,nclass_minus_1);
      
      
    }else{
      a*=step_length;
      if(a*direction.cwiseAbs().maxCoeff()<tol) break;
    }
    
    
    
  }
  
  
  // eta_p=eta_cpp(beta,Xprev,nxprev,nclass);
  // llik=(h_p.array()*eta_p.array().log()).sum();
  
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  // 如果達到最大迭代次數，可能有誤
  // 只適用於驗證，可以不要算llik
  // llik_p= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  List result = List::create(
    Named("beta_p") = beta,
    Named("eta") = eta,
    Named("llik") = llik,
    Named("direction") = direction,
    Named("a") = a,
    Named("iter") = iter);
  
  return result;
}



// [[Rcpp::export]]
// 用backtracking algorithm
std::vector<MatrixXd> max_beta_6_no_covariate_omp_cpp(MatrixXd& beta,
                                                      const MatrixXd& Xprev,
                                                      MatrixXd& eta_p,
                                                      const MatrixXd& h_p,
                                                      const int& npeop,
                                                      const int& nxprev,
                                                      const int& nclass,
                                                      const int& maxiter,
                                                      const double& step_length,
                                                      const double& tol,
                                                      const int& it) {
  
  const int nclass_minus_1 = nclass - 1;
  int count=0;
  MatrixXd eta=eta_p;
  
  
  // MatrixXd eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
  // VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
  VectorXd deri =(h_p.leftCols(nclass_minus_1)-eta_p.leftCols(nclass_minus_1)).colwise().sum();
  // MatrixXd eta_i_diag= MatrixXd::Zero(nclass - 1,nclass - 1) ;
  
  // VectorXd eta_i = eta_p.row(0).segment(0, nclass - 1);
  
  // MatrixXd eta_i_diag(nclass - 1,nclass - 1);
  // eta_i_diag.setZero();
  // eta_i_diag.diagonal() = eta_i; // 更新值而不是重新分配
  // 
  // MatrixXd hess=-1*npeop*(eta_i_diag-eta_i * eta_i.transpose());
  VectorXd eta_i = eta_p.row(0).head(nclass_minus_1);
  MatrixXd hess = eta_i.asDiagonal();  // 先构造对角矩阵
  hess.noalias() -= eta_i * eta_i.transpose();  // 减去外积
  hess *= -npeop;  // 最后缩放
  
  // VectorXd eta_i =eta_p.row(0).head(nclass - 1);
  // MatrixXd hess = MatrixXd::Zero(nclass - 1,nclass - 1);
  // hess.diagonal() = eta_i; 
  // hess-=eta_i * eta_i.transpose();
  // // hess=-1*npeop*hess;
  // hess *= -1 * npeop;
  // MatrixXd hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
  
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
  // MatrixXd ginv_hess = cod.pseudoInverse();
  // // MatrixXd ginv_hess = hess.inverse();
  //
  //
  // MatrixXd direction = -1*ginv_hess * deri;
  MatrixXd direction = -cod.pseudoInverse() * deri;
  
  // MatrixXd direction = -hess.ldlt().solve(deri);
  // // 使用 CompleteOrthogonalDecomposition 來求解
  // CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_decomp(hess);
  // MatrixXd direction = -cod_decomp.solve(deri);
  
  // direction.resize(nxprev+1 ,nclass_minus_1);
  
  
  // MatrixXd eta=eta_cpp(beta,Xprev,nxprev,nclass);
  
  
  // MatrixXd result_temp=h_p.array()*eta_p.array().log();
  //
  // // 計算結果總和
  // double llik=result_temp.sum();
  
  
  double llik=(h_p.array()*eta_p.array().log()).sum();
  
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  
  double llik_p;
  
  MatrixXd beta_p(nxprev+1 ,nclass_minus_1);
  double a=1.0;
  
  int iter = 0;
  for(iter = 0; iter < maxiter; ++iter){
    
    
    beta_p.noalias() =beta+a*direction.transpose();
    eta_p.noalias() =eta_cpp(beta_p,Xprev,nxprev,nclass);
    llik_p=(h_p.array()*eta_p.array().log()).sum();
    
    // llik_p= beta_likeli_cpp(beta_p,h_p,Xprev,nxprev,nclass);
    
    
    // max_beta_score_bisection_1_cpp(beta,Xprev,eta_p,h_p,
    //                                     npeop,nxprev,nclass,err);
    
    
    // double max_diff = (beta_p - beta).cwiseAbs().maxCoeff();
    // Rcpp::Rcout << "Iteration " << iter << ", Max Difference: " << max_diff << std::endl;
    // beta=beta_p;
    
    
    
    if(llik_p>llik){
      // Rcpp::Rcout << "Iteration " << iter << ", llik: " << llik_p << std::endl;
      ++count;
      llik= llik_p;
      beta.noalias() =beta_p;
      eta.noalias() =eta_p;
      if(a*direction.cwiseAbs().maxCoeff()<tol|count==it) break;
      // if (((a * direction).cwiseAbs().array() < tol).all()) break;
      
      
      // eta_p重複計算了，在llik_p已經有計算過
      // eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
      // deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
      // hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
      
      
      deri.noalias() =(h_p.leftCols(nclass_minus_1)-eta_p.leftCols(nclass_minus_1)).colwise().sum();
      
      // eta_i_diag= MatrixXd::Zero(nclass - 1,nclass - 1) ;
      
      // eta_i = eta_p.row(0).segment(0, nclass - 1);
      // eta_i_diag.setZero();
      // eta_i_diag.diagonal() = eta_i; // 更新值而不是重新分配
      // hess=-1*npeop*(eta_i_diag-eta_i * eta_i.transpose());
      
      eta_i.noalias() =eta_p.row(0).head(nclass_minus_1);
      hess = eta_i.asDiagonal();  // 先构造对角矩阵
      hess.noalias() -= eta_i * eta_i.transpose();  // 减去外积
      hess *= -npeop;  // 最后缩放
      
      // eta_i.noalias() =eta_p.row(0).head(nclass - 1);
      // hess.setZero();
      // hess.diagonal() = eta_i; 
      // hess-=eta_i * eta_i.transpose();
      // // hess=-1*npeop*hess;
      // hess *= -1 * npeop;
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
      // ginv_hess = cod.pseudoInverse();
      // // MatrixXd ginv_hess = hess.inverse();
      //
      //
      // direction = -1*ginv_hess * deri;
      direction.noalias() = -cod.pseudoInverse() * deri;
      // direction = -hess.ldlt().solve(deri);
      // // 使用 CompleteOrthogonalDecomposition 來求解
      // CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_decomp(hess);
      // MatrixXd direction = -cod_decomp.solve(deri);
      
      // direction.resize(nxprev+1 ,nclass_minus_1);
      
      
    }else{
      a*=step_length;
      if(a*direction.cwiseAbs().maxCoeff()<tol) break;
    }
    
    
    
  }
  
  
  // eta_p=eta_cpp(beta,Xprev,nxprev,nclass);
  // llik=(h_p.array()*eta_p.array().log()).sum();
  
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  // 如果達到最大迭代次數，可能有誤
  // 只適用於驗證，可以不要算llik
  // llik_p= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  // List result = List::create(
  //   Named("beta_p") = beta,
  //   Named("eta") = eta,
  //   Named("llik") = llik,
  //   Named("direction") = direction,
  //   Named("a") = a,
  //   Named("iter") = iter);
  // 
  // return result;
  return {beta, eta};
}


// [[Rcpp::export]]
// 用backtracking algorithm
List max_beta_6_cpp(MatrixXd& beta,
                    const MatrixXd& Xprev,
                    MatrixXd& eta_p,
                    const MatrixXd& h_p,
                    const int& npeop,
                    const int& nxprev,
                    const int& nclass,
                    const int& maxiter,
                    const double& step_length,
                    const double& tol,
                    const int& it) {
  
  int count=0;
  MatrixXd eta=eta_p;
  // MatrixXd eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
  
  MatrixXd deri=Xprev.transpose() *
    ( h_p.leftCols(nclass - 1)-eta_p.leftCols(nclass - 1) );
  deri.resize(deri.size(),1);
  // VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
  MatrixXd hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
  // MatrixXd ginv_hess = cod.pseudoInverse();
  // // MatrixXd ginv_hess = hess.inverse();
  //
  //
  // MatrixXd direction = -1*ginv_hess * deri;
  MatrixXd direction = -cod.pseudoInverse() * deri;
  // MatrixXd direction = CompleteOrthogonalDecomposition<MatrixXd>(hess).solve(-deri);
  
  // MatrixXd direction = -hess.ldlt().solve(deri);
  
  
  direction.resize(nxprev+1 ,nclass-1);
  
  
  
  
  
  // MatrixXd eta=eta_cpp(beta,Xprev,nxprev,nclass);
  
  
  // MatrixXd result_temp=h_p.array()*eta_p.array().log();
  // 
  // // 計算結果總和
  // double llik=result_temp.sum();
  
  
  double llik=(h_p.array()*eta_p.array().log()).sum();
  
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  
  double llik_p;
  
  MatrixXd beta_p(nxprev+1 ,nclass-1);
  double a=1.0;
  
  int iter = 0;
  for(iter = 0; iter < maxiter; ++iter){
    
    
    beta_p.noalias()=beta+a*direction;
    eta_p.noalias()=eta_cpp(beta_p,Xprev,nxprev,nclass);
    llik_p=(h_p.array()*eta_p.array().log()).sum();
    
    // llik_p= beta_likeli_cpp(beta_p,h_p,Xprev,nxprev,nclass);
    
    
    // max_beta_score_bisection_1_cpp(beta,Xprev,eta_p,h_p,
    //                                     npeop,nxprev,nclass,err);
    
    
    // double max_diff = (beta_p - beta).cwiseAbs().maxCoeff();
    // Rcpp::Rcout << "Iteration " << iter << ", Max Difference: " << max_diff << std::endl;
    // beta=beta_p;
    
    
    
    if(llik_p>llik){
      // Rcpp::Rcout << "Iteration " << iter << ", llik: " << llik_p << std::endl;
      ++count;
      llik= llik_p;
      beta.noalias() =beta_p;
      eta.noalias() =eta_p;
      if(a*direction.cwiseAbs().maxCoeff()<tol|count==it) break;
      // if (((a * direction).cwiseAbs().array() < tol).all()) break;
      
      // 是否達到maxiter-1就不做下列計算
      // 
      // 這麼處理反而變慢
      // // 是否達到maxiter-1就不做下列計算
      // if(iter<(maxiter-1)){
      //   eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
      //   deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
      //   hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
      //   
      //   // 計算ginv
      //   CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
      //   ginv_hess = cod.pseudoInverse();
      //   // MatrixXd ginv_hess = hess.inverse();
      //   
      //   
      //   direction = -1*ginv_hess * deri;
      //   // MatrixXd dir=direction;
      //   
      //   direction.resize(nxprev+1 ,nclass-1);
      //   // llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
      // }
      
      // eta_p重複計算了，在llik_p已經有計算過
      // eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
      
      deri=Xprev.transpose() *
        ( h_p.leftCols(nclass - 1)-eta_p.leftCols(nclass - 1) );
      deri.resize(deri.size(),1);
      // deri.noalias()  = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
      hess.noalias()  = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
      // ginv_hess = cod.pseudoInverse();
      // // MatrixXd ginv_hess = hess.inverse();
      //
      //
      // direction = -1*ginv_hess * deri;
      direction  = -cod.pseudoInverse() * deri;
      // direction = CompleteOrthogonalDecomposition<MatrixXd>(hess).solve(-deri);
      // direction = -hess.ldlt().solve(deri);
      direction.resize(nxprev+1 ,nclass-1);
      
      
    }else{
      a*=step_length;
      if(a*direction.cwiseAbs().maxCoeff()<tol) break;
    }
    
    
    
  }
  
  
  // eta_p=eta_cpp(beta,Xprev,nxprev,nclass);
  // llik=(h_p.array()*eta_p.array().log()).sum();
  
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  // 如果達到最大迭代次數，可能有誤
  // 只適用於驗證，可以不要算llik
  // llik_p= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  List result = List::create(
    Named("beta_p") = beta,
    Named("eta") = eta,
    Named("llik") = llik,
    Named("direction") = direction,
    Named("a") = a,
    Named("iter") = iter);
  
  return result;
}





// [[Rcpp::export]]
// 用backtracking algorithm
std::vector<MatrixXd> max_beta_6_for_omp_cpp(MatrixXd& beta,
                                             const MatrixXd& Xprev,
                                             MatrixXd& eta_p,
                                             const MatrixXd& h_p,
                                             const int& npeop,
                                             const int& nxprev,
                                             const int& nclass,
                                             const int& maxiter,
                                             const double& step_length,
                                             const double& tol,
                                             const int& it) {
  
  int count=0;
  MatrixXd eta=eta_p;
  // MatrixXd eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
  
  MatrixXd deri=Xprev.transpose() *
    ( h_p.leftCols(nclass - 1)-eta_p.leftCols(nclass - 1) );
  // Eigen::VectorXd a1 = Eigen::Map<Eigen::VectorXd>(mat.data(), mat.size());
  deri.resize(deri.size(),1);
  
  // VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
  MatrixXd hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
  // MatrixXd ginv_hess = cod.pseudoInverse();
  // // MatrixXd ginv_hess = hess.inverse();
  // 
  // 
  // MatrixXd direction = -1*ginv_hess * deri;
  MatrixXd direction = -cod.pseudoInverse() * deri;
  
  
  direction.resize(nxprev+1 ,nclass-1);
  
  
  
  
  
  // MatrixXd eta=eta_cpp(beta,Xprev,nxprev,nclass);
  
  
  // MatrixXd result_temp=h_p.array()*eta_p.array().log();
  // 
  // // 計算結果總和
  // double llik=result_temp.sum();
  
  
  double llik=(h_p.array()*eta_p.array().log()).sum();
  
  double llik_p;
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  MatrixXd beta_p(nxprev+1 ,nclass-1);
  double a=1.0;
  
  for(int iter = 0; iter < maxiter; ++iter){
    
    
    beta_p.noalias()=beta+a*direction;
    eta_p.noalias()=eta_cpp(beta_p,Xprev,nxprev,nclass);
    llik_p=(h_p.array()*eta_p.array().log()).sum();
    
    // llik_p= beta_likeli_cpp(beta_p,h_p,Xprev,nxprev,nclass);
    
    
    // max_beta_score_bisection_1_cpp(beta,Xprev,eta_p,h_p,
    //                                     npeop,nxprev,nclass,err);
    
    
    // double max_diff = (beta_p - beta).cwiseAbs().maxCoeff();
    // Rcpp::Rcout << "Iteration " << iter << ", Max Difference: " << max_diff << std::endl;
    // beta=beta_p;
    
    
    
    if(llik_p>llik){
      // Rcpp::Rcout << "Iteration " << iter << ", llik: " << llik_p << std::endl;
      ++count;
      llik= llik_p;
      beta.noalias() =beta_p;
      eta.noalias() =eta_p;
      if(a*direction.cwiseAbs().maxCoeff()<tol|count==it) break;
      // if (((a * direction).cwiseAbs().array() < tol).all()) break;
      
      // 是否達到maxiter-1就不做下列計算
      // 
      // 這麼處理反而變慢
      // // 是否達到maxiter-1就不做下列計算
      // if(iter<(maxiter-1)){
      //   eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
      //   deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
      //   hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
      //   
      //   // 計算ginv
      //   CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
      //   ginv_hess = cod.pseudoInverse();
      //   // MatrixXd ginv_hess = hess.inverse();
      //   
      //   
      //   direction = -1*ginv_hess * deri;
      //   // MatrixXd dir=direction;
      //   
      //   direction.resize(nxprev+1 ,nclass-1);
      //   // llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
      // }
      
      // eta_p重複計算了，在llik_p已經有計算過
      // eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
      
      
      deri=Xprev.transpose() *
        ( h_p.leftCols(nclass - 1)-eta_p.leftCols(nclass - 1) );
      // Eigen::VectorXd a1 = Eigen::Map<Eigen::VectorXd>(mat.data(), mat.size());
      deri.resize(deri.size(),1);
      
      // deri.noalias() = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
      hess.noalias() = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
      // ginv_hess = cod.pseudoInverse();
      // // MatrixXd ginv_hess = hess.inverse();
      // 
      // 
      // direction = -1*ginv_hess * deri;
      direction= -cod.pseudoInverse() * deri;
      
      direction.resize(nxprev+1 ,nclass-1);
      
      
    }else{
      a*=step_length;
      if(a*direction.cwiseAbs().maxCoeff()<tol) break;
    }
    
    
    
  }
  
  
  // eta_p=eta_cpp(beta,Xprev,nxprev,nclass);
  // llik=(h_p.array()*eta_p.array().log()).sum();
  
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  // 如果達到最大迭代次數，可能有誤
  // 只適用於驗證，可以不要算llik
  // llik_p= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  // std::vector<MatrixXd> result(2);
  // result[0] = beta; 
  // result[1] = eta;  
  // return result;
  
  return {beta, eta};
  
  // List result = List::create(
  //   Named("beta_p") = beta,
  //   Named("eta") = eta,
  //   Named("llik") = llik,
  //   Named("direction") = direction,
  //   Named("a") = a,
  //   Named("iter") = iter);
  
  
} 


// [[Rcpp::export]]
Eigen::MatrixXd cbindEigen_matrix(const Eigen::MatrixXd& mat1, 
                                  const Eigen::MatrixXd& mat2) {
  // 预分配结果矩阵（直接构造时保留内存）
  Eigen::MatrixXd result(mat1.rows(), mat1.cols() + mat2.cols());
  
  // 使用 .leftCols() 和 .rightCols() 快速填充（无额外拷贝）
  result.leftCols(mat1.cols()) = mat1;
  result.rightCols(mat2.cols()) = mat2;
  
  return result;
}




// [[Rcpp::export]]
MatrixXd rbindEigen_matrix(const MatrixXd& mat1, 
                           const MatrixXd& mat2) {
  int nrow1 = mat1.rows();
  int nrow2 = mat2.rows();
  
  int ncol1 = mat1.cols();
  // int ncol2 = mat2.cols();
  
  // Create a new matrix with the combined number of columns
  MatrixXd result(nrow1+nrow2, ncol1);
  
  // Copy data from mat1 to the result matrix
  result.block(0, 0, nrow1, ncol1) = mat1;
  
  // Copy data from mat2 to the result matrix
  result.block(nrow1, 0, nrow2, ncol1) = mat2;
  
  return result;
}




// [[Rcpp::export]]
List conditional_prob_and_Pi_tau_cpp(const List& alpha0,
                                     const List& alpha,
                                     const MatrixXd& Y_comp,
                                     const MatrixXd& e_comp,
                                     const List& xcond,
                                     const VectorXi& nxcond,
                                     const int& npeop,
                                     const int& nitem,
                                     const int& nclass) {
  
  std::vector<MatrixXd> Pi_minus_1(nclass);
  std::vector<MatrixXd> cond_dist(nclass);
  
  // List Pi_minus_1(nclass);
  // List cond_dist(nclass);
  // List cond_dist_test(nclass);
  
  
  // onesVector可以寫在外面
  const MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  
  
  // MatrixXd Pi_minus_1_j=MatrixXd::Zero(npeop, nxcond.sum()-nitem);
  
  // 是否給起始值全為0
  MatrixXd cond_prob(npeop, nclass);
  
  MatrixXd Pimkj;
  MatrixXd temp;
  
  
  // MatrixXd cond_dist_j;
  
  for (int j = 0; j < nclass; ++j) {
    
    MatrixXd Pi_j(npeop, 0);
    MatrixXd Pi_minus_1_j(npeop, 0);
    
    for (int m = 0; m < nitem; ++m) {
      
      
      if ( nxcond[m]==0 ) {
        const MatrixXd& alpha0_m = alpha0[m];
        
        
        Eigen::MatrixXd eigen_B(1, alpha0_m.cols() + 1);
        eigen_B.leftCols(alpha0_m.cols()) = alpha0_m.row(j);
        eigen_B.rightCols(1).setZero();
        
        temp = onesVector * eigen_B;
        
        
      }else{
        const MatrixXd& alpha0_m = alpha0[m];
        const MatrixXd& alpha_m = alpha[m];
        const MatrixXd& xcond_m = xcond[m];
        
        
        
        // 直接构造最终矩阵 eigen_B，避免所有中间临时矩阵
        MatrixXd eigen_B(alpha_m.rows() + 1, alpha_m.cols() + 1);
        
        // 填充上半部分：[alpha0[m].row(j); alpha[m]]
        eigen_B.topRows(1) = alpha0_m.row(j);      // 第一行 = alpha0[m].row(j)
        eigen_B.bottomRows(alpha_m.rows()) = alpha_m; // 剩余行 = alpha[m]
        
        // 最右侧补零列
        eigen_B.rightCols(1).setZero();
        
        temp = cbindEigen_matrix(onesVector,xcond_m) * eigen_B;
      }
      
      // 计算 softmax
      temp.array().colwise() -= temp.rowwise().maxCoeff().array();
      // Eigen::MatrixXd Pimkj = temp.array().exp();
      Pimkj = temp.array().exp();
      Pimkj.array().colwise() /= Pimkj.rowwise().sum().array();
      
      
      // MatrixXd Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
      // Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
      // Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
      
      // 拼接结果
      Pi_minus_1_j.conservativeResize(npeop, Pi_minus_1_j.cols() + Pimkj.cols() - 1);
      Pi_minus_1_j.rightCols(Pimkj.cols() - 1) = Pimkj.leftCols(Pimkj.cols() - 1);
      
      Pi_j.conservativeResize(npeop, Pi_j.cols() + Pimkj.cols());
      Pi_j.rightCols(Pimkj.cols()) = Pimkj;
      
    }
    
    Pi_minus_1[j]=Pi_minus_1_j;
    
    
    // 第一部分的計算可以合併
    cond_prob.col(j) = (Pi_j.array() * Y_comp.array() + (1 - Y_comp.array()))
             .log().rowwise().sum().exp();
    
    
    
    // 第二部分的計算可以更簡潔
    cond_dist[j] = (e_comp * Pi_j.array().log().matrix().transpose())
                                 .array().exp().transpose();
    
    
    
    
    // MatrixXd Pi_j_log=Pi_j.array().log();
    // 
    // cond_dist_j=e_comp*Pi_j_log.transpose();
    // cond_dist[j]=cond_dist_j.array().exp().transpose();
    
  }
  
  
  return List::create(Named("cond_prob") = cond_prob,
                      Named("Pi_minus_1") = Pi_minus_1,
                      Named("cond_dist") = cond_dist);
}





// [[Rcpp::export]]
std::vector<std::vector<MatrixXd>>  
  conditional_prob_and_Pi_for_omp_cpp(const std::vector<MatrixXd>& alpha0,
                                      const std::vector<MatrixXd>& alpha,
                                      const MatrixXd& Y_comp,
                                      const MatrixXd& e_comp,
                                      const std::vector<MatrixXd> xcond_vec,
                                      const VectorXi& nxcond,
                                      const int& npeop,
                                      const int& nitem,
                                      const int& nclass) {
    
    // List Pi_minus_1(nclass);
    std::vector<MatrixXd> Pi_minus_1(nclass);
    std::vector<MatrixXd> cond_dist(nclass);
    // onesVector可以寫在外面
    const MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
    
    // MatrixXd Pi_j=MatrixXd::Zero(npeop, nxcond.sum());
    // MatrixXd Pi_minus_1_j=MatrixXd::Zero(npeop, nxcond.sum()-nitem);
    
    // 是否給起始值全為0
    MatrixXd cond_prob(npeop, nclass);
    
    Eigen::MatrixXd eigen_B;
    Eigen::MatrixXd temp;
    
    for (int j = 0; j < nclass; ++j) {
      
      MatrixXd Pi_j(npeop, 0);
      MatrixXd Pi_minus_1_j(npeop, 0);
      
      for (int m = 0; m < nitem; ++m) {
        // Rf_isNull(xcond[m])
        // Eigen::MatrixXd eigen_B;
        if ( nxcond[m]==0 ) {
          eigen_B.resize(1, alpha0[m].cols() + 1);
          
          // Eigen::MatrixXd eigen_B(1, alpha0[m].cols() + 1);
          // eigen_B(1, alpha0[m].cols() + 1);
          eigen_B.leftCols(alpha0[m].cols()) = alpha0[m].row(j);
          eigen_B.rightCols(1).setZero();
          
          temp = onesVector * eigen_B;
          
        }else{
          // MatrixXd eigen_A =cbindEigen_matrix(onesVector,xcond_vec[m]);
          
          // 直接构造最终矩阵 eigen_B，避免所有中间临时矩阵
          eigen_B.resize(alpha[m].rows() + 1, alpha[m].cols() + 1);
          
          // Eigen::MatrixXd eigen_B(alpha[m].rows() + 1, alpha[m].cols() + 1);
          // eigen_B(alpha[m].rows() + 1, alpha[m].cols() + 1);
          // 填充上半部分：[alpha0[m].row(j); alpha[m]]
          eigen_B.topRows(1) = alpha0[m].row(j);      // 第一行 = alpha0[m].row(j)
          eigen_B.bottomRows(alpha[m].rows()) = alpha[m]; // 剩余行 = alpha[m]
          
          // 最右侧补零列
          eigen_B.rightCols(1).setZero();
          // MatrixXd eigen_B = as<Map<MatrixXd> >( cbind(rbind(alpha0_m.row(j), alpha_m), 0.0) );
          
          // temp = eigen_A * eigen_B;
          temp = cbindEigen_matrix(onesVector,xcond_vec[m]) * eigen_B;
        }
        // // 计算 temp
        // Eigen::MatrixXd temp = (nxcond[m] == 0) ? 
        // onesVector * eigen_B : 
        //   cbindEigen_matrix(onesVector, xcond_vec[m]) * eigen_B;
        
        
        // 计算 softmax
        temp.array().colwise() -= temp.rowwise().maxCoeff().array();
        Eigen::MatrixXd Pimkj = temp.array().exp();
        Pimkj.array().colwise() /= Pimkj.rowwise().sum().array();
        
        
        // MatrixXd Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
        // Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
        // Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
        
        // 拼接结果
        Pi_minus_1_j.conservativeResize(npeop, Pi_minus_1_j.cols() + Pimkj.cols() - 1);
        Pi_minus_1_j.rightCols(Pimkj.cols() - 1) = Pimkj.leftCols(Pimkj.cols() - 1);
        
        Pi_j.conservativeResize(npeop, Pi_j.cols() + Pimkj.cols());
        Pi_j.rightCols(Pimkj.cols()) = Pimkj;
      }
      
      
      // temp_matrix= Pi_j.array()*Y_comp.array()+(1-Y_comp.array());
      // cond_prob.col(j)=temp_matrix.array().log().rowwise().sum().exp();
      
      Pi_minus_1[j] = Pi_minus_1_j;
      cond_prob.col(j) = ( Pi_j.array() * Y_comp.array() + (1 - Y_comp.array()) )
               .log().rowwise().sum().exp();
      
      
      // 第二部分的計算可以更簡潔
      cond_dist[j] = (e_comp * Pi_j.array().log().matrix().transpose())
                                   .array().exp().transpose();
    }
    
    return {{cond_prob}, Pi_minus_1,cond_dist};
    
    // std::vector<std::vector<MatrixXd>> result(2);
    // result[0] = {cond_prob}; // 將 cond_prob 包裝成 vector
    // result[1] = Pi_minus_1;  // 直接賦值
    // return result;
  }   




// [[Rcpp::export]]
VectorXd ipfp_cpp(VectorXd& initial_prob,
                  const int& nitem,
                  const MatrixXd& A_matrix,
                  const MatrixXd& A_matrix_complement,
                  const VectorXd& margin,
                  const int& maxiter,
                  const double& tol) {
  
  // MatrixXd initial_prob_matrix= MatrixXd::Zero(npeop, H);
  // VectorXd pseo_matrix;
  
  // double pseo_margin;
  double scale;
  // VectorXd pseo_matrix(initial_prob.size());
  double row_sum;
  
  for(int iter = 0; iter < maxiter; ++iter){
    for(int m = 0; m < nitem; ++m) {
      // // double pseo_margin=A_matrix.row(m)*initial_prob;
      // // double scale = margin[m] / pseo_margin;
      // pseo_margin=A_matrix.row(m)*initial_prob;
      // scale = margin[m] / pseo_margin;
      // 
      // // initial_prob = (  A_matrix.row(m) * ( margin[m]/sum(initial_prob*A_matrix.row(m)) ) +
      // //   A_matrix_complement.row(m) )*initial_prob;
      // // VectorXd pseo_matrix= scale*A_matrix.row(m)+A_matrix_complement.row(m);
      // pseo_matrix= scale*A_matrix.row(m)+A_matrix_complement.row(m);
      // initial_prob = pseo_matrix.array()*initial_prob.array();
      
      
      
      // const double pseo_margin = A_matrix.row(m).dot(initial_prob);
      // const double scale = margin[m] / pseo_margin;
      
      // pseo_margin = A_matrix.row(m).dot(initial_prob);
      scale = margin[m] / A_matrix.row(m).dot(initial_prob);
      
      // pseo_matrix= scale*A_matrix.row(m)+A_matrix_complement.row(m);
      // // initial_prob = pseo_matrix.array()*initial_prob.array();
      // 
      // initial_prob = initial_prob.cwiseProduct(pseo_matrix);
      
      // initial_prob = (scale*A_matrix.row(m)+A_matrix_complement.row(m)).array()*initial_prob.array();
      initial_prob.array() *= (scale*A_matrix.row(m) + 
        A_matrix_complement.row(m)).array();
      
      
      // // 更高效的計算方式
      // initial_prob = initial_prob.cwiseProduct(
      //   scale * A_matrix.row(m) + A_matrix_complement.row(m));
    }
    
    // 正規化概率向量
    // initial_prob=initial_prob/initial_prob.sum();
    initial_prob /= initial_prob.sum();
    
    
    // 检查收敛条件
    bool converged = true;
    // VectorXd row_sum=A_matrix*initial_prob;
    for (int m = 0; m < nitem; ++m) {
      // double row_sum=A_matrix.row(m)*initial_prob;
      // row_sum=A_matrix.row(m)*initial_prob;
      row_sum=A_matrix.row(m).dot(initial_prob);
      
      if (std::abs(row_sum - margin[m]) >= tol) {
        converged = false;
        break;
      }
    }
    
    if (converged) break;
  }
  
  return initial_prob;
}





// [[Rcpp::export]]
MatrixXd calculateCondDist_2(const MatrixXd& e_tran_cor_nitem, 
                             const VectorXd& tau_j,
                             const std::vector<MatrixXd>& Uij,
                             const int& nitem,
                             const MatrixXd& A_matrix, 
                             const MatrixXd& A_matrix_complement,
                             const MatrixXd& margin_matrix, 
                             const int& maxiter, 
                             const double& tol, 
                             const int& npeop, 
                             const VectorXi& nxcond,
                             const int& H) {
  
  MatrixXd cond_dist_j = MatrixXd::Zero(npeop, H);
  
  if ( nxcond.sum() > 0 ) {
    
    
    for (int i = 0; i < npeop; ++i) {
      // VectorXd exponent = e_tran_cor_nitem * tau.col(j);
      // 
      // VectorXd prob_row = exponent.array() - exponent.maxCoeff();
      // 
      // prob_row=prob_row.array().exp();
      
      
      
      // prob_row=prob_row.array()/prob_row.sum();
      // // VectorXd prob_row_i=prob_row.array()+ (1e-300);
      // prob_row=prob_row.array()+ (1e-300);
      
      VectorXd Omaga_ij=Uij[i]*tau_j;
      VectorXd exponent = e_tran_cor_nitem * Omaga_ij;
      
      // VectorXd exponent = e_tran_cor_nitem * tau.col(j);
      // double max_exp = exponent.maxCoeff();
      VectorXd prob_row = (exponent.array() - exponent.maxCoeff()).exp();
      // double sum_exp = prob_row.sum();
      prob_row = (prob_row / prob_row.sum()).array() + 
        std::numeric_limits<double>::min();
      
      // VectorXd margin_matrix_row_i=margin_matrix.row(i);
      
      cond_dist_j.row(i).noalias() = 
        ipfp_cpp(prob_row,
                 nitem, 
                 A_matrix,
                 A_matrix_complement, 
                 margin_matrix.row(i),
                 maxiter, 
                 tol);
    }
    
  } else {
    
    // VectorXd exponent = e_tran_cor_nitem * tau.col(j);
    // 
    // VectorXd prob_row = exponent.array() - exponent.maxCoeff();
    // 
    // prob_row=prob_row.array().exp();
    
    
    
    // prob_row=prob_row.array()/prob_row.sum();
    // // VectorXd prob_row_i=prob_row.array()+ (1e-300);
    // prob_row=prob_row.array()+ (1e-300);
    
    VectorXd Omaga_j=Uij[0]*tau_j;
    VectorXd exponent = e_tran_cor_nitem * Omaga_j;
    
    
    
    // double max_exp = exponent.maxCoeff();
    VectorXd prob_row = (exponent.array() - exponent.maxCoeff()).exp();
    
    
    // double sum_exp = prob_row.sum();
    prob_row = (prob_row / prob_row.sum()).array() + 
      std::numeric_limits<double>::min();
    
    // Rcpp::Rcout << prob_row << std::endl;
    
    // 單次 IPFP 計算
    VectorXd ipfp_temp=ipfp_cpp(prob_row, 
                                nitem, 
                                A_matrix,
                                A_matrix_complement, 
                                margin_matrix.row(0),
                                maxiter, 
                                tol);
    
    // cond_dist_j= ipfp_temp.transpose().replicate(npeop, 1);
    // 4. 避免显式转置+复制，直接构造结果矩阵
    cond_dist_j.noalias() = ipfp_temp.transpose().colwise().replicate(npeop);
  }
  
  return cond_dist_j;
}




// [[Rcpp::export]]
MatrixXd calculateCondDistParallel_2(const MatrixXd& e_tran_cor_nitem, 
                                     const VectorXd& tau_j,
                                     const std::vector<MatrixXd>& Uij,
                                     const int& nitem,
                                     const MatrixXd& A_matrix, 
                                     const MatrixXd& A_matrix_complement,
                                     const MatrixXd& margin_matrix, 
                                     const int& maxiter, 
                                     const double& tol, 
                                     const int& npeop, 
                                     const VectorXi& nxcond,
                                     const int& H,
                                     const int ncores){
  MatrixXd cond_dist_j = MatrixXd::Zero(npeop, H);
  
  if ( nxcond.sum() > 0 ) {
    
#pragma omp parallel for num_threads(ncores)    
    for (int i = 0; i < npeop; ++i) {
      // VectorXd exponent = e_tran_cor_nitem * tau.col(j);
      // 
      // VectorXd prob_row = exponent.array() - exponent.maxCoeff();
      // 
      // prob_row=prob_row.array().exp();
      
      
      
      // prob_row=prob_row.array()/prob_row.sum();
      // // VectorXd prob_row_i=prob_row.array()+ (1e-300);
      // prob_row=prob_row.array()+ (1e-300);
      
      VectorXd Omaga_ij=Uij[i]*tau_j;
      VectorXd exponent = e_tran_cor_nitem * Omaga_ij;
      
      // VectorXd exponent = e_tran_cor_nitem * tau.col(j);
      // double max_exp = exponent.maxCoeff();
      VectorXd prob_row = (exponent.array() - exponent.maxCoeff()).exp();
      // double sum_exp = prob_row.sum();
      prob_row = (prob_row / prob_row.sum()).array() + 
        std::numeric_limits<double>::min();
      
      // VectorXd margin_matrix_row_i=margin_matrix.row(i);
      
      cond_dist_j.row(i).noalias() = 
        ipfp_cpp(prob_row,
                 nitem, 
                 A_matrix,
                 A_matrix_complement, 
                 margin_matrix.row(i),
                 maxiter, 
                 tol);
    }
    
  } else {
    
    // VectorXd exponent = e_tran_cor_nitem * tau.col(j);
    // 
    // VectorXd prob_row = exponent.array() - exponent.maxCoeff();
    // 
    // prob_row=prob_row.array().exp();
    
    
    
    // prob_row=prob_row.array()/prob_row.sum();
    // // VectorXd prob_row_i=prob_row.array()+ (1e-300);
    // prob_row=prob_row.array()+ (1e-300);
    
    VectorXd Omaga_j=Uij[0]*tau_j;
    VectorXd exponent = e_tran_cor_nitem * Omaga_j;
    
    
    
    // double max_exp = exponent.maxCoeff();
    VectorXd prob_row = (exponent.array() - exponent.maxCoeff()).exp();
    
    
    // double sum_exp = prob_row.sum();
    prob_row = (prob_row / prob_row.sum()).array() + 
      std::numeric_limits<double>::min();
    
    // Rcpp::Rcout << prob_row << std::endl;
    
    // 單次 IPFP 計算
    VectorXd ipfp_temp=ipfp_cpp(prob_row, 
                                nitem, 
                                A_matrix,
                                A_matrix_complement, 
                                margin_matrix.row(0),
                                maxiter, 
                                tol);
    
    // cond_dist_j= ipfp_temp.transpose().replicate(npeop, 1);
    // 4. 避免显式转置+复制，直接构造结果矩阵
    cond_dist_j.noalias() = ipfp_temp.transpose().colwise().replicate(npeop);
  }
  
  return cond_dist_j;
  
}




// [[Rcpp::export]]
List conditional_prob_tau_2_cpp(const List& alpha0,
                                const List& alpha,
                                const std::vector<VectorXd>& tau,
                                const std::vector<std::vector<MatrixXd>>& U,
                                const List& xcond,
                                const VectorXi& nxcond,
                                const int& npeop,
                                const int& nitem,
                                const int& nclass,
                                const int& H,
                                const MatrixXd& Y_cor,
                                const MatrixXd& e_tran_cor_nitem,
                                const MatrixXd& A_matrix,
                                const MatrixXd& A_matrix_complement,
                                const int& maxiter,
                                const double& tol){
  
  
  std::vector<MatrixXd> Pi_minus_1(nclass);
  std::vector<MatrixXd> cond_dist(nclass);
  
  // List cond_dist_test(nclass);
  
  
  // onesVector可以寫在外面
  const MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  
  
  // MatrixXd Pi_minus_1_j=MatrixXd::Zero(npeop, nxcond.sum()-nitem);
  
  // 是否給起始值全為0
  MatrixXd cond_prob(npeop, nclass);
  
  MatrixXd Pimkj;
  MatrixXd temp;
  
  
  for (int j = 0; j < nclass; ++j) {
    
    
    MatrixXd Pi_minus_1_j(npeop, 0);
    
    
    for (int m = 0; m < nitem; ++m) {
      
      
      // Rf_isNull(xcond[m])
      if ( nxcond[m]==0 ) {
        const MatrixXd& alpha0_m = alpha0[m];
        
        
        Eigen::MatrixXd eigen_B(1, alpha0_m.cols() + 1);
        eigen_B.leftCols(alpha0_m.cols()) = alpha0_m.row(j);
        eigen_B.rightCols(1).setZero();
        
        temp = onesVector * eigen_B;
      }else{
        const MatrixXd& alpha0_m = alpha0[m];
        const MatrixXd& alpha_m = alpha[m];
        const MatrixXd& xcond_m = xcond[m];
        
        
        
        // 直接构造最终矩阵 eigen_B，避免所有中间临时矩阵
        MatrixXd eigen_B(alpha_m.rows() + 1, alpha_m.cols() + 1);
        
        // 填充上半部分：[alpha0[m].row(j); alpha[m]]
        eigen_B.topRows(1) = alpha0_m.row(j);      // 第一行 = alpha0[m].row(j)
        eigen_B.bottomRows(alpha_m.rows()) = alpha_m; // 剩余行 = alpha[m]
        
        // 最右侧补零列
        eigen_B.rightCols(1).setZero();
        
        temp = cbindEigen_matrix(onesVector,xcond_m) * eigen_B;
      }
      
      // 计算 softmax
      temp.array().colwise() -= temp.rowwise().maxCoeff().array();
      // Eigen::MatrixXd Pimkj = temp.array().exp();
      Pimkj = temp.array().exp();
      Pimkj.array().colwise() /= Pimkj.rowwise().sum().array();
      
      
      // MatrixXd Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
      // Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
      // Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
      
      // 拼接结果
      Pi_minus_1_j.conservativeResize(npeop, Pi_minus_1_j.cols() + Pimkj.cols() - 1);
      Pi_minus_1_j.rightCols(Pimkj.cols() - 1) = Pimkj.leftCols(Pimkj.cols() - 1);
      
      // Pi_j.conservativeResize(npeop, Pi_j.cols() + Pimkj.cols());
      // Pi_j.rightCols(Pimkj.cols()) = Pimkj;
      
    }
    
    
    Pi_minus_1[j]=Pi_minus_1_j;
    
    
    
    
    
    cond_dist[j]=calculateCondDist_2(e_tran_cor_nitem,
                                     tau[j],
                                     U[j],
                                     nitem,
                                     A_matrix,
                                     A_matrix_complement,
                                     Pi_minus_1_j,
                                     maxiter,
                                     tol,
                                     npeop,
                                     nxcond,
                                     H);
    
    
    cond_prob.col(j) =( cond_dist[j].cwiseProduct(Y_cor) ).rowwise().sum().array()+
      std::numeric_limits<double>::min();
  }
  
  return List::create(Named("cond_prob") = cond_prob,
                      Named("Pi_minus_1") = Pi_minus_1,
                      Named("cond_dist") = cond_dist);
}




// [[Rcpp::export]]
std::vector<std::vector<MatrixXd>> 
  conditional_prob_tau_2_for_omp_cpp(const std::vector<MatrixXd>& alpha0,
                                     const std::vector<MatrixXd>& alpha,
                                     const std::vector<VectorXd>& tau,
                                     const std::vector<std::vector<MatrixXd>>& U,
                                     const std::vector<MatrixXd> xcond_vec,
                                     const VectorXi& nxcond,
                                     const int& npeop,
                                     const int& nitem,
                                     const int& nclass,
                                     const int& H,
                                     const MatrixXd& Y_cor,
                                     const MatrixXd& e_tran_cor_nitem,
                                     const MatrixXd& A_matrix,
                                     const MatrixXd& A_matrix_complement,
                                     const int& maxiter,
                                     const double& tol){
    
    
  // 先暫定為tau
  std::vector<MatrixXd> Pi_minus_1(nclass);
  std::vector<MatrixXd> cond_dist(nclass);
  
  // List cond_dist_test(nclass);
  
  
  // onesVector可以寫在外面
  const MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  
  
  // MatrixXd Pi_minus_1_j=MatrixXd::Zero(npeop, nxcond.sum()-nitem);
  
  // 是否給起始值全為0
  MatrixXd cond_prob(npeop, nclass);
  
  MatrixXd Pimkj;
  MatrixXd temp;
  Eigen::MatrixXd eigen_B;
  
  for (int j = 0; j < nclass; ++j) {
    
    MatrixXd Pi_minus_1_j(npeop, 0);
    
    for (int m = 0; m < nitem; ++m) {
      
      // Rf_isNull(xcond[m])
      if ( nxcond[m]==0 ) {
        
        eigen_B.resize(1, alpha0[m].cols() + 1);
        
        // Eigen::MatrixXd eigen_B(1, alpha0[m].cols() + 1);
        // eigen_B(1, alpha0[m].cols() + 1);
        eigen_B.leftCols(alpha0[m].cols()) = alpha0[m].row(j);
        eigen_B.rightCols(1).setZero();
        
        temp = onesVector * eigen_B;
        
      }else{
        eigen_B.resize(alpha[m].rows() + 1, alpha[m].cols() + 1);
        
        // Eigen::MatrixXd eigen_B(alpha[m].rows() + 1, alpha[m].cols() + 1);
        // eigen_B(alpha[m].rows() + 1, alpha[m].cols() + 1);
        // 填充上半部分：[alpha0[m].row(j); alpha[m]]
        eigen_B.topRows(1) = alpha0[m].row(j);      // 第一行 = alpha0[m].row(j)
        eigen_B.bottomRows(alpha[m].rows()) = alpha[m]; // 剩余行 = alpha[m]
        
        // 最右侧补零列
        eigen_B.rightCols(1).setZero();
        // MatrixXd eigen_B = as<Map<MatrixXd> >( cbind(rbind(alpha0_m.row(j), alpha_m), 0.0) );
        
        // temp = eigen_A * eigen_B;
        temp = cbindEigen_matrix(onesVector,xcond_vec[m]) * eigen_B;
      }
      
      // 计算 softmax
      temp.array().colwise() -= temp.rowwise().maxCoeff().array();
      // Eigen::MatrixXd Pimkj = temp.array().exp();
      Pimkj = temp.array().exp();
      Pimkj.array().colwise() /= Pimkj.rowwise().sum().array();
      
      
      // MatrixXd Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
      // Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
      // Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
      
      // 拼接结果
      Pi_minus_1_j.conservativeResize(npeop, Pi_minus_1_j.cols() + Pimkj.cols() - 1);
      Pi_minus_1_j.rightCols(Pimkj.cols() - 1) = Pimkj.leftCols(Pimkj.cols() - 1);
      
      // Pi_j.conservativeResize(npeop, Pi_j.cols() + Pimkj.cols());
      // Pi_j.rightCols(Pimkj.cols()) = Pimkj;
      
    }
    
    
    Pi_minus_1[j]=Pi_minus_1_j;
    
    cond_dist[j]=calculateCondDist_2(e_tran_cor_nitem,
                                     tau[j],
                                     U[j],
                                     nitem,
                                     A_matrix,
                                     A_matrix_complement,
                                     Pi_minus_1_j,
                                     maxiter,
                                     tol,
                                     npeop,
                                     nxcond,
                                     H);
    
    
    
    cond_prob.col(j) =( cond_dist[j].cwiseProduct(Y_cor) ).rowwise().sum().array()+
      std::numeric_limits<double>::min();
  }
  
  
  return {{cond_prob}, Pi_minus_1, cond_dist};
}




// [[Rcpp::export]]
List conditional_prob_tau_Parallel_2_cpp(const List& alpha0,
                                         const List& alpha,
                                         const std::vector<VectorXd>& tau,
                                         const std::vector<std::vector<MatrixXd>>& U,
                                         const List& xcond,
                                         const VectorXi& nxcond,
                                         const int& npeop,
                                         const int& nitem,
                                         const int& nclass,
                                         const int& H,
                                         const MatrixXd& Y_cor,
                                         const MatrixXd& e_tran_cor_nitem,
                                         const MatrixXd& A_matrix,
                                         const MatrixXd& A_matrix_complement,
                                         const int& maxiter,
                                         const double& tol,
                                         const int ncores){
  // 先暫定為tau
  std::vector<MatrixXd> Pi_minus_1(nclass);
  std::vector<MatrixXd> cond_dist(nclass);
  
  // List cond_dist_test(nclass);
  
  
  // onesVector可以寫在外面
  const MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  
  
  // MatrixXd Pi_minus_1_j=MatrixXd::Zero(npeop, nxcond.sum()-nitem);
  
  // 是否給起始值全為0
  MatrixXd cond_prob(npeop, nclass);
  
  MatrixXd Pimkj;
  MatrixXd temp;
  
  
  for (int j = 0; j < nclass; ++j) {
    
    
    MatrixXd Pi_minus_1_j(npeop, 0);
    
    
    for (int m = 0; m < nitem; ++m) {
      
      
      // Rf_isNull(xcond[m])
      if ( nxcond[m]==0 ) {
        const MatrixXd& alpha0_m = alpha0[m];
        
        
        Eigen::MatrixXd eigen_B(1, alpha0_m.cols() + 1);
        eigen_B.leftCols(alpha0_m.cols()) = alpha0_m.row(j);
        eigen_B.rightCols(1).setZero();
        
        temp = onesVector * eigen_B;
      }else{
        const MatrixXd& alpha0_m = alpha0[m];
        const MatrixXd& alpha_m = alpha[m];
        const MatrixXd& xcond_m = xcond[m];
        
        
        
        // 直接构造最终矩阵 eigen_B，避免所有中间临时矩阵
        MatrixXd eigen_B(alpha_m.rows() + 1, alpha_m.cols() + 1);
        
        // 填充上半部分：[alpha0[m].row(j); alpha[m]]
        eigen_B.topRows(1) = alpha0_m.row(j);      // 第一行 = alpha0[m].row(j)
        eigen_B.bottomRows(alpha_m.rows()) = alpha_m; // 剩余行 = alpha[m]
        
        // 最右侧补零列
        eigen_B.rightCols(1).setZero();
        
        temp = cbindEigen_matrix(onesVector,xcond_m) * eigen_B;
      }
      
      // 计算 softmax
      temp.array().colwise() -= temp.rowwise().maxCoeff().array();
      // Eigen::MatrixXd Pimkj = temp.array().exp();
      Pimkj = temp.array().exp();
      Pimkj.array().colwise() /= Pimkj.rowwise().sum().array();
      
      
      // MatrixXd Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
      // Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
      // Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
      
      // 拼接结果
      Pi_minus_1_j.conservativeResize(npeop, Pi_minus_1_j.cols() + Pimkj.cols() - 1);
      Pi_minus_1_j.rightCols(Pimkj.cols() - 1) = Pimkj.leftCols(Pimkj.cols() - 1);
      
      // Pi_j.conservativeResize(npeop, Pi_j.cols() + Pimkj.cols());
      // Pi_j.rightCols(Pimkj.cols()) = Pimkj;
      
    }
    
    
    Pi_minus_1[j]=Pi_minus_1_j;
    
    cond_dist[j]=calculateCondDistParallel_2(e_tran_cor_nitem,
                                             tau[j],
                                             U[j],
                                             nitem,
                                             A_matrix,
                                             A_matrix_complement,
                                             Pi_minus_1_j,
                                             maxiter,
                                             tol,
                                             npeop,
                                             nxcond,
                                             H,
                                             ncores);
    
    
    
    cond_prob.col(j) =( cond_dist[j].cwiseProduct(Y_cor) ).rowwise().sum().array()+
      std::numeric_limits<double>::min();
  }
  
  
  
  return List::create(Named("cond_prob") = cond_prob,
                      Named("Pi_minus_1") = Pi_minus_1,
                      Named("cond_dist") = cond_dist);
}




// [[Rcpp::export]]
List conditional_prob_tau_j_2_cpp(const List& alpha0,
                                  const List& alpha,
                                  const std::vector<VectorXd>& tau,
                                  const std::vector<std::vector<MatrixXd>>& U,
                                  const List& xcond,
                                  const VectorXi& nxcond,
                                  const int& npeop,
                                  const int& nitem,
                                  const int& nclass,
                                  const int& H,
                                  const MatrixXd& Y_cor,
                                  const MatrixXd& e_tran_cor_nitem,
                                  const MatrixXd& A_matrix,
                                  const MatrixXd& A_matrix_complement,
                                  const int& maxiter,
                                  const double& tol,
                                  const int& j){
  // 先暫定為tau
  
  
  // MatrixXd cond_prob(npeop, nclass);
  // MatrixXd cond_prob = MatrixXd::Zero(npeop, nclass);
  VectorXd cond_prob_j=VectorXd::Zero(npeop);
  // onesVector可以寫在外面
  MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  
  
  
  MatrixXd Pimkj;
  MatrixXd temp;
  
  
  MatrixXd Pi_minus_1_j(npeop, 0);
  
  
  for (int m = 0; m < nitem; ++m) {
    
    
    // Rf_isNull(xcond[m])
    if ( nxcond[m]==0 ) {
      const MatrixXd& alpha0_m = alpha0[m];
      
      
      Eigen::MatrixXd eigen_B(1, alpha0_m.cols() + 1);
      eigen_B.leftCols(alpha0_m.cols()) = alpha0_m.row(j);
      eigen_B.rightCols(1).setZero();
      
      temp = onesVector * eigen_B;
      
    }else{
      const MatrixXd& alpha0_m = alpha0[m];
      const MatrixXd& alpha_m = alpha[m];
      const MatrixXd& xcond_m = xcond[m];
      
      
      
      // 直接构造最终矩阵 eigen_B，避免所有中间临时矩阵
      MatrixXd eigen_B(alpha_m.rows() + 1, alpha_m.cols() + 1);
      
      // 填充上半部分：[alpha0[m].row(j); alpha[m]]
      eigen_B.topRows(1) = alpha0_m.row(j);      // 第一行 = alpha0[m].row(j)
      eigen_B.bottomRows(alpha_m.rows()) = alpha_m; // 剩余行 = alpha[m]
      
      // 最右侧补零列
      eigen_B.rightCols(1).setZero();
      
      temp = cbindEigen_matrix(onesVector,xcond_m) * eigen_B;
      
      
      
    }
    
    // 计算 softmax
    temp.array().colwise() -= temp.rowwise().maxCoeff().array();
    // Eigen::MatrixXd Pimkj = temp.array().exp();
    Pimkj = temp.array().exp();
    Pimkj.array().colwise() /= Pimkj.rowwise().sum().array();
    
    
    // MatrixXd Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
    // Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
    // Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
    
    // 拼接结果
    Pi_minus_1_j.conservativeResize(npeop, Pi_minus_1_j.cols() + Pimkj.cols() - 1);
    Pi_minus_1_j.rightCols(Pimkj.cols() - 1) = Pimkj.leftCols(Pimkj.cols() - 1);
    
    // Pi_j.conservativeResize(npeop, Pi_j.cols() + Pimkj.cols());
    // Pi_j.rightCols(Pimkj.cols()) = Pimkj;
    
  }
  
  MatrixXd cond_dist_j=calculateCondDist_2(e_tran_cor_nitem,
                                           tau[j],
                                           U[j],
                                           nitem,
                                           A_matrix,
                                           A_matrix_complement,
                                           Pi_minus_1_j,
                                           maxiter,
                                           tol,
                                           npeop,
                                           nxcond,
                                           H);
  
  
  // Map<MatrixXd> cond_dist_j_Xd(as<Map<MatrixXd> >(cond_dist_j));
  // MatrixXd cond_dist_j_Xd=cond_dist[j];
  cond_prob_j=(cond_dist_j.cwiseProduct(Y_cor)).rowwise().sum().array()+
    std::numeric_limits<double>::min();
  
  
  
  return List::create(Named("cond_prob_j") = cond_prob_j,
                      Named("Pi_minus_1_j") = Pi_minus_1_j,
                      Named("cond_dist_j") = cond_dist_j);
}




// [[Rcpp::export]]
std::vector<MatrixXd>
  conditional_prob_tau_j_2_for_omp_cpp(const std::vector<MatrixXd>& alpha0,
                                       const std::vector<MatrixXd>& alpha,
                                       const std::vector<VectorXd>& tau,
                                       const std::vector<std::vector<MatrixXd>>& U,
                                       const std::vector<MatrixXd> xcond_vec,
                                       const VectorXi& nxcond,
                                       const int& npeop,
                                       const int& nitem,
                                       const int& nclass,
                                       const int& H,
                                       const MatrixXd& Y_cor,
                                       const MatrixXd& e_tran_cor_nitem,
                                       const MatrixXd& A_matrix,
                                       const MatrixXd& A_matrix_complement,
                                       const int& maxiter,
                                       const double& tol,
                                       const int& j){
  // 先暫定為tau
  
  
  // MatrixXd cond_prob(npeop, nclass);
  // MatrixXd cond_prob = MatrixXd::Zero(npeop, nclass);
  VectorXd cond_prob_j=VectorXd::Zero(npeop);
  // onesVector可以寫在外面
  MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  
  
  
  MatrixXd Pimkj;
  MatrixXd temp;
  
  
  MatrixXd Pi_minus_1_j(npeop, 0);
  
  
  for (int m = 0; m < nitem; ++m) {
    
    
    // Rf_isNull(xcond[m])
    if ( nxcond[m]==0 ) {
      
      Eigen::MatrixXd eigen_B(1, alpha0[m].cols() + 1);
      eigen_B.leftCols(alpha0[m].cols()) = alpha0[m].row(j);
      eigen_B.rightCols(1).setZero();
      
      temp = onesVector * eigen_B;
      
    }else{
      
      // 直接构造最终矩阵 eigen_B，避免所有中间临时矩阵
      MatrixXd eigen_B(alpha[m].rows() + 1, alpha[m].cols() + 1);
      
      // 填充上半部分：[alpha0[m].row(j); alpha[m]]
      eigen_B.topRows(1) = alpha0[m].row(j);      // 第一行 = alpha0[m].row(j)
      eigen_B.bottomRows(alpha[m].rows()) = alpha[m]; // 剩余行 = alpha[m]
      
      // 最右侧补零列
      eigen_B.rightCols(1).setZero();
      
      temp = cbindEigen_matrix(onesVector,xcond_vec[m]) * eigen_B;
    }
    
    
    // 计算 softmax
    temp.array().colwise() -= temp.rowwise().maxCoeff().array();
    // Eigen::MatrixXd Pimkj = temp.array().exp();
    Pimkj = temp.array().exp();
    Pimkj.array().colwise() /= Pimkj.rowwise().sum().array();
    
    
    // MatrixXd Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
    // Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
    // Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
    
    // 拼接结果
    Pi_minus_1_j.conservativeResize(npeop, Pi_minus_1_j.cols() + Pimkj.cols() - 1);
    Pi_minus_1_j.rightCols(Pimkj.cols() - 1) = Pimkj.leftCols(Pimkj.cols() - 1);
    
    // Pi_j.conservativeResize(npeop, Pi_j.cols() + Pimkj.cols());
    // Pi_j.rightCols(Pimkj.cols()) = Pimkj;
    
  }
  
  MatrixXd cond_dist_j=calculateCondDist_2(e_tran_cor_nitem,
                                           tau[j],
                                           U[j],
                                           nitem,
                                           A_matrix,
                                           A_matrix_complement,
                                           Pi_minus_1_j,
                                           maxiter,
                                           tol,
                                           npeop,
                                           nxcond,
                                           H);
  
  
  // Map<MatrixXd> cond_dist_j_Xd(as<Map<MatrixXd> >(cond_dist_j));
  // MatrixXd cond_dist_j_Xd=cond_dist[j];
  cond_prob_j=(cond_dist_j.cwiseProduct(Y_cor)).rowwise().sum().array()+
    std::numeric_limits<double>::min();
  
  
  return {cond_prob_j.matrix(), Pi_minus_1_j, cond_dist_j};
}



// [[Rcpp::export]]
List conditional_prob_tau_j_Parallel_2_cpp(const List& alpha0,
                                           const List& alpha,
                                           const std::vector<VectorXd>& tau,
                                           const std::vector<std::vector<MatrixXd>>& U,
                                           const List& xcond,
                                           const VectorXi& nxcond,
                                           const int& npeop,
                                           const int& nitem,
                                           const int& nclass,
                                           const int& H,
                                           const MatrixXd& Y_cor,
                                           const MatrixXd& e_tran_cor_nitem,
                                           const MatrixXd& A_matrix,
                                           const MatrixXd& A_matrix_complement,
                                           const int& maxiter,
                                           const double& tol,
                                           const int& j,
                                           const int ncores){
  // 先暫定為tau
  
  
  // MatrixXd cond_prob(npeop, nclass);
  // MatrixXd cond_prob = MatrixXd::Zero(npeop, nclass);
  VectorXd cond_prob_j=VectorXd::Zero(npeop);
  // onesVector可以寫在外面
  MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  
  
  MatrixXd Pimkj;
  MatrixXd temp;
  
  
  MatrixXd Pi_minus_1_j(npeop, 0);
  
  for (int m = 0; m < nitem; ++m) {
    
    // Rf_isNull(xcond[m])
    if ( nxcond[m]==0 ) {
      const MatrixXd& alpha0_m = alpha0[m];
      
      
      Eigen::MatrixXd eigen_B(1, alpha0_m.cols() + 1);
      eigen_B.leftCols(alpha0_m.cols()) = alpha0_m.row(j);
      eigen_B.rightCols(1).setZero();
      
      temp = onesVector * eigen_B;
      
    }else{
      const MatrixXd& alpha0_m = alpha0[m];
      const MatrixXd& alpha_m = alpha[m];
      const MatrixXd& xcond_m = xcond[m];
      
      
      
      // 直接构造最终矩阵 eigen_B，避免所有中间临时矩阵
      MatrixXd eigen_B(alpha_m.rows() + 1, alpha_m.cols() + 1);
      
      // 填充上半部分：[alpha0[m].row(j); alpha[m]]
      eigen_B.topRows(1) = alpha0_m.row(j);      // 第一行 = alpha0[m].row(j)
      eigen_B.bottomRows(alpha_m.rows()) = alpha_m; // 剩余行 = alpha[m]
      
      // 最右侧补零列
      eigen_B.rightCols(1).setZero();
      
      temp = cbindEigen_matrix(onesVector,xcond_m) * eigen_B;
    }
    
    // 计算 softmax
    temp.array().colwise() -= temp.rowwise().maxCoeff().array();
    // Eigen::MatrixXd Pimkj = temp.array().exp();
    Pimkj = temp.array().exp();
    Pimkj.array().colwise() /= Pimkj.rowwise().sum().array();
    
    
    // MatrixXd Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
    // Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
    // Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
    
    // 拼接结果
    Pi_minus_1_j.conservativeResize(npeop, Pi_minus_1_j.cols() + Pimkj.cols() - 1);
    Pi_minus_1_j.rightCols(Pimkj.cols() - 1) = Pimkj.leftCols(Pimkj.cols() - 1);
    
    // Pi_j.conservativeResize(npeop, Pi_j.cols() + Pimkj.cols());
    // Pi_j.rightCols(Pimkj.cols()) = Pimkj;
    
  }
  
  MatrixXd cond_dist_j=calculateCondDistParallel_2(e_tran_cor_nitem,
                                                   tau[j],
                                                   U[j],
                                                   nitem,
                                                   A_matrix,
                                                   A_matrix_complement,
                                                   Pi_minus_1_j,
                                                   maxiter,
                                                   tol,
                                                   npeop,
                                                   nxcond,
                                                   H,
                                                   ncores);
  
  
  // Map<MatrixXd> cond_dist_j_Xd(as<Map<MatrixXd> >(cond_dist_j));
  // MatrixXd cond_dist_j_Xd=cond_dist[j];
  cond_prob_j=(cond_dist_j.cwiseProduct(Y_cor)).rowwise().sum().array()+
    std::numeric_limits<double>::min();
  
  
  
  return List::create(Named("cond_prob_j") = cond_prob_j,
                      Named("Pi_minus_1_j") = Pi_minus_1_j,
                      Named("cond_dist_j") = cond_dist_j);
}





// Function to transform alpha_vector to a list
// [[Rcpp::export]]
List tran_vec_to_list_cpp(const VectorXd& alpha_vector,
                          const VectorXi& nlevels,
                          const VectorXi& nxcond,
                          const int& num_alpha0,
                          const int& nitem,
                          const int& nclass) {
  
  
  List alpha0_p(nitem);
  List alpha_p(nitem);
  
  
  
  // 分割 alpha_vector 为 alpha0 和 alpha
  // VectorXd alpha0 = alpha_vector.head(num_alpha0);
  // VectorXd alpha = alpha_vector.tail(num_alpha);
  
  
  int temp_1=0;
  int temp_3=num_alpha0;
  
  int temp_2;
  int temp_4;
  
  for(int m = 0; m < nitem; ++m ){
    
    temp_2=nclass*(nlevels[m] - 1);
    
    // 0是起始索引，nclass*(nlevels[0] - 1)是元素数量
    MatrixXd temp_A = alpha_vector.segment(temp_1, temp_2); 
    
    // Reshape the matrix to the specified dimensions
    temp_A.resize(nclass, (nlevels[m] - 1));
    // 這兩式無法合在一起
    alpha0_p[m]=temp_A;
    
    
    
    temp_1=temp_1+temp_2;
    
    
    if(nxcond[m]!=0){
      
      temp_4=nxcond[m]*(nlevels[m] - 1);
      
      // 0是起始索引，nclass*(nlevels[0] - 1)是元素数量
      MatrixXd temp_B = alpha_vector.segment(temp_3, temp_4); 
      
      // Reshape the matrix to the specified dimensions
      temp_B.resize(nxcond[m], (nlevels[m] - 1));
      // 這兩式無法合在一起
      alpha_p[m]=temp_B;
      
      
      temp_3=temp_3+temp_4;
    }
    
  }
  return List::create(Named("alpha0") = alpha0_p,
                      Named("alpha") = alpha_p);
}




// Function to transform alpha_vector to a list
// [[Rcpp::export]]
std::vector<std::vector<MatrixXd>> 
  tran_vec_to_list_for_omp_cpp(const VectorXd& alpha_vector,
                               const VectorXi& nlevels,
                               const VectorXi& nxcond,
                               const int& num_alpha0,
                               const int& nitem,
                               const int& nclass) {
    
  std::vector<MatrixXd> alpha0_p(nitem);
  std::vector<MatrixXd> alpha_p(nitem);
  // List alpha0_p(nitem);
  // List alpha_p(nitem);
  
  
  
  // 分割 alpha_vector 为 alpha0 和 alpha
  // VectorXd alpha0 = alpha_vector.head(num_alpha0);
  // VectorXd alpha = alpha_vector.tail(num_alpha);
  
  
  int temp_1=0;
  int temp_3=num_alpha0;
  
  int temp_2;
  int temp_4;
  
  for(int m = 0; m < nitem; ++m ){
    
    temp_2=nclass*(nlevels[m] - 1);
    
    // 0是起始索引，nclass*(nlevels[0] - 1)是元素数量
    MatrixXd temp_A = alpha_vector.segment(temp_1, temp_2); 
    
    // Reshape the matrix to the specified dimensions
    temp_A.resize(nclass, (nlevels[m] - 1));
    // 這兩式無法合在一起
    alpha0_p[m]=temp_A;
    
    
    
    temp_1=temp_1+temp_2;
    
    
    if(nxcond[m]!=0){
      
      temp_4=nxcond[m]*(nlevels[m] - 1);
      
      // 0是起始索引，nclass*(nlevels[0] - 1)是元素数量
      MatrixXd temp_B = alpha_vector.segment(temp_3, temp_4); 
      
      // Reshape the matrix to the specified dimensions
      temp_B.resize(nxcond[m], (nlevels[m] - 1));
      // 這兩式無法合在一起
      alpha_p[m]=temp_B;
      
      
      temp_3=temp_3+temp_4;
    }
    
  }
  
  std::vector<std::vector<MatrixXd>> result(2);
  result[0] = alpha0_p; 
  result[1] = alpha_p;  
  return result;
  
}





// [[Rcpp::export]]
VectorXd direction_alpha_like_cpp(const List& Pi_minus_1,
                                  const List& cond_dist_p,
                                  const MatrixXd& h_p,
                                  const MatrixXd& e_tran_cor,
                                  const MatrixXd& y_w,
                                  const MatrixXd& Y,
                                  const List& xcond,
                                  const int& npeop,
                                  const int& nitem,
                                  const int& nclass,
                                  const VectorXi& nlevels,
                                  const VectorXi& nxcond,
                                  const int& H,
                                  const MatrixXd& diag_nclass,
                                  const int& alpha_length) {
  
  
  
  
  VectorXd deri = VectorXd::Zero(alpha_length);
  MatrixXd hessian = MatrixXd::Zero(alpha_length,alpha_length);
  // 這樣後續會有錯
  // VectorXd deri(alpha_length);
  // MatrixXd hessian(alpha_length,alpha_length);
  
  // int cols_Vij_eigen = Y.cols();
  const int cov_var_col = e_tran_cor.cols();
  const int nrow_result = Y.cols();
  
  
  // 预分配循环内变量
  MatrixXd cov_var(cov_var_col, cov_var_col);
  VectorXd margin_prob(cov_var_col);
  MatrixXd V_ij_11(nitem, nitem);
  MatrixXd V_ij_11_ginv(nitem, nitem);
  // MatrixXd eigen_Vh_ij_inver_h(nitem, nitem);
  // VectorXd muij(nitem);
  // VectorXd muij_var(nitem);
  MatrixXd Vij_eigen(nitem, nitem);
  MatrixXd deri_1(nitem, 1);
  
  
  for (int j = 0; j < nclass; ++j) {
    
    MatrixXd diag_nclass_j = diag_nclass.row(j);
    
    MatrixXd Pi_minus_1_j = Pi_minus_1[j];
    MatrixXd S = Y - Pi_minus_1_j;
    
    MatrixXd cond_dist_p_j=cond_dist_p[j];
    
    for (int i = 0; i < npeop; ++i) {
      
      margin_prob.noalias() = cond_dist_p_j.row(i) * e_tran_cor;
      
      // Calculate cov_var
      // MatrixXd cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
      cov_var.setZero();
      for(int h = 0; h < H; ++h) {
        cov_var.noalias() += cond_dist_p_j(i,h) *
          (e_tran_cor.row(h).transpose() - margin_prob) *
          (e_tran_cor.row(h).transpose() - margin_prob).transpose();
      }
      
      V_ij_11.noalias() = cov_var.topLeftCorner(nitem, nitem);
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(V_ij_11);
      V_ij_11_ginv.noalias() = cod.pseudoInverse();
      
      
      // muij = Pi_minus_1_j.row(i);
      // muij_var=muij.array()*(1-muij.array());
      // Vij_eigen=muij_var.asDiagonal();
      
      Vij_eigen=(  Pi_minus_1_j.row(i).array()*
        (1-Pi_minus_1_j.row(i).array()) ).
        matrix().asDiagonal();
      
      // MatrixXd eigen_Vh_ij_inver_h = V_ij_11_ginv * h_p(i,j);
      
      
      
      // MatrixXd deri_1 = eigen_Vh_ij_inver_h * S_i.transpose();
      deri_1.noalias() = h_p(i,j) * V_ij_11_ginv  * S.row(i).transpose();
      
      
      MatrixXd eigen_der_alpha0=KroneckerProduct(Vij_eigen,diag_nclass_j);
      
      
      
      // 計算對alpha微分
      MatrixXd  eigen_deri_mu_ij = eigen_der_alpha0;
      if(nxcond.sum() > 0){
        
        int u = -1;
        // NumericMatrix der_alpha(nitem, 0);
        // MatrixXd der_alpha(nlevels.sum()-nitem, 0);
        MatrixXd der_alpha(nrow_result, 0);
        
        VectorXd xcond_mi;
        
        for(int m = 0; m < nitem; ++m) {
          
          // Rf_isNull(xcond[m])
          // if (nxcond[m]==0) {
          //   // 如果条件满足，跳过当前循环
          //   continue;
          // }
          
          // const NumericMatrix& xcond_m_ref = xcond[m]; // 使用引用
          
          if(nxcond[m]!=0){
            MatrixXd xcond_m = xcond[m];
            xcond_mi= xcond_m.row(i);
            // xcond_mi= xcond_m.row(i);
          }
          
          
          for(int k = 0; k < (nlevels[m] - 1); ++k) {
            
            ++u;
            
            if (nxcond[m]==0) {
              // 如果条件满足，跳过当前循环
              continue;
            }
            
            
            MatrixXd b_temp=Vij_eigen.col(u)*xcond_mi.transpose();
            
            der_alpha = cbindEigen_matrix(der_alpha, b_temp);
          }
        }
        
        eigen_deri_mu_ij = cbindEigen_matrix(eigen_der_alpha0, der_alpha);
      }
      
      
      deri.noalias() += eigen_deri_mu_ij.transpose() * deri_1;
      hessian.noalias() += h_p(i,j) * eigen_deri_mu_ij.transpose() * V_ij_11_ginv *
        eigen_deri_mu_ij;
    }
  }
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hessian);
  
  VectorXd direction_alpha = cod.pseudoInverse() * deri;
  // VectorXd direction_alpha = CompleteOrthogonalDecomposition<MatrixXd>(hessian).solve(deri);
  
  
  // VectorXd direction_alpha = hessian.ldlt().solve(deri);
  return direction_alpha;
  // return  CompleteOrthogonalDecomposition<MatrixXd>(hessian).solve(deri);
}





// [[Rcpp::export]]
VectorXd direction_alpha_like_for_omp_cpp(const std::vector<MatrixXd> Pi_minus_1,
                                          const std::vector<MatrixXd>& cond_dist_p,
                                          const MatrixXd& h_p,
                                          const MatrixXd& e_tran_cor,
                                          const MatrixXd& y_w,
                                          const MatrixXd& Y,
                                          const std::vector<MatrixXd> xcond_vec,
                                          const int& npeop,
                                          const int& nitem,
                                          const int& nclass,
                                          const VectorXi& nlevels,
                                          const VectorXi& nxcond,
                                          const int& H,
                                          const MatrixXd& diag_nclass,
                                          const int& alpha_length) {
  
  VectorXd deri = VectorXd::Zero(alpha_length);
  MatrixXd hessian = MatrixXd::Zero(alpha_length,alpha_length);
  // 這樣後續會有錯
  // VectorXd deri(alpha_length);
  // MatrixXd hessian(alpha_length,alpha_length);
  
  // int cols_Vij_eigen = Y.cols();
  const int cov_var_col = e_tran_cor.cols();
  const int nrow_result = Y.cols();
  
  
  // 预分配循环内变量
  MatrixXd cov_var(cov_var_col, cov_var_col);
  VectorXd margin_prob(cov_var_col);
  MatrixXd V_ij_11(nitem, nitem);
  MatrixXd V_ij_11_ginv(nitem, nitem);
  // MatrixXd eigen_Vh_ij_inver_h(nitem, nitem);
  // VectorXd muij(nitem);
  // VectorXd muij_var(nitem);
  MatrixXd Vij_eigen(nitem, nitem);
  MatrixXd deri_1(nitem, 1);
  
  
  for (int j = 0; j < nclass; ++j) {
    
    MatrixXd diag_nclass_j = diag_nclass.row(j);
    
    MatrixXd S = Y - Pi_minus_1[j];
    
    MatrixXd cond_dist_p_j=cond_dist_p[j];
    
    for (int i = 0; i < npeop; ++i) {
      
      margin_prob.noalias() = cond_dist_p_j.row(i) * e_tran_cor;
      
      // Calculate cov_var
      // MatrixXd cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
      cov_var.setZero();
      for(int h = 0; h < H; ++h) {
        cov_var.noalias() += cond_dist_p_j(i,h) *
          (e_tran_cor.row(h).transpose() - margin_prob) *
          (e_tran_cor.row(h).transpose() - margin_prob).transpose();
      }
      
      V_ij_11.noalias() = cov_var.topLeftCorner(nitem, nitem);
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(V_ij_11);
      V_ij_11_ginv.noalias() = cod.pseudoInverse();
      
      
      // muij = Pi_minus_1_j.row(i);
      // muij_var=muij.array()*(1-muij.array());
      // Vij_eigen=muij_var.asDiagonal();
      
      Vij_eigen=(  Pi_minus_1[j].row(i).array()*
        (1-Pi_minus_1[j].row(i).array()) ).
        matrix().asDiagonal();
      
      // MatrixXd eigen_Vh_ij_inver_h = V_ij_11_ginv * h_p(i,j);
      
      
      
      // MatrixXd deri_1 = eigen_Vh_ij_inver_h * S_i.transpose();
      deri_1.noalias() = h_p(i,j) * V_ij_11_ginv  * S.row(i).transpose();
      
      
      MatrixXd eigen_der_alpha0=KroneckerProduct(Vij_eigen,diag_nclass_j);
      
      
      
      // 計算對alpha微分
      MatrixXd  eigen_deri_mu_ij = eigen_der_alpha0;
      if(nxcond.sum() > 0){
        
        int u = -1;
        // NumericMatrix der_alpha(nitem, 0);
        // MatrixXd der_alpha(nlevels.sum()-nitem, 0);
        MatrixXd der_alpha(nrow_result, 0);
        
        VectorXd xcond_mi;
        
        for(int m = 0; m < nitem; ++m) {
          
          // Rf_isNull(xcond[m])
          // if (nxcond[m]==0) {
          //   // 如果条件满足，跳过当前循环
          //   continue;
          // }
          
          // const NumericMatrix& xcond_m_ref = xcond[m]; // 使用引用
          
          // if(nxcond[m]!=0){
          //   MatrixXd xcond_m = xcond[m];
          //   xcond_mi= xcond_m.row(i);
          //   // xcond_mi= xcond_m.row(i);
          // }
          
          
          for(int k = 0; k < (nlevels[m] - 1); ++k) {
            
            ++u;
            
            if (nxcond[m]==0) {
              // 如果条件满足，跳过当前循环
              continue;
            }
            
            
            MatrixXd b_temp=Vij_eigen.col(u)*xcond_vec[m].row(i);
            
            der_alpha = cbindEigen_matrix(der_alpha, b_temp);
          }
        }
        
        eigen_deri_mu_ij = cbindEigen_matrix(eigen_der_alpha0, der_alpha);
      }
      
      
      deri.noalias() += eigen_deri_mu_ij.transpose() * deri_1;
      hessian.noalias() += h_p(i,j) * eigen_deri_mu_ij.transpose() * V_ij_11_ginv *
        eigen_deri_mu_ij;
    }
  }
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hessian);
  
  VectorXd direction_alpha = cod.pseudoInverse() * deri;
  // VectorXd direction_alpha = CompleteOrthogonalDecomposition<MatrixXd>(hessian).solve(deri);
  
  
  // VectorXd direction_alpha = hessian.ldlt().solve(deri);
  return direction_alpha;
  // return  CompleteOrthogonalDecomposition<MatrixXd>(hessian).solve(deri);
}




// 先定義對 Eigen::MatrixXd 的 reduction
#pragma omp declare reduction(MatrixAdd : MatrixXd : omp_out += omp_in) \
initializer(omp_priv = MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))
  
  
#pragma omp declare reduction(VectorAdd : VectorXd : omp_out += omp_in) \
  initializer(omp_priv = VectorXd::Zero(omp_orig.size()))  
    
    
    
// [[Rcpp::export]]
VectorXd direction_alpha_like_Para_1_cpp(const List& Pi_minus_1,
                                         const List& cond_dist_p,
                                         const MatrixXd& h_p,
                                         const MatrixXd& e_tran_cor,
                                         const MatrixXd& y_w,
                                         const MatrixXd& Y,
                                         const List& xcond,
                                         const int& npeop,
                                         const int& nitem,
                                         const int& nclass,
                                         const VectorXi& nlevels,
                                         const VectorXi& nxcond,
                                         const int& H,
                                         const MatrixXd& diag_nclass,
                                         const int& alpha_length,
                                         const int ncores) {
  
  // // 平行化過程中存取List會出問題
  std::vector<MatrixXd> xcond_vec(xcond.size());
  for (int m = 0; m < xcond.size();++m ) {
    if ( !Rf_isNull(xcond[m]) ) {
      xcond_vec[m] = as<MatrixXd>(xcond[m]);
    }
  }
  
  
  VectorXd eigen_result_deri = VectorXd::Zero(alpha_length);
  MatrixXd eigen_result_hessian = MatrixXd::Zero(alpha_length,alpha_length);
  
  // 這樣後續會有錯
  // VectorXd eigen_result_deri(alpha_length);
  // MatrixXd eigen_result_hessian(alpha_length,alpha_length);
  
  // int cols_Vij_eigen = Y.cols();
  int nrow_result = Y.cols() ;
  // int ncol_result = Y.cols() * nclass;
  int cov_var_col=e_tran_cor.cols();
  // VectorXd direction_tau((cov_var_col-nitem)*nclass);
  // MatrixXd direction_tau((cov_var_col-nitem),nclass);
  
  
  // #pragma omp parallel num_threads(ncores)
  // {
  //   // 外层循环
  // #pragma omp for schedule(dynamic) nowait
  for (int j = 0; j < nclass; ++j) {
    
    
    
    // RowVectorXd diag_nclass_j = diag_nclass.row(j);
    MatrixXd diag_nclass_j = diag_nclass.row(j);
    
    MatrixXd Pi_minus_1_j = Pi_minus_1[j];
    MatrixXd S = Y - Pi_minus_1_j;
    
    
    MatrixXd cond_dist_p_j=cond_dist_p[j];
    
    
    
    // 使用自定义归约的直接结果变量 
    VectorXd eigen_result_deri_j = VectorXd::Zero(alpha_length);
    MatrixXd eigen_result_hessian_j = MatrixXd::Zero(alpha_length, alpha_length);
    
    
    // #pragma omp for schedule(dynamic)
    // #pragma omp parallel for num_threads(ncores)
    
    // #pragma omp parallel for reduction(VectorAdd : eigen_result_deri_j) \
    // reduction(MatrixAdd : eigen_result_hessian_j)     
#pragma omp parallel for num_threads(ncores)           \
    schedule(dynamic)                              \
      reduction(VectorAdd:eigen_result_deri_j)     \
      reduction(MatrixAdd:eigen_result_hessian_j)
    for (int i = 0; i < npeop; ++i) {
      // int tid = omp_get_thread_num();
      
      VectorXd margin_prob = cond_dist_p_j.row(i) * e_tran_cor;
      // VectorXd y_minus_mu = y_w.row(i) - margin_prob;
      // VectorXd y_w_row_i=y_w.row(i);
      // // VectorXd y_minus_mu = y_w_row_i - margin_prob;
      // VectorXd y_minus_mu = y_w_row_i - margin_prob;
      
      
      // VectorXd y_minus_mu = y_w.row(i).transpose() - margin_prob;
      
      
      // S與前面重複
      // VectorXd S = y_minus_mu.head(nitem);
      // S = y_minus_mu.head(nitem);
      
      
      // e_tran_cor.array().rowwise()-margin_prob;
      
      // Calculate cov_var
      // cov_var.setZero();
      MatrixXd cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
      for(int h = 0; h < H; ++h) {
        // VectorXd diff = e_tran_cor.row(h).transpose() - margin_prob;
        // cov_var+= cond_dist_p_j(i,h) * diff * diff.transpose();
        // 
        cov_var.noalias() += cond_dist_p_j(i,h) * 
          (e_tran_cor.row(h).transpose() - margin_prob) * 
          (e_tran_cor.row(h).transpose() - margin_prob).transpose();
      }
      
      
      // MatrixXd cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
      // for(int h=0; h < H; ++h){
      //   VectorXd e_tran_cor_row_h=e_tran_cor.row(h);
      //   VectorXd diff = e_tran_cor_row_h - margin_prob;
      //   cov_var+=(diff*diff.transpose())*cond_dist_p_j(i,h);
      // }
      
      // MatrixXd V_ij_11 = cov_var.block(0, 0, nitem, nitem);
      
      MatrixXd V_ij_11 = cov_var.topLeftCorner(nitem, nitem);
      
      
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(V_ij_11);
      MatrixXd V_ij_11_ginv = cod.pseudoInverse();
      // MatrixXd V_ij_11_ginv = 
      //   CompleteOrthogonalDecomposition<MatrixXd>(V_ij_11).
      //   pseudoInverse();
      
      // VectorXd S_i = S.row(i);
      
      
      // MatrixXd eigen_Vh_ij_inver_h = V_ij_11_ginv * h_p(i,j);
      
      
      MatrixXd deri_1 = h_p(i,j)*V_ij_11_ginv * S.row(i).transpose();
      
      
      
      // VectorXd muij = Pi_minus_1_j.row(i);
      
      // VectorXd muij_var=Pi_minus_1_j.row(i).array()*(1-Pi_minus_1_j.row(i).array());
      // MatrixXd Vij_eigen=muij_var.asDiagonal();
      
      MatrixXd Vij_eigen=(  Pi_minus_1_j.row(i).array()*
        (1-Pi_minus_1_j.row(i).array()) ).
        matrix().asDiagonal();
      
      
      
      
      // MatrixXd eigen_der_alpha0=MatrixXd::Zero(nrow_result, ncol_result);
      // for (int k = 0; k < nrow_result; ++k) {
      //   
      //   // MatrixXd Vij_eigen_temp=Vij_eigen.col(k)*diag_nclass_j;
      //   
      //   eigen_der_alpha0.block( 0, k*nclass, nrow_result, nclass )=
      //     Vij_eigen.col(k)*diag_nclass_j;
      // }
      
      MatrixXd eigen_der_alpha0=KroneckerProduct(Vij_eigen,diag_nclass_j);
      
      
      // 計算對alpha微分
      MatrixXd eigen_deri_mu_ij=eigen_der_alpha0;
      if(nxcond.sum() > 0){
        
        int u = -1;
        MatrixXd der_alpha(nrow_result, 0);
        
        // MatrixXd der_alpha=MatrixXd::Zero(nrow_result, 0);
        // VectorXd xcond_mi;
        
        for(int m = 0; m < nitem; ++m) {
          
          // if(nxcond[m]!=0){
          //   // std::cout << "Thread " <<  " processing i = " << "\n";
          //   
          //   
          //   xcond_mi = xcond_vec[m].row(i);
          //   
          //   // xcond_mi = xcond[m].row(i);
          //   
          //   
          //   // VectorXd xcond_mi = xcond_m.row(i);
          //   // MatrixXd xcond_m = xcond[m];
          //   // xcond_mi= xcond_m.row(i);
          //   
          // }
          
          
          for(int k = 0; k < (nlevels[m] - 1); ++k) {
            
            ++u;
            
            if (nxcond[m]==0) {
              // 如果条件满足，跳过当前循环
              continue;
            }
            
            // MatrixXd Vij_u = Vij_eigen.col(u);
            // 
            // MatrixXd b_temp=Vij_u*xcond_mi.transpose();
            
            MatrixXd b_temp=Vij_eigen.col(u)*xcond_vec[m].row(i);
            // MatrixXd b_temp=Vij_eigen.col(u)*xcond_mi.transpose();
            
            der_alpha = cbindEigen_matrix(der_alpha, b_temp);
            
          }
        }
        
        eigen_deri_mu_ij = cbindEigen_matrix(eigen_der_alpha0, der_alpha);
      }
      
      // eigen_der_alpha0_local=eigen_der_alpha0_local+eigen_der_alpha0;
      // 
      // 
      
      // VectorXd deri = eigen_deri_mu_ij.transpose() * deri_1;
      // 
      // MatrixXd hessian = eigen_deri_mu_ij.transpose() * eigen_Vh_ij_inver_h*
      //   eigen_deri_mu_ij;
      
      // 累加到线程局部存储
      // deri_results[tid] += eigen_deri_mu_ij.transpose() * deri_1;
      // hessian_results[tid] += eigen_deri_mu_ij.transpose() * eigen_Vh_ij_inver_h * 
      //   eigen_deri_mu_ij;
      // 
      
      // // eigen_result_deri = eigen_result_deri + deri;
      // //
      // // eigen_result_hessian=eigen_result_hessian+hessian;
      // eigen_result_deri_local = eigen_result_deri_local + deri;
      // 
      // eigen_result_hessian_local=eigen_result_hessian_local+hessian;
      
      // 直接累加到归约变量（不再需要tid或thread-local存储）
      eigen_result_deri_j += eigen_deri_mu_ij.transpose() * deri_1;
      
      eigen_result_hessian_j += h_p(i,j)*eigen_deri_mu_ij.transpose() * 
        V_ij_11_ginv * eigen_deri_mu_ij;
    } 
      
    eigen_result_deri+= eigen_result_deri_j;
    eigen_result_hessian +=eigen_result_hessian_j;
  }
  // }
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(eigen_result_hessian);
  MatrixXd alpha_ginv = cod.pseudoInverse();
  // VectorXd direction_alpha = alpha_ginv * eigen_result_deri;
  
  
  // return direction_alpha;
  return alpha_ginv * eigen_result_deri;
}  
  



// [[Rcpp::export]]
List max_alpha_likeli_4_2_cpp(VectorXd& alpha_vector,
                            List& alpha0,
                            List& alpha,
                            const std::vector<VectorXd>& tau,
                            const std::vector<std::vector<MatrixXd>>& U,
                            const MatrixXd& Y,
                            const MatrixXd& Y_comp,
                            List& Pi_minus_1,
                            List& cond_dist,
                            MatrixXd& cond_prob,
                            const MatrixXd& h_p,
                            const MatrixXd& e_tran_cor,
                            const MatrixXd& y_w,
                            const VectorXi& nlevels,
                            const VectorXi& nxcond,
                            const int& H,
                            const List& xcond,
                            const MatrixXd& diag_nclass,
                            const int& num_alpha0,
                            const int& alpha_length,
                            const int& npeop,
                            const int& nitem,
                            const int& nclass,
                            const MatrixXd& Y_cor,
                            const MatrixXd& e_tran_cor_nitem,
                            const MatrixXd& A_matrix,
                            const MatrixXd& A_matrix_complement,
                            const int& maxiter,
                            const double& tol,
                            const int& maxit,
                            const double& tol_par,
                            const double& step_length,
                            const int& it) {
  
  int count=0;
  
  double llik=(h_p.array()*cond_prob.array().log()).sum(); 
  
  
  VectorXd direction_alpha=direction_alpha_like_cpp(Pi_minus_1,
                                                    cond_dist,
                                                    h_p,
                                                    e_tran_cor,
                                                    y_w,
                                                    Y,
                                                    xcond,
                                                    npeop,
                                                    nitem,
                                                    nclass,
                                                    nlevels,
                                                    nxcond,
                                                    H,
                                                    diag_nclass,
                                                    alpha_length);
  
  
  
  VectorXd alpha_vector_p(alpha_vector.size());
  MatrixXd cond_prob_p(npeop, nclass);
  List temp;
  List alpha0_p;
  List alpha_p;
  
  List cond;
  double llik_p;
  double a=1.0;
  
  int iter = 0;
  for( iter = 0; iter < maxiter; ++iter) {
    
    alpha_vector_p.noalias()=alpha_vector+a*direction_alpha;
    
    temp=tran_vec_to_list_cpp(alpha_vector_p,
                              nlevels,
                              nxcond,
                              num_alpha0,
                              nitem,
                              nclass);
    
    
    alpha0_p=temp["alpha0"];
    alpha_p=temp["alpha"];
    
    cond=conditional_prob_tau_2_cpp(alpha0_p,
                                    alpha_p,
                                    tau,
                                    U,
                                    xcond,
                                    nxcond,
                                    npeop,
                                    nitem,
                                    nclass,
                                    H,
                                    Y_cor,
                                    e_tran_cor_nitem,
                                    A_matrix,
                                    A_matrix_complement,
                                    maxit,
                                    tol);
    
    
    // List Pi_minus_1_p=cond["Pi_minus_1"];
    
    cond_prob_p=cond["cond_prob"];
    // cond_prob_p=cond_prob_p.array()+(1e-300);
    
    llik_p=(h_p.array()*cond_prob_p.array().log()).sum(); 
    
    
    
    if(llik_p>llik){
      ++count;
      alpha_vector.noalias()=alpha_vector_p;
      
      llik=llik_p;
      Pi_minus_1=cond["Pi_minus_1"];
      cond_prob.noalias()=cond_prob_p;
      cond_dist=cond["cond_dist"];
      
      
      alpha0=alpha0_p;
      alpha=alpha_p;
      
      if((a*direction_alpha).cwiseAbs().maxCoeff()<tol_par|
         count==it) break;
      
      // Pi_minus_1=Pi_minus_1_cpp(alpha_vector,
      //                           xcond,
      //                           nlevels,
      //                           nxcond,
      //                           num_alpha0,
      //                           npeop,
      //                           nitem,
      //                           nclass);
      
      direction_alpha.noalias()=
        direction_alpha_like_cpp(Pi_minus_1,
                                 cond_dist,
                                 h_p,
                                 e_tran_cor,
                                 y_w,
                                 Y,
                                 xcond,
                                 npeop,
                                 nitem,
                                 nclass,
                                 nlevels,
                                 nxcond,
                                 H,
                                 diag_nclass,
                                 alpha_length);
      
      
      
    }else{
      
      a*=step_length;
      if((a*direction_alpha).cwiseAbs().maxCoeff()<tol_par) break;
      
    }
    
    
  }
  
  
  
  List result = List::create(
    Named("alpha_vector_p") = alpha_vector,
    Named("alpha0") = alpha0,
    Named("alpha") = alpha,
    Named("Pi_minus_1") = Pi_minus_1,
    Named("cond_dist") = cond_dist,
    Named("cond_prob") = cond_prob,
    Named("direction") = direction_alpha,
    Named("a") = a,
    Named("llik") = llik,
    Named("iter") = iter);
  
  
  return result;
}  

  
  
  

// [[Rcpp::export]]
std::vector<std::vector<MatrixXd>> 
  max_alpha_likeli_4_2_for_omp_cpp(VectorXd& alpha_vector,
                                 std::vector<MatrixXd>& alpha0,
                                 std::vector<MatrixXd>& alpha,
                                 const std::vector<VectorXd>& tau,
                                 const std::vector<std::vector<MatrixXd>>& U,
                                 const MatrixXd& Y,
                                 const MatrixXd& Y_comp,
                                 std::vector<MatrixXd>& Pi_minus_1,
                                 std::vector<MatrixXd>& cond_dist,
                                 MatrixXd& cond_prob,
                                 const MatrixXd& h_p,
                                 const MatrixXd& e_tran_cor,
                                 const MatrixXd& y_w,
                                 const VectorXi& nlevels,
                                 const VectorXi& nxcond,
                                 const int& H,
                                 const std::vector<MatrixXd> xcond_vec,
                                 const MatrixXd& diag_nclass,
                                 const int& num_alpha0,
                                 const int& alpha_length,
                                 const int& npeop,
                                 const int& nitem,
                                 const int& nclass,
                                 const MatrixXd& Y_cor,
                                 const MatrixXd& e_tran_cor_nitem,
                                 const MatrixXd& A_matrix,
                                 const MatrixXd& A_matrix_complement,
                                 const int& maxiter,
                                 const double& tol,
                                 const int& maxit,
                                 const double& tol_par,
                                 const double& step_length,
                                 const int& it) {
  int count=0;
  
  double llik=(h_p.array()*cond_prob.array().log()).sum();
  
  VectorXd direction_alpha=direction_alpha_like_for_omp_cpp(Pi_minus_1,
                                                            cond_dist,
                                                            h_p,
                                                            e_tran_cor,
                                                            y_w,
                                                            Y,
                                                            xcond_vec,
                                                            npeop,
                                                            nitem,
                                                            nclass,
                                                            nlevels,
                                                            nxcond,
                                                            H,
                                                            diag_nclass,
                                                            alpha_length);
  
  
  VectorXd alpha_vector_p(alpha_vector.size());
  MatrixXd cond_prob_p(npeop, nclass);
  
  
  
  double llik_p;
  double a=1.0;
  
  for(int iter = 0; iter < maxiter; ++iter) {
    
    alpha_vector_p=alpha_vector+a*direction_alpha;
    
    auto temp=tran_vec_to_list_for_omp_cpp(alpha_vector_p,
                                           nlevels,
                                           nxcond,
                                           num_alpha0,
                                           nitem,
                                           nclass);
    
    
    auto alpha0_p=temp[0];
    auto alpha_p=temp[1];
    
    
    auto cond=conditional_prob_tau_2_for_omp_cpp(alpha0_p,
                                                 alpha_p,
                                                 tau,
                                                 U,
                                                 xcond_vec,
                                                 nxcond,
                                                 npeop,
                                                 nitem,
                                                 nclass,
                                                 H,
                                                 Y_cor,
                                                 e_tran_cor_nitem,
                                                 A_matrix,
                                                 A_matrix_complement,
                                                 maxit,
                                                 tol);
    
    
    MatrixXd cond_prob_p=cond[0][0];
    
    
    
    llik_p=(h_p.array()*cond_prob_p.array().log()).sum();
    
    
    
    if(llik_p>llik){
      ++count;
      alpha_vector=alpha_vector_p;
      
      llik=llik_p;
      Pi_minus_1=cond[1];
      cond_prob=cond_prob_p;
      cond_dist=cond[2];
      
      
      alpha0=alpha0_p;
      alpha=alpha_p;
      
      
      if((a*direction_alpha).cwiseAbs().maxCoeff()<tol_par|
         count==it) break;
      
      // Pi_minus_1=Pi_minus_1_cpp(alpha_vector,
      //                           xcond,
      //                           nlevels,
      //                           nxcond,
      //                           num_alpha0,
      //                           npeop,
      //                           nitem,
      //                           nclass);
      
      direction_alpha.noalias()=
        direction_alpha_like_for_omp_cpp(Pi_minus_1,
                                         cond_dist,
                                         h_p,
                                         e_tran_cor,
                                         y_w,
                                         Y,
                                         xcond_vec,
                                         npeop,
                                         nitem,
                                         nclass,
                                         nlevels,
                                         nxcond,
                                         H,
                                         diag_nclass,
                                         alpha_length);
      
    }else{
      
      a*=step_length;
      if((a*direction_alpha).cwiseAbs().maxCoeff()<tol_par) break;
      
    }
    
    
  }
  
  std::vector<std::vector<MatrixXd>> result(6);
  result[0] = {alpha_vector.matrix()}; // VectorXd -> MatrixXd
  result[1] = alpha0; 
  result[2] = alpha; 
  result[3] = Pi_minus_1; 
  result[4] = {cond_prob}; 
  result[5] = cond_dist; 
  
  
  return result;
}



// [[Rcpp::export]]
List max_alpha_likeli_4_Parallel_2_cpp(VectorXd& alpha_vector,
                                       List& alpha0,
                                       List& alpha,
                                       const std::vector<VectorXd>& tau,
                                       const std::vector<std::vector<MatrixXd>>& U,
                                       const MatrixXd& Y,
                                       const MatrixXd& Y_comp,
                                       List& Pi_minus_1,
                                       List& cond_dist,
                                       MatrixXd& cond_prob,
                                       const MatrixXd& h_p,
                                       const MatrixXd& e_tran_cor,
                                       const MatrixXd& y_w,
                                       const VectorXi& nlevels,
                                       const VectorXi& nxcond,
                                       const int& H,
                                       const List& xcond,
                                       const MatrixXd& diag_nclass,
                                       const int& num_alpha0,
                                       const int& alpha_length,
                                       const int& npeop,
                                       const int& nitem,
                                       const int& nclass,
                                       const MatrixXd& Y_cor,
                                       const MatrixXd& e_tran_cor_nitem,
                                       const MatrixXd& A_matrix,
                                       const MatrixXd& A_matrix_complement,
                                       const int& maxiter,
                                       const double& tol,
                                       const int& maxit,
                                       const double& tol_par,
                                       const double& step_length,
                                       const int& it,
                                       const int ncores) {
  
  int count=0;
  
  double llik=(h_p.array()*cond_prob.array().log()).sum();
  
  VectorXd direction_alpha=direction_alpha_like_Para_1_cpp(Pi_minus_1,
                                                           cond_dist,
                                                           h_p,
                                                           e_tran_cor,
                                                           y_w,
                                                           Y,
                                                           xcond,
                                                           npeop,
                                                           nitem,
                                                           nclass,
                                                           nlevels,
                                                           nxcond,
                                                           H,
                                                           diag_nclass,
                                                           alpha_length,
                                                           ncores);
  
  
  VectorXd alpha_vector_p(alpha_vector.size());
  MatrixXd cond_prob_p(npeop, nclass);
  List temp;
  List alpha0_p;
  List alpha_p;
  
  List cond;
  double llik_p;
  double a=1.0;
  int iter = 0;
  
  for( iter = 0; iter < maxiter; ++iter) {
    
    alpha_vector_p=alpha_vector+a*direction_alpha;
    
    temp=tran_vec_to_list_cpp(alpha_vector_p,
                              nlevels,
                              nxcond,
                              num_alpha0,
                              nitem,
                              nclass);
    
    
    alpha0_p=temp["alpha0"];
    alpha_p=temp["alpha"];
    
    cond=conditional_prob_tau_Parallel_2_cpp(alpha0_p,
                                             alpha_p,
                                             tau,
                                             U,
                                             xcond,
                                             nxcond,
                                             npeop,
                                             nitem,
                                             nclass,
                                             H,
                                             Y_cor,
                                             e_tran_cor_nitem,
                                             A_matrix,
                                             A_matrix_complement,
                                             maxit,
                                             tol,
                                             ncores);
    
    // List Pi_minus_1_p=cond["Pi_minus_1"];
    
    cond_prob_p=cond["cond_prob"];
    // cond_prob_p=cond_prob_p.array()+(1e-300);
    
    llik_p=(h_p.array()*cond_prob_p.array().log()).sum();
    
    
    
    if(llik_p>llik){
      ++count;
      alpha_vector.noalias()=alpha_vector_p;
      
      llik=llik_p;
      Pi_minus_1=cond["Pi_minus_1"];
      cond_prob.noalias()=cond_prob_p;
      cond_dist=cond["cond_dist"];
      
      
      alpha0=alpha0_p;
      alpha=alpha_p;
      
      
      if((a*direction_alpha).cwiseAbs().maxCoeff()<tol_par|
         count==it) break;
      
      // Pi_minus_1=Pi_minus_1_cpp(alpha_vector,
      //                           xcond,
      //                           nlevels,
      //                           nxcond,
      //                           num_alpha0,
      //                           npeop,
      //                           nitem,
      //                           nclass);
      
      direction_alpha.noalias()=
        direction_alpha_like_Para_1_cpp(Pi_minus_1,
                                        cond_dist,
                                        h_p,
                                        e_tran_cor,
                                        y_w,
                                        Y,
                                        xcond,
                                        npeop,
                                        nitem,
                                        nclass,
                                        nlevels,
                                        nxcond,
                                        H,
                                        diag_nclass,
                                        alpha_length,
                                        ncores);
      
    }else{
      
      a*=step_length;
      if((a*direction_alpha).cwiseAbs().maxCoeff()<tol_par) break;
      
    }
    
    
  }
  
  
  
  List result = List::create(
    Named("alpha_vector_p") = alpha_vector,
    Named("alpha0") = alpha0,
    Named("alpha") = alpha,
    Named("Pi_minus_1") = Pi_minus_1,
    Named("cond_dist") = cond_dist,
    Named("cond_prob") = cond_prob,
    Named("direction") = direction_alpha,
    Named("a") = a,
    Named("llik") = llik,
    Named("iter") = iter);
  
  
  return result;
}


  
  
// [[Rcpp::export]]
List Score_hess_tau_cpp(const List& Pi_minus_1,
                        const List& cond_dist_p,
                        const MatrixXd& h_p,
                        const MatrixXd& e_tran_cor,
                        const MatrixXd& y_w,
                        const MatrixXd& Y,
                        const std::vector<std::vector<MatrixXd>>& U,
                        const std::vector<std::vector<int>> tau_labels,
                        const std::vector<int> labels,
                        const int& num_tau,
                        const int& npeop,
                        const int& nitem,
                        const int& nclass,
                        const VectorXi& nlevels,
                        const VectorXi& nxcond,
                        const int& H) {
  
  // const MatrixXd& G,
  VectorXd tau_score = VectorXd::Zero(num_tau);
  MatrixXd tau_hessian=MatrixXd::Zero(num_tau,num_tau);
  
  MatrixXd Uij;
  
  
  const int cov_var_col = e_tran_cor.cols();
  const int tau_dim = cov_var_col - nitem;
  
  // MatrixXd U_comp_T=MatrixXd::Zero(num_tau,cov_var_col);
  
  VectorXd margin_prob(cov_var_col);
  VectorXd y_minus_mu(cov_var_col);
  MatrixXd cov_var(cov_var_col, cov_var_col);
  
  
  MatrixXd V_ij_11(nitem, nitem);
  MatrixXd V_ij_12(nitem, tau_dim);
  MatrixXd V_ij_21(tau_dim, nitem);
  MatrixXd V_ij_22(tau_dim, tau_dim);
  
  // VectorXd direction_tau((cov_var_col-nitem)*nclass);
  // MatrixXd direction_tau((cov_var_col-nitem),nclass);
  


  // VectorXd direction_tau_j;
  


  // int nrow_result = Y.cols() ;
  // int ncol_result = Y.cols() * nclass;


  VectorXd tau_score_temp;
  
  for (int j = 0; j < nclass; ++j) {
    
    
    VectorXd tau_score_j=VectorXd::Zero(tau_labels[j].size());
    // MatrixXd tau_hessian_j=MatrixXd::Zero(tau_labels[j].size(),tau_labels[j].size());
    
    
    MatrixXd Pi_minus_1_j = Pi_minus_1[j];
    MatrixXd S = Y - Pi_minus_1_j;
    MatrixXd cond_dist_p_j=cond_dist_p[j];
    
    for (int i = 0; i < npeop; ++i) {
      
      margin_prob.noalias() = cond_dist_p_j.row(i) * e_tran_cor;
      // // VectorXd y_minus_mu = y_w.row(i) - margin_prob;
      // VectorXd y_w_row_i=y_w.row(i);
      // // VectorXd y_minus_mu = y_w_row_i - margin_prob;
      // y_minus_mu = y_w_row_i - margin_prob;
      y_minus_mu.noalias() = y_w.row(i).transpose() - margin_prob;
      
      
      // Calculate cov_var
      // cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
      cov_var.setZero();
      for(int h = 0; h < H; ++h) {
        cov_var.noalias() += cond_dist_p_j(i,h) * 
          (e_tran_cor.row(h).transpose() - margin_prob) * 
          (e_tran_cor.row(h).transpose() - margin_prob).transpose();
      }
      
      
      // 对于固定位置的块，可以使用更具语义的方法
      V_ij_11 = cov_var.topLeftCorner(nitem, nitem);
      V_ij_12 = cov_var.topRightCorner(nitem, tau_dim);
      V_ij_21 = cov_var.bottomLeftCorner(tau_dim, nitem);
      V_ij_22 = cov_var.bottomRightCorner(tau_dim, tau_dim);
      
      // V_ij_11 = cov_var.block(0, 0, nitem, nitem);
      // V_ij_12 = cov_var.block(0, nitem, nitem, cov_var_col-nitem);
      // V_ij_21 = cov_var.block(nitem, 0, cov_var_col-nitem, nitem);
      // V_ij_22 = cov_var.block(nitem, nitem, cov_var_col-nitem, cov_var_col-nitem );
      
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(V_ij_11);
      MatrixXd V_ij_11_ginv = cod.pseudoInverse();
      
      
      MatrixXd U_comp_T=MatrixXd::Zero(num_tau,tau_dim);
      // for (int k = 0; k < tau_labels[j].size(); ++k) {
      //   U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][i].col(k).transpose();
      // }

      
      if(nxcond.sum() == 0){
        Uij=U[j][0];
        for (int k = 0; k < tau_labels[j].size(); ++k) {
          U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][0].col(k).transpose();
        }
      }else{
        Uij=U[j][i];
        for (int k = 0; k < tau_labels[j].size(); ++k) {
          U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][i].col(k).transpose();
        }
      }
      
      // 計算tau
      tau_score_j.noalias() += h_p(i,j) * Uij.transpose()*
        (y_minus_mu.tail(tau_dim) - V_ij_21 * V_ij_11_ginv * S.row(i).transpose());
      
      
      tau_hessian.noalias() += h_p(i,j) * U_comp_T*
        (V_ij_22 - V_ij_21 * V_ij_11_ginv * V_ij_12) * U_comp_T.transpose();
      
      
    }
    tau_score_temp.conservativeResize(tau_score_temp.size() + tau_score_j.size());
    tau_score_temp.tail(tau_score_j.size()) << tau_score_j;
  }
  
  for (int k = 0; k < tau_score_temp.size(); ++k) {
    tau_score(labels[k]-1) += tau_score_temp(k);
  }
  
  // VectorXd tau_score_test=G.transpose()*tau_score_temp;
    
    
  // List result = List::create(
  //   Named("tau_score") = tau_score,
  //   Named("tau_score_test") = tau_score_test,
  //   Named("tau_hessian") = tau_hessian);
  
  List result = List::create(
    Named("tau_score") = tau_score,
    Named("tau_hessian") = tau_hessian);
  
  return result;
}
  
  


  
// [[Rcpp::export]]
std::vector<MatrixXd>
    Score_hess_tau_for_omp_cpp(const std::vector<MatrixXd> Pi_minus_1,
                               const std::vector<MatrixXd>& cond_dist_p,
                               const MatrixXd& h_p,
                               const MatrixXd& e_tran_cor,
                               const MatrixXd& y_w,
                               const MatrixXd& Y,
                               const std::vector<std::vector<MatrixXd>>& U,
                               const std::vector<std::vector<int>> tau_labels,
                               const std::vector<int> labels,
                               const int& num_tau,
                               const int& npeop,
                               const int& nitem,
                               const int& nclass,
                               const VectorXi& nlevels,
                               const VectorXi& nxcond,
                               const int& H) {
  
  // const MatrixXd& G,
  VectorXd tau_score = VectorXd::Zero(num_tau);
  MatrixXd tau_hessian=MatrixXd::Zero(num_tau,num_tau);
  
  MatrixXd Uij;
  
  
  const int cov_var_col = e_tran_cor.cols();
  const int tau_dim = cov_var_col - nitem;
  
  // MatrixXd U_comp_T=MatrixXd::Zero(num_tau,cov_var_col);
  
  VectorXd margin_prob(cov_var_col);
  VectorXd y_minus_mu(cov_var_col);
  MatrixXd cov_var(cov_var_col, cov_var_col);
  
  
  MatrixXd V_ij_11(nitem, nitem);
  MatrixXd V_ij_12(nitem, tau_dim);
  MatrixXd V_ij_21(tau_dim, nitem);
  MatrixXd V_ij_22(tau_dim, tau_dim);
  
  // VectorXd direction_tau((cov_var_col-nitem)*nclass);
  // MatrixXd direction_tau((cov_var_col-nitem),nclass);
  
  
  
  // VectorXd direction_tau_j;
  
  
  
  // int nrow_result = Y.cols() ;
  // int ncol_result = Y.cols() * nclass;
  
  
  VectorXd tau_score_temp;
  
  for (int j = 0; j < nclass; ++j) {
    
    
    VectorXd tau_score_j=VectorXd::Zero(tau_labels[j].size());
    // MatrixXd tau_hessian_j=MatrixXd::Zero(tau_labels[j].size(),tau_labels[j].size());
    
    
    MatrixXd Pi_minus_1_j = Pi_minus_1[j];
    MatrixXd S = Y - Pi_minus_1_j;
    MatrixXd cond_dist_p_j=cond_dist_p[j];
    
    for (int i = 0; i < npeop; ++i) {
      
      margin_prob.noalias() = cond_dist_p_j.row(i) * e_tran_cor;
      // // VectorXd y_minus_mu = y_w.row(i) - margin_prob;
      // VectorXd y_w_row_i=y_w.row(i);
      // // VectorXd y_minus_mu = y_w_row_i - margin_prob;
      // y_minus_mu = y_w_row_i - margin_prob;
      y_minus_mu.noalias() = y_w.row(i).transpose() - margin_prob;
      
      
      // Calculate cov_var
      // cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
      cov_var.setZero();
      for(int h = 0; h < H; ++h) {
        cov_var.noalias() += cond_dist_p_j(i,h) * 
          (e_tran_cor.row(h).transpose() - margin_prob) * 
          (e_tran_cor.row(h).transpose() - margin_prob).transpose();
      }
      
      
      // 对于固定位置的块，可以使用更具语义的方法
      V_ij_11 = cov_var.topLeftCorner(nitem, nitem);
      V_ij_12 = cov_var.topRightCorner(nitem, tau_dim);
      V_ij_21 = cov_var.bottomLeftCorner(tau_dim, nitem);
      V_ij_22 = cov_var.bottomRightCorner(tau_dim, tau_dim);
      
      // V_ij_11 = cov_var.block(0, 0, nitem, nitem);
      // V_ij_12 = cov_var.block(0, nitem, nitem, cov_var_col-nitem);
      // V_ij_21 = cov_var.block(nitem, 0, cov_var_col-nitem, nitem);
      // V_ij_22 = cov_var.block(nitem, nitem, cov_var_col-nitem, cov_var_col-nitem );
      
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(V_ij_11);
      MatrixXd V_ij_11_ginv = cod.pseudoInverse();
      
      
      MatrixXd U_comp_T=MatrixXd::Zero(num_tau,tau_dim);
      // for (int k = 0; k < tau_labels[j].size(); ++k) {
      //   U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][i].col(k).transpose();
      // }
      
      
      if(nxcond.sum() == 0){
        Uij=U[j][0];
        for (int k = 0; k < tau_labels[j].size(); ++k) {
          U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][0].col(k).transpose();
        }
      }else{
        Uij=U[j][i];
        for (int k = 0; k < tau_labels[j].size(); ++k) {
          U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][i].col(k).transpose();
        }
      }
      
      // 計算tau
      tau_score_j.noalias() += h_p(i,j) * Uij.transpose()*
        (y_minus_mu.tail(tau_dim) - V_ij_21 * V_ij_11_ginv * S.row(i).transpose());
      
      
      tau_hessian.noalias() += h_p(i,j) * U_comp_T*
        (V_ij_22 - V_ij_21 * V_ij_11_ginv * V_ij_12) * U_comp_T.transpose();
      
      
    }
    tau_score_temp.conservativeResize(tau_score_temp.size() + tau_score_j.size());
    tau_score_temp.tail(tau_score_j.size()) << tau_score_j;
  }
  
  for (int k = 0; k < tau_score_temp.size(); ++k) {
    tau_score(labels[k]-1) += tau_score_temp(k);
  }
  
  // VectorXd tau_score_test=G.transpose()*tau_score_temp;
  
  
  // List result = List::create(
  //   Named("tau_score") = tau_score,
  //   Named("tau_score_test") = tau_score_test,
  //   Named("tau_hessian") = tau_hessian);
  
  // List result = List::create(
  //   Named("tau_score") = tau_score,
  //   Named("tau_hessian") = tau_hessian);
  
  std::vector<MatrixXd> result(2);
  result[0] = tau_score.matrix(); // VectorXd -> MatrixXd
  result[1] = tau_hessian;
  
  return result;
  
  
 
}  
    
  
    


// [[Rcpp::export]]
List Score_hess_tau_Parallel_1_cpp(const List& Pi_minus_1,
                                   const List& cond_dist_p,
                                   const MatrixXd& h_p,
                                   const MatrixXd& e_tran_cor,
                                   const MatrixXd& y_w,
                                   const MatrixXd& Y,
                                   const std::vector<std::vector<MatrixXd>>& U,
                                   const std::vector<std::vector<int>> tau_labels,
                                   const std::vector<int> labels,
                                   const int& num_tau,
                                   const int& npeop,
                                   const int& nitem,
                                   const int& nclass,
                                   const VectorXi& nlevels,
                                   const VectorXi& nxcond,
                                   const int& H,
                                   const int ncores) {
  
  
  
  
  

  MatrixXd tau_hessian=MatrixXd::Zero(num_tau,num_tau);
  

  
  
  const int cov_var_col = e_tran_cor.cols();
  const int tau_dim = cov_var_col - nitem;
  
  
  // std::vector<std::vector<MatrixXd>> Uij(nclass);
  // std::vector<std::vector<MatrixXd>> U_comp_T(nclass);
  // 
  // for (int j = 0; j < nclass; ++j) {
  //   std::vector<MatrixXd> Uij_j(npeop);
  //   std::vector<MatrixXd> U_comp_T_j(npeop);
  //   for (int i = 0; i < npeop; ++i) {
  //     // MatrixXd Uij;
  //     MatrixXd U_comp_T_ij=MatrixXd::Zero(num_tau,tau_dim);
  //     if(nxcond.sum() == 0){
  //       Uij_j[i]=U[j][0];
  //       for (int k = 0; k < tau_labels[j].size(); ++k) {
  // 
  //         U_comp_T_ij.row(tau_labels[j][k]-1) += U[j][0].col(k).transpose();
  //       }
  //     }else{
  //       Uij_j[i]=U[j][i];
  //       for (int k = 0; k < tau_labels[j].size(); ++k) {
  //         U_comp_T_ij.row(tau_labels[j][k]-1) += U[j][i].col(k).transpose();
  //       }
  //     }
  //     U_comp_T_j[i]=U_comp_T_ij;
  //   }
  //   Uij[j]=Uij_j;
  //   U_comp_T[j]=U_comp_T_j;
  // }
  
  
  
  
  
  // MatrixXd U_comp_T=MatrixXd::Zero(num_tau,cov_var_col);
  
  // VectorXd margin_prob(cov_var_col);
  // VectorXd y_minus_mu(cov_var_col);
  // MatrixXd cov_var(cov_var_col, cov_var_col);
  
  
  // MatrixXd V_ij_11(nitem, nitem);
  // MatrixXd V_ij_12(nitem, tau_dim);
  // MatrixXd V_ij_21(tau_dim, nitem);
  // MatrixXd V_ij_22(tau_dim, tau_dim);
  
  // VectorXd direction_tau((cov_var_col-nitem)*nclass);
  // MatrixXd direction_tau((cov_var_col-nitem),nclass);
  
  
  
  // VectorXd direction_tau_j;
  
  
  
  // int nrow_result = Y.cols() ;
  // int ncol_result = Y.cols() * nclass;
  
  
  VectorXd tau_score_temp;
  
  for (int j = 0; j < nclass; ++j) {
    
    
    VectorXd tau_score_j=VectorXd::Zero(tau_labels[j].size());
    // MatrixXd tau_hessian_j=MatrixXd::Zero(tau_labels[j].size(),tau_labels[j].size());
    MatrixXd tau_hessian_j=MatrixXd::Zero(num_tau,num_tau);
    
    MatrixXd Pi_minus_1_j = Pi_minus_1[j];
    MatrixXd S = Y - Pi_minus_1_j;
    MatrixXd cond_dist_p_j=cond_dist_p[j];
    
#pragma omp parallel for num_threads(ncores)       \
    schedule(static)                                  \
      reduction(VectorAdd:tau_score_j)                 \
      reduction(MatrixAdd:tau_hessian_j)
    for (int i = 0; i < npeop; ++i) {
      
      VectorXd margin_prob= cond_dist_p_j.row(i) * e_tran_cor;
      // // VectorXd y_minus_mu = y_w.row(i) - margin_prob;
      // VectorXd y_w_row_i=y_w.row(i);
      // // VectorXd y_minus_mu = y_w_row_i - margin_prob;
      // y_minus_mu = y_w_row_i - margin_prob;
      VectorXd y_minus_mu = y_w.row(i).transpose() - margin_prob;
      
      
      // Calculate cov_var
     
      // cov_var.setZero();
      MatrixXd cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
      for(int h = 0; h < H; ++h) {
        cov_var += cond_dist_p_j(i,h) * 
          (e_tran_cor.row(h).transpose() - margin_prob) * 
          (e_tran_cor.row(h).transpose() - margin_prob).transpose();
      }
      
      
      // 对于固定位置的块，可以使用更具语义的方法
      MatrixXd V_ij_11 = cov_var.topLeftCorner(nitem, nitem);
      MatrixXd V_ij_12 = cov_var.topRightCorner(nitem, tau_dim);
      MatrixXd V_ij_21 = cov_var.bottomLeftCorner(tau_dim, nitem);
      MatrixXd V_ij_22 = cov_var.bottomRightCorner(tau_dim, tau_dim);
      
      // V_ij_11 = cov_var.block(0, 0, nitem, nitem);
      // V_ij_12 = cov_var.block(0, nitem, nitem, cov_var_col-nitem);
      // V_ij_21 = cov_var.block(nitem, 0, cov_var_col-nitem, nitem);
      // V_ij_22 = cov_var.block(nitem, nitem, cov_var_col-nitem, cov_var_col-nitem );
      
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(V_ij_11);
      MatrixXd V_ij_11_ginv = cod.pseudoInverse();
      
      
      MatrixXd U_comp_T=MatrixXd::Zero(num_tau,tau_dim);
      // for (int k = 0; k < tau_labels[j].size(); ++k) {
      //   U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][i].col(k).transpose();
      // }

      MatrixXd Uij;
      if(nxcond.sum() == 0){
        Uij=U[j][0];
        for (int k = 0; k < (int)tau_labels[j].size(); ++k) {
          U_comp_T.row(tau_labels[j][k]-1) += U[j][0].col(k).transpose();
        }
      }else{
        Uij=U[j][i];
        for (int k = 0; k < (int)tau_labels[j].size(); ++k) {
          U_comp_T.row(tau_labels[j][k]-1) += U[j][i].col(k).transpose();
        }
      }
      
      // 計算tau
      tau_score_j += h_p(i,j) * Uij.transpose()*
        (y_minus_mu.tail(tau_dim) - V_ij_21 * V_ij_11_ginv * S.row(i).transpose());
      
      
      tau_hessian_j += h_p(i,j) * U_comp_T*
        (V_ij_22 - V_ij_21 * V_ij_11_ginv * V_ij_12) * U_comp_T.transpose();

      // // 計算tau
      // tau_score_j += h_p(i,j) * Uij[j][i].transpose()*
      //   (y_minus_mu.tail(tau_dim) - V_ij_21 * V_ij_11_ginv * S.row(i).transpose());
      // 
      // 
      // tau_hessian_j += h_p(i,j) * U_comp_T[j][i]*
      //   (V_ij_22 - V_ij_21 * V_ij_11_ginv * V_ij_12) * U_comp_T[j][i].transpose();
      
      
    }
    tau_score_temp.conservativeResize(tau_score_temp.size() + tau_score_j.size());
    tau_score_temp.tail(tau_score_j.size()) << tau_score_j;
    tau_hessian += tau_hessian_j;
  }
  
  
  VectorXd tau_score = VectorXd::Zero(num_tau);
  for (int k = 0; k < tau_score_temp.size(); ++k) {
    tau_score(labels[k]-1) += tau_score_temp(k);
  }
  
  // VectorXd tau_score_test=G.transpose()*tau_score_temp;
  
  
  // List result = List::create(
  //   Named("tau_score") = tau_score,
  //   Named("tau_score_test") = tau_score_test,
  //   Named("tau_hessian") = tau_hessian);
  
  List result = List::create(
    Named("tau_score") = tau_score,
    Named("tau_hessian") = tau_hessian);
  
  return result;
} 
  

// 
// // [[Rcpp::export]]
// List Score_hess_tau_j_1_cpp(const List& Pi_minus_1,
//                             const List& cond_dist_p,
//                             const MatrixXd& h_p,
//                             const MatrixXd& e_tran_cor,
//                             const MatrixXd& y_w,
//                             const MatrixXd& Y,
//                             const int& npeop,
//                             const int& nitem,
//                             const int& nclass,
//                             const VectorXi& nlevels,
//                             const VectorXi& nxcond,
//                             const int& H,
//                             const int& j) {
//               // const std::vector<std::vector<MatrixXd>>& U,
//   // MatrixXd Uij;
//   // if(nxcond.sum() == 0){
//   //   Uij=U[j][0];
//   // }else{
//   //   Uij=U[j][i];
//   // }
//   // VectorXd direction_tau_j;
// 
//   const int cov_var_col = e_tran_cor.cols();
//   const int tau_dim = cov_var_col - nitem;
// 
//   VectorXd margin_prob(cov_var_col);
//   VectorXd y_minus_mu(cov_var_col);
//   MatrixXd cov_var(cov_var_col, cov_var_col);
// 
// 
//   MatrixXd V_ij_11(nitem, nitem);
//   MatrixXd V_ij_12(nitem, tau_dim);
//   MatrixXd V_ij_21(tau_dim, nitem);
//   MatrixXd V_ij_22(tau_dim, tau_dim);
// 
//   // VectorXd direction_tau((cov_var_col-nitem)*nclass);
//   MatrixXd direction_tau((cov_var_col-nitem),nclass);
// 
// 
// 
//   MatrixXd Pi_minus_1_j = Pi_minus_1[j];
//   MatrixXd S = Y - Pi_minus_1_j;
//   MatrixXd cond_dist_p_j=cond_dist_p[j];
// 
// 
//   // VectorXd tau_score_j=VectorXd::Zero(cov_var_col-nitem);
//   // MatrixXd tau_hessian_j=MatrixXd::Zero(cov_var_col-nitem,cov_var_col-nitem);
// 
//   VectorXd tau_score_j=VectorXd::Zero(cov_var_col-nitem);
//   MatrixXd tau_hessian_j=MatrixXd::Zero(cov_var_col-nitem,cov_var_col-nitem);
// 
// 
//   for (int i = 0; i < npeop; ++i) {
// 
//     margin_prob.noalias() = cond_dist_p_j.row(i) * e_tran_cor;
//     // // VectorXd y_minus_mu = y_w.row(i) - margin_prob;
//     // VectorXd y_w_row_i=y_w.row(i);
//     // // VectorXd y_minus_mu = y_w_row_i - margin_prob;
//     // y_minus_mu = y_w_row_i - margin_prob;
//     y_minus_mu.noalias() = y_w.row(i).transpose() - margin_prob;
// 
// 
// 
//     // Calculate cov_var
//     // cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
//     cov_var.setZero();
//     for(int h = 0; h < H; ++h) {
//       cov_var.noalias() += cond_dist_p_j(i,h) *
//         (e_tran_cor.row(h).transpose() - margin_prob) *
//         (e_tran_cor.row(h).transpose() - margin_prob).transpose();
//     }
// 
// 
//     // 对于固定位置的块，可以使用更具语义的方法
//     V_ij_11 = cov_var.topLeftCorner(nitem, nitem);
//     V_ij_12 = cov_var.topRightCorner(nitem, tau_dim);
//     V_ij_21 = cov_var.bottomLeftCorner(tau_dim, nitem);
//     V_ij_22 = cov_var.bottomRightCorner(tau_dim, tau_dim);
// 
//     // V_ij_11 = cov_var.block(0, 0, nitem, nitem);
//     // V_ij_12 = cov_var.block(0, nitem, nitem, cov_var_col-nitem);
//     // V_ij_21 = cov_var.block(nitem, 0, cov_var_col-nitem, nitem);
//     // V_ij_22 = cov_var.block(nitem, nitem, cov_var_col-nitem, cov_var_col-nitem );
// 
// 
//     // 計算ginv
//     CompleteOrthogonalDecomposition<MatrixXd> cod(V_ij_11);
//     MatrixXd V_ij_11_ginv = cod.pseudoInverse();
// 
// 
// 
// 
//     // 計算tau
//     tau_score_j.noalias() += h_p(i,j) *
//       (y_minus_mu.tail(tau_dim) - V_ij_21 * V_ij_11_ginv * S.row(i).transpose());
// 
//     tau_hessian_j.noalias() += h_p(i,j) *
//       (V_ij_22 - V_ij_21 * V_ij_11_ginv * V_ij_12);
// 
//   }
// 
// 
// 
//   // // // 計算tau ginv
//   // CompleteOrthogonalDecomposition<MatrixXd> cod(tau_hessian_j);
//   // MatrixXd tau_hessian_j_ginv = cod.pseudoInverse();
//   // // VectorXd direction_tau_j = tau_hessian_j_ginv * tau_score_j;
//   //
//   // direction_tau_j = tau_hessian_j_ginv * tau_score_j;
//   //
// 
// 
// 
//   // List result = List::create(
//   //   Named("deri") = eigen_result_deri,
//   //   Named("hessian") = eigen_result_hessian,
//   //   Named("cov_var") = cov_var,
//   //   Named("y_minus_mu") = y_minus_mu,
//   //   Named("direction_alpha") = direction_alpha,
//   //   Named("direction_tau") = direction_tau,
//   //   Named("direction_tau_j") = direction_tau_j,
//   //   Named("tau_score_j") = tau_score_j,
//   //   Named("tau_hessian_j") = tau_hessian_j,
//   //   Named("V_ij_11") = V_ij_11,
//   //   Named("V_ij_12") = V_ij_12,
//   //   Named("V_ij_21") = V_ij_21,
//   //   Named("V_ij_22") = V_ij_22);
// 
// 
// 
//   List result = List::create(
//     Named("tau_score_j") = tau_score_j,
//     Named("tau_hessian_j") = tau_hessian_j);
// 
//   return result;
// }







// 
// 
// // [[Rcpp::export]]
// // tau一起算
// 
// List max_tau_2_cpp(const List& alpha0,
//                    const List& alpha,
//                    MatrixXd& tau,
//                    const MatrixXd& Y,
//                    const MatrixXd& Y_comp,
//                    List& Pi_minus_1,
//                    List& cond_dist,
//                    MatrixXd& cond_prob,
//                    const MatrixXd& h_p,
//                    const MatrixXd& e_tran_cor,
//                    const MatrixXd& y_w,
//                    const VectorXi& nlevels,
//                    const VectorXi& nxcond,
//                    const int& H,
//                    const List& xcond,
//                    const MatrixXd& diag_nclass,
//                    const int& num_alpha0,
//                    const int& alpha_length,
//                    const int& npeop,
//                    const int& nitem,
//                    const int& nclass,
//                    const MatrixXd& Y_cor,
//                    const MatrixXd& e_tran_cor_nitem,
//                    const NumericMatrix& A_matrix,
//                    const NumericMatrix& A_matrix_complement,
//                    const int& maxiter,
//                    const double& tol,
//                    const int& maxit,
//                    const double& tol_par,
//                    const double& step_length,
//                    double& a,
//                    const int& it) {
// 
//   int count=0;
//   double llik=(h_p.array()*cond_prob.array().log()).sum();
//   MatrixXd cond_prob_p;
//   MatrixXd tau_p=tau;
// 
//   List temp=Score_tau_cpp(Pi_minus_1,
//                           cond_dist,
//                           h_p,
//                           e_tran_cor,
//                           y_w,
//                           Y,
//                           xcond,
//                           npeop,
//                           nitem,
//                           nclass,
//                           nlevels,
//                           nxcond,
//                           H,
//                           diag_nclass,
//                           alpha_length);
// 
// 
// 
//   MatrixXd direction_tau=temp["direction_tau"];
// 
// 
// 
//   int iter;
//   for(iter = 0; iter < maxiter; ++iter) {
// 
//     tau_p=tau+a*direction_tau;
// 
// 
//     List cond=conditional_prob_tau_cpp(alpha0,alpha,
//                                        tau_p,xcond,
//                                        nxcond,
//                                        npeop,
//                                        nitem,
//                                        nclass,
//                                        H,
//                                        Y_cor,
//                                        e_tran_cor_nitem,
//                                        A_matrix,
//                                        A_matrix_complement,
//                                        maxit,
//                                        tol);
//     cond_prob_p=cond["cond_prob"];
//     // cond_prob_p=cond_prob_p.array()+(1e-300);
// 
//     double llik_p=(h_p.array()*cond_prob_p.array().log()).sum();
// 
// 
//     if(llik_p>=llik) {
//       tau=tau_p;
//       llik=llik_p;
//       Pi_minus_1=cond["Pi_minus_1"];
//       cond_prob=cond_prob_p;
//       cond_dist=cond["cond_dist"];
//       ++count;
// 
//       if((a*direction_tau).cwiseAbs().maxCoeff()<tol_par|count==it) break;
// 
// 
//       temp=Score_tau_cpp(Pi_minus_1,
//                          cond_dist,
//                          h_p,
//                          e_tran_cor,
//                          y_w,
//                          Y,
//                          xcond,
//                          npeop,
//                          nitem,
//                          nclass,
//                          nlevels,
//                          nxcond,
//                          H,
//                          diag_nclass,
//                          alpha_length);
// 
// 
//       direction_tau=temp["direction_tau"];
//     } else {
//       a=a*step_length;
//       if((a*direction_tau).cwiseAbs().maxCoeff()<tol_par) break;
//     }
// 
// 
//   }
// 
// 
// 
//   List result = List::create(
//     Named("tau_p") = tau_p,
//     Named("Pi_minus_1") = Pi_minus_1,
//     Named("cond_dist") = cond_dist,
//     Named("cond_prob") = cond_prob,
//     Named("direction_tau") = direction_tau,
//     Named("a") = a,
//     Named("llik") = llik,
//     Named("count") = count,
//     Named("iter") = iter);
// 
//   return result;
// }
// 



// [[Rcpp::export]]
std::vector<VectorXd> combine_tau_cpp(
    std::vector<VectorXd> tau,                          // tau: list of numeric vectors
    const std::vector<std::vector<int>>& tau_labels,    // labels: 每個 tau[j] 的標籤
    const VectorXd& direction_tau,                      // 要加的 direction
    int nclass) {
  // 遍歷每個 class
  for (int j = 0; j < nclass; ++j) {
    for (int k = 0; k < tau[j].size(); ++k) {
      int lab = tau_labels[j][k] - 1;  // R index -> C++ (0-based)
      tau[j][k] += direction_tau(lab);
    }
  }
  return tau;
}

  
// [[Rcpp::export]]
List max_tau_Parallel_2_cpp(const List& alpha0,
                            const List& alpha,
                            std::vector<VectorXd>& tau,
                            const std::vector<std::vector<MatrixXd>>& U,
                            const std::vector<std::vector<int>> tau_labels,
                            const std::vector<int> labels,
                            const int& num_tau,
                            const MatrixXd& Y,
                            const MatrixXd& Y_comp,
                            List& Pi_minus_1,
                            List& cond_dist,
                            MatrixXd& cond_prob,
                            const MatrixXd& h_p,
                            const MatrixXd& e_tran_cor,
                            const MatrixXd& y_w,
                            const VectorXi& nlevels,
                            const VectorXi& nxcond,
                            const int& H,
                            const List& xcond,
                            const MatrixXd& diag_nclass,
                            const int& num_alpha0,
                            const int& alpha_length,
                            const int& npeop,
                            const int& nitem,
                            const int& nclass,
                            const MatrixXd& Y_cor,
                            const MatrixXd& e_tran_cor_nitem,
                            const MatrixXd& A_matrix,
                            const MatrixXd& A_matrix_complement,
                            const int& maxiter,
                            const double& tol,
                            const int& maxit,
                            const double& tol_par,
                            const int& it,
                            const int ncores) {

  double llik=(h_p.array()*cond_prob.array().log()).sum();
  MatrixXd cond_prob_p=cond_prob;
  // std::vector<VectorXd> tau_p = tau;

  // VectorXd cond_prob_j_p(npeop);

  // 预计算常用矩阵
  // MatrixXd identity = MatrixXd::Identity(tau.cols(), tau.cols());


  double v=2.0;

  int count=0;

  List temp=Score_hess_tau_Parallel_1_cpp(Pi_minus_1,
                                          cond_dist,
                                          h_p,
                                          e_tran_cor,
                                          y_w,
                                          Y,
                                          U,
                                          tau_labels,
                                          labels,
                                          num_tau,
                                          npeop,
                                          nitem,
                                          nclass,
                                          nlevels,
                                          nxcond,
                                          H,
                                          ncores);



  VectorXd tau_score=temp["tau_score"];
  MatrixXd tau_hessian=temp["tau_hessian"];

  MatrixXd A = tau_hessian.transpose() * tau_hessian;
  double Lambda0= A.diagonal().array().maxCoeff();

  for(int iter = 0; iter < maxiter; ++iter) {

    MatrixXd mmm = tau_hessian.transpose()* tau_hessian +
      Lambda0 * MatrixXd::Identity(num_tau, num_tau);
    // VectorXd direction_tau=mmm.inverse()*tau_hessian_j.transpose()*tau_score_j;

    // MatrixXd mmm = tau_hessian_j.transpose() * tau_hessian_j +
    //   Lambda0 * identity;
    // for symmetric
    // VectorXd direction_tau = mmm.ldlt().solve(tau_hessian_j.transpose() * tau_score_j);

    // for positive definite
    VectorXd direction_tau = mmm.llt().solve(tau_hessian.transpose() * tau_score);

    auto tau_p=combine_tau_cpp(tau,       
                               tau_labels,
                               direction_tau,   
                               nclass);
      
    
    // tau_p.col(tau_which)=tau.col(tau_which)+direction_tau;


    List cond=conditional_prob_tau_Parallel_2_cpp(alpha0,
                                                  alpha,
                                                  tau_p,
                                                  U,
                                                  xcond,
                                                  nxcond,
                                                  npeop,
                                                  nitem,
                                                  nclass,
                                                  H,
                                                  Y_cor,
                                                  e_tran_cor_nitem,
                                                  A_matrix,
                                                  A_matrix_complement,
                                                  maxit,
                                                  tol,
                                                  ncores);



    cond_prob_p=cond["cond_prob"];
    // cond_prob_j_p=cond_prob_j_p.array()+(1e-300);


    double llik_p=(h_p.array()*cond_prob_p.array().log()).sum();

    // bool tau_ok = !(tau_p.col(tau_which).array().unaryExpr([](double x){ return std::isnan(x); }).any());

    // 检查有效性
    bool tau_ok = !direction_tau.array().isNaN().any();

    if ((llik_p > llik) && tau_ok) {
      // if(llik_p>llik) {
      // if ((llik_p > llik) && !(direction_tau.array().unaryExpr([](double x){ return std::isnan(x); }).any())) {
      // 接受更新
      tau=tau_p;
      // llik=llik_p;
      Pi_minus_1=cond["Pi_minus_1"];
      cond_prob=cond_prob_p;
      cond_dist=cond["cond_dist"];
      ++count;

      if((direction_tau).cwiseAbs().maxCoeff()<tol_par
           |count==it) break;


      // double  rho = (llik_p - llik) /
      // (direction_tau.dot(tau_score_j) -
      // 0.5 * direction_tau.transpose() *
      // tau_hessian_j * direction_tau);
      double  rho = (llik_p - llik) /
        (direction_tau.dot(direction_tau)*Lambda0+
          direction_tau.transpose() *
          tau_score.transpose() * tau_score);

      llik=llik_p;

      Lambda0 *= std::max(1/3.0, 1 - std::pow(2 * rho - 1, 3));
      v = 2.0;



      temp=Score_hess_tau_Parallel_1_cpp(Pi_minus_1,
                                         cond_dist,
                                         h_p,
                                         e_tran_cor,
                                         y_w,
                                         Y,
                                         U,
                                         tau_labels,
                                         labels,
                                         num_tau,
                                         npeop,
                                         nitem,
                                         nclass,
                                         nlevels,
                                         nxcond,
                                         H,
                                         ncores);

      tau_score=temp["tau_score"];
      tau_hessian=temp["tau_hessian"];

    } else {
      if((direction_tau).cwiseAbs().maxCoeff()<tol_par) break;
      Lambda0 *= v;
      v *= 2.0;
      // Lambda0=Lambda0*v;
      // v=2*v;
    }
  }




  List result = List::create(
    Named("tau_p") = tau,
    Named("Pi_minus_1") = Pi_minus_1,
    Named("cond_dist") = cond_dist,
    Named("cond_prob") = cond_prob,
    Named("llik") = llik);

  return result;
}



  
// [[Rcpp::export]]
std::vector<std::vector<MatrixXd>> 
  max_tau_Parallel_2_for_omp_cpp(const std::vector<MatrixXd>& alpha0,
                            const std::vector<MatrixXd>& alpha,
                            std::vector<VectorXd>& tau,
                            const std::vector<std::vector<MatrixXd>>& U,
                            const std::vector<std::vector<int>> tau_labels,
                            const std::vector<int> labels,
                            const int& num_tau,
                            const MatrixXd& Y,
                            const MatrixXd& Y_comp,
                            std::vector<MatrixXd> Pi_minus_1,
                            std::vector<MatrixXd>& cond_dist,
                            MatrixXd& cond_prob,
                            const MatrixXd& h_p,
                            const MatrixXd& e_tran_cor,
                            const MatrixXd& y_w,
                            const VectorXi& nlevels,
                            const VectorXi& nxcond,
                            const int& H,
                            const std::vector<MatrixXd> xcond_vec,
                            const MatrixXd& diag_nclass,
                            const int& num_alpha0,
                            const int& alpha_length,
                            const int& npeop,
                            const int& nitem,
                            const int& nclass,
                            const MatrixXd& Y_cor,
                            const MatrixXd& e_tran_cor_nitem,
                            const MatrixXd& A_matrix,
                            const MatrixXd& A_matrix_complement,
                            const int& maxiter,
                            const double& tol,
                            const int& maxit,
                            const double& tol_par,
                            const int& it) {
  
  double llik=(h_p.array()*cond_prob.array().log()).sum();
  MatrixXd cond_prob_p=cond_prob;
  // std::vector<VectorXd> tau_p = tau;
  
  // VectorXd cond_prob_j_p(npeop);
  
  // 预计算常用矩阵
  // MatrixXd identity = MatrixXd::Identity(tau.cols(), tau.cols());
  
  
  double v=2.0;
  
  int count=0;
  
  auto temp=Score_hess_tau_for_omp_cpp(Pi_minus_1,
                                          cond_dist,
                                          h_p,
                                          e_tran_cor,
                                          y_w,
                                          Y,
                                          U,
                                          tau_labels,
                                          labels,
                                          num_tau,
                                          npeop,
                                          nitem,
                                          nclass,
                                          nlevels,
                                          nxcond,
                                          H);
  
  
  VectorXd tau_score=temp[0].col(0);
  MatrixXd tau_hessian=temp[1];
  
  MatrixXd A = tau_hessian.transpose() * tau_hessian;
  double Lambda0= A.diagonal().array().maxCoeff();
  
  for(int iter = 0; iter < maxiter; ++iter) {
    
    MatrixXd mmm = tau_hessian.transpose()* tau_hessian +
      Lambda0 * MatrixXd::Identity(num_tau, num_tau);
    // VectorXd direction_tau=mmm.inverse()*tau_hessian_j.transpose()*tau_score_j;
    
    // MatrixXd mmm = tau_hessian_j.transpose() * tau_hessian_j +
    //   Lambda0 * identity;
    // for symmetric
    // VectorXd direction_tau = mmm.ldlt().solve(tau_hessian_j.transpose() * tau_score_j);
    
    // for positive definite
    VectorXd direction_tau = mmm.llt().solve(tau_hessian.transpose() * tau_score);
    
    auto tau_p=combine_tau_cpp(tau,       
                               tau_labels,
                               direction_tau,   
                               nclass);
    
    
    // tau_p.col(tau_which)=tau.col(tau_which)+direction_tau;
    
    
    auto cond=conditional_prob_tau_2_for_omp_cpp(alpha0,
                                                  alpha,
                                                  tau_p,
                                                  U,
                                                  xcond_vec,
                                                  nxcond,
                                                  npeop,
                                                  nitem,
                                                  nclass,
                                                  H,
                                                  Y_cor,
                                                  e_tran_cor_nitem,
                                                  A_matrix,
                                                  A_matrix_complement,
                                                  maxit,
                                                  tol);
    
  
    
    cond_prob_p=cond[0][0];
    // cond_prob_j_p=cond_prob_j_p.array()+(1e-300);
    
    
    double llik_p=(h_p.array()*cond_prob_p.array().log()).sum();
    
    // bool tau_ok = !(tau_p.col(tau_which).array().unaryExpr([](double x){ return std::isnan(x); }).any());
    
    // 检查有效性
    bool tau_ok = !direction_tau.array().isNaN().any();
    
    if ((llik_p > llik) && tau_ok) {
      // if(llik_p>llik) {
      // if ((llik_p > llik) && !(direction_tau.array().unaryExpr([](double x){ return std::isnan(x); }).any())) {
      // 接受更新
      tau=tau_p;
      // llik=llik_p;
      Pi_minus_1=cond[1];
      cond_prob=cond_prob_p;
      cond_dist=cond[2];
      ++count;
      
      if((direction_tau).cwiseAbs().maxCoeff()<tol_par
           |count==it) break;
      
      
      // double  rho = (llik_p - llik) /
      // (direction_tau.dot(tau_score_j) -
      // 0.5 * direction_tau.transpose() *
      // tau_hessian_j * direction_tau);
      double  rho = (llik_p - llik) /
        (direction_tau.dot(direction_tau)*Lambda0+
          direction_tau.transpose() *
          tau_score.transpose() * tau_score);
      
      llik=llik_p;
      
      Lambda0 *= std::max(1/3.0, 1 - std::pow(2 * rho - 1, 3));
      v = 2.0;
      
      
      
      temp=Score_hess_tau_for_omp_cpp(Pi_minus_1,
                                         cond_dist,
                                         h_p,
                                         e_tran_cor,
                                         y_w,
                                         Y,
                                         U,
                                         tau_labels,
                                         labels,
                                         num_tau,
                                         npeop,
                                         nitem,
                                         nclass,
                                         nlevels,
                                         nxcond,
                                         H);
      

      tau_score=temp[0].col(0);
      tau_hessian=temp[1];
      
    } else {
      if((direction_tau).cwiseAbs().maxCoeff()<tol_par) break;
      Lambda0 *= v;
      v *= 2.0;
      // Lambda0=Lambda0*v;
      // v=2*v;
    }
  }
  
  
  std::vector<MatrixXd> tau_as_matrix;
  tau_as_matrix.reserve(tau.size());
  for (const auto& v : tau) {
    tau_as_matrix.push_back(MatrixXd(v));  // VectorXd → MatrixXd(n,1)
  }
  
  
  std::vector<std::vector<MatrixXd>> result(4);
  result[0] = tau_as_matrix; 
  result[1] = Pi_minus_1; 
  result[2] = cond_dist; 
  result[3] = {cond_prob}; 

  

  
  return result;
}  
    

  

// [[Rcpp::export]]
List Estimation_likeli_omp_cpp(std::vector<MatrixXd> beta_list,
                                          const std::vector<VectorXd>& alpha_vector_list,
                                          const std::vector<VectorXd>& tau,
                                          const std::vector<std::vector<MatrixXd>>& U,
                                          const std::vector<std::vector<int>> tau_labels,
                                          const std::vector<int> labels,
                                          const int& num_tau,
                                          const List& alpha0_list,
                                          const List& alpha_list,
                                          const int& npeop,
                                          const int& nitem,
                                          const int nclass,
                                          const MatrixXd& Xprev,
                                          const int nxprev,
                                          const MatrixXd& Y,
                                          const MatrixXd& Y_comp,
                                          const MatrixXd& y_w,
                                          const MatrixXd& e_comp,
                                          const MatrixXd& Y_cor,
                                          const MatrixXd& e_tran_cor,
                                          const MatrixXd& e_tran_cor_nitem,
                                          const MatrixXd& A_matrix,
                                          const MatrixXd& A_matrix_complement,
                                          const VectorXi& nlevels,
                                          const VectorXi& nxcond,
                                          const int& H,
                                          const List& xcond,
                                          const MatrixXd& direct,
                                          const MatrixXd& diag_nclass,
                                          const int& alpha_length,
                                          const int& num_alpha0,
                                          const int& maxiter,
                                          const int& maxiter_para,
                                          const int& maxit,
                                          const double& tol,
                                          const double& tol_beta,
                                          const double& tol_alpha,
                                          const double& tol_tau,
                                          const double& tol_para,
                                          const double& tol_likeli,
                                          const double& step_length,
                                          const int& it_beta,
                                          const int& it_alpha,
                                          const int& it_tau,
                                          const int ncores) {
  
  int nsets = beta_list.size();
  std::vector<MatrixXd> beta_p_list(nsets);
  std::vector<MatrixXd> eta_list(nsets);
  std::vector<double> iter_list(nsets);
  std::vector<double> log_lik_list(nsets);
  
  std::vector<MatrixXd> cond_prob_p_list(nsets);
  std::vector<std::vector<MatrixXd>> Pi_minus_1_list(nsets);
  std::vector<std::vector<MatrixXd>> cond_dist_p_list(nsets);
  
  std::vector<VectorXd> alpha_vector_p_list(nsets);
  
  std::vector<std::vector<VectorXd>> tau_p_list(nsets);
  // std::vector<MatrixXd> al_list(nsets);
  
  
  std::vector<MatrixXd> xcond_vec(xcond.size());
  for (int m = 0; m < xcond.size(); ++m) {
    if (!Rf_isNull(xcond[m])) {
      xcond_vec[m] = as<MatrixXd>(xcond[m]);
    }
  }
  
  // 預先轉換 List → std::vector<MatrixXd>
  std::vector<std::vector<MatrixXd>> alpha0_list_std(nsets);
  std::vector<std::vector<MatrixXd>> alpha_list_std(nsets);
  for (int i = 0; i < nsets; ++i) {
    List alpha0 = alpha0_list[i];
    List alpha = alpha_list[i];
    for (int j = 0; j < alpha0.size(); ++j) {
      alpha0_list_std[i].push_back(as<MatrixXd>(alpha0[j]));
      alpha_list_std[i].push_back(as<MatrixXd>(alpha[j]));
    }
  }
  
  
  
  // #pragma omp for schedule(dynamic)    
#pragma omp parallel for num_threads(ncores)
  for (int idx = 0; idx < nsets; ++idx) {
    
    
    // const MatrixXd& beta = betas[i];
    auto& beta_p=beta_list[idx];
    auto& alpha0_p = alpha0_list_std[idx];
    auto& alpha_p = alpha_list_std[idx];
    VectorXd alpha_vector_p =alpha_vector_list[idx];
    
    auto tau_p=tau;
    
    VectorXd log_lik= VectorXd::Zero(maxiter+1);
    log_lik[0] = R_NegInf;
    
    
    auto conditional_prob_tau=
      conditional_prob_and_Pi_for_omp_cpp(alpha0_p,
                                          alpha_p,
                                          Y_comp,
                                          e_comp,
                                          xcond_vec,
                                          nxcond,
                                          npeop,
                                          nitem,
                                          nclass);

    
    MatrixXd cond_prob_p=conditional_prob_tau[0][0];
    
    std::vector<MatrixXd> Pi_minus_1=conditional_prob_tau[1];
    std::vector<MatrixXd> cond_dist_p=conditional_prob_tau[2];
    
    
    MatrixXd eta_p=eta_cpp(beta_p,Xprev,nxprev,nclass);
    
    MatrixXd h_p(npeop,nclass);
    
    MatrixXd beta_temp_p(nxprev + 1,nclass-1);
    VectorXd alpha_vector(alpha_length);
    double max_diff_alpha;
    double max_diff_beta;
    double log_lik_diff; 
    
    int iter=0;
    for(iter = 0; iter < maxiter; ++iter){
      // MatrixXd beta_temp_p=beta_p;
      beta_temp_p.noalias()=beta_p;
      // MatrixXd h_p= h_p_cpp(eta_p,cond_prob_p);
      // h_p.noalias()= h_p_cpp(eta_p,cond_prob_p);
      h_p=eta_p.cwiseProduct(cond_prob_p);
      h_p.array().colwise() /= h_p.rowwise().sum().array();      // 歸一化
      
      
      auto maxbeta=max_beta_6_for_omp_cpp(beta_p,
                                          Xprev,
                                          eta_p,
                                          h_p,
                                          npeop,
                                          nxprev,
                                          nclass,
                                          maxiter_para,
                                          step_length,
                                          tol_beta,
                                          it_beta);
      
      beta_p=maxbeta[0];
      // Rcpp::Rcout << "beta_p = " << beta_p << std::endl;  
      
      eta_p =maxbeta[1];
      
      // VectorXd alpha_vector=alpha_vector_p;
      alpha_vector.noalias()=alpha_vector_p;
      
      auto temp=max_alpha_likeli_4_2_for_omp_cpp(alpha_vector_p,
                                                 alpha0_p,
                                                 alpha_p,
                                                 tau_p,
                                                 U,
                                                 Y,
                                                 Y_comp,
                                                 Pi_minus_1,
                                                 cond_dist_p,
                                                 cond_prob_p,
                                                 h_p,
                                                 e_tran_cor,
                                                 y_w,
                                                 nlevels,
                                                 nxcond,
                                                 H,
                                                 xcond_vec,
                                                 diag_nclass,
                                                 num_alpha0,
                                                 alpha_length,
                                                 npeop,
                                                 nitem,
                                                 nclass,
                                                 Y_cor,
                                                 e_tran_cor_nitem,
                                                 A_matrix,
                                                 A_matrix_complement,
                                                 maxiter_para,
                                                 tol,
                                                 maxit,
                                                 tol_alpha,
                                                 step_length,
                                                 it_alpha);
      // VectorXd al = temp[0][0].col(0);
      // alpha_vector_p = al;
      alpha_vector_p = temp[0][0].col(0);
      alpha0_p=temp[1];
      alpha_p =temp[2];
      Pi_minus_1=temp[3];
      cond_prob_p=temp[4][0];
      cond_dist_p=temp[5];
      
      
      // Rcpp::Rcout << "alpha_vector_p = " << alpha_vector_p << std::endl;  
      auto testets_tau=max_tau_Parallel_2_for_omp_cpp(alpha0_p,
                                                      alpha_p,
                                                      tau_p,
                                                      U,
                                                      tau_labels,
                                                      labels,
                                                      num_tau,
                                                      Y,
                                                      Y_comp,
                                                      Pi_minus_1,
                                                      cond_dist_p,
                                                      cond_prob_p,
                                                      h_p,
                                                      e_tran_cor,
                                                      y_w,
                                                      nlevels,
                                                      nxcond,
                                                      H,
                                                      xcond_vec,
                                                      diag_nclass,
                                                      num_alpha0,
                                                      alpha_length,
                                                      npeop,
                                                      nitem,
                                                      nclass,
                                                      Y_cor,
                                                      e_tran_cor_nitem,
                                                      A_matrix,
                                                      A_matrix_complement,
                                                      maxiter_para,
                                                      tol,
                                                      maxit,
                                                      tol_tau,
                                                      it_tau);
        
        
        
        // 取出存進去的 tau_as_matrix
        std::vector<MatrixXd> tau_as_matrix = testets_tau[0];
        // for (size_t ll = 0; ll < tau_as_matrix.size(); ll++) {
        //   Rcpp::Rcout << "tau_as_matrix[" << ll << "] =\n"
        //               << tau_as_matrix[ll] << "\n";
        // }
        
        // 建立一個新的 tau_recovered
        std::vector<VectorXd> tau_recovered;
        tau_recovered.reserve(tau_as_matrix.size());
        
        for (const auto& m : tau_as_matrix) {
          // 假設 m 是 n×1，直接取第0欄轉回 VectorXd
          tau_recovered.push_back(m.col(0));
        }
        
      tau_p=tau_recovered;
      Pi_minus_1=testets_tau[1];
      cond_dist_p=testets_tau[2];
      cond_prob_p=testets_tau[3][0];  

      
      
      log_lik[iter+1]=((cond_prob_p.array()*eta_p.array()).rowwise().sum()).log().sum();
      
      max_diff_alpha = (alpha_vector_p - alpha_vector).cwiseAbs().maxCoeff();
      max_diff_beta = (beta_temp_p - beta_p).array().abs().maxCoeff();
      
      // 條件 1：參數收斂
      if ((max_diff_alpha < tol_para) && (max_diff_beta < tol_para)) break;
      
      // 條件 2：對數概似函數收斂
      log_lik_diff = std::abs(log_lik[iter+1] - log_lik[iter]);
      if (log_lik_diff < tol_likeli) break;
    }
    
    
    if(iter==maxiter) iter=iter-1;
    
    eta_list[idx]=eta_p;
    beta_p_list[idx]=beta_p;
    alpha0_list_std[idx]=alpha0_p;
    alpha_list_std[idx]=alpha_p;
    tau_p_list[idx]=tau_p;
    
    iter_list[idx]=iter+1;
    cond_prob_p_list[idx]=cond_prob_p;
    cond_dist_p_list[idx]=cond_dist_p;
    Pi_minus_1_list[idx]=Pi_minus_1;
    log_lik_list[idx]=log_lik[iter+1];
    // al_list[idx]=alpha_vector_p;
    alpha_vector_p_list[idx]=alpha_vector_p;
  }
  
  
  return List::create(
    Named("iter_list") = iter_list,
    Named("eta_list") = eta_list,
    Named("beta_list") = beta_p_list,
    Named("alpha0_list") = alpha0_list_std,
    Named("alpha_list") = alpha_list_std,
    Named("alpha_vector_list") = alpha_vector_p_list,
    Named("tau_list") = tau_p_list,
    Named("cond_dist_p_list") = cond_dist_p_list,
    Named("cond_prob_p_list") = cond_prob_p_list,
    Named("Pi_minus_1_list") = Pi_minus_1_list,
    Named("log_lik_list") = log_lik_list);
  
}

  
  



// [[Rcpp::export]]
List Estimation_likeli_no_covariate_omp_cpp(std::vector<MatrixXd> beta_list,
                               const std::vector<VectorXd>& alpha_vector_list,
                               const std::vector<VectorXd>& tau,
                               const std::vector<std::vector<MatrixXd>>& U,
                               const std::vector<std::vector<int>> tau_labels,
                               const std::vector<int> labels,
                               const int& num_tau,
                               const List& alpha0_list,
                               const List& alpha_list,
                               const int& npeop,
                               const int& nitem,
                               const int nclass,
                               const MatrixXd& Xprev,
                               const int nxprev,
                               const MatrixXd& Y,
                               const MatrixXd& Y_comp,
                               const MatrixXd& y_w,
                               const MatrixXd& e_comp,
                               const MatrixXd& Y_cor,
                               const MatrixXd& e_tran_cor,
                               const MatrixXd& e_tran_cor_nitem,
                               const MatrixXd& A_matrix,
                               const MatrixXd& A_matrix_complement,
                               const VectorXi& nlevels,
                               const VectorXi& nxcond,
                               const int& H,
                               const List& xcond,
                               const MatrixXd& direct,
                               const MatrixXd& diag_nclass,
                               const int& alpha_length,
                               const int& num_alpha0,
                               const int& maxiter,
                               const int& maxiter_para,
                               const int& maxit,
                               const double& tol,
                               const double& tol_beta,
                               const double& tol_alpha,
                               const double& tol_tau,
                               const double& tol_para,
                               const double& tol_likeli,
                               const double& step_length,
                               const int& it_beta,
                               const int& it_alpha,
                               const int& it_tau,
                               const int ncores) {
  
  int nsets = beta_list.size();
  std::vector<MatrixXd> beta_p_list(nsets);
  std::vector<MatrixXd> eta_list(nsets);
  std::vector<double> iter_list(nsets);
  std::vector<double> log_lik_list(nsets);
  
  std::vector<MatrixXd> cond_prob_p_list(nsets);
  std::vector<std::vector<MatrixXd>> Pi_minus_1_list(nsets);
  std::vector<std::vector<MatrixXd>> cond_dist_p_list(nsets);
  
  std::vector<VectorXd> alpha_vector_p_list(nsets);
  
  std::vector<std::vector<VectorXd>> tau_p_list(nsets);
  // std::vector<MatrixXd> al_list(nsets);
  
  
  std::vector<MatrixXd> xcond_vec(xcond.size());
  for (int m = 0; m < xcond.size(); ++m) {
    if (!Rf_isNull(xcond[m])) {
      xcond_vec[m] = as<MatrixXd>(xcond[m]);
    }
  }
  
  // 預先轉換 List → std::vector<MatrixXd>
  std::vector<std::vector<MatrixXd>> alpha0_list_std(nsets);
  std::vector<std::vector<MatrixXd>> alpha_list_std(nsets);
  for (int i = 0; i < nsets; ++i) {
    List alpha0 = alpha0_list[i];
    List alpha = alpha_list[i];
    for (int j = 0; j < alpha0.size(); ++j) {
      alpha0_list_std[i].push_back(as<MatrixXd>(alpha0[j]));
      alpha_list_std[i].push_back(as<MatrixXd>(alpha[j]));
    }
  }
  
  
  
  // #pragma omp for schedule(dynamic)    
#pragma omp parallel for num_threads(ncores)
  for (int idx = 0; idx < nsets; ++idx) {
    
    
    // const MatrixXd& beta = betas[i];
    auto& beta_p=beta_list[idx];
    auto& alpha0_p = alpha0_list_std[idx];
    auto& alpha_p = alpha_list_std[idx];
    VectorXd alpha_vector_p =alpha_vector_list[idx];
    
    auto tau_p=tau;
    
    VectorXd log_lik= VectorXd::Zero(maxiter+1);
    log_lik[0] = R_NegInf;
    
    
    auto conditional_prob_tau=
      conditional_prob_and_Pi_for_omp_cpp(alpha0_p,
                                          alpha_p,
                                          Y_comp,
                                          e_comp,
                                          xcond_vec,
                                          nxcond,
                                          npeop,
                                          nitem,
                                          nclass);
    
    
    MatrixXd cond_prob_p=conditional_prob_tau[0][0];
    
    std::vector<MatrixXd> Pi_minus_1=conditional_prob_tau[1];
    std::vector<MatrixXd> cond_dist_p=conditional_prob_tau[2];
    
    
    MatrixXd eta_p=eta_cpp(beta_p,Xprev,nxprev,nclass);
    
    MatrixXd h_p(npeop,nclass);
    
    MatrixXd beta_temp_p(nxprev + 1,nclass-1);
    VectorXd alpha_vector(alpha_length);
    double max_diff_alpha;
    double max_diff_beta;
    double log_lik_diff; 
    
    int iter=0;
    for(iter = 0; iter < maxiter; ++iter){
      // MatrixXd beta_temp_p=beta_p;
      beta_temp_p.noalias()=beta_p;
      // MatrixXd h_p= h_p_cpp(eta_p,cond_prob_p);
      // h_p.noalias()= h_p_cpp(eta_p,cond_prob_p);
      h_p=eta_p.cwiseProduct(cond_prob_p);
      h_p.array().colwise() /= h_p.rowwise().sum().array();      // 歸一化
      
      
      auto maxbeta=max_beta_6_no_covariate_omp_cpp(beta_p,
                                                   Xprev,
                                                   eta_p,
                                                   h_p,
                                                   npeop,
                                                   nxprev,
                                                   nclass,
                                                   maxiter_para,
                                                   step_length,
                                                   tol_beta,
                                                   it_beta);
      
      beta_p=maxbeta[0];
      // Rcpp::Rcout << "beta_p = " << beta_p << std::endl;  
      
      eta_p =maxbeta[1];
      
      // VectorXd alpha_vector=alpha_vector_p;
      alpha_vector.noalias()=alpha_vector_p;
      
      auto temp=max_alpha_likeli_4_2_for_omp_cpp(alpha_vector_p,
                                                 alpha0_p,
                                                 alpha_p,
                                                 tau_p,
                                                 U,
                                                 Y,
                                                 Y_comp,
                                                 Pi_minus_1,
                                                 cond_dist_p,
                                                 cond_prob_p,
                                                 h_p,
                                                 e_tran_cor,
                                                 y_w,
                                                 nlevels,
                                                 nxcond,
                                                 H,
                                                 xcond_vec,
                                                 diag_nclass,
                                                 num_alpha0,
                                                 alpha_length,
                                                 npeop,
                                                 nitem,
                                                 nclass,
                                                 Y_cor,
                                                 e_tran_cor_nitem,
                                                 A_matrix,
                                                 A_matrix_complement,
                                                 maxiter_para,
                                                 tol,
                                                 maxit,
                                                 tol_alpha,
                                                 step_length,
                                                 it_alpha);
      // VectorXd al = temp[0][0].col(0);
      // alpha_vector_p = al;
      alpha_vector_p = temp[0][0].col(0);
      alpha0_p=temp[1];
      alpha_p =temp[2];
      Pi_minus_1=temp[3];
      cond_prob_p=temp[4][0];
      cond_dist_p=temp[5];
      
      
      // Rcpp::Rcout << "alpha_vector_p = " << alpha_vector_p << std::endl;  
      auto testets_tau=max_tau_Parallel_2_for_omp_cpp(alpha0_p,
                                                      alpha_p,
                                                      tau_p,
                                                      U,
                                                      tau_labels,
                                                      labels,
                                                      num_tau,
                                                      Y,
                                                      Y_comp,
                                                      Pi_minus_1,
                                                      cond_dist_p,
                                                      cond_prob_p,
                                                      h_p,
                                                      e_tran_cor,
                                                      y_w,
                                                      nlevels,
                                                      nxcond,
                                                      H,
                                                      xcond_vec,
                                                      diag_nclass,
                                                      num_alpha0,
                                                      alpha_length,
                                                      npeop,
                                                      nitem,
                                                      nclass,
                                                      Y_cor,
                                                      e_tran_cor_nitem,
                                                      A_matrix,
                                                      A_matrix_complement,
                                                      maxiter_para,
                                                      tol,
                                                      maxit,
                                                      tol_tau,
                                                      it_tau);
      
      
      
      // 取出存進去的 tau_as_matrix
      std::vector<MatrixXd> tau_as_matrix = testets_tau[0];
      // for (size_t ll = 0; ll < tau_as_matrix.size(); ll++) {
      //   Rcpp::Rcout << "tau_as_matrix[" << ll << "] =\n"
      //               << tau_as_matrix[ll] << "\n";
      // }
      
      // 建立一個新的 tau_recovered
      std::vector<VectorXd> tau_recovered;
      tau_recovered.reserve(tau_as_matrix.size());
      
      for (const auto& m : tau_as_matrix) {
        // 假設 m 是 n×1，直接取第0欄轉回 VectorXd
        tau_recovered.push_back(m.col(0));
      }
      
      tau_p=tau_recovered;
      Pi_minus_1=testets_tau[1];
      cond_dist_p=testets_tau[2];
      cond_prob_p=testets_tau[3][0];  
      
      
      
      log_lik[iter+1]=((cond_prob_p.array()*eta_p.array()).rowwise().sum()).log().sum();
      
      max_diff_alpha = (alpha_vector_p - alpha_vector).cwiseAbs().maxCoeff();
      max_diff_beta = (beta_temp_p - beta_p).array().abs().maxCoeff();
      
      // 條件 1：參數收斂
      if ((max_diff_alpha < tol_para) && (max_diff_beta < tol_para)) break;
      
      // 條件 2：對數概似函數收斂
      log_lik_diff = std::abs(log_lik[iter+1] - log_lik[iter]);
      if (log_lik_diff < tol_likeli) break;
    }
    
    
    if(iter==maxiter) iter=iter-1;
    
    eta_list[idx]=eta_p;
    beta_p_list[idx]=beta_p;
    alpha0_list_std[idx]=alpha0_p;
    alpha_list_std[idx]=alpha_p;
    tau_p_list[idx]=tau_p;
    
    iter_list[idx]=iter+1;
    cond_prob_p_list[idx]=cond_prob_p;
    cond_dist_p_list[idx]=cond_dist_p;
    Pi_minus_1_list[idx]=Pi_minus_1;
    log_lik_list[idx]=log_lik[iter+1];
    // al_list[idx]=alpha_vector_p;
    alpha_vector_p_list[idx]=alpha_vector_p;
  }
  
  
  return List::create(
    Named("iter_list") = iter_list,
    Named("eta_list") = eta_list,
    Named("beta_list") = beta_p_list,
    Named("alpha0_list") = alpha0_list_std,
    Named("alpha_list") = alpha_list_std,
    Named("alpha_vector_list") = alpha_vector_p_list,
    Named("tau_list") = tau_p_list,
    Named("cond_dist_p_list") = cond_dist_p_list,
    Named("cond_prob_p_list") = cond_prob_p_list,
    Named("Pi_minus_1_list") = Pi_minus_1_list,
    Named("log_lik_list") = log_lik_list);
  
}
  





// [[Rcpp::export]]
List Score_alpha_tau_matrix_cpp(const List& Pi_minus_1,
                                const MatrixXd& eta_p,
                                const List& cond_dist_p,
                                const MatrixXd& e_tran_cor,
                                const MatrixXd& e_tran,
                                const List& xcond,
                                const int& nitem,
                                const int& nclass,
                                const VectorXi& nlevels,
                                const VectorXi& nxcond,
                                const int& H,
                                const MatrixXd& diag_nclass,
                                const std::vector<std::vector<MatrixXd>>& U,
                                const std::vector<std::vector<int>> tau_labels,
                                const std::vector<int> labels,
                                const int& num_tau,
                                const int& alpha_length,
                                const int& i) {

  int nrow_result = e_tran.cols() ;
  int ncol_result = e_tran.cols() * nclass;
  int cov_var_col=e_tran_cor.cols();

  const int tau_dim = cov_var_col - nitem;

  // MatrixXd Score=MatrixXd::Zero(alpha_length+(2^(nitem)-nitem-1)*nclass, H);
  MatrixXd Score=MatrixXd::Zero(alpha_length+num_tau, H);

  MatrixXd cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
  VectorXd y_minus_mu;
  // VectorXd tau_score_j;

  MatrixXd V_ij_11;
  MatrixXd V_ij_12;
  MatrixXd V_ij_21;
  MatrixXd V_ij_22;

  
  // MatrixXd tau_score=MatrixXd::Zero((cov_var_col-nitem),nclass);
  
  
  for(int h = 0; h < H; ++h){
    VectorXd tau_score_temp;
    VectorXd eigen_result_deri = VectorXd::Zero(alpha_length);
    for (int j = 0; j < nclass; ++j) {

      // RowVectorXd diag_nclass_j = diag_nclass.row(j);
      MatrixXd diag_nclass_j = diag_nclass.row(j);

      MatrixXd Pi_minus_1_j = Pi_minus_1[j];

      // MatrixXd S = e_tran - Pi_minus_1_j;

      MatrixXd cond_dist_p_j=cond_dist_p[j];

      // VectorXd tau_score_j=VectorXd::Zero(cov_var_col-nitem);
      // MatrixXd tau_hessian_j=MatrixXd::Zero(cov_var_col-nitem,cov_var_col-nitem);

      // tau_score_j=VectorXd::Zero(cov_var_col-nitem);

      // for (int i = 0; i < npeop; ++i) {





      VectorXd margin_prob = cond_dist_p_j.row(i) * e_tran_cor;
      // VectorXd y_minus_mu = y_w.row(i) - margin_prob;

      // VectorXd y_minus_mu = y_w_row_i - margin_prob;
      VectorXd e_tran_cor_h=e_tran_cor.row(h);
      y_minus_mu = e_tran_cor_h - margin_prob;





      cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
      for(int hh=0;hh < H; ++hh){
        cov_var+= cond_dist_p_j(i,hh) *
          (e_tran_cor.row(hh).transpose() - margin_prob) *
          (e_tran_cor.row(hh).transpose() - margin_prob).transpose();
      }


      V_ij_11 = cov_var.topLeftCorner(nitem, nitem);
      V_ij_12 = cov_var.topRightCorner(nitem, tau_dim);
      V_ij_21 = cov_var.bottomLeftCorner(tau_dim, nitem);
      V_ij_22 = cov_var.bottomRightCorner(tau_dim, tau_dim);



      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(V_ij_11);
      MatrixXd V_ij_11_ginv = cod.pseudoInverse();



      VectorXd S_i = e_tran.row(h) - Pi_minus_1_j.row(i);



      MatrixXd Vij_eigen=(  Pi_minus_1_j.row(i).array()*
        (1-Pi_minus_1_j.row(i).array()) ).
        matrix().asDiagonal();



      MatrixXd eigen_Vh_ij_inver_h = V_ij_11_ginv *eta_p(i,j)* cond_dist_p_j(i,h);


      MatrixXd deri_1 = eigen_Vh_ij_inver_h * S_i;



      MatrixXd eigen_der_alpha0(nrow_result, ncol_result);

      for (int k = 0; k < nrow_result; ++k) {

        MatrixXd Vij_eigen_temp=Vij_eigen.col(k)*diag_nclass_j;

        eigen_der_alpha0.block( 0, k*nclass, nrow_result, nclass )=Vij_eigen_temp;
      }


      // MatrixXd eigen_der_alpha0 = kroneckerProduct_matrix_eigen(Vij_eigen,eigen_diag_nclass_j);



      // 計算對alpha微分
      MatrixXd eigen_deri_mu_ij;
      if(nxcond.sum() == 0){
        eigen_deri_mu_ij = eigen_der_alpha0;
      }else{

        int u = -1;
        // NumericMatrix der_alpha(nitem, 0);
        // MatrixXd der_alpha(nlevels.sum()-nitem, 0);
        MatrixXd der_alpha(nrow_result, 0);

        VectorXd xcond_mi;

        for(int m = 0; m < nitem; ++m) {

          // Rf_isNull(xcond[m])
          // if (nxcond[m]==0) {
          //   // 如果条件满足，跳过当前循环
          //   continue;
          // }

          // const NumericMatrix& xcond_m_ref = xcond[m]; // 使用引用

          if(nxcond[m]!=0){
            MatrixXd xcond_m = xcond[m];
            xcond_mi= xcond_m.row(i);
            // xcond_mi= xcond_m.row(i);
          }


          for(int k = 0; k < (nlevels[m] - 1); ++k) {

            ++u;

            if (nxcond[m]==0) {
              // 如果条件满足，跳过当前循环
              continue;
            }

            // const NumericVector& Vij_u = Vij(_,u);
            // VectorXd Vij_u = Vij_eigen.col(u);

            MatrixXd Vij_u = Vij_eigen.col(u);
            // NumericVector Vij_u = Vij(_,u);


            // Map<MatrixXd> eigen_Vij_u(as<Map<MatrixXd> >(Vij_u));

            MatrixXd b_temp=Vij_u*xcond_mi.transpose();

            // MatrixXd b_temp=Vij_u*xcond_mi;

            der_alpha = cbindEigen_matrix(der_alpha, b_temp);

          }
        }

        eigen_deri_mu_ij = cbindEigen_matrix(eigen_der_alpha0, der_alpha);
      }




      VectorXd deri = eigen_deri_mu_ij.transpose() * deri_1;

      eigen_result_deri = eigen_result_deri + deri;






      MatrixXd U_comp_T=MatrixXd::Zero(num_tau,tau_dim);
      // for (int k = 0; k < tau_labels[j].size(); ++k) {
      //   U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][i].col(k).transpose();
      // }
      MatrixXd Uij;

      if(nxcond.sum() == 0){
        Uij=U[j][0];
        for (int k = 0; k < tau_labels[j].size(); ++k) {
          U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][0].col(k).transpose();
        }
      }else{
        Uij=U[j][i];
        for (int k = 0; k < tau_labels[j].size(); ++k) {
          U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][i].col(k).transpose();
        }
      }

      // 計算tau
      VectorXd tau_score_j= eta_p(i,j)*cond_dist_p_j(i,h) * Uij.transpose()*
        (y_minus_mu.tail(tau_dim) - V_ij_21 * V_ij_11_ginv * S_i);



      tau_score_temp.conservativeResize(tau_score_temp.size() + tau_score_j.size());
      tau_score_temp.tail(tau_score_j.size()) << tau_score_j;




      
      
      // tau_score_j=eta_p(i,j)*cond_dist_p_j(i,h)*
      //   (y_minus_mu.tail(cov_var_col - nitem)-
      //   V_ij_21*V_ij_11_ginv*S_i);
      // 
      // 
      // Score.block(alpha_length + j*tau_score_j.size(), h,
      //             tau_score_j.size() , 1 )=tau_score_j;
      
      
      // Score.block(alpha_length + j*tau_score_j.size(), h,
      //             tau_score_j.size() , 1 )=tau_score_j;

    }
    
    VectorXd tau_score = VectorXd::Zero(num_tau);
    for (int k = 0; k < tau_score_temp.size(); ++k) {
      tau_score(labels[k]-1) += tau_score_temp(k);
    }

    // tau_score.resize((cov_var_col-nitem)*nclass,1);

    // tau_score.resize(tau_score_j.size()*nclass,1);

    // Score.block(0, h, alpha_length, 1 )=eigen_result_deri;
    
    Score.col(h)<< eigen_result_deri,tau_score;
    
    // Score.block(alpha_length, h, (2^(nitem)-nitem-1), 1 )=tau_score;


    // Score.block(alpha_length, h, (pow(2,nitem)-nitem-1), 1 )=tau_score;
  }






  List result = List::create(
    Named("Score") = Score,
    // Named("tau_score_j") = tau_score_j,
    Named("y_minus_mu") = y_minus_mu);

  return result;
}




// [[Rcpp::export]]
List Fisher_like_information_cpp(const MatrixXd& eta_p,
                                 const List& cond_prob_for_all,
                                 const List& Pi_minus_1,
                                 const MatrixXd& Xprev,
                                 const List& xcond,
                                 const IntegerMatrix& e,
                                 const MatrixXd& e_tran_cor,
                                 const MatrixXd& e_tran,
                                 const MatrixXd& p_i,
                                 const VectorXi& nlevels,
                                 const int& npeop,
                                 const int& nitem,
                                 const int& nclass,
                                 const VectorXi& nxcond,
                                 const MatrixXd& diag_nclass,
                                 const std::vector<std::vector<MatrixXd>>& U,
                                 const std::vector<std::vector<int>> tau_labels,
                                 const std::vector<int> labels,
                                 const int& num_tau,
                                 const int& npar,
                                 const int& alpha_length,
                                 const int& H) {
  
  List cond_prob_list(npeop);
  // MatrixXd cond_prob(H,nclass);
  
  
  MatrixXd Fisher_information= MatrixXd::Zero(npar, npar);
  
  MatrixXd joint_prob_i;
  
  // MatrixXd der_beta;
  
  IntegerVector emk;
  VectorXd em;
  VectorXd Pimkj;
  VectorXd ymk;
  MatrixXd ymk_Pimkj;
  MatrixXd der_parmeter;
  MatrixXd F_hat;
  
  
  
  for(int i = 0; i < npeop; ++i){
    
    MatrixXd cond_prob(H,nclass);
    
    for (int j = 0; j < nclass; ++j) {
      
      MatrixXd cond_prob_for_all_j=cond_prob_for_all[j];
      
      cond_prob.col(j)=cond_prob_for_all_j.row(i);
    }
    
    cond_prob_list[i]=cond_prob;
    // VectorXd eta_p_i=eta_p.row(i);
    joint_prob_i=cond_prob.array().rowwise()*eta_p.row(i).array();
    
    
    
    // MatrixXd eta_i=eta_p.row(i);
    
    VectorXd eta_i = eta_p.row(i).segment(0, nclass - 1);
    // VectorXd pii=p_i.row(i);
    VectorXd pii=p_i.col(i);
    MatrixXd eta_pi=eta_i*pii.transpose();
    
    MatrixXd temp=joint_prob_i.leftCols(nclass - 1).transpose()-eta_pi;
    
    MatrixXd Xprev_i = Xprev.row(i);
    
    // MatrixXd der_beta =
    //   kroneckerProduct_matrix_eigen(temp,Xprev_i.transpose());
    MatrixXd der_beta =
      KroneckerProduct(temp,Xprev_i.transpose());
    
    // MatrixXd der_gamma;
    // MatrixXd der_alpha;
    MatrixXd der_gamma(0, H);
    MatrixXd der_alpha(0, H);
    
    
    List tmp=Score_alpha_tau_matrix_cpp(Pi_minus_1,
                                        eta_p,
                                        cond_prob_for_all,
                                        e_tran_cor,
                                        e_tran,
                                        xcond,
                                        nitem,
                                        nclass,
                                        nlevels,
                                        nxcond,
                                        H,
                                        diag_nclass,
                                        U,
                                        tau_labels,
                                        labels,
                                        num_tau,
                                        alpha_length,
                                        i);
    
    


    
    MatrixXd Score_matrix=tmp["Score"];
    
    // for (int m = 0; m < nitem; ++m) {
    //
    //
    //   for (int k = 0; k < (nlevels[m] - 1); ++k) {
    //
    //     // IntegerVector em=e.col(m);
    //
    //     IntegerVector emk=which_cpp(e(_,m), (k+1));
    //
    //     VectorXd Pimkj= VectorXd::Zero(cond_prob.cols());
    //     VectorXd ymk= VectorXd::Zero(H);
    //     for(int u=0; u<emk.size(); ++u){
    //       Pimkj+=cond_prob.row(emk[u]);
    //
    //       if (emk[u] >= 0 ) {
    //         ymk[emk[u]] = 1.0;
    //       }
    //     }
    //
    //     MatrixXd ymk_Pimkj = ymk.replicate(1, nclass) -
    //       Pimkj.transpose().replicate(H, 1);
    //
    //
    //     MatrixXd der_gamma_mk=
    //       (joint_prob_i.array()*ymk_Pimkj.array()).transpose();
    //
    //     der_gamma=rbindEigen_matrix(der_gamma,der_gamma_mk);
    //
    //
    //
    //     if (nxcond[m]!=0) {
    //       MatrixXd xcond_m = xcond[m];
    //       MatrixXd xcond_m_i = xcond_m.row(i);
    //       MatrixXd der_gamma_mk_sum=der_gamma_mk.array().colwise().sum();
    //
    //       MatrixXd der_alpha_mk=xcond_m_i.transpose()*der_gamma_mk_sum;
    //       der_alpha=rbindEigen_matrix(der_alpha,der_alpha_mk);
    //
    //     }
    //
    //
    //
    //   }
    // }
    
    
    der_parmeter=rbindEigen_matrix(der_beta,Score_matrix);
    // der_parmeter=rbindEigen_matrix(der_parmeter,der_alpha);
    
    // VectorXd pii_1=1.0/pii.array();
    // MatrixXd pii_1_as_diag=pii_1.asDiagonal();
    
    
    // MatrixXd pii_1_as_diag=pii.asDiagonal().inverse();
    
    F_hat=der_parmeter.transpose().array().colwise()/pii.array();
    MatrixXd Fisher_information_i=der_parmeter*F_hat;
    
    
    Fisher_information=Fisher_information+Fisher_information_i;
    
  }
  
  // asymptotic covariance
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(Fisher_information);
  MatrixXd asymptotic_covariance = cod.pseudoInverse();
  
  
  
  List result = List::create(
    Named("Fisher_information") = Fisher_information,
    // Named("cond_prob_list") = cond_prob_list,
    // Named("der_parmeter") = der_parmeter,
    Named("asymptotic_covariance") = asymptotic_covariance);
  
  
  return result;
}
  


// [[Rcpp::export]]
List Fisher_like_information_by_score_cpp(const MatrixXd& eta_p,
                                          const MatrixXd& h_p,
                                          const MatrixXd& Xprev,
                                          const List& Pi_minus_1,
                                          const List& cond_dist_p,
                                          const MatrixXd& e_tran_cor,
                                          const MatrixXd& y_w,
                                          const MatrixXd& Y,
                                          const List& xcond,
                                          const MatrixXd& direct,
                                          const MatrixXd& diag_nclass,
                                          const std::vector<std::vector<MatrixXd>>& U,
                                          const std::vector<std::vector<int>> tau_labels,
                                          const std::vector<int> labels,
                                          const int& num_tau,
                                          const int& npeop,
                                          const int& nxprev,
                                          const int& nitem,
                                          const int& nclass,
                                          const VectorXi& nlevels,
                                          const VectorXi& nxcond,
                                          const int& npar,
                                          const int& alpha_length,
                                          const int& H) {
  
  const int cov_var_col = e_tran_cor.cols();
  const int tau_dim = cov_var_col - nitem;
  
  VectorXd tau_score = VectorXd::Zero(num_tau);
  MatrixXd tau_hessian=MatrixXd::Zero(num_tau,num_tau);
  
  MatrixXd s=h_p.leftCols(nclass-1)-eta_p.leftCols(nclass-1);
  
  
  MatrixXd s_Matrix(npeop,(nclass-1)*(nxprev+1));
  MatrixXd Xprev_Matrix(npeop,(nclass-1)*(nxprev+1));
  
  
  // 利用replicate函數
  for (int j = 0; j < (nclass-1); ++j) {
    s_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = s.col(j).replicate(1, nxprev+1);
    // Xprev_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = Xprev.block(0, 0, npeop, nxprev+1);
    Xprev_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = Xprev;
  }
  
  
  MatrixXd a1_temp=s_Matrix.array()*Xprev_Matrix.array();
  
  
  // VectorXd a1 = a1_temp.colwise().sum();
  // // a1 = a1_temp.colwise().sum();
  //
  // return a1;
  
  
  MatrixXd cov_var;
  VectorXd y_minus_mu;
  VectorXd direction_tau_j;
  
  MatrixXd V_ij_11;
  MatrixXd V_ij_12;
  MatrixXd V_ij_21;
  MatrixXd V_ij_22;
  
  
  
  VectorXd eigen_result_deri = VectorXd::Zero(alpha_length);
  MatrixXd eigen_result_hessian = MatrixXd::Zero(alpha_length,alpha_length);
  // 這樣後續會有錯
  // VectorXd eigen_result_deri(alpha_length);
  // MatrixXd eigen_result_hessian(alpha_length,alpha_length);
  
  // int cols_Vij_eigen = Y.cols();
  int nrow_result = Y.cols() ;
  int ncol_result = Y.cols() * nclass;

  // VectorXd direction_tau((cov_var_col-nitem)*nclass);
  MatrixXd direction_tau((cov_var_col-nitem),nclass);
  
  
  
  // VectorXd tau_score_j;
  // MatrixXd tau_hessian_j;
  
  
  
  
  
  
  // 這樣後續會有錯
  // VectorXd eigen_result_deri(alpha_length);
  // MatrixXd eigen_result_hessian(alpha_length,alpha_length);
  
  MatrixXd der = MatrixXd::Zero(npar,npar);
  
  // int cols_Vij_eigen = Y.cols();
  
  for (int i = 0; i < npeop; ++i) {
    
    VectorXd eigen_result_deri = VectorXd::Zero(alpha_length);
    
    VectorXd der_i(npar);
    
    VectorXd tau_score_temp;
    
    for (int j = 0; j < nclass; ++j) {
      
      // RowVectorXd diag_nclass_j = diag_nclass.row(j);
      MatrixXd diag_nclass_j = diag_nclass.row(j);
      
      MatrixXd Pi_minus_1_j = Pi_minus_1[j];
      MatrixXd S = Y - Pi_minus_1_j;
      
      
      MatrixXd cond_dist_p_j=cond_dist_p[j];
      
      
      // VectorXd tau_score_j=VectorXd::Zero(cov_var_col-nitem);
      // MatrixXd tau_hessian_j=MatrixXd::Zero(cov_var_col-nitem,cov_var_col-nitem);
      
      // tau_score_j=VectorXd::Zero(cov_var_col-nitem);
      // tau_hessian_j=MatrixXd::Zero(cov_var_col-nitem,cov_var_col-nitem);
      // 

      

      
      
      VectorXd margin_prob = cond_dist_p_j.row(i) * e_tran_cor;
      // VectorXd y_minus_mu = y_w.row(i) - margin_prob;
      VectorXd y_w_row_i=y_w.row(i);
      // VectorXd y_minus_mu = y_w_row_i - margin_prob;
      y_minus_mu = y_w_row_i - margin_prob;
      
      // S與前面重複
      // VectorXd S = y_minus_mu.head(nitem);
      // S = y_minus_mu.head(nitem);
      
      
      // e_tran_cor.array().rowwise()-margin_prob;
      
      // Calculate cov_var
      // MatrixXd cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
      cov_var=MatrixXd::Zero(cov_var_col, cov_var_col);
      for(int h=0; h < H; ++h){
        VectorXd e_tran_cor_row_h=e_tran_cor.row(h);
        VectorXd diff = e_tran_cor_row_h - margin_prob;
        cov_var+=(diff*diff.transpose())*cond_dist_p_j(i,h);
      }
      
      // MatrixXd V_ij_11 = cov_var.block(0, 0, nitem, nitem);
      // MatrixXd V_ij_12 = cov_var.block(0, nitem, nitem, cov_var_col);
      // MatrixXd V_ij_21 = cov_var.block(nitem, 0, cov_var_col, nitem);
      // MatrixXd V_ij_22 = cov_var.block(nitem, nitem, cov_var_col, cov_var_col );
      V_ij_11 = cov_var.block(0, 0, nitem, nitem);
      V_ij_12 = cov_var.block(0, nitem, nitem, cov_var_col-nitem);
      V_ij_21 = cov_var.block(nitem, 0, cov_var_col-nitem, nitem);
      V_ij_22 = cov_var.block(nitem, nitem, cov_var_col-nitem, cov_var_col-nitem );
      
      
      
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(V_ij_11);
      MatrixXd V_ij_11_ginv = cod.pseudoInverse();
      
      
      
      
      // RowVectorXd S_i = S.row(i);
      
      VectorXd S_i = S.row(i);
      // MatrixXd S_i=S(i,_);
      
      
      // NumericVector muij = Pi_minus_1_j(i, _);
      
      VectorXd muij = Pi_minus_1_j.row(i);
      
      VectorXd muij_var=muij.array()*(1-muij.array());
      MatrixXd Vij_eigen=muij_var.asDiagonal();
      
      
      
      MatrixXd eigen_Vh_ij_inver_h = V_ij_11_ginv * h_p(i,j);
      
      
      
      // MatrixXd deri_1 = eigen_Vh_ij_inver_h * S_i.transpose();
      
      MatrixXd deri_1 = eigen_Vh_ij_inver_h * S_i;
      
      
      // int nrow_result = Vij_eigen.rows() * diag_nclass_j.rows();
      // int ncol_result = Vij_eigen.cols() * diag_nclass_j.cols();
      
      
      MatrixXd eigen_der_alpha0(nrow_result, ncol_result);
      // for (int k = 0; k < Vij_eigen.cols(); ++k) {
      //
      //   MatrixXd Vij_eigen_temp=Vij_eigen.col(k)*diag_nclass_j;
      //
      //   eigen_der_alpha0.block( 0, k*nclass, Vij_eigen.rows(), nclass )=Vij_eigen_temp;
      // }
      
      for (int k = 0; k < nrow_result; ++k) {
        
        MatrixXd Vij_eigen_temp=Vij_eigen.col(k)*diag_nclass_j;
        
        eigen_der_alpha0.block( 0, k*nclass, nrow_result, nclass )=Vij_eigen_temp;
      }
      
      
      // MatrixXd eigen_der_alpha0 = kroneckerProduct_matrix_eigen(Vij_eigen,eigen_diag_nclass_j);
      
      
      
      // 計算對alpha微分
      MatrixXd eigen_deri_mu_ij;
      if(nxcond.sum() == 0){
        eigen_deri_mu_ij = eigen_der_alpha0;
      }else{
        
        int u = -1;
        // NumericMatrix der_alpha(nitem, 0);
        // MatrixXd der_alpha(nlevels.sum()-nitem, 0);
        MatrixXd der_alpha(nrow_result, 0);
        
        VectorXd xcond_mi;
        
        for(int m = 0; m < nitem; ++m) {
          
          // Rf_isNull(xcond[m])
          // if (nxcond[m]==0) {
          //   // 如果条件满足，跳过当前循环
          //   continue;
          // }
          
          // const NumericMatrix& xcond_m_ref = xcond[m]; // 使用引用
          
          if(nxcond[m]!=0){
            MatrixXd xcond_m = xcond[m];
            xcond_mi= xcond_m.row(i);
            // xcond_mi= xcond_m.row(i);
          }
          
          
          for(int k = 0; k < (nlevels[m] - 1); ++k) {
            
            ++u;
            
            if (nxcond[m]==0) {
              // 如果条件满足，跳过当前循环
              continue;
            }
            
            // const NumericVector& Vij_u = Vij(_,u);
            // VectorXd Vij_u = Vij_eigen.col(u);
            
            MatrixXd Vij_u = Vij_eigen.col(u);
            // NumericVector Vij_u = Vij(_,u);
            
            
            // Map<MatrixXd> eigen_Vij_u(as<Map<MatrixXd> >(Vij_u));
            
            MatrixXd b_temp=Vij_u*xcond_mi.transpose();
            
            // MatrixXd b_temp=Vij_u*xcond_mi;
            
            der_alpha = cbindEigen_matrix(der_alpha, b_temp);
            
          }
        }
        
        eigen_deri_mu_ij = cbindEigen_matrix(eigen_der_alpha0, der_alpha);
      }
      
      
      
      
      
      VectorXd deri = eigen_deri_mu_ij.transpose() * deri_1;
      
      eigen_result_deri = eigen_result_deri + deri;
      
      
      
      
      
      
      
      
      MatrixXd Uij;
      MatrixXd U_comp_T=MatrixXd::Zero(num_tau,tau_dim);
      // for (int k = 0; k < tau_labels[j].size(); ++k) {
      //   U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][i].col(k).transpose();
      // }
      
      
      if(nxcond.sum() == 0){
        Uij=U[j][0];
        for (int k = 0; k < tau_labels[j].size(); ++k) {
          U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][0].col(k).transpose();
        }
      }else{
        Uij=U[j][i];
        for (int k = 0; k < tau_labels[j].size(); ++k) {
          U_comp_T.row(tau_labels[j][k]-1).noalias() += U[j][i].col(k).transpose();
        }
      }
      

      // 計算tau
      VectorXd tau_score_j= h_p(i,j) * Uij.transpose()*
        (y_minus_mu.tail(tau_dim) - V_ij_21 * V_ij_11_ginv * S_i);
      
      tau_score_temp.conservativeResize(tau_score_temp.size() + tau_score_j.size());
      tau_score_temp.tail(tau_score_j.size()) << tau_score_j;
      

      
      // tau_score_j=h_p(i,j)*
      //   (y_minus_mu.tail(cov_var_col - nitem)-
      //   V_ij_21*V_ij_11_ginv*S_i);
      // 
      // tau_score.segment(j*tau_score_j.size(), tau_score_j.size())=tau_score_j;
      
    }
    
    
    VectorXd tau_score = VectorXd::Zero(num_tau);
    for (int k = 0; k < tau_score_temp.size(); ++k) {
      tau_score(labels[k]-1) += tau_score_temp(k);
    }
    
    
    // beta一次就算完
    VectorXd a1=a1_temp.row(i);
    // der_i=rbindEigen_matrix(der_i,eigen_result_deri);
    der_i << a1, eigen_result_deri,tau_score;
    // der_i << a1, eigen_result_deri;
    der+=der_i*der_i.transpose();
    
    
  }
  
  
  
  
  // asymptotic covariance
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(der);
  MatrixXd asymptotic_covariance = cod.pseudoInverse();
  
  
  
  List result = List::create(
    Named("der") = der,
    Named("asymptotic_covariance") = asymptotic_covariance);
  
  return result;
  
}

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
