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


// 
// // [[Rcpp::export]]
// MatrixXd eta_cpp(const MatrixXd& beta,
//                  const MatrixXd& Xprev,
//                  const int& nxprev,
//                  const int& nclass) {
//   
//   // Create the betamatrix by adding a column of zeros
//   MatrixXd betamatrix = MatrixXd::Zero(nxprev + 1, nclass);
//   betamatrix.leftCols(nclass - 1) = beta;
//   
//   // MatrixXd betamatrix(nxprev + 1, nclass);
//   // betamatrix.setZero();
//   // 
//   // betamatrix.block(0, 0, nxprev + 1, nclass - 1) = beta;
//   
//   
//   // Calculate temp = Xprev %*% betamatrix
//   MatrixXd temp = Xprev * betamatrix;
//   
//   // Subtract the maximum value of each row from temp
//   // temp = temp - temp.rowwise().maxCoeff().replicate(1, nclass);
//   
//   temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
// 
//   temp=temp.array().exp();
//   
//   
//   // Calculate eta as the softmax of temp
//   VectorXd rowSumsTemp = temp.rowwise().sum();
//   MatrixXd eta = temp.array().colwise() / rowSumsTemp.array();
//   
//   // MatrixXd eta = temp.array().colwise() / temp.rowwise().sum().array();
//   return eta;
// }



// [[Rcpp::export]]
MatrixXd eta_1_cpp(const MatrixXd& beta,
                   const MatrixXd& Xprev,
                   const int& npeop,
                   const int& nclass) {
  
  // Create the betamatrix by adding a column of zeros
  // MatrixXd betamatrix = MatrixXd::Zero(nxprev + 1, nclass);
  
  // MatrixXd betamatrix(nxprev + 1, nclass); 
  // betamatrix.setZero();
  // 
  // betamatrix.block(0, 0, nxprev + 1, nclass - 1) = beta;
  // 
  // // Calculate temp = Xprev %*% betamatrix
  // MatrixXd temp = Xprev * betamatrix;
  
  MatrixXd temp(npeop,nclass);
  temp.setZero();
  // MatrixXd temp =MatrixXd::Zero(npeop,nclass);
  temp.block(0, 0, npeop, nclass - 1) = Xprev * beta;
  
  // Subtract the maximum value of each row from temp
  // temp = temp - temp.rowwise().maxCoeff().replicate(1, nclass);
  
  temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
  
  
  temp=temp.array().exp();
  
  // Calculate eta as the softmax of temp
  // VectorXd rowSumsTemp = temp.rowwise().sum();
  // MatrixXd eta = temp.array().colwise() / rowSumsTemp.array();

  MatrixXd eta = temp.array().colwise() / temp.rowwise().sum().array();
  return eta;
}


// [[Rcpp::export]]
MatrixXd h_p_1_cpp(const MatrixXd& eta,
                   const MatrixXd& cond_prob) {

  
  
  // const MatrixXd Xprev
  // const int nxprev
  // const int nclass
  // MatrixXd eta = eta_cpp(beta,Xprev,nxprev,nclass);
  
  MatrixXd h_p_temp=cond_prob.array()*eta.array();
  
  
  // 计算每行的和
  VectorXd rowSumsTemp = h_p_temp.rowwise().sum();
  // 按列进行除法
  MatrixXd h_p = h_p_temp.array().colwise() / rowSumsTemp.array();
  
  return h_p;
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
double beta_likeli_cpp(const MatrixXd& beta,
                       const MatrixXd& h_p,
                       const MatrixXd& Xprev,
                       const int& nxprev,
                       const int& nclass) {
  
  MatrixXd eta=eta_cpp(beta,Xprev,nxprev,nclass);
    

  MatrixXd result_temp=h_p.array()*eta.array().log();
  
  // 計算結果總和
  double result=result_temp.sum();
  // double result=result_temp.array().sum();
  
  return result;
}




// [[Rcpp::export]]
VectorXd g1_cpp(const MatrixXd& eta_p,
                const MatrixXd& h_p,
                const MatrixXd& Xprev,
                const int& npeop,
                const int& nxprev,
                const int& nclass) {
  
  
  // // MatrixXd eta_p =eta_cpp(beta,Xprev,nxprev,nclass);
  // 
  // 
  // MatrixXd s=h_p.leftCols(nclass-1)-eta_p.leftCols(nclass-1);
  // 
  // 
  // // MatrixXd s_Matrix(npeop,(nclass-1)*(nxprev+1));
  // // MatrixXd Xprev_Matrix(npeop,(nclass-1)*(nxprev+1));
  // VectorXd a1((nclass-1)*(nxprev+1));
  // 
  // // 利用replicate函數
  // // for (int j = 0; j < (nclass-1); ++j) {
  // //   s_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = s.col(j).replicate(1, nxprev+1);
  // //   // 不確定這樣改會不會有影響
  // //   // Xprev_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = Xprev.block(0, 0, npeop, nxprev+1);
  // //   Xprev_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = Xprev;
  // // }
  // // for (int j = 0; j < (nclass-1); ++j) {
  // //   s_Matrix.middleCols(j*(nxprev+1), nxprev+1)= s.col(j).replicate(1, nxprev+1);
  // //   // 不確定這樣改會不會有影響
  // //   // Xprev_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = Xprev.block(0, 0, npeop, nxprev+1);
  // //   Xprev_Matrix.middleCols(j*(nxprev+1), nxprev+1) = Xprev;
  // // }
  // 
  // 
  // for (int j = 0; j < (nclass-1); ++j) {
  // 
  //   a1.segment(j*(nxprev+1),nxprev+1)=Xprev.transpose() * s.col(j);
  // }

  
  MatrixXd tmp=Xprev.transpose() *
    ( h_p.leftCols(nclass - 1)-eta_p.leftCols(nclass - 1) );
  // Eigen::VectorXd a1 = Eigen::Map<Eigen::VectorXd>(mat.data(), mat.size());
  tmp.resize(tmp.size(),1);
  return tmp;
  
  // MatrixXd a1_temp=s_Matrix.array()*Xprev_Matrix.array();
  // VectorXd a1 = a1_temp.colwise().sum();
  // // a1 = a1_temp.colwise().sum();
  
  // VectorXd a1 = (s_Matrix.array()*Xprev_Matrix.array()).colwise().sum();
  // return a1;
  // return s_Matrix.cwiseProduct(Xprev_Matrix).colwise().sum();
  
}





// [[Rcpp::export]]
VectorXd g11_cpp_optimized(const MatrixXd& eta_p,
                           const MatrixXd& h_p,
                           const MatrixXd& Xprev,
                           int npeop,
                           int nxprev,
                           int nclass) {
  
  const int nclass_minus_1 = nclass - 1;
  const int block_size = nxprev + 1;
  const int total_cols = nclass_minus_1 * block_size;
  
  // 预分配结果向量
  VectorXd result = VectorXd::Zero(total_cols);
  
  // 计算差值矩阵
  MatrixXd s_diff = h_p.leftCols(nclass_minus_1) - eta_p.leftCols(nclass_minus_1);
  
  
  for (int j = 0; j < nclass_minus_1; ++j) {
    const int start_col = j * block_size;
    const VectorXd s_col = s_diff.col(j);
    
    for (int k = 0; k < block_size; ++k) {
      // 直接计算点积，避免存储完整矩阵
      result(start_col + k) = s_col.dot(Xprev.col(k));
    }
  }
  
  return result;
}



// Function to compute Kronecker product of two matrices
// [[Rcpp::export]]
MatrixXd kroneckerProduct_matrix_eigen(const MatrixXd& A, 
                                       const MatrixXd& B) {
  
  int nrow_A = A.rows();
  int ncol_A = A.cols();
  int nrow_B = B.rows();
  int ncol_B = B.cols();
  
  // int nrow_result = nrow_A * nrow_B;
  // int ncol_result = ncol_A * ncol_B;
  // 
  // MatrixXd result(nrow_result, ncol_result);
  
  MatrixXd result(nrow_A * nrow_B, ncol_A * ncol_B);
  
  for (int i = 0; i < nrow_A; ++i) {
    for (int j = 0; j < ncol_A; ++j) {
      result.block(i * nrow_B, j * ncol_B, nrow_B, ncol_B) = A(i, j) * B;
    }
  }
  
  return result;
}



// 并行 Kronecker Product
MatrixXd kroneckerProductParallel(const MatrixXd &A, 
                                  const MatrixXd &B) {
  int m = A.rows(), n = A.cols();
  int p = B.rows(), q = B.cols();
  MatrixXd C(m * p, n * q);
  
#pragma omp parallel for collapse(2)  // 使用 OpenMP 并行化
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C.block(i * p, j * q, p, q) = A(i, j) * B;
    }
  }
  return C;
}





// [[Rcpp::export]]
VectorXd g1_test_cpp(const MatrixXd& eta_p,
                     const MatrixXd& h_p,
                     const MatrixXd& Xprev,
                     const int& npeop,
                     const int& nxprev,
                     const int& nclass) {
  
  
  // MatrixXd eta_p =eta_cpp(beta,Xprev,nxprev,nclass);
  
  
  MatrixXd s=h_p.leftCols(nclass-1)-eta_p.leftCols(nclass-1);
  
  // 
  // MatrixXd s_Matrix(npeop,(nclass-1)*(nxprev+1));
  // MatrixXd Xprev_Matrix(npeop,(nclass-1)*(nxprev+1));
  // 
  // 
  // // 利用replicate函數
  // for (int j = 0; j < (nclass-1); ++j) {
  //   s_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = s.col(j).replicate(1, nxprev+1);
  //   // 不確定這樣改會不會有影響
  //   // Xprev_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = Xprev.block(0, 0, npeop, nxprev+1);
  //   Xprev_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = Xprev;
  // }
  // 
  
  // RowVectorXd ones_Xprev = RowVectorXd::Ones(nclass-1);
  MatrixXd Xprev_Matrix = kroneckerProductParallel(RowVectorXd::Ones(nclass-1),Xprev); 
  
  // RowVectorXd ones_s = RowVectorXd::Ones(nxprev+1);
  MatrixXd s_Matrix=kroneckerProductParallel(s,RowVectorXd::Ones(nxprev+1)); 
  
  // MatrixXd a1_temp=s_Matrix.array()*Xprev_Matrix.array();
  // VectorXd a1 = a1_temp.colwise().sum();
  // // a1 = a1_temp.colwise().sum();
  
  VectorXd a1 = (s_Matrix.array()*Xprev_Matrix.array()).colwise().sum();
  return a1;
}


// [[Rcpp::export]]
VectorXd g1_test_2_cpp(const MatrixXd& eta_p,
                       const MatrixXd& h_p,
                       const MatrixXd& Xprev,
                       const int& npeop,
                       const int& nxprev,
                       const int& nclass) {
  
  
  // MatrixXd eta_p =eta_cpp(beta,Xprev,nxprev,nclass);
  
  
  MatrixXd s=h_p.leftCols(nclass-1)-eta_p.leftCols(nclass-1);
  
  // 
  // MatrixXd s_Matrix(npeop,(nclass-1)*(nxprev+1));
  // MatrixXd Xprev_Matrix(npeop,(nclass-1)*(nxprev+1));
  // 
  // 
  // // 利用replicate函數
  // for (int j = 0; j < (nclass-1); ++j) {
  //   s_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = s.col(j).replicate(1, nxprev+1);
  //   // 不確定這樣改會不會有影響
  //   // Xprev_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = Xprev.block(0, 0, npeop, nxprev+1);
  //   Xprev_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = Xprev;
  // }
  // 
  
  // RowVectorXd ones_Xprev = RowVectorXd::Ones(nclass-1);
  MatrixXd Xprev_Matrix = KroneckerProduct(RowVectorXd::Ones(nclass-1),Xprev); 
  
  // RowVectorXd ones_s = RowVectorXd::Ones(nxprev+1);
  MatrixXd s_Matrix = KroneckerProduct(s,RowVectorXd::Ones(nxprev+1)); 
  
  // MatrixXd a1_temp=s_Matrix.array()*Xprev_Matrix.array();
  // VectorXd a1 = a1_temp.colwise().sum();
  // // a1 = a1_temp.colwise().sum();
  
  VectorXd a1 = (s_Matrix.array()*Xprev_Matrix.array()).colwise().sum();
  return a1;
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
MatrixXd Hess1_parallel_cpp(const MatrixXd& eta_p,
                            const MatrixXd& Xprev,
                            const int& npeop,
                            const int& nxprev,
                            const int& nclass) {
  
  MatrixXd hess= MatrixXd::Zero((nxprev+1)*(nclass-1), (nxprev+1)*(nclass-1));
  
  
  // 平行處理的部分
#pragma omp parallel
{
  MatrixXd hess_local = MatrixXd::Zero(hess.rows(), hess.cols());
  
  VectorXd eta_i=VectorXd::Zero(nclass-1);
  MatrixXd eta_i_diag= MatrixXd::Zero(nclass - 1,nclass - 1) ;
  MatrixXd term4 = MatrixXd::Zero(nclass - 1,nclass - 1);
  MatrixXd term2 = MatrixXd::Zero(nxprev+1,nxprev+1);
  // #pragma omp critical
  
#pragma omp for schedule(dynamic)
  for (int i = 0; i < npeop; ++i) {
    // 提取當前的 eta_i 和 Xprev_i
    eta_i = eta_p.row(i).segment(0, nclass - 1);
    eta_i_diag = MatrixXd::Zero(nclass - 1, nclass - 1);
    eta_i_diag.diagonal() = eta_i;
    
    term4 = eta_i_diag - eta_i * eta_i.transpose();
    term2 = Xprev.row(i).transpose() * Xprev.row(i);
    
    // 局部更新 hess_local
    hess_local -= KroneckerProduct(term4, term2);
    // hess -= kroneckerProduct_matrix_eigen(term4, term2);
  }
  
  // 將每個執行緒的 hess_local 累加到全局 hess
#pragma omp critical
{
  hess += hess_local;
}

// hess += hess_local;
}

return hess;
}



// 先定義對 Eigen::MatrixXd 的 reduction
#pragma omp declare reduction(MatrixAdd : MatrixXd : omp_out += omp_in) \
    initializer(omp_priv = MatrixXd::Zero(omp_orig.rows(), omp_orig.cols()))

    
#pragma omp declare reduction(VectorAdd : VectorXd : omp_out += omp_in) \
    initializer(omp_priv = VectorXd::Zero(omp_orig.size()))  
  

// [[Rcpp::export]]
MatrixXd Hess1_parallel_test_cpp(const MatrixXd& eta_p,
                                 const MatrixXd& Xprev,
                                 const int& npeop,
                                 const int& nxprev,
                                 const int& nclass) {
    
    // MatrixXd hess = MatrixXd::Zero((nxprev+1)*(nclass-1), (nxprev+1)*(nclass-1));
    MatrixXd hess((nxprev+1)*(nclass-1), (nxprev+1)*(nclass-1));
    hess.setZero();
    
    // 使用 reduction 直接累加
#pragma omp parallel for reduction(MatrixAdd: hess) schedule(dynamic)
    for (int i = 0; i < npeop; ++i) {
      // 提取當前的 eta_i 和 Xprev_i
      VectorXd eta_i = eta_p.row(i).segment(0, nclass - 1);
      // MatrixXd eta_i_diag = MatrixXd::Zero(nclass - 1, nclass - 1);
      MatrixXd eta_i_diag(nclass - 1, nclass - 1);
      eta_i_diag.setZero();
      eta_i_diag.diagonal() = eta_i;
      
      // MatrixXd term4 = eta_i_diag - eta_i * eta_i.transpose();
      // MatrixXd term2 = Xprev.row(i).transpose() * Xprev.row(i);
      // 
      // // 每次迭代直接累加到 hess 中
      // hess -= KroneckerProduct(term4, term2);
      
      hess -= KroneckerProduct(eta_i_diag - eta_i * eta_i.transpose(), 
                               Xprev.row(i).transpose() * Xprev.row(i));
    }
    
    return hess;
}





// [[Rcpp::export]]
List max_beta_bisection_cpp(const MatrixXd& beta,
                            const MatrixXd& Xprev,
                            const MatrixXd& eta_p,
                            const MatrixXd& h_p,
                            const int& npeop,
                            const int& nxprev,
                            const int& nclass,
                            double& c,
                            const double& err) {
  
  
  // MatrixXd eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
  VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
  MatrixXd hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);

  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
  MatrixXd ginv_hess = cod.pseudoInverse();
  MatrixXd direction = -1*ginv_hess * deri;
  

  direction.resize(nxprev+1 ,nclass-1);



  // NumericVector temp_aa(50);
  // NumericVector temp_cc(50);

  double temp_a;
  double temp_c;
  double temp_b;
  double temp_a1;
  double temp_a2;
  
  
  double a = 0.0;
  int iter = 0;
  for( iter = 0; iter < 1000; ++iter) {
    temp_a = a;
    temp_c = c;

    
    temp_b = (a + c) / 2.0;

    temp_a1 = beta_likeli_cpp( beta + temp_b * direction,
                                      h_p,
                                      Xprev,
                                      nxprev,
                                      nclass);


    temp_a2 = beta_likeli_cpp( beta + (temp_b + 0.02) * direction,
                                      h_p,
                                      Xprev,
                                      nxprev,
                                      nclass);


    if(temp_a1 < temp_a2) {
      a = temp_b - 0.02;
      c = c;

      // temp_aa[iter]=a;
      // temp_cc[iter]=c;

      if( abs(temp_a - a) < 0.001 ) break;
    } else {
      a = a;
      c = temp_b + 0.02;

      // temp_aa[iter]=a;
      // temp_cc[iter]=c;

      if( abs(temp_c - c) < 0.001 ) break;
    }

    // if( std::max( abs(temp_a - a),abs(temp_c - c) ) < 0.001 ) break;
  }



  int len = static_cast<int>((c - a) / err) + 1;
  NumericVector  likeli(len+3);
  NumericVector ak(len+3);
  List beta_list(len+3);


  
  ak[0]=0;
  beta_list[0]=beta;
  likeli[0] = beta_likeli_cpp( beta,
                                   h_p,
                                   Xprev,
                                   nxprev,
                                   nclass);
  
  
  int max_idx=0;

  for(int i = 0; i < len; ++i) {
    ak[i+1]=a + i * err;
    beta_list[i+1]=beta + ak[i]* direction;
    likeli[i+1] = beta_likeli_cpp( beta_list[i+1],
                                 h_p,
                                 Xprev,
                                 nxprev,
                                 nclass);
    
    if( likeli[i+1] >= likeli[i] ){
      max_idx=i+1;
    }

  }

  ak[len+1]=c;
  beta_list[len+1]=beta + c* direction;
  likeli[len+1] = beta_likeli_cpp( beta_list[len+1],
                                 h_p,
                                 Xprev,
                                 nxprev,
                                 nclass);
  
  if( likeli[len+1] >= likeli[max_idx] ){
    max_idx=len+1;
  }


  
  ak[len+2]=1;
  beta_list[len+2]=beta+direction;
  likeli[len+2] = beta_likeli_cpp( beta_list[len+2],
                                   h_p,
                                   Xprev,
                                   nxprev,
                                   nclass);

  if( likeli[len+2] >= likeli[max_idx] ){
    max_idx=len+2;
  }



  // double max_tempp = likeli.maxCoeff();


  // int max_idx = Rcpp::which_max(likeli);

  MatrixXd beta_p=beta_list[max_idx];


  // NumericVector alpha_vector_p=alpha_vector;
  double ak_temp=ak[max_idx];

  // 创建一个 List 对象包含多个值
  List result = List::create(
    Named("a") = a,
    Named("c") = c,
    Named("likeli") = likeli,
    Named("len") = len,
    Named("ak") = ak,
    Named("ak_temp") = ak_temp,
    Named("beta_p") = beta_p,
    Named("iter") = iter);
  
  
  return result;
}



// [[Rcpp::export]]
// 用backtracking algorithm
List max_beta_4_cpp(MatrixXd& beta,
                    const MatrixXd& Xprev,
                    MatrixXd& eta_p,
                    const MatrixXd& h_p,
                    const int& npeop,
                    const int& nxprev,
                    const int& nclass,
                    const int& maxiter,
                    const double& step_length,
                    const double& tol) {
  
  
  MatrixXd eta=eta_p;
  // MatrixXd eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
  VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
  MatrixXd hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
  // MatrixXd ginv_hess = cod.pseudoInverse();
  // // MatrixXd ginv_hess = hess.inverse();
  // 
  // 
  // MatrixXd direction = -1*ginv_hess * deri;
  MatrixXd direction = -1*cod.pseudoInverse() * deri;
  
  
  direction.resize(nxprev+1 ,nclass-1);
  
  
  // MatrixXd eta=eta_cpp(beta,Xprev,nxprev,nclass);
  
  
  // MatrixXd result_temp=h_p.array()*eta_p.array().log();
  // 
  // // 計算結果總和
  // double llik=result_temp.sum();
  
  
  double llik=(h_p.array()*eta_p.array().log()).sum();
  
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  
  double llik_p;
  
  MatrixXd beta_p;
  double a=1.0;
  
  int iter = 0;
  for(iter = 0; iter < maxiter; ++iter){
    
    
    beta_p=beta+a*direction;
    eta_p=eta_cpp(beta_p,Xprev,nxprev,nclass);
    llik_p=(h_p.array()*eta_p.array().log()).sum();
    // llik_p= beta_likeli_cpp(beta_p,h_p,Xprev,nxprev,nclass);
    
    
    // max_beta_score_bisection_1_cpp(beta,Xprev,eta_p,h_p,
    //                                     npeop,nxprev,nclass,err);
    
    
    // double max_diff = (beta_p - beta).cwiseAbs().maxCoeff();
    // Rcpp::Rcout << "Iteration " << iter << ", Max Difference: " << max_diff << std::endl;
    // beta=beta_p;
    
    
    
    if(llik_p>llik){
      // Rcpp::Rcout << "Iteration " << iter << ", llik: " << llik_p << std::endl;
      beta=beta_p;
      llik= llik_p;
      eta=eta_p;
      if((a*direction).cwiseAbs().maxCoeff()<tol) break;
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
      deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
      hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
      // ginv_hess = cod.pseudoInverse();
      // // MatrixXd ginv_hess = hess.inverse();
      // 
      // 
      // direction = -1*ginv_hess * deri;
      direction = -1*cod.pseudoInverse() * deri;
      
      direction.resize(nxprev+1 ,nclass-1);
      
      
    }else{
      a=a*step_length;
      if((a*direction).cwiseAbs().maxCoeff()<tol) break;
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
    
    
// // [[Rcpp::export]]
// std::tuple<MatrixXd, MatrixXd, double> max_beta_6_for_omp_cpp(MatrixXd beta,
//                              const MatrixXd& Xprev,
//                              MatrixXd eta_p,
//                              const MatrixXd& h_p,
//                              const int npeop,
//                              const int nxprev,
//                              const int nclass,
//                              const int maxiter,
//                              const double step_length,
//                              const double tol,
//                              const int it) {
//   
//   int count = 0;
//   MatrixXd eta = eta_p;
//   VectorXd deri = g1_cpp(eta_p, h_p, Xprev, npeop, nxprev, nclass);
//   MatrixXd hess = Hess1_cpp(eta_p, Xprev, npeop, nxprev, nclass);
//   
//   CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
//   MatrixXd direction = -1 * cod.pseudoInverse() * deri;
//   direction.resize(nxprev + 1, nclass - 1);
//   
//   double llik = (h_p.array() * eta_p.array().log()).sum();
//   double llik_p;
//   MatrixXd beta_p;
//   double a = 1.0;
//   
//   int iter = 0;
//   for (iter = 0; iter < maxiter; ++iter) {
//     beta_p = beta + a * direction;
//     eta_p = eta_cpp(beta_p, Xprev, nxprev, nclass);
//     llik_p = (h_p.array() * eta_p.array().log()).sum();
//     
//     if (llik_p > llik) {
//       ++count;
//       beta = beta_p;
//       eta = eta_p;
//       llik = llik_p;
//       
//       if (a * direction.cwiseAbs().maxCoeff() < tol || count == it)
//         break;
//       
//       deri = g1_cpp(eta_p, h_p, Xprev, npeop, nxprev, nclass);
//       hess = Hess1_cpp(eta_p, Xprev, npeop, nxprev, nclass);
//       cod.compute(hess);
//       direction = -1 * cod.pseudoInverse() * deri;
//       direction.resize(nxprev + 1, nclass - 1);
//       
//     } else {
//       a *= step_length;
//       if (a * direction.cwiseAbs().maxCoeff() < tol)
//         break;
//     }
//   }
//   
//   return std::make_tuple(beta, eta, llik);
// }
    
    
    
    
// [[Rcpp::export]]
// 用backtracking algorithm
List max_beta_7_cpp(MatrixXd& beta,
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
  VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
  MatrixXd hess = Hess1_parallel_test_cpp(eta_p,Xprev,npeop,nxprev,nclass);
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
  // MatrixXd ginv_hess = cod.pseudoInverse();
  // // MatrixXd ginv_hess = hess.inverse();
  //
  //
  // MatrixXd direction = -1*ginv_hess * deri;


  MatrixXd direction = -1*cod.pseudoInverse() * deri;
  
  // // 使用 CompleteOrthogonalDecomposition 來求解
  // CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_decomp(hess);
  // MatrixXd direction = -cod_decomp.solve(deri);
  direction.resize(nxprev+1 ,nclass-1);
  
  
  
  // MatrixXd temp = -1 * cod.pseudoInverse() * deri;
  // MatrixXd direction = Map<MatrixXd>(temp.data(), nxprev+1, nclass-1);

  
  // MatrixXd direction = -1 * cod.pseudoInverse() * 
  //   Eigen::Map<MatrixXd>(deri.data(), nxprev+1, nclass-1);
  
  // MatrixXd eta=eta_cpp(beta,Xprev,nxprev,nclass);
  
  
  // MatrixXd result_temp=h_p.array()*eta_p.array().log();
  // 
  // // 計算結果總和
  // double llik=result_temp.sum();
  
  
  double llik=(h_p.array()*eta_p.array().log()).sum();
  
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  
  double llik_p;
  
  MatrixXd beta_p;
  double a=1.0;
  
  int iter = 0;
  for(iter = 0; iter < maxiter; ++iter){
    
    
    beta_p=beta+a*direction;
    eta_p=eta_cpp(beta_p,Xprev,nxprev,nclass);
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
      beta=beta_p;
      llik= llik_p;
      eta=eta_p;
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
      deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
      hess = Hess1_parallel_test_cpp(eta_p,Xprev,npeop,nxprev,nclass);
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
      // ginv_hess = cod.pseudoInverse();
      // // MatrixXd ginv_hess = hess.inverse();
      //
      //
      // direction = -1*ginv_hess * deri;

      direction = -1*cod.pseudoInverse() * deri;
      
      // // 使用 CompleteOrthogonalDecomposition 來求解
      // CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_decomp(hess);
      // direction = -cod_decomp.solve(deri);
      
      direction.resize(nxprev+1 ,nclass-1);
      
      // temp = -1 * cod.pseudoInverse() * deri;
      // direction = Eigen::Map<MatrixXd>(temp.data(), nxprev+1, nclass-1);

      
      // direction = -1 * cod.pseudoInverse() * 
      //   Eigen::Map<MatrixXd>(deri.data(), nxprev+1, nclass-1);
      
    }else{
      a=a*step_length;
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
List calculate_eta_beta_vec(const Eigen::MatrixXd& h_p, int nclass, int npeop) {
  // eta_p = colMeans(h_p)
  VectorXd eta_p = h_p.colwise().mean();
  
  // beta_p = log(eta_p[1:(nclass-1)]/(1-eta_p[1:(nclass-1)]))
  VectorXd eta_sub = eta_p.head(nclass - 1);
  VectorXd beta_vec = (eta_sub.array() / (1.0 - eta_sub.array())).log();
  MatrixXd beta_p = beta_vec.transpose(); // 1 x (nclass-1) matrix
  
  // eta_p = matrix(eta_p, npeop, nclass, byrow = T)
  MatrixXd eta_p_matrix = eta_p.transpose().replicate(npeop, 1);
  
  return List::create(
    _["eta_p"] = eta_p,
    _["beta_p"] = beta_p,
    _["eta_p_matrix"] = eta_p_matrix
  );
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
std::vector<MatrixXd> max_beta_6_no_covariate_for_omp_cpp(MatrixXd& beta,
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
  // MatrixXd hess= MatrixXd::Zero(nclass - 1,nclass - 1);
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
  

  for(int iter = 0; iter < maxiter; ++iter){
    
    
    beta_p.noalias()=beta+a*direction.transpose();
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
  
  // std::vector<MatrixXd> result(2);
  // result[0] = beta; // 將 cond_prob 包裝成 vector
  // result[1] = eta;  // 直接賦值
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
std::vector<std::vector<MatrixXd>>  tran_vec_to_list_for_omp_cpp(const VectorXd& alpha_vector,
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





// // [[Rcpp::export]]
// MatrixXd rbindEigen_matrix(const MatrixXd& mat1, 
//                            const MatrixXd& mat2) {
//   int nrow1 = mat1.rows();
//   int nrow2 = mat2.rows();
//   
//   int ncol1 = mat1.cols();
//   // int ncol2 = mat2.cols();
//   
//   // Create a new matrix with the combined number of columns
//   MatrixXd result(nrow1+nrow2, ncol1);
//   
//   // Copy data from mat1 to the result matrix
//   result.block(0, 0, nrow1, ncol1) = mat1;
//   
//   // Copy data from mat2 to the result matrix
//   result.block(nrow1, 0, nrow2, ncol1) = mat2;
//   
//   return result;
// }


// [[Rcpp::export]]
Eigen::MatrixXd rbindEigen_matrix(const Eigen::MatrixXd& mat1, 
                                  const Eigen::MatrixXd& mat2) {
  // 预分配结果矩阵（直接指定最终尺寸）
  Eigen::MatrixXd result(mat1.rows() + mat2.rows(), mat1.cols());
  
  // 使用 .topRows() 和 .bottomRows() 快速填充
  result.topRows(mat1.rows()) = mat1;
  result.bottomRows(mat2.rows()) = mat2;
  
  return result;
}    
    
    



// // Function to perform column-wise concatenation
// // [[Rcpp::export]]
// MatrixXd cbindEigen_matrix(const MatrixXd& mat1, 
//                            const MatrixXd& mat2) {
//   int nrow = mat1.rows();
//   int ncol1 = mat1.cols();
//   int ncol2 = mat2.cols();
//   
//   // Create a new matrix with the combined number of columns
//   MatrixXd result(nrow, ncol1 + ncol2);
//   
//   // Copy data from mat1 to the result matrix
//   result.block(0, 0, nrow, ncol1) = mat1;
//   
//   // Copy data from mat2 to the result matrix
//   result.block(0, ncol1, nrow, ncol2) = mat2;
//   
//   return result;
// }


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
  int cov_var_col=e_tran_cor.cols();
  // VectorXd direction_tau((cov_var_col-nitem)*nclass);
  MatrixXd direction_tau((cov_var_col-nitem),nclass);
  
  
  
  VectorXd tau_score_j;
  MatrixXd tau_hessian_j;
  
  
  for (int j = 0; j < nclass; ++j) {
    
    // RowVectorXd diag_nclass_j = diag_nclass.row(j);
    MatrixXd diag_nclass_j = diag_nclass.row(j);
    
    MatrixXd Pi_minus_1_j = Pi_minus_1[j];
    MatrixXd S = Y - Pi_minus_1_j;
    
    
    MatrixXd cond_dist_p_j=cond_dist_p[j];
    
    
    // VectorXd tau_score_j=VectorXd::Zero(cov_var_col-nitem);
    // MatrixXd tau_hessian_j=MatrixXd::Zero(cov_var_col-nitem,cov_var_col-nitem);
    
    tau_score_j=VectorXd::Zero(cov_var_col-nitem);
    tau_hessian_j=MatrixXd::Zero(cov_var_col-nitem,cov_var_col-nitem);
    
    
    for (int i = 0; i < npeop; ++i) {
      
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
      
      // NumericMatrix temp = calculateTemp(muij);
      
      
      // MatrixXd temp = muij * muij.transpose();
      // 
      // 
      // MatrixXd temp_dir=temp.array()*direct.array();
      // 
      // // int nn = muij.size();
      // // MatrixXd mat=MatrixXd::Zero(nn, nn);
      // // for (int ii = 0; ii < nn; ++ii) {
      // //   mat(ii, ii) = muij[ii];
      // // }
      // 
      // MatrixXd muij_as_diag=muij.asDiagonal();
      // 
      // MatrixXd Vij_eigen =muij_as_diag-temp_dir;
      // 
      // 
      // 
      // // NumericMatrix Vij=wrap(Vij_eigen);
      // 
      // 
      // // 計算ginv
      // CompleteOrthogonalDecomposition<MatrixXd> cod(Vij_eigen);
      // MatrixXd Vij_inver = cod.pseudoInverse();
      // // MatrixXd Vij_inver = Vij_eigen.inverse();
      // 
      // 
      // 
      // // NumericMatrix Vij_inver = ginv_rcpp(Vij);
      // // NumericMatrix Vh_ij_inver_h = Vij_inver * h_p(i,j);
      // // double element = h_p(i,j);
      // // MatrixXd eigen_Vh_ij_inver_h = Vij_inver * element;
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
      
      MatrixXd hessian = eigen_deri_mu_ij.transpose() * eigen_Vh_ij_inver_h*
        eigen_deri_mu_ij;
      
      
      eigen_result_deri = eigen_result_deri + deri;
      
      eigen_result_hessian=eigen_result_hessian+hessian;
      
    }
    
    
    
    
  }
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(eigen_result_hessian);
  MatrixXd alpha_ginv = cod.pseudoInverse();
  VectorXd direction_alpha = alpha_ginv * eigen_result_deri;
  
  
  // List result = List::create(
  //   Named("deri") = eigen_result_deri,
  //   Named("hessian") = eigen_result_hessian,
  //   Named("cov_var") = cov_var,
  //   Named("y_minus_mu") = y_minus_mu,
  //   Named("direction_alpha") = direction_alpha,
  //   Named("direction_tau") = direction_tau,
  //   Named("direction_tau_j") = direction_tau_j,
  //   Named("tau_score_j") = tau_score_j,
  //   Named("tau_hessian_j") = tau_hessian_j,
  //   Named("V_ij_11") = V_ij_11,
  //   Named("V_ij_12") = V_ij_12,
  //   Named("V_ij_21") = V_ij_21,
  //   Named("V_ij_22") = V_ij_22);
  
  
  
  // List result = List::create(
  //   Named("deri") = eigen_result_deri,
  //   Named("hessian") = eigen_result_hessian,
  //   Named("direction_alpha") = direction_alpha,
  //   Named("direction_tau") = direction_tau);
  
  // return result;
  return direction_alpha;
}




// [[Rcpp::export]]
// only use for the function log_like_cpp
MatrixXd cond_prob_cpp(const List& alpha0, 
                       const List& alpha,
                       const MatrixXd& Y_comp,
                       const List& xcond,
                       const VectorXi& nxcond,
                       const int& npeop,
                       const int& nitem,
                       const int& nclass) {
  
  
  MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  // MatrixXd Pi_j=MatrixXd::Zero(npeop, nxcond.sum());
  
  // MatrixXd temp_matrix(npeop,Y_comp.cols());
  MatrixXd temp_matrix;
  MatrixXd cond_prob(npeop, nclass);
  
  
  
  for (int j = 0; j < nclass; ++j) {
    

    MatrixXd Pi_j(npeop, 0);
    
    // int temp_nxcond=0;
    for (int m = 0; m < nitem; ++m) {
      
      // Rf_isNull(xcond[m])
      if ( nxcond[m]==0 ) {
        MatrixXd alpha0_m = alpha0[m];
        VectorXd alpha0_m_j=alpha0_m.row(j);
        
        // MatrixXd eigen_A =onesVector;
        
        // 创建一个MatrixXd
        MatrixXd matrixA(1, alpha0_m_j.size());
        matrixA.row(0)=alpha0_m_j;
        
        
        
        // MatrixXd mat_R=matrixA;
        
        MatrixXd zeroVector_c = MatrixXd::Zero(matrixA.rows(), 1);
        
        MatrixXd eigen_B =cbindEigen_matrix(matrixA,zeroVector_c);
        
        
        
        // MatrixXd eigen_B = as<Map<MatrixXd> >( cbind(rbind(alpha0_m.row(j), alpha_m), 0.0) );
        // MatrixXd temp = eigen_A * eigen_B;
        MatrixXd temp = onesVector * eigen_B;
        // temp = temp - temp.rowwise().maxCoeff().replicate(1, temp.cols());
        
        temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
        
        MatrixXd inputMatrix = temp.array().exp();
        VectorXd inputVector = inputMatrix.rowwise().sum();
        MatrixXd Pimkj=inputMatrix.array().colwise() / inputVector.array();
        
        
        Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
        // Pi_j.block(0, temp_nxcond, npeop, nxcond[m]) = Pimkj;

      }else{
        MatrixXd alpha0_m = alpha0[m];
        MatrixXd alpha_m = alpha[m];
        MatrixXd xcond_m = xcond[m];
        
        
        
        
        
        MatrixXd eigen_A =cbindEigen_matrix(onesVector,xcond_m);
        
        
        // MatrixXd eigen_A = as<Map<MatrixXd> >(cbind(1.0, xcond_m));
        
        // NumericVector alpha0_m_j=alpha0_m(j,_);
        // Map<VectorXd> eigen_alpha0_m_j(as<Map<VectorXd> >(alpha0_m_j));
        
        VectorXd alpha0_m_j=alpha0_m.row(j);
        
        
        // 创建一个MatrixXd
        MatrixXd matrixA(1, alpha0_m_j.size());
        matrixA.row(0)=alpha0_m_j;
        
        
        // Map<MatrixXd> eigen_alpha_m(as<Map<MatrixXd> >(alpha_m));
        MatrixXd mat_R=rbindEigen_matrix(matrixA, alpha_m);
        
        MatrixXd zeroVector_c = MatrixXd::Zero(mat_R.rows(), 1);
        
        MatrixXd eigen_B =cbindEigen_matrix(mat_R,zeroVector_c);
        
        
        // MatrixXd eigen_B = as<Map<MatrixXd> >( cbind(rbind(alpha0_m.row(j), alpha_m), 0.0) );
        MatrixXd temp = eigen_A * eigen_B;
        // temp = temp - temp.rowwise().maxCoeff().replicate(1, temp.cols());
        
        temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
        // 以後再回來處理
        // NumericVector aa = apply(temp, 1, max);
        // temp = exp(temp - aa);
        
        // NumericMatrix Pimkj = temp / rowSumsCpp(temp);
        
        // Convert Eigen matrix to NumericMatrix
        
        
        MatrixXd inputMatrix=temp.array().exp();
        VectorXd inputVector = inputMatrix.rowwise().sum();
        MatrixXd Pimkj=inputMatrix.array().colwise() / inputVector.array();
        
        Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
        // Pi_j.block(0, temp_nxcond, npeop, nxcond[m]) = Pimkj;
      }
      
      // temp_nxcond=temp_nxcond+nxcond[m];
      
    }
    
    temp_matrix = Pi_j.array()*Y_comp.array()+(1-Y_comp.array());
    
    cond_prob.col(j) = temp_matrix.array().log().rowwise().sum().exp();
  }
  
  return cond_prob;
}




// [[Rcpp::export]]
double log_like_cpp(const MatrixXd& beta,
                    const List& alpha0, 
                    const List& alpha,
                    const MatrixXd& Y_comp,
                    const MatrixXd& Xprev,
                    const List& xcond,
                    const VectorXi& nxcond,
                    const int& npeop,
                    const int& nitem,
                    const int& nxprev,
                    const int& nclass) {
  
  // MatrixXd cond_temp=cond_prob_cpp(alpha0,alpha);
  
  MatrixXd cond_temp=cond_prob_cpp(alpha0,alpha,Y_comp,
                                   xcond,nxcond,npeop,nitem,nclass); 
  
  MatrixXd eta=eta_cpp(beta,Xprev,nxprev,nclass);
  
  MatrixXd temp=cond_temp.array()*eta.array();
  
  VectorXd rowSumsTemp = temp.rowwise().sum();
  double likeli=rowSumsTemp.array().log().sum(); 
  
  return likeli;
}




// [[Rcpp::export]]
double log_like_test_cpp(const MatrixXd& cond_prob,
                         const MatrixXd& eta) {
  
  MatrixXd temp=cond_prob.array()*eta.array();
  
  VectorXd rowSumsTemp = temp.rowwise().sum();
  double likeli=rowSumsTemp.array().log().sum(); 
  
  return likeli;
}





// [[Rcpp::export]]
List conditional_prob_and_Pi_cpp(const List& alpha0,
                                 const List& alpha,
                                 const MatrixXd& Y_comp,
                                 const List& xcond,
                                 const VectorXi& nxcond,
                                 const int& npeop,
                                 const int& nitem,
                                 const int& nclass) {
  
  // List Pi_minus_1(nclass);
  std::vector<MatrixXd> Pi_minus_1(nclass);
  
  // onesVector可以寫在外面
  const MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  
  // MatrixXd Pi_j=MatrixXd::Zero(npeop, nxcond.sum());
  // MatrixXd Pi_minus_1_j=MatrixXd::Zero(npeop, nxcond.sum()-nitem);
  
  // 是否給起始值全為0
  MatrixXd cond_prob(npeop, nclass);
  
  
  Eigen::MatrixXd temp;
  
  for (int j = 0; j < nclass; ++j) {
    
    MatrixXd Pi_j(npeop, 0);
    MatrixXd Pi_minus_1_j(npeop, 0);
    
    for (int m = 0; m < nitem; ++m) {
      
      
      // Rf_isNull(xcond[m])
      if ( nxcond[m]==0 ) {
        
        MatrixXd alpha0_m = alpha0[m];
        
        Eigen::MatrixXd eigen_B(1, alpha0_m.cols() + 1);
        eigen_B.leftCols(alpha0_m.cols()) = alpha0_m.row(j);
        eigen_B.rightCols(1).setZero();
        
        temp = onesVector * eigen_B;
      
      }else{
        
        MatrixXd alpha0_m = alpha0[m];
        MatrixXd alpha_m = alpha[m];
        MatrixXd xcond_m = xcond[m];
        
        
        // 直接构造最终矩阵 eigen_B，避免所有中间临时矩阵
        MatrixXd eigen_B(alpha_m.rows() + 1, alpha_m.cols() + 1);
        
        // 填充上半部分：[alpha0[m].row(j); alpha[m]]
        eigen_B.topRows(1) = alpha0_m.row(j);      // 第一行 = alpha0[m].row(j)
        eigen_B.bottomRows(alpha_m.rows()) = alpha_m; // 剩余行 = alpha[m]
        
        // 最右侧补零列
        eigen_B.rightCols(1).setZero();
        
        
        // MatrixXd eigen_B = as<Map<MatrixXd> >( cbind(rbind(alpha0_m.row(j), alpha_m), 0.0) );
        temp = cbindEigen_matrix(onesVector,xcond_m) * eigen_B;
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
    Pi_minus_1[j]=Pi_minus_1_j;
    
    // temp_matrix= Pi_j.array()*Y_comp.array()+(1-Y_comp.array());
    // cond_prob.col(j)=temp_matrix.array().log().rowwise().sum().exp();
    
    cond_prob.col(j) = (Pi_j.array() * Y_comp.array() + (1 - Y_comp.array()))
             .log().rowwise().sum().exp();
    
  }
  
  return List::create(Named("cond_prob") = cond_prob,
                      Named("Pi_minus_1") = Pi_minus_1);
  
}



    
    
// [[Rcpp::export]]
std::vector<std::vector<MatrixXd>>  
  conditional_prob_and_Pi_for_omp_cpp( const std::vector<MatrixXd>& alpha0,
                                       const std::vector<MatrixXd>& alpha,
                                       const MatrixXd& Y_comp,
                                       const std::vector<MatrixXd> xcond_vec,
                                       const VectorXi& nxcond,
                                       const int& npeop,
                                       const int& nitem,
                                       const int& nclass) {
  
  
  // List Pi_minus_1(nclass);
  std::vector<MatrixXd> Pi_minus_1(nclass);
  
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
    
  }
  
  return {{cond_prob}, Pi_minus_1};
  
  // std::vector<std::vector<MatrixXd>> result(2);
  // result[0] = {cond_prob}; // 將 cond_prob 包裝成 vector
  // result[1] = Pi_minus_1;  // 直接賦值
  // return result;
  
}        

    
    
    
            
    
// [[Rcpp::export]]
List conditional_prob_and_Pi_omp_multi_flat(const std::vector<List>& alpha0_list,
                                            const std::vector<List>& alpha_list,
                                            const MatrixXd& Y_comp,
                                            const List& xcond,
                                            const VectorXi& nxcond,
                                            const int npeop,
                                            const int nitem,
                                            const int nclass,
                                            const int ncores = 4) {
  
  int nsets = alpha0_list.size();
  if (alpha_list.size() != nsets) stop("alpha0_list 與 alpha_list 長度不符");
  
  // 預處理 xcond（List -> vector<MatrixXd>）
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
  
  // 建立結果儲存空間
  std::vector<MatrixXd> cond_prob_list(nsets);
  std::vector<std::vector<MatrixXd>> Pi_minus_1_list(nsets);
  
#pragma omp parallel for num_threads(ncores)
  for (int idx = 0; idx < nsets; ++idx) {
    const auto& alpha0 = alpha0_list_std[idx];
    const auto& alpha = alpha_list_std[idx];
    
    std::vector<MatrixXd> Pi_minus_1(nclass);
    MatrixXd cond_prob = MatrixXd::Zero(npeop, nclass);
    MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
    
    for (int j = 0; j < nclass; ++j) {
      MatrixXd Pi_j(npeop, 0);
      MatrixXd Pi_minus_1_j(npeop, 0);
      
      for (int m = 0; m < nitem; ++m) {
        MatrixXd Pimkj;
        if (nxcond[m] == 0) {
          VectorXd alpha0_m_j = alpha0[m].row(j);
          MatrixXd matrixA(1, alpha0_m_j.size());
          matrixA.row(0) = alpha0_m_j;
          MatrixXd eigen_B = cbindEigen_matrix(matrixA, MatrixXd::Zero(matrixA.rows(), 1));
          
          MatrixXd temp = onesVector * eigen_B;
          temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
          MatrixXd inputMatrix = temp.array().exp();
          VectorXd inputVector = inputMatrix.rowwise().sum();
          Pimkj = inputMatrix.array().colwise() / inputVector.array();
        } else {
          VectorXd alpha0_m_j = alpha0[m].row(j);
          MatrixXd matrixA(1, alpha0_m_j.size());
          matrixA.row(0) = alpha0_m_j;
          MatrixXd A = cbindEigen_matrix(onesVector, xcond_vec[m]);
          MatrixXd B = rbindEigen_matrix(matrixA, alpha[m]);
          MatrixXd Bz = cbindEigen_matrix(B, MatrixXd::Zero(B.rows(), 1));
          MatrixXd temp = A * Bz;
          temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
          MatrixXd inputMatrix = temp.array().exp();
          VectorXd inputVector = inputMatrix.rowwise().sum();
          Pimkj = inputMatrix.array().colwise() / inputVector.array();
        }
        
        MatrixXd Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
        Pi_minus_1_j = cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
        Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
      }
      
      Pi_minus_1[j] = Pi_minus_1_j;
      MatrixXd temp_matrix = Pi_j.array() * Y_comp.array() + (1 - Y_comp.array());
      cond_prob.col(j) = (temp_matrix.array().log().rowwise().sum()).array().exp();
    }
    
    cond_prob_list[idx] = cond_prob;
    Pi_minus_1_list[idx] = Pi_minus_1;
  }
  
  return List::create(
    Named("cond_prob_list") = cond_prob_list,
    Named("Pi_minus_1_list") = Pi_minus_1_list
  );
}


    

    
// // [[Rcpp::export]]
// std::vector<std::pair<MatrixXd, std::vector<MatrixXd>>> conditional_prob_and_Pi_omp_alpha(
//     const std::vector<std::vector<MatrixXd>>& alpha0_list_std,
//     const std::vector<std::vector<MatrixXd>>& alpha_list_std,
//     const MatrixXd& Y_comp,
//     const std::vector<MatrixXd>& xcond_vec,
//     const VectorXi& nxcond,
//     const int npeop,
//     const int nitem,
//     const int nclass,
//     const int ncores = 4) {
//   
//   int nsets = alpha0_list_std.size();
//   if (alpha_list_std.size() != nsets)
//     stop("alpha0_list_std 與 alpha_list_std 長度不符");
//   
//   std::vector<std::pair<MatrixXd, std::vector<MatrixXd>>> results(nsets);
//   
// #pragma omp parallel for num_threads(ncores)
//   for (int idx = 0; idx < nsets; ++idx) {
//     const std::vector<MatrixXd>& alpha0 = alpha0_list_std[idx];
//     const std::vector<MatrixXd>& alpha = alpha_list_std[idx];
//     
//     std::vector<MatrixXd> Pi_minus_1(nclass);
//     MatrixXd cond_prob = MatrixXd::Zero(npeop, nclass);
//     MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
//     
//     for (int j = 0; j < nclass; ++j) {
//       MatrixXd Pi_j(npeop, 0);
//       MatrixXd Pi_minus_1_j(npeop, 0);
//       
//       for (int m = 0; m < nitem; ++m) {
//         MatrixXd Pimkj;
//         if (nxcond[m] == 0) {
//           VectorXd alpha0_m_j = alpha0[m].row(j);
//           MatrixXd matrixA(1, alpha0_m_j.size());
//           matrixA.row(0) = alpha0_m_j;
//           MatrixXd eigen_B = cbindEigen_matrix(matrixA, MatrixXd::Zero(1, 1));
//           MatrixXd temp = onesVector * eigen_B;
//           temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
//           MatrixXd inputMatrix = temp.array().exp();
//           VectorXd inputVector = inputMatrix.rowwise().sum();
//           Pimkj = inputMatrix.array().colwise() / inputVector.array();
//           
//         } else {
//           VectorXd alpha0_m_j = alpha0[m].row(j);
//           MatrixXd matrixA(1, alpha0_m_j.size());
//           matrixA.row(0) = alpha0_m_j;
//           MatrixXd A = cbindEigen_matrix(onesVector, xcond_vec[m]);
//           MatrixXd B = rbindEigen_matrix(matrixA, alpha[m]);
//           MatrixXd Bz = cbindEigen_matrix(B, MatrixXd::Zero(B.rows(), 1));
//           MatrixXd temp = A * Bz;
//           temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
//           MatrixXd inputMatrix = temp.array().exp();
//           VectorXd inputVector = inputMatrix.rowwise().sum();
//           Pimkj = inputMatrix.array().colwise() / inputVector.array();
//         }
//         
//         MatrixXd Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
//         Pi_minus_1_j = cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
//         Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
//       }
//       
//       Pi_minus_1[j] = Pi_minus_1_j;
//       MatrixXd temp_matrix = Pi_j.array() * Y_comp.array() + (1 - Y_comp.array());
//       cond_prob.col(j) = (temp_matrix.array().log().rowwise().sum()).array().exp();
//     }
//     
//     results[idx] = std::make_pair(cond_prob, Pi_minus_1);
//   }
//   
//   return results;
// }      
    
    
// [[Rcpp::export]]
double f2_rcpp(const VectorXd& alpha_vector,
               const MatrixXd& Y_comp,
               const List& xcond,
               const VectorXi& nlevels,
               const VectorXi& nxcond,
               const MatrixXd& h_p,
               const int& num_alpha0,
               const int& npeop,
               const int& nitem,
               const int& nclass) {
  
  
  List temp=tran_vec_to_list_cpp(alpha_vector,
                                 nlevels,
                                 nxcond,
                                 num_alpha0,
                                 nitem,
                                 nclass);
  
  
  List alpha0=temp["alpha0"];
  List alpha=temp["alpha"];
  
  MatrixXd cond_prob=cond_prob_cpp(alpha0,
                                   alpha,
                                   Y_comp,
                                   xcond,
                                   nxcond,
                                   npeop,
                                   nitem,
                                   nclass);
  
  MatrixXd f2_temp=h_p.array()*cond_prob.array().log();
  
  double f2=f2_temp.sum(); 
  // double f2=f2_temp.array().sum(); 
  
  return f2;
}




// [[Rcpp::export]]
MatrixXi calculateDirectSum(const int& nitem, 
                            const VectorXi& nlevels) {
  // 创建一个初始矩阵
  MatrixXi directSum = MatrixXi::Zero(nlevels.sum() - nitem, nlevels.sum() - nitem);
  
  // MatrixXd unitMatrix =MatrixXd::Identity(nlevels[0] - 1, nlevels[0] - 1);
  
  // directSum.block(0,0,nlevels[0] - 1,nlevels[0] - 1)=unitMatrix;
  // directSum.topLeftCorner(unitMatrix.rows(), unitMatrix.cols()) =unitMatrix
  
  int temp=0;
  for (int m = 0; m < nitem; ++m) {
    // 创建一个单位矩阵
    MatrixXi unitMatrix = MatrixXi::Ones(nlevels[m] - 1, nlevels[m] - 1);
    
    directSum.block(temp,temp,nlevels[m] - 1,nlevels[m] - 1) = unitMatrix;
    
    temp=temp+nlevels[m] - 1;
    // 计算直和
    // directSum.conservativeResize(directSum.rows() + unitMatrix.rows(), directSum.cols() + unitMatrix.cols());
    // directSum.bottomRightCorner(unitMatrix.rows(), unitMatrix.cols()) = unitMatrix;
  }
  
  return directSum;
}







// [[Rcpp::export]]
List Score_hess_alpha_cpp(const List& Pi_minus_1,
                          const MatrixXd& Y,
                          const MatrixXd& h_p,
                          const int& npeop,
                          const int& nitem,
                          const int& nclass,
                          const VectorXi& nlevels,
                          const VectorXi& nxcond,
                          const List& xcond,
                          const MatrixXd& direct,
                          const MatrixXd& diag_nclass,
                          const int& alpha_length) {
  
  
  
  VectorXd eigen_result_deri = VectorXd::Zero(alpha_length);
  MatrixXd eigen_result_hessian = MatrixXd::Zero(alpha_length,alpha_length);
  // 這樣後續會有錯
  // VectorXd eigen_result_deri(alpha_length);
  // MatrixXd eigen_result_hessian(alpha_length,alpha_length);
  
  // int cols_Vij_eigen = Y.cols();
  // int nrow_result = Y.cols() ;
  int ncol_result = Y.cols() * nclass;
  MatrixXd Vij_eigen= MatrixXd::Zero(Y.cols(),Y.cols());
  
  MatrixXd Pi_minus_1_j(nitem,Y.cols());
  MatrixXd S(nitem,Y.cols());
  
  MatrixXd eigen_der_alpha0(Y.cols(), ncol_result);
  
  for (int j = 0; j < nclass; ++j) {
    
    // RowVectorXd diag_nclass_j = diag_nclass.row(j);
    MatrixXd diag_nclass_j = diag_nclass.row(j);
    

    Pi_minus_1_j = Pi_minus_1[j];
    
    // MatrixXd S = Y - Pi_minus_1_j;
    S.noalias()= Y - Pi_minus_1_j;
    
    for (int i = 0; i < npeop; ++i) {
      
      // RowVectorXd S_i = S.row(i);
      
      // VectorXd S_i = S.row(i);
    
      
      
      // NumericVector muij = Pi_minus_1_j(i, _);
      
      VectorXd muij = Pi_minus_1_j.row(i);
      
      // NumericMatrix temp = calculateTemp(muij);
      
      
      MatrixXd temp = muij * muij.transpose();
      
      
      // MatrixXd temp_dir=temp.array()*direct.array();
      
      // int nn = muij.size();
      // MatrixXd mat=MatrixXd::Zero(nn, nn);
      // for (int ii = 0; ii < nn; ++ii) {
      //   mat(ii, ii) = muij[ii];
      // }
      
      // MatrixXd muij_as_diag=muij.asDiagonal();
      // MatrixXd Vij_eigen =muij_as_diag-temp_dir;
      
      
      Vij_eigen.setZero();    
      Vij_eigen.diagonal()=muij;  
      // MatrixXd Vij_eigen=muij.asDiagonal();
      temp=temp.array()*direct.array();
      Vij_eigen -=temp;
      
      
      // NumericMatrix Vij=wrap(Vij_eigen);
      
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(Vij_eigen);
      MatrixXd Vij_inver = cod.pseudoInverse();
      // MatrixXd Vij_inver = Vij_eigen.inverse();
      
      
      
      // NumericMatrix Vij_inver = ginv_rcpp(Vij);
      // NumericMatrix Vh_ij_inver_h = Vij_inver * h_p(i,j);
      // double element = h_p(i,j);
      // MatrixXd eigen_Vh_ij_inver_h = Vij_inver * element;
      MatrixXd eigen_Vh_ij_inver_h = Vij_inver * h_p(i,j);
      
      
      
      // MatrixXd deri_1 = eigen_Vh_ij_inver_h * S_i.transpose();
      
      // MatrixXd deri_1 = eigen_Vh_ij_inver_h * S_i;
      MatrixXd deri_1 = eigen_Vh_ij_inver_h * S.row(i).transpose();
      
      // int nrow_result = Vij_eigen.rows() * diag_nclass_j.rows();
      // int ncol_result = Vij_eigen.cols() * diag_nclass_j.cols();
      
      
      // MatrixXd eigen_der_alpha0(Y.cols(), ncol_result);
      // for (int k = 0; k < Vij_eigen.cols(); ++k) {
      //   
      //   MatrixXd Vij_eigen_temp=Vij_eigen.col(k)*diag_nclass_j;
      //   
      //   eigen_der_alpha0.block( 0, k*nclass, Vij_eigen.rows(), nclass )=Vij_eigen_temp;
      // }
      
      for (int k = 0; k < Y.cols(); ++k) {
        
        // MatrixXd Vij_eigen_temp=Vij_eigen.col(k)*diag_nclass_j;
        // 
        // eigen_der_alpha0.block( 0, k*nclass, Y.cols(), nclass )=Vij_eigen_temp;
        
        eigen_der_alpha0.block( 0, k*nclass, Y.cols(), nclass )=
          Vij_eigen.col(k)*diag_nclass_j;
      }
      
      
      // MatrixXd eigen_der_alpha0 = kroneckerProduct_matrix_eigen(Vij_eigen,eigen_diag_nclass_j);
      
      
  
      
      MatrixXd eigen_deri_mu_ij;
      if(nxcond.sum() == 0){
        eigen_deri_mu_ij = eigen_der_alpha0;
      }else{
        
        int u = -1;
        // NumericMatrix der_alpha(nitem, 0);
        // MatrixXd der_alpha(nlevels.sum()-nitem, 0);
        MatrixXd der_alpha(Y.cols(), 0);
        
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
            
            // MatrixXd Vij_u = Vij_eigen.col(u);
            // NumericVector Vij_u = Vij(_,u);
            
            
            // Map<MatrixXd> eigen_Vij_u(as<Map<MatrixXd> >(Vij_u));
            
            // MatrixXd b_temp=Vij_u*xcond_mi.transpose();
            MatrixXd b_temp=Vij_eigen.col(u)*xcond_mi.transpose();
            
            // MatrixXd b_temp=Vij_u*xcond_mi;
            
            der_alpha = cbindEigen_matrix(der_alpha, b_temp);
            
          }
        }
        
        eigen_deri_mu_ij = cbindEigen_matrix(eigen_der_alpha0, der_alpha);
      }
      

      
      
      // VectorXd deri = eigen_deri_mu_ij.transpose() * deri_1;
      // 
      // MatrixXd hessian = eigen_deri_mu_ij.transpose() * eigen_Vh_ij_inver_h*
      //   eigen_deri_mu_ij;
      // 
      // 
      // eigen_result_deri = eigen_result_deri + deri;
      // 
      // eigen_result_hessian=eigen_result_hessian+hessian;
      
      eigen_result_deri += eigen_deri_mu_ij.transpose() * deri_1;
      
      eigen_result_hessian += eigen_deri_mu_ij.transpose() * eigen_Vh_ij_inver_h*
        eigen_deri_mu_ij;
      
    }
  }
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(eigen_result_hessian);
  MatrixXd ginv = cod.pseudoInverse();
  VectorXd direction = ginv * eigen_result_deri;
  
  
  List result = List::create(
    Named("deri") = eigen_result_deri,
    Named("hessian") = eigen_result_hessian,
    Named("direction") = direction);
  
  return result;
}


    

    


// [[Rcpp::export]]
VectorXd Score_hess_alpha_Para_cpp(const List& Pi_minus_1,
                                   const MatrixXd& Y,
                                   const MatrixXd& h_p,
                                   const int& npeop,
                                   const int& nitem,
                                   const int& nclass,
                                   const VectorXi& nlevels,
                                   const VectorXi& nxcond,
                                   const List& xcond,
                                   const MatrixXd& direct,
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
  // int nrow_result = Y.cols() ;
  int ncol_result = Y.cols() * nclass;
  // MatrixXd Vij_eigen= MatrixXd::Zero(Y.cols(),Y.cols());
  
  MatrixXd Pi_minus_1_j(nitem,Y.cols());
  MatrixXd S(nitem,Y.cols());
  
  MatrixXd eigen_der_alpha0(Y.cols(), ncol_result);
  
  
  
  for (int j = 0; j < nclass; ++j) {
    
    // RowVectorXd diag_nclass_j = diag_nclass.row(j);
    MatrixXd diag_nclass_j = diag_nclass.row(j);
    
    
    Pi_minus_1_j = Pi_minus_1[j];
    
    // MatrixXd S = Y - Pi_minus_1_j;
    S.noalias()= Y - Pi_minus_1_j;
    
    // 使用自定义归约的直接结果变量 
    VectorXd eigen_result_deri_j = VectorXd::Zero(alpha_length);
    MatrixXd eigen_result_hessian_j = MatrixXd::Zero(alpha_length, alpha_length);
    
    
#pragma omp parallel for num_threads(ncores)           \
    schedule(dynamic)                                  \
      reduction(VectorAdd:eigen_result_deri_j)         \
      reduction(MatrixAdd:eigen_result_hessian_j)
    for (int i = 0; i < npeop; ++i) {
      
      // RowVectorXd S_i = S.row(i);
      
      // VectorXd S_i = S.row(i);
      
      
      
      // NumericVector muij = Pi_minus_1_j(i, _);
      
      VectorXd muij = Pi_minus_1_j.row(i);
      
      // NumericMatrix temp = calculateTemp(muij);
      
      
      MatrixXd temp = muij * muij.transpose();
      
      
      // MatrixXd temp_dir=temp.array()*direct.array();
      
      // int nn = muij.size();
      // MatrixXd mat=MatrixXd::Zero(nn, nn);
      // for (int ii = 0; ii < nn; ++ii) {
      //   mat(ii, ii) = muij[ii];
      // }
      
      // MatrixXd muij_as_diag=muij.asDiagonal();
      // MatrixXd Vij_eigen =muij_as_diag-temp_dir;
      
      
      // Vij_eigen.setZero();   
      MatrixXd Vij_eigen= MatrixXd::Zero(Y.cols(),Y.cols());
      Vij_eigen.diagonal()=muij;  
      // MatrixXd Vij_eigen=muij.asDiagonal();
      temp=temp.array()*direct.array();
      Vij_eigen -=temp;
      
      
      // NumericMatrix Vij=wrap(Vij_eigen);
      
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(Vij_eigen);
      MatrixXd Vij_inver = cod.pseudoInverse();
      // MatrixXd Vij_inver = Vij_eigen.inverse();
      
      
      
      // NumericMatrix Vij_inver = ginv_rcpp(Vij);
      // NumericMatrix Vh_ij_inver_h = Vij_inver * h_p(i,j);
      // double element = h_p(i,j);
      // MatrixXd eigen_Vh_ij_inver_h = Vij_inver * element;
      MatrixXd eigen_Vh_ij_inver_h = Vij_inver * h_p(i,j);
      
      
      
      // MatrixXd deri_1 = eigen_Vh_ij_inver_h * S_i.transpose();
      
      // MatrixXd deri_1 = eigen_Vh_ij_inver_h * S_i;
      MatrixXd deri_1 = eigen_Vh_ij_inver_h * S.row(i).transpose();
      
      // int nrow_result = Vij_eigen.rows() * diag_nclass_j.rows();
      // int ncol_result = Vij_eigen.cols() * diag_nclass_j.cols();
      
      
      // MatrixXd eigen_der_alpha0(Y.cols(), ncol_result);
      // for (int k = 0; k < Vij_eigen.cols(); ++k) {
      //   
      //   MatrixXd Vij_eigen_temp=Vij_eigen.col(k)*diag_nclass_j;
      //   
      //   eigen_der_alpha0.block( 0, k*nclass, Vij_eigen.rows(), nclass )=Vij_eigen_temp;
      // }
      
      // for (int k = 0; k < Y.cols(); ++k) {
      //   
      //   // MatrixXd Vij_eigen_temp=Vij_eigen.col(k)*diag_nclass_j;
      //   // 
      //   // eigen_der_alpha0.block( 0, k*nclass, Y.cols(), nclass )=Vij_eigen_temp;
      //   
      //   eigen_der_alpha0.block( 0, k*nclass, Y.cols(), nclass )=
      //     Vij_eigen.col(k)*diag_nclass_j;
      // }
      
      
      MatrixXd eigen_der_alpha0=KroneckerProduct(Vij_eigen,diag_nclass_j);
      // MatrixXd eigen_der_alpha0 = kroneckerProduct_matrix_eigen(Vij_eigen,eigen_diag_nclass_j);
      
      
      
      
      MatrixXd eigen_deri_mu_ij;
      if(nxcond.sum() == 0){
        eigen_deri_mu_ij = eigen_der_alpha0;
      }else{
        
        int u = -1;
        // NumericMatrix der_alpha(nitem, 0);
        // MatrixXd der_alpha(nlevels.sum()-nitem, 0);
        MatrixXd der_alpha(Y.cols(), 0);
        
        // VectorXd xcond_mi;
        
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
            
            // const NumericVector& Vij_u = Vij(_,u);
            // VectorXd Vij_u = Vij_eigen.col(u);
            
            // MatrixXd Vij_u = Vij_eigen.col(u);
            // NumericVector Vij_u = Vij(_,u);
            
            
            // Map<MatrixXd> eigen_Vij_u(as<Map<MatrixXd> >(Vij_u));
            
            // MatrixXd b_temp=Vij_u*xcond_mi.transpose();
            // MatrixXd b_temp=Vij_eigen.col(u)*xcond_mi.transpose();
            
            MatrixXd b_temp=Vij_eigen.col(u)*xcond_vec[m].row(i);
            // MatrixXd b_temp=Vij_u*xcond_mi;
            
            der_alpha = cbindEigen_matrix(der_alpha, b_temp);
            
          }
        }
        
        eigen_deri_mu_ij = cbindEigen_matrix(eigen_der_alpha0, der_alpha);
      }
      
      
      
      
      // VectorXd deri = eigen_deri_mu_ij.transpose() * deri_1;
      // 
      // MatrixXd hessian = eigen_deri_mu_ij.transpose() * eigen_Vh_ij_inver_h*
      //   eigen_deri_mu_ij;
      // 
      // 
      // eigen_result_deri = eigen_result_deri + deri;
      // 
      // eigen_result_hessian=eigen_result_hessian+hessian;
      
      eigen_result_deri_j += eigen_deri_mu_ij.transpose() * deri_1;
      
      eigen_result_hessian_j += eigen_deri_mu_ij.transpose() * 
        eigen_Vh_ij_inver_h*
        eigen_deri_mu_ij;

    }
    
    
    eigen_result_deri += eigen_result_deri_j;
    eigen_result_hessian += eigen_result_hessian_j;
    
  }
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(eigen_result_hessian);
  MatrixXd ginv = cod.pseudoInverse();
  
  
  VectorXd direction = ginv * eigen_result_deri;
  
  
  return direction;
  
  // List result = List::create(
  //   Named("deri") = eigen_result_deri,
  //   Named("hessian") = eigen_result_hessian,
  //   Named("direction") = direction);
  // 
  // return result;
}
    

  
    


      
    
    
// [[Rcpp::export]]
VectorXd Score_hess_alpha_for_omp_cpp(const std::vector<MatrixXd> Pi_minus_1,
                                      const MatrixXd& Y,
                                      const MatrixXd& h_p,
                                      const int& npeop,
                                      const int& nitem,
                                      const int& nclass,
                                      const VectorXi& nlevels,
                                      const VectorXi& nxcond,
                                      const std::vector<MatrixXd> xcond_vec,
                                      const MatrixXd& direct,
                                      const MatrixXd& diag_nclass,
                                      const int& alpha_length) {
  
  VectorXd eigen_result_deri = VectorXd::Zero(alpha_length);
  MatrixXd eigen_result_hessian = MatrixXd::Zero(alpha_length,alpha_length);
  // 這樣後續會有錯
  // VectorXd eigen_result_deri(alpha_length);
  // MatrixXd eigen_result_hessian(alpha_length,alpha_length);
  
  // int cols_Vij_eigen = Y.cols();
  int y_cols  = Y.cols();
  // int ncol_result = Y.cols() * nclass;
  
  MatrixXd Vij_eigen= MatrixXd::Zero(y_cols ,y_cols );
  MatrixXd Vij_inver(y_cols ,y_cols );
    
  MatrixXd Pi_minus_1_j(nitem,y_cols );
  MatrixXd S(nitem,y_cols );
  
  VectorXd deri_1(y_cols);
  VectorXd muij(y_cols);
  
  MatrixXd temp(y_cols ,y_cols );
  
  MatrixXd eigen_der_alpha0(y_cols, y_cols  * nclass);
  // MatrixXd xcond_m;
  for (int j = 0; j < nclass; ++j) {
    
    // RowVectorXd diag_nclass_j = diag_nclass.row(j);
    // const auto& diag_nclass_j = diag_nclass.row(j);
    const MatrixXd& diag_nclass_j = diag_nclass.row(j);
    // MatrixXd Pi_minus_1_j = Pi_minus_1[j];
    
    Pi_minus_1_j.noalias() = Pi_minus_1[j];
    
    // MatrixXd S = Y - Pi_minus_1_j;
    S.noalias()= Y - Pi_minus_1_j;
    
    for (int i = 0; i < npeop; ++i) {
      // VectorXd S_i = S.row(i);
      
      // VectorXd muij = Pi_minus_1_j.row(i);
      
      muij .noalias()= Pi_minus_1_j.row(i);
      
      temp.noalias() = muij * muij.transpose();
      
      // temp.noalias() = Pi_minus_1_j.row(i).transpose() * 
      //   Pi_minus_1_j.row(i);

      // MatrixXd temp_dir=temp.array()*direct.array();
      
      // int nn = muij.size();
      // MatrixXd mat=MatrixXd::Zero(nn, nn);
      // for (int ii = 0; ii < nn; ++ii) {
      //   mat(ii, ii) = muij[ii];
      // }
      
      // MatrixXd muij_as_diag=muij.asDiagonal();
      // 
      // MatrixXd Vij_eigen =muij_as_diag-temp_dir;
      
      Vij_eigen=muij.asDiagonal();
      // Vij_eigen.setZero();    
      // Vij_eigen.diagonal()=Pi_minus_1_j.row(i);  
      // MatrixXd Vij_eigen=muij.asDiagonal();
      
      
      // temp=temp.array()*direct.array();
      // Vij_eigen.noalias() -=temp;
      
      Vij_eigen.noalias() -=temp.cwiseProduct(direct);
      
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(Vij_eigen);
      // MatrixXd Vij_inver = cod.pseudoInverse();
      Vij_inver.noalias() = cod.pseudoInverse();
      // MatrixXd Vij_inver = Vij_eigen.inverse();
      
      
      
      // NumericMatrix Vij_inver = ginv_rcpp(Vij);
      // NumericMatrix Vh_ij_inver_h = Vij_inver * h_p(i,j);
      // double element = h_p(i,j);
      // MatrixXd eigen_Vh_ij_inver_h = Vij_inver * element;
      // MatrixXd eigen_Vh_ij_inver_h = Vij_inver * h_p(i,j);
      
      
      
      // MatrixXd deri_1 = eigen_Vh_ij_inver_h * S_i.transpose();
      
      // MatrixXd deri_1 = eigen_Vh_ij_inver_h * S_i;
      
      // MatrixXd deri_1 = eigen_Vh_ij_inver_h * S.row(i).transpose(); 
      
      // deri_1.noalias() = eigen_Vh_ij_inver_h * S.row(i).transpose(); 
      deri_1.noalias() =  h_p(i,j)*Vij_inver * S.row(i).transpose(); 
      
      // MatrixXd eigen_der_alpha0(Y.cols(), ncol_result);
      // for (int k = 0; k < Vij_eigen.cols(); ++k) {
      //   
      //   MatrixXd Vij_eigen_temp=Vij_eigen.col(k)*diag_nclass_j;
      //   
      //   eigen_der_alpha0.block( 0, k*nclass, Vij_eigen.rows(), nclass )=Vij_eigen_temp;
      // }
      
      // for (int k = 0; k < y_cols; ++k) {
      //   
      //   // MatrixXd Vij_eigen_temp=Vij_eigen.col(k)*diag_nclass_j;
      //   // 
      //   // eigen_der_alpha0.block( 0, k*nclass, Y.cols(), nclass )=Vij_eigen_temp;
      //   
      //   eigen_der_alpha0.middleCols( k*nclass, nclass ).noalias() =
      //     Vij_eigen.col(k)*diag_nclass_j;
      // }
      eigen_der_alpha0.noalias()=KroneckerProduct(Vij_eigen,diag_nclass_j);
      
      // MatrixXd eigen_der_alpha0 = kroneckerProduct_matrix_eigen(Vij_eigen,eigen_diag_nclass_j);
      
      
      MatrixXd eigen_deri_mu_ij=eigen_der_alpha0;
      
      if(nxcond.sum() > 0){
      
        int u = -1;
        MatrixXd der_alpha(y_cols, 0);
        
        // VectorXd xcond_mi;
        for(int m = 0; m < nitem; ++m) {
          
          // Rf_isNull(xcond[m])
          // if (nxcond[m]==0) {
          //   // 如果条件满足，跳过当前循环
          //   continue;
          // }
          
          // const NumericMatrix& xcond_m_ref = xcond[m]; // 使用引用
          
          // if(nxcond[m]!=0){
          //   // MatrixXd xcond_m = xcond_vec[m];
          //   // xcond_mi= xcond_m.row(i);
          // 
          //   // xcond_mi= xcond_vec[m].row(i);
          //   // MatrixXd xcond_m= xcond_vec[m];
          //   xcond_m= xcond_vec[m];
          // }
          
          
          for(int k = 0; k < (nlevels[m] - 1); ++k) {
            
            ++u;
            
            if (nxcond[m]==0) {
              // 如果条件满足，跳过当前循环
              continue;
            }
            
            // const NumericVector& Vij_u = Vij(_,u);
            // VectorXd Vij_u = Vij_eigen.col(u);
            
            // MatrixXd Vij_u = Vij_eigen.col(u);
            // NumericVector Vij_u = Vij(_,u);
            
            
            // Map<MatrixXd> eigen_Vij_u(as<Map<MatrixXd> >(Vij_u));
            
            // MatrixXd b_temp=Vij_u*xcond_mi.transpose();
            
            // MatrixXd b_temp=Vij_eigen.col(u)*xcond_mi.transpose();
            
            MatrixXd b_temp=Vij_eigen.col(u)*xcond_vec[m].row(i);
            // MatrixXd b_temp=Vij_eigen.col(u)*xcond_m.row(i);
            
            // MatrixXd b_temp=Vij_u*xcond_mi;
            
            der_alpha = cbindEigen_matrix(der_alpha, b_temp);
            // der_alpha.conservativeResize(y_cols, der_alpha.cols() + b_temp.cols());
            // der_alpha.rightCols(b_temp.cols()) = b_temp;
            
          }
        }
        
        eigen_deri_mu_ij = cbindEigen_matrix(eigen_der_alpha0, der_alpha);
      }
      
      
      
      
      // VectorXd deri = eigen_deri_mu_ij.transpose() * deri_1;
      
      // MatrixXd hessian = eigen_deri_mu_ij.transpose() * eigen_Vh_ij_inver_h*
      //   eigen_deri_mu_ij;
      
      
      // eigen_result_deri = eigen_result_deri + deri;
      // eigen_result_hessian=eigen_result_hessian+hessian;
      eigen_result_deri.noalias() += eigen_deri_mu_ij.transpose() * deri_1;
      
      // eigen_result_hessian += eigen_deri_mu_ij.transpose() * eigen_Vh_ij_inver_h*
      //   eigen_deri_mu_ij;
      eigen_result_hessian.noalias() +=
        h_p(i,j)* eigen_deri_mu_ij.transpose() * Vij_inver *eigen_deri_mu_ij;
    }
  }
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(eigen_result_hessian);
  // MatrixXd ginv = cod.pseudoInverse();
  // VectorXd direction = ginv * eigen_result_deri;
  
  VectorXd direction = cod.pseudoInverse() * eigen_result_deri;
  
  // List result = List::create(
  //   Named("deri") = eigen_result_deri,
  //   Named("hessian") = eigen_result_hessian,
  //   Named("direction") = direction);
  
  return direction;
}    


// [[Rcpp::export]]
List Score_hess_alpha_test_cpp(const List& Pi_minus_1,
                               const MatrixXd& Y,
                               const MatrixXd& h_p,
                               const int& npeop,
                               const int& nitem,
                               const int& nclass,
                               const VectorXi& nlevels,
                               const VectorXi& nxcond,
                               const List& xcond,
                               const MatrixXd& direct,
                               const MatrixXd& diag_nclass,
                               const int& alpha_length) {
  
  // // 平行化過程中存取List會出問題
  std::vector<MatrixXd> xcond_vec(xcond.size());
  for (int m = 0; m < xcond.size(); ++m) {
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
  int ncol_result = Y.cols() * nclass;



  for (int j = 0; j < nclass; ++j) {

    // RowVectorXd diag_nclass_j = diag_nclass.row(j);
    MatrixXd diag_nclass_j = diag_nclass.row(j);

    MatrixXd Pi_minus_1_j = Pi_minus_1[j];
    MatrixXd S = Y - Pi_minus_1_j;
#pragma omp parallel
{
  VectorXd eigen_result_deri_local = VectorXd::Zero(alpha_length);
  MatrixXd eigen_result_hessian_local = MatrixXd::Zero(alpha_length,alpha_length);
#pragma omp parallel for  reduction(VectorAdd : eigen_result_deri_local) reduction(MatrixAdd : eigen_result_hessian_local)  schedule(dynamic)
    for (int i = 0; i < npeop; ++i) {

      // RowVectorXd S_i = S.row(i);

      VectorXd S_i = S.row(i);
      // MatrixXd S_i=S(i,_);


      // NumericVector muij = Pi_minus_1_j(i, _);

      VectorXd muij = Pi_minus_1_j.row(i);

      // NumericMatrix temp = calculateTemp(muij);


      MatrixXd temp = muij * muij.transpose();


      MatrixXd temp_dir=temp.array()*direct.array();

      // int nn = muij.size();
      // MatrixXd mat=MatrixXd::Zero(nn, nn);
      // for (int ii = 0; ii < nn; ++ii) {
      //   mat(ii, ii) = muij[ii];
      // }

      MatrixXd muij_as_diag=muij.asDiagonal();

      MatrixXd Vij_eigen =muij_as_diag-temp_dir;



      // NumericMatrix Vij=wrap(Vij_eigen);


      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(Vij_eigen);
      MatrixXd Vij_inver = cod.pseudoInverse();
      // MatrixXd Vij_inver = Vij_eigen.inverse();



      // NumericMatrix Vij_inver = ginv_rcpp(Vij);
      // NumericMatrix Vh_ij_inver_h = Vij_inver * h_p(i,j);
      // double element = h_p(i,j);
      // MatrixXd eigen_Vh_ij_inver_h = Vij_inver * element;
      MatrixXd eigen_Vh_ij_inver_h = Vij_inver * h_p(i,j);



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
            // MatrixXd xcond_m = xcond[m];
            // xcond_mi= xcond_m.row(i);
            // // xcond_mi= xcond_m.row(i);
            
            xcond_mi = xcond_vec[m].row(i);
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

      MatrixXd hessian = eigen_deri_mu_ij.transpose() * eigen_Vh_ij_inver_h*
        eigen_deri_mu_ij;


      eigen_result_deri_local = eigen_result_deri_local + deri;

      eigen_result_hessian_local=eigen_result_hessian_local+hessian;

    }
    
    eigen_result_deri=eigen_result_deri+eigen_result_deri_local;
    eigen_result_hessian=eigen_result_hessian+eigen_result_hessian_local;
}

  }

  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(eigen_result_hessian);
  MatrixXd ginv = cod.pseudoInverse();
  VectorXd direction = ginv * eigen_result_deri;


  List result = List::create(
    Named("deri") = eigen_result_deri,
    Named("hessian") = eigen_result_hessian,
    Named("direction") = direction);

  return result;
}



    
// [[Rcpp::export]]
List Score_hess_alpha_test_1_cpp(const List& Pi_minus_1,
                                 const MatrixXd& Y,
                                 const MatrixXd& h_p,
                                 const int& npeop,
                                 const int& nitem,
                                 const int& nclass,
                                 const VectorXi& nlevels,
                                 const VectorXi& nxcond,
                                 const List& xcond,
                                 const MatrixXd& direct,
                                 const MatrixXd& diag_nclass,
                                 const int& alpha_length) {
  
  // // 平行化過程中存取List會出問題
  std::vector<MatrixXd> xcond_vec(xcond.size());
  for (int m = 0; m < xcond.size(); ++m) {
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
  int ncol_result = Y.cols() * nclass;
  

  
#pragma omp parallel for collapse(2) reduction(VectorAdd : eigen_result_deri) reduction(MatrixAdd : eigen_result_hessian)  schedule(dynamic)
  for (int i = 0; i < npeop; ++i) {
for (int j = 0; j < nclass; ++j) {
  
  // RowVectorXd diag_nclass_j = diag_nclass.row(j);
  MatrixXd diag_nclass_j = diag_nclass.row(j);
  
  MatrixXd Pi_minus_1_j = Pi_minus_1[j];
  MatrixXd S = Y - Pi_minus_1_j;
// RowVectorXd S_i = S.row(i);

VectorXd S_i = S.row(i);
// MatrixXd S_i=S(i,_);


// NumericVector muij = Pi_minus_1_j(i, _);

VectorXd muij = Pi_minus_1_j.row(i);

// NumericMatrix temp = calculateTemp(muij);


MatrixXd temp = muij * muij.transpose();


MatrixXd temp_dir=temp.array()*direct.array();

// int nn = muij.size();
// MatrixXd mat=MatrixXd::Zero(nn, nn);
// for (int ii = 0; ii < nn; ++ii) {
//   mat(ii, ii) = muij[ii];
// }

MatrixXd muij_as_diag=muij.asDiagonal();

MatrixXd Vij_eigen =muij_as_diag-temp_dir;



// NumericMatrix Vij=wrap(Vij_eigen);


// 計算ginv
CompleteOrthogonalDecomposition<MatrixXd> cod(Vij_eigen);
MatrixXd Vij_inver = cod.pseudoInverse();




// MatrixXd Vij_inver = Vij_eigen.inverse();



// NumericMatrix Vij_inver = ginv_rcpp(Vij);
// NumericMatrix Vh_ij_inver_h = Vij_inver * h_p(i,j);
// double element = h_p(i,j);
// MatrixXd eigen_Vh_ij_inver_h = Vij_inver * element;
MatrixXd eigen_Vh_ij_inver_h = Vij_inver * h_p(i,j);



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
      // MatrixXd xcond_m = xcond[m];
      // xcond_mi= xcond_m.row(i);
      // // xcond_mi= xcond_m.row(i);
      
      xcond_mi = xcond_vec[m].row(i);
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

MatrixXd hessian = eigen_deri_mu_ij.transpose() * eigen_Vh_ij_inver_h*
  eigen_deri_mu_ij;


eigen_result_deri = eigen_result_deri + deri;

eigen_result_hessian=eigen_result_hessian+hessian;

}




  }
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(eigen_result_hessian);
  MatrixXd ginv = cod.pseudoInverse();
  VectorXd direction = ginv * eigen_result_deri;
  
  
  List result = List::create(
    Named("deri") = eigen_result_deri,
    Named("hessian") = eigen_result_hessian,
    Named("direction") = direction);
  
  return result;
}


    
    
    
    
    
    
    
// [[Rcpp::export]]
List max_alpha_bisection_cpp(const VectorXd& alpha_vector, 
                             const MatrixXd& Y,
                             const MatrixXd& Y_comp,
                             const List& Pi_minus_1,
                             const MatrixXd& h_p,
                             const VectorXi& nlevels,
                             const VectorXi& nxcond,
                             const List& xcond,
                             const MatrixXd& direct,
                             const MatrixXd& diag_nclass,
                             const int& alpha_length,
                             const int& num_alpha0,
                             const int& npeop,
                             const int& nitem,
                             const int& nclass,
                             double& c,
                             const double& err) {
 



  
  
  List temp=Score_hess_alpha_cpp(Pi_minus_1,
                                 Y,
                                 h_p,
                                 npeop,
                                 nitem,
                                 nclass,
                                 nlevels,
                                 nxcond,
                                 xcond,
                                 direct,
                                 diag_nclass,
                                 alpha_length);
  


  VectorXd direction=temp["direction"];

  
  // NumericVector temp_aa(20);
  // NumericVector temp_cc(20);
  
  double a = 0.0;  
  int iter = 0;
  for( iter = 0; iter < 1000; ++iter) {
    double temp_a = a;
    double temp_c = c;
    
    
    double temp_b = (a + c) / 2.0;
    
    
    double temp_a1 = f2_rcpp(alpha_vector + temp_b * direction,
                             Y_comp,xcond,nlevels,
                             nxcond,h_p,
                             num_alpha0,npeop,
                             nitem,nclass);
    
    double temp_a2 = f2_rcpp(alpha_vector + (temp_b + 0.02) * direction,
                             Y_comp,xcond,nlevels,
                             nxcond,h_p,
                             num_alpha0,npeop,
                             nitem,nclass);
    
    // double temp_a1 = as<double>(f2(alpha_vector + temp_b * direction));
    // double temp_a2 = as<double>(f2(alpha_vector + (temp_b + 0.05) * direction));
    
    if(temp_a1 < temp_a2) {
      a = temp_b - 0.02;
      c = c;
      
      // temp_aa[iter]=a;
      // temp_cc[iter]=c;
      
      if( abs(temp_a - a) < 0.001 ) break;
    } else {
      a = a;
      c = temp_b + 0.02;
      
      // temp_aa[iter]=a;
      // temp_cc[iter]=c;
      
      if( abs(temp_c - c) < 0.001 ) break;
    }
    
    // if( std::max( abs(temp_a - a),abs(temp_c - c) ) < 0.001 ) break;
  }
  
  
  
  int len = static_cast<int>((c - a) / err) + 1;
  NumericVector likeli(len+3);
  NumericVector ak(len+3);
  List alpha_vector_list(len+3);
  
  
  
  ak[0]=0;
  alpha_vector_list[0]=  alpha_vector;
  likeli[0] =   f2_rcpp(alpha_vector,
                        Y_comp,xcond,nlevels,
                        nxcond,h_p,
                        num_alpha0,npeop,
                        nitem,nclass);
  
  int max_idx=0;
  for(int i = 0; i < len; ++i) {
    ak[i+1]=a + i * err;
    alpha_vector_list[i+1]=alpha_vector + (a + i * err) * direction;
    likeli[i+1] = f2_rcpp(alpha_vector_list[i+1],
                          Y_comp,xcond,nlevels,
                          nxcond,h_p,
                          num_alpha0,npeop,
                          nitem,nclass);
    
    if(likeli[i+1]>=likeli[i]){
      max_idx=i+1;
    }
    
    
  }
  

  ak[len+1]=c;

  alpha_vector_list[len+1]=alpha_vector + c * direction;

  likeli[len+1] = f2_rcpp(alpha_vector_list[len+1],
                          Y_comp,xcond,nlevels,
                          nxcond,h_p,
                          num_alpha0,npeop,
                          nitem,nclass);

  if(likeli[len+1]>=likeli[max_idx]){
    max_idx=len+1;
  }
  
  
  ak[len+2]=1;
  alpha_vector_list[len+2]=alpha_vector+direction;
  likeli[len+2] = f2_rcpp(alpha_vector_list[len+2],
                          Y_comp,xcond,nlevels,
                          nxcond,h_p,
                          num_alpha0,npeop,
                          nitem,nclass);
  
  if(likeli[len+2]>=likeli[max_idx]){
    max_idx=len+2;
  }

  
  
  // double max_tempp = Rcpp::max(likeli);
  // NumericVector alpha_vector_p(alpha_vector.size());
  // 
  // int max_idx = Rcpp::which_max(likeli);
  // 
  // if(max_tempp < a2) {
  //   alpha_vector_p = alpha_vector;
  // } else {
  //   
  //   alpha_vector_p = alpha_vector + ak[max_idx] * direction;
  // }
  
  
  
  // int max_idx = Rcpp::which_max(likeli);
  
  
  // NumericVector alpha_vector_p(alpha_vector.size());
  VectorXd alpha_vector_p=alpha_vector_list[max_idx];
  
  // if(max_tempp >= a2) {
  //   alpha_vector_p = alpha_vector + ak[max_idx] * direction;
  // } else{
  //   alpha_vector_p = alpha_vector;
  // }
  
  // NumericVector alpha_vector_p=alpha_vector;
  double ak_temp=ak[max_idx];
  
  // 创建一个 List 对象包含多个值
  List result = List::create(
    Named("a") = a,
    Named("c") = c,
    Named("likeli") = likeli,
    Named("len") = len,
    Named("ak") = ak,
    Named("ak_temp") = ak_temp,
    Named("alpha_vector_p") = alpha_vector_p,
    Named("iter") = iter);
  
  
  return result;
}



// [[Rcpp::export]]
// 作為rlca alpha最大值的修改
// 仿max_beta_1
List max_alpha_4_cpp(VectorXd& alpha_vector,
                     List& alpha0,
                     List& alpha,
                     const MatrixXd& Y,
                     const MatrixXd& Y_comp,
                     List& Pi_minus_1,
                     MatrixXd& cond_prob,
                     const MatrixXd& h_p,
                     const VectorXi& nlevels,
                     const VectorXi& nxcond,
                     const List& xcond,
                     const MatrixXd& direct,
                     const MatrixXd& diag_nclass,
                     const int& alpha_length,
                     const int& num_alpha0,
                     const int& npeop,
                     const int& nitem,
                     const int& nclass,
                     const int& maxiter,
                     const double& step_length,
                     const double& tol) {
  
  
  // double llik= f2_rcpp(alpha_vector,
  //                      Y_comp,xcond,nlevels,
  //                      nxcond,h_p,
  //                      num_alpha0,npeop,
  //                      nitem,nclass);
  
  double llik=( h_p.array()*cond_prob.array().log() ).sum();
  
  List temp=Score_hess_alpha_cpp(Pi_minus_1,
                                 Y,
                                 h_p,
                                 npeop,
                                 nitem,
                                 nclass,
                                 nlevels,
                                 nxcond,
                                 xcond,
                                 direct,
                                 diag_nclass,
                                 alpha_length);
  
  VectorXd direction=temp["direction"];
  
  
  // List temp_1=tran_vec_to_list_cpp(alpha_vector,
  //                                  nlevels,
  //                                  nxcond,
  //                                  num_alpha0,
  //                                  nitem,
  //                                  nclass);
  // List alpha0=temp_1["alpha0"];
  // List alpha=temp_1["alpha"];
  
  
  
  double llik_p;
  
  
  double a=1.0;
  // List alpha0;
  // List al;
  // 
  List temp_1;
  
  VectorXd alpha_vector_p;
  List alpha0_p;
  List alpha_p;
  List cond_prob_Pi_minus_1;
  // List Pi_minus_1_p;
  MatrixXd cond_prob_p;
  
  int iter = 0;
  for( iter = 0; iter < maxiter; ++iter) {
    
    alpha_vector_p=alpha_vector+a*direction;
    
    temp_1=tran_vec_to_list_cpp(alpha_vector_p,
                                nlevels,
                                nxcond,
                                num_alpha0,
                                nitem,
                                nclass);
    
    
    alpha0_p=temp_1["alpha0"];
    alpha_p=temp_1["alpha"];
    
    
    cond_prob_Pi_minus_1=conditional_prob_and_Pi_cpp(alpha0_p,
                                                     alpha_p,
                                                     Y_comp,
                                                     xcond,
                                                     nxcond,
                                                     npeop,
                                                     nitem,
                                                     nclass);
    cond_prob_p=cond_prob_Pi_minus_1["cond_prob"];
    // Pi_minus_1_p=cond_prob_Pi_minus_1["Pi_minus_1"];
    
    llik_p=( h_p.array()*cond_prob_p.array().log() ).sum(); 
    
    
    // Rcpp::Rcout << " llik_p = " << llik_p << ", llik = " << llik <<direction<< std::endl;
    
    
    if(llik_p>llik){
      alpha_vector=alpha_vector_p;
      
      llik=llik_p;
      Pi_minus_1=cond_prob_Pi_minus_1["Pi_minus_1"];;
      cond_prob=cond_prob_p;
      
      alpha0=alpha0_p;
      alpha=alpha_p;
      
      if((a*direction).cwiseAbs().maxCoeff()<tol) break;
      
      // Pi_minus_1=Pi_minus_1_cpp(alpha_vector,
      //                           xcond,
      //                           nlevels,
      //                           nxcond,
      //                           num_alpha0,
      //                           npeop,
      //                           nitem,
      //                           nclass);
      
      temp=Score_hess_alpha_cpp(Pi_minus_1,
                                Y,
                                h_p,
                                npeop,
                                nitem,
                                nclass,
                                nlevels,
                                nxcond,
                                xcond,
                                direct,
                                diag_nclass,
                                alpha_length);
      direction=temp["direction"];
    }else{
      
      a=a*step_length;
      if((a*direction).cwiseAbs().maxCoeff()<tol) break;
    }
    
    // if((a*direction).cwiseAbs().maxCoeff()<tol) break;
  }
  
  
  
  
  
  List result = List::create(
    Named("alpha_vector_p") = alpha_vector,
    Named("alpha0") = alpha0,
    Named("alpha") = alpha,
    Named("Pi_minus_1") = Pi_minus_1,
    Named("cond_prob") = cond_prob,
    Named("direction") = direction,
    Named("a") = a,
    Named("llik") = llik,
    Named("iter") = iter);
  
  
  return result;
}




// [[Rcpp::export]]
// 作為rlca alpha最大值的修改
// 仿max_beta_1
List max_alpha_5_cpp(VectorXd& alpha_vector,
                     List& alpha0,
                     List& alpha,
                     const MatrixXd& Y,
                     const MatrixXd& Y_comp,
                     List& Pi_minus_1,
                     MatrixXd& cond_prob,
                     const MatrixXd& h_p,
                     const VectorXi& nlevels,
                     const VectorXi& nxcond,
                     const List& xcond,
                     const MatrixXd& direct,
                     const MatrixXd& diag_nclass,
                     const int& alpha_length,
                     const int& num_alpha0,
                     const int& npeop,
                     const int& nitem,
                     const int& nclass,
                     const int& maxiter,
                     const double& step_length,
                     const double& tol,
                     const int& it) {
  
  
  // double llik= f2_rcpp(alpha_vector,
  //                      Y_comp,xcond,nlevels,
  //                      nxcond,h_p,
  //                      num_alpha0,npeop,
  //                      nitem,nclass);
  int count=0;
  double llik=( h_p.array()*cond_prob.array().log() ).sum();
  
  List temp=Score_hess_alpha_cpp(Pi_minus_1,
                                 Y,
                                 h_p,
                                 npeop,
                                 nitem,
                                 nclass,
                                 nlevels,
                                 nxcond,
                                 xcond,
                                 direct,
                                 diag_nclass,
                                 alpha_length);
  
  VectorXd direction=temp["direction"];
  
  
  // List temp_1=tran_vec_to_list_cpp(alpha_vector,
  //                                  nlevels,
  //                                  nxcond,
  //                                  num_alpha0,
  //                                  nitem,
  //                                  nclass);
  // List alpha0=temp_1["alpha0"];
  // List alpha=temp_1["alpha"];
  
  
  
  double llik_p;
  
  
  double a=1.0;
  // List alpha0;
  // List al;
  // 
  List temp_1;
  
  VectorXd alpha_vector_p;
  List alpha0_p;
  List alpha_p;
  List cond_prob_Pi_minus_1;
  // List Pi_minus_1_p;
  MatrixXd cond_prob_p;
  
  int iter = 0;
  for( iter = 0; iter < maxiter; ++iter) {
    
    alpha_vector_p=alpha_vector+a*direction;
    
    temp_1=tran_vec_to_list_cpp(alpha_vector_p,
                                nlevels,
                                nxcond,
                                num_alpha0,
                                nitem,
                                nclass);
    
    
    alpha0_p=temp_1["alpha0"];
    alpha_p=temp_1["alpha"];
    
    
    cond_prob_Pi_minus_1=conditional_prob_and_Pi_cpp(alpha0_p,
                                                     alpha_p,
                                                     Y_comp,
                                                     xcond,
                                                     nxcond,
                                                     npeop,
                                                     nitem,
                                                     nclass);
    cond_prob_p=cond_prob_Pi_minus_1["cond_prob"];
    // Pi_minus_1_p=cond_prob_Pi_minus_1["Pi_minus_1"];
    
    llik_p=( h_p.array()*cond_prob_p.array().log() ).sum(); 
    
    
    // Rcpp::Rcout << " llik_p = " << llik_p << ", llik = " << llik <<direction<< std::endl;
    
    
    if(llik_p>llik){
      ++count;
      alpha_vector=alpha_vector_p;
      
      llik=llik_p;
      Pi_minus_1=cond_prob_Pi_minus_1["Pi_minus_1"];;
      cond_prob=cond_prob_p;
      
      alpha0=alpha0_p;
      alpha=alpha_p;
      
      if(a*direction.cwiseAbs().maxCoeff()<tol|count==it) break;
      
      // Pi_minus_1=Pi_minus_1_cpp(alpha_vector,
      //                           xcond,
      //                           nlevels,
      //                           nxcond,
      //                           num_alpha0,
      //                           npeop,
      //                           nitem,
      //                           nclass);
      
      temp=Score_hess_alpha_cpp(Pi_minus_1,
                                Y,
                                h_p,
                                npeop,
                                nitem,
                                nclass,
                                nlevels,
                                nxcond,
                                xcond,
                                direct,
                                diag_nclass,
                                alpha_length);
      direction=temp["direction"];
    }else{
      a*=step_length;
      // a=a*step_length;
      if(a*direction.cwiseAbs().maxCoeff()<tol) break;
    }
    
    // if((a*direction).cwiseAbs().maxCoeff()<tol) break;
  }
  
  
  
  
  
  List result = List::create(
    Named("alpha_vector_p") = alpha_vector,
    Named("alpha0") = alpha0,
    Named("alpha") = alpha,
    Named("Pi_minus_1") = Pi_minus_1,
    Named("cond_prob") = cond_prob,
    Named("direction") = direction,
    Named("a") = a,
    Named("llik") = llik,
    Named("iter") = iter);
  
  
  return result;
}

    


    
    

// [[Rcpp::export]]
// 作為rlca alpha最大值的修改
// 仿max_beta_1
List max_alpha_5_Para_cpp(VectorXd& alpha_vector,
                          List& alpha0,
                          List& alpha,
                          const MatrixXd& Y,
                          const MatrixXd& Y_comp,
                          List& Pi_minus_1,
                          MatrixXd& cond_prob,
                          const MatrixXd& h_p,
                          const VectorXi& nlevels,
                          const VectorXi& nxcond,
                          const List& xcond,
                          const MatrixXd& direct,
                          const MatrixXd& diag_nclass,
                          const int& alpha_length,
                          const int& num_alpha0,
                          const int& npeop,
                          const int& nitem,
                          const int& nclass,
                          const int& maxiter,
                          const double& step_length,
                          const double& tol,
                          const int& it,
                          const int ncores) {
  
  
  // // // 平行化過程中存取List會出問題
  // std::vector<MatrixXd> xcond_vec(xcond.size());
  // for (int m = 0; m < xcond.size();++m ) {
  //   if ( !Rf_isNull(xcond[m]) ) {
  //     xcond_vec[m] = as<MatrixXd>(xcond[m]);
  //   }
  // }
  
  
  // double llik= f2_rcpp(alpha_vector,
  //                      Y_comp,xcond,nlevels,
  //                      nxcond,h_p,
  //                      num_alpha0,npeop,
  //                      nitem,nclass);
  int count=0;
  double llik=( h_p.array()*cond_prob.array().log() ).sum();
  
  VectorXd direction=Score_hess_alpha_Para_cpp(Pi_minus_1,
                                               Y,
                                               h_p,
                                               npeop,
                                               nitem,
                                               nclass,
                                               nlevels,
                                               nxcond,
                                               xcond,
                                               direct,
                                               diag_nclass,
                                               alpha_length,
                                               ncores);

  
  // VectorXd direction=temp["direction"];
  
  
  // List temp_1=tran_vec_to_list_cpp(alpha_vector,
  //                                  nlevels,
  //                                  nxcond,
  //                                  num_alpha0,
  //                                  nitem,
  //                                  nclass);
  // List alpha0=temp_1["alpha0"];
  // List alpha=temp_1["alpha"];
  
  
  
  double llik_p;
  
  
  double a=1.0;
  // List alpha0;
  // List al;
  // 
  List temp_1;
  
  VectorXd alpha_vector_p;
  List alpha0_p;
  List alpha_p;
  List cond_prob_Pi_minus_1;
  // List Pi_minus_1_p;
  MatrixXd cond_prob_p;
  
  int iter = 0;
  for( iter = 0; iter < maxiter; ++iter) {
    
    alpha_vector_p=alpha_vector+a*direction;
    
    temp_1=tran_vec_to_list_cpp(alpha_vector_p,
                                nlevels,
                                nxcond,
                                num_alpha0,
                                nitem,
                                nclass);
    
    
    alpha0_p=temp_1["alpha0"];
    alpha_p=temp_1["alpha"];
    
    
    cond_prob_Pi_minus_1=conditional_prob_and_Pi_cpp(alpha0_p,
                                                     alpha_p,
                                                     Y_comp,
                                                     xcond,
                                                     nxcond,
                                                     npeop,
                                                     nitem,
                                                     nclass);
    cond_prob_p=cond_prob_Pi_minus_1["cond_prob"];
    // Pi_minus_1_p=cond_prob_Pi_minus_1["Pi_minus_1"];
    
    llik_p=( h_p.array()*cond_prob_p.array().log() ).sum(); 
    
    
    // Rcpp::Rcout << " llik_p = " << llik_p << ", llik = " << llik <<direction<< std::endl;
    
    
    if(llik_p>llik){
      ++count;
      alpha_vector=alpha_vector_p;
      
      llik=llik_p;
      Pi_minus_1=cond_prob_Pi_minus_1["Pi_minus_1"];
      cond_prob=cond_prob_p;
      
      alpha0=alpha0_p;
      alpha=alpha_p;
      
      if(a*direction.cwiseAbs().maxCoeff()<tol|count==it) break;
      
      // Pi_minus_1=Pi_minus_1_cpp(alpha_vector,
      //                           xcond,
      //                           nlevels,
      //                           nxcond,
      //                           num_alpha0,
      //                           npeop,
      //                           nitem,
      //                           nclass);
      
      direction=Score_hess_alpha_Para_cpp(Pi_minus_1,
                                          Y,
                                          h_p,
                                          npeop,
                                          nitem,
                                          nclass,
                                          nlevels,
                                          nxcond,
                                          xcond,
                                          direct,
                                          diag_nclass,
                                          alpha_length,
                                          ncores);
      // direction=temp["direction"];
    }else{
      a*=step_length;
      // a=a*step_length;
      if(a*direction.cwiseAbs().maxCoeff()<tol) break;
    }
    
    // if((a*direction).cwiseAbs().maxCoeff()<tol) break;
  }
  
  
  
  
  
  List result = List::create(
    Named("alpha_vector_p") = alpha_vector,
    Named("alpha0") = alpha0,
    Named("alpha") = alpha,
    Named("Pi_minus_1") = Pi_minus_1,
    Named("cond_prob") = cond_prob,
    Named("direction") = direction,
    Named("a") = a,
    Named("llik") = llik,
    Named("iter") = iter);
  
  
  return result;
}
    
       
    
    
    

// [[Rcpp::export]]
// 作為rlca alpha最大值的修改
// 仿max_beta_1
std::vector<std::vector<MatrixXd>> max_alpha_5_for_omp_cpp(VectorXd& alpha_vector,
                                                           std::vector<MatrixXd>& alpha0,
                                                           std::vector<MatrixXd>& alpha,
                                                           const MatrixXd& Y,
                                                           const MatrixXd& Y_comp,
                                                           std::vector<MatrixXd> Pi_minus_1,
                                                           MatrixXd& cond_prob,
                                                           const MatrixXd& h_p,
                                                           const VectorXi& nlevels,
                                                           const VectorXi& nxcond,
                                                           const std::vector<MatrixXd> xcond_vec,
                                                           const MatrixXd& direct,
                                                           const MatrixXd& diag_nclass,
                                                           const int& alpha_length,
                                                           const int& num_alpha0,
                                                           const int& npeop,
                                                           const int& nitem,
                                                           const int& nclass,
                                                           const int& maxiter,
                                                           const double& step_length,
                                                           const double& tol,
                                                           const int& it) {


  // double llik= f2_rcpp(alpha_vector,
  //                      Y_comp,xcond,nlevels,
  //                      nxcond,h_p,
  //                      num_alpha0,npeop,
  //                      nitem,nclass);
  int count=0;
  double llik=( h_p.array()*cond_prob.array().log() ).sum();

  VectorXd direction=Score_hess_alpha_for_omp_cpp(Pi_minus_1,
                                                  Y,
                                                  h_p,
                                                  npeop,
                                                  nitem,
                                                  nclass,
                                                  nlevels,
                                                  nxcond,
                                                  xcond_vec,
                                                  direct,
                                                  diag_nclass,
                                                  alpha_length);



  // List temp_1=tran_vec_to_list_cpp(alpha_vector,
  //                                  nlevels,
  //                                  nxcond,
  //                                  num_alpha0,
  //                                  nitem,
  //                                  nclass);
  // List alpha0=temp_1["alpha0"];
  // List alpha=temp_1["alpha"];


  double a=1.0;



  // int iter = 0;
  for(int iter = 0; iter < maxiter; ++iter) {

    VectorXd alpha_vector_p=alpha_vector+a*direction;

    auto temp_1=tran_vec_to_list_for_omp_cpp(alpha_vector_p,
                                             nlevels,
                                             nxcond,
                                             num_alpha0,
                                             nitem,
                                             nclass);


    auto alpha0_p=temp_1[0];
    auto alpha_p=temp_1[1];


    // List cond_prob_Pi_minus_1=conditional_prob_and_Pi_cpp(alpha0_p,alpha_p,
    //                                                  Y_comp,xcond,
    //                                                  nxcond,
    //                                                  npeop,
    //                                                  nitem,
    //                                                  nclass);
  // MatrixXd cond_prob_p=cond_prob_Pi_minus_1["cond_prob"];
  auto cond_prob_Pi_minus_1=
    conditional_prob_and_Pi_for_omp_cpp(alpha0_p,
                                        alpha_p,
                                        Y_comp,
                                        xcond_vec,
                                        nxcond,
                                        npeop,
                                        nitem,
                                        nclass);

  MatrixXd cond_prob_p=cond_prob_Pi_minus_1[0][0];





    // Pi_minus_1_p=cond_prob_Pi_minus_1["Pi_minus_1"];

    double llik_p=( h_p.array()*cond_prob_p.array().log() ).sum();


    // Rcpp::Rcout << " llik_p = " << llik_p << ", llik = " << llik <<direction<< std::endl;


    if(llik_p>llik){
      ++count;
      alpha_vector=alpha_vector_p;

      llik=llik_p;
      // Pi_minus_1=cond_prob_Pi_minus_1["Pi_minus_1"];
      Pi_minus_1=cond_prob_Pi_minus_1[1];
      cond_prob=cond_prob_p;

      alpha0=alpha0_p;
      alpha=alpha_p;

      if(a*direction.cwiseAbs().maxCoeff()<tol|count==it) break;

      // Pi_minus_1=Pi_minus_1_cpp(alpha_vector,
      //                           xcond,
      //                           nlevels,
      //                           nxcond,
      //                           num_alpha0,
      //                           npeop,
      //                           nitem,
      //                           nclass);

      direction.noalias()=
        Score_hess_alpha_for_omp_cpp(Pi_minus_1,
                                     Y,
                                     h_p,
                                     npeop,
                                     nitem,
                                     nclass,
                                     nlevels,
                                     nxcond,
                                     xcond_vec,
                                     direct,
                                     diag_nclass,
                                     alpha_length);

    }else{
      a*=step_length;
      // a=a*step_length;
      if(a*direction.cwiseAbs().maxCoeff()<tol) break;
    }

    // if((a*direction).cwiseAbs().maxCoeff()<tol) break;
  }


  std::vector<std::vector<MatrixXd>> result(5);
  result[0] = {alpha_vector.matrix()}; // VectorXd -> MatrixXd
  result[1] = alpha0; 
  result[2] = alpha; 
  result[3] = Pi_minus_1; 
  result[4] = {cond_prob}; 
  return result;


  
  
  // List result = List::create(
  //   Named("alpha_vector_p") = alpha_vector,
  //   Named("alpha0") = alpha0,
  //   Named("alpha") = alpha,
  //   Named("Pi_minus_1") = Pi_minus_1,
  //   Named("cond_prob") = cond_prob,
  //   Named("direction") = direction,
  //   Named("a") = a,
  //   Named("llik") = llik,
  //   Named("iter") = iter);
  // 
  // 
  // return result;
}

    
    
    
    

// [[Rcpp::export]]
List Estimation_cpp(MatrixXd& beta_p,
                    VectorXd& alpha_vector_p,
                    List& alpha0_p,
                    List& alpha_p,
                    const MatrixXd& Xprev,
                    const int& nxprev,
                    const int& nclass,
                    const MatrixXd& Y,
                    const MatrixXd& Y_comp,
                    const VectorXi& nlevels,
                    const VectorXi& nxcond,
                    const List& xcond,
                    const MatrixXd& direct,
                    const MatrixXd& diag_nclass,
                    const int& alpha_length,
                    const int& num_alpha0,
                    const int& npeop,
                    const int& nitem,
                    const double& step_length,
                    const double& tol,
                    const int& it,
                    const int& maxiter,
                    const int& maxiter_para,
                    const double& tol_para,
                    const double& tol_likeli) {
  
  VectorXd log_lik(maxiter+1);
  log_lik.setZero();
  log_lik[0] = R_NegInf;
  
  List cond_prob_and_Pi_minus_1=
    conditional_prob_and_Pi_cpp(alpha0_p,
                                alpha_p,
                                Y_comp,
                                xcond,
                                nxcond,
                                npeop,
                                nitem,
                                nclass);
  
  MatrixXd cond_prob_p=cond_prob_and_Pi_minus_1["cond_prob"];
  
  List Pi_minus_1=cond_prob_and_Pi_minus_1["Pi_minus_1"];
  
  MatrixXd eta_p = eta_cpp(beta_p,Xprev,nxprev,nclass);
  int iter=0; 
  for(iter = 0; iter < maxiter; ++iter){
    
    MatrixXd beta_temp_p=beta_p;
    MatrixXd h_p= h_p_cpp(eta_p,cond_prob_p);
    
    List maxbeta=max_beta_6_cpp(beta_p,
                                Xprev,
                                eta_p,
                                h_p,
                                npeop,
                                nxprev,
                                nclass,
                                maxiter_para,
                                step_length,
                                tol,
                                it);
    
    beta_p=maxbeta["beta_p"];
    
    eta_p =maxbeta["eta"];
    
    VectorXd alpha_vector=alpha_vector_p;
    
    
    List temp=max_alpha_5_cpp(alpha_vector,
                              alpha0_p,
                              alpha_p,
                              Y,
                              Y_comp,
                              Pi_minus_1,
                              cond_prob_p,
                              h_p,
                              nlevels,
                              nxcond,
                              xcond,
                              direct,
                              diag_nclass,
                              alpha_length,
                              num_alpha0,
                              npeop,
                              nitem,
                              nclass,
                              maxiter_para,
                              step_length,
                              tol,
                              it);
    
    
    
  
    alpha_vector_p=temp["alpha_vector_p"];
    alpha0_p=temp["alpha0"];
    alpha_p =temp["alpha"];
      
      
    Pi_minus_1=temp["Pi_minus_1"];
    cond_prob_p=temp["cond_prob"];
    
    log_lik[iter+1]=((cond_prob_p.array()*eta_p.array()).rowwise().sum()).log().sum();
    
    
    
    
    double max_diff_alpha = (alpha_vector_p - alpha_vector).cwiseAbs().maxCoeff();
    double max_diff_beta = (beta_temp_p - beta_p).array().abs().maxCoeff();
    
    // 條件 1：參數收斂
    if ((max_diff_alpha < tol_para) && (max_diff_beta < tol_para)) break;
    
    // 條件 2：對數概似函數收斂
    double log_lik_diff = std::abs(log_lik[iter+1] - log_lik[iter]);
    if (log_lik_diff < tol_likeli) break;  
     
        
  }
  
  if(iter==maxiter) iter=iter-1;
  
  List result = List::create(
    Named("log_lik") = log_lik[iter+1],
    // Named("log_lik") =((cond_prob_p.array()*eta_p.array()).rowwise().sum()).log().sum(),
    Named("beta_p") = beta_p,
    Named("alpha0_p") = alpha0_p,
    Named("alpha_p") = alpha_p,
    Named("eta_p") = eta_p,
    Named("Pi_minus_1") = Pi_minus_1,
    Named("cond_prob_p") = cond_prob_p,
    Named("iter") = iter+1);
  
  
  return result;
}
    
    
    
// [[Rcpp::export]]
List Estimation_test_cpp(MatrixXd& beta_p,
                         VectorXd& alpha_vector_p,
                         List& alpha0_p,
                         List& alpha_p,
                         const MatrixXd& Xprev,
                         const int& nxprev,
                         const int& nclass,
                         const MatrixXd& Y,
                         const MatrixXd& Y_comp,
                         const VectorXi& nlevels,
                         const VectorXi& nxcond,
                         const List& xcond,
                         const MatrixXd& direct,
                         const MatrixXd& diag_nclass,
                         const int& alpha_length,
                         const int& num_alpha0,
                         const int& npeop,
                         const int& nitem,
                         const double& step_length,
                         const double& tol,
                         const int& it,
                         const int& maxiter,
                         const int& maxiter_para,
                         const double& tol_para,
                         const double& tol_likeli) {

  VectorXd log_lik(maxiter+1);
  log_lik.setZero();
  log_lik[0] = R_NegInf;

  
  
  List cond_prob_and_Pi_minus_1=
    conditional_prob_and_Pi_cpp(alpha0_p,
                                alpha_p,
                                Y_comp,
                                xcond,
                                nxcond,
                                npeop,
                                nitem,
                                nclass);

  // MatrixXd cond_prob_p=cond_prob;
  MatrixXd cond_prob_p=cond_prob_and_Pi_minus_1["cond_prob"];
  
  List Pi_minus_1=cond_prob_and_Pi_minus_1["Pi_minus_1"];

  MatrixXd eta_p = eta_cpp(beta_p,Xprev,nxprev,nclass);
  
  int iter=0;
  for(iter = 0; iter < maxiter; ++iter){

    MatrixXd beta_temp_p=beta_p;
    MatrixXd h_p= h_p_cpp(eta_p,cond_prob_p);


    int count=0;
    MatrixXd eta=eta_p;
    VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
    MatrixXd hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);

    // 計算ginv
    CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
    MatrixXd direction = -1*cod.pseudoInverse() * deri;
    direction.resize(nxprev+1 ,nclass-1);
    double llik=(h_p.array()*eta_p.array().log()).sum();

    double a=1.0;
    for(int k = 0; k < maxiter; ++k){

      MatrixXd beta_pp=beta_p+a*direction;
      eta_p=eta_cpp(beta_pp,Xprev,nxprev,nclass);
      double llik_p=(h_p.array()*eta_p.array().log()).sum();


      if(llik_p>llik){
        // Rcpp::Rcout << "Iteration " << iter << ", llik: " << llik_p << std::endl;
        ++count;
        beta_p=beta_pp;
        llik= llik_p;
        eta=eta_p;
        if(a*direction.cwiseAbs().maxCoeff()<tol|count==it) break;

        deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
        hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);

        // 計算ginv
        CompleteOrthogonalDecomposition<MatrixXd> cod(hess);

        direction = -1*cod.pseudoInverse() * deri;

        direction.resize(nxprev+1 ,nclass-1);

      }else{
        a=a*step_length;
        if(a*direction.cwiseAbs().maxCoeff()<tol) break;
      }
    }
    eta_p=eta;



    VectorXd alpha_vector=alpha_vector_p;


    List temp=max_alpha_5_cpp(alpha_vector,
                              alpha0_p,
                              alpha_p,
                              Y,
                              Y_comp,
                              Pi_minus_1,
                              cond_prob_p,
                              h_p,
                              nlevels,
                              nxcond,
                              xcond,
                              direct,
                              diag_nclass,
                              alpha_length,
                              num_alpha0,
                              npeop,
                              nitem,
                              nclass,
                              maxiter_para,
                              step_length,
                              tol,
                              it);




    alpha_vector_p=temp["alpha_vector_p"];
    alpha0_p=temp["alpha0"];
    alpha_p =temp["alpha"];


    Pi_minus_1=temp["Pi_minus_1"];
    cond_prob_p=temp["cond_prob"];

    log_lik[iter+1]=((cond_prob_p.array()*eta_p.array()).rowwise().sum()).log().sum();

    // 條件 2：對數概似函數收斂
    double log_lik_diff = std::abs(log_lik[iter+1] - log_lik[iter]);
    if (log_lik_diff < tol_likeli) break;


    double max_diff_alpha = (alpha_vector_p - alpha_vector).cwiseAbs().maxCoeff();
    double max_diff_beta = (beta_temp_p - beta_p).array().abs().maxCoeff();

    // 條件 1：參數收斂
    if ((max_diff_alpha < tol_para) && (max_diff_beta < tol_para)) break;



  }

  if(iter==maxiter) iter=iter-1;

  List result = List::create(Named("log_lik") = log_lik[iter+1],
                             // Named("log_lik") =((cond_prob_p.array()*eta_p.array()).rowwise().sum()).log().sum(),
                             Named("beta_p") = beta_p,
                             Named("alpha0_p") = alpha0_p,
                             Named("alpha_p") = alpha_p,
                             Named("eta_p") = eta_p,
                             Named("Pi_minus_1") = Pi_minus_1,
                             Named("cond_prob_p") = cond_prob_p,
                             Named("iter") = iter+1);


  return result;
}

    
    
// [[Rcpp::export]]
List parallel_estimation_cpp(List beta_list,
                             VectorXd alpha_vector_p,
                             List alpha0_p,
                             List alpha_p,
                             const MatrixXd& Xprev,
                             const int& nxprev,
                             const int& nclass,
                             const MatrixXd& Y,
                             const MatrixXd& Y_comp,
                             const VectorXi& nlevels,
                             const VectorXi& nxcond,
                             const List& xcond,
                             const MatrixXd& direct,
                             const MatrixXd& diag_nclass,
                             const int& alpha_length,
                             const int& num_alpha0,
                             const int& npeop,
                             const int& nitem,
                             const double& step_length,
                             const double& tol,
                             const int& it,
                             const int& maxiter,
                             const int& maxiter_para,
                             const double& tol_para,
                             const double& tol_likeli) {
  
  int B = beta_list.size();
  List output(B);
  
#pragma omp parallel for 
  for(int b = 0; b < B; ++b){
    MatrixXd beta_p = as<MatrixXd>(beta_list[b]);
    
    // 因為 alpha_vector_p, alpha0_p, alpha_p 是共用的，要複製副本進入每條 thread
    VectorXd alpha_vector_copy = alpha_vector_p;
    List alpha0_copy = clone(alpha0_p);
    List alpha_copy = clone(alpha_p);
    
    List result = Estimation_cpp(beta_p,
                                 alpha_vector_copy,
                                 alpha0_copy,
                                 alpha_copy,
                                 Xprev,
                                 nxprev,
                                 nclass,
                                 Y,
                                 Y_comp,
                                 nlevels,
                                 nxcond,
                                 xcond,
                                 direct,
                                 diag_nclass,
                                 alpha_length,
                                 num_alpha0,
                                 npeop,
                                 nitem,
                                 step_length,
                                 tol,
                                 it,
                                 maxiter,
                                 maxiter_para,
                                 tol_para,
                                 tol_likeli);
    
    output[b] = result;
  }
  
  return output;
}
    
    
    

// [[Rcpp::export]]
List prob_cpp(const MatrixXd& beta,
              const List& alpha0, 
              const List& alpha,
              const MatrixXd& Xprev,
              const List& xcond,
              const MatrixXd& e_comp,
              const int& npeop,
              const int& nitem,
              const int& nclass,
              const int& nxprev,
              const VectorXi& nxcond,
              const int& H) {
  
  List temp_Pi_jtemp_Pi_j(nclass);
  List Pi_minus_1(nclass);
  MatrixXd eta_p=eta_cpp(beta,Xprev,nxprev,nclass);  
    
  List cond_prob_for_all(nclass);
  MatrixXd p_i = MatrixXd::Zero(npeop,H);
  
  
  VectorXd eta_p_j;
  MatrixXd p_i_temp;
  
  MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  
  // MatrixXd Pi_j=MatrixXd::Zero(npeop, nxcond.sum());
  // MatrixXd Pi_minus_1_j=MatrixXd::Zero(npeop, nxcond.sum()-nitem);
  
  MatrixXd cond_prob(npeop, nclass);
  
  MatrixXd Pimkj;
  
  MatrixXd temp_matrix;
  
  
  // MatrixXd Pi_j;
  MatrixXd Pi_minus_1_j;
  MatrixXd Pimkj_min_1;
  for (int j = 0; j < nclass; ++j) {
    
    // NumericMatrix Pi_j(npeop, 0);
    MatrixXd Pi_j(npeop, 0);
    MatrixXd Pi_minus_1_j(npeop, 0);
    
    
    int temp_nxcond=0;
    int temp_num_Pi_minus_1_j=0;
    for (int m = 0; m < nitem; ++m) {
      
      
      // Rf_isNull(xcond[m])
      if ( nxcond[m]==0 ) {
        MatrixXd alpha0_m = alpha0[m];
        // MatrixXd eigen_A =onesVector;
        
        // NumericVector alpha0_m_j=alpha0_m(j,_);
        // Map<VectorXd> eigen_alpha0_m_j(as<Map<VectorXd> >(alpha0_m_j));
        
        VectorXd alpha0_m_j=alpha0_m.row(j);
        
        // 创建一个MatrixXd
        MatrixXd matrixA(1, alpha0_m_j.size());
        matrixA.row(0)=alpha0_m_j;
        
        
        // MatrixXd mat_R=matrixA;
        
        MatrixXd zeroVector_c = MatrixXd::Zero(matrixA.rows(), 1);
        
        MatrixXd eigen_B =cbindEigen_matrix(matrixA,zeroVector_c);
        
        // MatrixXd eigen_B = as<Map<MatrixXd> >( cbind(rbind(alpha0_m.row(j), alpha_m), 0.0) );
        // MatrixXd temp = eigen_A * eigen_B;
        MatrixXd temp = onesVector * eigen_B;
        // temp = temp - temp.rowwise().maxCoeff().replicate(1, temp.cols());
        
        temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
        
        
        MatrixXd inputMatrix = temp.array().exp();
        VectorXd inputVector = inputMatrix.rowwise().sum();
        Pimkj=inputMatrix.array().colwise() / inputVector.array();
        
        
        
         Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
        Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
        // Pi_minus_1_j.block(0, temp_num_Pi_minus_1_j, npeop, Pimkj.cols() - 1) = Pimkj_min_1;
        
        
        
        Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
        // Pi_j.block(0, temp_nxcond, npeop, nxcond[m]) = Pimkj;
      }else{
        
        MatrixXd alpha0_m = alpha0[m];
        MatrixXd alpha_m = alpha[m];
        MatrixXd xcond_m = xcond[m];
        
        // Map<MatrixXd> eigen_A(as<Map<MatrixXd> >(cbindCpp(1.0, xcond_m)));
        // Map<MatrixXd> eigen_B(as<Map<MatrixXd> >(cbindCpp(rbindCpp(alpha0_m(j, _), alpha_m), 0.0)));
        // MatrixXd temp = eigen_A * eigen_B;
        
        
        // Map<MatrixXd> eigen_xcond_m(as<Map<MatrixXd> >(xcond_m));
        MatrixXd eigen_A =cbindEigen_matrix(onesVector,xcond_m);
        
        
        // MatrixXd eigen_A = as<Map<MatrixXd> >(cbind(1.0, xcond_m));
        
        // NumericVector alpha0_m_j=alpha0_m(j,_);
        // Map<VectorXd> eigen_alpha0_m_j(as<Map<VectorXd> >(alpha0_m_j));
        
        VectorXd alpha0_m_j=alpha0_m.row(j);
        
        
        // 创建一个MatrixXd
        MatrixXd matrixA(1, alpha0_m_j.size());
        matrixA.row(0)=alpha0_m_j;
        
        
        // Map<MatrixXd> eigen_alpha_m(as<Map<MatrixXd> >(alpha_m));
        MatrixXd mat_R=rbindEigen_matrix(matrixA, alpha_m);
        
        MatrixXd zeroVector_c = MatrixXd::Zero(mat_R.rows(), 1);
        
        MatrixXd eigen_B =cbindEigen_matrix(mat_R,zeroVector_c);
        
        
        // MatrixXd eigen_B = as<Map<MatrixXd> >( cbind(rbind(alpha0_m.row(j), alpha_m), 0.0) );
        MatrixXd temp = eigen_A * eigen_B;
        // temp = temp - temp.rowwise().maxCoeff().replicate(1, temp.cols());
        
        temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
        
        
        
        // 以後再回來處理
        // NumericVector aa = apply(temp, 1, max);
        // temp = exp(temp - aa);
        
        // NumericMatrix Pimkj = temp / rowSumsCpp(temp);
        
        
        MatrixXd inputMatrix=temp.array().exp();
        VectorXd inputVector = inputMatrix.rowwise().sum();
        Pimkj=inputMatrix.array().colwise() / inputVector.array();
        
        
        // Pimkj = divideByRowSums(expMatrix(wrap(temp)));
        // 
        // // Pi_j = cbind(Pi_j, Pimkj);
        // 
        // Map<MatrixXd> eigen_Pimkj(as<Map<MatrixXd> >(Pimkj));
        
         Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
        Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
        
        // Pi_minus_1_j.block(0, temp_num_Pi_minus_1_j, npeop, Pimkj.cols() - 1) = Pimkj_min_1;
        
        Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
        // Pi_j.block(0, temp_nxcond, npeop, nxcond[m]) = Pimkj;
      }
      // temp_nxcond=temp_nxcond+nxcond[m];
      // temp_num_Pi_minus_1_j=temp_num_Pi_minus_1_j+Pimkj.cols() - 1;
      
    }
    Pi_minus_1[j]=Pi_minus_1_j;
    
    MatrixXd cond_prob_for_all_j = MatrixXd::Zero(npeop,H);
    
    for (int h = 0; h < H; ++h) {

      // VectorXi indices(e_comp.cols());
      // int index = 0;
      // for (int k = 0; k < e_comp.cols(); ++k) {
      //   if (e_comp(h, k) == 1) {
      //     indices(index) = k;
      //     ++index;
      //   }
      // }
      //
      // indices.conservativeResize(index);
      //
      // // 现在indices中包含了满足条件的列的索引
      // // MatrixXd selected_cols = Pi_j.col(indices);
      
      // VectorXd tempeh=e_comp.row(0);
      // MatrixXd temp_Pi_j=Pi_j.transpose().array().colwise()*tempeh.array();
      
      RowVectorXd tempeh=e_comp.row(h);
      MatrixXd temp_Pi_j=Pi_j.array().rowwise()*tempeh.array();
      temp_Pi_j=temp_Pi_j.array().rowwise()+(1-tempeh.array());
      // temp_Pi_jtemp_Pi_j[j]=temp_Pi_j;
      

      cond_prob_for_all_j.col(h)=temp_Pi_j.array().rowwise().prod();


      // MatrixXd temp=Pi_j.col(indices);
      //
      // cond_prob_for_all_j.col(h)=
      //
      // temp.array().log().rowwise().prod().exp();
    }


    cond_prob_for_all[j]=cond_prob_for_all_j;

    eta_p_j=eta_p.col(j);

    p_i_temp=cond_prob_for_all_j.array().colwise() * eta_p_j.array();

    p_i = p_i+p_i_temp;
    
  }
  
    
    
    
  List result = List::create(
    // Named("temp_Pi_jtemp_Pi_j") =temp_Pi_jtemp_Pi_j,
      
      // Named("Pi_j") = Pi_j,
      Named("Pimkj") = Pimkj,
      Named("Pi_minus_1") =Pi_minus_1,
      Named("Pimkj_min_1") =Pimkj_min_1,
      Named("eta") =eta_p,
      Named("cond_prob_for_all") = cond_prob_for_all,
      Named("p_i") = p_i.transpose());  
    
    
    
    // which(e_comp.row(h).array() == 1)
    // 
    // 
    // cond_prob_for_all[[j]]=
    //   sapply(  1:H, function(h) exp( rowSums(  
    //       log(Pi_j[ ,which(e_comp[h,]==1) ])  )  )    )
    // 
    // 
    //   Pi_j[ ,which(e_comp[1,]==1) ]
    
    
  return result;
  // return p_i;
  // return p_i.transpose();
}



    
// [[Rcpp::export]]
List prob_1_cpp(const MatrixXd& beta,
                const List& alpha0, 
                const List& alpha,
                const MatrixXd& eta_p,
                const MatrixXd& Xprev,
                const List& xcond,
                const MatrixXd& e_comp,
                const int& npeop,
                const int& nitem,
                const int& nclass,
                const int& nxprev,
                const VectorXi& nxcond,
                const int& H) {
  
  // List temp_Pi_jtemp_Pi_j(nclass);
  List Pi_minus_1(nclass);
  // MatrixXd eta_p=eta_cpp(beta,Xprev,nxprev,nclass);  
  
  List cond_prob_for_all(nclass);
  MatrixXd p_i = MatrixXd::Zero(npeop,H);
  
  
  // VectorXd eta_p_j;
  // MatrixXd p_i_temp;
  
  MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  
  // MatrixXd Pi_j=MatrixXd::Zero(npeop, nxcond.sum());
  // MatrixXd Pi_minus_1_j=MatrixXd::Zero(npeop, nxcond.sum()-nitem);
  
  MatrixXd cond_prob(npeop, nclass);
  
  MatrixXd Pimkj;
  
  MatrixXd temp_matrix;
  
  
  // Eigen::MatrixXd eigen_B;
  Eigen::MatrixXd temp;
  
  // MatrixXd Pi_j;
  // MatrixXd Pi_minus_1_j;
  MatrixXd Pimkj_min_1;
  for (int j = 0; j < nclass; j++) {
    
    MatrixXd Pi_j(npeop, 0);
    MatrixXd Pi_minus_1_j(npeop, 0);
    

    for (int m = 0; m < nitem; m++) {
      
      
      // Rf_isNull(xcond[m])
      if ( nxcond[m]==0 ) {
        MatrixXd alpha0_m = alpha0[m];

        // eigen_B.resize(1, alpha0[m].cols() + 1);
        
        Eigen::MatrixXd eigen_B(1, alpha0_m.cols() + 1);
        // eigen_B(1, alpha0[m].cols() + 1);
        eigen_B.leftCols(alpha0_m.cols()) = alpha0_m.row(j);
        eigen_B.rightCols(1).setZero();
        
        temp = onesVector * eigen_B;
    
      }else{
        MatrixXd alpha0_m = alpha0[m];
        MatrixXd alpha_m = alpha[m];
        MatrixXd xcond_m = xcond[m];
        // eigen_B.resize(alpha[m].rows() + 1, alpha[m].cols() + 1);
        
        Eigen::MatrixXd eigen_B(alpha_m.rows() + 1, alpha_m.cols() + 1);
        // eigen_B(alpha[m].rows() + 1, alpha[m].cols() + 1);
        // 填充上半部分：[alpha0[m].row(j); alpha[m]]
        eigen_B.topRows(1) = alpha0_m.row(j);      // 第一行 = alpha0[m].row(j)
        eigen_B.bottomRows(alpha_m.rows()) = alpha_m; // 剩余行 = alpha[m]
        
        // 最右侧补零列
        eigen_B.rightCols(1).setZero();
        temp = cbindEigen_matrix(onesVector,xcond_m) * eigen_B;
      }
      
      
      // 计算 softmax
      temp.array().colwise() -= temp.rowwise().maxCoeff().array();
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
    
    MatrixXd cond_prob_for_all_j = MatrixXd::Zero(npeop,H);
    
   
    
    MatrixXd Pi_j_log=Pi_j.array().log();

    MatrixXd cond_dist_j=e_comp*Pi_j_log.transpose();
    cond_dist_j=cond_dist_j.array().exp();
    cond_prob_for_all_j=cond_dist_j.transpose();
    

    
    
    cond_prob_for_all[j]=cond_prob_for_all_j;
    
    // eta_p_j=eta_p.col(j);
    
    // p_i_temp=cond_prob_for_all_j.array().colwise() * eta_p_j.array();
    // p_i_temp=cond_prob_for_all_j.array().colwise() * eta_p.col(j).array();
    // p_i =p_i+ p_i_temp;

    
    p_i.noalias() += cond_prob_for_all_j.cwiseProduct(
      eta_p.col(j).replicate(1, H));
  }
  
  
  
  
  List result = List::create(
    // Named("temp_Pi_jtemp_Pi_j") =temp_Pi_jtemp_Pi_j,
    
    // Named("Pi_j") = Pi_j,
    Named("Pimkj") = Pimkj,
    Named("Pi_minus_1") =Pi_minus_1,
    Named("Pimkj_min_1") =Pimkj_min_1,
    Named("eta") =eta_p,
    Named("cond_prob_for_all") = cond_prob_for_all,
    Named("p_i") = p_i.transpose());  
  
  
  
  // which(e_comp.row(h).array() == 1)
  // 
  // 
  // cond_prob_for_all[[j]]=
  //   sapply(  1:H, function(h) exp( rowSums(  
  //       log(Pi_j[ ,which(e_comp[h,]==1) ])  )  )    )
  // 
  // 
  //   Pi_j[ ,which(e_comp[1,]==1) ]
  
  
  return result;
  // return p_i;
  // return p_i.transpose();
}

    
    

// [[Rcpp::export]]
List cond_prob_for_all_cpp(const List& alpha0, 
                           const List& alpha,
                           const List& xcond,
                           const MatrixXd& e_comp,
                           const int& npeop,
                           const int& nitem,
                           const int& nclass,
                           const VectorXi& nxcond,
                           const int& H) {
                            
  
  // List temp_Pi_jtemp_Pi_j(nclass);
  List Pi_minus_1(nclass);
  // MatrixXd eta_p=eta_cpp(beta,Xprev,nxprev,nclass);  
  
  List cond_prob_for_all(nclass);
  MatrixXd p_i = MatrixXd::Zero(npeop,H);
  
  
  // VectorXd eta_p_j;
  MatrixXd p_i_temp;
  
  MatrixXd onesVector = MatrixXd::Ones(npeop, 1);
  
  // MatrixXd Pi_j=MatrixXd::Zero(npeop, nxcond.sum());
  // MatrixXd Pi_minus_1_j=MatrixXd::Zero(npeop, nxcond.sum()-nitem);
  
  MatrixXd cond_prob(npeop, nclass);
  
  MatrixXd Pimkj;
  
  MatrixXd temp_matrix;
  
  
  // MatrixXd Pi_j;
  // MatrixXd Pi_minus_1_j;
  MatrixXd Pimkj_min_1;
  for (int j = 0; j < nclass; ++j) {
    
    // NumericMatrix Pi_j(npeop, 0);
    MatrixXd Pi_j(npeop, 0);
    MatrixXd Pi_minus_1_j(npeop, 0);
    
    
    int temp_nxcond=0;
    int temp_num_Pi_minus_1_j=0;
    for (int m = 0; m < nitem; ++m) {
      
      
      // Rf_isNull(xcond[m])
      if ( nxcond[m]==0 ) {
        MatrixXd alpha0_m = alpha0[m];
        // MatrixXd eigen_A =onesVector;
        
        // NumericVector alpha0_m_j=alpha0_m(j,_);
        // Map<VectorXd> eigen_alpha0_m_j(as<Map<VectorXd> >(alpha0_m_j));
        
        VectorXd alpha0_m_j=alpha0_m.row(j);
        
        // 创建一个MatrixXd
        MatrixXd matrixA(1, alpha0_m_j.size());
        matrixA.row(0)=alpha0_m_j;
        
        
        // MatrixXd mat_R=matrixA;
        
        MatrixXd zeroVector_c = MatrixXd::Zero(matrixA.rows(), 1);
        
        MatrixXd eigen_B =cbindEigen_matrix(matrixA,zeroVector_c);
        
        // MatrixXd eigen_B = as<Map<MatrixXd> >( cbind(rbind(alpha0_m.row(j), alpha_m), 0.0) );
        // MatrixXd temp = eigen_A * eigen_B;
        MatrixXd temp = onesVector * eigen_B;
        // temp = temp - temp.rowwise().maxCoeff().replicate(1, temp.cols());
        
        temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
        
        
        MatrixXd inputMatrix = temp.array().exp();
        VectorXd inputVector = inputMatrix.rowwise().sum();
        Pimkj=inputMatrix.array().colwise() / inputVector.array();
        
        
        
        Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
        Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
        // Pi_minus_1_j.block(0, temp_num_Pi_minus_1_j, npeop, Pimkj.cols() - 1) = Pimkj_min_1;
        
        
        
        Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
        // Pi_j.block(0, temp_nxcond, npeop, nxcond[m]) = Pimkj;
      }else{
        
        MatrixXd alpha0_m = alpha0[m];
        MatrixXd alpha_m = alpha[m];
        MatrixXd xcond_m = xcond[m];
        
        // Map<MatrixXd> eigen_A(as<Map<MatrixXd> >(cbindCpp(1.0, xcond_m)));
        // Map<MatrixXd> eigen_B(as<Map<MatrixXd> >(cbindCpp(rbindCpp(alpha0_m(j, _), alpha_m), 0.0)));
        // MatrixXd temp = eigen_A * eigen_B;
        
        
        // Map<MatrixXd> eigen_xcond_m(as<Map<MatrixXd> >(xcond_m));
        MatrixXd eigen_A =cbindEigen_matrix(onesVector,xcond_m);
        
        
        // MatrixXd eigen_A = as<Map<MatrixXd> >(cbind(1.0, xcond_m));
        
        // NumericVector alpha0_m_j=alpha0_m(j,_);
        // Map<VectorXd> eigen_alpha0_m_j(as<Map<VectorXd> >(alpha0_m_j));
        
        VectorXd alpha0_m_j=alpha0_m.row(j);
        
        
        // 创建一个MatrixXd
        MatrixXd matrixA(1, alpha0_m_j.size());
        matrixA.row(0)=alpha0_m_j;
        
        
        // Map<MatrixXd> eigen_alpha_m(as<Map<MatrixXd> >(alpha_m));
        MatrixXd mat_R=rbindEigen_matrix(matrixA, alpha_m);
        
        MatrixXd zeroVector_c = MatrixXd::Zero(mat_R.rows(), 1);
        
        MatrixXd eigen_B =cbindEigen_matrix(mat_R,zeroVector_c);
        
        
        // MatrixXd eigen_B = as<Map<MatrixXd> >( cbind(rbind(alpha0_m.row(j), alpha_m), 0.0) );
        MatrixXd temp = eigen_A * eigen_B;
        // temp = temp - temp.rowwise().maxCoeff().replicate(1, temp.cols());
        
        temp = temp.array().colwise() - temp.rowwise().maxCoeff().array();
        
        
        
        // 以後再回來處理
        // NumericVector aa = apply(temp, 1, max);
        // temp = exp(temp - aa);
        
        // NumericMatrix Pimkj = temp / rowSumsCpp(temp);
        
        
        MatrixXd inputMatrix=temp.array().exp();
        VectorXd inputVector = inputMatrix.rowwise().sum();
        Pimkj=inputMatrix.array().colwise() / inputVector.array();
        
        
        // Pimkj = divideByRowSums(expMatrix(wrap(temp)));
        // 
        // // Pi_j = cbind(Pi_j, Pimkj);
        // 
        // Map<MatrixXd> eigen_Pimkj(as<Map<MatrixXd> >(Pimkj));
        
        Pimkj_min_1 = Pimkj.leftCols(Pimkj.cols() - 1);
        Pi_minus_1_j=cbindEigen_matrix(Pi_minus_1_j, Pimkj_min_1);
        
        // Pi_minus_1_j.block(0, temp_num_Pi_minus_1_j, npeop, Pimkj.cols() - 1) = Pimkj_min_1;
        
        Pi_j = cbindEigen_matrix(Pi_j, Pimkj);
        // Pi_j.block(0, temp_nxcond, npeop, nxcond[m]) = Pimkj;
      }
      // temp_nxcond=temp_nxcond+nxcond[m];
      // temp_num_Pi_minus_1_j=temp_num_Pi_minus_1_j+Pimkj.cols() - 1;
      
    }
    Pi_minus_1[j]=Pi_minus_1_j;
    
    MatrixXd cond_prob_for_all_j = MatrixXd::Zero(npeop,H);
    
    // for (int h = 0; h < H; ++h) {
    //   
    //   // VectorXi indices(e_comp.cols());
    //   // int index = 0;
    //   // for (int k = 0; k < e_comp.cols(); ++k) {
    //   //   if (e_comp(h, k) == 1) {
    //   //     indices(index) = k;
    //   //     ++index;
    //   //   }
    //   // }
    //   //
    //   // indices.conservativeResize(index);
    //   //
    //   // // 现在indices中包含了满足条件的列的索引
    //   // // MatrixXd selected_cols = Pi_j.col(indices);
    //   
    //   // VectorXd tempeh=e_comp.row(0);
    //   // MatrixXd temp_Pi_j=Pi_j.transpose().array().colwise()*tempeh.array();
    //   
    //   RowVectorXd tempeh=e_comp.row(h);
    //   MatrixXd temp_Pi_j=Pi_j.array().rowwise()*tempeh.array();
    //   temp_Pi_j=temp_Pi_j.array().rowwise()+(1-tempeh.array());
    //   // temp_Pi_jtemp_Pi_j[j]=temp_Pi_j;
    //   
    //   
    //   cond_prob_for_all_j.col(h)=temp_Pi_j.array().rowwise().prod();
    //   
    //   
    //   // MatrixXd temp=Pi_j.col(indices);
    //   //
    //   // cond_prob_for_all_j.col(h)=
    //   //
    //   // temp.array().log().rowwise().prod().exp();
    // }
    
    MatrixXd Pi_j_log=Pi_j.array().log();
    
    MatrixXd cond_dist_j=e_comp*Pi_j_log.transpose();
    cond_dist_j=cond_dist_j.array().exp();
    cond_prob_for_all_j=cond_dist_j.transpose();
    
    
    cond_prob_for_all[j]=cond_prob_for_all_j;
    
  
  }
  
  
  
  List result = List::create(
    // Named("temp_Pi_jtemp_Pi_j") =temp_Pi_jtemp_Pi_j,
    
    // Named("Pi_j") = Pi_j,
    Named("Pimkj") = Pimkj,
    Named("Pi_minus_1") =Pi_minus_1,
    Named("Pimkj_min_1") =Pimkj_min_1,
    Named("cond_prob_for_all") = cond_prob_for_all);  
  
  
  return result;
 
}


// [[Rcpp::export]]
IntegerVector which_cpp(IntegerVector x, double y) {
  IntegerVector indices = seq_len(x.size()) - 1;
  return (indices[x == y]);
}


// [[Rcpp::export]]
VectorXd ymk_cpp(IntegerVector indices,
                 int length) {
  
  
  VectorXd result= VectorXd::Zero(length);
  
  
  // 设置指定索引的元素为 1
  for (int i = 0; i < indices.size(); ++i) {
    int index = indices[i];
    
    if (index >= 0 ) {
      result[index] = 1.0;
    } 
    
  }
  return result;
}



// [[Rcpp::export]]
List Fisher_information_cpp(const MatrixXd& eta_p, 
                            const List& cond_prob_for_all,
                            const MatrixXd& Xprev, 
                            const List& xcond, 
                            const IntegerMatrix& e,
                            const MatrixXd& p_i,
                            const VectorXi& nlevels,
                            const int& npeop,
                            const int& nitem,
                            const int& nclass,
                            const VectorXi& nxcond,
                            const int& npar,
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
  
  // MatrixXd der_alpha_mk;
  // MatrixXd der_alpha;
  // MatrixXd der_beta = MatrixXd::Zero(nclass - 1, Xprev.cols());
  // 
  // MatrixXd der_gamma = MatrixXd::Zero(H * (nclass - 1), nitem);
  // MatrixXd der_alpha = MatrixXd::Zero(nitem * (nclass - 1), H);
  
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

    MatrixXd der_beta = KroneckerProduct(temp,Xprev_i.transpose());

    
    // MatrixXd der_gamma;
    // MatrixXd der_alpha;
    MatrixXd der_gamma(0, H);
    MatrixXd der_alpha(0, H);
    
    for (int m = 0; m < nitem; ++m) {
  

      for (int k = 0; k < (nlevels[m] - 1); ++k) {
        
        // IntegerVector em=e.col(m);
        
        IntegerVector emk=which_cpp(e(_,m), (k+1));
        
        VectorXd Pimkj= VectorXd::Zero(cond_prob.cols());
        VectorXd ymk= VectorXd::Zero(H);
        for(int u=0; u<emk.size(); ++u){
          Pimkj+=cond_prob.row(emk[u]);
          
          if (emk[u] >= 0 ) {
            ymk[emk[u]] = 1.0;
            }
        }
        
        MatrixXd ymk_Pimkj = ymk.replicate(1, nclass) - 
          Pimkj.transpose().replicate(H, 1);
        
        
        MatrixXd der_gamma_mk=(joint_prob_i.array()*ymk_Pimkj.array()).transpose();
        der_gamma=rbindEigen_matrix(der_gamma,der_gamma_mk);
        
        

        if (nxcond[m]!=0) {
          MatrixXd xcond_m = xcond[m];
          MatrixXd xcond_m_i = xcond_m.row(i);
          MatrixXd der_gamma_mk_sum=der_gamma_mk.array().colwise().sum();
          
          MatrixXd der_alpha_mk=xcond_m_i.transpose()*der_gamma_mk_sum;
          der_alpha=rbindEigen_matrix(der_alpha,der_alpha_mk);

        }

        
        // MatrixXd xcond_m = xcond[m];
        // MatrixXd xcond_m_i = xcond_m.row(i);
        // MatrixXd der_gamma_mk_sum=der_gamma_mk.array().colwise().sum();
        // 
        // MatrixXd der_alpha_mk=xcond_m_i.transpose()*der_gamma_mk_sum;
        // der_alpha=rbindEigen_matrix(der_alpha,der_alpha_mk);
         
         
//         emk=(e[,m]==k)
//           Pimkj=colSums(cond_prob[[i]][emk,])
//           ymk=ifelse(emk,1,0)
//           
// #ymk_Pimkj=ymk-Pimkj
//           ymk_Pimkj=matrix(ymk,H,nclass)-matrix(Pimkj,H,nclass,byrow = T)
//             
//             der_gamma_mk= t( joint_prob*ymk_Pimkj )
//             der_alpha_mk=matrix(xcond[[m]][i,],,1)%*%matrix(colSums(der_gamma_mk),1,H)
//             
//             der_gamma=rbind(der_gamma,der_gamma_mk)
//             der_alpha=rbind(der_alpha,der_alpha_mk)
        
        
        
        
        
        // 
        // VectorXi emk = (e.col(m).cast<int>() == (k+1));
        // VectorXd Pimkj = cond_prob[m].transpose() * emk.cast<double>();
        // VectorXd ymk = emk.cast<double>();
        // 
        // MatrixXd ymk_Pimkj = ymk.replicate(1, nclass) - Pimkj.replicate(H, 1);
        // 
        // MatrixXd der_gamma_mk = joint_prob.array() * ymk_Pimkj.array();
        // MatrixXd der_alpha_mk = xcond[m].transpose() * der_gamma_mk.rowwise().sum();
        // 
        // der_gamma.conservativeResize(der_gamma.rows() + der_gamma_mk.rows(), der_gamma.cols());
        // der_gamma.bottomRows(der_gamma_mk.rows()) = der_gamma_mk;
        // 
        // der_alpha.conservativeResize(der_alpha.rows() + der_alpha_mk.rows(), der_alpha.cols());
        // der_alpha.bottomRows(der_alpha_mk.rows()) = der_alpha_mk;
      }
    }

    
    der_parmeter=rbindEigen_matrix(der_beta,der_gamma);
    der_parmeter=rbindEigen_matrix(der_parmeter,der_alpha);
    
    // VectorXd pii_1=1.0/pii.array();
    // MatrixXd pii_1_as_diag=pii_1.asDiagonal();
    
    
    // MatrixXd pii_1_as_diag=pii.asDiagonal().inverse();
    
    F_hat=der_parmeter.transpose().array().colwise()/pii.array();
    MatrixXd Fisher_information_i=der_parmeter*F_hat;
    
    // MatrixXd Fisher_information_i=der_parmeter*pii_1_as_diag*der_parmeter.transpose();
    
//     der_parmeter=rbind(der_beta,der_gamma,der_alpha)
//       F_hat_i=t(der_parmeter)
// # F_hat_sum=F_hat_sum+t(der_parmeter)
// #
//       Fisher_information_i=der_parmeter%*%(F_hat_i/p_i[,i])



    // MatrixXd der_parmeter = der_beta.rowwise().sum();
    // der_parmeter.conservativeResize(der_parmeter.rows() + der_gamma.rows() + der_alpha.rows(),
    //                                 der_parmeter.cols());
    // 
    // MatrixXd F_hat_i = der_parmeter.transpose();
    // 
    // MatrixXd Fisher_information_i = der_parmeter *
    //   (F_hat_i.cwiseProduct(p_i.col(i).replicate(1, H))).transpose();
    // 
    // 
    // 

    // 
    // 
    // 
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
List Fisher_information_by_score_cpp(const MatrixXd& eta_p,
                                     const MatrixXd& h_p,
                                     const MatrixXd& Xprev,
                                     const List& Pi_minus_1,
                                     const MatrixXd& Y,
                                     const List& xcond,
                                     const MatrixXd& direct,
                                     const MatrixXd& diag_nclass,
                                     const int& alpha_length,
                                     const int& npeop,
                                     const int& nxprev,
                                     const int& nitem,
                                     const int& nclass,
                                     const VectorXi& nlevels,
                                     const VectorXi& nxcond,
                                     const int& npar) {
  
  MatrixXd s=h_p.leftCols(nclass-1)-eta_p.leftCols(nclass-1);
  
  
  MatrixXd s_Matrix(npeop,(nclass-1)*(nxprev+1));
  MatrixXd Xprev_Matrix(npeop,(nclass-1)*(nxprev+1));
  
  
  // 利用replicate函數
  for (int j = 0; j < (nclass-1); ++j) {
    s_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = s.col(j).replicate(1, nxprev+1);
    Xprev_Matrix.block(0, j*(nxprev+1), npeop, nxprev+1) = Xprev.block(0, 0, npeop, nxprev+1);
  }
  
  
  MatrixXd a1_temp=s_Matrix.array()*Xprev_Matrix.array();
  
  
  // VectorXd a1 = a1_temp.colwise().sum();
  // // a1 = a1_temp.colwise().sum();
  // 
  // return a1;
  
  
  
  
  

  // 這樣後續會有錯
  // VectorXd eigen_result_deri(alpha_length);
  // MatrixXd eigen_result_hessian(alpha_length,alpha_length);
  
  MatrixXd der = MatrixXd::Zero(npar,npar);
  
  // int cols_Vij_eigen = Y.cols();
  int nrow_result = Y.cols() ;
  int ncol_result = Y.cols() * nclass;
  
  for (int i = 0; i < npeop; ++i) {
    
    VectorXd eigen_result_deri = VectorXd::Zero(alpha_length);
    VectorXd der_i(npar);
    
    for (int j = 0; j < nclass; ++j) {
    
    // RowVectorXd diag_nclass_j = diag_nclass.row(j);
    
    
      MatrixXd diag_nclass_j = diag_nclass.row(j);
      
      MatrixXd Pi_minus_1_j = Pi_minus_1[j];
      
      MatrixXd S = Y - Pi_minus_1_j;
      
      
      // RowVectorXd S_i = S.row(i);
      
      VectorXd S_i = S.row(i);
      // MatrixXd S_i=S(i,_);
      
      
      // NumericVector muij = Pi_minus_1_j(i, _);
      
      VectorXd muij = Pi_minus_1_j.row(i);
      
      // NumericMatrix temp = calculateTemp(muij);
      
      
      MatrixXd temp = muij * muij.transpose();
      
      
      MatrixXd temp_dir=temp.array()*direct.array();
      
      // int nn = muij.size();
      // MatrixXd mat=MatrixXd::Zero(nn, nn);
      // for (int ii = 0; ii < nn; ++ii) {
      //   mat(ii, ii) = muij[ii];
      // }
      
      MatrixXd muij_as_diag=muij.asDiagonal();
      
      MatrixXd Vij_eigen =muij_as_diag-temp_dir;
      
      
      
      // NumericMatrix Vij=wrap(Vij_eigen);
      
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(Vij_eigen);
      MatrixXd Vij_inver = cod.pseudoInverse();
      // MatrixXd Vij_inver = Vij_eigen.inverse();
      
      
      
      // NumericMatrix Vij_inver = ginv_rcpp(Vij);
      // NumericMatrix Vh_ij_inver_h = Vij_inver * h_p(i,j);
      // double element = h_p(i,j);
      // MatrixXd eigen_Vh_ij_inver_h = Vij_inver * element;
      MatrixXd eigen_Vh_ij_inver_h = Vij_inver * h_p(i,j);
      
      
      
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
      
      }
    
    VectorXd a1=a1_temp.row(i);
    // der_i=rbindEigen_matrix(der_i,eigen_result_deri);
    der_i << a1, eigen_result_deri;
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








// [[Rcpp::export]]
// 之後用於取代max_beta_bisection_cpp
List modify_max_beta_bisection_cpp(const MatrixXd& beta,
                                   const MatrixXd& Xprev,
                                   const MatrixXd& eta_p,
                                   const MatrixXd& h_p,
                                   const int& npeop,
                                   const int& nxprev,
                                   const int& nclass,
                                   double& c,
                                   const double& err,
                                   const int& len) {
  
  
  // MatrixXd eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
  VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
  MatrixXd hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
  MatrixXd ginv_hess = cod.pseudoInverse();
  // MatrixXd ginv_hess = hess.inverse();
  
  
  MatrixXd direction = -1*ginv_hess * deri;
  
  
  direction.resize(nxprev+1 ,nclass-1);
  
  
  
  // NumericVector temp_aa(50);
  // NumericVector temp_cc(50);
  
  
  // double temp_a;
  // double temp_c;
  double temp_b;
  double temp_a1;
  double temp_a2;
  
  
  double a = 0.0;
  int iter = 0;
  for( iter = 0; iter < 1000; ++iter) {
    // temp_a = a;
    // temp_c = c;
    
    
    temp_b = (a + c) / 2.0;
    
    temp_a1 = beta_likeli_cpp( beta + temp_b * direction,
                               h_p,
                               Xprev,
                               nxprev,
                               nclass);
    
    
    temp_a2 = beta_likeli_cpp( beta + (temp_b + 0.00000001) * direction,
                               h_p,
                               Xprev,
                               nxprev,
                               nclass);
    
    
    if(temp_a1 < temp_a2) {
      a = temp_b;
      // c = c;
      
      
      // temp_aa[iter]=a;
      // temp_cc[iter]=c;
      
      // if( abs(temp_a - a) < err ) break;
    } else {
      // a = a;
      c = temp_b;
      
      
      // temp_aa[iter]=a;
      // temp_cc[iter]=c;
      
      // if( abs(temp_c - c) < err ) break;
    }
    
    if( (c-a) < err ) break;
    // if( std::max( abs(temp_a - a),abs(temp_c - c) ) < 0.001 ) break;
  }
  
  
  
  
  // int len = static_cast<int>((c - a) / err) + 1;
  // int len = 10;
  double step = (c - a) / (len-1);
  
  // NumericVector  likeli(len+3);
  // NumericVector ak(len+3);
  // List beta_list(len+3);
  
  
  NumericVector likeli(len+2);
  NumericVector ak(len+2);
  List beta_list(len+2);
  
  
  ak[0]=0;
  beta_list[0]=beta;
  likeli[0] = beta_likeli_cpp( beta_list[0],
                               h_p,
                               Xprev,
                               nxprev,
                               nclass);
  
  
  
  
  int max_idx=0;
  
  for(int i = 0; i < len; ++i) {
    ak[i+1]=a + i * step;
    beta_list[i+1]=beta + ak[i+1]* direction;
    likeli[i+1] = beta_likeli_cpp( beta_list[i+1],
                                   h_p,
                                   Xprev,
                                   nxprev,
                                   nclass);
    
    if( likeli[i+1] >= likeli[max_idx] ){
      max_idx=i+1;
    }
    
  }
  
  ak[len+1]=1;
  beta_list[len+1]=beta+direction;
  likeli[len+1] = beta_likeli_cpp( beta_list[len+1],
                                   h_p,
                                   Xprev,
                                   nxprev,
                                   nclass);
  
  if( likeli[len+1] >= likeli[max_idx] ){
    max_idx=len+1;
  }
  
  
  
  // ak[len+1]=c;
  // beta_list[len+1]=beta + c* direction;
  // likeli[len+1] = beta_likeli_cpp( beta_list[len+1],
  //                                  h_p,
  //                                  Xprev,
  //                                  nxprev,
  //                                  nclass);
  // 
  // if( likeli[len+1] >= likeli[max_idx] ){
  //   max_idx=len+1;
  // }
  // 
  // 
  // 
  // ak[len+2]=1;
  // beta_list[len+2]=beta+direction;
  // likeli[len+2] = beta_likeli_cpp( beta_list[len+2],
  //                                  h_p,
  //                                  Xprev,
  //                                  nxprev,
  //                                  nclass);
  // 
  // if( likeli[len+2] >= likeli[max_idx] ){
  //   max_idx=len+2;
  // }
  
  
  
  // double max_tempp = likeli.maxCoeff();
  
  
  // int max_idx = Rcpp::which_max(likeli);
  
  MatrixXd beta_p=beta_list[max_idx];
  
  
  // NumericVector alpha_vector_p=alpha_vector;
  double ak_temp=ak[max_idx];
  
  // 创建一个 List 对象包含多个值
  List result = List::create(
    Named("a") = a,
    Named("c") = c,
    Named("likeli") = likeli,
    Named("len") = len,
    Named("ak") = ak,
    Named("ak_temp") = ak_temp,
    Named("beta_p") = beta_p,
    Named("max_idx") = max_idx+1,
    Named("direction") = direction,
    Named("llik") = likeli[max_idx],
    Named("iter") = iter);
  
  
  return result;
}






// [[Rcpp::export]]
// 作為rlca alpha最大值的修改
// 之後用於取代max_alpha_bisection_cpp
List modify_max_alpha_bisection_cpp(const VectorXd& alpha_vector, 
                                    const MatrixXd& Y,
                                    const MatrixXd& Y_comp,
                                    const List& Pi_minus_1,
                                    const MatrixXd& h_p,
                                    const VectorXi& nlevels,
                                    const VectorXi& nxcond,
                                    const List& xcond,
                                    const MatrixXd& direct,
                                    const MatrixXd& diag_nclass,
                                    const int& alpha_length,
                                    const int& num_alpha0,
                                    const int& npeop,
                                    const int& nitem,
                                    const int& nclass,
                                    double& c,
                                    const double& err,
                                    const int& len) {
  
  
  
  
  
  
  List temp=Score_hess_alpha_cpp(Pi_minus_1,
                                 Y,
                                 h_p,
                                 npeop,
                                 nitem,
                                 nclass,
                                 nlevels,
                                 nxcond,
                                 xcond,
                                 direct,
                                 diag_nclass,
                                 alpha_length);
  
  
  
  VectorXd direction=temp["direction"];
  
  
  // NumericVector temp_aa(20);
  // NumericVector temp_cc(20);
  
  double a = 0.0;  
  int iter = 0;
  for( iter = 0; iter < 1000; ++iter) {
    // double temp_a = a;
    // double temp_c = c;
    
    
    double temp_b = (a + c) / 2.0;
    
    
    double temp_a1 = f2_rcpp(alpha_vector + temp_b * direction,
                             Y_comp,xcond,nlevels,
                             nxcond,h_p,
                             num_alpha0,npeop,
                             nitem,nclass);
    
    double temp_a2 = f2_rcpp(alpha_vector + (temp_b + 0.00000001) * direction,
                             Y_comp,xcond,nlevels,
                             nxcond,h_p,
                             num_alpha0,npeop,
                             nitem,nclass);
    
    // double temp_a1 = as<double>(f2(alpha_vector + temp_b * direction));
    // double temp_a2 = as<double>(f2(alpha_vector + (temp_b + 0.05) * direction));
    
    if(temp_a1 < temp_a2) {
      a = temp_b;
      // c = c;
      
      // temp_aa[iter]=a;
      // temp_cc[iter]=c;
      
      // if( abs(temp_a - a) < 0.001 ) break;
    } else {
      // a = a;
      c = temp_b;
      
      // temp_aa[iter]=a;
      // temp_cc[iter]=c;
      
      // if( abs(temp_c - c) < 0.001 ) break;
    }
    
    if( (c-a) < err ) break;
    // if( std::max( abs(temp_a - a),abs(temp_c - c) ) < 0.001 ) break;
  }
  
  
  
  // int len = static_cast<int>((c - a) / err) + 1;
  
  double step = (c - a) / (len-1);
  
  NumericVector likeli(len+2);
  NumericVector ak(len+2);
  List alpha_vector_list(len+2);
  
  
  
  ak[0]=0;
  alpha_vector_list[0]=  alpha_vector;
  likeli[0] =   f2_rcpp(alpha_vector_list[0],
                        Y_comp,xcond,nlevels,
                        nxcond,h_p,
                        num_alpha0,npeop,
                        nitem,nclass);
  
  
  int max_idx=0;
  for(int i = 0; i < len; ++i) {
    ak[i+1]=a + i * step;
    alpha_vector_list[i+1]=alpha_vector + ak[i+1] * direction;
    likeli[i+1] = f2_rcpp(alpha_vector_list[i+1],
                          Y_comp,xcond,nlevels,
                          nxcond,h_p,
                          num_alpha0,npeop,
                          nitem,nclass);
    
    if(likeli[i+1]>=likeli[max_idx]){
      max_idx=i+1;
    }
    
    
  }
  
  
  ak[len+1]=1;
  alpha_vector_list[len+1]=alpha_vector+direction;
  likeli[len+1] = f2_rcpp(alpha_vector_list[len+1],
                          Y_comp,xcond,nlevels,
                          nxcond,h_p,
                          num_alpha0,npeop,
                          nitem,nclass);
  
  if(likeli[len+1]>=likeli[max_idx]){
    max_idx=len+1;
  }
  
  
  // ak[len+1]=c;
  // 
  // alpha_vector_list[len+1]=alpha_vector + c * direction;
  // 
  // likeli[len+1] = f2_rcpp(alpha_vector_list[len+1],
  //                         Y_comp,xcond,nlevels,
  //                         nxcond,h_p,
  //                         num_alpha0,npeop,
  //                         nitem,nclass);
  // 
  // if(likeli[len+1]>=likeli[max_idx]){
  //   max_idx=len+1;
  // }
  // 
  // 
  // ak[len+2]=1;
  // alpha_vector_list[len+2]=alpha_vector+direction;
  // likeli[len+2] = f2_rcpp(alpha_vector_list[len+2],
  //                         Y_comp,xcond,nlevels,
  //                         nxcond,h_p,
  //                         num_alpha0,npeop,
  //                         nitem,nclass);
  // 
  // if(likeli[len+2]>=likeli[max_idx]){
  //   max_idx=len+2;
  // }
  
  
  
  
  
  // double max_tempp = Rcpp::max(likeli);
  // NumericVector alpha_vector_p(alpha_vector.size());
  // 
  // int max_idx = Rcpp::which_max(likeli);
  // 
  // if(max_tempp < a2) {
  //   alpha_vector_p = alpha_vector;
  // } else {
  //   
  //   alpha_vector_p = alpha_vector + ak[max_idx] * direction;
  // }
  
  
  
  // int max_idx = Rcpp::which_max(likeli);
  
  
  // NumericVector alpha_vector_p(alpha_vector.size());
  VectorXd alpha_vector_p=alpha_vector_list[max_idx];
  
  // if(max_tempp >= a2) {
  //   alpha_vector_p = alpha_vector + ak[max_idx] * direction;
  // } else{
  //   alpha_vector_p = alpha_vector;
  // }
  
  // NumericVector alpha_vector_p=alpha_vector;
  double ak_temp=ak[max_idx];
  
  // 创建一个 List 对象包含多个值
  List result = List::create(
    Named("a") = a,
    Named("c") = c,
    Named("likeli") = likeli,
    Named("len") = len,
    Named("ak") = ak,
    Named("ak_temp") = ak_temp,
    Named("alpha_vector_p") = alpha_vector_p,
    Named("max_idx") = max_idx+1,
    Named("direction") = direction,
    Named("llik") = likeli[max_idx],
    Named("iter") = iter);
  
  
  return result;
}



// [[Rcpp::export]]
// 用backtracking algorithm
List max_beta_1_cpp(MatrixXd& beta,
                    const MatrixXd& Xprev,
                    MatrixXd& eta_p,
                    const MatrixXd& h_p,
                    const int& npeop,
                    const int& nxprev,
                    const int& nclass,
                    const int& maxiter,
                    const double& step_length,
                    const double& tol) {
  
  
  
  // MatrixXd eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
  VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
  MatrixXd hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
  MatrixXd ginv_hess = cod.pseudoInverse();
  // MatrixXd ginv_hess = hess.inverse();
  
  
  MatrixXd direction = -1*ginv_hess * deri;
  // MatrixXd dir=direction;
  
  direction.resize(nxprev+1 ,nclass-1);
  double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  double llik_p;
  
  MatrixXd beta_p;
  double a=1.0;
  
  int iter = 0;
  for(iter = 0; iter < maxiter; ++iter){
    
    
    beta_p=beta+a*direction;
    llik_p= beta_likeli_cpp(beta_p,h_p,Xprev,nxprev,nclass);
    
    // max_beta_score_bisection_1_cpp(beta,Xprev,eta_p,h_p,
    //                                     npeop,nxprev,nclass,err);
    
    
    // double max_diff = (beta_p - beta).cwiseAbs().maxCoeff();
    // Rcpp::Rcout << "Iteration " << iter << ", Max Difference: " << max_diff << std::endl;
    // beta=beta_p;
    
    
    
    if(llik_p>=llik){
      // Rcpp::Rcout << "Iteration " << iter << ", llik: " << llik_p << std::endl;
      beta=beta_p;
      llik= llik_p;
      if((a*direction).cwiseAbs().maxCoeff()<tol) break;
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
      
      eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
      deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
      hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
      ginv_hess = cod.pseudoInverse();
      // MatrixXd ginv_hess = hess.inverse();
      
      
      direction = -1*ginv_hess * deri;
      // MatrixXd dir=direction;
      
      direction.resize(nxprev+1 ,nclass-1);
      
      
    }else{
      a=a*step_length;
    }
    
    
    // if(max_diff < tol){
    //   break;
    // }else{
    //   eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
    // }
    
    
  }
  
  
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  // 如果達到最大迭代次數，可能有誤
  // 只適用於驗證，可以不要算llik
  // llik_p= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  List result = List::create(
    Named("beta_p") = beta,
    Named("llik") = llik,
    Named("iter") = iter);
  
  return result;
}



// [[Rcpp::export]]
// 用backtracking algorithm
List max_beta_2_cpp(MatrixXd& beta,
                    const MatrixXd& Xprev,
                    MatrixXd& eta_p,
                    const MatrixXd& h_p,
                    const int& npeop,
                    const int& nxprev,
                    const int& nclass,
                    const int& maxiter,
                    const double& step_length,
                    const double& tol) {
  
  
  MatrixXd eta=eta_p;
  // MatrixXd eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
  VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
  MatrixXd hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
  MatrixXd ginv_hess = cod.pseudoInverse();
  // MatrixXd ginv_hess = hess.inverse();
  
  
  MatrixXd direction = -1*ginv_hess * deri;
  // MatrixXd dir=direction;
  
  direction.resize(nxprev+1 ,nclass-1);
  
  // MatrixXd eta=eta_cpp(beta,Xprev,nxprev,nclass);
  
  
  // MatrixXd result_temp=h_p.array()*eta_p.array().log();
  // 
  // // 計算結果總和
  // double llik=result_temp.sum();
  
  
  double llik=(h_p.array()*eta_p.array().log()).sum();
  
  
  
  // return result;
  // 
  // 
  // 
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  double llik_p;
  
  MatrixXd beta_p;
  double a=1.0;
  
  int iter = 0;
  for(iter = 0; iter < maxiter; ++iter){
    
    
    beta_p=beta+a*direction;
    eta_p=eta_cpp(beta_p,Xprev,nxprev,nclass);
    llik_p=(h_p.array()*eta_p.array().log()).sum();
    
    
    // llik_p= beta_likeli_cpp(beta_p,h_p,Xprev,nxprev,nclass);
    
    // max_beta_score_bisection_1_cpp(beta,Xprev,eta_p,h_p,
    //                                     npeop,nxprev,nclass,err);
    
    
    // double max_diff = (beta_p - beta).cwiseAbs().maxCoeff();
    // Rcpp::Rcout << "Iteration " << iter << ", Max Difference: " << max_diff << std::endl;
    // beta=beta_p;
    
    
    
    if(llik_p>=llik){
      // Rcpp::Rcout << "Iteration " << iter << ", llik: " << llik_p << std::endl;
      beta=beta_p;
      llik= llik_p;
      eta=eta_p;
      if((a*direction).cwiseAbs().maxCoeff()<tol) break;
      
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
      deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
      hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
      
      // 計算ginv
      CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
      ginv_hess = cod.pseudoInverse();
      // MatrixXd ginv_hess = hess.inverse();
      
      
      direction = -1*ginv_hess * deri;
      // MatrixXd dir=direction;
      
      direction.resize(nxprev+1 ,nclass-1);
      
      
    }else{
      a=a*step_length;
    }
    
    
    // if(max_diff < tol){
    //   break;
    // }else{
    //   eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
    // }
    
    
  }
  
  
  // double llik= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  // 如果達到最大迭代次數，可能有誤
  // 只適用於驗證，可以不要算llik
  // llik_p= beta_likeli_cpp(beta,h_p,Xprev,nxprev,nclass);
  
  List result = List::create(
    Named("beta_p") = beta,
    Named("eta") = eta,
    Named("llik") = llik,
    Named("iter") = iter);
  
  return result;
}

// [[Rcpp::export]]
List max_beta_score_bisection_cpp(const MatrixXd& beta,
                                  const MatrixXd& Xprev,
                                  const MatrixXd& eta_p,
                                  const MatrixXd& h_p,
                                  const int& npeop,
                                  const int& nxprev,
                                  const int& nclass,
                                  double& c,
                                  const double& err,
                                  const int& len) {
  
  
  // MatrixXd eta_p = eta_cpp(beta,Xprev,nxprev,nclass);
  VectorXd deri = g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
  MatrixXd hess = Hess1_cpp(eta_p,Xprev,npeop,nxprev,nclass);
  
  // 計算ginv
  CompleteOrthogonalDecomposition<MatrixXd> cod(hess);
  MatrixXd ginv_hess = cod.pseudoInverse();
  // MatrixXd ginv_hess = hess.inverse();
  
  
  MatrixXd direction = -1*ginv_hess * deri;
  MatrixXd dir=direction;
  
  direction.resize(nxprev+1 ,nclass-1);
  
  
  
  // double temp_a;
  // double temp_c;
  double temp_b;
  double temp_a1;
  VectorXd g1;
  
  
  double a = 0.0;
  int iter = 0;
  for( iter = 0; iter < 1000; ++iter) {
    // temp_a = a;
    // temp_c = c;
    
    temp_b = (a + c) / 2.0;
    // temp_b = a+(c-a) / 2.0;
    
    MatrixXd eta_p = eta_cpp(beta + temp_b * direction,Xprev,nxprev,nclass);
    g1=g1_cpp(eta_p,h_p,Xprev,npeop,nxprev,nclass);
    temp_a1 = (dir.array()*g1.array()).sum();
    
    
    //   beta_likeli_cpp( beta + temp_b * direction,
    //                            h_p,
    //                            Xprev,
    //                            nxprev,
    //                            nclass);
    // 
    // 
    // temp_a2 = beta_likeli_cpp( beta + (temp_b + 0.00000001) * direction,
    //                            h_p,
    //                            Xprev,
    //                            nxprev,
    //                            nclass);
    
    
    if(temp_a1 > 0) {
      a = temp_b;
      // c = c;
      
      
      // temp_aa[iter]=a;
      // temp_cc[iter]=c;
      
      // if( abs(temp_a - a) < err ) break;
    } else {
      // a = a;
      c = temp_b;
      
      
      // temp_aa[iter]=a;
      // temp_cc[iter]=c;
      
      // if( abs(temp_c - c) < err ) break;
    }
    
    if( (c-a) < err ) break;
    // if( g1.cwiseAbs().maxCoeff() < err ) break;
    // if( std::max( abs(temp_a - a),abs(temp_c - c) ) < 0.001 ) break;
  }
  
  
  
  
  // int len = static_cast<int>((c - a) / err) + 1;
  // int len = 10;
  double step = (c - a) / (len-1);
  
  
  // NumericVector likeli(len+2);
  // NumericVector ak(len+2);
  // List beta_list(len+2);
  // ak[0]=0;
  // beta_list[0]=beta;
  // likeli[0] = beta_likeli_cpp( beta_list[0],
  //                              h_p,
  //                              Xprev,
  //                              nxprev,
  //                              nclass);
  // 
  // 
  // 
  // 
  // int max_idx=0;
  // 
  // for(int i = 0; i < len; ++i) {
  //   ak[i+1]=a + i * step;
  //   beta_list[i+1]=beta + ak[i+1]* direction;
  //   likeli[i+1] = beta_likeli_cpp( beta_list[i+1],
  //                                  h_p,
  //                                  Xprev,
  //                                  nxprev,
  //                                  nclass);
  //   
  //   if( likeli[i+1] >= likeli[max_idx] ){
  //     max_idx=i+1;
  //   }
  //   
  // }
  // 
  // ak[len+1]=1;
  // beta_list[len+1]=beta+direction;
  // likeli[len+1] = beta_likeli_cpp( beta_list[len+1],
  //                                  h_p,
  //                                  Xprev,
  //                                  nxprev,
  //                                  nclass);
  // 
  // if( likeli[len+1] >= likeli[max_idx] ){
  //   max_idx=len+1;
  // }
  
  
  
  NumericVector likeli(len);
  NumericVector ak(len);
  List beta_list(len);
  
  int max_idx=0;
  
  for(int i = 0; i < len; ++i) {
    ak[i]=a + i * step;
    beta_list[i]=beta + ak[i]* direction;
    likeli[i] = beta_likeli_cpp( beta_list[i],
                                 h_p,
                                 Xprev,
                                 nxprev,
                                 nclass);
    
    if( likeli[i] >= likeli[max_idx] ){
      max_idx=i;
    }
    
  }
  
  
  
  
  
  // double max_tempp = likeli.maxCoeff();
  
  
  // int max_idx = Rcpp::which_max(likeli);
  
  MatrixXd beta_p=beta_list[max_idx];
  
  
  // NumericVector alpha_vector_p=alpha_vector;
  double ak_temp=ak[max_idx];
  
  // 创建一个 List 对象包含多个值
  List result = List::create(
    Named("a") = a,
    Named("c") = c,
    Named("likeli") = likeli,
    Named("len") = len,
    Named("ak") = ak,
    Named("ak_temp") = ak_temp,
    Named("beta_p") = beta_p,
    Named("max_idx") = max_idx+1,
    Named("direction") = direction,
    Named("llik") = likeli[max_idx],
    Named("iter") = iter);
  
  
  return result;
}


    
    
    
    
    
    
// [[Rcpp::export]]
MatrixXd EM_cpp(MatrixXd& beta_p,
                List& alpha0_p,
                List& alpha_p,
                const MatrixXd& Y_comp,
                const List& xcond,
                const VectorXi& nxcond,
                const int& nitem,
                const int& nclass,
                const MatrixXd& Xprev,
                const int& npeop,
                const int& nxprev,
                const int& maxiter,
                const double& step_length,
                const double& tol,
                const int& it){
  
  
  List cond_prob_and_Pi_minus_1=
    conditional_prob_and_Pi_cpp(alpha0_p,alpha_p,Y_comp,
                                xcond,nxcond,npeop,nitem,nclass);
  
  
  MatrixXd cond_prob_p=cond_prob_and_Pi_minus_1["cond_prob"];
  
  List Pi_minus_1=cond_prob_and_Pi_minus_1["Pi_minus_1"];
  MatrixXd eta_p = eta_cpp(beta_p,Xprev,nxprev,nclass);
  
  for( int k = 0; k < maxiter; ++k) {
    
    MatrixXd beta_temp_p=beta_p;
    MatrixXd h_p=h_p_cpp(eta_p,cond_prob_p);


    List maxbeta=max_beta_6_cpp(beta_p,Xprev,eta_p,h_p,npeop,nxprev,nclass,
                       maxiter,step_length,
                       tol,it);
    
    MatrixXd beta_p=maxbeta["beta_p"];
    MatrixXd eta_p =maxbeta["eta"];
    // 
    // alpha_vector=alpha_vector_p;
    // 
    // 
    // temp=max_alpha_5_cpp(alpha_vector,
    //                      alpha0_p,alpha_p,
    //                      Y,Y_comp,
    //                      Pi_minus_1,
    //                      cond_prob_p,
    //                      h_p,
    //                      nlevels,
    //                      nxcond,
    //                      xcond,
    //                      direct,
    //                      diag_nclass,
    //                      alpha_length,
    //                      num_alpha0,
    //                      npeop,
    //                      nitem,
    //                      nclass,
    //                      maxiter,
    //                      step_length=0.5,
    //                      tol=tol_alpha,it=it_alpha);
    //   
    //   
    //   
    //   
    //   alpha_vector_p=temp["alpha_vector_p"];
    //   alpha0_p=temp["alpha0"];
    //   alpha_p =temp["alpha"];
    //   
    //   
    //   Pi_minus_1=temp["Pi_minus_1"];
    //   MatrixXd cond_prob_p=temp["cond_prob"];
    //   
    // 
    //   
      MatrixXd likeli_temp=cond_prob_p.array()*eta_p.array();

      VectorXd rowSumsTemp = likeli_temp.rowwise().sum();
      double likeli=rowSumsTemp.array().log().sum();
  }
  
  
  return eta_p;
  
}    
    
    
    
    
    
    
    
    
// // [[Rcpp::export]]
// std::vector<MatrixXd> Estimation_omp_cpp(std::vector<MatrixXd> beta_list,
//                                          List& alpha0_p,
//                                          List& alpha_p,
//                                          const int& npeop,
//                                          const int& nitem,
//                                          const int nclass,
//                                          const MatrixXd& Xprev,
//                                          const int nxprev,
//                                          const MatrixXd& Y,
//                                          const MatrixXd& Y_comp,
//                                          const VectorXi& nlevels,
//                                          const VectorXi& nxcond,
//                                          const List& xcond,
//                                          const int& maxiter,
//                                          const int& maxiter_para,
//                                          const double& step_length,
//                                          const double& tol,
//                                          const int& it) {
//   
//   int nsets = beta_list.size();
//   std::vector<MatrixXd> result(nsets);
//   
//   
//   
//   // #pragma omp parallel for num_threads(ncores)
// #pragma omp for schedule(dynamic)  
//   for (int idx = 0; idx < nsets; ++idx) {
//     
//     
//     // const MatrixXd& beta = betas[i];
//     auto& beta_p=beta_list[idx];
//     
//     
//     VectorXd log_lik= VectorXd::Zero(maxiter+1);
//     log_lik[0] = R_NegInf;
//     
//     
//     List cond_prob_and_Pi_minus_1=
//       conditional_prob_and_Pi_cpp(alpha0_p,alpha_p,Y_comp,
//                                   xcond,nxcond,npeop,nitem,nclass);
//     
//     MatrixXd cond_prob_p=cond_prob_and_Pi_minus_1["cond_prob"];
//     
//     List Pi_minus_1=cond_prob_and_Pi_minus_1["Pi_minus_1"];
//     
//     
//     MatrixXd eta_p=eta_cpp(beta_p,Xprev,nxprev,nclass);
//     int iter=0;
//     for(iter = 0; iter < maxiter; ++iter){
//       MatrixXd beta_temp_p=beta_p;
//       MatrixXd h_p= h_p_cpp(eta_p,cond_prob_p);
//       
//       List maxbeta=max_beta_6_cpp(beta_p,Xprev,eta_p,h_p,npeop,nxprev,nclass,
//                                   maxiter_para,step_length,tol,it);
//       
//       beta_p=maxbeta["beta_p"];
//       
//       eta_p =maxbeta["eta"];
//     }
//     if(iter==maxiter) iter=iter-1;
//     
//     result[idx]=eta_p;
//   }
//   
//   
//   
//   return result;
// }
    
    
   
// [[Rcpp::export]]
List Estimation_omp_cpp(std::vector<MatrixXd> beta_list,
                        const std::vector<VectorXd>& alpha_vector_list,
                        const List& alpha0_list,
                        const List& alpha_list,
                        const int& npeop,
                        const int& nitem,
                        const int nclass,
                        const MatrixXd& Xprev,
                        const int nxprev,
                        const MatrixXd& Y,
                        const MatrixXd& Y_comp,
                        const VectorXi& nlevels,
                        const VectorXi& nxcond,
                        const List& xcond,
                        const MatrixXd& direct,
                        const MatrixXd& diag_nclass,
                        const int& alpha_length,
                        const int& num_alpha0,
                        const int& maxiter,
                        const int& maxiter_para,
                        const double& step_length,
                        const double& tol_beta,
                        const double& tol_alpha,
                        const int& it_beta,
                        const int& it_alpha,
                        const double& tol_para,
                        const double& tol_likeli,
                        const int ncores) {
  
  int nsets = beta_list.size();
  std::vector<MatrixXd> beta_p_list(nsets);
  std::vector<MatrixXd> eta_list(nsets);
  std::vector<double> iter_list(nsets);
  std::vector<double> log_lik_list(nsets);
  
  std::vector<MatrixXd> cond_prob_p_list(nsets);
  std::vector<std::vector<MatrixXd>> Pi_minus_1_list(nsets);
  
  std::vector<VectorXd> alpha_vector_p_list(nsets);
  
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
    
    VectorXd log_lik= VectorXd::Zero(maxiter+1);
    log_lik[0] = R_NegInf;
    
    
    auto cond_prob_and_Pi_minus_1=
      conditional_prob_and_Pi_for_omp_cpp(alpha0_p,
                                          alpha_p,
                                          Y_comp,
                                          xcond_vec,
                                          nxcond,
                                          npeop,
                                          nitem,
                                          nclass);

    MatrixXd cond_prob_p=cond_prob_and_Pi_minus_1[0][0];

    std::vector<MatrixXd> Pi_minus_1=cond_prob_and_Pi_minus_1[1];
    
    
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
      h_p.array().colwise() /= h_p.rowwise().sum().array();// 歸一化
      
      
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

      eta_p =maxbeta[1];

      // VectorXd alpha_vector=alpha_vector_p;
      alpha_vector.noalias()=alpha_vector_p;
      
      auto temp=max_alpha_5_for_omp_cpp(alpha_vector_p,
                                        alpha0_p,
                                        alpha_p,
                                        Y,
                                        Y_comp,
                                        Pi_minus_1,
                                        cond_prob_p,
                                        h_p,
                                        nlevels,
                                        nxcond,
                                        xcond_vec,
                                        direct,
                                        diag_nclass,
                                        alpha_length,
                                        num_alpha0,
                                        npeop,
                                        nitem,
                                        nclass,
                                        maxiter_para,
                                        step_length,
                                        tol_alpha,
                                        it_alpha);
      
      // VectorXd al = temp[0][0].col(0);
      // alpha_vector_p = al;
      alpha_vector_p = temp[0][0].col(0);
      alpha0_p=temp[1];
      alpha_p =temp[2];
      Pi_minus_1=temp[3];
      cond_prob_p=temp[4][0];
      
      
      
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
     
    iter_list[idx]=iter+1;
    cond_prob_p_list[idx]=cond_prob_p;
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
    Named("cond_prob_p_list") = cond_prob_p_list,
    Named("Pi_minus_1_list") = Pi_minus_1_list,
    Named("log_lik_list") = log_lik_list
    
  );
  
}
    
    

    
    
// [[Rcpp::export]]
List Estimation_no_beta_covaruate_omp_cpp(std::vector<MatrixXd> beta_list,
                                          const std::vector<VectorXd>& alpha_vector_list,
                                          const List& alpha0_list,
                                          const List& alpha_list,
                                          const int& npeop,
                                          const int& nitem,
                                          const int nclass,
                                          const MatrixXd& Xprev,
                                          const int nxprev,
                                          const MatrixXd& Y,
                                          const MatrixXd& Y_comp,
                                          const VectorXi& nlevels,
                                          const VectorXi& nxcond,
                                          const List& xcond,
                                          const MatrixXd& direct,
                                          const MatrixXd& diag_nclass,
                                          const int& alpha_length,
                                          const int& num_alpha0,
                                          const int& maxiter,
                                          const int& maxiter_para,
                                          const double& step_length,
                                          const double& tol_beta,
                                          const double& tol_alpha,
                                          const int& it_beta,
                                          const int& it_alpha,
                                          const double& tol_para,
                                          const double& tol_likeli,
                                          const int ncores) {
  
  int nsets = beta_list.size();
  std::vector<MatrixXd> beta_p_list(nsets);
  std::vector<MatrixXd> eta_list(nsets);
  std::vector<double> iter_list(nsets);
  std::vector<double> log_lik_list(nsets);
  
  std::vector<MatrixXd> cond_prob_p_list(nsets);
  std::vector<std::vector<MatrixXd>> Pi_minus_1_list(nsets);
  
  std::vector<VectorXd> alpha_vector_p_list(nsets);
  
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
    
    VectorXd log_lik= VectorXd::Zero(maxiter+1);
    log_lik[0] = R_NegInf;
    
    
    auto cond_prob_and_Pi_minus_1=
      conditional_prob_and_Pi_for_omp_cpp(alpha0_p,
                                          alpha_p,
                                          Y_comp,
                                          xcond_vec,
                                          nxcond,
                                          npeop,
                                          nitem,
                                          nclass);
    
    MatrixXd cond_prob_p=cond_prob_and_Pi_minus_1[0][0];
    
    std::vector<MatrixXd> Pi_minus_1=cond_prob_and_Pi_minus_1[1];
    
    
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
      
      
      auto maxbeta=max_beta_6_no_covariate_for_omp_cpp(beta_p,
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
      
      eta_p =maxbeta[1];
      
      // VectorXd alpha_vector=alpha_vector_p;
      alpha_vector.noalias()=alpha_vector_p;
      
      auto temp=max_alpha_5_for_omp_cpp(alpha_vector_p,
                                        alpha0_p,
                                        alpha_p,
                                        Y,
                                        Y_comp,
                                        Pi_minus_1,
                                        cond_prob_p,
                                        h_p,
                                        nlevels,
                                        nxcond,
                                        xcond_vec,
                                        direct,
                                        diag_nclass,
                                        alpha_length,
                                        num_alpha0,
                                        npeop,
                                        nitem,
                                        nclass,
                                        maxiter_para,
                                        step_length,
                                        tol_alpha,
                                        it_alpha);
      
      // VectorXd al = temp[0][0].col(0);
      // alpha_vector_p = al;
      alpha_vector_p = temp[0][0].col(0);
      alpha0_p=temp[1];
      alpha_p =temp[2];
      Pi_minus_1=temp[3];
      cond_prob_p=temp[4][0];
      
      
      
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
    
    iter_list[idx]=iter+1;
    cond_prob_p_list[idx]=cond_prob_p;
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
    Named("cond_prob_p_list") = cond_prob_p_list,
    Named("Pi_minus_1_list") = Pi_minus_1_list,
    Named("log_lik_list") = log_lik_list);
  
}

    
    
    
    
    
    
    
