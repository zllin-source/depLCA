
rlca=function(y=y,
                xprev=NULL,
                xcond=vector("list",ncol(y)),
                nlevels=rep(2,ncol(y)),
                nclass=2,
                beta_p=NULL,
                alpha0_p=NULL,
                alpha_p=NULL,
                maxiter=2000,
                it_beta=1,
                it_alpha=1,
                tol_beta=1e-20,
                tol_alpha=1e-20,
                tol_para=1e-8,
                tol_likeli = 1e-16,
                nrep=10,
                ncores=parallel::detectCores()-2,
                verbose=F,
                seed=NULL){
  
  starttime <- Sys.time()
  # if (nclass<=1){
  #   stop("\n Error: not allowed when nclass<=1; will be ignored. \n \n")
  #   # ret <- NULL
  # }
  
  nitem=ncol(y)
  
  
  # if(length(xcond)!= nitem|
  #    !mode(xcond)=="list"){
  #   stop("\n Error: xcond  format error \n \n")
  #   # cat("\n ALERT: xcond  format error \n \n")
  #   # ret <- NULL
  # }
  
  if(!mode(xcond)=="list"){
    stop("\n Error: xcond  format error \n \n")
    # cat("\n ALERT: xcond  format error \n \n")
    # ret <- NULL
  }
  
  npeop=nrow(y)
  
  
  # if(is.null(xcond)){
  #   vector("list",ncol(y))
  # } 
  
  ###將y轉成multinomial######
  e_comp=NULL
  # nlevels=c(2,2,2,2,2)
  Y=NULL
  Y_comp=NULL
  nxcond=rep(0,nitem)
  H=prod(nlevels)
  # e為所有可能結果
  e=matrix(0,H,nitem)
  
  for(m in 1:nitem){
    #利用循環數
    e[,m]=rep(1:nlevels[m], each = prod(nlevels[-c(1:m)]), len = H)
    
    
    temp=diag(rep(1,nlevels[m]))[ y[,m],]
    
    Y_comp=cbind(Y_comp,temp)
    temp=temp[,-nlevels[m]]
    Y=cbind(Y,temp)
    
    temp_e=diag(rep(1,nlevels[m]))[  e[,m], ]
    e_comp=cbind(e_comp,temp_e)
    
    
    if(is.null(xcond[[m]])) next
    nxcond[m]=ncol(xcond[[m]])
  }
  
  
  
  
  
  outcome=((y-1))%*%rev(cumprod(rev(c(nlevels[-1],1))))+1
  #O為(O_1,...,O_H)表示觀測次數able(c(outcome,1:H))-1
  O=table(c(outcome,1:H))-1
  
  outcome_table=cbind(e,O)
  colnames(outcome_table)[nitem+1]="count" 
  
  ret <- list()
  ret$observed=O
  
  
  
  
  
  
  if(is.null(xprev)){
    nxprev=0
    Xprev=matrix(1,npeop,1)
  }else{
    nxprev=ncol(xprev)
    Xprev=cbind(1,xprev)
  }
  
  
  # if(nclass>1){
  #   max_beta_fun_cpp=max_beta_6_cpp
  # }else{
  #   max_beta_fun_cpp=function() {
  #     # 函數內部什麼也不做
  #   }
  # }
  if(nxprev==0){
    max_beta_cpp=max_beta_6_no_covariate_cpp
  }else{
    max_beta_cpp=max_beta_6_cpp
    # if(parallel==T) max_beta_cpp=max_beta_7_cpp
    
  }
  
  
  diag_nclass=diag(nclass)
  direct=calculateDirectSum(nitem,nlevels)
  
  # ret <- list()
  
  
  
  num_alpha0=sum(nclass*(nlevels- 1))
  num_alpha=sum(nxcond*(nlevels - 1))
  alpha_length=num_alpha0+num_alpha
  
  
  if(is.null(beta_p)&is.null(alpha0_p)&is.null(alpha_p)){
    ret$log_lik=-Inf
    # beta_nrep=NULL
    # alpha0_nrep=NULL
    # alpha_nrep=NULL
    log_lik_list=rep(0,nrep)
    set.seed(seed)
    for(repl in 1:nrep){
      
      starttime_model <- Sys.time()
      
      beta0_p=runif(nclass)
      beta0_p=beta0_p/sum(beta0_p)
      beta0_p=log(beta0_p/(1-beta0_p))[-nclass]
      beta_p=matrix(0,nxprev,(nclass-1))
      beta_p=rbind(beta0_p,beta_p)
      beta0init=beta0_p
      
      
      
      alpha0_p=NULL
      alpha_p=NULL
      
      for(m in 1:nitem){
        # use for binary case
        # temp=runif(nclass)
        # a=list(matrix(log(temp/(1-temp)),
        #               nclass,(nlevels[m] - 1)))
        
        temp=matrix(runif(nclass*(nlevels[m])),nclass,(nlevels[m] ))
        temp=(temp/rowSums(temp))[,-nlevels[m]]
        
        a=list(matrix(log(temp/(1-temp)),
                      nclass,(nlevels[m] - 1)))
        
        
        
        alpha0_p=c(alpha0_p,a)
        
        # a=list(matrix(runif(10,-1,1),nxcond[m],(nlevels[m] - 1)))
        a=list(matrix(0,nxcond[m],(nlevels[m] - 1)))
        alpha_p=c(alpha_p,a)
      }
      
      alpha0init=alpha0_p
      # alphainit=alpha_p
      alpha_vector_p=c(unlist(alpha0_p),unlist(alpha_p))
      
      log_lik=c()
      log_lik[1]=-Inf
      
      cond_prob_and_Pi_minus_1=
        conditional_prob_and_Pi_cpp(alpha0_p,alpha_p,Y_comp,
                                    xcond,nxcond,npeop,nitem,nclass)
      
      
      cond_prob_p=cond_prob_and_Pi_minus_1$cond_prob
      
      Pi_minus_1=cond_prob_and_Pi_minus_1$Pi_minus_1
      eta_p = eta_cpp(beta_p,Xprev,nxprev,nclass)
      
      for(k in 1:maxiter){
        
        # cond_prob_and_Pi_minus_1=
        #   conditional_prob_and_Pi_cpp(alpha0_p,alpha_p,Y_comp,
        #                               xcond,nxcond,npeop,nitem,nclass)
        # 
        # cond_prob_p=cond_prob_and_Pi_minus_1$cond_prob
        # 
        # Pi_minus_1=cond_prob_and_Pi_minus_1$Pi_minus_1
        
        
        beta_temp_p=beta_p
        
        
        # eta_p = eta_cpp(beta_p,Xprev,nxprev,nclass)
        # h_p= h_p_cpp(eta_p,cond_prob_p,Xprev,nxprev,nclass)
        h_p= h_p_cpp(eta_p,cond_prob_p)
        
        # beta_p=max_beta_1_cpp(beta_p,Xprev,eta_p,h_p,npeop,nxprev,nclass,maxiter,
        #                       0.5,tol=1e-5)$beta_p
        
        
        # beta_p=max_beta_score_bisection_cpp(beta_p,Xprev,eta_p,h_p,
        #                                     npeop,nxprev,nclass,
        #                                     c=2,err=1e-6,len=10)$beta_p
        # eta_p = eta_cpp(beta_p,Xprev,nxprev,nclass)
        
        # if(nxprev==0){
        #   # eta_p=colMeans(h_p)
        #   # beta_p=matrix(log(eta_p[1:(nclass-1)]/(1-eta_p[1:(nclass-1)])),nrow = 1)
        #   # eta_p = matrix(eta_p,npeop,nclass,byrow = T)
        #   maxbeta=calculate_eta_beta_vec(h_p,nclass, npeop)
        #   beta_p=maxbeta$beta_p
        #   eta_p=maxbeta$eta_p_matrix
        # }else{
        #   maxbeta=max_beta_cpp(beta_p,Xprev,eta_p,h_p,npeop,nxprev,nclass,
        #                        maxiter,step_length=0.5,
        #                        tol=tol_beta,it=it_beta)
        #   beta_p=maxbeta$beta_p
        #   eta_p =maxbeta$eta
        # }
        
        maxbeta=max_beta_cpp(beta_p,Xprev,eta_p,h_p,npeop,nxprev,nclass,
                             maxiter,step_length=0.5,
                             tol=tol_beta,it=it_beta)
        beta_p=maxbeta$beta_p
        eta_p =maxbeta$eta
        
        
        # maxbeta=max_beta_6_cpp(beta_p,Xprev,eta_p,h_p,npeop,nxprev,nclass,
        #                        maxiter,step_length=0.5,
        #                        tol=tol_beta,it=it_beta)
        # beta_p=maxbeta$beta_p
        # eta_p =maxbeta$eta
        
        
        
        # beta_p=modify_max_beta_bisection_cpp(beta_p,Xprev,eta_p,h_p,
        #                               npeop,nxprev,nclass,
        #                               c=1.2,err=1e-6,len=10)$beta_p
        
        # beta_p=modify_max_beta_bisection_cpp(beta_p,Xprev,eta_p,h_p,
        #                                      npeop,nxprev,nclass,
        #                                      c=2,err=1e-6,len=10)$beta_p
        
        
        
        
        # beta_p=max_beta_bisection_cpp(beta_p,Xprev,eta_p,h_p,
        #                               npeop,nxprev,nclass,
        #                               c=1.2,err=0.01)$beta_p
        
        # beta_p=matrix(optim( c(beta_p), fn=f1 ,gr=g1,
        #                        control = list(fnscale = -1,factr=1e-20,
        #                                       maxit=1e6))$par,
        #         (nxprev+1),(nclass-1) )
        
        
        alpha_vector=alpha_vector_p
        
        
        # alpha_vector_p=max_alpha_bisection_cpp(alpha_vector,
        #                                        Y,
        #                                        Y_comp,
        #                                        Pi_minus_1,
        #                                        h_p,
        #                                        nlevels,
        #                                        nxcond,
        #                                        xcond,
        #                                        direct,
        #                                        diag_nclass,
        #                                        alpha_length,
        #                                        num_alpha0,
        #                                        npeop,
        #                                        nitem,
        #                                        nclass,
        #                                        c=1.2,
        #                                        err=0.01)$alpha_vector_p
        
        
        # alpha_vector_p=modify_max_alpha_bisection_cpp(alpha_vector,
        #                                        Y,
        #                                        Y_comp,
        #                                        Pi_minus_1,
        #                                        h_p,
        #                                        nlevels,
        #                                        nxcond,
        #                                        xcond,
        #                                        direct,
        #                                        diag_nclass,
        #                                        alpha_length,
        #                                        num_alpha0,
        #                                        npeop,
        #                                        nitem,
        #                                        nclass,
        #                                        c=1.2,
        #                                        err=1e-2,
        #                                        len=10)$alpha_vector_p
        
        
        # alpha_vector_p=modify_max_alpha_bisection_cpp(alpha_vector,
        #                                               Y,
        #                                               Y_comp,
        #                                               Pi_minus_1,
        #                                               h_p,
        #                                               nlevels,
        #                                               nxcond,
        #                                               xcond,
        #                                               direct,
        #                                               diag_nclass,
        #                                               alpha_length,
        #                                               num_alpha0,
        #                                               npeop,
        #                                               nitem,
        #                                               nclass,
        #                                               c=2,
        #                                               err=1e-2,
        #                                               len=10)$alpha_vector_p
        # # cat("\n",alpha_vector_p,"\n")
        # temp=tran_vec_to_list_cpp(alpha_vector_p,
        #                           nlevels,
        #                           nxcond,
        #                           num_alpha0,
        #                           nitem,
        #                           nclass)
        # 
        # cond_prob_and_Pi_minus_1=
        #   conditional_prob_and_Pi_cpp(alpha0_p,alpha_p,Y_comp,
        #                               xcond,nxcond,npeop,nitem,nclass)
        # 
        # cond_prob_p=cond_prob_and_Pi_minus_1$cond_prob
        # 
        # Pi_minus_1=cond_prob_and_Pi_minus_1$Pi_minus_1
        
        
        
        temp=max_alpha_5_Para_cpp(alpha_vector_p,
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
                                  maxiter,
                                  step_length=0.5,
                                  tol=tol_alpha,
                                  it=it_alpha,
                                  ncores)
        
        
        
        
        alpha_vector_p=temp$alpha_vector_p
        alpha0_p=temp$alpha0
        alpha_p =temp$alpha
        
        
        Pi_minus_1=temp$Pi_minus_1
        cond_prob_p=temp$cond_prob
        
        # cat("\n",temp$alpha_vector_p,"\n")
        
        
        
        
        log_lik[k+1]=sum(log(rowSums(cond_prob_p*eta_p) ))
        
        # log_lik[k+1]=log_like_test_cpp(cond_prob_p,eta_p)
        
        # log_lik[k+1]=log_like_cpp(beta_p,alpha0_p,alpha_p,
        #                           Y_comp,Xprev,xcond,nxcond,
        #                           npeop,nitem,nxprev,nclass)
        
        
        if(  (max(abs(alpha_vector_p-alpha_vector))<(tol_para))&
             (max(abs(beta_temp_p-beta_p))<(tol_para))  ) break
        
        
        if(abs(log_lik[k+1]-log_lik[k]<tol_likeli)) break
      }
      
      
      
      if(log_lik[k+1]>ret$log_lik){
        ret$log_lik=log_lik[k+1]
        ret$beta=beta_p
        ret$alpha0=alpha0_p
        ret$alpha=alpha_p
        
        ret$eta=eta_p
        Pi_minus_1_nrep=Pi_minus_1
        cond_prob_nrep=cond_prob_p
        
        ret$beta0init=beta0init
        ret$alpha0init=alpha0init
        # ret$alphainit=alphainit
      }
      log_lik_list[repl]=log_lik[k+1]
      
      time=Sys.time()-starttime_model
      # cat(log_lik[k+1]-log_lik[k]," ")
      # if(log_lik[k+1]-log_lik[k]<0) cat("X ")
      if(verbose==F){
        cat("Model ", repl, ": loglik = ", log_lik[k+1], 
            " ... best loglik = ", ret$log_lik,
            " ... time = ",time," ",attr(time, "units"),
            "\n", sep = "")
      }
      
      
      # cat("Model ", repl, ": loglik = ", log_lik[k+1], 
      #     " ... best loglik = ", log_lik_nrep,
      #     " ... time = ",Sys.time()-starttime_model,
      #     "\n", sep = "")
      
      # cat("Model ", repl, ": log_lik = ", log_lik[k+1], 
      #     " ... best log_lik = ", log_lik_nrep, "\n", sep = "")
      
      
    }
  }else{
    # 直接給真實值作計算
    ret$beta=beta_p
    ret$alpha0=alpha0_p
    ret$alpha=alpha_p
    # log_lik_nrep=log_like_cpp(beta_p,alpha0_p,alpha_p,
    #                           Y_comp,Xprev,xcond,nxcond,
    #                           npeop,nitem,nxprev,nclass)
    
    cond_prob_and_Pi_minus_1=
      conditional_prob_and_Pi_cpp(ret$alpha0,ret$alpha,Y_comp,
                                  xcond,nxcond,npeop,nitem,nclass)
    cond_prob_nrep=cond_prob_and_Pi_minus_1$cond_prob
    Pi_minus_1_nrep=cond_prob_and_Pi_minus_1$Pi_minus_1
    
    ret$eta = eta_cpp(beta_p,Xprev,nxprev,nclass)
    ret$log_lik=sum(log(rowSums(cond_prob_nrep*ret$eta) ))
    k=1
    
  }
  ret$log_lik_list=log_lik_list
  
  ret$cond_prob=cond_prob_nrep
  ret$posterior=h_p_cpp(ret$eta,cond_prob_nrep)
  
  ret$item_prob=Pi_minus_1_nrep
  # fisher_infor
  # prob_temp=prob_cpp(beta_nrep,alpha0_nrep,alpha_nrep,
  #                    Xprev,xcond,e_comp,
  #                    npeop,nitem,nclass,nxprev,nxcond,H)
  
  prob_temp=prob_1_cpp(ret$beta,ret$alpha0,ret$alpha,ret$eta,
                       Xprev,xcond,e_comp,
                       npeop,nitem,nclass,nxprev,nxcond,H)
  
  ret$Y=Y
  ret$N=npeop
  ret$nxprev=nxprev
  ret$nxcond=nxcond
  
  # ret$beta=beta_nrep
  # ret$alpha0=alpha0_nrep
  # ret$alpha=alpha_nrep
  
  
  ret$npar=(1+nxprev)*(nclass-1)+alpha_length
  
  Fisher_temp=Fisher_information_cpp(ret$eta, 
                                     prob_temp$cond_prob_for_all,
                                     Xprev, 
                                     xcond, 
                                     e,
                                     prob_temp$p_i,
                                     nlevels,
                                     npeop,
                                     nitem,
                                     nclass,
                                     nxcond,
                                     ret$npar,
                                     H)
  
  
  ret$Fisher_information=Fisher_temp$Fisher_information
  ret$cov=Fisher_temp$asymptotic_covariance
  ret$Std_Error=sqrt(diag(ret$cov))
  
  # 這是不重覆計算likeli才刪去
  # cond_prob_and_Pi_minus_1=
  #   conditional_prob_and_Pi_cpp(alpha0_nrep,alpha_nrep,Y_comp,
  #                               xcond,nxcond,npeop,nitem,nclass)
  # 
  # cond_prob_p=cond_prob_and_Pi_minus_1$cond_prob
  
  
  
  
  
  
  
  
  # ret$Pi_minus_1=cond_prob_and_Pi_minus_1$Pi_minus_1
  
  
  
  
  
  
  
  # ret$log_lik=log_lik_nrep
  # ret$AIC=-2*log_lik_nrep+2*ret$npar
  # ret$BIC=-2*log_lik_nrep+log(npeop)*ret$npar
  ret$AIC=-2*ret$log_lik+2*ret$npar
  ret$BIC=-2*ret$log_lik+log(npeop)*ret$npar
  
  ret$numiter=k
  
  ret$fitted=rowSums(prob_temp$p_i)
  ret$freq=cbind(outcome_table,rowSums(prob_temp$p_i) )
  
  if(sum(nxcond)+nxprev==0){
    ret$X2=sum(((ret$observed-ret$fitted)^2)/ret$fitted)
    ret$p_value=1-pchisq(ret$X2,H-1-ret$npar)
  }
  
  # ret$eta=prob_temp$eta
  
  # ret$posterior=h_p_cpp(ret$eta,
  #                       cond_prob_cpp(alpha0_nrep,alpha_nrep,
  #                                     Y_comp,xcond,nxcond,
  #                                     npeop,nitem,nclass))
  
  
  
  # Fisher_information_by_score=
  #   Fisher_information_by_score_cpp(ret$eta,
  #                                   ret$posterior,
  #                                   Xprev,
  #                                   cond_prob_and_Pi_minus_1$Pi_minus_1,
  #                                   Y,
  #                                   xcond,
  #                                   direct,
  #                                   diag_nclass,
  #                                   alpha_length,
  #                                   npeop,
  #                                   nxprev,
  #                                   nitem,
  #                                   nclass,
  #                                   nlevels,
  #                                   nxcond,
  #                                   ret$npar)
  
  Fisher_information_by_score=
    Fisher_information_by_score_cpp(ret$eta,
                                    ret$posterior,
                                    Xprev,
                                    Pi_minus_1_nrep,
                                    Y,
                                    xcond,
                                    direct,
                                    diag_nclass,
                                    alpha_length,
                                    npeop,
                                    nxprev,
                                    nitem,
                                    nclass,
                                    nlevels,
                                    nxcond,
                                    ret$npar)
  
  ret$cov_by_score=Fisher_information_by_score$asymptotic_covariance
  ret$Std_Error_by_score=sqrt(diag(ret$cov_by_score))
  
  
  ret$robust_cov=(ret$cov)%*%(Fisher_information_by_score$der)%*%(ret$cov)
  
  ret$robust_Std_Error=sqrt(diag(ret$robust_cov))
  
  
  ret$robust_cov_by_score=
    (ret$cov_by_score)%*%(Fisher_information_by_score$der)%*%(ret$cov_by_score)
  ret$robust_Std_Error_by_score=sqrt(diag(ret$robust_cov_by_score))
  
  
  ret$predclass=apply(ret$posterior,1,which.max)
  
  ret$cond_prob_for_all=prob_temp$cond_prob_for_all
  ret$prob_estimated=prob_temp$p_i
  
  
  ret$der=Fisher_information_by_score$der
  ret$time=Sys.time() - starttime
  # verbose(ret)
  return(ret)
}








slcm_goodness=function(y,eta,pikj){
  
  npeop=nrow(y)
  nclass=nrow(pikj) 
  nitem=ncol(y)
  q=nitem*nclass+1
  
  
  nlevels=rep(2,nitem)
  H=prod(nlevels)
  outcome=((y-1))%*%rev(cumprod(rev(c(nlevels[-1],1))))+1
  #obs_counts 為(O_1,...,O_H)表示觀測次數able(c(outcome,1:H))-1
  obs_counts =table(c(outcome,1:H))-1
  
  obs_counts=unname(obs_counts)
  e=matrix(0,H,nitem)
  for(m in 1:nitem){
    #利用循環數
    e[,m]=rep(1:nlevels[m], each = prod(nlevels[-c(1:m)]), len = H)
  }
  outcome_table=cbind(e,obs_counts )
  YY=2-e
  
  model_probs=rep(0,2^nitem)
  for (j in 1:nclass) {
    # YY=as.matrix(expand.grid(rep(list(0:1), 5)))
    
    log_p_yj <- YY %*% log(c(pikj[j,])) + (1 - YY) %*% log(1 - c(pikj[j,]))
    # posterior[,j] <- eta[j] * exp(log_p_yj)
    model_probs =model_probs+ eta[j] * exp(log_p_yj)
  }
  exp_counts =c(npeop*model_probs)
  
  outcome_table=cbind(outcome_table,exp_counts)
  
  
  X2 <- sum((obs_counts - exp_counts)^2 / exp_counts)
  df <- (length(obs_counts) - 1) - q
  
  # p-value
  p_value = 1 - pchisq(X2, df)
  
  return(list(
    X2 = X2,
    df = df,
    p_value = p_value,
    obs_counts = obs_counts,
    exp_counts = exp_counts,
    outcome_table=outcome_table
  ))
}



# only for binary case  no covariate
slcm_fun=function(y, nclass, maxiter=10000, tol=1e-15, nrun=100, ncores=parallel::detectCores()-2){
  
  starttime <- Sys.time()
  
  ret <- list()
  
  temp=slcm_rcpp_omp(as.matrix(2-y),    
                nclass,
                maxiter,
                tol,
                nrun,
                ncores)

  ret$niter_list=temp$niter
  ret$loglik_list=temp$loglik
  
  whmax=which.max(ret$loglik_list) 
  ret$eta=temp$eta[[whmax]]
  ret$pikj=temp$pikj[[whmax]]
  ret$posterior=temp$posterior[[whmax]]
  ret$log_lik=temp$loglik[[whmax]]
  ret$numiter=temp$niter[[whmax]]
  
  
  goodness=slcm_goodness(y,ret$eta,ret$pikj)
  ret$X2=goodness$X2
  ret$df=goodness$df
  ret$p_value=goodness$p_value
  ret$observed=goodness$obs_counts
  ret$fitted=goodness$exp_counts
  ret$freq=goodness$outcome_table
  
  ret$npar=nitem*nclass+1

  npeop=nrow(y)
  
  ret$AIC=-2*ret$log_lik+2*ret$npar
  ret$BIC=-2*ret$log_lik+log(npeop)*ret$npar


  ret$time=Sys.time() - starttime
  
  return(ret)
}






depLCA=function(y,
                     xprev=NULL,
                     xcond=vector("list",ncol(y)),
                     nlevels=rep(2,ncol(y)),
                     nclass=2,
                     U,
                     tau_labels,
                     beta_p=NULL,
                     alpha0_p=NULL,
                     alpha_p=NULL,
                     tau_p=NULL,
                     beta_init=NULL,
                     alpha0_init=NULL,
                     alpha_init=NULL,
                     tau_init=NULL,
                     maxiter_it=1000,
                     maxiter=1000,
                     maxit=1e3,
                     it_beta=1,
                     it_alpha=1,
                     it_tau=1,
                     tol_beta=1e-12,
                     tol_alpha=1e-12,
                     tol_tau=1e-6,
                     tol=1e-5,
                     tol_par=1e-7,
                     tol_likeli = 1e-7,
                     nrep=1,
                     ncores=parallel::detectCores()-2,
                     verbose=F,
                     seed=NULL){
  
  
  starttime <- Sys.time()
  
  if(ncores<0) stop("\n Error: ncores error. \n \n")
  
  nitem=ncol(y)
  
  tau_pp=vector("list",nclass)
  for(j in 1:nclass){
    
    if(dim(U[[j]][[1]])[2]!=length(tau_labels[[j]])
       |dim(U[[j]][[1]])[1]!=2^nitem-nitem-1) stop("\n Error: tau dimension error. \n \n")
    
    tau_pp[[j]]=rep(0,length(tau_labels[[j]]))
  }
  
  
  if (ncores<0){
    stop("\n Error: not allowed when ncores<0. \n \n")
    # ret <- NULL
  }
  
  if (nclass<=1){
    stop("\n Error: not allowed when nclass<=1; will be ignored. \n \n")
    # ret <- NULL
  }
  
  
  
  if(length(xcond)!= nitem|
     !mode(xcond)=="list"){
    stop("\n Error: xcond  format error \n \n")
    # cat("\n ALERT: xcond  format error \n \n")
    # ret <- NULL
  }
  
  ret <- list()
  ret$U=U
  ret$tau_labels=tau_labels
  
  
  npeop=nrow(y)
  
  
  #binary的簡單方法
  # 總組合數
  H=prod(nlevels)
  
  # e為所有可能結果
  e=matrix(0,H,nitem)
  for(i in 1:nitem){
    #利用循環數
    temp=rep(1:nlevels[i], each = prod(nlevels[-c(1:i)]), len = H)
    e[,i]=temp
  }
  
  ###將y轉成multinomial######
  e_comp=NULL
  for(i in 1:nitem){
    temp=diag(rep(1,nlevels[i]))[  e[,i], ]
    e_comp=cbind(e_comp,temp)
  }
  
  # 處理exponential 裡的correlation
  e_tran=ifelse(e>=2,0,e)
  e_tran_cor=t(e_tran)
  
  for(i in 1:(nitem-1)){
    atemp=t( e_tran[,i]*e_tran[,-c(1:i)] )
    e_tran_cor=rbind(e_tran_cor,atemp)
  }
  
  for(i in 1:nitem){
    atemp=t( e_tran_cor[i,]*t(e_tran_cor[-c(1:i),]) )
    e_tran_cor=rbind(e_tran_cor,atemp)
  }
  
  e_tran_cor=unique(e_tran_cor)
  e_tran_cor=t(e_tran_cor)
  
  simu_item=nitem+choose(nitem,2)
  
  
  
  outcome=((y-1))%*%rev(cumprod(rev(c(nlevels[-1],1))))+1
  #O為(O_1,...,O_H)表示觀測次數able(c(outcome,1:H))-1
  O=table(c(outcome,1:H))-1
  
  outcome_table=cbind(e,O)
  colnames(outcome_table)[nitem+1]="freq" 
  
  
  Y_cor=NULL
  for(i in 1:npeop){
    temp=diag(H)[ outcome[i,],]
    Y_cor=rbind(Y_cor,temp)
  }
  
  y_w=e_tran_cor[c(outcome),]
  
  
  
  e_comp=NULL
  
  Y=NULL
  Y_comp=NULL
  nxcond=rep(0,nitem)
  H=prod(nlevels)
  # e為所有可能結果
  e=matrix(0,H,nitem)
  
  for(m in 1:nitem){
    #利用循環數
    e[,m]=rep(1:nlevels[m], each = prod(nlevels[-c(1:m)]), len = H)
    
    
    temp=diag(rep(1,nlevels[m]))[ y[,m],]
    
    Y_comp=cbind(Y_comp,temp)
    temp=temp[,-nlevels[m]]
    Y=cbind(Y,temp)
    
    temp_e=diag(rep(1,nlevels[m]))[  e[,m], ]
    e_comp=cbind(e_comp,temp_e)
    
    if(is.null(xcond[[m]])) next
    nxcond[m]=ncol(xcond[[m]])
  }
  
  
  
  
  outcome=((y-1))%*%rev(cumprod(rev(c(nlevels[-1],1))))+1
  #O為(O_1,...,O_H)表示觀測次數able(c(outcome,1:H))-1
  O=table(c(outcome,1:H))-1
  
  # outcome_table=cbind(e,O)
  # colnames(outcome_table)[nitem+1]="count" 
  # 
  
  # ret <- list()
  ret$observed=O
  
  if(is.null(xprev)){
    nxprev=0
    Xprev=matrix(1,npeop,1)
  }else{
    nxprev=ncol(xprev)
    Xprev=cbind(1,xprev)
  }
  
  
  if(nxprev==0){
    max_beta_cpp=max_beta_6_no_covariate_cpp
  }else{
    max_beta_cpp=max_beta_6_cpp
  }
  
  
  
  diag_nclass=diag(nclass)
  direct=calculateDirectSum(nitem,nlevels)
  
  # ret <- list()
  # ret$observed=O
  num_alpha0=sum(nclass*(nlevels- 1))
  num_alpha=sum(nxcond*(nlevels - 1))
  alpha_length=num_alpha0+num_alpha
  
  
  


  if(nxprev==0&sum(nxcond)==0){
    # beta_p=matrix(0,(nxprev+1),(nclass-1))
    temp=slcm_rcpp_omp(as.matrix(2-y),
                       nclass,
                       maxiter=2000,
                       tol=1e-16,
                       nrun=nrep,
                       ncores)
    
    
    whmax=which.max(temp$loglik)
    # cat(whmax)
    eta_p=temp$eta[[whmax]]
    # beta_p[1,]=matrix(log(eta_p[1:(nclass-1)]/(1-eta_p[1:(nclass-1)])),nrow = 1)
    beta_p=matrix(log(eta_p[1:(nclass-1)]/(1-eta_p[1:(nclass-1)])),nrow = 1)
    
    
    
    
    pikj=temp$pikj[[whmax]]
    pikj=ifelse(abs(1-pikj)<0.01,0.99,pikj)
    pikj=ifelse(abs(pikj)<0.01,0.01,pikj)
    
    uuu=log(pikj/(1-pikj))
    
    ret$uuu=uuu
    
    
    
    alpha0_p=NULL
    alpha_p=NULL
    
    for(m in 1:nitem){
      # use for binary case
      # temp=runif(nclass)
      # a=list(matrix(log(temp/(1-temp)),
      #               nclass,(nlevels[m] - 1)))
      
      temp=matrix(runif(nclass*(nlevels[m])),nclass,(nlevels[m] ))
      temp=(temp/rowSums(temp))[,-nlevels[m]]
      
      a=list(matrix(uuu[1:(nclass*(nlevels[m] - 1))],
                    nclass,(nlevels[m] - 1)))
      
      uuu=uuu[-c(1:(nclass*(nlevels[m] - 1)))]
      
      alpha0_p=c(alpha0_p,a)
      
      # a=list(matrix(runif(10,-1,1),nxcond[m],(nlevels[m] - 1)))
      a=list(matrix(0,nxcond[m],(nlevels[m] - 1)))
      alpha_p=c(alpha_p,a)
    }
    
    # alpha0init=alpha0_p
    # alphainit=alpha_p
    
    # tau_p=matrix(0 ,(2^(nitem)-nitem-1),nclass)
    ret$alpha0_int=alpha0_p
  }else{
    temp=rlca_6(y=y,
                xprev=xprev,
                xcond=xcond,
                nlevels,
                nclass=nclass,
                beta_p=NULL,
                alpha0_p=NULL,
                alpha_p=NULL,
                nrep=nrep,
                ncores=ncores,
                seed=seed)
    
    beta_p=temp$beta
    alpha0_p=temp$alpha0
    alpha_p=temp$alpha
  }
  

  
  
  
  

  # cat(unlist(alpha_p))
  
  tau_p=tau_pp
  
  
  alpha_vector_p=c(unlist(alpha0_p),unlist(alpha_p))
  
  # cat("\n",beta_p,"\n")
  # cat("\n",alpha_vector_p,"\n")
  
  log_lik=c()
  log_lik[1]=-Inf
  conditional_prob_tau=
    conditional_prob_and_Pi_tau_cpp(alpha0_p,
                                    alpha_p,
                                    Y_comp,
                                    e_comp,
                                    xcond,
                                    nxcond,
                                    npeop,
                                    nitem,
                                    nclass)
  
  
  cond_dist_p=conditional_prob_tau$cond_dist
  cond_prob_p=conditional_prob_tau$cond_prob
  Pi_minus_1=conditional_prob_tau$Pi_minus_1
  eta_p = eta_cpp(beta_p,Xprev,nxprev,nclass)
  # cat(beta_p)
  for(k in 1:maxiter_it){
    
    beta_temp_p=beta_p
    tau_temp_p=tau_p
    
    
    h_p= h_p_cpp(eta_p,cond_prob_p)
    
    
    
    
    
    testets_tau= max_tau_Parallel_2_cpp(alpha0_p,
                                        alpha_p,
                                        tau_p,
                                        U,
                                        tau_labels,
                                        labels=unlist(tau_labels),
                                        num_tau=max(unlist(tau_labels)),
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
                                        xcond,
                                        diag_nclass,
                                        num_alpha0,
                                        alpha_length,
                                        npeop,
                                        nitem,
                                        nclass,
                                        Y_cor,
                                        e_tran_cor_nitem=e_tran_cor[,-c(1:nitem)],
                                        A_matrix=t(e_tran_cor[,c(1:nitem)]),
                                        A_matrix_complement=1-t(e_tran_cor[,c(1:nitem)]),
                                        maxiter,
                                        tol,
                                        maxit,
                                        tol_par=tol_tau,
                                        it=it_tau,
                                        ncores)
    
    tau_p=testets_tau$tau
    Pi_minus_1=testets_tau$Pi_minus_1
    cond_dist_p=testets_tau$cond_dist
    cond_prob_p=testets_tau$cond_prob
    
    
    
    h_p= h_p_cpp(eta_p,cond_prob_p)
    
    alpha_vector=alpha_vector_p
    
    
    temp=max_alpha_likeli_4_Parallel_2_cpp(alpha_vector_p,
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
                                           xcond,
                                           diag_nclass,
                                           num_alpha0,
                                           alpha_length,
                                           npeop,
                                           nitem,
                                           nclass,
                                           Y_cor,
                                           e_tran_cor_nitem=e_tran_cor[,-c(1:nitem)],
                                           A_matrix=t(e_tran_cor[,c(1:nitem)]),
                                           A_matrix_complement=1-t(e_tran_cor[,c(1:nitem)]),
                                           maxiter,
                                           tol,
                                           maxit,
                                           tol_par=tol_alpha,
                                           step_length=0.5,
                                           it=it_alpha,
                                           ncores)
    
    alpha_vector_p=temp$alpha_vector_p
    alpha0_p=temp$alpha0
    alpha_p =temp$alpha
    
    cond_prob_p=temp$cond_prob
    cond_dist_p=temp$cond_dist
    Pi_minus_1=temp$Pi_minus_1
    
    # if(nxprev==0){
    #   # eta_p=colMeans(h_p)
    #   # beta_p=matrix(log(eta_p[1:(nclass-1)]/(1-eta_p[1:(nclass-1)])),nrow = 1)
    #   # eta_p = matrix(eta_p,npeop,nclass,byrow = T)
    #   maxbeta=calculate_eta_beta_vec(h_p,nclass, npeop)
    #   beta_p=maxbeta$beta_p
    #   eta_p=maxbeta$eta_p_matrix
    # }else{
    #   maxbeta=max_beta_cpp(beta_p,Xprev,eta_p,h_p,npeop,nxprev,nclass,
    #                        maxiter,step_length=0.5,
    #                        tol=tol_beta,it=it_beta)
    #   beta_p=maxbeta$beta_p
    #   eta_p =maxbeta$eta
    # }
    
    h_p= h_p_cpp(eta_p,cond_prob_p)
    
    maxbeta=max_beta_cpp(beta_p,Xprev,eta_p,h_p,npeop,nxprev,nclass,
                         maxiter,step_length=0.5,
                         tol=tol_beta,it=it_beta)
    beta_p=maxbeta$beta_p
    eta_p=maxbeta$eta
    
    
    # cat("\n",beta_p,"\n")
    
    
    
    
    # cat("\n",alpha_vector_p,"\n")
    
    
    # cat("\n",unlist(tau_p),"\n")
    
    # temp_cond_prob_p2 <<- testets_tau
    # cat("\n",sum(cond_prob_p),"\n")
    
    log_lik[k+1]=sum(log(rowSums(cond_prob_p*eta_p) ))
    
    
    # cat("\n",k," ",sum(abs(unlist(alpha_p)-unlist(alpha)))," ",
    #     sum(abs(beta_temp_p-beta_p))," ",
    #     sum(abs(alpha_vector_p-alpha_vector))," ",
    #     sum(abs(unlist(tau_temp_p)-unlist(tau_p)))," ",
    #     log_lik[k+1]," ",
    #     log_lik[k+1]-log_lik[k],"\n")
    
    # cat("\n",k," ",
    #     sum(abs(beta_temp_p-beta_p))," ",
    #     sum(abs(alpha_vector_p-alpha_vector))," ",
    #     sum(abs(tau_temp_p-tau_p))," ",
    #     log_lik[k+1]," ",
    #     log_lik[k+1]-log_lik[k],"\n")
    
    if(  (max(abs(alpha_vector_p-alpha_vector))<(tol_par))&
         (max(abs(beta_temp_p-beta_p))<(tol_par))&
         (max(abs(unlist(tau_temp_p)-unlist(tau_p)))<(tol_par))) break
    
    
    if(abs(log_lik[k+1]-log_lik[k])<tol_likeli) break
    
  }
  # if(log_lik[k+1]>ret$log_lik){
  #   ret$log_lik=log_lik[k+1]
  #   ret$beta=beta_p
  #   ret$alpha0=alpha0_p
  #   ret$alpha=alpha_p
  #   ret$tau=tau_p
  #   
  #   
  #   ret$eta=eta_p
  #   Pi_minus_1_nrep=Pi_minus_1
  #   cond_prob_nrep=cond_prob_p
  #   cond_dist_nrep=cond_dist_p
  #   
  #   # ret$beta0init=beta0init
  #   # ret$alpha0init=alpha0init
  #   # ret$alphainit=alphainit
  # }
  
  
  ret$log_lik=log_lik[k+1]
  ret$beta=beta_p
  ret$alpha0=alpha0_p
  ret$alpha=alpha_p
  ret$tau=tau_p
  
  
  ret$eta=eta_p
  Pi_minus_1_nrep=Pi_minus_1
  cond_prob_nrep=cond_prob_p
  cond_dist_nrep=cond_dist_p
  
  ret$item_prob=Pi_minus_1_nrep
  
  ret$numiter=k
  
  ret$posterior=h_p_cpp(ret$eta,cond_prob_nrep)
  ret$predclass=apply(ret$posterior,1,which.max)
  
  ret$cond_prob_for_all=cond_dist_nrep
  
  prob=0
  for(j in 1:nclass){
    prob=prob+cond_dist_nrep[[j]]*ret$eta[,j]
  }
  # p_i=t(prob)
  # ret$prob_estimated=p_i
  ret$prob_estimated=t(prob)
  ret$fitted=rowSums(ret$prob_estimated)
  
  ret$freq=cbind(outcome_table,rowSums(ret$prob_estimated) )
  colnames(ret$freq)[nitem+2]="fitted"
  
  ret$Y=Y
  ret$N=npeop
  ret$nxprev=nxprev
  ret$nxcond=nxcond
  
  num_tau=max(unlist(tau_labels))
  ret$npar=(1+nxprev)*(nclass-1)+alpha_length+num_tau
  
  if(sum(nxcond)+nxprev==0){
    ret$X2=sum(((ret$observed-ret$fitted)^2)/ret$fitted)
    ret$p_value=1-pchisq(ret$X2,H-1-ret$npar)
  }
  
  # Fisher_temp=Fisher_like_information_cpp(ret$eta,
  #                                         ret$cond_prob_for_all,
  #                                         Pi_minus_1_nrep,
  #                                         Xprev,
  #                                         xcond,
  #                                         e,
  #                                         e_tran_cor,
  #                                         e_tran,
  #                                         ret$prob_estimated,
  #                                         nlevels,
  #                                         npeop,
  #                                         nitem,
  #                                         nclass,
  #                                         nxcond,
  #                                         diag_nclass,
  #                                         ret$npar,
  #                                         alpha_length,
  #                                         H)
  # 
  # ret$Fisher_information=Fisher_temp$Fisher_information
  # ret$cov=Fisher_temp$asymptotic_covariance
  # ret$Std_Error=sqrt(diag(ret$cov))
  
  
  
  
  
  
  # Fisher_information_by_score=
  #   Fisher_like_information_by_score_cpp(ret$eta,
  #                                        ret$posterior,
  #                                        Xprev,
  #                                        Pi_minus_1_nrep,
  #                                        ret$cond_prob_for_all,
  #                                        e_tran_cor,
  #                                        y_w,
  #                                        Y,
  #                                        xcond,
  #                                        direct,
  #                                        diag_nclass,
  #                                        npeop,
  #                                        nxprev,
  #                                        nitem,
  #                                        nclass,
  #                                        nlevels,
  #                                        nxcond,
  #                                        ret$npar,
  #                                        alpha_length,
  #                                        H)
  # ret$cov_by_score=Fisher_information_by_score$asymptotic_covariance
  # ret$Std_Error_by_score=sqrt(diag(ret$cov_by_score))
  
  
  # ret$robust_cov=(ret$cov)%*%(Fisher_information_by_score$der)%*%(ret$cov)
  # 
  # ret$robust_Std_Error=sqrt(diag(ret$robust_cov))
  
  
  # ret$robust_cov_by_score=
  #   (ret$cov_by_score)%*%(Fisher_information_by_score$der)%*%(ret$cov_by_score)
  # ret$robust_Std_Error_by_score=sqrt(diag(ret$robust_cov_by_score))
  
  
  
  
  ret$AIC=-2*ret$log_lik+2*ret$npar
  ret$BIC=-2*ret$log_lik+log(npeop)*ret$npar
  
  ret$cond_dist=cond_dist_nrep
  
  ret$time=Sys.time() - starttime
  
  # ret$Pi_minus_1=Pi_minus_1_nrep
  # ret$cond_dist=cond_dist_nrep
  # ret$e_tran_cor=e_tran_cor
  # ret$e_tran=e_tran
  # ret$alpha_length=alpha_length
  return(ret)
}

