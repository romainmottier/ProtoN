
#ifndef SSPRKStepOpt_hpp
#define SSPRKStepOpt_hpp

void SSPRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &alpha, Matrix<double, Dynamic, Dynamic> &beta, TSSPRKHHOAnalyses & ssprk_an, double & dt, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof);

void SSPRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &alpha, Matrix<double, Dynamic, Dynamic> &beta, TSSPRKHHOAnalyses & ssprk_an, double & dt, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof){
    
    size_t n_dof = x_dof_n_m.rows();
    Matrix<double, Dynamic, Dynamic> ys = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s+1);

    Matrix<double, Dynamic, 1> yn, ysi, yj;
    ys.block(0, 0, n_dof, 1) = x_dof_n_m;
    for (int i = 0; i < s; i++) {

        ysi = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
        for (int j = 0; j <= i; j++) {
            yn = ys.block(0, j, n_dof, 1);
            ERKWeightOpt(ssprk_an, yn, yj, n_f_dof, dt, alpha(i,j), beta(i,j));
            ysi += yj;
        }
        ys.block(0, i+1, n_dof, 1) = ysi;
    }
    
    x_dof_n = ys.block(0, s, n_dof, 1);

}

#endif
