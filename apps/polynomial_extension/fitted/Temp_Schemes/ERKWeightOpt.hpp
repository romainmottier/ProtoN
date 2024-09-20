
#ifndef ERKWeightOpt_hpp
#define ERKWeightOpt_hpp


void ERKWeightOpt(TSSPRKHHOAnalyses & ssprk_an,  Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof, double dt, double a, double b);




void ERKWeightOpt(TSSPRKHHOAnalyses & ssprk_an,  Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof, double dt, double a, double b){
    
    timecounter tc;
    tc.tic();
    
    size_t n_c_dof = x_dof_n_m.rows() - n_f_dof;
    Matrix<double, Dynamic, 1> x_c_dof = x_dof_n_m.block(0, 0, n_c_dof, 1);
    Matrix<double, Dynamic, 1> x_f_dof = x_dof_n_m.block(n_c_dof, 0, n_f_dof, 1);
    
    // Faces update (last state)
    {
        Matrix<double, Dynamic, 1> RHSf = ssprk_an.Kfc()*x_c_dof;
        x_f_dof = -ssprk_an.FacesAnalysis().solve(RHSf);
    }
    
    // Cells update
    Matrix<double, Dynamic, 1> RHSc = ssprk_an.Fc() - ssprk_an.Kc()*x_c_dof - ssprk_an.Kcf()*x_f_dof;
    Matrix<double, Dynamic, 1> delta_x_c_dof = ssprk_an.CellsAnalysis().solve(RHSc);
    Matrix<double, Dynamic, 1> x_n_c_dof = a * x_c_dof + b * dt * delta_x_c_dof; // new state
    
    // Faces update
    Matrix<double, Dynamic, 1> RHSf = ssprk_an.Kfc()*x_n_c_dof;
    Matrix<double, Dynamic, 1> x_n_f_dof = -ssprk_an.FacesAnalysis().solve(RHSf); // new state
    
    // Composing global solution
    x_dof_n = x_dof_n_m;
    x_dof_n.block(0, 0, n_c_dof, 1) = x_n_c_dof;
    x_dof_n.block(n_c_dof, 0, n_f_dof, 1) = x_n_f_dof;
    tc.toc();
    
}


#endif
