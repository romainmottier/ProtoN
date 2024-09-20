
#ifndef ERKWeight_hpp
#define ERKWeight_hpp

void ERKWeight(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_n_dof, Matrix<double, Dynamic, 1> & x_dof, size_t n_f_dof, double dt, double a, double b);


void ERKWeight(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof, double dt, double a, double b){
    
    using RealType = double;
    timecounter tc;
    tc.tic();
      
    size_t n_c_dof = Kg.rows() - n_f_dof;
    
    // Composing objects
    SparseMatrix<RealType> Mc = Mg.block(0, 0, n_c_dof, n_c_dof);
    SparseMatrix<RealType> Kc = Kg.block(0, 0, n_c_dof, n_c_dof);
    SparseMatrix<RealType> Kcf = Kg.block(0, n_c_dof, n_c_dof, n_f_dof);
    SparseMatrix<RealType> Kfc = Kg.block(n_c_dof,0, n_f_dof, n_c_dof);
    SparseMatrix<RealType> Sff = Kg.block(n_c_dof,n_c_dof, n_f_dof, n_f_dof);
    Matrix<double, Dynamic, 1> Fc = Fg.block(0, 0, n_c_dof, 1);
    Matrix<double, Dynamic, 1> x_c_dof = x_dof_n_m.block(0, 0, n_c_dof, 1);
    Matrix<double, Dynamic, 1> x_f_dof = x_dof_n_m.block(n_c_dof, 0, n_f_dof, 1);
    
    
    SparseLU<SparseMatrix<RealType>> analysis_f;
    analysis_f.analyzePattern(Sff);
    analysis_f.factorize(Sff);
    {
        // Faces update (last state)
        Matrix<double, Dynamic, 1> RHSf = Kfc*x_c_dof;
        Matrix<double, Dynamic, 1> x_f_dof_c = -analysis_f.solve(RHSf);
        x_f_dof = x_f_dof_c;
    }
    
    // Cells update
    SparseLU<SparseMatrix<RealType>> analysis_c;
    analysis_c.analyzePattern(Mc);
    analysis_c.factorize(Mc);
    Matrix<double, Dynamic, 1> RHSc = Fc - Kc*x_c_dof - Kcf*x_f_dof;
    Matrix<double, Dynamic, 1> delta_x_c_dof = analysis_c.solve(RHSc);
    Matrix<double, Dynamic, 1> x_n_c_dof = a * x_c_dof + b * dt * delta_x_c_dof; // new state
    
    // Faces update
    Matrix<double, Dynamic, 1> RHSf = Kfc*x_n_c_dof;
    Matrix<double, Dynamic, 1> x_n_f_dof = -analysis_f.solve(RHSf); // new state
    
    // Composing global solution
    x_dof_n = x_dof_n_m;
    x_dof_n.block(0, 0, n_c_dof, 1) = x_n_c_dof;
    x_dof_n.block(n_c_dof, 0, n_f_dof, 1) = x_n_f_dof;
    tc.toc();
    
}

#endif
