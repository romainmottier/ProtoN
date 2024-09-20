#ifndef FaceDoFUpdate_hpp
#define FaceDoFUpdate_hpp

void FaceDoFUpdate(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_dof, size_t n_f_dof);

void FaceDoFUpdate(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_dof, size_t n_f_dof){
    
    using RealType = double;
    timecounter tc;
    tc.tic();
    
    size_t n_c_dof = x_dof.rows() - n_f_dof;
    
    // Composing objects
    SparseMatrix<RealType> Mc = Mg.block(0, 0, n_c_dof, n_c_dof);
    SparseMatrix<RealType> Kc = Kg.block(0, 0, n_c_dof, n_c_dof);
    SparseMatrix<RealType> Kcf = Kg.block(0, n_c_dof, n_c_dof, n_f_dof);
    SparseMatrix<RealType> Kfc = Kg.block(n_c_dof,0, n_f_dof, n_c_dof);
    SparseMatrix<RealType> Sff = Kg.block(n_c_dof,n_c_dof, n_f_dof, n_f_dof);
    Matrix<double, Dynamic, 1> x_c_dof = x_dof.block(0, 0, n_c_dof, 1);
    Matrix<double, Dynamic, 1> x_f_dof = x_dof.block(n_c_dof, 0, n_f_dof, 1);
    
    std::cout << "x_f_dof = " << x_f_dof << std::endl;
    
    // Faces update (last state)
    Matrix<double, Dynamic, 1> RHSf = Kfc*x_c_dof;
    SparseLU<SparseMatrix<RealType>> analysis_f;
    analysis_f.analyzePattern(Sff);
    analysis_f.factorize(Sff);
    Matrix<double, Dynamic, 1> x_f_dof_c = -analysis_f.solve(RHSf); // new state
    std::cout << "x_f_dof_c = " << x_f_dof_c << std::endl;
    x_f_dof = x_f_dof_c;
    x_dof.block(n_c_dof, 0, n_f_dof, 1) = x_f_dof;

    tc.toc();
    
}

#endif
