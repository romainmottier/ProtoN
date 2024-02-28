
#ifndef DIRKStep_hpp
#define DIRKStep_hpp

void DIRKStep(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, SparseMatrix<double> & Mg, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n);

void IRKWeight(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k, double dt, double a);

void DIRKStep(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, SparseMatrix<double> & Mg, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n){
    

    size_t n_dof = x_dof_n_m.rows();
    Matrix<double, Dynamic, Dynamic> k = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s);
    SparseMatrix<double> Kg;
    Matrix<double, Dynamic, 1> Fg;
    
    double t;
    Matrix<double, Dynamic, 1> yn, ki;

    x_dof_n = x_dof_n_m;
    for (int i = 0; i < s; i++) {
        
        yn = x_dof_n_m;
        for (int j = 0; j < s - 1; j++) {
            yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
        }
        
        t = tn + c(i,0) * dt;

#ifdef InhomogeneousQ
        ComputeInhomogeneousKGFG(Kg, Fg, msh, hho_di, assembler, t, functions);
#else
        ComputeKGFG(Kg, Fg, msh, hho_di, assembler, t, functions);
#endif
    
        IRKWeight(Kg, Fg, Mg, yn, ki, dt, a(i,i));
        
        // Accumulated solution
        x_dof_n += dt*b(i,0)*ki;
        k.block(0, i, n_dof, 1) = ki;
    }
    
}

#endif
