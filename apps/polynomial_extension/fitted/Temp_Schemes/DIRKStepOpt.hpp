
#ifndef DIRKStepOpt_hpp
#define DIRKStepOpt_hpp

void DIRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, TDIRKHHOAnalyses & dirk_an, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, bool is_sdirk_Q = false);

void IRKWeightOpt(TDIRKHHOAnalyses & dirk_an, Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k, double dt, double a, bool is_sdirk_Q = false);

void DIRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, TDIRKHHOAnalyses & dirk_an, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, bool is_sdirk_Q){
    
    size_t n_dof = x_dof_n_m.rows();
    Matrix<double, Dynamic, Dynamic> k = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s);
    Matrix<double, Dynamic, 1> Fg, Fg_c,xd;
    xd = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
    
    double t;
    Matrix<double, Dynamic, 1> yn, ki;

    x_dof_n = x_dof_n_m;
    for (int i = 0; i < s; i++) {
        
        yn = x_dof_n_m;
        for (int j = 0; j < s - 1; j++) {
            yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
        }
        
        t = tn + c(i,0) * dt;
        ComputeFG(Fg, msh, hho_di, assembler, t, functions);
        dirk_an.SetFg(Fg);
        
        IRKWeightOpt(dirk_an, yn, ki, dt, a(i,i),is_sdirk_Q);
        
        // Accumulated solution
        x_dof_n += dt*b(i,0)*ki;
        k.block(0, i, n_dof, 1) = ki;
    }
    
}

#endif
