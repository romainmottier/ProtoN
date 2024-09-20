#ifndef ComputeFG_hpp
#define ComputeFG_hpp

void ComputeFG(Matrix<double, Dynamic, 1> & Fg, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions);

void ComputeFG(Matrix<double, Dynamic, 1> & Fg, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions){
    
    using RealType = double;
    auto exact_vel_sol_fun = functions.Evaluate_v(t);
    auto rhs_fun = functions.Evaluate_f(t);
    
    assembler.LHS *= 0.0;
    assembler.RHS *= 0.0;
    
    for (auto& cell : msh.cells)
    {
        
        double c = 1.0;
        auto bar = barycenter(msh, cell);
        double x = bar.x();
        if (x < 0.5) {
            c *= contrast;
        }
    
        auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
        auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
        auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif

        auto n_rows = reconstruction_operator.second.rows();
        auto n_cols = reconstruction_operator.second.cols();
        
        auto n_s_rows = stabilization_operator.rows();
        auto n_s_cols = stabilization_operator.cols();
        
        Matrix<RealType, Dynamic, Dynamic> S_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
        
        Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
        R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) *= 0.0;
        // Compossing objects
        Matrix<RealType, Dynamic, Dynamic> laplacian_loc = R_operator + (1.0/c)*S_operator;
        Matrix<RealType, Dynamic, 1> f_loc = make_mixed_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
        assembler.assemble_mixed_RHS(msh, cell, laplacian_loc, f_loc, exact_vel_sol_fun);
    }
    assembler.finalize();
    Fg = assembler.RHS;
    
}

#endif
