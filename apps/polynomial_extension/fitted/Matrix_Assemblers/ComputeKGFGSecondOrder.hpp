
#ifndef ComputeKGFGSecondOrder_hpp
#define ComputeKGFGSecondOrder_hpp

void ComputeKGFGSecondOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions);

void ComputeKGFGSecondOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions){
    using RealType = double;
    

    auto exact_scal_sol_fun = functions.Evaluate_u(t);
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
        
        auto reconstruction_operator = make_hho_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
        auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
        auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
//        Matrix<RealType, Dynamic, Dynamic> laplacian_loc = reconstruction_operator.second + (c*c)* stabilization_operator;
        Matrix<RealType, Dynamic, Dynamic> laplacian_loc = reconstruction_operator.second + stabilization_operator;
        Matrix<RealType, Dynamic, 1> f_loc = make_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
        assembler.assemble(msh, cell, laplacian_loc, f_loc, exact_scal_sol_fun);
    }
    assembler.finalize();
}

#endif
