
#ifndef erk_step_cuthho_interface_cfl_hpp
#define erk_step_cuthho_interface_cfl_hpp

template<typename Mesh, typename testType, typename meth>
void
erk_step_cuthho_interface_cfl(size_t it, size_t s, RealType ti, RealType dt, Matrix<RealType, Dynamic, Dynamic> a, Matrix<RealType, Dynamic, Dynamic> b, Matrix<RealType, Dynamic, Dynamic> c, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> & x_dof, erk_hho_scheme<RealType> & analysis, bool write_error_Q = false);

template<typename Mesh, typename testType, typename meth>
void
erk_step_cuthho_interface_cfl(size_t it, size_t s, RealType ti, RealType dt, Matrix<RealType, Dynamic, Dynamic> a, Matrix<RealType, Dynamic, Dynamic> b, Matrix<RealType, Dynamic, Dynamic> c, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> & x_dof, erk_hho_scheme<RealType> & analysis, bool write_error_Q){
    
    bool write_silo_Q = false;
    auto level_set_function = test_case.level_set_;

    auto bcs_fun = test_case.bcs_fun;

    timecounter tc;
    auto assembler = make_two_fields_interface_assembler(msh, bcs_fun, hdi);
    
    if (x_dof.rows() == 0) {
        RealType t = ti;
        auto test_t_case = make_test_case_laplacian_waves_mixed(t,msh, level_set_function);
        auto vel_fun = test_t_case.sol_fun;
        auto flux_fun = test_t_case.sol_grad;
        auto rhs_fun = test_t_case.rhs_fun;

        assembler.project_over_cells(msh, hdi, x_dof, vel_fun, flux_fun);
    }
    


    // ERK step
    analysis.refresh_faces_unknowns(x_dof);
    Matrix<RealType, Dynamic, 1> x_dof_n;
    RealType tn = dt*(it-1)+ti;
    tc.tic();
    {
        size_t n_dof = x_dof.rows();
        Matrix<RealType, Dynamic, Dynamic> k = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof, s);
        Matrix<RealType, Dynamic, 1> Fg, Fg_c,xd;
        xd = Matrix<RealType, Dynamic, 1>::Zero(n_dof, 1);
        
        Matrix<RealType, Dynamic, 1> yn, ki;

        x_dof_n = x_dof;
        for (int i = 0; i < s; i++) {
            
            yn = x_dof;
            for (int j = 0; j < s - 1; j++) {
                yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
            }
            
            {
                assembler.RHS.setZero(); // assuming null dirichlet data on boundary.
                analysis.SetFg(assembler.RHS);
                analysis.erk_weight(yn, ki);
            }

            // Accumulated solution
            x_dof_n += dt*b(i,0)*ki;
            k.block(0, i, n_dof, 1) = ki;
        }
    }
    tc.toc();
    std::cout << bold << cyan << "ERK step completed: " << tc << " seconds" << reset << std::endl;
    x_dof = x_dof_n;
    
}

#endif
