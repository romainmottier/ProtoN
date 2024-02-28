
#ifndef sdirk_step_cuthho_interface_hpp
#define sdirk_step_cuthho_interface_hpp

template<typename Mesh, typename testType, typename meth>
void
sdirk_step_cuthho_interface(size_t it, size_t s, RealType ti, RealType dt, Matrix<RealType, Dynamic, Dynamic> a, Matrix<RealType, Dynamic, Dynamic> b, Matrix<RealType, Dynamic, Dynamic> c, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> & x_dof, dirk_hho_scheme<RealType> & analysis, bool write_error_Q = false);

template<typename Mesh, typename testType, typename meth>
void
sdirk_step_cuthho_interface(size_t it, size_t s, RealType ti, RealType dt, Matrix<RealType, Dynamic, Dynamic> a, Matrix<RealType, Dynamic, Dynamic> b, Matrix<RealType, Dynamic, Dynamic> c, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> & x_dof, dirk_hho_scheme<RealType> & analysis, bool write_error_Q){
    
    bool write_silo_Q = false;
    auto level_set_function = test_case.level_set_;

    auto rhs_fun                  = test_case.rhs_fun;
    auto sol_fun                  = test_case.sol_fun;
    auto sol_grad                 = test_case.sol_grad;
    auto bcs_fun                  = test_case.bcs_fun;
    auto dirichlet_jump           = test_case.dirichlet_jump;
    auto neumann_jump             = test_case.neumann_jump;
    struct params<RealType> parms = test_case.parms;

    timecounter tc;
    auto assembler = make_two_fields_interface_assembler(msh, bcs_fun, hdi);
    
    if (x_dof.rows() == 0) {
        RealType t = ti;
        auto test_t_case = make_test_case_laplacian_waves_mixed(t,msh, level_set_function);
        auto vel_fun     = test_t_case.sol_fun;
        auto flux_fun    = test_t_case.sol_grad;
        auto rhs_fun     = test_t_case.rhs_fun;

        assembler.project_over_cells(msh, hdi, x_dof, vel_fun, flux_fun);
                
        size_t it = 0;
        if(write_silo_Q || true){
            std::string silo_file_name = "cut_hho_two_fields_";
            postprocessor<Mesh>::write_silo_two_fields(silo_file_name, it, msh, hdi, assembler, x_dof, vel_fun, false);
        }
    }
    
    // SDIRK step
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
                RealType t = tn + c(i,0) * dt;
                auto test_t_case = make_test_case_laplacian_waves_mixed(t,msh, level_set_function);
                assembler.RHS.setZero(); // assuming null dirichlet data on boundary.
                #ifdef HAVE_INTEL_TBB
                        size_t n_cells = msh.cells.size();
                        tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
                            [&msh,&method,&test_t_case,&hdi,&assembler] (size_t & cell_ind){
                                auto& cell = msh.cells.at(cell_ind);
                                auto f = method.make_contrib_rhs(msh, cell, test_t_case, hdi);
                                assembler.assemble_rhs(msh, cell, f);
                        }
                    );
                #else
                    for (auto& cell : msh.cells)
                    {
                        auto f = method.make_contrib_rhs(msh, cell, test_t_case, hdi);
                        assembler.assemble_rhs(msh, cell, f);
                    }
                #endif
                analysis.SetFg(assembler.RHS);
                analysis.irk_weight(yn, ki, dt, a(i,i),true);
            }

            // Accumulated solution
            x_dof_n += dt*b(i,0)*ki;
            k.block(0, i, n_dof, 1) = ki;
        }
    }
    tc.toc();
    std::cout << bold << cyan << "SDIRK step completed: " << tc << " seconds" << reset << std::endl;
    x_dof = x_dof_n;

    if(write_silo_Q || write_error_Q){
        std::string silo_file_name = "cut_hho_two_fields_";
        postprocessor<Mesh>::write_silo_two_fields(silo_file_name, it, msh, hdi, assembler, x_dof, sol_fun, false);
    }

    if(write_error_Q){
        postprocessor<Mesh>::compute_errors_two_fields(msh, hdi, assembler, x_dof, sol_fun, sol_grad);
    }

    
}

#endif

