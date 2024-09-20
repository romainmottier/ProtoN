
#ifndef newmark_step_cuthho_interface_hpp
#define newmark_step_cuthho_interface_hpp

template<typename Mesh, typename testType, typename meth>
void newmark_step_cuthho_interface(size_t it, double  t, typename Mesh::coordinate_type dt, typename Mesh::coordinate_type beta, typename Mesh::coordinate_type gamma, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<double, Dynamic, 1> & u_dof_n, Matrix<double, Dynamic, 1> & v_dof_n, Matrix<double, Dynamic, 1> & a_dof_n, SparseMatrix<typename Mesh::coordinate_type> & Kg, linear_solver<typename Mesh::coordinate_type> & analysis, bool write_error_Q = false);

template<typename Mesh, typename testType, typename meth>
void newmark_step_cuthho_interface(size_t it, double  t, typename Mesh::coordinate_type dt, typename Mesh::coordinate_type beta, typename Mesh::coordinate_type gamma, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<double, Dynamic, 1> & u_dof_n, Matrix<double, Dynamic, 1> & v_dof_n, Matrix<double, Dynamic, 1> & a_dof_n, SparseMatrix<typename Mesh::coordinate_type> & Kg, linear_solver<typename Mesh::coordinate_type> & analysis, bool write_error_Q)
{
    using RealType = typename Mesh::coordinate_type;
    bool write_silo_Q = false;
    auto level_set_function = test_case.level_set_;

    auto rhs_fun = test_case.rhs_fun;
    auto sol_fun = test_case.sol_fun;
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    auto dirichlet_jump = test_case.dirichlet_jump;
    auto neumann_jump = test_case.neumann_jump;
    struct params<RealType> parms = test_case.parms;
    
    timecounter tc;
    
    tc.tic();
    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
    
    if (u_dof_n.rows() == 0) {
        size_t n_dof = assembler.LHS.rows();
        u_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        v_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        a_dof_n = Matrix<RealType, Dynamic, 1>::Zero(n_dof,1);
        
    
    RealType t = 0;
    
    
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

       auto u_fun = [&t](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
           return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
       };

       auto v_fun = [&t](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
           return 2*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
       };
        
       auto a_fun = [](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
           return 2.0*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
       };

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
//         auto u_fun = [&t](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
//             return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2.0)*M_PI*t);
//         };
//         assembler.project_over_cells(msh, hdi, u_dof_n, u_fun);
// 
//         auto v_fun = [&t](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
//             return std::sqrt(2.0)*M_PI*(1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::cos(std::sqrt(2.0)*M_PI*t);
//         };
//         assembler.project_over_cells(msh, hdi, v_dof_n, v_fun);
// 
//         auto a_fun = [&t](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
//             return -2*M_PI*M_PI*(1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2)*M_PI*t);
//         };
//         assembler.project_over_cells(msh, hdi, a_dof_n, a_fun);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        
//        auto u_fun = [&t](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
//            return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//        };
//        assembler.project_over_cells(msh, hdi, u_dof_n, u_fun);
//
//        auto v_fun = [&t](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
//            return std::cos(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//        };
//        assembler.project_over_cells(msh, hdi, v_dof_n, v_fun);
//
//        auto a_fun = [&t](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
//            return -std::sqrt(2.0) * M_PI * std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
//        };
//        assembler.project_over_cells(msh, hdi, a_dof_n, a_fun);
        

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        
        size_t it = 0;
        if(write_silo_Q){
            std::string silo_file_name = "cut_hho_one_field_";
            postprocessor<Mesh>::write_silo_one_field(silo_file_name, it, msh, hdi, assembler, v_dof_n, v_fun, false);
        }
    }
    
    assembler.RHS.setZero(); // assuming null dirichlet data on boundary.
    #ifdef HAVE_INTEL_TBB
            size_t n_cells = msh.cells.size();
            tbb::parallel_for(size_t(0), size_t(n_cells), size_t(1),
                [&msh,&method,&test_case,&hdi,&assembler] (size_t & cell_ind){
                    auto& cell = msh.cells.at(cell_ind);
                    auto f = method.make_contrib_rhs(msh, cell, test_case, hdi);
                    assembler.assemble_rhs(msh, cell, f);
            }
        );
    #else
        for (auto& cell : msh.cells)
        {
            auto f = method.make_contrib_rhs(msh, cell, test_case, hdi);
            assembler.assemble_rhs(msh, cell, f);
        }
    #endif
    
    


    tc.toc();
    std::cout << bold << yellow << "RHS assembly: " << tc << " seconds" << reset << std::endl;
    
    // Compute intermediate state for scalar and rate
    u_dof_n = u_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
    v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
    Matrix<RealType, Dynamic, 1> res = Kg*u_dof_n;
    assembler.RHS -= res;
    
    tc.tic();
    a_dof_n = analysis.solve(assembler.RHS); // new acceleration
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    // update scalar and rate
    u_dof_n += beta*dt*dt*a_dof_n;
    v_dof_n += gamma*dt*a_dof_n;
    
    if(write_silo_Q){
        std::string silo_file_name = "cut_hho_one_field_";
        postprocessor<Mesh>::write_silo_one_field(silo_file_name, it, msh, hdi, assembler, u_dof_n, sol_fun, false);
    }
    
    if(write_error_Q){
       std::ofstream error_file_test("Error_file");
        postprocessor<Mesh>::compute_errors_one_field(msh, hdi, assembler, u_dof_n, sol_fun, sol_grad, error_file_test);
    }
}



#endif




