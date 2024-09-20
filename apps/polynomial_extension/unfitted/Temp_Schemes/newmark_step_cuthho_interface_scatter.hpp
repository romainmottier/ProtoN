
#ifndef newmark_step_cuthho_interface_scatter_hpp
#define newmark_step_cuthho_interface_scatter_hpp

template<typename Mesh, typename testType, typename meth>
void newmark_step_cuthho_interface_scatter(size_t it, typename Mesh::coordinate_type dt, typename Mesh::coordinate_type beta, typename Mesh::coordinate_type gamma, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<double, Dynamic, 1> & u_dof_n, Matrix<double, Dynamic, 1> & v_dof_n, Matrix<double, Dynamic, 1> & a_dof_n, SparseMatrix<typename Mesh::coordinate_type> & Kg, linear_solver<typename Mesh::coordinate_type> & analysis, std::ofstream &sensor_1_log, std::ofstream &sensor_2_log, std::ofstream &sensor_3_log, std::pair<typename Mesh::point_type,size_t> &s1_pt_cell, std::pair<typename Mesh::point_type,size_t> &s2_pt_cell, std::pair<typename Mesh::point_type,size_t> &s3_pt_cell);

template<typename Mesh, typename testType, typename meth>
void newmark_step_cuthho_interface_scatter(size_t it, typename Mesh::coordinate_type dt, typename Mesh::coordinate_type beta, typename Mesh::coordinate_type gamma, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<double, Dynamic, 1> & u_dof_n, Matrix<double, Dynamic, 1> & v_dof_n, Matrix<double, Dynamic, 1> & a_dof_n, SparseMatrix<typename Mesh::coordinate_type> & Kg, linear_solver<typename Mesh::coordinate_type> & analysis, std::ofstream &sensor_1_log, std::ofstream &sensor_2_log, std::ofstream &sensor_3_log, std::pair<typename Mesh::point_type,size_t> &s1_pt_cell, std::pair<typename Mesh::point_type,size_t> &s2_pt_cell, std::pair<typename Mesh::point_type,size_t> &s3_pt_cell)
{
    using RealType = typename Mesh::coordinate_type;
    bool write_silo_Q = true;
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
        auto u_fun = test_case.sol_fun;
        
//        auto u_fun = [](const typename Mesh::point_type& pt) -> typename Mesh::coordinate_type {
//            RealType x,y,xc,yc,r,wave,vx,vy,v,c,lp,factor;
//            x = pt.x();
//            y = pt.y();
//            xc = 0.0;
//            yc = 2.0/3.0;
//            c = 10.0;
//            lp = std::sqrt(9.0)/10.0;
//            r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
//            wave = (c)/(std::exp((1.0/(lp*lp))*r*r*M_PI*M_PI));
//            factor = (lp*lp/(2.0*M_PI*M_PI));
//            return factor*wave;
//        };

        assembler.project_over_cells(msh, hdi, u_dof_n, u_fun);
        
        size_t it = 0;
        if(write_silo_Q){
            std::string silo_file_name = "cut_hho_one_field_";
            postprocessor<Mesh>::write_silo_one_field(silo_file_name, it, msh, hdi, assembler, v_dof_n, sol_fun, false);
        }
        
        postprocessor<mesh_type>::record_data_acoustic_one_field(it, s1_pt_cell, msh, hdi, assembler, u_dof_n, sensor_1_log);
        postprocessor<mesh_type>::record_data_acoustic_one_field(it, s2_pt_cell, msh, hdi, assembler, u_dof_n, sensor_2_log);
        postprocessor<mesh_type>::record_data_acoustic_one_field(it, s3_pt_cell, msh, hdi, assembler, u_dof_n, sensor_3_log);
        
    }
    
    assembler.RHS = 0.0*u_dof_n;
//    assembler.RHS.setZero(); // assuming null dirichlet data on boundary.
//    for (auto& cl : msh.cells)
//    {
//        auto f = method.make_contrib_rhs(msh, cl, test_case, hdi);
//        assembler.assemble_rhs(msh, cl, f);
//    }

    tc.toc();
    std::cout << bold << yellow << "RHS assembly: " << tc << " seconds" << reset << std::endl;
    
    // Compute intermediate state for scalar and rate
    u_dof_n = u_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
    v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
    Matrix<RealType, Dynamic, 1> res = Kg*u_dof_n;
    assembler.RHS -= res;
    
    std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    tc.tic();
    a_dof_n = analysis.solve(assembler.RHS); // new acceleration
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    // update scalar and rate
    u_dof_n += beta*dt*dt*a_dof_n;
    v_dof_n += gamma*dt*a_dof_n;
    
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    
    if(write_silo_Q){
        std::string silo_file_name = "cut_hho_one_field_";
        postprocessor<Mesh>::write_silo_one_field(silo_file_name, it, msh, hdi, assembler, v_dof_n, sol_fun, false);
    }
    
    postprocessor<mesh_type>::record_data_acoustic_one_field(it, s1_pt_cell, msh, hdi, assembler, u_dof_n, sensor_1_log);
    postprocessor<mesh_type>::record_data_acoustic_one_field(it, s2_pt_cell, msh, hdi, assembler, u_dof_n, sensor_2_log);
    postprocessor<mesh_type>::record_data_acoustic_one_field(it, s3_pt_cell, msh, hdi, assembler, u_dof_n, sensor_3_log);
    
}

#endif
