
#ifndef erk_step_cuthho_interface_scatter_hpp
#define erk_step_cuthho_interface_scatter_hpp

template<typename Mesh, typename testType, typename meth>
void
erk_step_cuthho_interface_scatter(size_t it, size_t s, RealType ti, RealType dt, Matrix<RealType, Dynamic, Dynamic> a, Matrix<RealType, Dynamic, Dynamic> b, Matrix<RealType, Dynamic, Dynamic> c, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> & x_dof, erk_hho_scheme<RealType> & analysis, std::ofstream &sensor_1_log, std::ofstream &sensor_2_log, std::ofstream &sensor_3_log, std::pair<typename Mesh::point_type,size_t> &s1_pt_cell, std::pair<typename Mesh::point_type,size_t> &s2_pt_cell, std::pair<typename Mesh::point_type,size_t> &s3_pt_cell);

template<typename Mesh, typename testType, typename meth>
void
erk_step_cuthho_interface_scatter(size_t it, size_t s, RealType ti, RealType dt, Matrix<RealType, Dynamic, Dynamic> a, Matrix<RealType, Dynamic, Dynamic> b, Matrix<RealType, Dynamic, Dynamic> c, Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, Matrix<RealType, Dynamic, 1> & x_dof, erk_hho_scheme<RealType> & analysis, std::ofstream &sensor_1_log, std::ofstream &sensor_2_log, std::ofstream &sensor_3_log, std::pair<typename Mesh::point_type,size_t> &s1_pt_cell, std::pair<typename Mesh::point_type,size_t> &s2_pt_cell, std::pair<typename Mesh::point_type,size_t> &s3_pt_cell){
    
    
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
    auto assembler = make_two_fields_interface_assembler(msh, bcs_fun, hdi);
    
    if (x_dof.rows() == 0) {
        RealType t = ti;

        auto vel_fun = [](const typename Mesh::point_type& pt) -> RealType {
            return 0.0;
        };

        auto flux_fun = [](const typename Mesh::point_type& pt) -> Matrix<RealType, 1, 2> {
            Matrix<RealType, 1, 2> v;
            RealType x,y,xc,yc,r,wave,vx,vy,c,lp;
            x = pt.x();
            y = pt.y();
            xc = 0.0;
            yc = 0.0;//2.0/3.0;
            c = 10.0;
            lp = std::sqrt(9.0)/10.0;
            r = std::sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc));
            wave = (c)/(std::exp((1.0/(lp*lp))*r*r*M_PI*M_PI));
            vx = -wave*(x-xc);
            vy = -wave*(y-yc);
            v(0) = vx;
            v(1) = vy;
            return v;
        };
        
        assembler.project_over_cells(msh, hdi, x_dof, vel_fun, flux_fun);
        
        size_t it = 0;
        if(write_silo_Q){
            std::string silo_file_name = "cut_hho_two_fields_";
            postprocessor<Mesh>::write_silo_two_fields(silo_file_name, it, msh, hdi, assembler, x_dof, vel_fun, false);
        }
        
        postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s1_pt_cell, msh, hdi, assembler, x_dof, sensor_1_log);
        postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s2_pt_cell, msh, hdi, assembler, x_dof, sensor_2_log);
        postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s3_pt_cell, msh, hdi, assembler, x_dof, sensor_3_log);
        
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

    if(write_silo_Q){
        if (it % 64 == 0){
            std::string silo_file_name = "cut_hho_e_two_fields_";
            postprocessor<Mesh>::write_silo_two_fields(silo_file_name, it, msh, hdi, assembler, x_dof, sol_fun, false);
        }
    }
    
    postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s1_pt_cell, msh, hdi, assembler, x_dof, sensor_1_log);
    postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s2_pt_cell, msh, hdi, assembler, x_dof, sensor_2_log);
    postprocessor<mesh_type>::record_data_acoustic_two_fields(it, s3_pt_cell, msh, hdi, assembler, x_dof, sensor_3_log);
    
}

#endif
