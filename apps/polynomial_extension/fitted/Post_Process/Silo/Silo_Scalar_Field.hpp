
#ifndef Silo_Scalar_Field_hpp
#define Silo_Scalar_Field_hpp


void RenderSiloFileScalarField(std::string silo_file_name, size_t it, poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
                               std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun, bool cell_centered_Q = true);

void RenderSiloFileScalarField(std::string silo_file_name, size_t it, poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
                             std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun, bool cell_centered_Q){
    
    auto num_cells = msh.cells.size();
    auto num_points = msh.points.size();
    using RealType = double;
    std::vector<RealType> exact_u, approx_u;
    std::vector<RealType> exact_dux, exact_duy, approx_dux, approx_duy;

    if (cell_centered_Q) {
        exact_u.reserve( num_cells );
        approx_u.reserve( num_cells );
        
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            auto bar = barycenter(msh, cell);
            exact_u.push_back( scal_fun(bar) );

            size_t cell_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
            // scalar evaluation
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
                auto t_phi = cell_basis.eval_basis( bar );
                RealType uh = scalar_cell_dof.dot( t_phi );
                approx_u.push_back(uh);
            }
            cell_i++;
        }
        
    }else{
        
        exact_u.reserve( num_points );
        approx_u.reserve( num_points );
        
        // scan for selected cells, common cells are discardable
        std::map<size_t, size_t> point_to_cell;
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            size_t n_p = cell.ptids.size();
            for (size_t l = 0; l < n_p; l++)
            {
                auto pt_id = cell.ptids[l];
                point_to_cell[pt_id] = cell_i;
            }
            cell_i++;
        }
        
        size_t cell_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
        for (auto& pt_id : point_to_cell)
        {
            auto bar = msh.points.at( pt_id.first );
            exact_u.push_back( scal_fun(bar) );
            
            cell_i = pt_id.second;
            auto& cell = msh.cells.at(cell_i);


            // scalar evaluation
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
                auto t_phi = cell_basis.eval_basis( bar );
                RealType uh = scalar_cell_dof.dot( t_phi );
                approx_u.push_back(uh);
            }
        }
        
    }
    
    silo_database silo;
    silo_file_name += std::to_string(it) + ".silo";
    silo.create(silo_file_name.c_str());
    silo.add_mesh(msh, "mesh");
    if (cell_centered_Q) {
        silo.add_variable("mesh", "v", exact_u.data(), exact_u.size(), zonal_variable_t);
        silo.add_variable("mesh", "vh", approx_u.data(), approx_u.size(), zonal_variable_t);
    }else{
        silo.add_variable("mesh", "v", exact_u.data(), exact_u.size(), nodal_variable_t);
        silo.add_variable("mesh", "vh", approx_u.data(), approx_u.size(), nodal_variable_t);
    }
    
    silo.close();
    
}

#endif
