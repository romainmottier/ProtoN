//
//  postprocessor.hpp
//  acoustics
//
//  Created by Omar Durán on 4/10/20.
//

#pragma once
#ifndef postprocessor_hpp
#define postprocessor_hpp

#include <iomanip>

template<typename Mesh>
class postprocessor {
    
public:
    
    
    // Write a silo file for one field approximation
    static void write_silo_one_field(std::string silo_file_name, size_t it, Mesh & msh, disk::hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
    std::function<double(const typename Mesh::point_type& )> scal_fun, bool cell_centered_Q){

        timecounter tc;
        tc.tic();
        
        auto dim = Mesh::dimension;
        auto num_cells = msh.cells_size();
        auto num_points = msh.points_size();
        using RealType = double;
        std::vector<RealType> exact_u, approx_u;
        size_t cell_dof = disk::scalar_basis_size(hho_di.cell_degree(), dim);
        
        if (cell_centered_Q) {
            exact_u.reserve( num_cells );
            approx_u.reserve( num_cells );

            size_t cell_i = 0;
            for (auto& cell : msh)
            {
                auto bar = barycenter(msh, cell);
                exact_u.push_back( scal_fun(bar) );
                
                // scalar evaluation
                {
                    auto cell_basis = make_scalar_monomial_basis(msh, cell, hho_di.cell_degree());
                    Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
                    auto t_phi = cell_basis.eval_functions( bar );
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
            for (auto& cell : msh)
            {
                auto points = cell.point_ids();
                size_t n_p = points.size();
                for (size_t l = 0; l < n_p; l++)
                {
                    auto pt_id = points[l];
                    point_to_cell[pt_id] = cell_i;
                }
                cell_i++;
            }

            for (auto& pt_id : point_to_cell)
            {
                auto bar = *std::next(msh.points_begin(), pt_id.first);
                exact_u.push_back( scal_fun(bar) );

                cell_i = pt_id.second;
                auto cell = *std::next(msh.cells_begin(), cell_i);
                // scalar evaluation
                {
                    auto cell_basis = make_scalar_monomial_basis(msh, cell, hho_di.cell_degree());
                    Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
                    auto t_phi = cell_basis.eval_functions( bar );
                    RealType uh = scalar_cell_dof.dot( t_phi );
                    approx_u.push_back(uh);
                }
            }

        }

        disk::silo_database silo;
        silo_file_name += std::to_string(it) + ".silo";
        silo.create(silo_file_name.c_str());
        silo.add_mesh(msh, "mesh");
        if (cell_centered_Q) {
            disk::silo_zonal_variable<double> v_silo("v", exact_u);
            disk::silo_zonal_variable<double> vh_silo("vh", approx_u);
            silo.add_variable("mesh", v_silo);
            silo.add_variable("mesh", vh_silo);
        }else{
            disk::silo_nodal_variable<double> v_silo("v", exact_u);
            disk::silo_nodal_variable<double> vh_silo("vh", approx_u);
            silo.add_variable("mesh", v_silo);
            silo.add_variable("mesh", vh_silo);
        }

        silo.close();
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
    }
    
};


#endif /* postprocessor_hpp */
