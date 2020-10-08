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
    static void write_silo_one_field(std::string silo_file_name, size_t it, Mesh & msh, hho_degree_info & hho_di, newmark_interface_assembler<Mesh, std::function<double(const typename Mesh::point_type& )>> & assembler, Matrix<double, Dynamic, 1> & x_dof,
    std::function<double(const typename Mesh::point_type& )> scal_fun, bool cell_centered_Q = false){

        timecounter tc;
        tc.tic();
        
        auto dim = 2;
        auto num_cells = msh.cells.size();
        auto num_points = msh.points.size();
        using RealType = double;
        std::vector<RealType> exact_u, approx_u;
        
        if (cell_centered_Q) {
            exact_u.reserve( num_cells );
            approx_u.reserve( num_cells );

            size_t cell_i = 0;
            for (auto& cell : msh.cells)
            {
                auto bar = barycenter(msh, cell);
                exact_u.push_back( scal_fun(bar) );
                
                // scalar evaluation
                {
                    cell_basis<cuthho_poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                    if ( location(msh, cell) == element_location::ON_INTERFACE )
                    {
                        // negative side
                        {
                            Matrix<RealType, Dynamic, 1> scalar_cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_NEGATIVE_SIDE);
                            auto t_phi = cell_basis.eval_basis( bar );
                            RealType uh = scalar_cell_dof.dot( t_phi );
                            approx_u.push_back(uh);
                        }
                        
//                        // positive side
//                        {
//                            Matrix<RealType, Dynamic, 1> scalar_cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_POSITIVE_SIDE);
//                            auto t_phi = cell_basis.eval_basis( bar );
//                            RealType uh = scalar_cell_dof.dot( t_phi );
//                            approx_u.push_back(uh);
//                        }
                        
                    }else{
                        Matrix<RealType, Dynamic, 1> scalar_cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,location(msh, cell));
                        auto t_phi = cell_basis.eval_basis( bar );
                        RealType uh = scalar_cell_dof.dot( t_phi );
                        approx_u.push_back(uh);
                    }
                    

                }
                cell_i++;
            }

        }else{

            exact_u.reserve( num_points );
            approx_u.reserve( num_points );

            // scan for selected cells, common cells are discardable
            std::map<size_t, size_t> node_to_cell;
            size_t cell_i = 0;
            for (auto& cell : msh.cells)
            {
                auto cell_nodes = nodes(msh,cell);
                size_t n_p = cell_nodes.size();
                for (size_t l = 0; l < n_p; l++)
                {
                    auto node = cell_nodes[l];
                    node_to_cell[node.ptid] = cell_i;
                }
                cell_i++;
            }
            

            for (auto& node_id : node_to_cell)
            {
                auto bar = msh.points.at(node_id.first);
                exact_u.push_back( scal_fun(bar) );

                cell_i = node_id.second;
                auto cell = msh.cells.at(cell_i);

                // scalar evaluation
                {
                    cell_basis<cuthho_poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                    if ( location(msh, cell) == element_location::ON_INTERFACE )
                    {
                        auto node = msh.nodes.at(node_id.first);
                        
                        if (location(msh, node) == element_location::IN_NEGATIVE_SIDE)
                        // negative side
                        {
                            Matrix<RealType, Dynamic, 1> scalar_cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_NEGATIVE_SIDE);
                            auto t_phi = cell_basis.eval_basis( bar );
                            RealType uh = scalar_cell_dof.dot( t_phi );
                            approx_u.push_back(uh);
                        }else
                        // positive side
                        {
                            Matrix<RealType, Dynamic, 1> scalar_cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_POSITIVE_SIDE);
                            auto t_phi = cell_basis.eval_basis( bar );
                            RealType uh = scalar_cell_dof.dot( t_phi );
                            approx_u.push_back(uh);
                        }

                    }else{
                        Matrix<RealType, Dynamic, 1> scalar_cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,location(msh, cell));
                        auto t_phi = cell_basis.eval_basis( bar );
                        RealType uh = scalar_cell_dof.dot( t_phi );
                        approx_u.push_back(uh);
                    }


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
        tc.toc();
        std::cout << std::endl;
        std::cout << bold << cyan << "Silo file rendered in : " << tc << " seconds" << reset << std::endl;
    }
    
};


#endif /* postprocessor_hpp */
