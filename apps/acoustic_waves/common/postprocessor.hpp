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
    static void write_silo_one_field(std::string silo_file_name, size_t it, Mesh & msh, hho_degree_info & hho_di, one_field_interface_assembler<Mesh, std::function<double(const typename Mesh::point_type& )>> & assembler, Matrix<double, Dynamic, 1> & x_dof,
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
    
    // Write a silo file for two fields approximation
    static void write_silo_two_fields(std::string silo_file_name, size_t it, Mesh & msh, hho_degree_info & hho_di, two_fields_interface_assembler<Mesh, std::function<double(const typename Mesh::point_type& )>> & assembler, Matrix<double, Dynamic, 1> & x_dof,
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
                    auto cbs = cell_basis.size();
                    if ( location(msh, cell) == element_location::ON_INTERFACE )
                    {
                        // negative side
                        {
                            Matrix<RealType, Dynamic, 1> cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_NEGATIVE_SIDE);
                            Matrix<RealType, Dynamic, 1> scal_cell_dof = cell_dof.tail(cbs);
                            auto t_phi = cell_basis.eval_basis( bar );
                            RealType uh = scal_cell_dof.dot( t_phi );
                            approx_u.push_back(uh);
                        }
                        
//                        // positive side
//                        {
//                            Matrix<RealType, Dynamic, 1> cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_POSITIVE_SIDE);
//                            Matrix<RealType, Dynamic, 1> scal_cell_dof = cell_dof.tail(cbs);
//                            auto t_phi = cell_basis.eval_basis( bar );
//                            RealType uh = scal_cell_dof.dot( t_phi );
//                            approx_u.push_back(uh);
//                        }
                        
                    }else{
                        Matrix<RealType, Dynamic, 1> cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,location(msh, cell));
                        Matrix<RealType, Dynamic, 1> scal_cell_dof = cell_dof.tail(cbs);
                        auto t_phi = cell_basis.eval_basis( bar );
                        RealType uh = scal_cell_dof.dot( t_phi );
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
                    auto cbs = cell_basis.size();
                    if ( location(msh, cell) == element_location::ON_INTERFACE )
                    {
                        auto node = msh.nodes.at(node_id.first);
                        
                        if (location(msh, node) == element_location::IN_NEGATIVE_SIDE)
                        // negative side
                        {
                            Matrix<RealType, Dynamic, 1> cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_NEGATIVE_SIDE);
                            Matrix<RealType, Dynamic, 1> scal_cell_dof = cell_dof.tail(cbs);
                            
                            auto t_phi = cell_basis.eval_basis( bar );
                            RealType uh = scal_cell_dof.dot( t_phi );
                            approx_u.push_back(uh);
                        }else
                        // positive side
                        {
                            Matrix<RealType, Dynamic, 1> cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_POSITIVE_SIDE);
                            Matrix<RealType, Dynamic, 1> scal_cell_dof = cell_dof.tail(cbs);
                            auto t_phi = cell_basis.eval_basis( bar );
                            RealType uh = scal_cell_dof.dot( t_phi );
                            approx_u.push_back(uh);
                        }

                    }else{
                        Matrix<RealType, Dynamic, 1> cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,location(msh, cell));
                        Matrix<RealType, Dynamic, 1> scal_cell_dof = cell_dof.tail(cbs);
                        auto t_phi = cell_basis.eval_basis( bar );
                        RealType uh = scal_cell_dof.dot( t_phi );
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
    
    /// Compute L2 and H1 errors for one field approximation
    static void compute_errors_one_field(Mesh & msh, hho_degree_info & hho_di, one_field_interface_assembler<Mesh, std::function<double(const typename Mesh::point_type& )>> & assembler, Matrix<double, Dynamic, 1> & x_dof,std::function<double(const typename Mesh::point_type& )> scal_fun, std::function<Matrix<double, 1, 2>(const typename Mesh::point_type& )> flux_fun, std::ostream & error_file = std::cout){

       timecounter tc;
       tc.tic();

       using RealType = double;

       RealType scalar_l2_error = 0.0;
       RealType flux_l2_error = 0.0;
       size_t cell_i = 0;
       RealType h = 10.0;
       std::vector<RealType> l2_error_vec(msh.cells.size());
       std::vector<RealType> flux_l2_error_vec(msh.cells.size());
       for (auto& cell : msh.cells)
       {
           auto offset_cl = offset(msh,cell);
           std::cout << "Postprocess of Uncut cells: " << offset_cl << std::endl;

           l2_error_vec[cell_i] = 0.0;
            RealType h_l = diameter(msh, cell);
           if (h_l < h) {
               h = h_l;
           }
           
           cell_basis<cuthho_poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
           auto cbs = cell_basis.size();
           if ( location(msh, cell) == element_location::ON_INTERFACE )
           {
               
               auto dofs_n = assembler.take_local_data(msh, cell, x_dof, element_location::IN_NEGATIVE_SIDE);
               auto dofs_p = assembler.take_local_data(msh, cell, x_dof, element_location::IN_POSITIVE_SIDE);

               auto cell_dofs_n = dofs_n.head(cbs);
               auto cell_dofs_p = dofs_p.head(cbs);
               
               // negative side
               auto qps_n = integrate(msh, cell, 2*hho_di.cell_degree(), element_location::IN_NEGATIVE_SIDE);
               for (auto& qp : qps_n)
               {
                   /* Compute H1-error */
                   auto t_dphi = cell_basis.eval_gradients( qp.first );
                   Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                   for (size_t i = 1; i < cbs; i++ )
                       grad += cell_dofs_n(i) * t_dphi.block(i, 0, 1, 2);

                   flux_l2_error_vec[cell_i] += qp.second * (flux_fun(qp.first) - grad).dot(flux_fun(qp.first) - grad);
                   

                   auto t_phi = cell_basis.eval_basis( qp.first );
                   auto v = cell_dofs_n.dot(t_phi);
                   
                   /* Compute L2-error */
                   l2_error_vec[cell_i] += qp.second * (scal_fun(qp.first) - v) * (scal_fun(qp.first) - v);
               }
               
               // positive side
               auto qps_p = integrate(msh, cell, 2*hho_di.cell_degree(), element_location::IN_POSITIVE_SIDE);
               for (auto& qp : qps_p)
               {
                   /* Compute H1-error */
                   auto t_dphi = cell_basis.eval_gradients( qp.first );
                   Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                   for (size_t i = 1; i < cbs; i++ )
                       grad += cell_dofs_n(i) * t_dphi.block(i, 0, 1, 2);

                   flux_l2_error_vec[cell_i] += qp.second * (flux_fun(qp.first) - grad).dot(flux_fun(qp.first) - grad);

                   auto t_phi = cell_basis.eval_basis( qp.first );
                   auto v = cell_dofs_n.dot(t_phi);
                   
                   /* Compute L2-error */
                   l2_error_vec[cell_i] += qp.second * (scal_fun(qp.first) - v) * (scal_fun(qp.first) - v);
               }

           }else{
               
               auto dofs = assembler.take_local_data(msh, cell, x_dof);
                auto cell_dofs = dofs.head(cbs);

               // uncut case
               auto qps = integrate(msh, cell, 2*hho_di.cell_degree());
               for (auto& qp : qps)
               {
                   /* Compute H1-error */
                   auto t_dphi = cell_basis.eval_gradients( qp.first );
                   Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                   for (size_t i = 1; i < cbs; i++ )
                       grad += cell_dofs(i) * t_dphi.block(i, 0, 1, 2);

                   flux_l2_error_vec[cell_i] += qp.second * (flux_fun(qp.first) - grad).dot(flux_fun(qp.first) - grad);

                   auto t_phi = cell_basis.eval_basis( qp.first );
                   auto v = cell_dofs.dot(t_phi);
                   
                   /* Compute L2-error */
                   l2_error_vec[cell_i] += qp.second * (scal_fun(qp.first) - v) * (scal_fun(qp.first) - v);
                   
               }
           }
           cell_i++;
       }
       
       scalar_l2_error = std::accumulate(l2_error_vec.begin(), l2_error_vec.end(),0.0);
       flux_l2_error   = std::accumulate(flux_l2_error_vec.begin(), flux_l2_error_vec.end(),0.0);
       tc.toc();
       
       std::cout << bold << cyan << "Error completed: " << tc << " seconds" << reset << std::endl;
       error_file << "Characteristic h size = " << std::setprecision(16) << h << std::endl;
       error_file << "L2-norm error = " << std::setprecision(16) << std::sqrt(scalar_l2_error) << std::endl;
       error_file << "H1-norm error = " << std::setprecision(16) << std::sqrt(flux_l2_error) << std::endl;
       error_file << std::endl;
       error_file.flush();
       
    }
    
    /// Compute L2 and H1 errors for two fields approximation
    static void compute_errors_two_fields(Mesh & msh, hho_degree_info & hho_di, two_fields_interface_assembler<Mesh, std::function<double(const typename Mesh::point_type& )>> & assembler, Matrix<double, Dynamic, 1> & x_dof,std::function<double(const typename Mesh::point_type& )> scal_fun, std::function<Matrix<double, 1, 2>(const typename Mesh::point_type& )> flux_fun, std::ostream & error_file = std::cout){

        timecounter tc;
        tc.tic();

        using RealType = double;
 
        RealType scalar_l2_error = 0.0;
        RealType flux_l2_error = 0.0;
        size_t cell_i = 0;
        RealType h = 10.0;
        std::vector<RealType> l2_error_vec(msh.cells.size());
        std::vector<RealType> flux_l2_error_vec(msh.cells.size());
        for (auto& cell : msh.cells)
        {
            l2_error_vec[cell_i] = 0.0;
             RealType h_l = diameter(msh, cell);
            if (h_l < h) {
                h = h_l;
            }

            
            {
                cell_basis<cuthho_poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> vec_cell_basis(msh, cell, hho_di.grad_degree());
                
                auto cbs = cell_basis.size();
                auto gbs = vec_cell_basis.size();
                if ( location(msh, cell) == element_location::ON_INTERFACE )
                {
                    
                    Matrix<RealType, Dynamic, 1> cell_dof_n = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_NEGATIVE_SIDE);
                    Matrix<RealType, Dynamic, 1> cell_dof_p = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_POSITIVE_SIDE);
                    
                    // negative side
                    auto qps_n = integrate(msh, cell, 2*hho_di.cell_degree(), element_location::IN_NEGATIVE_SIDE);
                    for (auto& qp : qps_n)
                    {
                        // scalar evaluation
                        Matrix<RealType, Dynamic, 1> scal_cell_dof = cell_dof_n.tail(cbs);
                        auto t_phi = cell_basis.eval_basis( qp.first );
                        RealType uh = scal_cell_dof.dot( t_phi );
                        l2_error_vec[cell_i] += qp.second * (scal_fun(qp.first) - uh) * (scal_fun(qp.first) - uh);
                    
                        // flux evaluation
                        Matrix<RealType, Dynamic, 1> vec_cell_dof = cell_dof_n.head(gbs);
                        auto t_phi_v = vec_cell_basis.eval_basis( qp.first );
                        Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
                        for (size_t i = 0; i < t_phi_v.rows(); i++){
                          grad_uh = grad_uh + vec_cell_dof(i)*t_phi_v.block(i, 0, 1, 2);
                        }
                        auto grad_u_exact = flux_fun(qp.first);
                        flux_l2_error_vec[cell_i] += qp.second * (grad_u_exact - grad_uh).dot(grad_u_exact - grad_uh);
                        

                    }
                    
                    // positive side
                    auto qps_p = integrate(msh, cell, 2*hho_di.cell_degree(), element_location::IN_POSITIVE_SIDE);
                    for (auto& qp : qps_p)
                    {
                        // scalar evaluation
                        Matrix<RealType, Dynamic, 1> scal_cell_dof = cell_dof_p.tail(cbs);
                        auto t_phi = cell_basis.eval_basis( qp.first );
                        RealType uh = scal_cell_dof.dot( t_phi );
                        l2_error_vec[cell_i] += qp.second * (scal_fun(qp.first) - uh) * (scal_fun(qp.first) - uh);
                        
                        // flux evaluation
                        Matrix<RealType, Dynamic, 1> vec_cell_dof = cell_dof_p.head(gbs);
                        auto t_phi_v = vec_cell_basis.eval_basis( qp.first );
                        Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
                        for (size_t i = 0; i < t_phi_v.rows(); i++){
                          grad_uh = grad_uh + vec_cell_dof(i)*t_phi_v.block(i, 0, 1, 2);
                        }
                        auto grad_u_exact = flux_fun(qp.first);
                        flux_l2_error_vec[cell_i] += qp.second * (grad_u_exact - grad_uh).dot(grad_u_exact - grad_uh);
                        
                    }

                }else{
                    
                    // uncut case
                    // scalar evaluation
                    Matrix<RealType, Dynamic, 1> cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,location(msh, cell));
                    auto qps = integrate(msh, cell, 2*hho_di.cell_degree());
                    for (auto& qp : qps)
                    {
                        Matrix<RealType, Dynamic, 1> scal_cell_dof = cell_dof.tail(cbs);
                        auto t_phi = cell_basis.eval_basis( qp.first );
                        RealType uh = scal_cell_dof.dot( t_phi );
                        l2_error_vec[cell_i] += qp.second * (scal_fun(qp.first) - uh) * (scal_fun(qp.first) - uh);
                        
                        // flux evaluation
                        Matrix<RealType, Dynamic, 1> vec_cell_dof = cell_dof.head(gbs);
                        auto t_phi_v = vec_cell_basis.eval_basis( qp.first );
                        Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
                        for (size_t i = 0; i < t_phi_v.rows(); i++){
                          grad_uh = grad_uh + vec_cell_dof(i)*t_phi_v.block(i, 0, 1, 2);
                        }
                        auto grad_u_exact = flux_fun(qp.first);
                        flux_l2_error_vec[cell_i] += qp.second * (grad_u_exact - grad_uh).dot(grad_u_exact - grad_uh);
                        
                    }
                }

            }
            cell_i++;
        }
        scalar_l2_error = std::accumulate(l2_error_vec.begin(), l2_error_vec.end(),0.0);
        flux_l2_error = std::accumulate(flux_l2_error_vec.begin(), flux_l2_error_vec.end(),0.0);
        tc.toc();
        
        std::cout << bold << cyan << "Error completed: " << tc << " seconds" << reset << std::endl;
        error_file << "Characteristic h size = " << std::setprecision(16) << h << std::endl;
        error_file << "L2-norm error = " << std::setprecision(16) << std::sqrt(scalar_l2_error) << std::endl;
        error_file << "H1-norm error = " << std::setprecision(16) << std::sqrt(flux_l2_error) << std::endl;
        error_file << std::endl;
        error_file.flush();
        
    }
    
    /// Find the cells associated to the requested point
    static std::set<size_t> find_cells(typename Mesh::point_type & pt, Mesh & msh, bool verbose_Q = false){
        
        using RealType = double;
        auto norm =  [](const typename Mesh::point_type& a, const typename Mesh::point_type& b ) -> RealType {
            RealType dx = (b.x() - a.x());
            RealType dy = (b.y() - a.y());
            RealType norm = std::sqrt(dx*dx + dy*dy);
            return norm;
        };
        
        // find minimum distance to the requested point
        size_t np = msh.points.size();
        std::vector<RealType> distances(np);
        
        size_t ip = 0;
        for (auto& point : msh.points)
        {
            RealType dist = norm(pt,point);
            distances[ip] = dist;
            ip++;
        }
        
        size_t index = std::min_element(distances.begin(),distances.end()) - distances.begin();
        if(verbose_Q){
            RealType min_dist = *std::min_element(distances.begin(), distances.end());
            typename Mesh::point_type nearest_point = msh.points.at(index);
            std::cout << "Nearest point detected : " << std::endl;
            std::cout << "  x =  " << nearest_point.x() << std::endl;
            std::cout << "  y =  " << nearest_point.y() << std::endl;
            std::cout << "Distance = " << min_dist << std::endl;
            std::cout << "Global index = " << index << std::endl;
        }
        
        std::set<size_t> cell_indexes;
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            auto cell_nodes = nodes(msh,cell);
            size_t n_p = cell_nodes.size();
            for (size_t l = 0; l < n_p; l++)
            {
                auto node = cell_nodes[l];
                if(index == node.ptid){
                    cell_indexes.insert(cell_i);
                }
            }
            cell_i++;
        }
        
        if(verbose_Q){
            std::cout << "Detected cells indexes : " << std::endl;
            for(auto index : cell_indexes){
                std::cout << index << std::endl;
            }
        }

        return cell_indexes;
    }
    
    /// Pick the cell that contains the requested point
    static size_t pick_cell(typename Mesh::point_type & pt, Mesh & msh, std::set<size_t> & cell_indexes, bool verbose_Q = false){
        
        using RealType = double;
        
        auto triangle_member_Q = [] (typename Mesh::point_type & p, typename Mesh::point_type & p0, typename Mesh::point_type & p1, typename Mesh::point_type & p2)
        {
            RealType dx = p.x()-p2.x();
            RealType dy = p.y()-p2.y();
            RealType dx21 = p2.x()-p1.x();
            RealType dy12 = p1.y()-p2.y();
            RealType d = dy12*(p0.x()-p2.x()) + dx21*(p0.y()-p2.y());
            RealType s = dy12*dx + dx21*dy;
            RealType t = (p2.y()-p0.y())*dx + (p0.x()-p2.x())*dy;
            if (d < 0.0) {
                return s<=0.0 && t<=0.0 && s+t>=d;
            }
            return s>=0 && t>=0 && s+t<=d;
        };
        
        size_t n_cells = cell_indexes.size();
        if (n_cells == 1) {
            size_t first_index = *cell_indexes.begin();
            return first_index;
        }
        bool is_member_Q = false;
        for(auto index : cell_indexes){
            auto& cell = msh.cells.at(index);
            auto bar = barycenter(msh, cell);
            auto cell_nodes = nodes(msh,cell);
            size_t n_p = cell_nodes.size();
            
            // building teselation
            std::vector<std::vector<typename Mesh::point_type>> triangles(n_p);
            for (size_t l = 0; l < n_p; l++)
            {

                std::vector<typename Mesh::point_type> chunk(3);
                if( l == n_p - 1){
                    chunk[0] = msh.points.at(cell_nodes[l].ptid);
                    chunk[1] = msh.points.at(cell_nodes[0].ptid);
                    chunk[2] = bar;
                }else{
                    chunk[0] = msh.points.at(cell_nodes[l].ptid);
                    chunk[1] = msh.points.at(cell_nodes[l+1].ptid);
                    chunk[2] = bar;
                }
                triangles[l] = chunk;
            }
            
            // check whether the point is memeber of any triangle
            for (auto triangle : triangles) {
                is_member_Q = triangle_member_Q(pt,triangle[0],triangle[1],triangle[2]);
                if (is_member_Q) {
                    std::cout << "Detected cell index = " << index << std::endl;
                    return index;
                }
            }

        }
        
        if(!is_member_Q){
            if(verbose_Q){
                std::cout << "Point is not member of cells set. Returning cell_indexes[0] " << std::endl;
            }
            size_t first_index = *cell_indexes.begin();
            return first_index;
        }
        
        return -1;
    }
    
    /// Record data at provided point for one field approximation
    static void record_data_acoustic_one_field(size_t it, std::pair<typename Mesh::point_type,size_t> & pt_cell_index, Mesh & msh, hho_degree_info & hho_di, one_field_interface_assembler<Mesh, std::function<double(const typename Mesh::point_type& )>> & assembler, Matrix<double, Dynamic, 1> & x_dof, std::ostream & seismogram_file = std::cout){

        timecounter tc;
        tc.tic();

        using RealType = double;
        auto dim = 2;

        Matrix<double, Dynamic, 1> vh = Matrix<double, Dynamic, 1>::Zero(2, 1);

        typename Mesh::point_type pt = pt_cell_index.first;
        
        if(pt_cell_index.second == -1){
            std::set<size_t> cell_indexes = find_cells(pt, msh, true);
            size_t cell_index = pick_cell(pt, msh, cell_indexes, true);
            assert(cell_index != -1);
            pt_cell_index.second = cell_index;
            seismogram_file << "\"Time\"" << "," << "\"vhx\"" << "," << "\"vhy\"" << std::endl;
        }

        {
            size_t cell_ind = pt_cell_index.second;
            // scalar evaluation
            auto cell = msh.cells.at(cell_ind);

            // scalar evaluation
            {
                cell_basis<cuthho_poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                if ( location(msh, cell) == element_location::ON_INTERFACE )
                {
                    auto node = msh.nodes.at(0);
                    throw std::invalid_argument("Recoding at cut cell. Not implemented.");
                    if (location(msh, node) == element_location::IN_NEGATIVE_SIDE)
                    // negative side
                    {
                        Matrix<RealType, Dynamic, 1> scalar_cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_NEGATIVE_SIDE);
                        auto t_dphi = cell_basis.eval_gradients( pt );
                        size_t cbs = cell_basis.size();
                           Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();
                           for (size_t i = 1; i < cbs; i++ )
                               grad += scalar_cell_dof(i) * t_dphi.block(i, 0, 1, 2);
                        vh = grad;
                    }else
                    // positive side
                    {
                        Matrix<RealType, Dynamic, 1> scalar_cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_POSITIVE_SIDE);
                        auto t_dphi = cell_basis.eval_gradients( pt );
                        size_t cbs = cell_basis.size();
                           Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();
                           for (size_t i = 1; i < cbs; i++ )
                               grad += scalar_cell_dof(i) * t_dphi.block(i, 0, 1, 2);
                        vh = grad;
                    }

                }else{
                    Matrix<RealType, Dynamic, 1> scalar_cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,location(msh, cell));
                    auto t_dphi = cell_basis.eval_gradients( pt );
                    size_t cbs = cell_basis.size();
                       Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();
                       for (size_t i = 1; i < cbs; i++ )
                           grad += scalar_cell_dof(i) * t_dphi.block(i, 0, 1, 2);
                    
                    vh = grad;
                }


            }
        }
        tc.toc();
        std::cout << bold << cyan << "Value recorded: " << tc << " seconds" << reset << std::endl;
        seismogram_file << it << "," << std::setprecision(16) <<  vh(0,0) << "," << std::setprecision(16) <<  vh(1,0) << std::endl;
        seismogram_file.flush();

    }
    
    /// Record data at provided point for two fields approximation
    static void record_data_acoustic_two_fields(size_t it, std::pair<typename Mesh::point_type,size_t> & pt_cell_index, Mesh & msh, hho_degree_info & hho_di, two_fields_interface_assembler<Mesh, std::function<double(const typename Mesh::point_type& )>> & assembler, Matrix<double, Dynamic, 1> & x_dof, std::ostream & seismogram_file = std::cout){

        timecounter tc;
        tc.tic();

        using RealType = double;
        auto dim = 2;

        Matrix<double, Dynamic, 1> vh = Matrix<double, Dynamic, 1>::Zero(2, 1);

        typename Mesh::point_type pt = pt_cell_index.first;
        
        if(pt_cell_index.second == -1){
            std::set<size_t> cell_indexes = find_cells(pt, msh, true);
            size_t cell_index = pick_cell(pt, msh, cell_indexes, true);
            assert(cell_index != -1);
            pt_cell_index.second = cell_index;
            seismogram_file << "\"Time\"" << "," << "\"vhx\"" << "," << "\"vhy\"" << std::endl;
        }

        {
            size_t cell_ind = pt_cell_index.second;
            auto cell = msh.cells.at(cell_ind);

            // flux evaluation
            {
                vector_cell_basis<cuthho_poly_mesh<RealType>, RealType> vec_cell_basis(msh, cell, hho_di.grad_degree());
                auto gbs = vec_cell_basis.size();
                
                if ( location(msh, cell) == element_location::ON_INTERFACE )
                {
                    auto node = msh.nodes.at(0);
                    throw std::invalid_argument("Recoding at cut cell. Not implemented.");
                    Matrix<RealType, Dynamic, 1> cell_dof_n = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_NEGATIVE_SIDE);
                    Matrix<RealType, Dynamic, 1> cell_dof_p = assembler.gather_cell_dof(msh,cell,x_dof,element_location::IN_POSITIVE_SIDE);
                    
                    if (location(msh, node) == element_location::IN_NEGATIVE_SIDE)
                    // negative side
                    {
                        Matrix<RealType, Dynamic, 1> vec_cell_dof = cell_dof_n.head(gbs);
                        auto t_phi_v = vec_cell_basis.eval_basis( pt );
                        Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
                        for (size_t i = 0; i < t_phi_v.rows(); i++){
                          grad_uh = grad_uh + vec_cell_dof(i)*t_phi_v.block(i, 0, 1, 2);
                        }
                        vh = grad_uh;
                    }else
                    // positive side
                    {
                        Matrix<RealType, Dynamic, 1> vec_cell_dof = cell_dof_p.head(gbs);
                        auto t_phi_v = vec_cell_basis.eval_basis( pt );
                        Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
                        for (size_t i = 0; i < t_phi_v.rows(); i++){
                          grad_uh = grad_uh + vec_cell_dof(i)*t_phi_v.block(i, 0, 1, 2);
                        }
                        vh = grad_uh;
                    }

                }else{
                    Matrix<RealType, Dynamic, 1> cell_dof = assembler.gather_cell_dof(msh,cell,x_dof,location(msh, cell));
                    
                    Matrix<RealType, Dynamic, 1> vec_cell_dof = cell_dof.head(gbs);
                    auto t_phi_v = vec_cell_basis.eval_basis( pt );
                    Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
                    for (size_t i = 0; i < t_phi_v.rows(); i++){
                      grad_uh = grad_uh + vec_cell_dof(i)*t_phi_v.block(i, 0, 1, 2);
                    }
                    vh = grad_uh;
                }


            }
        }
        tc.toc();
        std::cout << bold << cyan << "Value recorded: " << tc << " seconds" << reset << std::endl;
        seismogram_file << it << "," << std::setprecision(16) <<  vh(0,0) << "," << std::setprecision(16) <<  vh(1,0) << std::endl;
        seismogram_file.flush();

    }
    
};


#endif /* postprocessor_hpp */
