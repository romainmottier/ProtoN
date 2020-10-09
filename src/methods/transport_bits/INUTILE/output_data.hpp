/*
 *       /\        Matteo Cicuttin (C) 2017,2018; Guillaume Delay 2018,2019
 *      /__\       matteo.cicuttin@enpc.fr        guillaume.delay@enpc.fr
 *     /_\/_\      École Nationale des Ponts et Chaussées - CERMICS
 *    /\    /\
 *   /__\  /__\    This is ProtoN, a library for fast Prototyping of
 *  /_\/_\/_\/_\   Numerical methods.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * If you use this code or parts of it for scientific publications, you
 * are required to cite it as following:
 *
 * Implementation of Discontinuous Skeletal methods on arbitrary-dimensional,
 * polytopal meshes using generic programming.
 * M. Cicuttin, D. A. Di Pietro, A. Ern.
 * Journal of Computational and Applied Mathematics.
 * DOI: 10.1016/j.cam.2017.09.017
 */

/// Useful to plot level set pre FE transport problem
/// in cuthho_export.hpp
template<typename Mesh, typename Function>
void
output_mesh_info2_pre_FEM(const Mesh& msh, const Function& level_set_function)
{
    using RealType = typename Mesh::coordinate_type;

    /************** OPEN SILO DATABASE **************/
    silo_database silo;
    silo.create("cuthho_meshinfo_preFEM_Stokes.silo");
    silo.add_mesh(msh, "mesh");

    /************** MAKE A SILO VARIABLE FOR CELL POSITIONING **************/
    std::vector<RealType> cut_cell_markers;
    for (auto& cl : msh.cells)
    {
        if ( location(msh, cl) == element_location::IN_POSITIVE_SIDE )
            cut_cell_markers.push_back(1.0);
        else if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            cut_cell_markers.push_back(-1.0);
        else if ( location(msh, cl) == element_location::ON_INTERFACE )
            cut_cell_markers.push_back(0.0);
        else
            throw std::logic_error("shouldn't have arrived here...");
    }
    silo.add_variable("mesh", "cut_cells", cut_cell_markers.data(), cut_cell_markers.size(), zonal_variable_t);

    /************** MAKE A SILO VARIABLE FOR CELL HIGHLIGHT **************/
    std::vector<RealType> highlight_markers;
    for (auto& cl : msh.cells)
    {
        if ( cl.user_data.highlight )
            highlight_markers.push_back(1.0);
        else
            highlight_markers.push_back(0.0);

    }
    silo.add_variable("mesh", "highlighted_cells", highlight_markers.data(), highlight_markers.size(), zonal_variable_t);

    /************** MAKE A SILO VARIABLE FOR LEVEL SET FUNCTION **************/
    std::vector<RealType> level_set_vals;
    // for (auto& pt : msh.points)
    //    level_set_vals.push_back( level_set_function(pt) );
    for (auto& n : msh.nodes)
        level_set_vals.push_back( level_set_function(n) );
    
    silo.add_variable("mesh", "level_set", level_set_vals.data(), level_set_vals.size(), nodal_variable_t);

    /************** MAKE A SILO VARIABLE FOR NODE POSITIONING **************/
    std::vector<RealType> node_pos;
    for (auto& n : msh.nodes)
        node_pos.push_back( location(msh, n) == element_location::IN_POSITIVE_SIDE ? +1.0 : -1.0 );
    silo.add_variable("mesh", "node_pos", node_pos.data(), node_pos.size(), nodal_variable_t);

    std::vector<RealType> cell_set;
    for (auto& cl : msh.cells)
    {
        RealType r;

        switch ( cl.user_data.agglo_set )
        {
            case cell_agglo_set::UNDEF:
                r = 0.0;
                break;

            case cell_agglo_set::T_OK:
                r = 1.0;
                break;

            case cell_agglo_set::T_KO_NEG:
                r = 2.0;
                break;

            case cell_agglo_set::T_KO_POS:
                r = 3.0;
                break;

        }

        cell_set.push_back( r );
    }
    silo.add_variable("mesh", "agglo_set", cell_set.data(), cell_set.size(), zonal_variable_t);

    silo.close();

    /*************  MAKE AN OUTPUT FOR THE INTERSECTION POINTS *************/
    std::vector<RealType> int_pts_x;
    std::vector<RealType> int_pts_y;

    for (auto& fc : msh.faces)
    {
        if( fc.user_data.location != element_location::ON_INTERFACE ) continue;

        RealType x = fc.user_data.intersection_point.x();
        RealType y = fc.user_data.intersection_point.y();

        int_pts_x.push_back(x);
        int_pts_y.push_back(y);
    }

    std::ofstream points_file_pre_FEM("int_points_pre_FEM.3D", std::ios::out | std::ios::trunc);

    if(points_file_pre_FEM)
    {
        // instructions
        points_file_pre_FEM << "X   Y   Z   val" << std::endl;

        for( size_t i = 0; i<int_pts_x.size(); i++)
        {
            points_file_pre_FEM << int_pts_x[i] << "   " <<  int_pts_y[i]
                        << "   0.0     0.0" << std::endl;
        }

        points_file_pre_FEM.close();
    }

    else
        std::cerr << "points_file_pre_FEM has not been opened" << std::endl;


    /*************  MAKE AN OUTPUT FOR THE INTERFACE *************/
    std::vector<RealType> int_x;
    std::vector<RealType> int_y;

    for (auto& cl : msh.cells)
    {
        if( cl.user_data.location != element_location::ON_INTERFACE ) continue;

        for(size_t i = 0; i < cl.user_data.interface.size(); i++)
        {
            RealType x = cl.user_data.interface.at(i).x();
            RealType y = cl.user_data.interface.at(i).y();

            int_x.push_back(x);
            int_y.push_back(y);
        }
    }
    std::ofstream interface_file_preFEM("interface_pre_FEM_Stokes.3D", std::ios::out | std::ios::trunc);

    if(interface_file_preFEM)
    {
        // instructions
        interface_file_preFEM << "X   Y   Z   val" << std::endl;

        for( size_t i = 0; i<int_x.size(); i++)
        {
            interface_file_preFEM << int_x[i] << "   " <<  int_y[i]
                        << "   0.0     0.0" << std::endl;
        }

        interface_file_preFEM.close();
    }

    else
        std::cerr << "interface_file_preFEM has not been opened" << std::endl;
}



template<typename Mesh, typename VEC , typename T >
void
goal_quantities_time(const Mesh& msh, T time ,const VEC& interface_points , const std::vector<T>& val_u_nx , const std::vector<T>& val_u_ny , const std::vector<T>& val_u_n , const std::vector<std::pair<T,T>>& vec_u_n  )
{
    
    std::string filename_stokes0 = "‎⁨val_u_nx_" + std::to_string(time) + ".3D";
    std::ofstream interface_file0(filename_stokes0, std::ios::out | std::ios::trunc);

    if(interface_file0)
    {
        // instructions
        interface_file0 << "X   Y   Z   val" << std::endl;
        size_t i = 0;
        for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
        {
            //std::cout<<val_u_nx[i]<<std::endl;
            interface_file0 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
            << val_u_nx[i] << "    0.0"<< std::endl;
            
            i++;
            
        }
            
        interface_file0.close();
    }
    else
        std::cerr << "File 'val_u_nx' has not been opened" << std::endl;
    
    std::string filename_stokes1 = "val_u_ny_" + std::to_string(time) + ".3D";
    std::ofstream interface_file1(filename_stokes1, std::ios::out | std::ios::trunc);

       if(interface_file1)
       {
           // instructions
           interface_file1 << "X   Y   Z   val" << std::endl;
           size_t i = 0;
           for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
           {
               //std::cout<<val_u_ny[i]<<std::endl;
               interface_file1 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
               << val_u_ny[i] << "    0.0"<< std::endl;
               i++;
           }
               
           interface_file1.close();
       }
       else
           std::cerr << "File 'val_u_ny' has not been opened" << std::endl;
    
    
    std::string filename_stokes2 = "val_u_n_" + std::to_string(time) + ".3D";
    std::ofstream interface_file2(filename_stokes2, std::ios::out | std::ios::trunc);

    if(interface_file2)
    {
        // instructions
        interface_file2 << "X   Y   Z   val" << std::endl;
        size_t i = 0;
        for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
        {
            //std::cout<<val_u_n[i]<<std::endl;
            interface_file2 << (*interface_point).x() << "   " <<  (*interface_point).y() << "   "
            << val_u_n[i] << "    0.0"<< std::endl;
            i++;
        }
               
        interface_file2.close();
    }
    else
        std::cerr << "File 'val_u_n' has not been opened" << std::endl;
    
    
     
    postprocess_output<double> postoutput_vec_u_n;
    std::string filename_stokes3 = "vec_u_n_" + std::to_string(time) + ".dat";
    auto test_vec_u_n = std::make_shared< gnuplot_output_object_vec<double> >(filename_stokes3);

    size_t i = 0;
    for(auto interface_point = interface_points.begin() ; interface_point < interface_points.end() ; interface_point++ )
    {
        test_vec_u_n->add_data(*interface_point,std::make_pair( val_u_n[i]*vec_u_n[i].first , val_u_n[i]*vec_u_n[i].second ) );
        i++;
    }
    
    postoutput_vec_u_n.add_object(test_vec_u_n);
    postoutput_vec_u_n.write();
    
    
    
}

/// Useful to plot level set post FE transport problem
/// in cuthho_export.hpp
template<typename Mesh, typename Function>
void
output_mesh_info2(const Mesh& msh, const Function& level_set_function)
{
    using RealType = typename Mesh::coordinate_type;

    /************** OPEN SILO DATABASE **************/
    silo_database silo;
    silo.create("cuthho_meshinfo_Stokes.silo");
    silo.add_mesh(msh, "mesh");

    /************** MAKE A SILO VARIABLE FOR CELL POSITIONING **************/
    std::vector<RealType> cut_cell_markers;
    for (auto& cl : msh.cells)
    {
        if ( location(msh, cl) == element_location::IN_POSITIVE_SIDE )
            cut_cell_markers.push_back(1.0);
        else if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            cut_cell_markers.push_back(-1.0);
        else if ( location(msh, cl) == element_location::ON_INTERFACE )
            cut_cell_markers.push_back(0.0);
        else
            throw std::logic_error("shouldn't have arrived here...");
    }
    silo.add_variable("mesh", "cut_cells", cut_cell_markers.data(), cut_cell_markers.size(), zonal_variable_t);

    /************** MAKE A SILO VARIABLE FOR CELL HIGHLIGHT **************/
    std::vector<RealType> highlight_markers;
    for (auto& cl : msh.cells)
    {
        if ( cl.user_data.highlight )
            highlight_markers.push_back(1.0);
        else
            highlight_markers.push_back(0.0);

    }
    silo.add_variable("mesh", "highlighted_cells", highlight_markers.data(), highlight_markers.size(), zonal_variable_t);

    /************** MAKE A SILO VARIABLE FOR LEVEL SET FUNCTION **************/
    std::vector<RealType> level_set_vals;
    // for (auto& pt : msh.points)
    //    level_set_vals.push_back( level_set_function(pt) );
    for (auto& n : msh.nodes)
        level_set_vals.push_back( level_set_function(n) );
    
    silo.add_variable("mesh", "level_set", level_set_vals.data(), level_set_vals.size(), nodal_variable_t);

    /************** MAKE A SILO VARIABLE FOR NODE POSITIONING **************/
    std::vector<RealType> node_pos;
    for (auto& n : msh.nodes)
        node_pos.push_back( location(msh, n) == element_location::IN_POSITIVE_SIDE ? +1.0 : -1.0 );
    silo.add_variable("mesh", "node_pos", node_pos.data(), node_pos.size(), nodal_variable_t);

    std::vector<RealType> cell_set;
    for (auto& cl : msh.cells)
    {
        RealType r;

        switch ( cl.user_data.agglo_set )
        {
            case cell_agglo_set::UNDEF:
                r = 0.0;
                break;

            case cell_agglo_set::T_OK:
                r = 1.0;
                break;

            case cell_agglo_set::T_KO_NEG:
                r = 2.0;
                break;

            case cell_agglo_set::T_KO_POS:
                r = 3.0;
                break;

        }

        cell_set.push_back( r );
    }
    silo.add_variable("mesh", "agglo_set", cell_set.data(), cell_set.size(), zonal_variable_t);

    silo.close();

    /*************  MAKE AN OUTPUT FOR THE INTERSECTION POINTS *************/
    std::vector<RealType> int_pts_x;
    std::vector<RealType> int_pts_y;

    for (auto& fc : msh.faces)
    {
        if( fc.user_data.location != element_location::ON_INTERFACE ) continue;

        RealType x = fc.user_data.intersection_point.x();
        RealType y = fc.user_data.intersection_point.y();

        int_pts_x.push_back(x);
        int_pts_y.push_back(y);
    }

    std::ofstream points_file("int_points.3D", std::ios::out | std::ios::trunc);

    if(points_file)
    {
        // instructions
        points_file << "X   Y   Z   val" << std::endl;

        for( size_t i = 0; i<int_pts_x.size(); i++)
        {
            points_file << int_pts_x[i] << "   " <<  int_pts_y[i]
                        << "   0.0     0.0" << std::endl;
        }

        points_file.close();
    }

    else
        std::cerr << "Points_file has not been opened" << std::endl;


    /*************  MAKE AN OUTPUT FOR THE INTERFACE *************/
    std::vector<RealType> int_x;
    std::vector<RealType> int_y;

    for (auto& cl : msh.cells)
    {
        if( cl.user_data.location != element_location::ON_INTERFACE ) continue;

        for(size_t i = 0; i < cl.user_data.interface.size(); i++)
        {
            RealType x = cl.user_data.interface.at(i).x();
            RealType y = cl.user_data.interface.at(i).y();

            int_x.push_back(x);
            int_y.push_back(y);
        }
    }
    std::ofstream interface_file("interface_Stokes.3D", std::ios::out | std::ios::trunc);

    if(interface_file)
    {
        // instructions
        interface_file << "X   Y   Z   val" << std::endl;

        for( size_t i = 0; i<int_x.size(); i++)
        {
            interface_file << int_x[i] << "   " <<  int_y[i]
                        << "   0.0     0.0" << std::endl;
        }

        interface_file.close();
    }

    else
        std::cerr << "Interface_file has not been opened" << std::endl;
}



// Qualitative testing of the discrete level set function wrt the analytical one
template< typename Fonction , typename Mesh  >
void
testing_velocity_field_L2projected(const Mesh msh , const Fonction& vel )
{
    //typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    
    auto test_discx  = std::make_shared< gnuplot_output_object<double> >("L2vel_HHOX.dat");
    auto test_discy  = std::make_shared< gnuplot_output_object<double> >("L2vel_HHOY.dat");
    
    
    for(auto& cl:msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<double,Mesh> (msh,cl,vel.degree_FEM);
        for(auto&pt : pts)
        {
            auto value = vel(pt,msh,cl);
            test_discx->add_data(pt,value.first);
            test_discy->add_data(pt,value.second);
            
            
        }
    }
   
    postoutput1.add_object(test_discx);
    postoutput1.add_object(test_discy);
    

    
    postoutput1.write();
    
}
template< typename Fonction , typename Mesh  >
void
testing_velocity_field(const Mesh msh , const Fonction& vel )
{
    //typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    
    auto test_discx  = std::make_shared< gnuplot_output_object<double> >("vel_HHOX.dat");
    auto test_discy  = std::make_shared< gnuplot_output_object<double> >("vel_HHOY.dat");
    
    
    for(auto& cl:msh.cells)
    {
        auto pts = equidistriduted_nodes_ordered_bis<double,Mesh> (msh,cl,vel.degree_FEM);
        for(auto&pt : pts)
        {
            auto value = vel(pt,msh,cl);
            test_discx->add_data(pt,value.first);
            test_discy->add_data(pt,value.second);
            
            
        }
    }
   
    postoutput1.add_object(test_discx);
    postoutput1.add_object(test_discy);
    

    
    postoutput1.write();
    
}
