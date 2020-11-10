/*
 *       /\        Guillaume Delay 2018,2019
 *      /__\       guillaume.delay@enpc.fr
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

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <list>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>


#include <unsupported/Eigen/MatrixFunctions> // ADD STEFANO


using namespace Eigen;

#include "core/core"
#include "core/solvers"
#include "dataio/silo_io.hpp"

#include "methods/hho"
#include "methods/cuthho"

//#ifndef __MY_GLOBALS_H__
//#define __MY_GLOBALS_H__
//static cuthho_poly_mesh<double>::cell_type agglo_LS_cl;
//#endif

//#include "sol2/sol.hpp"


// Function to be modified in order to use the new operator()

/*
// utils.hpp
template<typename Mesh, typename Function>
Matrix<typename Mesh::coordinate_type, Dynamic, 1>
make_rhs2(const Mesh& msh, const typename Mesh::cell_type& cl,
         size_t degree, const Function& f, size_t di = 0)
{
    using T = typename Mesh::coordinate_type;

    cell_basis<Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    auto qps = integrate(msh, cl, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi * f(qp.first,cl);
    }

    return ret;
}

template<typename Mesh, typename Function>
Matrix<typename Mesh::coordinate_type, Dynamic, 1>
make_rhs2(const Mesh& msh, const typename Mesh::face_type& fc,
         size_t degree, const Function& f, size_t di = 0)
{
    using T = typename Mesh::coordinate_type;

    face_basis<Mesh,T> fb(msh, fc, degree);
    auto fbs = fb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(fbs);

    auto qps = integrate(msh, fc, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto phi = fb.eval_basis(qp.first);
        ret += qp.second * phi * f(qp.first,fc);
    }

    return ret;
}
*/

// Per ora inutili, non ho ancora definito il caso vettoriale..
/*
//////  RHS
template<typename Mesh, typename Function>
Matrix<typename Mesh::coordinate_type, Dynamic, 1>
make_vector_rhs(const Mesh& msh, const typename Mesh::cell_type& cl,
                size_t degree, const Function& f, size_t di = 0)
{
    using T = typename Mesh::coordinate_type;

    vector_cell_basis<Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    auto qps = integrate(msh, cl, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi * f(qp.first,msh,cl);
    }

    return ret;
}


// make_vector_rhs on faces
template<typename Mesh, typename Function>
Matrix<typename Mesh::coordinate_type, Dynamic, 1>
make_vector_rhs(const Mesh& msh, const typename Mesh::face_type& fc,
         size_t degree, const Function& f, size_t di = 0)
{
    using T = typename Mesh::coordinate_type;

    vector_face_basis<Mesh,T> fb(msh, fc, degree);
    auto fbs = fb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(fbs);

    auto qps = integrate(msh, fc, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto phi = fb.eval_basis(qp.first);
        ret += qp.second * phi * f(qp.first,msh,fc);
    }

    return ret;
}

*/

/*
template<typename T, typename Mesh >
std::vector< point<T,2> >
r_points_negative_area(const point<T,2>& p0, const point<T,2>& p1, const point<T,2>& p2, size_t deg , const Mesh& msh,
const typename Mesh::cell_type& cl)
{
    if (deg == 0)
        deg = 1;

    if (deg > 8)
        throw std::invalid_argument("Quadrature order too high");

    auto v0 = p1 - p0;
    auto v1 = p2 - p0;
    std::vector< point<T,2> > ret;
    auto area = (v0.x() * v1.y() - v0.y() * v1.x()) / 2.0;
    // the area is negative when the points are sorted clockwise
    if(area < 0){
        std::cout << "negative weights !!" << std::endl;
        for( auto& r : cl.user_data.interface ){
            ret.push_back(r);
        }
    
    }
    return ret;

}


template<typename T, typename Mesh >
void
check_negative_weights(const Mesh& msh,
          const typename Mesh::cell_type& cl,
          size_t degree)
{
    auto bar = barycenter(msh, cl);
    auto pts = points(msh, cl);
    auto counter = offset(msh,cl);
    auto num_points = pts.size();
    std::cout<<"In cella num "<<counter<<" with degree of r = "<<degree<<std::endl;
    for (size_t i = 0; i < num_points; i++)
    {
        auto p0 = pts[i];
        auto p1 = pts[(i+1)%num_points];
        auto p2 = bar;
        auto qps = r_points_negative_area(p0, p1, p2, degree, msh ,cl );
        for(auto& qp:qps){
            std::cout<<"In cella num "<<counter<<std::endl;
            std::cout<<"Point r: x is "<<qp.x()<<", y is "<<qp.y()<<std::endl;
           //negative_areas->add_data(qp)
        }

    }

}

*/


template<typename Mesh>
void offset_definition( Mesh& msh)
{
    //size_t counter = 0;
    for (auto& cl:msh.cells) {
        auto offset_orig = offset(msh,cl);
        cl.user_data.offset_subcells.push_back(offset_orig);
       // counter++;
       // std::cout<<"Initialisation cell num "<< cl.user_data.offset_subcells[offset_orig]<<std::endl;
    }
}



// in cuthho_export.hpp
template<typename Mesh, typename Function>
void
output_mesh_info2_pre_FEM(const Mesh& msh, const Function& level_set_function)
{
    using RealType = typename Mesh::coordinate_type;

    /************** OPEN SILO DATABASE **************/
    silo_database silo;
    silo.create("cuthho_meshinfo.silo");
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
    std::ofstream interface_file_preFEM("interface_pre_FEM.3D", std::ios::out | std::ios::trunc);

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





// in cuthho_export.hpp
template<typename Mesh, typename Function>
void
output_mesh_info2(const Mesh& msh, const Function& level_set_function)
{
    using RealType = typename Mesh::coordinate_type;

    /************** OPEN SILO DATABASE **************/
    silo_database silo;
    silo.create("cuthho_meshinfo.silo");
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
    std::ofstream interface_file("interface.3D", std::ios::out | std::ios::trunc);

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







// in cuthho_geom.hpp

template<typename T, typename Function, typename Mesh>
point<T, 2>
find_zero_crossing_on_face(const point<T,2>& p0, const point<T,2>& p1, const Function& level_set_function,
                   const T& threshold, const Mesh & msh, const typename Mesh::face_type& fc)
{
    /* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */
    
    // I SHOULD CHECK THAT pm IS ALWAYS ON THE FACE p0-p1 ???????
    auto pa = p0;
    auto pb = p1;
    auto pm = (pa+pb)/2.0;
    auto pm_prev = pm;

    T x_diff_sq, y_diff_sq;

    /* A threshold of 1/10000 the diameter of the element is considered
     * acceptable. Since with 24 iterations we reduce the error by 16384
     * and the worst case is that the two points are at the opposite sides
     * of the element, we put 30 as limit. */
    size_t max_iter = 50;

    do {
        auto la = level_set_function(pa,msh,fc);
        auto lb = level_set_function(pb,msh,fc);
        auto lm = level_set_function(pm,msh,fc);

        if ( (lb >= 0 && lm >= 0) || (lb < 0 && lm < 0) )
        {   /* intersection is between pa and pm */
            pm_prev = pm;
            pb = pm;
            pm = (pa+pb)/2.0;
        }
        else
        {   /* intersection is between pm and pb */
            pm_prev = pm;
            pa = pm;
            pm = (pa+pb)/2.0;
        }

        x_diff_sq = (pm_prev.x() - pm.x()) * (pm_prev.x() - pm.x());
        y_diff_sq = (pm_prev.y() - pm.y()) * (pm_prev.y() - pm.y());

    } while ( (sqrt(x_diff_sq + y_diff_sq) > threshold) && max_iter-- );

    return pm;

    /* Affine zero crossing was like that: */
    //auto t = l0/(l0-l1);
    //auto ip = (pts[1] - pts[0]) * t + pts[0];
}


// in cuthho_geom.hpp
template<typename T, typename Function, typename Mesh >
point<T, 2>
find_zero_crossing_in_cell(const point<T,2>& p0, const point<T,2>& p1, const Function& level_set_function,
                   const T& threshold, const Mesh & msh, const typename Mesh::cell_type& cl)
{
    /* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */
    
    // I SHOULD CHECK THAT pm IS ALWAYS IN THE CELL ???????
    auto pa = p0;
    auto pb = p1;
    auto pm = (pa+pb)/2.0;
    auto pm_prev = pm;

    T x_diff_sq, y_diff_sq;

    /* A threshold of 1/10000 the diameter of the element is considered
     * acceptable. Since with 24 iterations we reduce the error by 16384
     * and the worst case is that the two points are at the opposite sides
     * of the element, we put 30 as limit. */
    size_t max_iter = 50; // ERA 50, METTO 100

    do {
        auto la = level_set_function(pa,msh,cl);
        auto lb = level_set_function(pb,msh,cl);
        auto lm = level_set_function(pm,msh,cl);

        if ( (lb >= 0 && lm >= 0) || (lb < 0 && lm < 0) )
        {   /* intersection is between pa and pm */
            pm_prev = pm;
            pb = pm;
            pm = (pa+pb)/2.0;
        }
        else
        {   /* intersection is between pm and pb */
            pm_prev = pm;
            pa = pm;
            pm = (pa+pb)/2.0;
        }

        x_diff_sq = (pm_prev.x() - pm.x()) * (pm_prev.x() - pm.x());
        y_diff_sq = (pm_prev.y() - pm.y()) * (pm_prev.y() - pm.y());

    } while ( (sqrt(x_diff_sq + y_diff_sq) > threshold) && max_iter-- );

    return pm;

    /* Affine zero crossing was like that: */
    //auto t = l0/(l0-l1);
    //auto ip = (pts[1] - pts[0]) * t + pts[0];
}




// in cuthho_geom.hpp
template<typename T, size_t ET, typename Function>
void
detect_node_position2(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    for (auto& n : msh.nodes)
    {
        //auto pt = points(msh, n); //deleted by Stefano
        //if ( level_set_function(pt) < 0 ) //deleted by Stefano
        if ( level_set_function(n) < 0 ) // add by Stefano
            n.user_data.location = element_location::IN_NEGATIVE_SIDE;
        else
            n.user_data.location = element_location::IN_POSITIVE_SIDE;
    }
}

// in cuthho_geom.hpp
template<typename T, size_t ET, typename Function>
void
detect_cut_faces2(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    for (auto& fc : msh.faces)
    {
        auto pts = points(msh, fc);
        //auto l0 = level_set_function(pts[0]);      //deleted by Stefano
        //auto l1 = level_set_function(pts[1]);       //deleted by Stefano
        
        auto l0 = level_set_function(pts[0],msh,fc);      // add by Stefano
        auto l1 = level_set_function(pts[1],msh,fc);       // add by Stefano
        
        if (l0 >= 0 && l1 >= 0)
        {
            fc.user_data.location = element_location::IN_POSITIVE_SIDE;
            continue;
        }

        if (l0 < 0 && l1 < 0)
        {
            fc.user_data.location = element_location::IN_NEGATIVE_SIDE;
            continue;
        }

        auto threshold = diameter(msh, fc) / 1e20;
        //auto pm = find_zero_crossing(pts[0], pts[1], level_set_function, threshold);
        auto pm = find_zero_crossing_on_face(pts[0], pts[1], level_set_function, threshold,msh,fc);

        /* If node 0 is in the negative region, mark it as node inside, otherwise mark node 1 */
        fc.user_data.node_inside = ( l0 < 0 ) ? 0 : 1;
        fc.user_data.location = element_location::ON_INTERFACE;
        fc.user_data.intersection_point = pm;
    }
}


// in cuthho_geom.hpp
template<typename T, size_t ET, typename Function>
void
detect_cut_cells2(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    std::cout<<"I AM IN DETECT CUT CELL2!!!!"<<std::endl;
    typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    typedef typename cuthho_mesh<T, ET>::cell_type cell_type;

    size_t cell_i = 0;
    for (auto& cl : msh.cells)
    {
        auto fcs = faces(msh, cl);
        std::array< std::pair<size_t, point_type>, 2 >  cut_faces;

        size_t k = 0;
        for (size_t i = 0; i < fcs.size(); i++)
        {
            if ( is_cut(msh, fcs[i]) )
                cut_faces.at(k++) = std::make_pair(i, fcs[i].user_data.intersection_point);
        }

        /* If a face is cut, the cells that own the face are cut. Is this
         * unconditionally true? It should...fortunately this isn't avionics
         * software */

        if (k == 0)
        {
            
            auto is_positive = [&](const point_type& pt) -> bool {
            return level_set_function(pt) > 0;
            };
            
            
            auto pts = points(msh, cl);
            
            if ( std::all_of(pts.begin(), pts.end(), is_positive) )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
     
            
            
            
            /*
            auto pts = points(msh, cl);
            auto pt = pts.begin();
            size_t counter = 0;
            while( ( pt!= pts.end() ) && ( level_set_function(*pt,msh,cl) > 0 ) )
            {
                counter++;
                pt++;
                
            }
             
            if ( counter == pts.size() )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
            */
             
        }

        if (k == 2)
        {
            cl.user_data.location = element_location::ON_INTERFACE;
            auto p0 = cut_faces[0].second;
            auto p1 = cut_faces[1].second;
            auto pt = p1 - p0;
            auto pn = p0 + point<T,2>(-pt.y(), pt.x());

            if ( level_set_function(pn,msh,cl) >= 0 )
            {
                cl.user_data.p0 = p1;
                cl.user_data.p1 = p0;
            }
            else
            {
                cl.user_data.p0 = p0;
                cl.user_data.p1 = p1;
            }

            cl.user_data.interface.push_back(cl.user_data.p0);
            cl.user_data.interface.push_back(cl.user_data.p1);
        }
        //std::cout<<"In cl "<<offset(msh,cl)<<" , k is "<<k<<std::endl;
        if ( k != 0 && k != 2 )
            throw std::logic_error("invalid number of cuts in cell");

        cell_i++;
    }
}


template<typename T, size_t ET, typename Function>
void
refine_interface2(cuthho_mesh<T, ET>& msh, typename cuthho_mesh<T, ET>::cell_type& cl,
                 const Function& level_set_function, size_t min, size_t max)
{
    if ( (max-min) < 2 )
        return;

    typedef typename cuthho_mesh<T, ET>::point_type     point_type;

    size_t mid = (max+min)/2;
    auto p0 = cl.user_data.interface.at(min);
    auto p1 = cl.user_data.interface.at(max);
    auto pm = (p0+p1)/2.0;
    auto pt = p1 - p0;
    auto pn = point_type(-pt.y(), pt.x());
    auto ps1 = pm + pn;
    auto ps2 = pm - pn;

    auto lm = level_set_function(pm,msh,cl);
    auto ls1 = level_set_function(ps1,msh,cl);
    auto ls2 = level_set_function(ps2,msh,cl);

    point_type ip;
   // std::cout<<"the node of interface are "<<p0<<" and "<<p1<<". I search pm= "<<pm<<" in which phi = "<<lm<<" and ps1 e ps2 "<<ps1<<" and "<<ps2<<"equal to "<<ls1<<" , "<<ls2<<std::endl;
    if ( !((lm >= 0 && ls1 >= 0) || (lm < 0 && ls1 < 0)) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing_in_cell(pm, ps1, level_set_function, threshold,msh,cl);
    }
    else if ( !((lm >= 0 && ls2 >= 0) || (lm < 0 && ls2 < 0)) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing_in_cell(pm, ps2, level_set_function, threshold,msh,cl);
    }
    else
        throw std::logic_error("interface not found in search range");

    cl.user_data.interface.at(mid) = ip;

    refine_interface2(msh, cl, level_set_function, min, mid);
    refine_interface2(msh, cl, level_set_function, mid, max);
}

template<typename T, size_t ET, typename Function>
void
refine_interface2(cuthho_mesh<T, ET>& msh, const Function& level_set_function, size_t levels)
{
    if (levels == 0)
        return;

    size_t interface_points = iexp_pow(2, levels);

    for (auto& cl : msh.cells)
    {
        if ( !is_cut(msh, cl) )
            continue;

        cl.user_data.interface.resize(interface_points+1);
        cl.user_data.interface.at(0)                = cl.user_data.p0;
        cl.user_data.interface.at(interface_points) = cl.user_data.p1;

        refine_interface2(msh, cl, level_set_function, 0, interface_points);
    }
}





/*****************************************************************************
 *   Test stuff ADDED by StePicca
 *****************************************************************************/

//template<typename T, typename Mesh>
//std::vector<size_t>
//subcell_finder<T,Mesh>(const Mesh& , const point<T,2>& , const typename Mesh::cell_type& , const mesh_init_params<T>&);

template<typename T, typename Mesh>
bool
pt_in_cell(const Mesh& msh, const point<T,2>& , const typename Mesh::cell_type& );

template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes(const Mesh& , const typename Mesh::cell_type& , size_t );

template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes_ordered(const Mesh& , const typename Mesh::cell_type& , size_t );

template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes_ordered_bis(const Mesh& , const typename Mesh::cell_type& , size_t );

template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes_subcell(const Mesh& ,
          const typename Mesh::cell_type& ,
                              size_t , const std::vector<size_t>& ) ;


template<typename T>
std::vector<point<T,1> >
reference_nodes(size_t );

template< typename FonctionD , typename Mesh , typename FonctionA >
void
testing_level_set(const Mesh msh , const FonctionD& ,  const FonctionA& );

template< typename FonctionD , typename Mesh >
void
test_new_method(const Mesh , const FonctionD&  , const typename Mesh::cell_type & );


// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh , typename FonctionA >
void
gradient_checking1(const Mesh msh , const FonctionD& level_set_disc , const FonctionA& level_set_anal , const typename Mesh::cell_type & cl)
{
    typedef typename Mesh::point_type       point_type;
   
    double derD1x , derD1y , derAx , derAy , derD2x , derD2y ;
    Eigen::Matrix<double,2,1> derD1 ,derD2 , derA;
    point<double,2> node;

    
    auto pts = points(msh,cl);
    for(auto & node:pts)
    {
        derD1 = level_set_disc.gradient(node);
        derD2 = level_set_disc.gradient(node,msh,cl);
        derA = level_set_anal.gradient(node);
    
            derD1x = derD1(0);
            derD1y = derD1(1);
            derD2x = derD2(0);
            derD2y = derD2(1);
            derAx = derA(0);
            derAy = derA(1);

        
        /*
            if((derD1x-derD2x)>1e-2)
            {
            std::cout<<"Differnce between two x-evaluation system "<<(derD1x-derD2x)<<std::endl;
            }
        */
        
        if((derAx-derD2x)>1e-2)
        {
            std::cout<<"Differnce between analytic and NEW X-evaluation system "<<(derAx-derD2x)<<std::endl;
        }
   
        if((derAx-derD1x)>1e-2)
        {
            std::cout<<"Differnce between analytic and OLD X-evaluation system "<<(derAx-derD1x)<<std::endl;
        }
        
        
        /*
        if((derD1y-derD2y)>1e-2)
        {
            std::cout<<"Differnce between two y-evaluation system "<<(derD1y-derD2y)<<std::endl;
        }
         */
        
        
        if((derAy-derD2y)>1e-2)
        {
            std::cout<<"Differnce between analytic and NEW Y-evaluation system "<<(derAy-derD2y)<<std::endl;
        }
        
        if((derAy-derD1y)>1e-2)
        {
            std::cout<<"Differnce between analytic and OLD Y-evaluation system "<<(derAy-derD1y)<<std::endl;
        }
             
            
    }
        


}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh , typename FonctionA >
void
gradient_checking(const Mesh msh , const FonctionD& level_set_disc , const FonctionA& level_set_anal , const typename Mesh::cell_type & cl)
{
    typedef typename Mesh::point_type       point_type;
   
    double derD1x , derD1y , derAx , derAy , derD2x , derD2y ;
    Eigen::Matrix<double,2,1> derD1 ,derD2 , derA;
    point<double,2> node;

    auto pts = points(msh,cl);
    for(auto & node:pts)
    {
        derD1 = level_set_disc.gradient(node);
       // derD2 = level_set_disc.gradient(node,msh,cl);
        derA = level_set_anal.gradient(node);
    
            derD1x = derD1(0);
            derD1y = derD1(1);
         //   derD2x = derD2(0);
         //   derD2y = derD2(1);
            derAx = derA(0);
            derAy = derA(1);

        
        /*
            if((derD1x-derD2x)>1e-2)
            {
            std::cout<<"Differnce between two x-evaluation system "<<(derD1x-derD2x)<<std::endl;
            }
        */
        /*
        if((derAx-derD2x)>1e-2)
        {
            std::cout<<"Differnce between analytic and NEW X-evaluation system "<<(derAx-derD2x)<<std::endl;
        }
        */
        if((derAx-derD1x)>1e-2)
        {
            std::cout<<"Differnce between analytic and OLD X-evaluation system "<<(derAx-derD1x)<<std::endl;
        }
        
        
        /*
        if((derD1y-derD2y)>1e-2)
        {
            std::cout<<"Differnce between two y-evaluation system "<<(derD1y-derD2y)<<std::endl;
        }
         */
        
        /*
        if((derAy-derD2y)>1e-2)
        {
            std::cout<<"Differnce between analytic and NEW Y-evaluation system "<<(derAy-derD2y)<<std::endl;
        }
        */
        if((derAy-derD1y)>1e-2)
        {
            std::cout<<"Differnce between analytic and OLD Y-evaluation system "<<(derAy-derD1y)<<std::endl;
        }
             
            
    }
        


}


// Qualitative testing of the discrete level set function wrt the analytical one


template< typename FonctionD , typename Mesh , typename FonctionA >
void
testing_velocity(const Mesh msh , const FonctionD& vel_disc , const FonctionA& vel_anal)
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    
    Eigen::Matrix<double,2,1> valueA;
    point<double,2> node;
    size_t N, M;
    N = 40;
    M = 40;
    auto test_discx  = std::make_shared< gnuplot_output_object<double> >("vel_discX.dat");
    auto test_analx = std::make_shared< gnuplot_output_object<double> >("vel_analX.dat");
    auto test_discy  = std::make_shared< gnuplot_output_object<double> >("vel_discY.dat");
    auto test_analy = std::make_shared< gnuplot_output_object<double> >("vel_analY.dat");
    
    for(auto& cl:msh.cells)
    {
        auto pts = points(msh,cl);
        for(auto&pt : pts)
        {
            auto valueD = vel_disc(pt,msh,cl);
            valueA = vel_anal(pt);
            
            test_discx->add_data(pt,valueD.first);
            test_discy->add_data(pt,valueD.second);
            test_analx->add_data(pt,valueA(0));
            test_analy->add_data(pt,valueA(1));
            
            
        }
    }
   
    postoutput1.add_object(test_discx);
    postoutput1.add_object(test_analx);
    postoutput1.add_object(test_discy);
    postoutput1.add_object(test_analy);

    
    postoutput1.write();
    
}



template< typename FonctionD , typename Mesh , typename FonctionA >
void
testing_level_set(const Mesh msh , const FonctionD& level_set_disc , const FonctionA& level_set_anal)
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD , valueA , derDx , derDy , derAx , derAy;
    Eigen::Matrix<double,2,1> derD , derA;
    point<double,2> node;
    size_t N, M;
    N = 40;
    M = 40;
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_disc.dat");
    auto test_anal = std::make_shared< gnuplot_output_object<double> >("testing_interface_anal.dat");
    
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");
    
    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    
    for(size_t i = 0 ; i<= N ; i++ )
    {
        for (size_t j = 0 ; j<= M ; j++ )
        {
            double px = i*(1.0/N);
            double py = j*(1.0/M);
            node = point_type(px,py);
            /*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
            valueD = level_set_disc(node);
            /*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
            valueA = level_set_anal(node);
            
            derD = level_set_disc.gradient(node);
            derA = level_set_anal.gradient(node);
        
            
            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            
            test_disc->add_data(node,valueD);
            test_anal->add_data(node,valueA);
            
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            
        }
        
    }
    postoutput1.add_object(test_disc);
    postoutput1.add_object(test_anal);
    
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    
    postoutput1.write();
    
}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh  >
void
testing_level_set2(const Mesh msh , const FonctionD& level_set_disc ,const FonctionD& level_set_fin)
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD , valueA , derDx , derDy , derAx , derAy;
    Eigen::Matrix<double,2,1> derD , derA;
    point<double,2> node;
    size_t N, M;
    N = 40;
    M = 40;
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_disc.dat");
    auto test_anal = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_anal.dat");
    /*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");
    
    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
    for(size_t i = 0 ; i<= N ; i++ )
    {
        for (size_t j = 0 ; j<= M ; j++ )
        {
            double px = i*(1.0/N);
            double py = j*(1.0/M);
            node = point_type(px,py);
            /*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
            valueD = level_set_disc(node);
            /*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
            valueA = level_set_fin(node);
            /*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);
        
            
            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
            test_disc->add_data(node,valueD);
            test_anal->add_data(node,valueA);
            /*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
        }
        
    }
    postoutput1.add_object(test_disc);
    postoutput1.add_object(test_anal);
    /*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
    postoutput1.write();
    
}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh  >
void
testing_level_set_mesh2(const Mesh msh , const FonctionD& level_set_disc ,const FonctionD& level_set_fin)
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD , valueA , derDx , derDy , derAx , derAy;
    Eigen::Matrix<double,2,1> derD , derA;
    point<double,2> node;
    size_t N, M;
    N = 40;
    M = 40;
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_disc_mesh2.dat");
    auto test_anal = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_anal_mesh2.dat");
    /*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");
    
    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
    for(size_t i = 0 ; i<= N ; i++ )
    {
        for (size_t j = 0 ; j<= M ; j++ )
        {
            double px = i*(1.0/N);
            double py = j*(1.0/M);
            node = point_type(px,py);
            /*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
            valueD = level_set_disc(node);
            /*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
            valueA = level_set_fin(node);
            /*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);
        
            
            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
            test_disc->add_data(node,valueD);
            test_anal->add_data(node,valueA);
            /*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
        }
        
    }
    postoutput1.add_object(test_disc);
    postoutput1.add_object(test_anal);
    /*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
    postoutput1.write();
    
}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh  >
void
testing_level_set_mesh1(const Mesh msh , const FonctionD& level_set_disc ,const FonctionD& level_set_fin)
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD , valueA , derDx , derDy , derAx , derAy;
    Eigen::Matrix<double,2,1> derD , derA;
    point<double,2> node;
    size_t N, M;
    N = 40;
    M = 40;
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_disc_mesh1.dat");
    auto test_anal = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_anal_mesh1.dat");
    /*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");
    
    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
    for(size_t i = 0 ; i<= N ; i++ )
    {
        for (size_t j = 0 ; j<= M ; j++ )
        {
            double px = i*(1.0/N);
            double py = j*(1.0/M);
            node = point_type(px,py);
            /*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
            valueD = level_set_disc(node);
            /*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
            valueA = level_set_fin(node);
            /*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);
        
            
            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
            test_disc->add_data(node,valueD);
            test_anal->add_data(node,valueA);
            /*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
        }
        
    }
    postoutput1.add_object(test_disc);
    postoutput1.add_object(test_anal);
    /*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
    postoutput1.write();
    
}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh  >
void
testing_level_set_mesh0(const Mesh msh , const FonctionD& level_set_disc ,const FonctionD& level_set_fin)
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<double> postoutput1;
    double valueD , valueA , derDx , derDy , derAx , derAy;
    Eigen::Matrix<double,2,1> derD , derA;
    point<double,2> node;
    size_t N, M;
    N = 40;
    M = 40;
    auto test_disc  = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_disc_mesh0.dat");
    auto test_anal = std::make_shared< gnuplot_output_object<double> >("testing_interface_fin_anal_mesh0.dat");
    /*
    auto test_disc_gradX  = std::make_shared< gnuplot_output_object<double> >("testing_der_discX.dat");
    auto test_anal_gradX = std::make_shared< gnuplot_output_object<double> >("testing_der_analX.dat");
    
    auto test_disc_gradY  = std::make_shared< gnuplot_output_object<double> >("testing_der_discY.dat");
    auto test_anal_gradY = std::make_shared< gnuplot_output_object<double> >("testing_der_analY.dat");
    */
    for(size_t i = 0 ; i<= N ; i++ )
    {
        for (size_t j = 0 ; j<= M ; j++ )
        {
            double px = i*(1.0/N);
            double py = j*(1.0/M);
            node = point_type(px,py);
            /*
            if (std::abs( level_set_disc(node)) <1e-2 ) {
                valueD = 1;
            }
            else
                valueD = 0;
            */
            valueD = level_set_disc(node);
            /*
            if (std::abs( level_set_anal(node)) <1e-2 ) {
                valueA = 1;
            }
            else
                valueA = 0;
            */
            valueA = level_set_fin(node);
            /*
            derD = level_set_disc.gradient(node);
            derA = level_set_fin.gradient(node);
        
            
            derDx = derD(0);
            derDy = derD(1);
            derAx = derA(0);
            derAy = derA(1);
            */
            test_disc->add_data(node,valueD);
            test_anal->add_data(node,valueA);
            /*
            test_disc_gradX->add_data(node,derDx);
            test_anal_gradX->add_data(node,derAx);
            test_disc_gradY->add_data(node,derDy);
            test_anal_gradY->add_data(node,derAy);
            */
        }
        
    }
    postoutput1.add_object(test_disc);
    postoutput1.add_object(test_anal);
    /*
    postoutput1.add_object(test_disc_gradX);
    postoutput1.add_object(test_anal_gradX);
    postoutput1.add_object(test_disc_gradY);
    postoutput1.add_object(test_anal_gradY);
    */
    postoutput1.write();
    
}
// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh , typename FonctionA >
void
test_new_method(const Mesh msh , const FonctionD& level_set_disc , const FonctionA& level_set_anal , const typename Mesh::cell_type & cl)
{
    typedef typename Mesh::point_type       point_type;
    double valueD1 , valueD2 ,valueA ;
    
    auto pts =points(msh,cl);
    
    
    
    
    timecounter tc1;
    tc1.tic();
    for(auto& pt : pts)
    {
        valueD1 = level_set_disc(pt,msh,cl);
       // valueA = level_set_anal(pt);
       // valueD2 = level_set_disc(pt);
       // std::cout<<"Differnce between two evaluation system "<<(valueD1-valueD2)<<std::endl;
      //  std::cout<<"Im in test_new_method";
    }
    
   
    
    
    tc1.toc();
    std::cout << bold << yellow << "Time for the new evaluation: " << tc1 << " seconds" << reset << std::endl;
    
   tc1.tic();
       for(auto& pt : pts)
       {
           valueD2 = level_set_disc(pt);
         //  std::cout<<"Im in test_new_method";
       }
    tc1.toc();
    std::cout << bold << yellow << "Time for the old evaluation: " << tc1 << " seconds" << reset << std::endl;
    
    tc1.tic();
         for(auto& pt : pts)
         {
             valueA = level_set_anal(pt);
         }
      tc1.toc();
      std::cout << bold << yellow << "Time for the analytic evaluation: " << tc1 << " seconds" << reset << std::endl;
    
    
}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh >
void
time_NEWdiscrete_testing(const Mesh msh , const FonctionD& level_set_disc , const typename Mesh::cell_type & cl)
{
    typedef typename Mesh::point_type       point_type;
    double valueD1 , valueD2 ,valueA ;
    
    auto pts =points(msh,cl);
    
  
    for(auto& pt : pts)
    {
        valueD1 = level_set_disc(pt,msh,cl);
       
    }
    
    
}

// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh >
void
time_OLDdiscrete_testing(const Mesh msh , const FonctionD& level_set_disc , const typename Mesh::cell_type & cl)
{
    typedef typename Mesh::point_type       point_type;
    double valueD1 , valueD2 ,valueA ;
    
    auto pts =points(msh,cl);
    
  
    for(auto& pt : pts)
    {
        valueD1 = level_set_disc(pt);
       
    }
    
    
}

template< typename FonctionA , typename Mesh >
void
time_analytic_testing(const Mesh msh , const FonctionA& level_set_anal , const typename Mesh::cell_type & cl)
{
    typedef typename Mesh::point_type       point_type;
    double valueD1 , valueD2 ,valueA ;
    
    auto pts =points(msh,cl);
    
  
    for(auto& pt : pts)
    {
        valueA = level_set_anal(pt);
       
    }
    
    
}


template< typename FonctionD , typename Mesh >
void
time_face_testing(const Mesh msh , const FonctionD& level_set_disc , const typename Mesh::face_type & fc)
{
    
    typedef typename Mesh::point_type       point_type;
    double valueD1 , valueD2 ,valueA ;
       
    auto pts =points(msh,fc);
       
     
    for(auto& pt : pts)
    {
        valueD1 = level_set_disc(pt,msh,fc);
          
    }
       
    
    
}

template< typename FonctionA , typename Mesh >
void
time_faceANALITIC_testing(const Mesh msh , const FonctionA& level_set_anal , const typename Mesh::face_type & fc)
{
    
    typedef typename Mesh::point_type       point_type;
    double valueD1 , valueD2 ,valueA ;
       
    auto pts =points(msh,fc);
       
     
    for(auto& pt : pts)
    {
        valueA = level_set_anal(pt);
          
    }
       
    
    
}



// Qualitative testing of the discrete level set function wrt the analytical one
template< typename FonctionD , typename Mesh , typename FonctionA >
void
test_new_method(const Mesh msh , const FonctionD& level_set_disc , const FonctionA& level_set_anal , const typename Mesh::cell_type & cl , const typename Mesh::face_type & fc)
{
    typedef typename Mesh::point_type       point_type;
    double valueD1 , valueD2 ,valueA , valueD3;
    
    auto pts =points(msh,fc);
    
    
    
    
    timecounter tc1;
    tc1.tic();
    for(auto& pt : pts)
    {
        valueD1 = level_set_disc(pt,msh,fc);
        valueA = level_set_anal(pt);
        //valueD2 = level_set_disc(pt,msh,cl);
        valueD3 = level_set_disc(pt);
        
        std::cout<<"Differnce between FACE and OLD evaluation system "<<(valueD1-valueD3)<<std::endl;
        std::cout<<"Error between analytic and face evaluation "<<(valueD1-valueA)<<std::endl;
    }
    

    
}

// Lagrangian basis b_kl(x,y) = b_k(x)*b_l(y) over a set of equidistributed 2-dimensional nodes (3D CASE NOT YET IMPLEMENTED)


int binomial_coeff_fx(int n , int k) {

        int C[n + 1][k + 1];
        int i, j;
      
        // Caculate value of Binomial Coefficient
        // in bottom up manner
        for (i = 0; i <= n; i++)
        {
            for (j = 0; j <= std::min(i, k); j++)
            {
                // Base Cases
                if (j == 0 || j == i)
                    C[i][j] = 1;
      
                // Calculate value using previously
                // stored values
                else
                    C[i][j] = C[i - 1][j - 1] +
                              C[i - 1][j];
            }
        }
      
        return C[n][k];
}

//#define POWER_CACHE

template<typename Mesh, typename VT>
class cell_basis_Bernstein
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;
    coordinate_type min_x , max_x , min_y , max_y ;
    coordinate_type scaling_x , scaling_y ;

#ifdef POWER_CACHE
    std::vector<coordinate_type>  power_cache , power_cache_bis;
    std::vector<size_t>  binomial_coeff , binomial_coeff_bis ;
#endif

public:
    cell_basis_Bernstein(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree)
    {
        //cell_bar        = barycenter(msh, cl);
        //cell_h          = diameter(msh, cl);
        basis_degree    = degree;
        basis_size      = (basis_degree+1)*(basis_degree+1);
        min_x = points(msh,cl)[0].x();
        max_x = points(msh,cl)[1].x();
        min_y = points(msh,cl)[0].y();
        max_y = points(msh,cl)[2].y();
        scaling_x = 1.0/( pow( (max_x - min_x), basis_degree) );
        scaling_y = 1.0/( pow( (max_y - min_y), basis_degree) );
       
    }

    Matrix<VT, Dynamic, 1>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size);

        auto bx0 = pt.x() - min_x ;
        auto bx1 = max_x - pt.x() ;
        auto by0 = pt.y() - min_y ;
        auto by1 = max_y - pt.y() ;
        
#ifdef POWER_CACHE
        if ( power_cache.size() != (basis_degree+1)*2 )
          power_cache.resize( (basis_degree+1)*2);
        
        if ( binomial_coeff.size() != (basis_degree+1) )
        binomial_coeff.resize( basis_degree+1 );

    
        for (size_t i = 0 ; i <= basis_degree ; i++)
        {
            size_t j = basis_degree - i;
            //if(i == 0)
             //   binomial_coeff[0] = 1.0 ;
            //else if(i == basis_degree)
            //    binomial_coeff[basis_degree+1] = 1.0 ;
           // else
            
            binomial_coeff[i] = binomial_coeff_fx(basis_degree,i);
           
            
            power_cache[2*i]    = binomial_coeff[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j);
            power_cache[2*i+1]  = binomial_coeff[i]*iexp_pow(by0, i)*iexp_pow(by1, j);
          
        }
#endif

        size_t pos = 0;
        for (size_t pow_x = 0; pow_x <= basis_degree; pow_x++)
        {
            for (size_t pow_y = 0; pow_y <= basis_degree; pow_y++)
            {

#ifdef POWER_CACHE
                auto bv = scaling_x * power_cache[2*pow_x] * scaling_y * power_cache[2*pow_y+1];
#else
                size_t j_x = basis_degree - pow_x;
                size_t j_y = basis_degree - pow_y;
                
                auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
               
                auto bv = scaling_x * coeff_n_x * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x) * scaling_y * coeff_n_y * iexp_pow(by0, pow_y) * iexp_pow(by1, j_y) ;
#endif
                ret(pos++) = bv;
            }
        }

        assert(pos == basis_size);

        return ret;
    }

    Matrix<VT, Dynamic, 2>
    eval_gradients(const point_type& pt)
    {
        Matrix<VT, Dynamic, 2> ret = Matrix<VT, Dynamic, 2>::Zero(basis_size, 2);

        auto bx0 = pt.x() - min_x ;
        auto bx1 = max_x - pt.x() ;
        auto by0 = pt.y() - min_y ;
        auto by1 = max_y - pt.y() ;
        //The stuff below should be in the constructor, but gradient is used less times than eval.
        auto N = basis_degree - 1 ;
        auto coeff_dx = basis_degree/(max_x - min_x);
        auto coeff_dy = basis_degree/(max_y - min_y);
        auto scaling_x_bis = 1.0/(pow(max_x - min_x,N) );
        auto scaling_y_bis = 1.0/(pow(max_y - min_y,N) );
        
        
#ifdef POWER_CACHE
        if ( power_cache.size() != (basis_degree+1)*2 )
          power_cache.resize( (basis_degree+1)*2) ;
        
        if ( power_cache_bis.size() != (N+1)*2 )
            power_cache_bis.resize( (N+1)*2) ;
        
        if ( binomial_coeff.size() != (basis_degree+1) )
            binomial_coeff.resize( basis_degree+1 ) ;
            
        if ( binomial_coeff_bis.size() != (N+1) )
            binomial_coeff_bis.resize( N+1 ) ;
        
       
        for (size_t i = 0; i <= basis_degree ; i++)
        {
            size_t j = basis_degree - i;
            binomial_coeff[i] = binomial_coeff_fx( basis_degree , i );
            power_cache[2*i]    = scaling_x *binomial_coeff[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j);
            power_cache[2*i+1]  = scaling_y *binomial_coeff[i]*iexp_pow(by0, i)*iexp_pow(by1, j);
             
            if( i < basis_degree )
            {
                size_t j_bis = N - i;
                binomial_coeff_bis[i] = binomial_coeff_fx( N , i );
                power_cache_bis[2*i] = scaling_x_bis * binomial_coeff_bis[i]*iexp_pow(bx0, i)*iexp_pow(bx1, j_bis);
                power_cache_bis[2*i+1] = scaling_y_bis * binomial_coeff_bis[i]*iexp_pow(by0, i)*iexp_pow(by1, j_bis);
               
            }
            
        }
#endif

        size_t pos = 0;
        for (int pow_x = 0; pow_x <= basis_degree; pow_x++)
        {
            for (int pow_y = 0; pow_y <= basis_degree; pow_y++)
            {
                VT ix_bis = pow_x-1;
                VT iy_bis = pow_y-1;
               
#ifdef POWER_CACHE
                auto px = power_cache[2*pow_x];
                auto py = power_cache[2*pow_y+1];
                if ( pow_x == 0 )
                    auto dx = -coeff_dx*power_cache_bis[2*pow_x] ;
                else if ( pow_x == basis_degree )
                     auto dx = coeff_dx*power_cache_bis[2*ix_bis] ;
                else
                    auto dx = coeff_dx*( power_cache_bis[2*ix_bis] - power_cache_bis[2*pow_x] );

                if ( pow_y == 0 )
                    auto dy = -coeff_dy*power_cache_bis[2*pow_y+1] ;
                else if ( pow_y == basis_degree )
                    auto dy = coeff_dy*power_cache_bis[2*iy_bis+1] ;
                else
                    auto dy = coeff_dy*( power_cache_bis[2*iy_bis+1] - power_cache_bis[2*pow_y+1] );
#else
            
                size_t j_x = basis_degree - pow_x;
                size_t j_y = basis_degree - pow_y;
                
                auto coeff_n_x = binomial_coeff_fx(basis_degree,pow_x);
                auto coeff_n_y = binomial_coeff_fx(basis_degree,pow_y);
            
                auto px = scaling_x *coeff_n_x* iexp_pow(bx0, pow_x)*iexp_pow(bx1, j_x);
                auto py = scaling_y *coeff_n_y* iexp_pow(by0, pow_y)*iexp_pow(by1, j_y);
                
                // DERIVATIVES
                

                
                
                auto px_bis0 = 0.0 , py_bis0 = 0.0 , px_bis1 = 0.0 , py_bis1 = 0.0 ;
                if ( pow_x != 0 && pow_y != 0 )
                {
                    size_t j_x_bis1 = N - ix_bis;
                    size_t j_y_bis1 = N - iy_bis;
                    auto coeff_n_x_bis1 = binomial_coeff_fx(N,ix_bis);
                    auto coeff_n_y_bis1 = binomial_coeff_fx(N,iy_bis);
                    
                    auto px_bis1 = scaling_x_bis * coeff_n_x_bis1 * iexp_pow(bx0, ix_bis) * iexp_pow(bx1, j_x_bis1);
                    auto py_bis1 = scaling_y_bis * coeff_n_y_bis1 * iexp_pow(by0, iy_bis) * iexp_pow(by1, j_y_bis1);
                }
                
                
                if ( pow_x != N && pow_y != N)
                {
                    size_t j_x_bis0 = N - pow_x;
                    size_t j_y_bis0 = N - pow_y;
                    auto coeff_n_x_bis0 = binomial_coeff_fx(N,pow_x);
                    auto coeff_n_y_bis0 = binomial_coeff_fx(N,pow_y);
                 
                    auto px_bis0 = scaling_x_bis * coeff_n_x_bis0 * iexp_pow(bx0, pow_x) * iexp_pow(bx1, j_x_bis0);
                    auto py_bis0 = scaling_y_bis * coeff_n_x_bis0 * iexp_pow(by0, pow_y) * iexp_pow(by1, j_x_bis0);
                }
                 
                
                
                auto dx = coeff_dx * ( px_bis1 - px_bis0 );
                auto dy = coeff_dy * ( py_bis1 - py_bis0 );
                               
                
#endif
                ret(pos,0) = dx*py;
                ret(pos,1) = px*dy;
                pos++;
              
            }
        }

        assert(pos == basis_size);

        return ret;
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }
    
    static size_t size(size_t degree)
    {
        return (degree+1)*(degree+1);
    }
};


template< typename Mesh , typename T = typename Mesh::coordinate_type >
void
testing_basis (const Mesh msh , size_t degree_FEM )
{
    typedef typename Mesh::point_type       point_type;
    postprocess_output<T> postoutput1;
    T valueD , valueA , derDx , derDy , derAx , derAy;
    Eigen::Matrix<T,2,1> derD , derA;
    point<T,2> node;
   
    auto test_base  = std::make_shared< gnuplot_output_object<T> >("test_base.dat");
    auto test_base_gradx = std::make_shared< gnuplot_output_object<T> >("test_base_gradx.dat");
    auto test_base_grady = std::make_shared< gnuplot_output_object<T> >("test_base_grady.dat");
    size_t N = degree_FEM +1 ;
    size_t size_basis = (degree_FEM+1)*(degree_FEM+1);
   //  Matrix<T, 1, 4> error_conv32 = Eigen::Matrix<T, 1, 4>::Zero( 1 , 4 ) ;
    Eigen::Matrix<T,Dynamic,1> const_func_one =  Eigen::Matrix<T,Dynamic,1>::Ones(size_basis,1);
    const_func_one = const_func_one.col(0);
    //std::vector<T> const_func_one(size_basis,1.0);
    for(auto& nd:msh.nodes)
        std::cout<<"node is "<<nd<<std::endl;
    
    for(auto& pt:msh.points)
           std::cout<<"point is "<<pt<<std::endl;
    
   
    
    
    for( const auto& cl:msh.cells)
    {
       
        for(auto& fc : faces(msh,cl))
            std::cout<<"fc is "<<fc<<std::endl;
        
        for(auto& pt:points(msh,cl) )
            std::cout<<"points in the cell"<<offset(msh,cl)<<" is "<<pt<<std::endl;
        
        cell_basis_Bernstein <Mesh,T> cb(msh, cl, degree_FEM);
        T ax = points(msh,cl)[0].x();
        T bx = points(msh,cl)[1].x();
        T ay = points(msh,cl)[0].y();
        T by = points(msh,cl)[2].y();
        for(size_t i = 0 ; i<=degree_FEM ; i++ )
        {
            for (size_t j = 0 ; j<=degree_FEM ; j++ )
            {
                T px = i*(1.0/degree_FEM)*(bx-ax) + ax ;
                T py = j*(1.0/degree_FEM)*(by-ay) + ay ;
                node = point_type(px,py);
                
                auto values_cell = const_func_one.dot(cb.eval_basis(node)) ;
                
                auto gradx_cell = const_func_one.dot(cb.eval_gradients(node).col(0) );
                
                auto grady_cell = const_func_one.dot(cb.eval_gradients(node).col(1) );
                
                std::cout<<"eval_basis"<<'\n';
                for(size_t iii = 0 ; iii<size_basis ; iii++)
                    std::cout<<cb.eval_basis(node)[iii]<<" , ";
                std::cout<<std::endl;
                test_base->add_data(node,values_cell);
                test_base_gradx->add_data(node,gradx_cell);
                test_base_grady->add_data(node,grady_cell);
 
            
            }
        
        }
    
    }
    postoutput1.add_object(test_base);
    postoutput1.add_object(test_base_gradx);
    
    postoutput1.add_object(test_base_grady);
       
    postoutput1.write();
}




template<typename Mesh, typename T >
class cell_basis_Lagrangian
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          cell_bar;
    coordinate_type     cell_h;
    size_t              basis_degree, basis_size;
    std::vector<point<T, 2> >          nodes;
    //std::vector<size_t>         indeces;
    
public:
    cell_basis_Lagrangian(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree)
    {
        nodes           = equidistriduted_nodes<T,Mesh>(msh, cl, degree);
       // nodes           = equidistriduted_nodes_ordered<T,Mesh>(msh, cl, degree);
        //nodes           = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree);
        basis_degree    = degree;
        basis_size      = (basis_degree+1)*(basis_degree+1);
    }
    
    /*
    cell_basis_Lagrangian(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, const std::vector<size_t>& indeces)
    {
        
        nodes = equidistriduted_nodes_subcell<T,Mesh>(msh, cl, degree, indeces);
        basis_degree    = degree;
        basis_size      = (basis_degree+1)*(basis_degree+1);
        
    }
    */

    Matrix<T, Dynamic, 1>
    eval_basis(const point_type& pt)
    {
        Matrix<T, Dynamic, 1> rety = Matrix<T, Dynamic, 1>::Zero(basis_size);
        Matrix<T, Dynamic, 1> retx = Matrix<T, Dynamic, 1>::Zero(basis_size);
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(basis_size);

        // Per la y, trovo la colonna facendo col = l%(degree+1)
        // scorro su tutta la colonna tmpy = [col:(degree+1): col+(degree+1)*degree]
        // faccio base bl moltiplicando tutti tranne quando tmpy = l
        for ( size_t l = 0; l < basis_size ; l++ )
        {
            size_t col = l%(basis_degree+1);
            T bl = 1.0;
            for (size_t tmpy = col; tmpy <= col+(basis_degree+1)*basis_degree; tmpy+=(basis_degree+1))
            {
                if (tmpy!=l)
                {
                    bl *= ( ( pt.y() - (nodes.at(tmpy)).y() )/ ( (nodes.at(l)).y() - (nodes.at(tmpy)).y() ) );
                }
            }
            rety(l) = bl;
        }
        
        // Per la x, trovo la riga facendo riga = floor(k/(degree+1))
        //scorro su tutta la riga tmpx = [(degree+1)*riga: (degree+1)*(riga+1)-1]
        // faccio base bk moltiplicando tutti tranne quando tmpx = k
         
        for (size_t k = 0 ; k < basis_size ; k++ )
        {
            T bk = 1.0;
            size_t row=floor( k/(basis_degree+1) );
            for (size_t tmpx = (basis_degree+1)*row; tmpx <= (basis_degree+1)*(row+1)-1; tmpx++)
            {
                if (tmpx!=k) {
                    bk *= ( ( pt.x() - (nodes.at(tmpx)).x() )/ ( (nodes.at(k)).x() - (nodes.at(tmpx)).x() ) );
                }
            }
            retx(k) = bk;
        }
        
        for (size_t i = 0; i<basis_size; i++)
        {
            ret(i) = rety(i)*retx(i);
        }
        return ret;
        
    }


    
    Matrix<T, Dynamic, 2>
    eval_gradients(const point_type& pt)
    {
        // Modified Yves Daoust Algorithm (https://math.stackexchange.com/questions/809927/first-derivative-of-lagrange-polynomial)
        
        Matrix<T, Dynamic, 2> ret = Matrix<T, Dynamic, 2>::Zero(basis_size, 2);
       
        Matrix<T, Dynamic, 1> rety = Matrix<T, Dynamic, 1>::Zero(basis_size);
        Matrix<T, Dynamic, 1> retx = Matrix<T, Dynamic, 1>::Zero(basis_size);
        Matrix<T, Dynamic, 1> sy = Matrix<T, Dynamic, 1>::Zero(basis_size);
        Matrix<T, Dynamic, 1> sx = Matrix<T, Dynamic, 1>::Zero(basis_size);


        // for each l, b_l(y)' = {sum(tmpy!=l)[prod(jy!=l,jy!=tmpy)[x-x_jy]]}/prod(tmpy!=l)[x_l-x_tmpy]
        
        for ( size_t l = 0; l < basis_size ; l++ )
        {
            size_t col = l%(basis_degree+1);
            T bl = 1.0 , bl_der = 1.0 ;
            T sumy = 0.0;
            for (size_t tmpy = col; tmpy <= col+(basis_degree+1)*basis_degree; tmpy+=(basis_degree+1))
            {
                 T sumyy = 1.0 ;
                if (tmpy!=l)
                {

                    bl *= ( ( pt.y() - (nodes.at(tmpy)).y() )/ ( (nodes.at(l)).y() - (nodes.at(tmpy)).y() ) );

                    bl_der *= ( 1.0/ ( (nodes.at(l)).y() - (nodes.at(tmpy)).y() ) );
                    for ( size_t jy = col; jy <= col+(basis_degree+1)*basis_degree; jy+=(basis_degree+1) )
                   {
                        if (jy!=tmpy && jy!=l)
                        {
                            sumyy *= ( pt.y()-(nodes.at(jy)).y() );
                        }
                    }
                    sumy +=sumyy;
                }
            }
            rety(l) = bl;
            sy(l) = bl_der*sumy;
        }
        
        // For the x-derivative of b_k(x), same procedure of b_l(y)'
         
        for (size_t k = 0 ; k < basis_size ; k++ )
        {
            size_t row=floor( k/(basis_degree+1) );
            T bk = 1.0 , bk_der = 1.0 ;
            T sumx = 0.0;
            for (size_t tmpx = (basis_degree+1)*row; tmpx <= (basis_degree+1)*(row+1)-1; tmpx++)
            {
                T sumxx = 1.0 ;
                
                if (tmpx!=k) {
                    
                    bk *= ( ( pt.x() - (nodes.at(tmpx)).x() )/ ( (nodes.at(k)).x() - (nodes.at(tmpx)).x() ) );
                    bk_der *= ( 1.0/ ( (nodes.at(k)).x() - (nodes.at(tmpx)).x() ) );
                    for (size_t jx = (basis_degree+1)*row; jx <= (basis_degree+1)*(row+1)-1; jx++)
                    {
                        if (jx!=tmpx && jx!=k)
                        {
                            sumxx *= ( pt.x()-(nodes.at(jx)).x() );
                        }
                    }
                    sumx += sumxx;
                    
                }
            }
            retx(k) = bk;
            sx(k) = bk_der*sumx;
        }
        
        for (size_t i = 0; i<basis_size; i++)
        {
            ret(i,0) = rety(i)*sx(i);
            ret(i,1) = retx(i)*sy(i);
            
        }
        return ret;
        
    }

    size_t size() const
    {
        return basis_size;
    }

    size_t degree() const
    {
        return basis_degree;
    }

    static size_t size(size_t degree)
    {
        return (degree+1)*(degree+1);
    }
};





template<typename T>
std::vector<point<T,1> >
reference_nodes(size_t degree)
{
    auto comp_degree = degree + 1;

    size_t reqd_nodes = comp_degree;

    std::vector<point<T,1> > ret;
    ret.reserve(reqd_nodes);

    point<T,1>  qp;
    T           a1, a2;
    T           delta_x;
    switch(reqd_nodes)
    {
        case 1:
            qp = point<T,1>({0.0});
            ret.push_back(qp);
            return ret;

        case 2:
            qp = point<T,1>({ 1.0 });
            ret.push_back( -qp );
            ret.push_back( qp );
            return ret;

        case 3:
            qp = point<T,1>({ 1.0 });
            ret.push_back( -qp );
            ret.push_back( qp );
            qp = point<T,1>({0.0});
            ret.push_back( qp );
            return ret;

        case 4:
            a1 = 1.0/3.0;
            qp = point<T,1>({ 1.0 });
            ret.push_back( -qp );
            ret.push_back( qp );
            qp = point<T,1>({ a1 });
            ret.push_back( -qp );
            ret.push_back( qp );
            return ret;

        case 5:
            // Be carefull in what order data is inserted in ret!
            // In Gauss Legendre the first one was 0.0, now is the last one
            a2 = 0.5;
            a1 = 1.0;
            qp = point<T,1>({ a1 });
            ret.push_back( -qp );
            ret.push_back( qp );

            qp = point<T,1>({ a2 });
            ret.push_back( -qp );
            ret.push_back( qp );

            qp = point<T,1>({ 0.0 });
            ret.push_back( qp );

            return ret;

        default:

            delta_x = 2.0/degree;
            std::cout<<"delta x"<<delta_x<<std::endl;
            a1 = 1.0;
            while (a1>1e-10) {
                qp = point<T,1>({ a1 });
                std::cout<<"qp "<<qp<<std::endl;
                ret.push_back( -qp );
                ret.push_back( qp );
                a1-=delta_x;
                std::cout<<"a1 "<<a1<<std::endl;

            }
            if(a1<1e-10 && a1>-1e-10)
            {
                std::cout<<"a1 "<<a1<<std::endl;
                qp = point<T,1>({0.0});
                 std::cout<<"qp "<<qp<<std::endl;
                ret.push_back( qp );
            }
            return ret;
    }

    return ret;
}

template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes(const Mesh& msh,
          const typename Mesh::cell_type& cl,
          size_t degree)
{
    typedef typename Mesh::point_type    point_type;

    auto qps = reference_nodes<T>(degree);
   

    auto pts = points(msh, cl);
    
    auto v0 = pts[1] - pts[0];
    auto v1 = pts[2] - pts[1];
    auto v2 = pts[3] - pts[2];
    auto v3 = pts[3] - pts[0];

    std::vector< point<T,2> > ret;

    auto P = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].x() * (1-xi)*(1-eta) +
               0.25 * pts[1].x() * (1+xi)*(1-eta) +
               0.25 * pts[2].x() * (1+xi)*(1+eta) +
               0.25 * pts[3].x() * (1-xi)*(1+eta);
    };

    auto Q = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].y() * (1-xi)*(1-eta) +
               0.25 * pts[1].y() * (1+xi)*(1-eta) +
               0.25 * pts[2].y() * (1+xi)*(1+eta) +
               0.25 * pts[3].y() * (1-xi)*(1+eta);
    };

    for (auto jtor = qps.begin(); jtor != qps.end(); jtor++)
    {
        for (auto itor = qps.begin(); itor != qps.end(); itor++)
        {
            auto qp_x = *itor;
            auto qp_y = *jtor;

            auto xi = qp_x.x();
            auto eta = qp_y.x();

            auto px = P(xi, eta);
            auto py = Q(xi, eta);

            ret.push_back( point_type(px, py) );
        }
    }
    return ret;
}

template<typename T>
std::vector<point<T,1> >
reference_nodes_ordered(size_t degree)
{
    auto comp_degree = degree + 1;
    size_t reqd_nodes = comp_degree;

    std::vector<point<T,1> > ret;
    ret.reserve(reqd_nodes);

    point<T,1>  qp , qp_1;
    T           a1, a2;
    T           delta_x;
    switch(reqd_nodes)
    {
        case 1:
            qp = point<T,1>({0.0});
            ret.push_back(qp);
            return ret;

        case 2:
            qp = point<T,1>({ 1.0 });
            ret.push_back( -qp );
            ret.push_back( qp );
            return ret;

        case 3:
            qp = point<T,1>({ 1.0 });
            ret.push_back( -qp );
            ret.push_back( qp );
            qp = point<T,1>({0.0});
            ret.push_back( qp );
            return ret;

        case 4:
            a1 = 1.0/3.0;
            qp = point<T,1>({ 1.0 });
            ret.push_back( -qp );
            ret.push_back( qp );
            qp = point<T,1>({ a1 });
            ret.push_back( -qp );
            ret.push_back( qp );
            return ret;

        case 5:
            // Be carefull in what order data is inserted in ret!
            // In Gauss Legendre the first one was 0.0, now is the last one
            a2 = 0.5;
            a1 = 1.0;
            qp = point<T,1>({ a1 });
            ret.push_back( -qp );
            ret.push_back( qp );

            qp = point<T,1>({ a2 });
            ret.push_back( -qp );
            qp_1 = point<T,1>({ 0.0 });
            ret.push_back( qp_1 );
            ret.push_back( qp );

         //   qp = point<T,1>({ 0.0 });
         //   ret.push_back( qp );
            return ret;

        default:

            delta_x = 2.0/degree;
            a1 = 1.0;
            while (a1>1e-10) {
                qp = point<T,1>({ a1 });
                ret.push_back( -qp );
                ret.push_back( qp );
                a1-=delta_x;

            }
            if(a1<1e-10 && a1>-1e-10)
            {
                qp = point<T,1>({0.0});
                ret.push_back( qp );
            }
            std::sort(ret.begin()+2, ret.end() ,[](point<T,1>  a, point<T,1>  b) {
                return a.x() < b.x();
            } );
            return ret;
    }
    return ret;
}



template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes_ordered(const Mesh& msh,
          const typename Mesh::cell_type& cl,
          size_t degree)
{
    typedef typename Mesh::point_type    point_type;

    auto qps = reference_nodes_ordered<T>(degree);
   
    // 3 -  12 - 11 - 10 - 2
    // 13 - 22 - 21 - 20 - 9
    // 14 - 23 - 24 - 19 - 8
    // 15 - 16 - 17 - 18 - 7
    // 0 -  4 -  5  - 6  - 1

    auto pts = points(msh, cl);
    
    auto v0 = pts[1] - pts[0];
    auto v1 = pts[2] - pts[1];
    auto v2 = pts[3] - pts[2];
    auto v3 = pts[3] - pts[0];

    std::vector< point<T,2> > ret((degree+1)*(degree+1));

    auto P = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].x() * (1-xi)*(1-eta) +
               0.25 * pts[1].x() * (1+xi)*(1-eta) +
               0.25 * pts[2].x() * (1+xi)*(1+eta) +
               0.25 * pts[3].x() * (1-xi)*(1+eta);
    };

    auto Q = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].y() * (1-xi)*(1-eta) +
               0.25 * pts[1].y() * (1+xi)*(1-eta) +
               0.25 * pts[2].y() * (1+xi)*(1+eta) +
               0.25 * pts[3].y() * (1-xi)*(1+eta);
    };

    /// First adding vertices
    // (-1,-1)
    auto qp_x = qps[0];
    auto qp_y = qps[0];
    auto xi = qp_x.x();
    auto eta = qp_y.x();
    auto px = P(xi, eta);
    auto py = Q(xi, eta);
    ret[0] = ( point_type(px, py) );
    // (1,-1)
    qp_x = qps[1];
    qp_y = qps[0];
    xi = qp_x.x();
    eta = qp_y.x();
    px = P(xi, eta);
    py = Q(xi, eta);
    ret[1] = ( point_type(px, py) );
    // (1,1)
    qp_x = qps[1];
    qp_y = qps[1];
    xi = qp_x.x();
    eta = qp_y.x();
    px = P(xi, eta);
    py = Q(xi, eta);
    ret[2] = ( point_type(px, py) );
    // (0,1)
    qp_x = qps[0];
    qp_y = qps[1];
    xi = qp_x.x();
    eta = qp_y.x();
    px = P(xi, eta);
    py = Q(xi, eta);
    ret[3] = ( point_type(px, py) );
    
    /// counter for each side of the 2D - square
    //size_t k = degree + 1;
   
    int  count0 = 4 , count1 = 4 + degree - 1 , count2 = 4 + 2*(degree - 1), count3 = 4 + 3*(degree - 1)  ; // OK degree
    int j_max = floor((degree-1)/2) ; // OK degree
  //  std::cout<<"j max "<<j_max<<std::endl;
    int pos_right = 100;
    int pos_left = 100;
   // size_t i_min = 0;
 //   std::cout<<"inizia ciclo"<<std::endl;
    for (int j = 0 ; j <= j_max ; j++ ) {
       // bool i_loop = FALSE;
        for(int i = std::max(0,j-1) ; i < degree -1 - j ; i++ )
        {
          //  int convertdata_i = static_cast<int>(i);
         //   int convertdata_j = static_cast<int>(j-1);
         //   if( convertdata_i >= convertdata_j )
          //  {
          //      i_loop = TRUE;
         //   std::cout<<"i "<<i<<" j "<<j<<std::endl;
            if(j == 0)
                pos_left = 0;
            else if(j == 1)
                pos_left = 2;
            else
                pos_left = 2+(j-1);
            
         //   std::cout<<"pos_left "<<pos_left<<std::endl;
            qp_x = qps[2+i];
            qp_y = qps[pos_left]; //qps[0 + 2*j];
         //   std::cout<<"qp_y "<<qp_y<<std::endl;
            xi = qp_x.x();
            eta = qp_y.x();
            px = P(xi, eta);
            py = Q(xi, eta);
            ret[count0] = ( point_type(px, py) );
      //      std::cout<<"count0 is "<<count0<<" , pt0"<<point_type(px, py)<<std::endl;
            //std::cout<<ret[count0]<<std::endl;
            count0++;
            if(j==0)
                pos_right = 1;
            else
                pos_right = degree + 1 - j;
           // size_t pos = (1 + j*(degree-1));
           // if(pos>degree)
           //     pos -= j*degree ;
            
            qp_x =  qps[pos_right];
            qp_y = qps[2 + i];
            xi = qp_x.x();
            eta = qp_y.x();
            px = P(xi, eta);
            py = Q(xi, eta);
            ret[count1] = ( point_type(px, py) );
         //   std::cout<<"count1 is "<<count1<<" , pt1"<<point_type(px, py)<<std::endl;
            //std::cout<<ret[count1]<<std::endl;
            count1++;
            qp_x = qps[degree - i];
            qp_y =  qps[pos_right] ;
            xi = qp_x.x();
            eta = qp_y.x();
            px = P(xi, eta);
            py = Q(xi, eta);
            ret[count2] = ( point_type(px, py) );
          //  std::cout<<"count2 is "<<count2<<" , pt2"<<point_type(px, py)<<std::endl;
            count2++;
            qp_x = qps[pos_left];
            qp_y = qps[degree - i];
            xi = qp_x.x();
            eta = qp_y.x();
            px = P(xi, eta);
            py = Q(xi, eta);
            ret[count3] = ( point_type(px, py) );
          //  std::cout<<"count3 is "<<count3<<" , pt3"<<point_type(px, py)<<std::endl;
            count3++;
          //  }
            
        }
        //if( i_loop == TRUE )
        //{
        count0 = count3 ;
        count1 = count0 + (degree - 2*(j+1)); //  count0 + (degree - 2); it was
        count2 = count1 + (degree - 2*(j+1));
        count3 = count2 + (degree - 2*(j+1));
        //}
    }
    
    /// Middle point
    if( degree % 2 == 0)
    {
        qp_x = qps[degree - floor((degree-1)/2)];
        qp_y = qps[degree - floor((degree-1)/2)];
        xi = qp_x.x();
        eta = qp_y.x();
        px = P(xi, eta);
        py = Q(xi, eta);
        ret[count0] = ( point_type(px, py) );
    }
    return ret;
}


template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes_ordered_bis(const Mesh& msh,
          const typename Mesh::cell_type& cl,
          size_t degree)
{
    typedef typename Mesh::point_type    point_type;

    auto qps = reference_nodes_ordered<T>(degree);
   
    // 3 -  12 - 11 - 10 - 2
    // 13 - 22 - 21 - 20 - 9
    // 14 - 23 - 24 - 19 - 8
    // 15 - 16 - 17 - 18 - 7
    // 0 -  4 -  5  - 6  - 1

    auto pts = points(msh, cl);
    
    auto v0 = pts[1] - pts[0];
    auto v1 = pts[2] - pts[1];
    auto v2 = pts[3] - pts[2];
    auto v3 = pts[3] - pts[0];

    std::vector< point<T,2> > ret((degree+1)*(degree+1));

    auto P = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].x() * (1-xi)*(1-eta) +
               0.25 * pts[1].x() * (1+xi)*(1-eta) +
               0.25 * pts[2].x() * (1+xi)*(1+eta) +
               0.25 * pts[3].x() * (1-xi)*(1+eta);
    };

    auto Q = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].y() * (1-xi)*(1-eta) +
               0.25 * pts[1].y() * (1+xi)*(1-eta) +
               0.25 * pts[2].y() * (1+xi)*(1+eta) +
               0.25 * pts[3].y() * (1-xi)*(1+eta);
    };

    /// First adding vertices
    // (-1,-1)
    auto qp_x = qps[0];
    auto qp_y = qps[0];
    auto xi = qp_x.x();
    auto eta = qp_y.x();
    auto px = P(xi, eta);
    auto py = Q(xi, eta);
    ret[0] = ( point_type(px, py) );
    // (1,-1)
    qp_x = qps[1];
    qp_y = qps[0];
    xi = qp_x.x();
    eta = qp_y.x();
    px = P(xi, eta);
    py = Q(xi, eta);
    ret[1] = ( point_type(px, py) );
    // (1,1)
    qp_x = qps[1];
    qp_y = qps[1];
    xi = qp_x.x();
    eta = qp_y.x();
    px = P(xi, eta);
    py = Q(xi, eta);
    ret[2] = ( point_type(px, py) );
    // (0,1)
    qp_x = qps[0];
    qp_y = qps[1];
    xi = qp_x.x();
    eta = qp_y.x();
    px = P(xi, eta);
    py = Q(xi, eta);
    ret[3] = ( point_type(px, py) );
    
    /// counter for each side of the 2D - square
    
    int  count0 = 4 , count1 = 4 + degree - 1 , count2 = 4 + 2*(degree - 1), count3 = 4 + 3*(degree - 1)  ; // OK degree
    int count_bis0 = 4*degree ;
    int j_max = floor((degree-1)/2) ; // OK degree
  //  std::cout<<"j max "<<j_max<<std::endl;
    int pos_right = 100;
    int pos_left = 100;
   // size_t i_min = 0;
 //   std::cout<<"inizia ciclo"<<std::endl;
    for (int j = 0 ; j <= j_max ; j++ ) {
       // bool i_loop = FALSE;
        for(int i = std::max(0,j-1) ; i < degree -1 - j ; i++ )
        {
         
            if( i == std::max(0,j-1) && j > 0)
            {
                if(j == 0)
                    pos_left = 0;
                else if(j == 1)
                    pos_left = 2;
                else
                    pos_left = 2+(j-1);
                           
                //   std::cout<<"pos_left "<<pos_left<<std::endl;
                qp_x = qps[2+i];
                qp_y = qps[pos_left]; //qps[0 + 2*j];
                        //   std::cout<<"qp_y "<<qp_y<<std::endl;
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                count_bis0 = count0;
                std::cout<<"count bis 0 "<<count_bis0<<std::endl;
                ret[count_bis0] = ( point_type(px, py) );
                     //      std::cout<<"count0 is "<<count0<<" , pt0"<<point_type(px, py)<<std::endl;
                           //std::cout<<ret[count0]<<std::endl;
                count_bis0++;
                if(j==0)
                    pos_right = 1;
                else
                    pos_right = degree + 1 - j;
                // size_t pos = (1 + j*(degree-1));
                // if(pos>degree)
                //     pos -= j*degree ;
                           
                qp_x =  qps[pos_right];
                qp_y = qps[2 + i];
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count_bis0] = ( point_type(px, py) );
                //   std::cout<<"count1 is "<<count1<<" , pt1"<<point_type(px, py)<<std::endl;
                        //std::cout<<ret[count1]<<std::endl;
                count_bis0++;
                qp_x = qps[degree - i];
                qp_y =  qps[pos_right] ;
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count_bis0] = ( point_type(px, py) );
                //  std::cout<<"count2 is "<<count2<<" , pt2"<<point_type(px, py)<<std::endl;
                count_bis0++;
                qp_x = qps[pos_left];
                qp_y = qps[degree - i];
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count_bis0] = ( point_type(px, py) );
                         //  std::cout<<"count3 is "<<count3<<" , pt3"<<point_type(px, py)<<std::endl;
                count_bis0++;
                count0 = count_bis0 ;
                count1 = count0 + std::ceil((degree - 2.0*j)/2.0); //  count0 + (degree - 2); it was
                count2 = count1 + std::ceil((degree - 2.0*j)/2.0);
                count3 = count2 + std::ceil((degree - 2.0*j)/2.0);

            }
            else
            {
                
                //   std::cout<<"i "<<i<<" j "<<j<<std::endl;
                if(j == 0)
                    pos_left = 0;
                else if(j == 1)
                    pos_left = 2;
                else
                    pos_left = 2+(j-1);
                
         //   std::cout<<"pos_left "<<pos_left<<std::endl;
                qp_x = qps[2+i];
                qp_y = qps[pos_left]; //qps[0 + 2*j];
         //   std::cout<<"qp_y "<<qp_y<<std::endl;
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count0] = ( point_type(px, py) );
      //      std::cout<<"count0 is "<<count0<<" , pt0"<<point_type(px, py)<<std::endl;
            //std::cout<<ret[count0]<<std::endl;
                count0++;
                if(j==0)
                    pos_right = 1;
                else
                    pos_right = degree + 1 - j;
           // size_t pos = (1 + j*(degree-1));
           // if(pos>degree)
           //     pos -= j*degree ;
            
                qp_x =  qps[pos_right];
                qp_y = qps[2 + i];
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count1] = ( point_type(px, py) );
         //   std::cout<<"count1 is "<<count1<<" , pt1"<<point_type(px, py)<<std::endl;
            //std::cout<<ret[count1]<<std::endl;
                count1++;
                qp_x = qps[degree - i];
                qp_y =  qps[pos_right] ;
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count2] = ( point_type(px, py) );
          //  std::cout<<"count2 is "<<count2<<" , pt2"<<point_type(px, py)<<std::endl;
                count2++;
                qp_x = qps[pos_left];
                qp_y = qps[degree - i];
                xi = qp_x.x();
                eta = qp_y.x();
                px = P(xi, eta);
                py = Q(xi, eta);
                ret[count3] = ( point_type(px, py) );
          //  std::cout<<"count3 is "<<count3<<" , pt3"<<point_type(px, py)<<std::endl;
                count3++;
            
                
            }
        }
        
        count0 = count3 ;
        count1 = count0 + (degree - 2*(j+1)); //  count0 + (degree - 2); it was
        count2 = count1 + (degree - 2*(j+1));
        count3 = count2 + (degree - 2*(j+1));
        //}
    }
    
    /// Middle point
    if( degree % 2 == 0)
    {
        qp_x = qps[degree - floor((degree-1)/2)];
        qp_y = qps[degree - floor((degree-1)/2)];
        xi = qp_x.x();
        eta = qp_y.x();
        px = P(xi, eta);
        py = Q(xi, eta);
        ret[count0] = ( point_type(px, py) );
    }
    return ret;
}


/*
template<typename T,typename Mesh>
std::vector< point<T,2> >
equidistriduted_nodes_subcell(const Mesh& msh,
          const typename Mesh::cell_type& cl,
          size_t degree, const std::vector<size_t>& indeces)
{
    typedef typename Mesh::point_type    point_type;

    auto qps = reference_nodes<T>(degree);
   
    std::vector<typename Mesh::point_type> pts;
    auto all_points = points(msh, cl);
    for (size_t i = 0 ; i < 4 ; i++) {
        pts.push_back( all_points[indeces[i]] );
    }
        
    auto v0 = pts[1] - pts[0];
    auto v1 = pts[2] - pts[1];
    auto v2 = pts[3] - pts[2];
    auto v3 = pts[3] - pts[0];

    std::vector< point<T,2> > ret;

    auto P = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].x() * (1-xi)*(1-eta) +
               0.25 * pts[1].x() * (1+xi)*(1-eta) +
               0.25 * pts[2].x() * (1+xi)*(1+eta) +
               0.25 * pts[3].x() * (1-xi)*(1+eta);
    };

    auto Q = [&](T xi, T eta) -> T {
        return 0.25 * pts[0].y() * (1-xi)*(1-eta) +
               0.25 * pts[1].y() * (1+xi)*(1-eta) +
               0.25 * pts[2].y() * (1+xi)*(1+eta) +
               0.25 * pts[3].y() * (1-xi)*(1+eta);
    };

    for (auto jtor = qps.begin(); jtor != qps.end(); jtor++)
    {
        for (auto itor = qps.begin(); itor != qps.end(); itor++)
        {
            auto qp_x = *itor;
            auto qp_y = *jtor;

            auto xi = qp_x.x();
            auto eta = qp_y.x();

            auto px = P(xi, eta);
            auto py = Q(xi, eta);

            ret.push_back( point_type(px, py) );
        }
    }

    return ret;
}

*/

/*

template<typename T , typename Mesh>
size_t
cells_counter(const Mesh& msh , const typename Mesh::cell_type& cl)
{
    
}
*/


/*
// OLD VERSION OF pt_in_cell
template<typename T, typename Mesh>
bool
pt_in_cell(const Mesh& msh, const point<T,2>& point_to_find, const typename Mesh::cell_type& cl)
{
    bool ret = 0;
    auto pts =points(msh,cl);
    for (size_t i = 0; i < pts.size(); i++)
    {
        if( pts[i].x()>=point_to_find.x() && pts[i].y()>=point_to_find.y() )
            ret = 1;
    }
    return ret;

}
*/
template<typename T, typename Mesh>
bool
pt_in_cell(const Mesh& msh, const point<T,2>& point_to_find, const typename Mesh::cell_type& cl)
{
    auto pts =points(msh,cl);
 
    //std::cout<<"Point to find "<<std::setprecision(15)<<point_to_find.x()<<", "<<point_to_find.y()<<std::endl;

   // std::cout<<"Min x "<<std::setprecision(15)<<pts[0].x()<<", max x "<<pts[1].x()<<std::endl;

    //std::cout<<"Min y "<<std::setprecision(15)<<pts[1].y()<<", max y "<<pts[2].y()<<std::endl;

    T epsilon = 1e-10;
     if( (pts[0].x()-epsilon)<=point_to_find.x() && (pts[1].x()+epsilon)>=point_to_find.x() && (pts[1].y()-epsilon)<=point_to_find.y() && (pts[2].y()+epsilon)>=point_to_find.y() )
         return TRUE;
    else
        return FALSE;
  
}

template<typename T, typename Mesh>
size_t
pt_in_subcell(const Mesh& msh, const point<T,2>& point_to_find, const typename Mesh::cell_type& agglocl)
{
    
    for (auto& offset_subcells : agglocl.user_data.offset_subcells)
    {
        //std::cout<<"OFFSET ORIGINAL CELL "<<offset_subcells<<std::endl;
        auto cl = msh.cells[offset_subcells];
        if( pt_in_cell(msh,point_to_find,cl) )
            return offset_subcells;
    }
    // IF IT ARRIVES HERE, IT DIDN'T FIND THE POINT IN THE CELL.
    std::cout<<"IT DIDN'T FIND THE POINT IN SUBCELL "<<offset(msh,agglocl)<<std::endl;
    throw std::invalid_argument("Invalid point-> NOT IN AGGLO_CELL");

}

// SOSPETTO SIA UNITILE
/*
template<typename T , typename Mesh>
std::vector<size_t>
subcell_finder(const Mesh& msh, const point<T,2>& point_to_find, const typename Mesh::cell_type& cl, const mesh_init_params<T>& params)
{
    std::vector<size_t> ret(4);
    auto pts = points(msh,cl);
    auto hx = params.hx();
    auto hy = params.hy();
    bool check_already_found_x = FALSE;
    for (size_t i = 0; i < pts.size()-1; i++)
    {
        if( !check_already_found_x && pts[i].x()!= pts[i+1].x() && pts[i].x()<=point_to_find.x() && pts[i+1].x()>=point_to_find.x() )
        {
            if ( (pts[i].y()+hy)>=point_to_find.y() )
            {
                // point in the first row of subcells
                ret[0]=i;
                ret[1]=i+1;
                check_already_found_x = TRUE;
                for (size_t ii = 0; ii < pts.size(); ii++)
                {
                    if( (pts[i].y()+hy) == pts[ii].y() && pts[i].x() == pts[ii].x() ){
                        ret[2] = ii;
                    }
                    if( (pts[i].y()+hy) == pts[ii].y() && pts[i+1].x() == pts[ii].x() ){
                        ret[3] = ii;
                    }
                }
            }
            else
            {
                // point in the second row of subcells
                check_already_found_x = TRUE;
                for (size_t ii = 0; ii < pts.size(); ii++) // loop to find lower pts
                {
                    if( (pts[i].y()+hy) == pts[ii].y() && pts[i].x() == pts[ii].x() )
                    {
                        ret[0] = ii; // now I look for 3rd point
                        for (size_t j = 0; j < pts.size(); j++) {
                            if( (pts[ii].y()+hy) == pts[j].y() && pts[ii].x() == pts[j].x() ){
                                ret[2] = j;
                            }
                        }
                        
                    } // end of the research for 1st and 3rd points
                    if( (pts[i].y()+hy) == pts[ii].y() && pts[i+1].x() == pts[ii].x() )
                    {
                        ret[1] = ii; // now I look for 4th point
                        for (size_t j = 0; j < pts.size(); j++) {
                            if( (pts[ii].y()+hy) == pts[j].y() && pts[ii+1].x() == pts[j].x() ){
                                ret[3] = j;
                            }
                        }
                        
                    }// end of the research for 2nd and 4th points
                    
                } // END of loop to find lower pts
            }  // END of else "search in the second row of subcells"
            
        } // END of the initial if-> I dont enter if ALREADY CHECKED or consecutive points have: same x or doesn't include pt to find
        
    } // end of the initial loop over the point
    return ret;
}

*/





// Ho due possibilità.. se uso values_bis1 uso una vector notation,
//                      se uso values_bis uso la matrix notation -> USO QUESTA PER ORA, MA I TEST CHE HO FATTO PER N,M=32, 64 RILEVANO CHE SONO PRATICAMENTE LO STESSO

template<typename T, typename Mesh ,typename Fonction >
struct projected_level_set: public level_set<T>
{
    std::vector< T > values;
    Eigen::Matrix<T, Dynamic, Dynamic> values_bis; // MATRIX NOTATION
    //Eigen::Matrix<T, Dynamic, 1> values_bis1; // VECTOR NOTATION
    Eigen::Matrix<T, Dynamic, 1> vertices; // saving level_set on vertices mesh
    std::vector< size_t > boundary_cells;
    // In the case in which I use FEM with Qk , k>1 I need another vector for all the values in each node
    
    size_t degree_FEM;
    size_t number_elements;
    Mesh msh;
    size_t      Nx, Ny ;
    mesh_init_params<T> params;
    size_t  last_row_init, last_row_end, number_faces_one_row;
  //  size_t counter_cell , counter_face, num_cell_row;
    
    T phi_max , phi_min ;
    T cut_level;
    
    
    projected_level_set(const Fonction & level_set, const Mesh & msh, size_t degree_k , const mesh_init_params<T>& params)
        : number_elements((degree_k+1)*(degree_k+1)), msh(msh),degree_FEM(degree_k),Nx(params.Nx),Ny(params.Ny), params(params)
    {
        
        last_row_init = Ny*(2*Nx+1); // There are 2 faces for each row of cells + Ny
        last_row_end = last_row_init + Nx-1;
        number_faces_one_row = 2*Nx+1; // for each cell I count only the low and sx faces, respectevely 0-1 2-3 4-5 6-7 8 + the last on the right boundary
        vertices = Eigen::Matrix<T, Dynamic, 1>::Zero( ((Nx+1)*(Ny+1)), 1 );
       
    //#ifdef NODES
        // MATRIX NOTATION
        values_bis= Eigen::Matrix<T, Dynamic, Dynamic>::Zero(number_elements, msh.cells.size());
        // VECTOR NOTATION
        //values_bis1= Eigen::Matrix<T, Dynamic, 1>::Zero(number_elements*msh.cells.size(), 1 );
        
        //std::cout<<"Number of cells "<<msh.cells.size()<<std::endl;
        
        // MATRIX NOTATION
        // std::cout<<"Dimension of the basis "<<values_bis.size()<<std::endl;
        // VECTOR NOTATION
        // std::cout<<"Dimension of the basis "<<values_bis1.size()<<std::endl;
        size_t i_global = 0 , i_local=0 , i_vertex=0;
        for(auto& cl:msh.cells)
        {
            /*
            bool boundary_bool = FALSE;
            for (auto& fc:faces(msh,cl)) {
                if (boundary_bool)
                    break;
    
                if(fc.is_boundary && !boundary_bool){
                    boundary_cells.push_back(offset(msh,cl));
                    boundary_bool = TRUE;
                }
            }
            */
            
            auto qps = equidistriduted_nodes<T,Mesh>(msh, cl, degree_FEM);
            //auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);
            
            i_local = 0;
            for ( const auto & qp : qps)
            {
                
                values.push_back( level_set(qp) ); // I DONT KNOW IF IT IS USEFUL
                
               // if (boundary_bool) {
               //      values_bis(i_local,i_global) = 1 ;  // MATRIX NOTATION
               // }
                values_bis(i_local,i_global) = level_set(qp) ;  // MATRIX NOTATION
                //values_bis1(i_local+i_global) = level_set(qp) ; // VECTOR NOTATION
                i_vertex = i_global+floor(i_global/Nx);
                if( i_local==0 )
                    vertices(i_vertex) = level_set(qp) ;
                
                if( i_local==1 )
                    vertices(i_vertex+1) = level_set(qp) ;
                
                if( i_local==(degree_FEM+2) )
                    vertices(i_vertex+Nx+2) = level_set(qp) ;
                
                if( i_local==(degree_FEM+1) )
                    vertices(i_vertex+Nx+1) = level_set(qp) ;
                i_local++;
            }
             i_global++;  // MATRIX NOTATION
           //  i_global+=number_elements;       // VECTOR NOTATION
        }
    //#endif
        
        
    }
    
    projected_level_set()=default;
    
    
    void  set_discrete_points( Eigen::Matrix<T, Dynamic, Dynamic>& values_new)
    {
        values_bis=values_new;
    }
    
    
    void converting_into_HHO_formulation( const Eigen::Matrix<T, Dynamic, 1>& vertices )
    {
        assert(degree_FEM == 1);
        for (size_t j = 0; j < values_bis.cols() ; j++)
        {

            for (size_t i = 0; i < values_bis.rows() ; i++)
            {
                auto i_vertex = j + floor(j/Nx);
                if ( i == 0 )
                    values_bis(i,j) = vertices(i_vertex);
                
                if( i ==1 )
                    values_bis(i,j) = vertices(i_vertex+1) ;
                
                if( i ==(degree_FEM+2) )
                    values_bis(i,j) = vertices(i_vertex+Nx+2) ;
                
                if( i ==(degree_FEM+1) )
                    values_bis(i,j) = vertices(i_vertex+Nx+1) ;
            }
        }
                 
    }
    
    void converting_into_FE_formulation( const Eigen::Matrix<T, Dynamic, Dynamic>& values )
    {
        assert(degree_FEM == 1);
        for (size_t j = 0; j < values_bis.cols() ; j++)
        {

            for (size_t i = 0; i < values_bis.rows() ; i++)
            {
                auto i_vertex = j + floor(j/Nx);
                if ( i == 0 )
                    vertices(i_vertex) = values(i,j) ;
                
                if( i ==1 )
                    vertices(i_vertex+1) = values(i,j) ;
                
                if( i ==(degree_FEM+2) )
                    vertices(i_vertex+Nx+2) = values(i,j) ;
                
                if( i ==(degree_FEM+1) )
                    vertices(i_vertex+Nx+1) = values(i,j) ;
            }
        }
                 
    }
       
    
    
    // OK FINE, IT WORKS ALSO FOR AGGLOMERATED MESHES
    T operator()( const typename Mesh::node_type& node ) const
    {
        // Optimised to check the value of the level set only in the vertices
        // std::cout<<"Value in vertices "<<vertices(node.ptid)<<", at position "<<node.ptid<<std::endl;
        return vertices(node.ptid);
        
    }
    
 
    // OK FINE, IT WORKS ALSO FOR AGGLOMERATED MESHES
    T operator()(const point<T,2>& pt) const
    {
        //std::cout<<"I AM IN OPERATOR() SLOW !!!!"<<std::endl;
        size_t counter=0;
        
        // It looks for in what cell the point is
        for( const auto& cl:msh.cells)
        {
            if( pt_in_cell<T,Mesh>(msh,pt,cl) )
            {
                //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
                cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
               
                auto values_cell = (values_bis.block(0,counter,number_elements,1)).col(0);
                return values_cell.dot( cb.eval_basis(pt) );
                
               // T tmp=0;
               // for(auto i = 0; i<number_elements ; i++)
               // {
               //     tmp += (values.at(i+counter))*(cb.eval_basis(pt))[i];
               // }
               // return tmp;
            }
            //counter+=number_elements; // OLD VERSION OF THE CODE
            counter+=1;
        }
        std::cout<<"IF HERE, THERE IS A PROBLEM IN projected_level_set::operator()!!!"<<std::endl;
        return 1e10; //to check if doesn't enter in the loop
    }

    // IT WORKS ONLY FOR NOT-AGGLOMERATED MESHES
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        
        size_t counter = offset(msh,cl) ;
        // Checking if cell is agglomerated or not
        /*
        size_t cell_points = (points(msh,cl)).size();
        size_t cells_number = msh.cells.size();
        bool agglomeration=FALSE;
        if ( cells_number < (Nx*Ny-1) ) {
            agglomeration = TRUE;
        }
        if ( cells_number > (Nx*Ny-1) ) {
                  // throw std::logic_error("shouldn't have arrived here...");
               }
        */
        
        //if(cell_points<5)
        //{
            // MATRIX NOTATION
         //   if (!agglomeration) {
         //       counter = offset(msh,cl);
         //    }
         //   else{
         //       counter = 1;//cells_counter(msh_origin,msh,cl);
         //   }
        
            //std::cout<<"Value of offset "<<counter<<std::endl;
            //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
        cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (values_bis.block(0,counter,number_elements,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
        return tmp;
            
            // VECTOR NOTATION
            
            //size_t counter = offset(msh,cl)*number_elements;
            //cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
            //auto values_cell = (values_bis1.segment(counter,number_elements));
            //T tmp = values_cell.dot( cb.eval_basis(pt) );
            //return tmp;
       // }
        /*
        else
        {
            std::vector<size_t> indices = subcell_finder<T,Mesh>(msh, pt , cl , params);
            cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM,indices);
            size_t counter = 1; // DA TROVAREEE!!!!!!!!!!!!!!!
            auto values_cell = (values_bis.block(0,counter,number_elements,1)).col(0);
            T tmp = values_cell.dot( cb.eval_basis(pt) );
            return tmp;
        }
        */
        
        
    }
    
 // IT WORKS ONLY FOR NOT-AGGLOMERATED MESHES
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::face_type& fc ) const
       {
           // MATRIX NOTATION
           auto counter_face = offset(msh,fc);
           size_t counter_cell;
           // da fc devo trovare la cella in cui sono per la base
           //std::cout<<"Face number "<<counter_face<<std::endl;

           // ATTENTION, ALL THIS WORKS IN STRUCTURED QUADRANGULAR MESHES
           
           // Check if I am in the last row, upper faces (ordered differently)
           if(counter_face>=last_row_init && counter_face<=last_row_end)
           {
               counter_cell = (Ny-1)*Nx + counter_face%(last_row_init);
           }
           else
           {
            // Find in what row the face is
             auto  num_cell_row = floor(counter_face/(number_faces_one_row));
               if ( counter_face!= ( (2*Nx)*(num_cell_row+1)+num_cell_row ) )
               {
                // Faces not on the right boudary, to know in what cell are, it is sufficient to analyse the low and left face of each quadrangular cell.
                   counter_cell = floor( (counter_face-num_cell_row)/2.0 );
               }
               else
               {
                // Face on the right boudary,
                   counter_cell = ( num_cell_row+1 )*Nx -1;
               }
           
           }
           //std::cout<<"Face->Cell number "<<counter_cell<<std::endl;
           auto cl = msh.cells.at(counter_cell);
           //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
           cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
           auto values_cell = (values_bis.block(0,counter_cell,number_elements,1)).col(0);
           T tmp = values_cell.dot( cb.eval_basis(pt) );
           return tmp;

           
       }
    
    

    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();
    }
    
    
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        size_t counter=0;
        //std::cout<<"I AM IN GRADIENT SLOW !!!!"<<std::endl;
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        for( const auto& cl:msh.cells)
        {
            if(pt_in_cell<T,Mesh>(msh,pt,cl))
            {
                //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
                cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
                
                auto values_cell = values_bis.col(counter);
                auto grad_eval =  cb.eval_gradients(pt);
                ret(0) = values_cell.dot( grad_eval.col(0) );
                // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
                ret(1) = values_cell.dot( grad_eval.col(1) );

                return ret;
            }
            //counter+=number_elements; // OLD VERSION OF THE CODE
            counter+=1;

        }
        std::cout<<"Se compare questo problema in gradient()"<<std::endl;
        ret(0)+=1e10;
        ret(1)+=1e10;
        return ret; //to check if doesn't enter in the loop

    }

    
    Eigen::Matrix<T,2,1> gradient( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
    
        // MATRIX NOTATION
        size_t counter = offset(msh,cl);
        //std::cout<<"the cell in NEW is the number "<<counter<<std::endl;
        //std::cout<<"Value of offset "<<counter<<std::endl;
        //cell_basis_Qk<Mesh,T> cb(msh, cl, degree_FEM);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = values_bis.col(counter);
        auto grad_eval =  cb.eval_gradients(pt);
        ret(0) = values_cell.dot( grad_eval.col(0) );
       // std::cout<<"Value of derivative new along x"<<ret(0)<<std::endl;
        ret(1) = values_cell.dot( grad_eval.col(1) );
        //values_cell.dot( grad_eval.col(1) );
       // std::cout<<"Value of derivative new along y"<<ret(1)<<std::endl;
        return ret;
        
        // VECTOR NOTATION
        
        //size_t counter = offset(msh,cl)*number_elements;
        //cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        //auto values_cell = (values_bis1.segment(counter,number_elements));
        //T tmp = values_cell.dot( cb.eval_basis(pt) );
        //return tmp;
        
        
    }


    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt, const Mesh& msh, const typename Mesh::cell_type& cl) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret = gradient(pt,msh,cl);
        return ret/ret.norm();
        
    }
    
    
    
    
    
    // template<typename T , typename MATRIX , typename VECTOR >
    void smooth_cut_off( T C , T x_centre , T y_centre , T radius )
    {
        T hx = params.hx();
        T hy = params.hy();
        cut_level = C;
        // T r0 = radius + 2*sqrt(pow(hx,2) + pow(hy,2)); // check value
        //T r0 = radius + 2.5*std::max(hx , hy ); // check value
        //T r0 = radius + 2.5*0.0625; // check value
        T pos_r0 = 0.45; //std::min(x_centre , 1 - x_centre );
        // IF I wanna define like that I need to pass the value all'inizio, sennò cambia per valure le exact solution (cambiando x_centre).
        
        //T max_r0 = pos_r0-(pos_r0 - 2*0.0625); //2.5*std::max(hx,hy); // if mesh too much coarse
        //r0 = std::max(r0 , max_r0);
        //r0 = 0.25*max_r0 + 0.75*r0 ;
        //if(r0<radius)
          //  r0 += 0.0625 ;
        T dist = pos_r0 - radius + 2*0.0625;
        T r0 = radius + dist/2;
        T delta = r0/8;
        //T delta = 0.0625 ; // r0/20.0; // or better std::max(hx,hy) ??
        //T delta = sqrt(pow(hx,2) + pow(hy,2));
        //if(std::abs(r0-delta-radius)<2*std::max(hx,hy) )
         //   delta = delta/2;
       
        std::cout<<"radius = "<<radius<<" , r0 = "<<r0<<" , delta = "<<delta<<" , hx = hy = "<<hx<<std::endl;
        std::cout<<"value in alfa in r_int = "<<(radius-r0)/delta<<std::endl;
        std::cout<<"value in alfa in R = "<<(pos_r0-r0)/delta<<std::endl;
        auto alfa = [x_centre , y_centre , delta , r0](const typename Mesh::point_type& pt) { // sol
            return (1 - tanh( (sqrt(pow((pt.x()-x_centre),2) + pow((pt.y()-y_centre),2)) - r0 ) / delta ))/2;};
 
        size_t counter = 0;
       // Eigen::Matrix<T, Dynamic, 1> tmp_vert = Matrix<T, Dynamic, 1>::Zero(vertices.rows(), 1); ;
        for(auto& pt:msh.points )
        {
           // std::cout<<"the point is "<< pt<<" alfa is "<<alfa(pt)<<std::endl;
            vertices(counter) = ( 1 - alfa(pt)  )*C + alfa(pt)*vertices(counter);
            counter++;
        }
       
        converting_into_HHO_formulation(vertices);
        phi_max = vertices.maxCoeff();
        phi_min = vertices.minCoeff(); // DEVO CAMBIARLO O VA BENE COSI?
        
        postprocess_output<double> postoutput00;
        typedef typename Mesh::point_type       point_type;
        point<double,2> node;
        auto alfa_values = std::make_shared< gnuplot_output_object<double> >("alfa.dat");
        //auto interface_pos = std::make_shared< gnuplot_output_object<double> >("interface_alfa.dat");
        for(auto& pt:msh.points )
               {
                  alfa_values->add_data(pt,alfa(pt));
               }
        postoutput00.add_object(alfa_values);
        postoutput00.write();
    }
    
    
    
    
   // template<typename T , typename MATRIX , typename VECTOR >
    void cut_off( T d )
    {
        //auto vertices_abs = vertices.cwiseAbs();
        //auto phi_max_abs = d*vertices_abs.maxCoeff();
        //std::cout<<"MAX VALUE OF PHI ABS"<<phi_max_abs<<std::endl;
        
        cut_level = d ;

        
        // Test to compare cut_off vs smooth_cut_off
        /*
        T pos_r0 = 0.45;
        T radius = 1.0/9.0;
        T dist = pos_r0 - radius + 2*0.0625;
        T r0 = radius + dist/2;
        cut_level = r0*r0 - radius*radius;
        d = cut_level ;
        std::cout<<"ATTENCTION!!! STO USANDO CUT_OFF CON D CORRISPONDETEN A r0!!!!"<<std::endl;
        std::cout<<"r0 = "<<r0<<" , cut_level = "<<cut_level<<std::endl;
        */
        
        
        //std::cout<<"CUTTING AT d = "<<d<<std::endl;
        T level_set_max = vertices.maxCoeff();
        //std::cout<<"MAX VALUE OF PHI BEFORE CUTTING = "<<level_set_max<<std::endl;
        //assert(degree_FEM == 1)
        
        Eigen::Matrix<T, Dynamic, Dynamic> One_mat = Eigen::Matrix<T, Dynamic, Dynamic>::Ones(values_bis.rows(), values_bis.cols() ); // MATRIX NOTATION
           //Eigen::Matrix<T, Dynamic, 1> values_bis1; // VECTOR NOTATION
        Eigen::Matrix<T, Dynamic, 1> One_vec = Eigen::Matrix<T, Dynamic, 1>::Ones( vertices.rows(), 1 ); // saving level_set on vertices mesh
        
        
       auto cut_off_level_vec = d*One_vec ;
       auto cut_off_level_mat = d*One_mat ;
        
        vertices/=(level_set_max);
        values_bis/=(level_set_max);
       // std::cout<<"MAX VALUE OF NORMALISED PHI = "<<vertices.maxCoeff()<<std::endl;
        auto vertices_prova = vertices;
        auto values_bis_prova = values_bis ;
        
        vertices = vertices.cwiseMin(cut_off_level_vec);
        values_bis = values_bis.cwiseMin(cut_off_level_mat);
        // Cut off also in inner domain
        vertices = vertices.cwiseMax(-cut_off_level_vec);
        values_bis = values_bis.cwiseMax(-cut_off_level_mat);
        
        // NOT NORMALISED CUT
        //vertices = vertices.cwiseMin(cut_off_level_vec*level_set_max);
        //values_bis = values_bis.cwiseMin(cut_off_level_mat*level_set_max);
        
        
        phi_max = vertices.maxCoeff();
        phi_min = vertices.minCoeff(); // DEVO CAMBIARLO O VA BENE COSI?
        //std::cout<<"MAX VALUE OF PHI_CUT AND NORMALISED = "<<phi_max<<std::endl;
        std::cout<<"Cut at "<<phi_max<<" , MIN VALUE OF PHI_CUT AND NORMALISED = "<<phi_min<<" (NEGATIVE cut off ACTIVED)."<<std::endl;
        
        /*
        postprocess_output<double> postoutput2;
        typedef typename Mesh::point_type       point_type;
        point<double,2> node;
        auto test_activated_nodes = std::make_shared< gnuplot_output_object<double> >("test_activated_nodes.dat");
        auto test_before_cut_off = std::make_shared< gnuplot_output_object<double> >("test_before_cut_off.dat");
        
        for (size_t i = 0; i<vertices_prova.rows(); i++) {
            node = msh.points.at(i);
            test_before_cut_off->add_data(node,vertices_prova(i));
    
            if( std::abs(vertices_prova(i) - vertices(i))>1e-10 )
                vertices_prova(i) = 1;
            else
                vertices_prova(i) = 0;
            test_activated_nodes->add_data(node,vertices_prova(i));
            
            
        }
        
      
        for (size_t j = 0; j<values_bis.cols(); j++)
        {
            bool active_cell= FALSE;
            size_t i = 0;
            while (i<values_bis.rows() && !active_cell ) {
                if( values_bis_prova(i,j)!= values_bis(i,j) ){
                    active_cell = TRUE;
                //std::cout<<"values_bis_prova(i,j)= "<<values_bis_prova(i,j)<<" , values_bis(i,j)= "<<values_bis(i,j)<<std::endl;
              //  std::cout<<" the cell num "<<j<<" is active."<<std::endl;
                }
                i++;
            }
        }
        
        
        
        postoutput2.add_object(test_before_cut_off);
        postoutput2.add_object(test_activated_nodes);
        postoutput2.write();
        */
        /*
        for (size_t i = 0 ; i<values_bis.rows() ; i++ ) {
            for (size_t j = 0 ; j<values_bis.cols() ; j++ ) {
                
            }
            
        }
            */
        
    }

    void boundary_con( T d )
    {
        T level_set_max = vertices.maxCoeff();
        std::cout<<"MAX VALUE OF PHI"<<level_set_max<<std::endl;
        //assert(degree_FEM == 1)
        
        Eigen::Matrix<T, Dynamic, Dynamic> One_mat = Eigen::Matrix<T, Dynamic, Dynamic>::Ones(values_bis.rows(), values_bis.cols() ); // MATRIX NOTATION
           //Eigen::Matrix<T, Dynamic, 1> values_bis1; // VECTOR NOTATION
        Eigen::Matrix<T, Dynamic, 1> One_vec = Eigen::Matrix<T, Dynamic, 1>::Ones( vertices.rows(), 1 ); // saving level_set on vertices mesh
        
        
       auto cut_off_level_vec = d*One_vec ;
       auto cut_off_level_mat = d*One_mat ;
        
        vertices/=(level_set_max);
        values_bis/=(level_set_max);
        vertices = vertices.cwiseMin(cut_off_level_vec);
        values_bis = values_bis.cwiseMin(cut_off_level_mat);
        /*
        for (size_t i = 0 ; i<values_bis.rows() ; i++ ) {
            for (size_t j = 0 ; j<values_bis.cols() ; j++ ) {
                
            }
            
        }
            */
        
    }
    
    
    
    
};

/*
template< typename Mesh>
struct Current_Cell
{
    typedef typename Mesh::cell_type       cell_type;
    cell_type crr_cl;
    Current_Cell(const cell_type& cl):crr_cl(cl){}
    Current_Cell(){}
    
    void set_cell(const cell_type& cl)
    {
        crr_cl = cl;
    }
    
    cell_type get_cell() const
    {
       return crr_cl;
    }
    
};

*/

template< typename Mesh>
struct Current_Mesh
{
    Mesh current_mesh ;
    Current_Mesh(const Mesh& msh):current_mesh(msh){}
    
    void set_current_mesh(const Mesh& msh)
    {
        current_mesh = msh;
    }
    
    
};



template< typename T , typename Mesh ,typename Level_Set,typename Fonction >
struct LS_cell: public projected_level_set< T,  Mesh , Fonction>
{
    typedef typename Mesh::cell_type       cell_type;
    cell_type agglo_LS_cl;
    std::vector<cell_type> subcells;
    Mesh agglo_msh;
    Level_Set level_set;
    //LS_cell(const Level_Set & level_set, const Mesh & msh, const typename Mesh::cell_type& cl)
   // : agglo_cl(cl), agglo_msh(msh), level_set(level_set){}
    // I don't know if I have to define a copyconstructor for level_set.. TO BE CHECKED!
    LS_cell(const Level_Set & level_set_, const Mesh & msh)
    : agglo_msh(msh), level_set(level_set_){}
    //LS_cell(const Level_Set & level_set )
    //: level_set(level_set){}

    //LS_cell()=default;
    
    T operator()(const point<T,2>& pt) const
       {
           if (subcells.size()<1)
           {
               assert(agglo_LS_cl.user_data.offset_subcells.size()==1);
               auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
               auto cl_old = level_set.msh.cells[offset_old];
               return level_set( pt , level_set.msh , cl_old );
           }
       
           else
           {
               auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
               auto subcl = level_set.msh.cells[offset];
               return level_set( pt , level_set.msh , subcl );
           }
               
       }
    
    
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        if (subcells.size()<1)
            {
                assert(agglo_LS_cl.user_data.offset_subcells.size()==1);
                auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = level_set.msh.cells[offset_old];
                return level_set.normal( pt , level_set.msh , cl_old );
            }
        
            else
            {
                auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
                auto subcl = level_set.msh.cells[offset];
                return level_set.normal( pt , level_set.msh , subcl );
            }
    }
    
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        if (subcells.size()<1)
            {
                std::cout<<"I m here 2.5 and size is "<<agglo_LS_cl.user_data.offset_subcells.size()<<std::endl;
                assert(agglo_LS_cl.user_data.offset_subcells.size()==1);
                auto offset_old = agglo_LS_cl.user_data.offset_subcells[0];
                auto cl_old = level_set.msh.cells[offset_old];
                return level_set.gradient( pt , level_set.msh , cl_old );
            }
        
            else
            {
                auto offset = pt_in_subcell(level_set.msh,pt,agglo_LS_cl);
                auto subcl = level_set.msh.cells[offset];
                return level_set.gradient( pt , level_set.msh , subcl );
            }
        
    }
    
    
    T divergence2D(const Eigen::Matrix<T,2,1>& vec )const
    {
        
    }
  
    void cell_assignment(const cell_type& cl)
    {
        subcells.clear();
        agglo_LS_cl = cl;
        if (agglo_LS_cl.user_data.offset_subcells.size()>1)
        {

            for (auto& offset_subcells:agglo_LS_cl.user_data.offset_subcells)
            {
                subcells.push_back( level_set.msh.cells[offset_subcells] );
            
            }
        }
            
    }
    
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt , const cell_type& cl)
    {
        //agglo_LS_cl = cl;
        cell_assignment(cl);
        return normal( pt );
    }
    
    T operator()(const point<T,2>& pt,const cell_type& cl )
    {
        //agglo_LS_cl = cl;
        cell_assignment(cl);
        return operator()( pt );
    }
    
    
  //  void get_cell()
  //  {
  //      auto crr_cl = download_current_cell();
  //      std::cout<<"IN get_cell.. size cell "<<crr_cl.user_data.offset_subcells.size()<<std::endl;
  //      this->cell_assignment(crr_cl);
 //   }
    
};








/*****************************************************************************
 *   Test stuff
 *****************************************************************************/
 //   template<typename RealType>
 //   auto lagrange_interpolation<RealType>(const cuthho_poly_mesh<RealType>&, const T&, const size_t);

template<typename Mesh, typename Function>
void
test_quadrature_bis(const Mesh& msh, const Function& level_set_function, size_t degree)
{
    using T = typename Mesh::coordinate_type;

    // x^k
    auto ref_x = [](const typename cuthho_poly_mesh<T>::point_type& pt, const size_t k) -> T {
        return iexp_pow(pt.x(), k);
    };

    // y^k
    auto ref_y = [](const typename cuthho_poly_mesh<T>::point_type& pt, const size_t k) -> T {
        return iexp_pow(pt.y(), k);
    };

    T integral_x = 0.0;
    T integral_y = 0.0;

    auto cbs = cell_basis<Mesh, T>::size(degree);
    for (auto cl : msh.cells)
    {
        // basis functions
        cell_basis<Mesh, T> cb(msh, cl, degree);

        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            // local mass matrix
            Matrix<T, Dynamic, Dynamic> mass1 = Matrix<T, Dynamic, Dynamic>::Zero(cbs,cbs);
            Matrix<T, Dynamic, Dynamic> mass2 = Matrix<T, Dynamic, Dynamic>::Zero(cbs,cbs);

            // projection : right-hand side
            Matrix<T, Dynamic, 1> RHS_1x = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_2x = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_1y = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_2y = Matrix<T, Dynamic, 1>::Zero(cbs);

            auto qps_n = integrate(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                auto phi = cb.eval_basis(qp.first);
                RHS_1x += qp.second * ref_x(qp.first, degree) * phi;
                RHS_1y += qp.second * ref_y(qp.first, degree) * phi;

                mass1 += qp.second * phi * phi.transpose();
            }
            auto qps_p = integrate(msh, cl, 2*degree, element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                auto phi = cb.eval_basis(qp.first);
                RHS_2x += qp.second * ref_x(qp.first, degree) * phi;
                RHS_2y += qp.second * ref_y(qp.first, degree) * phi;

                mass2 += qp.second * phi * phi.transpose();
            }
            // computation of degrees of projection coefficients
            auto M1_ldlt = mass1.ldlt();
            auto M2_ldlt = mass2.ldlt();
            Matrix<T, Dynamic, 1> U_1x = M1_ldlt.solve(RHS_1x);
            Matrix<T, Dynamic, 1> U_1y = M1_ldlt.solve(RHS_1y);
            Matrix<T, Dynamic, 1> U_2x = M2_ldlt.solve(RHS_2x);
            Matrix<T, Dynamic, 1> U_2y = M2_ldlt.solve(RHS_2y);

            // computation of the integral value
            for (auto& qp : qps_n)
            {
                auto t_phi = cb.eval_basis( qp.first );
                integral_x += qp.second * U_1x.dot(t_phi);
                integral_y += qp.second * U_1y.dot(t_phi);
            }
            for (auto& qp : qps_p)
            {
                auto t_phi = cb.eval_basis( qp.first );
                integral_x += qp.second * U_2x.dot(t_phi);
                integral_y += qp.second * U_2y.dot(t_phi);
            }
        }
        else
        {
            // local mass matrix
            Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, cl, degree);

            // projection : right-hand side
            Matrix<T, Dynamic, 1> RHS_x = Matrix<T, Dynamic, 1>::Zero(cbs);
            Matrix<T, Dynamic, 1> RHS_y = Matrix<T, Dynamic, 1>::Zero(cbs);
            auto qps = integrate(msh, cl, degree);
            for (auto& qp : qps)
            {
                auto phi = cb.eval_basis(qp.first);
                RHS_x += qp.second * ref_x(qp.first, degree) * phi;
                RHS_y += qp.second * ref_y(qp.first, degree) * phi;
            }

            // computation of degrees of projection coefficients
            auto M_ldlt = mass.ldlt();
            Matrix<T, Dynamic, 1> U_x = M_ldlt.solve(RHS_x);
            Matrix<T, Dynamic, 1> U_y = M_ldlt.solve(RHS_y);

            // computation of the integral value
            for (auto& qp : qps)
            {
                auto t_phi = cb.eval_basis( qp.first );
                integral_x += qp.second * U_x.dot(t_phi);
                integral_y += qp.second * U_y.dot(t_phi);
            }
        }
    }

    // final error
    std::cout << "error x^k = " << 1.0/(degree+1) - integral_x << std::endl;
    std::cout << "error y^k = " << 1.0/(degree+1) - integral_y << std::endl;
}




template<typename Mesh, typename Function>
void test_projection(const Mesh& msh, const Function& level_set_function, size_t degree)
{
    using T = typename Mesh::coordinate_type;

    // reference function (to be projected)
    auto ref_fun = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> T {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };

    auto grad_ref_fun = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;

        ret(0) = M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        ret(1) = M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());

        return ret;
    };

    T H1_error_uncut = 0.0;
    T H1_error_cut = 0.0;
    T L2_error_uncut = 0.0;
    T L2_error_cut = 0.0;
    T interface_L2_error = 0.0;

    size_t proj_degree = degree + 1;
    auto cbs = cell_basis<Mesh, T>::size(proj_degree);
    for (auto cl : msh.cells)
    {
        // basis functions
        cell_basis<Mesh, T> cb(msh, cl, proj_degree);

        // local mass matrix
        Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, cl, proj_degree);

        // projection : right-hand side
        Matrix<T, Dynamic, 1> RHS = Matrix<T, Dynamic, 1>::Zero(cbs);
        auto qps = integrate(msh, cl, 2*proj_degree);
        for (auto& qp : qps)
        {
            auto phi = cb.eval_basis(qp.first);
            RHS += qp.second * ref_fun(qp.first) * phi;
        }

        // computation of projection coefficients
        auto M_ldlt = mass.ldlt();
        Matrix<T, Dynamic, 1> U = M_ldlt.solve(RHS);

        // computation of H1 and L2 errors
        if( location(msh, cl) == element_location::ON_INTERFACE )
        {
            auto qps_n = integrate(msh, cl, 2*proj_degree, element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                // H1_error
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<T, 1, 2> grad = Matrix<T, 1, 2>::Zero();
                for (size_t i = 1; i < cbs; i++ )
                    grad += U(i) * t_dphi.block(i, 0, 1, 2);

                Matrix<T, Dynamic, 1> delta_grad = grad_ref_fun(qp.first) - grad;
                H1_error_cut += qp.second * delta_grad.dot(delta_grad);

                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                L2_error_cut += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }

            auto qps_p = integrate(msh, cl, 2*proj_degree, element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                // H1_error
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<T, 1, 2> grad = Matrix<T, 1, 2>::Zero();
                for (size_t i = 1; i < cbs; i++ )
                    grad += U(i) * t_dphi.block(i, 0, 1, 2);

                Matrix<T, Dynamic, 1> delta_grad = grad_ref_fun(qp.first) - grad;
                H1_error_cut += qp.second * delta_grad.dot(delta_grad);

                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                L2_error_cut += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }

            auto qps_int = integrate_interface(msh, cl, 2*proj_degree,
                                               element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_int)
            {
                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                interface_L2_error += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }
        }
        else
        {
            auto qps = integrate(msh, cl, 2*proj_degree);
            for (auto& qp : qps)
            {
                // H1_error
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<T, 1, 2> grad = Matrix<T, 1, 2>::Zero();
                for (size_t i = 1; i < cbs; i++ )
                    grad += U(i) * t_dphi.block(i, 0, 1, 2);

                Matrix<T, Dynamic, 1> delta_grad = grad_ref_fun(qp.first) - grad;
                H1_error_uncut += qp.second * delta_grad.dot(delta_grad);

                // L2_error
                auto t_phi = cb.eval_basis( qp.first );
                auto v = U.dot(t_phi);
                L2_error_uncut += qp.second * (ref_fun(qp.first) - v) * (ref_fun(qp.first) - v);
            }
        }
    }

    // write the results
    std::cout << bold << green << "UNcut cells: " << std::endl;
    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error_uncut) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error_uncut) << std::endl;

    std::cout << bold << green << "CUT cells: " << std::endl;
    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error_cut) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error_cut) << std::endl;
    std::cout << bold << red << "L2-norm absolute error on interface :           " << std::sqrt(interface_L2_error) << std::endl;
}




void tests_stabilization()
{
    using T = double;
    using Mesh = cuthho_poly_mesh<T>;

    // reference function (to be projected)
    auto ref_fun = [](const typename cuthho_poly_mesh<T>::point_type& pt) -> T {
        return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
    };


    std::vector<size_t> mesh_sizes, pol_orders;

    // meshes
    mesh_sizes.push_back(8);
    mesh_sizes.push_back(16);
    mesh_sizes.push_back(32);
    mesh_sizes.push_back(64);
    mesh_sizes.push_back(128);

    // polynomial orders
    pol_orders.push_back(0);
    pol_orders.push_back(1);
    pol_orders.push_back(2);
    pol_orders.push_back(3);

    for (std::vector<size_t>::iterator it = pol_orders.begin(); it != pol_orders.end(); it++)
    {
        size_t k = *it;

        std::cout << bold << blue << "!!!!!!!! start tests for k = " << k << std::endl;

        for (std::vector<size_t>::iterator it_msh = mesh_sizes.begin();
             it_msh != mesh_sizes.end(); it_msh++)
        {
            // mesh
            size_t N = *it_msh;
            mesh_init_params<T> mip;
            mip.Nx = N;
            mip.Ny = N;
            cuthho_poly_mesh<T> msh(mip);


            T error_T = 0.0;
            T error_F = 0.0;
            T error_stab = 0.0;

            auto cbs = cell_basis<Mesh, T>::size(k+1);
            auto fbs = face_basis<Mesh, T>::size(k);
            for (auto cl : msh.cells)
            {
                T hT = diameter(msh, cl);

                /////////  CELL PROJECTION
                // basis functions
                cell_basis<Mesh, T> cb(msh, cl, k+1);

                // local mass matrix
                Matrix<T, Dynamic, Dynamic> mass_T = make_mass_matrix(msh, cl, k+1);

                // projection : right-hand side
                Matrix<T, Dynamic, 1> RHS_T = Matrix<T, Dynamic, 1>::Zero(cbs);
                auto qps_T = integrate(msh, cl, 2*(k+1));
                for (auto& qp : qps_T)
                {
                    auto phi = cb.eval_basis(qp.first);
                    RHS_T += qp.second * ref_fun(qp.first) * phi;
                }

                // computation of projection coefficients (cell)
                auto M_ldlt = mass_T.ldlt();
                Matrix<T, Dynamic, 1> U_T = M_ldlt.solve(RHS_T);

                // computation of cell projection error
                for (auto& qp : qps_T)
                {
                    auto phi = cb.eval_basis(qp.first);
                    auto delta = ref_fun(qp.first) - phi.dot(U_T);
                    error_T += qp.second * delta * delta;
                }

                // stabilization matrix
                hho_degree_info hdi(k+1, k);
                auto hho_stab = make_hho_naive_stabilization(msh, cl, hdi);

                auto fcs = faces(msh, cl);
                auto num_faces = fcs.size();
                Matrix<T, Dynamic, 1> loc_vect
                    = Matrix<T, Dynamic, 1>::Zero( cbs + num_faces * fbs );
                loc_vect.head(cbs) = U_T;

                ////////// FACE PROJECTION
                for (size_t i = 0; i < fcs.size(); i++)
                {
                    auto fc = fcs[i];
                    face_basis<Mesh, T> fb(msh, fc, k);

                    Matrix<T, Dynamic, Dynamic> mass_F = make_mass_matrix(msh, fc, k);
                    Matrix<T, Dynamic, 1> RHS_F = Matrix<T, Dynamic, 1>::Zero(fbs);
                    auto qps_F = integrate(msh, fc, 2*k);
                    for (auto& qp : qps_F)
                    {
                        auto phi_F = fb.eval_basis(qp.first);
                        RHS_F += qp.second * ref_fun(qp.first) * phi_F;
                    }

                    // computation of projection coefficients (face)
                    auto M_ldlt_F = mass_F.ldlt();
                    Matrix<T, Dynamic, 1> U_F = M_ldlt_F.solve(RHS_F);


                    ///////// Computation of errors
                    // computation of face projection error
                    auto qps_F_bis = integrate(msh, fc, 2*(k+1));
                    for (auto& qp : qps_F_bis)
                    {
                        auto phi_F = fb.eval_basis(qp.first);
                        auto delta = ref_fun(qp.first) - phi_F.dot(U_F);
                        error_F += hT * qp.second * delta * delta;
                    }
                    // computation of the stabilization error

                    loc_vect.block(cbs+i*fbs, 0, fbs, 1) = U_F;
                }
                auto loc_error = (hho_stab * loc_vect).dot(loc_vect);
                if( loc_error < 0)
                {
                    // std::cout << bold << green << "!!!!! loc_error < 0 !!!!! : "
                    //           << loc_error << std::endl;
                    error_stab -= loc_error;
                }
                else
                    error_stab += loc_error;
            }

            std::cout << bold << red
                      << "mesh_size = " << 1.0/N << std::endl;
            std::cout << bold << yellow
                      << "Errors : proj_cells = " << sqrt(error_T) << std::endl;
            std::cout << "         proj_faces = " << sqrt(error_F) << std::endl;
            std::cout << "         stab_error = " << sqrt(error_stab) << std::endl;
        }
    }
}


//////////////////////////   END  TESTS   /////////////////////////////


///////////////////////   FICTITIOUS DOMAIN METHODS  ///////////////////////////

template<typename T, size_t ET, typename testType>
class fictdom_method
{
    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

protected:
    fictdom_method(){}

    virtual std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi,
                     const element_location where = element_location::IN_NEGATIVE_SIDE,
                     const params<T>& parms = params<T>())
    {
    }

public:
    std::pair<Mat, Vect>
    make_contrib_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType test_case)
    {
        auto gr = make_hho_gradrec_vector(msh, cl, hdi);
        Mat stab = make_hho_naive_stabilization(msh, cl, hdi);
        Mat lc = gr.second + stab;
        Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        return std::make_pair(lc, f);
    }


    std::pair<Mat, Vect>
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType test_case, const hho_degree_info hdi,
                 const element_location where = element_location::IN_NEGATIVE_SIDE,
                 const params<T>& parms = params<T>())
    {
        if( location(msh, cl) == where )
            return make_contrib_uncut(msh, cl, hdi, test_case);
        else if( location(msh, cl) != element_location::ON_INTERFACE )
        {
            Mat lc;
            Vect f;
            return std::make_pair(lc, f);
        }
        else // on interface
            return make_contrib_cut(msh, cl, test_case, hdi, where, parms);
    }
};

/////////////////////////  GRADREC_FICTITIOUS_METHOD

template<typename T, size_t ET, typename testType>
class gradrec_fictdom_method : public fictdom_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    gradrec_fictdom_method(T eta_)
        : fictdom_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi,
                     const element_location where = element_location::IN_NEGATIVE_SIDE,
                     const params<T>& parms = params<T>())
    {
        // LHS
        auto gr = make_hho_gradrec_vector(msh, cl, test_case.level_set_, hdi, where, 1.0);
        Mat stab = make_hho_cut_stabilization(msh, cl, hdi, where)
            + make_hho_cut_interface_penalty(msh, cl, hdi, eta);
        Mat lc = gr.second + stab;


        // RHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        Vect f = Vect::Zero(lc.rows());
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, where);
        f.block(0, 0, cbs, 1) += make_rhs_penalty(msh, cl, celdeg, test_case.bcs_fun, eta);
        f += make_GR_rhs(msh, cl, celdeg, test_case.bcs_fun, test_case.level_set_, gr.first);

        return std::make_pair(lc, f);
    }
};



template<typename T, size_t ET, typename testType>
auto make_gradrec_fictdom_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                 const testType test_case)
{
    return gradrec_fictdom_method<T, ET, testType>(eta_);
}

////////////////////////  NITSCHE_FICTITIOUS_METHOD


template<typename T, size_t ET, typename testType>
class Nitsche_fictdom_method : public fictdom_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    Nitsche_fictdom_method(T eta_)
        : fictdom_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi,
                     const element_location where = element_location::IN_NEGATIVE_SIDE,
                     const params<T>& parms = params<T>())
    {

        // LHS
        auto gr = make_hho_gradrec_vector(msh, cl, test_case.level_set_, hdi, where, 0.0);
        Mat stab = make_hho_cut_stabilization(msh, cl, hdi, where)
            + make_hho_cut_interface_penalty(msh, cl, hdi, eta);
        Mat Nitsche = make_Nitsche(msh, cl, test_case.level_set_, hdi);
        Mat lc = gr.second + stab + Nitsche;


        // RHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        Vect f = Vect::Zero(lc.rows());
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, where);
        f.block(0, 0, cbs, 1) += make_rhs_penalty(msh, cl, celdeg, test_case.bcs_fun, eta);

        cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);

        auto qpsi = integrate_interface(msh, cl, 2*celdeg - 1, element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : qpsi)
        {
            const auto n = test_case.level_set_.normal(qp.first);
            const auto c_dphi  = cb.eval_gradients(qp.first);
            const auto c_dphi_n  = c_dphi * n;

            f.block(0, 0, cbs, 1) -= qp.second * test_case.bcs_fun(qp.first) * c_dphi_n;
        }

        return std::make_pair(lc, f);
    }
};

template<typename T, size_t ET, typename testType>
auto make_Nitsche_fictdom_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                 testType test_case)
{
    return Nitsche_fictdom_method<T, ET, testType>(eta_);
}

/////////////////////////////////

template<typename Mesh, typename testType>
test_info<typename Mesh::coordinate_type>
run_cuthho_fictdom(const Mesh& msh, size_t degree, testType test_case)
{
    using RealType = typename Mesh::coordinate_type;

    auto level_set_function = test_case.level_set_;

    auto rhs_fun = test_case.rhs_fun;
    auto sol_fun = test_case.sol_fun;
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;


    /************** OPEN SILO DATABASE **************/
    silo_database silo;
    silo.create("cuthho_fictdom.silo");
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

    /************** MAKE A SILO VARIABLE FOR LEVEL SET FUNCTION **************/
    std::vector<RealType> level_set_vals;
    for (auto& pt : msh.points)
        level_set_vals.push_back( level_set_function(pt) );
    silo.add_variable("mesh", "level_set", level_set_vals.data(), level_set_vals.size(), nodal_variable_t);

    /************** MAKE A SILO VARIABLE FOR NODE POSITIONING **************/
    std::vector<RealType> node_pos;
    for (auto& n : msh.nodes)
        node_pos.push_back( location(msh, n) == element_location::IN_POSITIVE_SIDE ? +1.0 : -1.0 );
    silo.add_variable("mesh", "node_pos", node_pos.data(), node_pos.size(), nodal_variable_t);


    timecounter tc;

    bool sc = true; // static condensation

    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    element_location where = element_location::IN_NEGATIVE_SIDE;

    tc.tic();
    auto assembler = make_fict_assembler(msh, bcs_fun, hdi, where);
    auto assembler_sc = make_fict_condensed_assembler(msh, bcs_fun, hdi, where);


    // method with gradient reconstruction (penalty-free)
    auto class_meth = make_gradrec_fictdom_method(msh, 1.0, test_case);
    // Nitsche's method
    // auto class_meth = make_Nitsche_fictdom_method(msh, 1.0, test_case);

    for (auto& cl : msh.cells)
    {
        auto contrib = class_meth.make_contrib(msh, cl, test_case, hdi,
                                               element_location::IN_NEGATIVE_SIDE);
        auto lc = contrib.first;
        auto f = contrib.second;

        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
    }

    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;


    /************** SOLVE **************/
    tc.tic();
#if 0
    SparseLU<SparseMatrix<RealType>>  solver;
    Matrix<RealType, Dynamic, 1> sol;

    if( sc )
    {
        solver.analyzePattern(assembler_sc.LHS);
        solver.factorize(assembler_sc.LHS);
        sol = solver.solve(assembler_sc.RHS);
    }
    else
    {
        solver.analyzePattern(assembler.LHS);
        solver.factorize(assembler.LHS);
        sol = solver.solve(assembler.RHS);
    }
#endif
#if 1
    Matrix<RealType, Dynamic, 1> sol;
    cg_params<RealType> cgp;
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    if( sc )
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
    }
    else
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
        cgp.max_iter = assembler.LHS.rows();
        conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
    }
#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;

    /************** POSTPROCESS **************/



    postprocess_output<RealType>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_uT.dat");
    auto Ru_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_Ru.dat");
    auto int_gp  = std::make_shared< gnuplot_output_object<RealType> >("ficdom_int.dat");
    auto diff_gp  = std::make_shared< gnuplot_output_object<RealType> >("fictdom_diff.dat");


    std::vector<RealType>   solution_uT, eigval_data;

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    size_t      cell_i   = 0;
    for (auto& cl : msh.cells)
    {
        bool hide_fict_dom = true; // hide the fictitious domain in the gnuplot outputs
        if (hide_fict_dom && location(msh,cl) == element_location::IN_POSITIVE_SIDE)
            continue;

        cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();

        Matrix<RealType, Dynamic, 1> locdata;
        if( sc )
            locdata = assembler_sc.take_local_data(msh, cl, sol);
        else
            locdata = assembler.take_local_data(msh, cl, sol);

        Matrix<RealType, Dynamic, 1> cell_dofs = locdata.head(cbs);

        auto bar = barycenter(msh, cl, element_location::IN_NEGATIVE_SIDE);

        Matrix<RealType, Dynamic, 1> c_phi = cb.eval_basis(bar);
        auto c_val = cell_dofs.dot( c_phi );
        solution_uT.push_back(c_val);


        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE ||
             location(msh, cl) == element_location::ON_INTERFACE )
        {
            Matrix<RealType, 1, 2> real_grad_int = Matrix<RealType, 1, 2>::Zero();
            Matrix<RealType, 1, 2> comp_grad_int = Matrix<RealType, 1, 2>::Zero();
            auto qps = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);



                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs.dot(t_phi);
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);

                int_gp->add_data( qp.first, 1.0 );
                uT_gp->add_data( qp.first, cell_dofs.dot(t_phi) );
            }
        }

        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;

    postoutput.add_object(uT_gp);
    postoutput.add_object(Ru_gp);
    postoutput.add_object(diff_gp);
    postoutput.add_object(int_gp);
    postoutput.write();

    silo.add_variable("mesh", "uT", solution_uT.data(), solution_uT.size(), zonal_variable_t);

    test_info<RealType> TI;
    TI.H1 = std::sqrt(H1_error);
    TI.L2 = std::sqrt(L2_error);

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;

    return TI;
}



//////////////////////////////  INTERFACE METHODS  ///////////////////////////

template<typename T, size_t ET, typename testType>
class interface_method
{
    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

protected:
    interface_method(){}

    virtual std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     testType test_case, const hho_degree_info hdi)
    {
    }

public:
    std::pair<Mat, Vect>
    make_contrib_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi,  testType test_case)
    {
        T kappa;
        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            kappa = test_case.parms.kappa_1;
        else
            kappa = test_case.parms.kappa_2;
        //std::cout<<"The size of offset cl CONTRIB UNCUT is "<<cl.user_data.offset_subcells.size()<<std::endl;
        
        auto gr = make_hho_gradrec_vector(msh, cl, hdi);
        Mat stab = make_hho_naive_stabilization(msh, cl, hdi);
        Mat lc = kappa * (gr.second + stab);
        Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        return std::make_pair(lc, f);
    }


    std::pair<Mat, Vect>
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl,
                  testType test_case, const hho_degree_info hdi)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_uncut(msh, cl, hdi, test_case);
        else // on interface
            return make_contrib_cut(msh, cl, test_case, hdi);
    }
    
    
};

////////////////////////  NITSCHE INTERFACE METHOD


template<typename T, size_t ET, typename testType>
class Nitsche_interface_method : public interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    Nitsche_interface_method(T eta_)
        : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ////////   LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 0.0);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 0.0);

        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);

        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_1 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_1 * penalty;

        Mat Nitsche = make_NS_Nitsche(msh, cl, level_set_function, hdi).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) -= parms.kappa_1 * Nitsche;
        stab.block(0, 0, cbs, cbs) -= parms.kappa_1 * Nitsche.transpose();
        stab.block(cbs, 0, cbs, cbs) += parms.kappa_1 * Nitsche;
        stab.block(0, cbs, cbs, cbs) += parms.kappa_1 * Nitsche.transpose();

        Mat lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        ////////    RHS
        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        f.head(cbs) += parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.head(cbs) += make_flux_jump(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                      test_case.neumann_jump);

        // pos part
        f.block(cbs, 0, cbs, 1) = make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) += parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);

        return std::make_pair(lc, f);
    }
};

template<typename T, size_t ET, typename testType>
auto make_Nitsche_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case)
{
    return Nitsche_interface_method<T, ET, testType>(eta_);
}


////////////////////////  SYMMETRIC GRADREC INTERFACE METHOD


template<typename T, size_t ET, typename testType>
class Sym_gradrec_interface_method : public interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    Sym_gradrec_interface_method(T eta_)
        : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ///////////////   LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 0.5);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 0.5);

        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);

        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_2 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_2 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_2 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_2 * penalty;

        Mat lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        ////////////////    RHS

        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
        // (see definition of make_Dirichlet_jump)
        f.head(cbs) -= parms.kappa_2 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.head(cbs) += 0.5*make_flux_jump(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                      test_case.neumann_jump);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) += parms.kappa_2 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.block(cbs, 0, cbs, 1)
            += 0.5 * make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                    test_case.neumann_jump);


        // rhs term with GR
        auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi    = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= 0.5 * F_bis.transpose() * (parms.kappa_1 * gr_n.first + parms.kappa_2 * gr_p.first);

        return std::make_pair(lc, f);
    }
};

template<typename T, size_t ET, typename testType>
auto make_sym_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case)
{
    return Sym_gradrec_interface_method<T, ET, testType>(eta_);
}


////////////////////////  GRADREC INTERFACE METHOD (method used in the article)


template<typename T, size_t ET, typename testType>
class gradrec_interface_method : public interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    gradrec_interface_method(T eta_)
        : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ///////////////    LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 1.0);
        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 0.0);

        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);

        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_1 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_1 * penalty;

        Mat lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;

        ///////////////    RHS
        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
        // (see definition of make_Dirichlet_jump)
        f.head(cbs) -= parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) += parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.block(cbs, 0, cbs, 1)
            += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                    test_case.neumann_jump);


        // rhs term with GR
        auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi    = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first );

        return std::make_pair(lc, f);
    }
};

template<typename T, size_t ET, typename testType>
auto make_gradrec_interface_method(const cuthho_mesh<T, ET>& msh, const T eta_,
                                   testType test_case)
{
    return gradrec_interface_method<T, ET, testType>(eta_);
}


////////////////////////  NITSCHE INTERFACE METHOD 2 (hat gradrec - hat gradrec)


template<typename T, size_t ET, typename testType>
class Nitsche_interface_method_2 : public interface_method<T, ET, testType>
{
    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:
    T eta;

    Nitsche_interface_method_2(T eta_)
        : interface_method<T,ET,testType>(), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                    testType test_case, const hho_degree_info hdi)
    {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        ///////////////////    LHS
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        // GR
        auto gr_n = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_NEGATIVE_SIDE, 1.0);

        auto gr_p = make_hho_gradrec_vector_interface(msh, cl, level_set_function, hdi,
                                                      element_location::IN_POSITIVE_SIDE, 1.0);
        // stab
        Mat stab = make_hho_stabilization_interface(msh, cl, level_set_function, hdi, parms);
        Mat penalty = make_hho_cut_interface_penalty(msh, cl, hdi, eta).block(0, 0, cbs, cbs);
        stab.block(0, 0, cbs, cbs) += parms.kappa_1 * penalty;
        stab.block(0, cbs, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, 0, cbs, cbs) -= parms.kappa_1 * penalty;
        stab.block(cbs, cbs, cbs, cbs) += parms.kappa_1 * penalty;
        
        Mat Nitsche = make_NS_Nitsche(msh, cl, level_set_function, hdi).block(0, 0, cbs, cbs);
        stab.block(0, cbs, cbs, cbs) += parms.kappa_2 * Nitsche;
        stab.block(cbs, 0, cbs, cbs) += parms.kappa_2 * Nitsche.transpose();
        stab.block(cbs, cbs, cbs, cbs) -= parms.kappa_2 * Nitsche;
        stab.block(cbs, cbs, cbs, cbs) -= parms.kappa_2 * Nitsche.transpose();
    
        Mat lc = stab + parms.kappa_1 * gr_n.second + parms.kappa_2 * gr_p.second;
        
        //////////////////////    RHS
        Vect f = Vect::Zero(lc.rows());
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                          element_location::IN_NEGATIVE_SIDE);
        //std::cout<<"I m here 10"<<std::endl;
        // we use element_location::IN_POSITIVE_SIDE to get rid of the Nitsche term
        // (see definition of make_Dirichlet_jump)
        f.head(cbs) -= parms.kappa_1 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);
        //std::cout<<"I m here 11"<<std::endl;

        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun,
                                           element_location::IN_POSITIVE_SIDE);
        f.block(cbs, 0, cbs, 1) -= parms.kappa_2 *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_NEGATIVE_SIDE,
                                level_set_function, dir_jump, eta);
        f.block(cbs, 0, cbs, 1) -= (parms.kappa_1-parms.kappa_2) *
            make_Dirichlet_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                level_set_function, dir_jump, eta);

        f.block(cbs, 0, cbs, 1)
            += make_flux_jump(msh, cl, celdeg, element_location::IN_POSITIVE_SIDE,
                                    test_case.neumann_jump);


        // rhs term with GR
        auto gbs = vector_cell_basis<cuthho_poly_mesh<T>,T>::size(hdi.grad_degree());
        vector_cell_basis<cuthho_poly_mesh<T>, T> gb( msh, cl, hdi.grad_degree() );
        Matrix<T, Dynamic, 1> F_bis = Matrix<T, Dynamic, 1>::Zero( gbs );
        auto iqps = integrate_interface(msh, cl, 2*hdi.grad_degree(),
                                        element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps)
        {
            const auto g_phi    = gb.eval_basis(qp.first);
            const Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            F_bis += qp.second * dir_jump(qp.first) * g_phi * n;
        }
        f -= F_bis.transpose() * (parms.kappa_1 * gr_n.first + parms.kappa_2 * gr_p.first);

        return std::make_pair(lc, f);
    }
    
    
};

template<typename T, size_t ET, typename testType>
auto make_Nitsche_interface_method_2(const cuthho_mesh<T, ET>& msh, const T eta_,
                                     testType test_case)
{
    return Nitsche_interface_method_2<T, ET, testType>(eta_);
}

/////////////////////////////////
// Matrix<RealType, Dynamic, 1>


/*

template<typename Mesh, typename testType, typename meth, typename Fonction , typename Velocity>
test_info<typename Mesh::coordinate_type>
run_cuthho_interface_velocity(const Mesh& msh, size_t degree, meth method, testType test_case,  Fonction level_set_function , Velocity velocity)
{
    using RealType = typename Mesh::coordinate_type;
   // using msh =  msh_storage.current_mesh;
   // using agglo_msh = msh_storage.agglomerated_mesh;
    //auto level_set_function = test_case.level_set_;
    
    //auto rhs_fun = test_case.rhs_fun;
    //auto sol_fun = test_case.sol_fun;
    //auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    //auto dirichlet_jump = test_case.dirichlet_jump;
    //auto neumann_jump = test_case.neumann_jump;
    
    struct params<RealType> parms = test_case.parms;

    timecounter tc;

    bool sc = true; // static condensation


    // ************** ASSEMBLE PROBLEM **************
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    // In assembler level_set_function doesn't enter
    auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
    auto assembler_sc = make_interface_condensed_assembler(msh, bcs_fun, hdi);
    for (auto& cl : msh.cells)
    {

        level_set_function.cell_assignment(cl);
       // test_case.test_case_cell_assignment(cl);
       // test_case.cl_uploaded = cl;
        
        auto test_case_cell = make_test_case_laplacian_jumps_3(msh, level_set_function);
        auto rhs_fun_cell = test_case_cell.rhs_fun;
        auto sol_fun_cell = test_case_cell.sol_fun;
        auto sol_grad_cell = test_case_cell.sol_grad;
        auto bcs_fun_cell = test_case_cell.bcs_fun;
        auto dirichlet_jump_cell = test_case_cell.dirichlet_jump;
        auto neumann_jump_cell = test_case_cell.neumann_jump;
        assembler_sc.set_dir_func( bcs_fun_cell);

        
       
        auto contrib = method.make_contrib(msh, cl, test_case_cell, hdi); // To be modified
        auto lc = contrib.first;
        auto f = contrib.second;
        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
        
    
    }
    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    // ************** SOLVE **************
    tc.tic();
#if 0
    SparseLU<SparseMatrix<RealType>>  solver;
    Matrix<RealType, Dynamic, 1> sol;

    if( sc )
    {
        solver.analyzePattern(assembler_sc.LHS);
        solver.factorize(assembler_sc.LHS);
        sol = solver.solve(assembler_sc.RHS);
    }
    else
    {
        solver.analyzePattern(assembler.LHS);
        solver.factorize(assembler.LHS);
        sol = solver.solve(assembler.RHS);
    }
#endif
//#if 0
    Matrix<RealType, Dynamic, 1> sol;
    cg_params<RealType> cgp;
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    if( sc )
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
    }
    else
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
        cgp.max_iter = assembler.LHS.rows();
        conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
    }
//#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;
    //for(auto it= 0;it<sol.size();it++)
     //   std::cout<<"The solution is "<<sol(it)<<std::endl;
    //std::cout<<"The solution size is "<<sol.size()<<std::endl;
    //std::cout<<"The solution is "<<sol<<std::endl;
    // ************** POSTPROCESS **************


    postprocess_output<RealType>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT.dat");
    auto diff_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_diff.dat");


    std::vector<RealType>   solution_uT;

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    size_t      cell_i   = 0;
    RealType    sol_norm = 0.0;
    RealType    sol_norm_max = 0.0;
    
    size_t i_global = 0 ;
    for (auto& cl : msh.cells)
    {
        cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto fbs = face_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.face_degree());

        level_set_function.cell_assignment(cl);
        
        auto test_case_cell = make_test_case_laplacian_jumps_3(msh, level_set_function);
    
        auto rhs_fun_cell = test_case_cell.rhs_fun;
        auto sol_fun_cell = test_case_cell.sol_fun;
        auto sol_grad_cell = test_case_cell.sol_grad;
        auto bcs_fun_cell = test_case_cell.bcs_fun;
        auto dirichlet_jump_cell = test_case_cell.dirichlet_jump;
        auto neumann_jump_cell = test_case_cell.neumann_jump;
        assembler_sc.set_dir_func( bcs_fun_cell); // WITHOUT THIS TERM Dirichlet bdry = 0!!!!
        
        Matrix<RealType, Dynamic, 1> locdata_n, locdata_p, locdata;
        Matrix<RealType, Dynamic, 1> cell_dofs_n, cell_dofs_p, cell_dofs;

        
        
        
        
        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            if( sc )
            {
                locdata_n = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                locdata_n = assembler.take_local_data(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }

            cell_dofs_n = locdata_n.head(cbs);
            cell_dofs_p = locdata_p.head(cbs);
            

            // Uploading velocity field
            auto Lagrange_nodes_Qk = equidistriduted_nodes<RealType,Mesh>(msh, cl, velocity.degree_FEM);
            size_t i_local = 0;
            for ( const auto & ln_Qk : Lagrange_nodes_Qk)
            {
                if( level_set_function(ln_Qk.first) > 0 )
                {
                    auto phi_HHO = cb.eval_basis( ln_Qk.first );
                    velocity.first(i_local,i_global) = cell_dofs_p.dot( phi_HHO );
                 //   velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                }
                else
                {
                    auto phi_HHO = cb.eval_basis( ln_Qk.first );
                    velocity.first(i_local,i_global) = cell_dofs_n.dot( phi_HHO );
                    //   velocity.second(i_local,i_global) = 0; // elliptic case is scalar
                }
            }
            
            
            
            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_n(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad_cell(qp.first) - grad).dot(sol_grad_cell(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_n.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                // Compute L2-error //
                L2_error += qp.second * (sol_fun_cell(qp.first) - v) * (sol_fun_cell(qp.first) - v);
                // Compute solution-norm //
                sol_norm += qp.second * v * v;
             //   std::cout<<"The v is "<<v<<", the qp.second is
            }


            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_p(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad_cell(qp.first) - grad).dot(sol_grad_cell(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_p.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                // Compute L2-error //
                L2_error += qp.second * (sol_fun_cell(qp.first) - v) * (sol_fun_cell(qp.first) - v);
                
                sol_norm += qp.second * v * v;
               
            }
            sol_norm = std::sqrt(sol_norm);
            sol_norm_max = std::max(sol_norm,sol_norm_max);
        }
        else
        {
            if( sc )
            {
                locdata = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
                locdata = assembler.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            cell_dofs = locdata.head(cbs);
            
            
            auto Lagrange_nodes_Qk = equidistriduted_nodes<RealType,Mesh>(msh, cl, velocity.degree_FEM);
            size_t i_local = 0;
            for ( const auto & ln_Qk : Lagrange_nodes_Qk)
            {
                auto phi_HHO = cb.eval_basis( ln_Qk.first );
                velocity.first(i_local,i_global) = cell_dofs.dot( phi_HHO );
                 //   velocity.second(i_local,i_global) = 0; // elliptic case is scalar

            }
            
            
            
            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            
            for (auto& qp : qps)
            {
                // Compute H1-error //
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad_cell(qp.first) - grad).dot(sol_grad_cell(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs.dot(t_phi);
                
                uT_gp->add_data(qp.first, v);

                // Compute L2-error //
                L2_error += qp.second * (sol_fun_cell(qp.first) - v) * (sol_fun_cell(qp.first) - v);
                sol_norm += qp.second * v * v;
            }
            sol_norm = std::sqrt(sol_norm);
            sol_norm_max = std::max(sol_norm,sol_norm_max);
        }

        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error) << std::endl;
    
    std::cout << bold << green << "Max solution norm:           " << sol_norm_max << std::endl;

    postoutput.add_object(uT_gp);
    postoutput.add_object(diff_gp);
    postoutput.write();

    RealType    cell_norm = 1.0/16.0; // Da mettere in main
    std::cout << bold << green << "Max temporal step:           " << cell_norm/sol_norm_max << std::endl;
    test_info<RealType> TI;
    TI.H1 = std::sqrt(H1_error);
    TI.L2 = std::sqrt(L2_error);
    
    
    
    
    if (false)
    {
        /////////////// compute condition number
        SparseMatrix<RealType> Mat;
        // Matrix<RealType, Dynamic, Dynamic> Mat;
        if (sc)
            Mat = assembler_sc.LHS;
        else
            Mat = assembler.LHS;


        RealType sigma_max, sigma_min;

        // Construct matrix operation object using the wrapper class SparseSymMatProd
        Spectra::SparseSymMatProd<RealType> op(Mat);
        // Construct eigen solver object, requesting the largest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 10);
        max_eigs.init();
        max_eigs.compute();
        if(max_eigs.info() == Spectra::SUCCESSFUL)
            sigma_max = max_eigs.eigenvalues()(0);


        // Construct eigen solver object, requesting the smallest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > min_eigs(&op, 1, 10);

        min_eigs.init();
        min_eigs.compute();
        if(min_eigs.info() == Spectra::SUCCESSFUL)
            sigma_min = min_eigs.eigenvalues()(0);

        // compute condition number
        RealType cond = sigma_max / sigma_min;
        TI.cond = cond;
        std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                  << sigma_min << "  cond = " << cond
                  << std::endl;
    }
    else
        TI.cond = 0.0;

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;

    //sol;
        return TI;
    }
*/


template<typename Mesh, typename testType, typename meth, typename Fonction>
test_info<typename Mesh::coordinate_type>
run_cuthho_interface2(const Mesh& msh, size_t degree, meth method, testType test_case,  Fonction level_set_function)
{
    using RealType = typename Mesh::coordinate_type;
   // using msh =  msh_storage.current_mesh;
   // using agglo_msh = msh_storage.agglomerated_mesh;
    //auto level_set_function = test_case.level_set_;
    
    //auto rhs_fun = test_case.rhs_fun;
    //auto sol_fun = test_case.sol_fun;
    //auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    //auto dirichlet_jump = test_case.dirichlet_jump;
    //auto neumann_jump = test_case.neumann_jump;
    
    struct params<RealType> parms = test_case.parms;

    timecounter tc;

    bool sc = true; // static condensation


    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    // In assembler level_set_function doesn't enter
    auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
    auto assembler_sc = make_interface_condensed_assembler(msh, bcs_fun, hdi);
    for (auto& cl : msh.cells)
    {

        level_set_function.cell_assignment(cl);
       // test_case.test_case_cell_assignment(cl);
       // test_case.cl_uploaded = cl;
        
        auto test_case_cell = make_test_case_laplacian_jumps_3(msh, level_set_function);
        auto rhs_fun_cell = test_case_cell.rhs_fun;
        auto sol_fun_cell = test_case_cell.sol_fun;
        auto sol_grad_cell = test_case_cell.sol_grad;
        auto bcs_fun_cell = test_case_cell.bcs_fun;
        auto dirichlet_jump_cell = test_case_cell.dirichlet_jump;
        auto neumann_jump_cell = test_case_cell.neumann_jump;
        assembler_sc.set_dir_func( bcs_fun_cell);
/*
        auto sol_fun = [level_set_function](const typename Mesh::point_type& pt) -> RealType
        { // sol
         
         if(level_set_function(pt) > 0)
             return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y())
                 + 2. + pt.x() * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
            else return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
            
        };
        
        auto rhs_fun = [cl,level_set_function](const typename Mesh::point_type& pt)  ->  RealType
        { // rhs
        std::cout<<"Here in LAMBDA FUNCTION.."<<std::endl;
        std::cout<<"The size of current_cell1 is "<<cl.user_data.offset_subcells.size()<<std::endl;
        std::cout<<"The size of current_cell2 is "<<level_set_function.agglo_LS_cl.user_data.offset_subcells.size()<<std::endl;
        
         if(level_set_function(pt)  > 0)
             return 2.0 * M_PI * M_PI * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y()) - 6 * pt.x() * pt.y() * (pt.x() * pt.x() + pt.y() * pt.y() );
        else return 2.0 * M_PI * M_PI * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
        };
        
        
       // auto bcs_fun = [level_set_function](const typename Mesh::point_type& pt) -> T
       // { // bcs
       //  if(level_set_function(pt)  > 0)
      //      return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y()) + 2.
        //        + pt.x() * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
       //     else return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
            
        //};
        
        
       auto dirichlet_jump = [](const typename Mesh::point_type& pt) -> RealType
        {// Dir
           return - 2. - pt.x() * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
           
       };
        
        
        auto neumann_jump = [level_set_function](const typename Mesh::point_type& pt) -> RealType
        {// Neu
        Matrix<RealType, 1, 2> normal = level_set_function.normal(pt);
        return -3 * pt.x() * pt.x() * pt.y() * pt.y() * (pt.y() * normal(0) + pt.x() * normal(1));
            
        };
        */
        
       
        auto contrib = method.make_contrib(msh, cl, test_case_cell, hdi); // To be modified
        auto lc = contrib.first;
        auto f = contrib.second;
        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
        
    
    }
    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    /************** SOLVE **************/
    tc.tic();
#if 0
    SparseLU<SparseMatrix<RealType>>  solver;
    Matrix<RealType, Dynamic, 1> sol;

    if( sc )
    {
        solver.analyzePattern(assembler_sc.LHS);
        solver.factorize(assembler_sc.LHS);
        sol = solver.solve(assembler_sc.RHS);
    }
    else
    {
        solver.analyzePattern(assembler.LHS);
        solver.factorize(assembler.LHS);
        sol = solver.solve(assembler.RHS);
    }
#endif
//#if 0
    Matrix<RealType, Dynamic, 1> sol;
    cg_params<RealType> cgp;
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    if( sc )
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
    }
    else
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
        cgp.max_iter = assembler.LHS.rows();
        conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
    }
//#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;
    //for(auto it= 0;it<sol.size();it++)
     //   std::cout<<"The solution is "<<sol(it)<<std::endl;
    //std::cout<<"The solution size is "<<sol.size()<<std::endl;
    //std::cout<<"The solution is "<<sol<<std::endl;
    /************** POSTPROCESS **************/


    postprocess_output<RealType>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT.dat");
    auto diff_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_diff.dat");


    std::vector<RealType>   solution_uT;

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    size_t      cell_i   = 0;
    RealType    sol_norm = 0.0;
    RealType    sol_norm_max = 0.0;
    
    for (auto& cl : msh.cells)
    {
        cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto fbs = face_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.face_degree());

        level_set_function.cell_assignment(cl);
        
        auto test_case_cell = make_test_case_laplacian_jumps_3(msh, level_set_function);
    
        auto rhs_fun_cell = test_case_cell.rhs_fun;
        auto sol_fun_cell = test_case_cell.sol_fun;
        auto sol_grad_cell = test_case_cell.sol_grad;
        auto bcs_fun_cell = test_case_cell.bcs_fun;
        auto dirichlet_jump_cell = test_case_cell.dirichlet_jump;
        auto neumann_jump_cell = test_case_cell.neumann_jump;
        assembler_sc.set_dir_func( bcs_fun_cell); // WITHOUT THIS TERM Dirichlet bdry = 0!!!!
        /*
        auto sol_grad = [level_set_function](const typename Mesh::point_type& pt) -> auto
        { // grad
        Matrix<RealType, 1, 2> ret;
        if(level_set_function(pt)  > 0)
        {
            ret(0) = M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y()) + 3*pt.x()*pt.x()*pt.y()*pt.y()*pt.y();
            ret(1) = M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y()) + 3*pt.x()*pt.x()*pt.x()*pt.y()*pt.y();
            return ret;
        }
        else {
            ret(0) = M_PI * std::cos(M_PI*pt.x()) * std::sin(M_PI*pt.y());
            ret(1) = M_PI * std::sin(M_PI*pt.x()) * std::cos(M_PI*pt.y());
            return ret;}
            
        };
        
        auto sol_fun = [level_set_function](const typename Mesh::point_type& pt) -> RealType
        { // sol
         
         if(level_set_function(pt) > 0)
             return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y())
                 + 2. + pt.x() * pt.x() * pt.x() * pt.y() * pt.y() * pt.y();
            else return std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
            
        };
        */
        Matrix<RealType, Dynamic, 1> locdata_n, locdata_p, locdata;
        Matrix<RealType, Dynamic, 1> cell_dofs_n, cell_dofs_p, cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            if( sc )
            {
                locdata_n = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                locdata_n = assembler.take_local_data(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }

            cell_dofs_n = locdata_n.head(cbs);
            cell_dofs_p = locdata_p.head(cbs);
            

            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_n(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad_cell(qp.first) - grad).dot(sol_grad_cell(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_n.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun_cell(qp.first) - v) * (sol_fun_cell(qp.first) - v);
                /* Compute solution-norm */
                sol_norm += qp.second * v * v;
             //   std::cout<<"The v is "<<v<<", the qp.second is
            }


            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_p(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad_cell(qp.first) - grad).dot(sol_grad_cell(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_p.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun_cell(qp.first) - v) * (sol_fun_cell(qp.first) - v);
                
                sol_norm += qp.second * v * v;
               
            }
            sol_norm = std::sqrt(sol_norm);
            sol_norm_max = std::max(sol_norm,sol_norm_max);
        }
        else
        {
            if( sc )
            {
                locdata = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
                locdata = assembler.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            cell_dofs = locdata.head(cbs);
            
            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            
            for (auto& qp : qps)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad_cell(qp.first) - grad).dot(sol_grad_cell(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs.dot(t_phi);
                
                uT_gp->add_data(qp.first, v);

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun_cell(qp.first) - v) * (sol_fun_cell(qp.first) - v);
                sol_norm += qp.second * v * v;
            }
            sol_norm = std::sqrt(sol_norm);
            sol_norm_max = std::max(sol_norm,sol_norm_max);
        }

        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error) << std::endl;
    
    std::cout << bold << green << "Max solution norm:           " << sol_norm_max << std::endl;

    postoutput.add_object(uT_gp);
    postoutput.add_object(diff_gp);
    postoutput.write();

    RealType    cell_norm = 1.0/16.0; // Da mettere in main
    std::cout << bold << green << "Max temporal step:           " << cell_norm/sol_norm_max << std::endl;
    test_info<RealType> TI;
    TI.H1 = std::sqrt(H1_error);
    TI.L2 = std::sqrt(L2_error);

    if (false)
    {
        /////////////// compute condition number
        SparseMatrix<RealType> Mat;
        // Matrix<RealType, Dynamic, Dynamic> Mat;
        if (sc)
            Mat = assembler_sc.LHS;
        else
            Mat = assembler.LHS;


        RealType sigma_max, sigma_min;

        // Construct matrix operation object using the wrapper class SparseSymMatProd
        Spectra::SparseSymMatProd<RealType> op(Mat);
        // Construct eigen solver object, requesting the largest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 10);
        max_eigs.init();
        max_eigs.compute();
        if(max_eigs.info() == Spectra::SUCCESSFUL)
            sigma_max = max_eigs.eigenvalues()(0);


        // Construct eigen solver object, requesting the smallest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > min_eigs(&op, 1, 10);

        min_eigs.init();
        min_eigs.compute();
        if(min_eigs.info() == Spectra::SUCCESSFUL)
            sigma_min = min_eigs.eigenvalues()(0);

        // compute condition number
        RealType cond = sigma_max / sigma_min;
        TI.cond = cond;
        std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                  << sigma_min << "  cond = " << cond
                  << std::endl;
    }
    else
        TI.cond = 0.0;

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;

    //sol;
        return TI;
    }

/////////////////////////////////
// Matrix<RealType, Dynamic, 1>

template<typename Mesh, typename testType, typename meth>
test_info<typename Mesh::coordinate_type>
run_cuthho_interface(const Mesh& msh, size_t degree, meth method, testType test_case)
{
    using RealType = typename Mesh::coordinate_type;

    auto level_set_function = test_case.level_set_;

    auto rhs_fun = test_case.rhs_fun;
    auto sol_fun = test_case.sol_fun;
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    auto dirichlet_jump = test_case.dirichlet_jump;
    auto neumann_jump = test_case.neumann_jump;
    struct params<RealType> parms = test_case.parms;

    timecounter tc;

    bool sc = true; // static condensation


    /************** ASSEMBLE PROBLEM **************/
    hho_degree_info hdi(degree+1, degree);

    tc.tic();
    // In assembler level_set_function doesn't enter
    auto assembler = make_interface_assembler(msh, bcs_fun, hdi);
    auto assembler_sc = make_interface_condensed_assembler(msh, bcs_fun, hdi);
    for (auto& cl : msh.cells)
    {
        auto contrib = method.make_contrib(msh, cl, test_case, hdi); // To be modified
        auto lc = contrib.first;
        auto f = contrib.second;

        if( sc )
            assembler_sc.assemble(msh, cl, lc, f);
        else
            assembler.assemble(msh, cl, lc, f);
    }
    if( sc )
        assembler_sc.finalize();
    else
        assembler.finalize();

    tc.toc();
    std::cout << bold << yellow << "Matrix assembly: " << tc << " seconds" << reset << std::endl;

    if( sc )
        std::cout << "System unknowns: " << assembler_sc.LHS.rows() << std::endl;
    else
        std::cout << "System unknowns: " << assembler.LHS.rows() << std::endl;

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::cout << "Faces: " << msh.faces.size() << std::endl;

    /************** SOLVE **************/
    tc.tic();
#if 0
    SparseLU<SparseMatrix<RealType>>  solver;
    Matrix<RealType, Dynamic, 1> sol;

    if( sc )
    {
        solver.analyzePattern(assembler_sc.LHS);
        solver.factorize(assembler_sc.LHS);
        sol = solver.solve(assembler_sc.RHS);
    }
    else
    {
        solver.analyzePattern(assembler.LHS);
        solver.factorize(assembler.LHS);
        sol = solver.solve(assembler.RHS);
    }
#endif
//#if 0
    Matrix<RealType, Dynamic, 1> sol;
    cg_params<RealType> cgp;
    cgp.histfile = "cuthho_cg_hist.dat";
    cgp.verbose = true;
    cgp.apply_preconditioner = true;
    if( sc )
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler_sc.RHS.rows());
        cgp.max_iter = assembler_sc.LHS.rows();
        conjugated_gradient(assembler_sc.LHS, assembler_sc.RHS, sol, cgp);
    }
    else
    {
        sol = Matrix<RealType, Dynamic, 1>::Zero(assembler.RHS.rows());
        cgp.max_iter = assembler.LHS.rows();
        conjugated_gradient(assembler.LHS, assembler.RHS, sol, cgp);
    }
//#endif
    tc.toc();
    std::cout << bold << yellow << "Linear solver: " << tc << " seconds" << reset << std::endl;
    //for(auto it= 0;it<sol.size();it++)
     //   std::cout<<"The solution is "<<sol(it)<<std::endl;
    std::cout<<"The solution size is "<<sol.size()<<std::endl;
    /************** POSTPROCESS **************/


    postprocess_output<RealType>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_uT.dat");
    auto diff_gp  = std::make_shared< gnuplot_output_object<RealType> >("interface_diff.dat");


    std::vector<RealType>   solution_uT;

    tc.tic();
    RealType    H1_error = 0.0;
    RealType    L2_error = 0.0;
    size_t      cell_i   = 0;
    RealType    sol_norm = 0.0;
    RealType    sol_norm_max = 0.0;
    
    for (auto& cl : msh.cells)
    {
        cell_basis<cuthho_poly_mesh<RealType>, RealType> cb(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto fbs = face_basis<cuthho_poly_mesh<RealType>,RealType>::size(hdi.face_degree());

        Matrix<RealType, Dynamic, 1> locdata_n, locdata_p, locdata;
        Matrix<RealType, Dynamic, 1> cell_dofs_n, cell_dofs_p, cell_dofs;

        if (location(msh, cl) == element_location::ON_INTERFACE)
        {
            if( sc )
            {
                locdata_n = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
            {
                locdata_n = assembler.take_local_data(msh, cl, sol, element_location::IN_NEGATIVE_SIDE);
                locdata_p = assembler.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }

            cell_dofs_n = locdata_n.head(cbs);
            cell_dofs_p = locdata_p.head(cbs);


            auto qps_n = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : qps_n)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_n(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_n.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
                /* Compute solution-norm */
                sol_norm += qp.second * v * v;
             //   std::cout<<"The v is "<<v<<", the qp.second is
            }


            auto qps_p = integrate(msh, cl, 2*hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
            for (auto& qp : qps_p)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs_p(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs_p.dot(t_phi);
                uT_gp->add_data(qp.first, v);

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
                
                sol_norm += qp.second * v * v;
               
            }
            sol_norm = std::sqrt(sol_norm);
            sol_norm_max = std::max(sol_norm,sol_norm_max);
        }
        else
        {
            if( sc )
            {
                locdata = assembler_sc.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            }
            else
                locdata = assembler.take_local_data(msh, cl, sol, element_location::IN_POSITIVE_SIDE);
            cell_dofs = locdata.head(cbs);

            auto qps = integrate(msh, cl, 2*hdi.cell_degree());
            for (auto& qp : qps)
            {
                /* Compute H1-error */
                auto t_dphi = cb.eval_gradients( qp.first );
                Matrix<RealType, 1, 2> grad = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < cbs; i++ )
                    grad += cell_dofs(i) * t_dphi.block(i, 0, 1, 2);

                H1_error += qp.second * (sol_grad(qp.first) - grad).dot(sol_grad(qp.first) - grad);

                auto t_phi = cb.eval_basis( qp.first );
                auto v = cell_dofs.dot(t_phi);
                
                uT_gp->add_data(qp.first, v);

                /* Compute L2-error */
                L2_error += qp.second * (sol_fun(qp.first) - v) * (sol_fun(qp.first) - v);
                sol_norm += qp.second * v * v;
            }
            sol_norm = std::sqrt(sol_norm);
            sol_norm_max = std::max(sol_norm,sol_norm_max);
        }

        cell_i++;
    }

    std::cout << bold << green << "Energy-norm absolute error:           " << std::sqrt(H1_error) << std::endl;
    std::cout << bold << green << "L2-norm absolute error:           " << std::sqrt(L2_error) << std::endl;
    
    std::cout << bold << green << "Max solution norm:           " << sol_norm_max << std::endl;

    postoutput.add_object(uT_gp);
    postoutput.add_object(diff_gp);
    postoutput.write();

    RealType    cell_norm = 1.0/16.0; // Da mettere in main
    std::cout << bold << green << "Max temporal step:           " << cell_norm/sol_norm_max << std::endl;
    test_info<RealType> TI;
    TI.H1 = std::sqrt(H1_error);
    TI.L2 = std::sqrt(L2_error);

    if (false)
    {
        /////////////// compute condition number
        SparseMatrix<RealType> Mat;
        // Matrix<RealType, Dynamic, Dynamic> Mat;
        if (sc)
            Mat = assembler_sc.LHS;
        else
            Mat = assembler.LHS;


        RealType sigma_max, sigma_min;

        // Construct matrix operation object using the wrapper class SparseSymMatProd
        Spectra::SparseSymMatProd<RealType> op(Mat);
        // Construct eigen solver object, requesting the largest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::LARGEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > max_eigs(&op, 1, 10);
        max_eigs.init();
        max_eigs.compute();
        if(max_eigs.info() == Spectra::SUCCESSFUL)
            sigma_max = max_eigs.eigenvalues()(0);


        // Construct eigen solver object, requesting the smallest eigenvalue
        Spectra::SymEigsSolver< RealType, Spectra::SMALLEST_MAGN,
                                Spectra::SparseSymMatProd<RealType> > min_eigs(&op, 1, 10);

        min_eigs.init();
        min_eigs.compute();
        if(min_eigs.info() == Spectra::SUCCESSFUL)
            sigma_min = min_eigs.eigenvalues()(0);

        // compute condition number
        RealType cond = sigma_max / sigma_min;
        TI.cond = cond;
        std::cout << "sigma_max = " << sigma_max << "   sigma_min = "
                  << sigma_min << "  cond = " << cond
                  << std::endl;
    }
    else
        TI.cond = 0.0;

    tc.toc();
    std::cout << bold << yellow << "Postprocessing: " << tc << " seconds" << reset << std::endl;

    //sol;
        return TI;
    }



/////////////////////////   AUTOMATIC TESTS  //////////////////////////


void convergence_test(void)
{
    using T = double;

    std::vector<size_t> mesh_sizes, pol_orders, FE_orders;

    // meshes
    mesh_sizes.push_back(8);
    mesh_sizes.push_back(16);
    mesh_sizes.push_back(32);
    //mesh_sizes.push_back(64);
    //mesh_sizes.push_back(128);
    // mesh_sizes.push_back(256);

    // polynomial orders
    //pol_orders.push_back(0);
    //pol_orders.push_back(1);
    pol_orders.push_back(2);
    //pol_orders.push_back(3);

    // Finite Element orders for Level Set
    //FE_orders.push_back(0);
    FE_orders.push_back(1);
    //FE_orders.push_back(2);
    //FE_orders.push_back(3);
    
    // export to files ...
    std::vector<std::string> files;
    files.push_back("./output/test_k0.txt");
    files.push_back("./output/test_k1.txt");
    files.push_back("./output/test_k2.txt");
    files.push_back("./output/test_k3.txt");

    for (std::vector<size_t>::iterator it = pol_orders.begin(); it != pol_orders.end(); it++)
    {
        size_t k = *it;

        std::cout << "start tests for k = " << k << std::endl;

        // init the file
        std::ofstream output;
        output.open (files.at(*it), std::ios::in | std::ios::trunc);
        if (!output.is_open())
            throw std::logic_error("file not open");

        // output << "N\th\tH1\tordre1\tL2\tordre2" << std::endl;
        output << "N\th\tH1\tordre1\tL2\tordre2\tcond" << std::endl;

        // convergence tests
        T previous_H1 = 0.0;
        T previous_L2 = 0.0;
        T previous_h = 0.0;
        for (std::vector<size_t>::iterator it_msh = mesh_sizes.begin();
             it_msh != mesh_sizes.end(); it_msh++)
        {
            size_t N = *it_msh;

            // init mesh (with agglomeration)
            mesh_init_params<T> mip;
            mip.Nx = N;
            mip.Ny = N;
            cuthho_poly_mesh<T> msh(mip);
            size_t int_refsteps = 1;
            T radius = 1.0/3.0;
            auto circle_level_set_function = circle_level_set<T>(radius, 0.5, 0.5);

            // auto level_set_function = flower_level_set<T>(0.31, 0.5, 0.5, 4, 0.04);
            // auto level_set_function = circle_level_set<T>(radius, 0.5, 0.5);
            // auto level_set_function = square_level_set<T>(1.05, -0.05, -0.05, 1.05);
            // auto level_set_function = square_level_set<T>(1.0, -0.0, -0.0, 1.0);
            // auto level_set_function = square_level_set<T>(0.76, 0.24, 0.24, 0.76);
           // auto level_set_function = square_level_set<T>(0.751, 0.249, 0.249, 0.751);
            
            
           // typedef RealType T;
            typedef  circle_level_set<T> Fonction;
            typedef cuthho_poly_mesh<T> Mesh;
            //typedef  flower_level_set<T> Fonction; //DA RIMETTERE POI PER LA DISCRETA
            //typedef line_level_set<T> Fonction;

             /************** LEVEL SET FUNCTION DISCRETISATION **************/
            size_t degree_FEM = 2;
            std::cout<<"degree FEM "<<degree_FEM<<std::endl;
            auto level_set_function = projected_level_set< T, Mesh,Fonction>(circle_level_set_function, msh, degree_FEM , mip);
            
            
            detect_node_position2(msh, level_set_function);
            detect_cut_faces2(msh, level_set_function);
            if(1)  // AGGLOMERATION
            {
                detect_cut_cells(msh, level_set_function);
                detect_cell_agglo_set(msh, level_set_function);
                make_neighbors_info_cartesian(msh);
                refine_interface2(msh, level_set_function, int_refsteps);
                make_agglomeration(msh, level_set_function);
            }
            else  // NODE DISPLACEMENT
            {
                move_nodes(msh, level_set_function);
                detect_cut_faces2(msh, level_set_function); //do it again to update intersection points
                detect_cut_cells(msh, level_set_function);
                refine_interface2(msh, level_set_function, int_refsteps);
            }

            // compute solution/errors
            test_info<T> TI;
            // auto TI = run_cuthho_interface(msh, level_set_function, k, 3);

            // auto TI = run_cuthho_interface(msh, level_set_function, k, 3, test_case);
            if(0) // sin(\pi x) * sin(\pi y)
            {
                auto test_case = make_test_case_laplacian_sin_sin(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
                // TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(0) // 1 + sin(\pi x) * sin(\pi y)
            {
                auto test_case = make_test_case_laplacian_sin_sin_bis(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                // TI = run_cuthho_interface(msh, k, meth3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(0) // sin(\pi (x-a)/(b-a)) * sin(\pi (y-c)/(d-c))
            {
                T a = 0.23;
                T b = 0.77;
                T c = 0.23;
                T d = 0.77;
                auto test_case = make_test_case_laplacian_sin_sin_gen(msh, level_set_function,
                                                                      a, b, c, d);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                // TI = run_cuthho_interface(msh, k, meth3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(0) // exp(x) * cos(y)
            {
                auto test_case = make_test_case_laplacian_exp_cos(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                // TI = run_cuthho_interface(msh, k, meth3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }
            if(1) // jumps sin_sin -> exp_cos
            {
                auto test_case = make_test_case_laplacian_jumps_1(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
            }
            if(0) // jumps2 exp_cos -> sin_sin
            {
                auto test_case = make_test_case_laplacian_jumps_2(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
            }
            if(0) // jumps3 sin_sin -> sin_sin + pol
            {
                auto test_case = make_test_case_laplacian_jumps_3(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
            }
            if(0) // contrast deg 2
            {
                auto parms = params<T>();
                parms.kappa_1 = 1.0;
                parms.kappa_2 = 1000.0;

                auto test_case = make_test_case_laplacian_contrast_2(msh, circle_level_set_function, parms);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
            }
            if(0) // contrast deg 6
            {
                auto parms = params<T>();
                parms.kappa_1 = 1.0;
                parms.kappa_2 = 1.0;

                auto test_case = make_test_case_laplacian_contrast_6(msh, circle_level_set_function, parms);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
            }

            if(0) // homogeneous test case on a circle
            {
                auto test_case = make_test_case_laplacian_circle_hom(msh, circle_level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                // TI = run_cuthho_interface(msh, k, meth3, test_case);
                TI = run_cuthho_fictdom(msh, k, test_case);
            }

            // auto TI = run_cuthho_fictdom(msh, level_set_function, k);


            // report info in the file
            T h = 1.0/N;
            if (it_msh == mesh_sizes.begin())
            {
                // output << N << "\t" << h << "\t" << TI.H1 << "\t" << "."
                //        << "\t" << TI.L2 << "\t" << "." << "\t" << "0.0"
                //        << std::endl;
                output << N << "\t" << h << "\t" << TI.H1 << "\t" << "."
                       << "\t" << TI.L2 << "\t" << "." << "\t" << TI.cond
                       << std::endl;
            }
            else
            {
                T orderH = log(previous_H1 / TI.H1) / log(previous_h / h);
                T orderL = log(previous_L2 / TI.L2) / log(previous_h / h);
                // output << N << "\t" << h << "\t" << TI.H1 << "\t" << orderH
                //        << "\t" << TI.L2 << "\t" << orderL << "\t" << "0.0"
                //        << std::endl;
                output << N << "\t" << h << "\t" << TI.H1 << "\t" << orderH
                       << "\t" << TI.L2 << "\t" << orderL << "\t" << TI.cond
                       << std::endl;
            }
            previous_H1 = TI.H1;
            previous_L2 = TI.L2;
            previous_h = h;
        }
        // close the file
        output.close();
    }

    // update the gnuplot curves
    system("gnuplot './output/gnuplot_script.txt'");

    // update the .pdf file
    system("pdflatex ./output/autom_tests.tex");

    // open the .pdf file
    system("xdg-open ./autom_tests.pdf");
}


//////////////////////////     MAIN        ////////////////////////////

#if 0
int main(int argc, char **argv)
{
    convergence_test();
    // tests_stabilization();
    // interface_residus();
    return 1;
}
#endif

#if 0
int main(int argc, char **argv)
{
    using RealType = double;

    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t degree_FEM       = 0;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;

    /* k <deg>:     method degree
     * g<deg>:  method FEM degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:g:M:N:r:ifDAd")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'g':
                degree_FEM = atoi(optarg);
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;


    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    typedef cuthho_poly_mesh<RealType> Mesh;
    offset_definition(msh);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    

    /************** ANALYTIC LEVEL SET FUNCTION  **************/
    RealType radius = 1.0/3.0;
    //auto level_set_function_anal = circle_level_set<RealType>(radius, 0.5, 0.5);
   auto level_set_function_anal = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);
    // auto level_set_function = line_level_set<RealType>(0.5);
    
    
    /************** ANALYTIC LEVEL SET FUNCTION, OLD IMPLEMENTATION **************/
    //auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    //auto level_set_function = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);
    // auto level_set_function = line_level_set<RealType>(0.5);


    typedef RealType T;
   // typedef  circle_level_set<T> Fonction;
    typedef  flower_level_set<T> Fonction; //DA RIMETTERE POI PER LA DISCRETA
    //typedef line_level_set<T> Fonction;

     /************** LEVEL SET FUNCTION DISCRETISATION **************/
    std::cout<<"degree FEM "<<degree_FEM<<std::endl;
    auto level_set_function = projected_level_set< T, Mesh,Fonction>(level_set_function_anal, msh, degree_FEM , mip);

    /************** Initialization of meshes-storage **************/
    auto msh_storage =  Current_Mesh<Mesh>(msh);
    
     /************** LEVEL SET FUNCTION TESTING **************/
    /*
     tc.tic();
     for (auto& cl : msh.cells)
     {
         //auto offset_curr = offset(msh,cl);
         level_set_function1.cell_assignment(cl);

         gradient_checking(msh , level_set_function1 ,  level_set_function_anal , cl);
     }
    tc.toc();
    std::cout << bold << yellow << "Total time for the NEW gradient operator() : " << tc << " seconds" << reset << std::endl;
    
    tc.tic();
     for (auto& cl : msh.cells)
     {
         gradient_checking1(msh , level_set_function ,  level_set_function_anal , cl);
     }
    tc.toc();
    std::cout << bold << yellow << "Total time for the OLD gradient operator() : " << tc << " seconds" << reset << std::endl;
    
    */
   //  testing_level_set(msh,level_set_function,level_set_function_anal);
    /*
    tc.tic();
       for (auto& n:msh.nodes)
       {
           std::cout<< bold << yellow <<"NODE NUMBER "<<n<< reset << std::endl;
           auto valueA = level_set_function(n);
                      
       }
    for (auto&cl:msh.cells) {
        auto nn = nodes(msh,cl);
        for (auto& n1:nn) {
            auto pt = points(msh, n1);
            auto valueA = level_set_function(pt,msh,cl);
            size_t counter_cell = offset(msh,cl);
            std::cout<< bold << yellow <<n1<<", with value "<<valueA<< reset << std::endl;
                
        }
    }
    
    
       tc.toc();
       std::cout << bold << yellow << "Total time for the FACE operator() : " << tc << " seconds" << reset << std::endl;
    */
    
    /*
    tc.tic();
    for (auto& cl:msh.cells)
    {
        auto fcs = faces(msh, cl);
        size_t counter_cell = offset(msh,cl);
        std::cout<< bold << yellow <<"CELL NUMBER "<<counter_cell<< reset << std::endl;
        for(auto& fc:fcs)
        {
        size_t counter_face = offset(msh,fc);
        //time_face_testing(msh , level_set_function , fc);
        test_new_method( msh , level_set_function , level_set_function_anal,cl,fc );
        }
        //test_new_method( msh , level_set_function , level_set_function_anal , cl );
        //std::cout<<"Sono dentro a cosa nuova"<<std::endl;
        // time_NEWdiscrete_testing(msh , level_set_function , cl);
    }
    tc.toc();
    std::cout << bold << yellow << "Total time for the FACE operator() : " << tc << " seconds" << reset << std::endl;
    */
    
    /*
    tc.tic();
       for (auto& cl:msh.cells) {
           auto fcs = faces(msh, cl);
           size_t counter_cell = offset(msh,cl);
           std::cout<< bold << yellow <<"CELL NUMBER "<<counter_cell<< reset << std::endl;
           for(auto& fc:fcs)
           {
           size_t counter_face = offset(msh,fc);
           time_faceANALITIC_testing(msh , level_set_function_anal , fc);
           }
           //test_new_method( msh , level_set_function , level_set_function_anal , cl );
           //std::cout<<"Sono dentro a cosa nuova"<<std::endl;
           // time_NEWdiscrete_testing(msh , level_set_function , cl);
       }
       tc.toc();
       std::cout << bold << yellow << "Total time for the ANALYTIC operator() : " << tc << " seconds" << reset << std::endl;
    */
    
    /*
    tc.tic();
    for (auto& cl:msh.cells) {
        //test_new_method( msh , level_set_function , level_set_function_anal , cl );
        //std::cout<<"Sono dentro a cosa nuova"<<std::endl;
        time_OLDdiscrete_testing(msh , level_set_function , cl);
    }
    tc.toc();
    std::cout << bold << yellow << "Total time for the old operator() : " << tc << " seconds" << reset << std::endl;
    */
    
    /*
    tc.tic();
    for (auto& cl:msh.cells) {
        //test_new_method( msh , level_set_function , level_set_function_anal , cl );
        //std::cout<<"Sono dentro a cosa nuova"<<std::endl;
        time_analytic_testing(msh , level_set_function_anal , cl);
    }
    tc.toc();
    std::cout << bold << yellow << "Total time for the analitic function operator() : " << tc << " seconds" << reset << std::endl;
    */
    
    
    
    /************** DO cutHHO MESH PROCESSING **************/
    tc.tic();
    detect_node_position2(msh, level_set_function); // IN cuthho_geom->da modificare operator() ->problema che giro su n:nodes
 
    detect_cut_faces2(msh, level_set_function); // IN cuthho_geom->da modificare operator() ->problema che giro su fc:faces

    //std::cout<<"Number of cells "<<msh.cells.size()<<std::endl;
    //std::cout<<"Number of nodes "<<msh.nodes.size()<<std::endl;
    //std::cout<<"Number of faces "<<msh.faces.size()<<std::endl;
    //std::cout<<"Number of points "<<msh.points.size()<<std::endl;
    
    
    
    if (agglomeration)
    {
        
        detect_cut_cells2(msh, level_set_function); // IN cuthho_geom

        detect_cell_agglo_set(msh, level_set_function); // Non serve modificarla
        std::cout << bold << yellow << "before make_neighbors_info_cartesian"<< reset << std::endl;
        make_neighbors_info_cartesian(msh); // Non serve modificarla, in cuthho_agglomeration
    
        refine_interface2(msh, level_set_function, int_refsteps); // IN cuthho_geom->da modificare operator() ->problema se esco da cella
        std::cout << bold << yellow << "after refine_interface2"<< reset << std::endl;
        //for (auto& cl : msh.cells)
          //     {
            //       check_negative_weights<T,Mesh>( msh, cl, int_refsteps);
               
             //  }
               
        make_agglomeration(msh, level_set_function); // Non serve modificarla
       std::cout << bold << yellow << "after make_agglomeration"<< reset << std::endl;
    }
    else
    {
        move_nodes(msh, level_set_function);
        detect_cut_faces2(msh, level_set_function); //do it again to update intersection points
        detect_cut_cells2(msh, level_set_function);
        refine_interface2(msh, level_set_function, int_refsteps);
    }
    
    tc.toc();
    std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

    // NON CREDO SI ENTRI MAI QUA DENTRO..
    if (dump_debug)
    {
        dump_mesh(msh);
        output_mesh_info(msh, level_set_function);
        test_projection(msh, level_set_function, degree);
    }
    
    output_mesh_info2(msh, level_set_function); // IN cuthho_export.. Points/Nodes doesn't change after agglomeration-> I can redefine operator()(n)
    /*
    std::cout<<"Number of cells after agglomeration "<<msh.cells.size()<<std::endl;
    std::cout<<"Number of nodes after agglomeration "<<msh.nodes.size()<<std::endl;
    std::cout<<"Number of faces after agglomeration "<<msh.faces.size()<<std::endl;
    std::cout<<"Number of points after agglomeration "<<msh.points.size()<<std::endl;
    */
    /*
    for(auto & cl : msh.cells)
    {
        std::cout<<"I m here 0.0 and size is "<<cl.user_data.offset_subcells.size()<<std::endl;
        
        auto pts = points(msh,cl);
        auto number_pts =pts.size();
        std::cout<<"In A-cell num "<<offset(msh,cl)<<", there is "<<number_pts<<" points."<<std::endl;
        if(number_pts>4)
        {
            size_t counter = 0;
            for(auto & pt : pts){
                std::cout<<"Point num "<<counter<<", position = ( "<<pt.x()<<" , "<<pt.y()<<" )"<<std::endl;
            }
        }
        
        
    }
    */
    
    // jumps sin_sin -> exp_cos
    // auto test_case = make_test_case_laplacian_jumps_1(msh, level_set_function);
    // jumps3 sin_sin -> sin_sin + pol
    
    msh_storage.set_agglomesh(msh);
    typedef projected_level_set< T, Mesh,Fonction> Level_Set;
    auto ls_cell = LS_cell< T , Mesh , Level_Set, Fonction >(level_set_function,msh);
    
    auto current_cell =  Current_Cell<Mesh>();
    
    //auto test_case = make_test_case_laplacian_jumps_3bis(msh, ls_cell,current_cell);
    auto test_case = make_test_case_laplacian_jumps_3(msh, ls_cell); // 2 possibilità.. 1) creo 3bis 2) anzichè passarle level_set_function le passo LS_cell

    // auto method = make_Nitsche_interface_method(msh, 1.0, test_case);
    // auto method = make_sym_gradrec_interface_method(msh, 1.0, test_case);
    // auto method = make_gradrec_interface_method(msh, 1.0, test_case);
    auto method = make_Nitsche_interface_method_2(msh, 1.0, test_case);
    //Matrix<RealType, Dynamic, 1> sol;
    if(solve_interface){
        run_cuthho_interface2(msh, degree, method,test_case, ls_cell);
        //run_cuthho_interface(msh, degree, method, test_case);
    }
    if (solve_fictdom)
        run_cuthho_fictdom(msh, degree, test_case);

    //sol =  run_cuthho_interface(msh, degree, method, test_case);
    return 0;
}
#endif


/**************MOVING INTERFACE: LEVEL SET METHOD  **************/

template<typename T, typename Mesh >
struct projection
{
    Eigen::Matrix<T, Dynamic, Dynamic> values_bis;
    Eigen::Matrix<T, Dynamic, 1> vertices; // saving level_set on vertices mesh
    
    size_t degree_FEM;
    size_t number_elements;
    Mesh msh;
    size_t      Nx, Ny ;
    mesh_init_params<T> params;
    //size_t  last_row_init, last_row_end, number_faces_one_row;

    projection( const Mesh & msh, size_t degree_k , const mesh_init_params<T>& params)
        : number_elements((degree_k+1)*(degree_k+1)), msh(msh),degree_FEM(degree_k),Nx(params.Nx),Ny(params.Ny), params(params)
    {
        vertices = Eigen::Matrix<T, Dynamic, 1>::Zero( ((Nx+1)*(Ny+1)), 1 );
        values_bis= Eigen::Matrix<T, Dynamic, Dynamic>::Zero(number_elements, msh.cells.size());
    }
    
    projection()=default;
    
    
    void  set_discrete_points( Eigen::Matrix<T, Dynamic, Dynamic>& values_new)
    {
        values_bis=values_new;
    }
    
    
    void converting_into_HHO_formulation( const Eigen::Matrix<T, Dynamic, 1>& vertices )
    {
        assert(degree_FEM == 1);
        for (size_t j = 0; j < values_bis.cols() ; j++)
        {

            for (size_t i = 0; i < values_bis.rows() ; i++)
            {
                auto i_vertex = j + floor(j/Nx);
                if ( i == 0 )
                    values_bis(i,j) = vertices(i_vertex);
                
                if( i ==1 )
                    values_bis(i,j) = vertices(i_vertex+1) ;
                
                if( i ==(degree_FEM+2) )
                    values_bis(i,j) = vertices(i_vertex+Nx+2) ;
                
                if( i ==(degree_FEM+1) )
                    values_bis(i,j) = vertices(i_vertex+Nx+1) ;
            }
        }
                 
    }
    
    void converting_into_FE_formulation( const Eigen::Matrix<T, Dynamic, Dynamic>& values )
    {
        assert(degree_FEM == 1);
        for (size_t j = 0; j < values_bis.cols() ; j++)
        {

            for (size_t i = 0; i < values_bis.rows() ; i++)
            {
                auto i_vertex = j + floor(j/Nx);
                if ( i == 0 )
                    vertices(i_vertex) = values(i,j) ;
                
                if( i ==1 )
                    vertices(i_vertex+1) = values(i,j) ;
                
                if( i ==(degree_FEM+2) )
                    vertices(i_vertex+Nx+2) = values(i,j) ;
                
                if( i ==(degree_FEM+1) )
                    vertices(i_vertex+Nx+1) = values(i,j) ;
            }
        }
                 
    }
       
    
    
   
    T operator()( const typename Mesh::node_type& node ) const
    {
        return vertices(node.ptid);
    }
    
    T operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        size_t counter = offset(msh,cl) ;
        cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = (values_bis.block(0,counter,number_elements,1)).col(0);
        T tmp = values_cell.dot( cb.eval_basis(pt) );
        return tmp;
    }
  
    Eigen::Matrix<T,2,1> gradient( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl ) const
    {
        size_t counter = offset(msh,cl);
        Eigen::Matrix<T,2,1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell = values_bis.col(counter);
        auto grad_eval =  cb.eval_gradients(pt);
        ret(0) = values_cell.dot( grad_eval.col(0) );
        ret(1) = values_cell.dot( grad_eval.col(1) );
        return ret;
    }


    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt, const Mesh& msh, const typename Mesh::cell_type& cl) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret = gradient(pt,msh,cl);
        return ret/ret.norm();
        
    }
    
};




template <typename T>
void positive_part( Eigen::Matrix<T, Dynamic, Dynamic>& mat) {
    for (size_t i = 0; i<mat.rows();i++) {
        for (size_t j = 0; j<mat.cols();j++) {
            if( mat(i,j) < 0 )
                mat(i,j) = 0.;
        }
    }

}





/*

template <typename T>
Eigen::Matrix<T, Dynamic, Dynamic> positive_part(const Eigen::Matrix<T, Dynamic, Dynamic>& mat) {
    //Eigen::Matrix<T, Dynamic, Dynamic> ret =mat;
    Eigen::Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(mat.rows(), mat.cols());
    auto positive = mat.cwiseSign();
    std::cout<<"positive: "<<'\n'<<positive<<std::endl;
    for (size_t i = 0; i<mat.rows();i++) {
        for (size_t j = 0; j<mat.cols();j++) {
            if (positive(i,j)==-1) {
                ret(i,j) = 0;
            }
            //if( mat(i,j) < 0 )
            //    ret(i,j) = 0;
        }
    }
    return ret;
}
*/


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template<typename T, typename Mesh>
class entropy
{
public:
    virtual T operator()(const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
    }
    
    virtual T derivative(const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
    }
   
};

template<typename T , typename Fonction , typename Mesh>
struct non_linear_entropy: public entropy<T,Mesh>
{
    const T eps;
    Fonction phi;
    T phi_max , phi_min ;
    Mesh msh;
    
    non_linear_entropy(const T eps , const Fonction& phi ,const Mesh& msh): eps(eps),phi(phi),msh(msh)
    {
        phi_max = phi.phi_max;
        phi_min = phi.phi_min;
        
    }
    
    /// THIS WAS FOR A GENERAL PHI, RE-WRITTEN FOR PHI BETWEEN 0 AND 1
    /*
    T operator()(const Fonction& f, const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
        return -std::log10( std::abs( (phi_max-f(pt,msh,cl))*(f(pt,msh,cl)-phi_min) ) + eps );
    }
    
    T operator()(const point<T,2>& pt , const typename Mesh::cell_type& cl) const
    {
        return -std::log10( std::abs( (phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min) ) + eps );
    }
    
    T derivative(const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
       // std::cout<<"il segno di "<<((phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ))<<" is "<< sgn( (phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ) )<<std::endl;
        return -1./(( std::abs( (phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ) ) +eps)*std::log(10)) * sgn( (phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ) ) * ( phi_max - 2*phi(pt,msh,cl) + phi_min ) ;
    }
    */
    
    T operator()(const Fonction& f, const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
        return -std::log10( std::abs( f(pt,msh,cl)*(1-f(pt,msh,cl)) ) + eps );
    }
    
    T operator()(const point<T,2>& pt , const typename Mesh::cell_type& cl) const
    {
        return -std::log10( std::abs( phi(pt,msh,cl)*(1-phi(pt,msh,cl)) ) + eps );
    }
    
    T derivative(const point<T,2>& pt, const typename Mesh::cell_type& cl) const
    {
       // std::cout<<"il segno di "<<((phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ))<<" is "<< sgn( (phi_max-phi(pt,msh,cl))*(phi(pt,msh,cl)-phi_min ) )<<std::endl;
        return -1./( ( std::abs( (1-phi(pt,msh,cl))*phi(pt,msh,cl)  ) +eps )*std::log(10) ) * sgn( (1-phi(pt,msh,cl))*phi(pt,msh,cl)  ) * ( 1 - 2*phi(pt,msh,cl)  ) ;
    }
   
};


template<typename T>
class velocity_field
{
   public:
    virtual Eigen::Matrix<T,2,1> operator()(const point<T,2>& pt) const
    {
    }
    
    /*
     virtual Eigen::Matrix<T,2,1> flux(const point<T,2>& pt) const
    {
    }
    
    Eigen::Matrix<T,2,1> normal(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;

        ret = gradient(pt);
        return ret/ret.norm();
    }
    */
   
};



template<typename T>
struct rotational_velocity_field: public velocity_field<T>
{
    T u1, xc , yc;

    rotational_velocity_field(T u1 )
    : u1(u1){}

    rotational_velocity_field(T xc , T yc , T u1 )
    : u1(u1) , xc(xc) , yc(yc)
    {}

    Eigen::Matrix<T,2,1> operator()(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret(0) = (  u1*pt.y() -xc )/xc ;
        ret(1) = (- u1*pt.x() + yc ) / yc ;
       
        return ret;
    }
    
    
};

template<typename T>
struct linear_velocity_field: public velocity_field<T>
{
    T u1, u2 , u3 , u4;

    linear_velocity_field(T u1 , T u2 , T u3 , T u4 )
        : u1(u1), u2(u2) , u3(u3) , u4(u4)
    {}

    Eigen::Matrix<T,2,1> operator()(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret(0) = u1*pt.x() + u2;
        ret(1) = u3*pt.y() + u4;
       
        return ret;
    }
/*
    Eigen::Matrix<T,2,1> gradient(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
        ret(0) = 2*pt.x() - 2*alpha;
        ret(1) = 2*pt.y() - 2*beta;
        return ret;
    }
    */
};

template<typename T>
struct stokes_field: public velocity_field<T>
{
    T a = 2.0 , b = 2.0 , c = 1.0/M_PI;
    stokes_field(T a , T b): a(a),b(b){};
    stokes_field(T a , T b , T c): a(a),b(b),c(c){};
    
    stokes_field(){};

    Eigen::Matrix<T,2,1> operator()(const point<T,2>& pt) const
    {
        Eigen::Matrix<T,2,1> ret;
    // ret(0) = 2*M_PI*(std::sin(2*M_PI*pt.x())*std::sin(2*M_PI*pt.y())) ;
        ret(0) = (b*M_PI*(std::sin(a*M_PI*pt.x())*std::sin(b*M_PI*pt.y())))/(c*M_PI) ;
    //ret(1) = 2*M_PI*(std::cos(2*M_PI*pt.x()) - std::cos(2*M_PI*pt.x())*std::cos(2*M_PI*pt.y()));
        ret(1) = ( a*M_PI*(std::cos(a*M_PI*pt.x()) - std::cos(a*M_PI*pt.x())*std::cos(b*M_PI*pt.y()) ) ) /(c*M_PI) ;
       
        return ret;
    }
    
    
};

template<typename Mesh, typename T = typename Mesh::coordinate_type>
Matrix<T, Dynamic, Dynamic>
make_lagrange_local_mass_matrix(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, size_t di = 0)
{
    
    //auto f_neigh = cl.user_data.f_neighbors;
    //auto d_neigh = cl.user_data.d_neighbors;
    
    cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    //Matrix<T, Dynamic, 1> ret2 = Matrix<T, Dynamic, 1>::Zero(cbs, 1);

    auto qps = integrate(msh, cl, 2*(degree+di)); // integration of order 2k

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi * phi.transpose();
        // phi * phi.transpose is (degree+1)^2 x (degree+1)^2 ; qp.second is a scalar
    }
    //ret2  =  ret.rowwise().sum(); // sum row; i.e. mi0 + mi1 + mi2 + mi3 , with i = 0 : 3
    
    return ret;
}

template<typename Mesh, typename T = typename Mesh::coordinate_type>
Matrix<T, Dynamic, Dynamic>
make_lagrange_lumped_mass_matrix(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, size_t di = 0)
{
    cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs, 1);

    auto qps = integrate(msh, cl, 2*(degree+di));

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi;
    }

    return ret;
}

template<typename Mesh, typename T = typename Mesh::coordinate_type>
std::pair< Matrix<T, Dynamic, Dynamic> , Matrix<T, Dynamic, Dynamic> >
make_local_cij_matrix(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, size_t di = 0)
{
    cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, Dynamic> ret0 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    Matrix<T, Dynamic, Dynamic> ret1 = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    
    // for Q1 , degree = 1-> integration of order 2
    auto qps = integrate(msh, cl, 2*(degree+di)); // integration of order 2k

    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        
        auto phi_grad = cb.eval_gradients(qp.first);
        
        ret0 += qp.second * phi * ((phi_grad).col(0)).transpose();
    
        ret1 += qp.second * phi * ((phi_grad).col(1)).transpose();
    }
    
    return std::make_pair(ret0,ret1);
}




template<typename T, typename Mesh ,typename Fonction >
struct projection_velocity: public velocity_field<T>
{
    // MATRIX NOTATION
    std::pair<Eigen::Matrix<T, Dynamic, Dynamic> , Eigen::Matrix<T, Dynamic, Dynamic> > values_bis;
    //Eigen::Matrix<T, Dynamic, 1> values_bis1; // VECTOR NOTATION
    
    // saving velocity fields on vertices mesh -> JUST FOR Q1, TO BE GENERALISED
    std::pair<Eigen::Matrix< T, Dynamic, 1>,Eigen::Matrix< T, Dynamic, 1>  >vertices;
    size_t degree_FEM;
    size_t number_elements;
    Mesh msh;
    size_t      Nx, Ny ;
  //  mesh_init_params<T> params;
    size_t  last_row_init, last_row_end, number_faces_one_row;
  //  size_t counter_cell , counter_face, num_cell_row;
    
    
    
    projection_velocity(const Mesh & msh, size_t degree_k , const mesh_init_params<T>& params)
        : number_elements((degree_k+1)*(degree_k+1)), msh(msh),degree_FEM(degree_k),Nx(params.Nx),Ny(params.Ny) //, params(params)
    {
        last_row_init = Ny*(2*Nx+1); // There are 2 faces for each row of cells + Ny
        last_row_end = last_row_init + Nx-1;
        number_faces_one_row = 2*Nx+1; // for each cell I count only the low and sx faces, respectevely 0-1 2-3 4-5 6-7 8 + the last on the right boundary
        vertices = std::make_pair( Eigen::Matrix<T, Dynamic, 1>::Zero( ((Nx+1)*(Ny+1)), 1 ) , Eigen::Matrix<T, Dynamic, 1>::Zero( ((Nx+1)*(Ny+1)), 1 ) );

        values_bis = std::make_pair(Eigen::Matrix<T, Dynamic, Dynamic>::Zero(number_elements, msh.cells.size()) , Eigen::Matrix<T, Dynamic, Dynamic>::Zero(number_elements, msh.cells.size()));
    //#endif
    }
    
    
    
    projection_velocity(const Fonction & u, const Mesh & msh, size_t degree_k , const mesh_init_params<T>& params)
        : number_elements((degree_k+1)*(degree_k+1)), msh(msh),degree_FEM(degree_k),Nx(params.Nx),Ny(params.Ny) //, params(params)
    {
        last_row_init = Ny*(2*Nx+1); // There are 2 faces for each row of cells + Ny
        last_row_end = last_row_init + Nx-1;
        number_faces_one_row = 2*Nx+1; // for each cell I count only the low and sx faces, respectevely 0-1 2-3 4-5 6-7 8 + the last on the right boundary
        vertices = std::make_pair( Eigen::Matrix<T, Dynamic, 1>::Zero( ((Nx+1)*(Ny+1)), 1 ) , Eigen::Matrix<T, Dynamic, 1>::Zero( ((Nx+1)*(Ny+1)), 1 ) );
        // MAYBE I COULD DO LIKE BEFORE -> (accelerate??)
    //#ifdef NODES
        std::cout<<"problema qui 0"<<std::endl;
        // MATRIX NOTATION
        Eigen::Matrix<T, Dynamic, Dynamic> ret0= Eigen::Matrix<T, Dynamic, Dynamic>::Zero(number_elements, msh.cells.size());
        Eigen::Matrix<T, Dynamic, Dynamic> ret1= Eigen::Matrix<T, Dynamic, Dynamic>::Zero(number_elements, msh.cells.size());
        // VECTOR NOTATION
        //values_bis1= Eigen::Matrix<T, Dynamic, 1>::Zero(number_elements*msh.cells.size(), 1 );
    
        //std::cout<<"Number of cells "<<msh.cells.size()<<std::endl;
        
        // MATRIX NOTATION
        // std::cout<<"Dimension of the basis "<<values_bis.size()<<std::endl;
        // VECTOR NOTATION
        // std::cout<<"Dimension of the basis "<<values_bis1.size()<<std::endl;
        size_t i_global = 0 , i_local=0 , i_vertex=0;
        for(auto& cl:msh.cells)
        {
            auto qps = equidistriduted_nodes<T,Mesh>(msh, cl, degree_FEM);
            //auto qps = equidistriduted_nodes_ordered_bis<T,Mesh>(msh, cl, degree_FEM);

            i_local = 0;
            for ( const auto & qp : qps)
            {
                ret0(i_local,i_global) = (u(qp))(0) ;  // MATRIX NOTATION
                ret1(i_local,i_global) = (u(qp))(1) ;  // MATRIX NOTATION
                    
                //values_bis1(i_local+i_global) = level_set(qp) ; // VECTOR NOTATION
                i_vertex = i_global+floor(i_global/Nx);
                if( i_local==0 ){
                    vertices.first(i_vertex) = (u(qp))(0) ;
                    vertices.second(i_vertex) = (u(qp))(1) ;
                }
            
                if( i_local==1 ){
                    vertices.first(i_vertex+1) = (u(qp))(0) ;
                    vertices.second(i_vertex+1) = (u(qp))(1) ;
                }
            
                if( i_local==(degree_FEM+2) ){
                    vertices.first(i_vertex+Nx+2) = (u(qp))(0) ;
                    vertices.second(i_vertex+Nx+2) = (u(qp))(1) ;
                }
                
                if( i_local==(degree_FEM+1) ){
                    vertices.first(i_vertex+Nx+1) = (u(qp))(0) ;
                    vertices.second(i_vertex+Nx+1) = (u(qp))(1) ;
                }
                
                i_local++;
            }
            i_global++;  // MATRIX NOTATION
            //  i_global+=number_elements;       // VECTOR NOTATION
        }
        values_bis=std::make_pair(ret0,ret1);
    //#endif
    }
    
    template<typename VERTEX>
    void converting_into_HHO_formulation( const VERTEX& vertices )
    {
        assert(degree_FEM == 1);
        for (size_t j = 0; j < values_bis.first.cols() ; j++)
        {

            for (size_t i = 0; i < values_bis.first.rows() ; i++)
            {
                auto i_vertex = j + floor(j/Nx);
                if ( i == 0 ){
                    values_bis.first(i,j) = vertices.first(i_vertex);
                    values_bis.second(i,j) = vertices.second(i_vertex);
                }
                
                if( i ==1 ){
                    values_bis.first(i,j) = vertices.first(i_vertex+1) ;
                    values_bis.second(i,j) = vertices.second(i_vertex+1);
                }
                
                if( i ==(degree_FEM+2) ){
                    values_bis.first(i,j) = vertices.first(i_vertex+Nx+2) ;
                    values_bis.second(i,j) = vertices.second(i_vertex+Nx+2);
                }
                
                if( i ==(degree_FEM+1) ){
                    values_bis.first(i,j) = vertices.first(i_vertex+Nx+1) ;
                    values_bis.second(i,j) = vertices.second(i_vertex+Nx+1);
                }
            }
        }
                 
    }
    
    template<typename MATRIX>
    void converting_into_FE_formulation( const MATRIX& values )
    {
        assert(degree_FEM == 1);
        for (size_t j = 0; j < values_bis.first.cols() ; j++)
        {

            for (size_t i = 0; i < values_bis.first.rows() ; i++)
            {
                auto i_vertex = j + floor(j/Nx);
                if ( i == 0 ){
                    vertices.first(i_vertex) = values.first(i,j) ;
                    vertices.second(i_vertex) = values.second(i,j) ;
                }
                
                if( i ==1 ){
                    vertices.first(i_vertex+1) = values.first(i,j) ;
                    vertices.second(i_vertex+1) = values.second(i,j) ;
                }
                
                if( i ==(degree_FEM+2) ){
                    vertices.first(i_vertex+Nx+2) = values.first(i,j) ;
                    vertices.second(i_vertex+Nx+2) = values.second(i,j) ;
                }
                
                if( i ==(degree_FEM+1) ){
                    vertices.first(i_vertex+Nx+1) = values.first(i,j) ;
                    vertices.second(i_vertex+Nx+1) = values.second(i,j) ;
                }
            }
        }
                 
    }
    
    std::pair<T,T> operator()( const point<T,2>& pt, const Mesh & msh,  const typename Mesh::cell_type& cl) const
    {
        size_t counter = offset(msh,cl) ;
        cell_basis_Lagrangian<Mesh,T> cb(msh, cl, degree_FEM);
        auto values_cell_first = (values_bis.first.block(0,counter,number_elements,1)).col(0);
        auto values_cell_second = (values_bis.second.block(0,counter,number_elements,1)).col(0);
        T tmp1 = values_cell_first.dot( cb.eval_basis(pt) );
        T tmp2 = values_cell_second.dot( cb.eval_basis(pt) );
        return std::make_pair(tmp1,tmp2);
    }
    
    
       
};


template < typename VECTOR, typename Mesh, typename MATRIX , typename T = typename Mesh::coordinate_type >
Matrix<T, Dynamic, 1>  make_dij_vector(const Mesh & msh, const MATRIX& dij, const VECTOR& phi_old )
{
    Matrix<T, Dynamic, 1> ret = Eigen::Matrix<T, Dynamic, 1>::Zero(dij.rows(), 1);
    
    
    for(size_t i = 0 ; i<dij.rows() ; i++ )
    {
        auto tmp = (dij.row(i)).adjoint();
        ret(i) = tmp.dot( phi_old ) - tmp(i)*( phi_old(i) );
    }
    return ret;
}

template < typename VECTOR, typename T >
Matrix<T, Dynamic, 1> solveFEM( const VECTOR& lumped_mass , const VECTOR& conv , const VECTOR& last_term, const VECTOR& phi_old , T dt , const std::vector<size_t>& bdry_nodes)
{
    /*
    Matrix<T, Dynamic, 1> sol =  Eigen::Matrix<T, Dynamic, 1>::Zero(lumped_mass.rows(), 1);
    size_t j = 0;
    for (size_t i = 0 ; i < phi_old.rows() ; i++ ) {
        if( i == bdry_nodes.at(j) ){
            sol(i) = phi_old(i);
            j++; // I can do since bdry_nodes is ordered
        }
        else
           sol(i) = phi_old(i) - dt * conv(i)/( lumped_mass(i) ) + dt * last_term(i)/( lumped_mass(i) );
    }
    return sol;
    */
    // if I dont consider boundary condition here, but just solve for all
    
    
    return phi_old - dt * conv.cwiseQuotient(lumped_mass) + dt * last_term.cwiseQuotient(lumped_mass);
    
}






template < typename Fonction, typename VECTOR, typename MATRIX , typename T >
Matrix<T, Dynamic, 1> solveFEM_Entropic( const MATRIX& mass,const VECTOR& conv, const VECTOR& last_term, const Fonction& phi_old , T dt )
{
    // if I dont consider boundary condition here, but just solve for all-> look to solveFEM
    auto b = phi_old - dt * conv + dt * last_term;
    
    return mass.completeOrthogonalDecomposition().solve(b);
    
    
}

template < typename Fonction, typename VECTOR, typename MATRIX , typename T >
Matrix<T, Dynamic, 1> solveFEM_Entropic_FAST( const MATRIX& llt,const VECTOR& conv, const VECTOR& last_term, const Fonction& phi_old , T dt )
{
    // if I dont consider boundary condition here, but just solve for all-> look to solveFEM
    auto b = phi_old - dt * conv + dt * last_term;
    
    return llt.solve(b);
    
    
}



/*

std::vector< size_t > find_neighbors(  const typename Mesh::node_type& node )
{
    size_t row = floor( node/(M+1) ) ;
    size_t col = n % (M+1) ;
    
    size_t central_cell = col + row*M ;
    
}
*/

template<typename T >
class Finite_Element
{
    size_t ndof;
    size_t order;
    T hx;
    T hy;
    
public:
    Finite_Element(size_t ndof , size_t order , const mesh_init_params<T>& params ): ndof(ndof) , order(order) , hx(params.hx() ) , hy(params.hy() ) {}
    
    size_t get_order() const { return order; }
    size_t get_n_dof() const { return ndof; }
    size_t get_hx() const { return hx; }
    size_t get_hy() const { return hy; }
    
    
};



template< typename VECTOR_LOC , typename VECTOR_GLOB , typename NODES>
void global_update_vector( const VECTOR_LOC& local , VECTOR_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
        // std::cout<<"i "<<i<<std::endl;
        global( kk ) += local ( i ) ;
        i++;
    }

}


template< typename VECTOR_LOC , typename VECTOR_GLOB , typename NODES>
void global_update_vector_NOSUM( const VECTOR_LOC& local , VECTOR_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
        // std::cout<<"i "<<i<<std::endl;
        global( kk ) = local ( i ) ;
        i++;
    }

}


template< typename VECTOR_LOC , typename VECTOR_GLOB , typename NODES>
void global_update_vector_MAX( const VECTOR_LOC& local , VECTOR_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
        // std::cout<<"i "<<i<<std::endl;
        auto max_glob = global( kk );
        global( kk ) = std::max( local ( i ) , max_glob ) ;
        i++;
    }

}

template< typename VECTOR_LOC , typename VECTOR_GLOB , typename NODES>
void global_update_vector_MIN( const VECTOR_LOC& local , VECTOR_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
        // std::cout<<"i "<<i<<std::endl;
        auto min_glob = global( kk );
        global( kk ) = std::min( local ( i ) , min_glob ) ;
        i++;
    }

}




template< typename MATRIX_LOC , typename MATRIX_GLOB , typename NODES>
void global_update_NOSUM( const MATRIX_LOC& local , MATRIX_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t j = 0;
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
       // std::cout<<"i "<<i<<std::endl;
        
        for(auto & l : nodes_position)
        {
            size_t ll = l.ptid;
            
            //std::cout<<"colonna "<<ll<<std::endl;
            //std::cout<<"j "<<j<<std::endl;
            global( kk  , ll  ) = local ( i , j ) ;
            j++;
        }
        i++;
        
    }
    
}



template< typename MATRIX_LOC , typename MATRIX_GLOB , typename NODES>
void global_update( const MATRIX_LOC& local , MATRIX_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
   // std::cout<<"numero nodi "<<nodes_position.size()<<std::endl;
    for(auto & k : nodes_position)
    {
        size_t j = 0;
        size_t kk = k.ptid;
        //std::cout<<"riga "<<kk<<std::endl;
       // std::cout<<"i "<<i<<std::endl;
        
        for(auto & l : nodes_position)
        {
            size_t ll = l.ptid;
            
            //std::cout<<"colonna "<<ll<<std::endl;
            //std::cout<<"j "<<j<<std::endl;
            global( kk  , ll  ) += local ( i , j ) ;
            j++;
        }
        i++;
        
    }
    
}



/*

// OLD IMPLEMENTATION
template< typename MATRIX_LOC , typename MATRIX_GLOB , typename NODES>
void global_update2( const MATRIX_LOC& local , MATRIX_GLOB& global , const NODES& nodes_position )
{
    size_t i = 0 ;
    
   std::cout<<"LOCAL "<<std::endl;
   std::cout<<local<<std::endl;
    
    
    for(auto & k : nodes_position)
    {
        size_t j = 0;
        //size_t kk = k.ptid;
        std::cout<<"riga "<<k<<std::endl;
        std::cout<<"i "<<i<<std::endl;
        
        for(auto & l : nodes_position)
        {
            //size_t ll = l.ptid;
            
            std::cout<<"colonna "<<l<<std::endl;
            std::cout<<"j "<<j<<std::endl;
            
            global( k  , l  ) += local ( i , j ) ;
            j++;
        }
        i++;
        
    }
    std::cout<<"GLOBAL "<<std::endl;
    std::cout<<global<<std::endl;
    
}
*/



std::vector<size_t> boundary_nodes_function( size_t Nx , size_t Ny )
{
    // IT WORKS ONLY FOR Q1!!!
    std::vector<size_t> bdry_nodes ;
    for (size_t j = 1 ; j<=Ny ; j++) {
        bdry_nodes.push_back( j*(Nx+1) -1 ) ;
        bdry_nodes.push_back( j*(Nx+1) ) ;
    }
    for( size_t j = 1 ; j< Nx ; j++){
        bdry_nodes.push_back( j ) ;
        bdry_nodes.push_back( Ny*(Nx+1)+j ) ;
    }
    bdry_nodes.push_back( 0 ) ;
    bdry_nodes.push_back( (Ny+1)*(Nx+1)-1 ) ;
    std::sort(bdry_nodes.begin(), bdry_nodes.end() );
    
    return bdry_nodes;
}


template<typename Node , typename SUPP_NODES>
void supporting_nodes( SUPP_NODES& ret ,  const Node& nodes_position )
{
    
    for( auto & i : nodes_position)
    {
        for( auto & j : nodes_position)
        {
            (ret.at(i.ptid)).insert(j.ptid);
         //    std::cout<<"ret("<<i.ptid<<") is "<<j<<std::endl;
        }
         //std::sort(ret(i.ptid).begin(), ret(i.ptid).end() ); // set already increasing ordered
    }
   
}


template<typename Node , typename MATRIX >
void division_Si( const Node& S_i , const MATRIX& mat1 , const MATRIX& mat2 , MATRIX& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(counter,elem) = mat1(counter,elem)/mat2(counter,elem);
        }
        counter++;
    }
}

template<typename Node , typename MATRIX >
void division_Si_T( const Node& S_i , const MATRIX& mat1 , const MATRIX& mat2 , MATRIX& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(elem,counter) = mat1(elem,counter)/mat2(elem,counter);
        }
        counter++;
    }
}

template<typename Node , typename MATRIX >
void moltiplication_Si_T( const Node& S_i , const MATRIX& mat1 , const MATRIX& mat2 , MATRIX& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(elem,counter) += ( mat1(elem,counter) *mat2(elem,counter) );
        }
        counter++;
    }
}


template<typename Node , typename MATRIX >
void moltiplication_Si( const Node& S_i , const MATRIX& mat1 , const MATRIX& mat2 , MATRIX& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(counter,elem) += ( mat1(counter,elem)*mat2(counter,elem) );
        }
        counter++;
    }
}

template<typename Node , typename MATRIX , typename VECTOR>
void sum_Si( const Node& S_i , const MATRIX& mat , const VECTOR& vec , VECTOR& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(counter) += mat(counter,elem)*vec(elem);
        }
        counter++;
    }
}

template<typename Node , typename MATRIX , typename VECTOR>
void averaged_sum_Si( const Node& S_i , const MATRIX& mat , const VECTOR& vec , VECTOR& sol )
{
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(counter) += mat(counter,elem)*(vec(elem)-vec(counter));
        }
        counter++;
    }
}


/*
template < typename VECTOR, typename MATRIX , typename T , typename POS >
void checking_phi_l( const VECTOR& lumped_mass, const MATRIX& cij_x , const MATRIX& cij_y , const MATRIX& d_ij, const VECTOR& phi_old , const VECTOR& phi_l , T dt , const POS& S_i )
{
 
    
    VECTOR K_term =  Eigen::Matrix<T, Dynamic, 1>::Zero(lumped_mass.rows(), 1);
    VECTOR Dl_term =  Eigen::Matrix<T, Dynamic, 1>::Zero(lumped_mass.rows(), 1);
    // SBAGLIATO NON è PHI_OLD MA IL FLUSSO CHE SERVE QUI DENTRO!
    averaged_sum_Si( S_i , cij_x , phi_old , K_term );
    averaged_sum_Si( S_i , cij_y , phi_old , K_term );
    sum_Si( S_i , d_ij , phi_old , Dl_term );
    //std::cout<<'\n'<<"check Dij "<<'\n'<<Dl_term-d_ij*phi_old<<std::endl;
    VECTOR tmp =  lumped_mass.cwiseProduct( (phi_l - phi_old)/dt ) + K_term - Dl_term;
    std::cout<<'\n'<<"check phi_l vec is "<<'\n'<<tmp<<std::endl;
}
*/
template < typename VECTOR , typename T  >
void checking_phi_lBIS( const VECTOR& lumped_mass, const VECTOR& cij_term , const VECTOR& dij_term, const VECTOR& phi_old , const VECTOR& phi_l , T dt )
{
 
    VECTOR tmp =  lumped_mass.cwiseProduct( (phi_l - phi_old)/dt ) + cij_term - dij_term;
    std::cout<<'\n'<<"CHECK BIS phi_l vec is "<<'\n'<<tmp<<std::endl;
}


/*
template < typename VECTOR , typename MATRIX , typename T , typename POS >
void checking_phi_h( const MATRIX& mass, const MATRIX& cij_x , const MATRIX& cij_y , const MATRIX& dC_ij , const VECTOR& phi_old , const VECTOR& phi_h , T dt , const POS& S_i)
{
 
    VECTOR K_term =  Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    VECTOR Dc_term =  Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    // SBAGLIATO NON è PHI_OLD MA IL FLUSSO CHE SERVE QUI DENTRO!
    averaged_sum_Si( S_i , cij_x , phi_old , K_term );
    averaged_sum_Si( S_i , cij_y , phi_old , K_term );
    sum_Si( S_i , dC_ij , phi_old , Dc_term );
    VECTOR tmp =  mass*( (phi_h - phi_old)/dt ) + K_term  - dC_ij*phi_old;
   // std::cout<<'\n'<<"check DCij "<<'\n'<<Dc_term-dC_ij*phi_old<<std::endl;
    std::cout<<'\n'<<"check phi_h vec is "<<'\n'<<tmp<<std::endl;
}
*/
template < typename VECTOR , typename T , typename MATRIX  >
void checking_phi_hBIS( const MATRIX& mass, const VECTOR& cij_term , const VECTOR& dij_term, const VECTOR& phi_old , const VECTOR& phi_h , T dt )
{
 
    VECTOR tmp =  mass*( (phi_h - phi_old)/dt ) + cij_term - dij_term;
    std::cout<<'\n'<<"CHECK BIS phi_H vec is "<<'\n'<<tmp<<std::endl;
}




template < typename Entropy , typename Fonction, typename Fonction_TILDE , typename Mesh, typename Vel_Field , typename T , typename POSITION , typename VECTOR >
  void r_i_calculator(const Mesh& msh, const typename Mesh::cell_type& cl , const Entropy& E , const Fonction_TILDE& phi_tilde , const Fonction& phi , T dt , const Vel_Field& u , const POSITION& S_i , VECTOR& Emax_global , VECTOR& Emin_global , VECTOR& R_i  )
{
    size_t di = 0;
    cell_basis_Lagrangian<Mesh,T> cb(msh, cl, phi.degree_FEM);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
    auto qps = integrate(msh, cl, 2*(phi.degree_FEM+di));
    
    
    auto nds = nodes(msh,cl);
    iter_swap(nds.end()-1 , nds.end()-2 );
    
    auto pts = points(msh,cl);
    iter_swap(pts.end()-1 , pts.end()-2 );
   
    
    
    size_t counter = 0;
    /// FARE FUNZIONE max_min_entropy PER RENDERE IL CODICE PULITO
    Eigen::Matrix<T, Dynamic, 1> Emax =  Eigen::Matrix<T, Dynamic, 1>::Ones(pts.size(), 1);
    Eigen::Matrix<T, Dynamic, 1> Emin =  Eigen::Matrix<T, Dynamic, 1>::Ones(pts.size(), 1);
    for(auto& nd : nds )
    {
        auto i = nd.ptid ;
        T max_loc = -1e20;
        T min_loc = 1e20;
        for(auto& pt_j : pts )
        {
            Emax(counter) = std::max ( E(pt_j , cl) , max_loc );
            Emin(counter) = std::min ( E(pt_j , cl) , min_loc );
            max_loc = std::max ( Emax(counter) , max_loc );
            min_loc = std::min ( Emin(counter) , min_loc );
        }
        
        counter++;
    }
    global_update_vector_MAX( Emax , Emax_global , nds );
    global_update_vector_MIN( Emin , Emin_global , nds );
    // std::cout<<"Emax_global: "<<'\n'<<Emax_global<<std::endl;
    //  std::cout<<"Emin_global: "<<'\n'<<Emin_global<<std::endl;
    
    for (auto& qp : qps)
    {
        auto bi = cb.eval_basis(qp.first);
        auto phi_grad0 = phi.gradient(qp.first,msh,cl)(0);
        auto phi_grad1 = phi.gradient(qp.first,msh,cl)(1);
        
        auto f = ( ( ( phi_tilde(qp.first,msh,cl)-phi(qp.first,msh,cl) )/dt + u(qp.first,msh,cl).first*phi_grad0 + u(qp.first,msh,cl).second*phi_grad1 ) *
                  E.derivative(qp.first,cl) ); //.cwiseQuotient( Emax - Emin ) ;
        ret += qp.second * bi * f;
    }
    global_update_vector( ret , R_i , nds );
    
}



template < typename T , typename MATRIX , typename VECTOR ,typename POSITION >
VECTOR alfaf_ij_creator( const VECTOR& lumped_mass , const MATRIX& mass , const VECTOR& delta_phi , const VECTOR& phi_L , T dt , const MATRIX& D_ij , const MATRIX& Dc_ij , const VECTOR& phi_old , const POSITION& S_i)
{
    
    MATRIX new_mass = - mass;
    for(size_t i = 0 ; i < new_mass.rows() ; i++){
        new_mass(i,i) += lumped_mass(i);
    }
    
    MATRIX new_D = (Dc_ij - D_ij); //(D_ij - Dc_ij);
    
    MATRIX f_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( mass.rows(), mass.cols() );
    MATRIX alpha_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( mass.rows(), mass.cols() );
    
    VECTOR ret = Eigen::Matrix<T, Dynamic, 1>::Zero( mass.rows(), 1 );
    
    VECTOR P_plus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    VECTOR P_minus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    VECTOR Q_plus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    VECTOR Q_minus = Eigen::Matrix<T, Dynamic, 1>::Zero(mass.rows(), 1);
    
    VECTOR R_plus = Eigen::Matrix<T, Dynamic, 1>::Ones(mass.rows(), 1);
    VECTOR R_minus = Eigen::Matrix<T, Dynamic, 1>::Ones(mass.rows(), 1);
    VECTOR phi_max = phi_old;
    VECTOR phi_min = phi_old;
    
               
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            phi_max(counter) = std::max(phi_old(elem),phi_max(counter));
            phi_min(counter) = std::min(phi_old(elem),phi_min(counter));
        }
        counter++;
    }
    
    for(size_t i = 0; i< f_ij.rows() ; i++)
    {
        for(size_t j = 0; j< f_ij.cols() ; j++)
        {
            f_ij(i, j) = new_mass(i,j)*(delta_phi(j)-delta_phi(i)) +dt*new_D(i,j)*(phi_old(j)-phi_old(i)) ;
            T f_tmp = f_ij(i, j);
            P_plus(i) += std::max( 0. , f_tmp );
            P_minus(i) += std::min( 0. , f_tmp );
        }
        Q_plus(i) = lumped_mass(i)*(phi_max(i)-phi_L(i));
        Q_minus(i) = lumped_mass(i)*(phi_min(i)-phi_L(i));
        if( std::abs(P_plus(i)) > 1e-20 ){
            T Q_P_plus = Q_plus(i)/P_plus(i);
            R_plus(i) = std::min( 1. , Q_P_plus );
        }
        if( std::abs(P_minus(i)) > 1e-20 ){
            T Q_P_minus = Q_minus(i)/P_minus(i);
            R_minus(i) = std::min( 1. , Q_P_minus );
        }
    }
    
    for(size_t i = 0; i< f_ij.rows() ; i++)
    {
        for(size_t j = 0; j< f_ij.cols() ; j++)
        {
            if( f_ij(i, j) > 0 )
                alpha_ij(i,j) = std::min( R_plus(i) , R_minus(j) );
            else
                alpha_ij(i,j) = std::min( R_plus(j) , R_minus(i) );
        }
       
    }
    
    size_t counter2 = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            ret(counter2) += ( alpha_ij(counter2,elem)*f_ij(counter2,elem) );
        }
        counter2++;
    }

    
    // CHECKING F and Alpha properties
    /*
    for(size_t i = 0; i< f_ij.rows() ; i++)
    {
        for(size_t j = 0; j< f_ij.cols() ; j++)
        {
            std::cout<<f_ij(i,j)+f_ij(j,i)<<" , ";
        }
       std::cout<<'\n'<<std::endl;
    }
    */
    
    /*
    std::cout<<"alpha_ij checking symmetry"<<'\n'<<std::endl;
    for(size_t i = 0; i< f_ij.rows() ; i++)
    {
        for(size_t j = 0; j< f_ij.cols() ; j++)
        {
            std::cout<<alpha_ij(i,j)-alpha_ij(j,i)<<" , ";
        }
       std::cout<<'\n'<<std::endl;
    }
    */
    
    /*
    size_t counter3 = 0;
    
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<f_ij(counter3,elem)+f_ij(elem,counter3)<<" , ";
        }
        counter3++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
     
    counter3 = 0;
     std::cout<<"alpha_ij checking symmetry: METODO 2"<<'\n'<<std::endl;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<alpha_ij(counter3,elem)-alpha_ij(elem,counter3)<<" , ";
        }
        counter3++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    */
    //  CHECKING F_IJ
    VECTOR ret0 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    VECTOR ret1 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    averaged_sum_Si( S_i , new_mass , delta_phi , ret0 );
    averaged_sum_Si( S_i , new_D , phi_old , ret1 );
    VECTOR f_i = (ret0 + dt*ret1);
    
    VECTOR f_i_NEW = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    VECTOR One_Vec = Eigen::Matrix<T, Dynamic, 1>::Ones(new_mass.rows(), 1);
    sum_Si( S_i , f_ij , One_Vec , f_i_NEW );
    
    //std::cout<<'\n'<<"Diff f_i 2 meths is:"<<'\n'<< f_i_NEW - f_i<<std::endl;
    
    return  ret ;
    
}




template < typename T , typename MATRIX , typename VECTOR ,typename POSITION >
VECTOR f_ij_creator( const VECTOR& lumped_mass , const MATRIX& mass , const VECTOR& delta_phi , T dt , const MATRIX& D_ij , const MATRIX& Dc_ij , const VECTOR& phi_old , const POSITION& S_i)
{
    
    MATRIX new_mass = - mass;
    MATRIX new_D = (Dc_ij - D_ij); //(D_ij - Dc_ij);
    for(size_t i = 0 ; i < new_mass.rows() ; i++){
        new_mass(i,i) += lumped_mass(i);
    }
    
    /*
    std::cout<<"new_mass sum check: "<<std::endl;
    for(size_t i = 0 ; i < new_mass.rows() ; i++){
        new_mass(i,i) += lumped_mass(i);
        std::cout<<(new_mass.row(i)).sum()<<std::endl;
    }
    std::cout<<"new_D check: "<<std::endl;
    for(size_t i = 0 ; i < new_D.rows() ; i++){
       std::cout<<(new_D.row(i).sum() )<<std::endl;
    }
    */
    
    VECTOR ret0 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    VECTOR ret1 = Eigen::Matrix<T, Dynamic, 1>::Zero(new_mass.rows(), 1);
    averaged_sum_Si( S_i , new_mass , delta_phi , ret0 );
    averaged_sum_Si( S_i , new_D , phi_old , ret1 );
    /*
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            sol(counter) += mat(counter,elem)*(vec(elem)-vec(counter));
        }
        counter++;
    }
    */
    //  std::cout<<'\n'<<"dt: "<<'\n'<<dt<<std::endl;
   
    //    std::cout<<'\n'<<"ret0: "<<'\n'<<ret0<<std::endl;
    //    std::cout<<'\n'<<"ret1: "<<'\n'<<dt*ret1<<std::endl;
  
    return (ret0 + dt*ret1) ;
    
}
template<typename FUNCTION, typename T>
void mapping_phi(FUNCTION& phi , T phi_max , T phi_min)
{
    Matrix<T, Dynamic, 1> Vec_One = Eigen::Matrix<T, Dynamic, 1>::Ones( phi.vertices.rows(), 1 );
    Matrix<T, Dynamic, Dynamic> Mat_One = Eigen::Matrix<T, Dynamic, Dynamic>::Ones( phi.values_bis.rows(), phi.values_bis.cols() );
    // mapping between 0 and 1
    phi.vertices = (phi.vertices-phi_min*Vec_One)/( phi_max - phi_min );
    phi.values_bis = (phi.values_bis-phi_min*Mat_One)/( phi_max - phi_min );
}

template<typename FUNCTION, typename T>
void inverse_mapping_phi(FUNCTION& phi, T phi_max , T phi_min)
{
    Matrix<T, Dynamic, 1> Vec_One = Eigen::Matrix<T, Dynamic, 1>::Ones( phi.vertices.rows() , 1 );
    Matrix<T, Dynamic, Dynamic> Mat_One = Eigen::Matrix<T, Dynamic, Dynamic>::Ones( phi.values_bis.rows(), phi.values_bis.cols() );
    phi.vertices = phi_min*Vec_One + phi.vertices*( phi_max - phi_min );
    phi.values_bis = phi_min*Mat_One + phi.values_bis*( phi_max - phi_min );
}

template < typename Fonction, typename Mesh, typename Vel_Field , typename T = typename Mesh::coordinate_type >
void run_FEM_levelset(const Mesh & msh, size_t degree, Fonction & phi, const Vel_Field & u , const T& dt , const mesh_init_params<T>& params)
{
    timecounter tc;
    tc.tic();
    size_t ndof = ( degree+1 ) * ( degree+1 ) ; // degrees of freedom
    const T eps = 1e-14; //constant into entropy
 
    // LOCAL MASS MATRIX ==> CHECK DEGREE == 1!!!
    assert(degree==1);
    
    
    mapping_phi(phi , phi.phi_max , phi.phi_min ); // mapping of phi to  have a phi between 0 and 1
    /// PHI MAPPED INTO 0-1 PLOT
    /*
    postprocess_output<double> postoutput4;
    auto test_phi_mapped = std::make_shared< gnuplot_output_object<double> >("phi_mapped.dat");
    for (auto cl:msh.cells) {
        auto pts = points(msh,cl) ;
        for (auto pt:pts) {
            T value = phi(pt,msh,cl);
            test_phi_mapped->add_data(pt,value);
        }
    }
    postoutput4.add_object(test_phi_mapped);
    postoutput4.write();
    */
    
    // ENTROPY -> for phi with codomain in (0,1)
    non_linear_entropy<T,Fonction,Mesh> E(eps , phi ,msh );
    auto phi_tilde = projection<T,Mesh> ( msh, degree , params );
    
    /// ENTROPY PLOT
    /*
    postprocess_output<double> postoutput6;
    auto cut_entropy = std::make_shared< gnuplot_output_object<double> >("entropy.dat");
    for (auto cl:msh.cells) {
        auto pts = points(msh,cl) ;
        for (auto pt:pts) {
            T value = E(pt,cl);
            cut_entropy->add_data(pt,value);
        }
    }
    postoutput6.add_object(cut_entropy);
    postoutput6.write();
    */

    // Global Matrix and Global Vector Definitions:
    size_t dim = msh.nodes.size();
    // Lumped mass vector
    Matrix<T, Dynamic, 1> global_lumped_mass = Eigen::Matrix<T, Dynamic, 1>::Zero( dim, 1 );
    // Mass matrix
    Matrix<T, Dynamic, Dynamic> global_mass = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim );
    // C_ij,1 matrix
    Matrix<T, Dynamic, Dynamic> global_cij_x = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim );
    // C_ij,2 matrix
    Matrix<T, Dynamic, Dynamic> global_cij_y = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim );
    // Convection term (vector)
    Matrix<T, Dynamic, 1> conv_global = Eigen::Matrix<T, Dynamic, 1>::Zero( dim, 1 );
    // D_ij term (vector)
    Matrix<T, Dynamic, 1> term_dij = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1 );
    // N_ij matrix
    std::pair<Matrix<T, Dynamic, Dynamic>,Matrix<T, Dynamic, Dynamic>> nij = std::make_pair(Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) , Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) ) ;
    // N_ji matrix
    std::pair<Matrix<T, Dynamic, Dynamic>,Matrix<T, Dynamic, Dynamic>> nij_transpose = std::make_pair(Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) , Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) ) ;
    // ||c_ij||^2
    Matrix<T, Dynamic, Dynamic> cij_square = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    // ||c_ji||^2
    Matrix<T, Dynamic, Dynamic> cji_square = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    
    // One vector
    Matrix<T, Dynamic, 1> Vec_One = Eigen::Matrix<T, Dynamic, 1>::Ones( dim, 1 );
    // One matrix
    Matrix<T, Dynamic, Dynamic> Mat_One = Eigen::Matrix<T, Dynamic, Dynamic>::Ones( dim, dim );
    
    
    // Boundary Nodes = Vertices (in Q1)
    auto bdry_nodes = boundary_nodes_function( phi.Nx , phi.Ny );

    
    // The Support of Global Shape Function: list of neighborood nodes for each node i
    std::vector<std::set<size_t>> S_i(dim);
    timecounter tc1;
    tc1.tic();
    for (auto cl : msh.cells)
    {
        
        
        auto nodes_position = nodes(msh, cl); // node_pos.ptid to have just the number
        
        // In basis (mesh creation) the order is
        /// 2 - - 3
        /// |       |
        /// 0 - - 1
        
        // In nodes position instead the order is
        /// 3 - - 2
        /// |       |
        /// 0 - - 1
        // I swap last two elements to have the first ordering
        iter_swap(nodes_position.end()-1 , nodes_position.end()-2 );
       
        
        
        // Support of global shape function: list of close nodes for each node i
        supporting_nodes( S_i , nodes_position );
        
        /*
        // CALCULATION OF THE SIZE + PLOTTING
        size_t size_supp_nodes = 0;
        std::cout<<"Supporting nodes:"<<std::endl;
        size_t jjjj = 0;
        for (auto& i:S_i) {
            size_supp_nodes+=i.size();
            std::cout <<"Node "<<jjjj<<":";
            for (auto it=i.begin(); it != i.end(); ++it)
                std::cout << ' ' << *it;
               // std::cout<<ii;
            std::cout<<'\n';
            jjjj++;
        }
        std::cout<<std::endl;
        std::cout<<"Supporting nodes size:"<<size_supp_nodes<<std::endl;
        */
        
        
        // Local mass matrix
        Matrix<T, Dynamic, Dynamic> mass = make_lagrange_local_mass_matrix(msh, cl, degree);
        
        // Assembling into global matrix
        //global_update2( mass , global_mass , nodes_position2 );
        global_update( mass , global_mass , nodes_position );
        
        // Local c_ij = b_i nabla(b_j)
        auto cij = make_local_cij_matrix(msh, cl, degree);
        
        // Assembling into global matrix
        global_update( cij.first , global_cij_x , nodes_position );
        global_update( cij.second , global_cij_y , nodes_position );
        
    // USEFUL COMMAND FOR NORMS of a Eigen::matrix A:
    // A.template lpNorm<2>()
    // A.norm()
        
    }
    
    // LUMPED MASS VECTOR (i.e. vector = diagonal of a matrix)
    sum_Si(S_i, global_mass , Vec_One , global_lumped_mass );
    
    
    
    // CHECKING OF MASS MATRICES' PROPERTIES-> OK, perfect precision 1e-19
    /*
    std::cout<<"First checking: Lumped Mass"<<std::endl;
    for(size_t i = 0; i<global_lumped_mass.rows();i++)
        if(global_lumped_mass(i)<0)
            std::cout<<"PROBLEM into lumped mass"<<std::endl;
    
    Matrix<T, Dynamic, Dynamic> mass_check = - global_mass;
    for(size_t i = 0 ; i < mass_check.rows() ; i++){
        mass_check(i,i) += global_lumped_mass(i);
        std::cout<<(mass_check.row(i)).sum()<<std::endl;
    }
    */
    
    
    
    /// CHECK OF THE SUM PROPERTY OF CIJ
    /*
     Matrix<T, Dynamic, 1> sum0 = Eigen::Matrix<T, Dynamic, 1>::Zero(global_cij_x.rows(), 1);
     Matrix<T, Dynamic, 1> sum1 = Eigen::Matrix<T, Dynamic, 1>::Zero(global_cij_x.rows(), 1);
    
    for (size_t k = 0; k<global_cij_x.cols(); k++)
    {
        sum0 += (global_cij_x).col(k);
        sum1 += (global_cij_y).col(k);
    }
    
    for (size_t i=0; i<global_cij_y.rows(); i++) {
         std::cout<<"The sum is "<<sum1(i)<<" and "<<sum0(i)<<std::endl;
    }
    
    size_t counter_sum = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
           sum0( counter_sum ) += global_cij_x( counter_sum , elem );
           sum1( counter_sum ) += global_cij_y( counter_sum , elem );
           if(elem==counter_sum)
               std::cout<<"In ("<<counter_sum<<" , "<<elem<<" ), c^0 = "<<global_cij_x( counter_sum , elem )<<" and c1 = "<<global_cij_y( counter_sum , elem )<<std::endl;
           else
                std::cout<<"In ("<<counter_sum<<" , "<<elem<<" ), c^0_ij + c^0_ji = "<<global_cij_x( counter_sum , elem ) + global_cij_x( elem , counter_sum )<<" and c^1_ij + c^1_ji = "<<global_cij_y( counter_sum , elem ) + global_cij_y( elem , counter_sum ) <<std::endl;
        }
        counter_sum++;
    }
    //std::cout<<"The sum0 is "<<'\n'<<sum0<<" and sum1 "<<'\n'<<sum1<<std::endl;
    */
     
   

    // FLUX TERM
    Matrix<T, Dynamic, 1> flux0 = (u.vertices.first).cwiseProduct(phi.vertices) ;
    Matrix<T, Dynamic, 1> flux1 = (u.vertices.second).cwiseProduct(phi.vertices) ;
    
    
    
    // RESOLUTION OF phi_tilde GLOBALE     ( with cij(phi_j - phi_i) )
    
    Matrix<T, Dynamic, 1> vec1 = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    sum_Si(S_i , global_mass , phi.vertices , vec1);
    
    Matrix<T, Dynamic, 1> vec2 = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    //Matrix<T, Dynamic, 1> vec2_bis = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    averaged_sum_Si(S_i , global_cij_x , flux0 , vec2);
    averaged_sum_Si(S_i , global_cij_y , flux1 , vec2);
    //sum_Si(S_i , global_cij_x , flux0 , vec2_bis);
    //sum_Si(S_i , global_cij_y , flux1 , vec2_bis);
    //std::cout<<"diff vec 2 = "<<'\n'<<vec2-vec2_bis<<std::endl;
    /// THERE IS A DIFFERENCE OF  <1E-18 DEPENDING IF I CONSIDER OR NOT THE AVERAGED SUM
    
    Matrix<T, Dynamic, 1> b = vec1 - dt*vec2.cwiseQuotient(global_lumped_mass);
    
    /// OLD IMPLEMENTATION
    //Matrix<T, Dynamic, 1> prova_FAST = global_mass.completeOrthogonalDecomposition().solve(b);
    //phi_tilde.vertices = global_mass.HouseholderQR().solve(b);
    timecounter tc00;
    tc00.tic();
    LLT<Matrix<T, Dynamic, Dynamic > > llt( global_mass );
    phi_tilde.vertices = llt.solve(b);
    tc00.toc();
    
    
    if (llt.info() != Success)
    {
        std::cout<<"not positive"<<std::endl;
        assert(0);
    }
    //std::cout<<"ERRORE FAST= "<<'\n'<<phi_tilde.vertices - prova_FAST <<std::endl;
   
    phi_tilde.converting_into_HHO_formulation( phi_tilde.vertices );
    std::cout << bold << yellow << "LLT RESOLUTION TIME: " << tc00 << " seconds" << reset << std::endl;
    
    
    //std::cout<<"phi_tilde= "<<'\n'<<phi_tilde.vertices<<std::endl;
    /*
    postprocess_output<double> postoutput3;
    typedef typename Mesh::point_type       point_type;
    point<double,2> node;
    auto test_phi_tilde = std::make_shared< gnuplot_output_object<double> >("phi_tilde.dat");
    for (size_t i = 0; i<phi_tilde.vertices.rows(); i++) {
        node = msh.points.at(i);
        test_phi_tilde->add_data(node,phi_tilde.vertices(i));
    }
    postoutput3.add_object(test_phi_tilde);
    postoutput3.write();
    */
    
    
    // Term R_i^n
    Matrix<T, Dynamic, 1> Emax_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    Matrix<T, Dynamic, 1> Emin_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    Emax_global *= -1e20;
    Emin_global *= 1e20;
    Matrix<T, Dynamic, 1> R_i = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    for( auto& cl: msh.cells )
    {
        r_i_calculator( msh , cl , E , phi_tilde , phi , dt , u ,S_i , Emax_global , Emin_global , R_i);
    }
    //std::cout<<"Emax_global - Emin_global = "<<'\n'<<Emax_global-Emin_global<<std::endl;
    R_i = R_i.cwiseQuotient( Emax_global - Emin_global );
    //std::cout<<"Ri = "<<'\n'<<R_i<<std::endl;
    
    
    
    // CONVOLUTION TERM
    //sum_Si(S_i, global_cij_x , flux0 , conv_global );
    //sum_Si(S_i, global_cij_y , flux1 , conv_global );
    averaged_sum_Si(S_i, global_cij_x , flux0 , conv_global );
    averaged_sum_Si(S_i, global_cij_y , flux1 , conv_global );
    // OLD IMPLEMENTATION
    //auto conv_global2 = global_cij_x*( flux0 ) + global_cij_y*( flux1 ) ;
    
    // Norm of c_ij
    moltiplication_Si( S_i , global_cij_x , global_cij_x , cij_square );
    moltiplication_Si( S_i , global_cij_y , global_cij_y , cij_square );
    Matrix<T, Dynamic, Dynamic> cij_norm = cij_square.cwiseSqrt();
    
    // Matrix n_ij
    division_Si( S_i , global_cij_x , cij_norm , nij.first );
    division_Si( S_i , global_cij_y , cij_norm , nij.second );
    // OLD IMPLEMENTATION
    //auto nij2 = std::make_pair( global_cij_x.cwiseQuotient(cij_norm) , (global_cij_y).cwiseQuotient( cij_norm ) ) ;
    
    // NOTICE:
    /// Normal Velocity calculation IS EQUIVALENT TO f', that is EQUIVALENT to lambda_r , lambda_l.  Moreover I don't need the calculation of the normal flux. Since then it has to be derived, it is enough to calculate f' = u dot n
     
    // Normal velocity
    Matrix<T, Dynamic, Dynamic> normal_vel = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    size_t counter_row = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            auto u_max_0 = std::max( u.vertices.first(counter_row),u.vertices.first(elem) );
            auto u_max_1 = std::max( u.vertices.second(counter_row),u.vertices.second(elem) );
            normal_vel(counter_row,elem) = u_max_0 * nij.first(counter_row,elem) + u_max_1*nij.second(counter_row,elem);
        }
        counter_row++;
    }
     
    
    
   
    // C_ji matrix
    std::pair<Matrix<T, Dynamic, Dynamic>,Matrix<T, Dynamic, Dynamic>> cji_global = std::make_pair(global_cij_x.adjoint() , global_cij_y.adjoint() );
    
    // Norm of c_ji -> i.e. c_ij transposed
    moltiplication_Si( S_i , cji_global.first , cji_global.first , cji_square );
    moltiplication_Si( S_i , cji_global.second , cji_global.second , cji_square );
    // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
    //moltiplication_Si_T( S_i , cji_global.first , cji_global.first , cji_square );
    //moltiplication_Si_T( S_i , cji_global.second , cji_global.second , cji_square );
    
    //C_ji Norm
    Matrix<T, Dynamic, Dynamic> cji_norm = cji_square.cwiseSqrt();
    //auto cji_square2 = (cji_global.first).cwiseProduct(cji_global.first) + (cji_global.second).cwiseProduct(cji_global.second);
    //Matrix<T, Dynamic, Dynamic> cji_norm2 = cji_square2.cwiseSqrt();
    
   
    // Matrix n_ij (TRANSPOSED)
    division_Si( S_i , cji_global.first , cji_norm , nij_transpose.first );
    division_Si( S_i , cji_global.second , cji_norm , nij_transpose.second );
     // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
    //division_Si_T( S_i , cji_global.first , cji_norm , nij_transpose.first );
    //division_Si_T( S_i , cji_global.second , cji_norm , nij_transpose.second );
    //auto  nij_transposex2 = cji_global.first.cwiseQuotient( cji_norm2 );
    //auto  nij_transposey2 = cji_global.second.cwiseQuotient( cji_norm2 );
    
    // Normal velocity (TRANSPOSED)
    Matrix<T, Dynamic, Dynamic> normal_vel_adj = Eigen::Matrix<T,Dynamic,Dynamic>::Zero(dim, dim);
    //Matrix<T, Dynamic, Dynamic> normal_vel_adj2 = Eigen::Matrix<T,Dynamic,Dynamic>::Zero(dim, dim);
    /*
    for(size_t i = 0 ; i< normal_vel_adj2.rows() ; i++)
    {
        for(size_t j = 0 ; j< normal_vel_adj2.cols() ; j++)
        {
            auto u_max_0 = std::max( u.vertices.first(i),u.vertices.first(j) );
            auto u_max_1 = std::max( u.vertices.second(i),u.vertices.second(j) );
            //normal_vel(counter_row,elem) = u_max_0 * nij.first(counter_row,elem) + u_max_1*nij.second(counter_row,elem);
            // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
            normal_vel_adj2(i,j) = u_max_0 * nij_transposex2(i,j) + u_max_1*nij_transposey2(i,j);
        }
    }
    
    */

    counter_row = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            auto u_max_0 = std::max( u.vertices.first(elem),u.vertices.first(counter_row) );
            auto u_max_1 = std::max( u.vertices.second(elem),u.vertices.second(counter_row) );
            //normal_vel_adj(elem,counter_row) = u_max_0 * nij.first(elem,counter_row) + u_max_1*nij.second(elem,counter_row);
            // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
            normal_vel_adj(counter_row , elem) = u_max_0 * nij_transpose.first(counter_row,elem) + u_max_1*nij_transpose.second(counter_row,elem);
        }
        counter_row++;
    }
    
    
    // Matrix Dij calculation
    Matrix<T, Dynamic, Dynamic> lambda_max = normal_vel.cwiseAbs() ;  // o lambda_max=normal_vel ?
    Matrix<T, Dynamic, Dynamic> tmp = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    moltiplication_Si( S_i , lambda_max , cij_norm , tmp );
    
    Matrix<T, Dynamic, Dynamic> lambda_max_adj=normal_vel_adj.cwiseAbs();// o lambda_max=normal_vel?
    Matrix<T, Dynamic, Dynamic> tmp_adj = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    moltiplication_Si( S_i , lambda_max_adj , cji_norm , tmp_adj );
    // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
    //moltiplication_Si_T( S_i , lambda_max_adj , cji_norm , tmp_adj );
    //Matrix<T, Dynamic, Dynamic> lambda_max_adj2=normal_vel_adj2.cwiseAbs();
    //auto tmp_adj2 = lambda_max_adj.cwiseProduct( cji_norm2 ) ;
    
    Matrix<T, Dynamic, Dynamic> dij = tmp.cwiseMax(tmp_adj);
    
    //Matrix<T, Dynamic, Dynamic> dij2 = tmp.cwiseMax(tmp_adj2);
    
    //std::cout<<"Dij: "<<'\n'<<dij<<std::endl;
    
    /*
      /// The  error is < 1e-15, at max
    std::cout<<"D_ij check: "<<std::endl;
    for(size_t i = 0; i<tmp.rows();i++){
        for(size_t j = i+1; j<tmp.rows();j++){
            if(std::abs(dij(i,j)-dij(j,i))>1e-15)
            std::cout<<dij(i,j)-dij(j,i);
        }
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    */
   
    
    tc1.toc();
    std::cout << bold << yellow << "Solving time method original: " << tc1 << " seconds" << reset << std::endl;
    
    timecounter tc2;
    tc2.tic();
       
    

    
    // Minimum between d_ij and R_i
    T c_e = 1.;
    Matrix<T, Dynamic, Dynamic> dE_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    Matrix<T, Dynamic, Dynamic> phi_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    counter_row = 0;
   
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            if(elem!=counter_row)
            {
            auto R_i_j = c_e * std::max( std::abs( R_i(counter_row) ) , std::abs( R_i(elem) ) );
            dE_ij( counter_row , elem ) = std::min( dij( counter_row , elem ), R_i_j );
            //std::cout<<"R_i_j "<<R_i_j<<" and dij"<<dij( counter_row , elem )<<std::endl;
            }
            phi_ij( counter_row , elem ) = 0.5*( phi.vertices(counter_row) + phi.vertices(elem) );
        }
        counter_row++;
    }
    //std::cout<<"dE_ij: "<<'\n'<<dE_ij<<std::endl;
    //std::cout<<"phi_ij"<<'\n'<<phi_ij<<std::endl;
    T c_comp = 1.;
    Matrix<T, Dynamic, Dynamic> dC_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    Matrix<T, Dynamic, Dynamic> tmp0 = phi_ij - phi_ij.cwiseProduct(phi_ij);
    //Matrix<T, Dynamic, Dynamic> tmp0 = (phi_ij-phi.phi_min*Mat_One).cwiseProduct(phi.phi_max*Mat_One-phi_ij);
    //std::cout<<"tmp0: "<<'\n'<<tmp0<<std::endl;
    positive_part( tmp0 );
    
    
   // Matrix<T, Dynamic, Dynamic> tmp_dC0 = positive_part( tmp0 );

   
    Matrix<T, Dynamic, Dynamic> tmp_dC1 =  Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
           // std::cout<<"In ("<<counter<<" , "<<elem<<") tmp_dC0 = "<<tmp_dC0(counter,elem)<<" and phi.ver diff "<<std::abs(phi.vertices(counter) - phi.vertices(elem))<<std::endl;
            if( std::abs(phi.vertices(counter) - phi.vertices(elem))>1e-15 )
            tmp_dC1(counter,elem) = tmp0(counter,elem)/( std::abs(phi.vertices(counter) - phi.vertices(elem)) );
        }
        counter++;
    }
    //std::cout<<'\n'<<"tmp_dC1: "<<'\n'<<tmp_dC1<<std::endl;
    Matrix<T, Dynamic, Dynamic> tmp1 = Mat_One-c_comp*tmp_dC1;
    
    //std::cout<<'\n'<<"tmp1: "<<'\n'<<tmp1<<std::endl;
    positive_part( tmp1  );
    //std::cout<<'\n'<<"tmp1 POSTIVIVE: "<<'\n'<<tmp1<<std::endl;
    
    moltiplication_Si( S_i , dE_ij , tmp1 , dC_ij );
   
    
    for(size_t i = 0 ; i < dC_ij.rows() ; i++)
    {
        dC_ij(i,i) = 0;
        dij(i,i) = 0;
        dC_ij(i,i) = -dC_ij.row(i).sum();
        dij(i,i) = -dij.row(i).sum();
    }
    //dC_ij = dE_ij.cwiseProduct( positive_part( tmp1  ) );
    //   std::cout<<'\n'<<"dC_ij: "<<'\n'<<dC_ij<<std::endl;
    //   std::cout<<'\n'<<"dij: "<<'\n'<<dij<<std::endl;
    
   
    //std::cout<<"dC_ij: "<<'\n'<<dC_ij<<std::endl;
    
    
    /*
    counter_row = 0;
    std::cout<<"DC_ij check symmetry: "<<std::endl;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<dC_ij(counter_row,elem) - dC_ij(elem,counter_row)<<std::endl;
        }
        counter_row++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    
    
    counter_row = 0;
    std::cout<<"D_ij check symmetry "<<std::endl;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<dij(counter_row,elem) - dij(elem,counter_row)<<std::endl;
        }
        counter_row++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    */
    
    
    // Term D_ij(phi_j - phi_i )
    averaged_sum_Si(S_i, dC_ij , phi.vertices , term_dij );
    //std::cout<<"term_dij : "<<'\n'<<term_dij<<std::endl;
    
    
    //averaged_sum_Si(S_i, dE_ij , phi.vertices , term_dij );
    Matrix<T, Dynamic, 1> term_dij_no_entropy = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1 );
    // VERSION WITHOUT ENTROPY CORRECTION
    averaged_sum_Si(S_i, dij , phi.vertices , term_dij_no_entropy );
    
    //std::cout<<"term_dij_no_entropy : "<<'\n'<<term_dij_no_entropy<<std::endl;
    
    // M*phi^n
    Matrix<T, Dynamic, 1> mass_phi_old = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    sum_Si( S_i , global_mass , phi.vertices , mass_phi_old );
    
    
    
  /*
    // **************** OLD IMPLEMENTATION **************** //
    
   for (size_t i = 0; i<global_lumped_mass.rows(); i++) {
          global_lumped_mass(i) = global_mass.row(i).sum() ;
      }
   
   
    Matrix<T, Dynamic, Dynamic> normal_vel2 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim1, dim2);
    Matrix<T, Dynamic, Dynamic> normal_vel_adj2 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim2, dim1);
    
    //normal velocity
    for (size_t i = 0; i < dim1; i++)
    {
        for (size_t j = 0; j < dim2; j++)
        {
            auto u_max_0 = std::max( u.vertices.first(i),u.vertices.first(j) );
            auto u_max_1 = std::max( u.vertices.second(i),u.vertices.second(j) );
            normal_vel2( i , j ) = u_max_0 * nij.first(i,j) + u_max_1*nij.second(i,j);
            //}
        }
           
    }
    
     //normal velocity (transposed)
    auto nij_transpose2 = std::make_pair( ( global_cij_x.transpose() ).cwiseQuotient(cji_norm) ,                                    ( global_cij_y. transpose() ).cwiseQuotient( cji_norm ) ) ;
    
    for (size_t i = 0; i < dim2; i++)
    {
        for (size_t j = 0; j < dim1; j++)
        {
            auto u_max_0 = std::max( u.vertices.first(i),u.vertices.first(j) );
            auto u_max_1 = std::max( u.vertices.second(i),u.vertices.second(j) );
            normal_vel_adj2( i , j ) = u_max_0 * nij_transpose2.first(i,j) + u_max_1*nij_transpose2.second(i,j);
        }
    }

    Matrix<T, Dynamic, Dynamic> lambda_max2 = normal_vel2.cwiseAbs() ;  // o lambda_max=normal_vel ?
    auto tmp2 = lambda_max2.cwiseProduct( cij_norm ) ;
    
    Matrix<T, Dynamic, Dynamic> lambda_max_adj2 = normal_vel_adj2.cwiseAbs() ;
    auto tmp_adj2 = lambda_max_adj2.cwiseProduct( cji_norm ) ;
  
    auto dij2 = tmp2.cwiseMax(tmp_adj2);
    Matrix<T, Dynamic, 1> term_dij2 = Eigen::Matrix<T, Dynamic, 1>::Zero(dim2, 1);
    /// CHECK OUT THE FUNCTION: make_dij_vector
    // auto phi_old = (phi.values_bis).col(counter);
    // auto last_term = make_dij_vector(msh , dij, phi_old );

    for (size_t i = 0; i<dij.rows(); i++)
    {
        auto diff =  phi.vertices - phi.vertices(i)*Vec_One;
        term_dij2(i) = ( dij2.row(i) ).dot( diff )  ;
    }
    std::cout<<"diff term dij: "<<term_dij-term_dij2<<std::endl;
   
    // ****************END OF OLD IMPLEMENTATION **************** //
    
 */
    
    tc2.toc();
    std::cout << bold << yellow << "Solving time method2: " << tc2 << " seconds" << reset << std::endl;
    
    
    timecounter tc3;
    tc3.tic();
       
    ///********* RESOLUTION OF THE SYSTEM: **********//
    //tc.tic();
    Matrix<T, Dynamic, 1> phi_L = solveFEM(global_lumped_mass , conv_global , term_dij_no_entropy , phi.vertices , dt , bdry_nodes); // VERSION WITHOUT ENTROPY CORRECTION
    tc3.toc();
    std::cout << bold << yellow << "Solving time ORIGINAL METHOD: " << tc3 << " seconds" << reset << std::endl;
    
    timecounter tc4;
    tc4.tic();

     Matrix<T, Dynamic, 1> phi_H = solveFEM_Entropic_FAST(llt , conv_global , term_dij , mass_phi_old , dt );
    
    //Matrix<T, Dynamic, 1> phi_H = solveFEM_Entropic(global_mass , conv_global , term_dij , mass_phi_old , dt );
     //std::cout<<'\n'<<"phi_H -phi_H_fast : "<<'\n'<<phi_H-phi_H_prova<<std::endl;
    tc4.toc();
    
    std::cout << bold << yellow << "Solving time CORRECTED METHOD: " << tc4 << " seconds" << reset << std::endl;
    
    
    
    // std::cout<<'\n'<<"phi_L : "<<'\n'<<phi_L<<std::endl;
    
    
    // CHECKING phi_h and phi_l
    Matrix<T, Dynamic, 1> zero_vec = Eigen::Matrix<T, Dynamic, 1>::Zero(phi.vertices.rows(), 1);
   // checking_phi_l( global_lumped_mass , global_cij_x , global_cij_y ,dij , phi.vertices ,phi_L,dt, S_i); // NO, DA MODIFICARE
    
    //checking_phi_lBIS( global_lumped_mass , conv_global , term_dij_no_entropy , phi.vertices , phi_L , dt );
    
    //checking_phi_h( global_mass , global_cij_x , global_cij_y , dC_ij , phi.vertices  , phi_H ,dt,S_i); // NO, DA MODIFICARE
    
    //checking_phi_hBIS( global_mass , conv_global , term_dij , phi.vertices , phi_H , dt );
    
   
    
    
    

    // EXTENSION: MAXIMUM PRINCIPLE PRESERVING
    
    Matrix<T, Dynamic, 1> delta_phi = phi_H - phi.vertices;
    //std::cout<<'\n'<<"delta_phi "<<'\n'<<delta_phi<<std::endl;
    
    Matrix<T, Dynamic, 1> f_i = f_ij_creator( global_lumped_mass , global_mass , delta_phi , dt , dij , dC_ij , phi.vertices , S_i );
    
    // CHECKING PHI
    auto check_phi = phi_H - phi_L - f_i.cwiseQuotient(global_lumped_mass) ;//+dt*Vec_One ;
    Matrix<T, Dynamic, 1>  phi_H2 = phi_L + f_i.cwiseQuotient(global_lumped_mass);
    //std::cout<<'\n'<<"check phi "<<'\n'<<check_phi<<std::endl;
   // std::cout<<'\n'<<"check phi_h: "<<'\n'<<phi_H - phi_H2<<std::endl;
    
    // CORRECTION TERM
    Matrix<T, Dynamic, 1> correction_fi = alfaf_ij_creator( global_lumped_mass , global_mass , delta_phi , phi_L , dt , dij , dC_ij , phi.vertices , S_i );
    Matrix<T, Dynamic, 1>  phi_new = phi_L + correction_fi.cwiseQuotient(global_lumped_mass);
    
    //T check_phi2 = ((phi_new - phi_L).cwiseProduct(global_lumped_mass)).sum();
   // std::cout<<'\n'<<"check PHI_NEW - PHI_L = "<<check_phi2<<std::endl;
    
    //T check_phi3 = ((phi_new - phi.vertices).cwiseProduct(global_lumped_mass)).sum();
   // std::cout<<'\n'<<"check PHI_NEW - PHI_OLD = "<<check_phi3<<std::endl;
    
   // T check_phi4 = ((phi_L - phi.vertices).cwiseProduct(global_lumped_mass)).sum();
   // std::cout<<'\n'<<"check PHI_L - PHI_OLD = "<<check_phi4<<std::endl;
    
    //T check_phi5 = ((phi_H - phi.vertices).cwiseProduct(global_lumped_mass)).sum();
    //std::cout<<'\n'<<"check PHI_H - PHI_OLD = "<<check_phi5<<std::endl;
    
    //T check_phi6 = ((phi_H - phi_L).cwiseProduct(global_lumped_mass)).sum();
    //std::cout<<'\n'<<"check PHI_H - PHI_L = "<<check_phi6<<std::endl;
    
    

    // BOUNDARY CONDITION IMPOSITION -->   NO ENTROPY
    for (auto& j : bdry_nodes )
        phi_L(j) = phi.vertices(j) ;
    
    // BOUNDARY CONDITION IMPOSITION -> ENTROPIC SOLUTION
    for (auto& j : bdry_nodes )
        phi_H(j) = phi.vertices(j) ;
    
    // BOUNDARY CONDITION IMPOSITION -> ENTROPIC AND MAX PRESERVING SOLUTION
    for (auto& j : bdry_nodes )
        phi_new(j) = phi.vertices(j) ;
    
    
    // SAVING AND UPLOAD phi_new  INTO CLASS projected_level_set
    phi.converting_into_HHO_formulation(phi_new);
    phi.vertices = phi_new;
    
    // INVERSE MAPPING BACK [0,1] --> [PHI_MIN,PHI_MAX]
    inverse_mapping_phi( phi , phi.phi_max , phi.phi_min );
    phi.phi_max = phi.vertices.maxCoeff();
    phi.phi_min = phi.vertices.minCoeff();
    
    tc.toc();
    std::cout << bold << yellow << "FEM method, time resolution: " << tc << " seconds" << reset << std::endl;
    
    /// PLOTTING SOLUTION (GNUPLOT)
    /*
    postprocess_output<double> postoutput5;
    // auto test_phi_h = std::make_shared< gnuplot_output_object<double> >("phi_h.dat");
    // auto test_phi_l = std::make_shared< gnuplot_output_object<double> >("phi_l.dat");
    auto test_phi_new = std::make_shared< gnuplot_output_object<double> >("phi_new.dat");
    
    size_t iii = 0;
    for (auto pt : msh.points) {
      //  test_phi_h->add_data(pt,phi_H(iii) );
      //  test_phi_l->add_data(pt,phi_L(iii) );
        test_phi_new->add_data(pt,phi_new(iii) );
        iii++;
    }
    //   postoutput5.add_object(test_phi_h);
    //  postoutput5.add_object(test_phi_l);
    postoutput5.add_object(test_phi_new);
    postoutput5.write();
    */
    
    std::cout << bold << yellow << "!!ATTENCTION: I'M COMPUTING BOTH phi AND phi_entropic!!" << reset <<std::endl;
    
    
    
    /// COUT OF PHI IN CELLWISE NOTATION
/*
    for (size_t i = 0; i<phi.values_bis.rows(); i++) {
        for (size_t j = 0; j<phi.values_bis.cols(); j++) {
            std::cout<<phi.values_bis(i,j);
        }
        std::cout<<'\n';
    }
    std::cout<<std::endl;
*/
  
    

}


template < typename Fonction, typename Mesh, typename Vel_Field , typename T = typename Mesh::coordinate_type >
void run_FEM_levelset_SPARSE(const Mesh & msh, size_t degree, Fonction & phi, const Vel_Field & u , const T& dt , const mesh_init_params<T>& params)
{
    timecounter tc;
    tc.tic();
    size_t ndof = ( degree+1 ) * ( degree+1 ) ; // degrees of freedom
    const T eps = 1e-14; //constant into entropy
 
    // LOCAL MASS MATRIX ==> CHECK DEGREE == 1!!!
    assert(degree==1);
    
    
    mapping_phi(phi , phi.phi_max , phi.phi_min ); // mapping of phi to  have a phi between 0 and 1
    /// PHI MAPPED INTO 0-1 PLOT
    /*
    postprocess_output<double> postoutput4;
    auto test_phi_mapped = std::make_shared< gnuplot_output_object<double> >("phi_mapped.dat");
    for (auto cl:msh.cells) {
        auto pts = points(msh,cl) ;
        for (auto pt:pts) {
            T value = phi(pt,msh,cl);
            test_phi_mapped->add_data(pt,value);
        }
    }
    postoutput4.add_object(test_phi_mapped);
    postoutput4.write();
    */
    
    // ENTROPY -> for phi with codomain in (0,1)
    non_linear_entropy<T,Fonction,Mesh> E(eps , phi ,msh );
    auto phi_tilde = projection<T,Mesh> ( msh, degree , params );
    
    /// ENTROPY PLOT
    /*
    postprocess_output<double> postoutput6;
    auto cut_entropy = std::make_shared< gnuplot_output_object<double> >("entropy.dat");
    for (auto cl:msh.cells) {
        auto pts = points(msh,cl) ;
        for (auto pt:pts) {
            T value = E(pt,cl);
            cut_entropy->add_data(pt,value);
        }
    }
    postoutput6.add_object(cut_entropy);
    postoutput6.write();
    */

    // Global Matrix and Global Vector Definitions:
    size_t dim = msh.nodes.size();
    
    //SPARSE
    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;
    std::vector< Triplet<T> >   triplets;
    LHS = SparseMatrix<T>( dim, dim );
    RHS = Matrix<T, Dynamic, 1>::Zero( dim );
    
    
    
    // Lumped mass vector
    Matrix<T, Dynamic, 1> global_lumped_mass = Eigen::Matrix<T, Dynamic, 1>::Zero( dim, 1 );
    // Mass matrix
    Matrix<T, Dynamic, Dynamic> global_mass = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim );
    // C_ij,1 matrix
    Matrix<T, Dynamic, Dynamic> global_cij_x = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim );
    // C_ij,2 matrix
    Matrix<T, Dynamic, Dynamic> global_cij_y = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim );
    // Convection term (vector)
    Matrix<T, Dynamic, 1> conv_global = Eigen::Matrix<T, Dynamic, 1>::Zero( dim, 1 );
    // D_ij term (vector)
    Matrix<T, Dynamic, 1> term_dij = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1 );
    // N_ij matrix
    std::pair<Matrix<T, Dynamic, Dynamic>,Matrix<T, Dynamic, Dynamic>> nij = std::make_pair(Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) , Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) ) ;
    // N_ji matrix
    std::pair<Matrix<T, Dynamic, Dynamic>,Matrix<T, Dynamic, Dynamic>> nij_transpose = std::make_pair(Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) , Eigen::Matrix<T, Dynamic, Dynamic>::Zero( dim, dim ) ) ;
    // ||c_ij||^2
    Matrix<T, Dynamic, Dynamic> cij_square = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    // ||c_ji||^2
    Matrix<T, Dynamic, Dynamic> cji_square = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    
    // One vector
    Matrix<T, Dynamic, 1> Vec_One = Eigen::Matrix<T, Dynamic, 1>::Ones( dim, 1 );
    // One matrix
    Matrix<T, Dynamic, Dynamic> Mat_One = Eigen::Matrix<T, Dynamic, Dynamic>::Ones( dim, dim );
    
    
    // Boundary Nodes = Vertices (in Q1)
    auto bdry_nodes = boundary_nodes_function( phi.Nx , phi.Ny );

    
    // The Support of Global Shape Function: list of neighborood nodes for each node i
    std::vector<std::set<size_t>> S_i(dim);
    timecounter tc1;
    tc1.tic();
    for (auto cl : msh.cells)
    {
        
        
        auto nodes_position = nodes(msh, cl); // node_pos.ptid to have just the number
        
        // In basis (mesh creation) the order is
        /// 2 - - 3
        /// |       |
        /// 0 - - 1
        
        // In nodes position instead the order is
        /// 3 - - 2
        /// |       |
        /// 0 - - 1
        // I swap last two elements to have the first ordering
        iter_swap(nodes_position.end()-1 , nodes_position.end()-2 );
       
        // Support of global shape function: list of close nodes for each node i
        supporting_nodes( S_i , nodes_position );
        
        /*
        std::vector<assembly_index> asm_map;
        for (auto& nd : nodes_position) {
            if(location(msh, n) == )
            asm_map.push_back( assembly_index(nd, true) );
            asm_map.push_back( assembly_index(nd, !dirichlet) );
        }
        
        triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
        */
        
        /*
        // CALCULATION OF THE SIZE + PLOTTING
        size_t size_supp_nodes = 0;
        std::cout<<"Supporting nodes:"<<std::endl;
        size_t jjjj = 0;
        for (auto& i:S_i) {
            size_supp_nodes+=i.size();
            std::cout <<"Node "<<jjjj<<":";
            for (auto it=i.begin(); it != i.end(); ++it)
                std::cout << ' ' << *it;
               // std::cout<<ii;
            std::cout<<'\n';
            jjjj++;
        }
        std::cout<<std::endl;
        std::cout<<"Supporting nodes size:"<<size_supp_nodes<<std::endl;
        */
        
        
        // Local mass matrix
        Matrix<T, Dynamic, Dynamic> mass = make_lagrange_local_mass_matrix(msh, cl, degree);
        
        // Assembling into global matrix
        //global_update2( mass , global_mass , nodes_position2 );
        global_update( mass , global_mass , nodes_position );
        
        // Local c_ij = b_i nabla(b_j)
        auto cij = make_local_cij_matrix(msh, cl, degree);
        
        // Assembling into global matrix
        global_update( cij.first , global_cij_x , nodes_position );
        global_update( cij.second , global_cij_y , nodes_position );
        
    // USEFUL COMMAND FOR NORMS of a Eigen::matrix A:
    // A.template lpNorm<2>()
    // A.norm()
        
    }
    
    // LUMPED MASS VECTOR (i.e. vector = diagonal of a matrix)
    sum_Si(S_i, global_mass , Vec_One , global_lumped_mass );
    
    
    
    // CHECKING OF MASS MATRICES' PROPERTIES-> OK, perfect precision 1e-19
    /*
    std::cout<<"First checking: Lumped Mass"<<std::endl;
    for(size_t i = 0; i<global_lumped_mass.rows();i++)
        if(global_lumped_mass(i)<0)
            std::cout<<"PROBLEM into lumped mass"<<std::endl;
    
    Matrix<T, Dynamic, Dynamic> mass_check = - global_mass;
    for(size_t i = 0 ; i < mass_check.rows() ; i++){
        mass_check(i,i) += global_lumped_mass(i);
        std::cout<<(mass_check.row(i)).sum()<<std::endl;
    }
    */
    
    
    
    /// CHECK OF THE SUM PROPERTY OF CIJ
    /*
     Matrix<T, Dynamic, 1> sum0 = Eigen::Matrix<T, Dynamic, 1>::Zero(global_cij_x.rows(), 1);
     Matrix<T, Dynamic, 1> sum1 = Eigen::Matrix<T, Dynamic, 1>::Zero(global_cij_x.rows(), 1);
    
    for (size_t k = 0; k<global_cij_x.cols(); k++)
    {
        sum0 += (global_cij_x).col(k);
        sum1 += (global_cij_y).col(k);
    }
    
    for (size_t i=0; i<global_cij_y.rows(); i++) {
         std::cout<<"The sum is "<<sum1(i)<<" and "<<sum0(i)<<std::endl;
    }
    
    size_t counter_sum = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
           sum0( counter_sum ) += global_cij_x( counter_sum , elem );
           sum1( counter_sum ) += global_cij_y( counter_sum , elem );
           if(elem==counter_sum)
               std::cout<<"In ("<<counter_sum<<" , "<<elem<<" ), c^0 = "<<global_cij_x( counter_sum , elem )<<" and c1 = "<<global_cij_y( counter_sum , elem )<<std::endl;
           else
                std::cout<<"In ("<<counter_sum<<" , "<<elem<<" ), c^0_ij + c^0_ji = "<<global_cij_x( counter_sum , elem ) + global_cij_x( elem , counter_sum )<<" and c^1_ij + c^1_ji = "<<global_cij_y( counter_sum , elem ) + global_cij_y( elem , counter_sum ) <<std::endl;
        }
        counter_sum++;
    }
    //std::cout<<"The sum0 is "<<'\n'<<sum0<<" and sum1 "<<'\n'<<sum1<<std::endl;
    */
     
   

    // FLUX TERM
    Matrix<T, Dynamic, 1> flux0 = (u.vertices.first).cwiseProduct(phi.vertices) ;
    Matrix<T, Dynamic, 1> flux1 = (u.vertices.second).cwiseProduct(phi.vertices) ;
    
    
    
    // RESOLUTION OF phi_tilde GLOBALE     ( with cij(phi_j - phi_i) )
    
    Matrix<T, Dynamic, 1> vec1 = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    sum_Si(S_i , global_mass , phi.vertices , vec1);
    
    Matrix<T, Dynamic, 1> vec2 = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    //Matrix<T, Dynamic, 1> vec2_bis = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    averaged_sum_Si(S_i , global_cij_x , flux0 , vec2);
    averaged_sum_Si(S_i , global_cij_y , flux1 , vec2);
    //sum_Si(S_i , global_cij_x , flux0 , vec2_bis);
    //sum_Si(S_i , global_cij_y , flux1 , vec2_bis);
    //std::cout<<"diff vec 2 = "<<'\n'<<vec2-vec2_bis<<std::endl;
    /// THERE IS A DIFFERENCE OF  <1E-18 DEPENDING IF I CONSIDER OR NOT THE AVERAGED SUM
    
    Matrix<T, Dynamic, 1> b = vec1 - dt*vec2.cwiseQuotient(global_lumped_mass);
    
    /// OLD IMPLEMENTATION
    //Matrix<T, Dynamic, 1> prova_FAST = global_mass.completeOrthogonalDecomposition().solve(b);
    //phi_tilde.vertices = global_mass.HouseholderQR().solve(b);
    timecounter tc00;
    tc00.tic();
    LLT<Matrix<T, Dynamic, Dynamic > > llt( global_mass );
    phi_tilde.vertices = llt.solve(b);
    tc00.toc();
    
    
    if (llt.info() != Success)
    {
        std::cout<<"not positive"<<std::endl;
        assert(0);
    }
    //std::cout<<"ERRORE FAST= "<<'\n'<<phi_tilde.vertices - prova_FAST <<std::endl;
   
    phi_tilde.converting_into_HHO_formulation( phi_tilde.vertices );
    std::cout << bold << yellow << "LLT RESOLUTION TIME: " << tc00 << " seconds" << reset << std::endl;
    
    
    //std::cout<<"phi_tilde= "<<'\n'<<phi_tilde.vertices<<std::endl;
    /*
    postprocess_output<double> postoutput3;
    typedef typename Mesh::point_type       point_type;
    point<double,2> node;
    auto test_phi_tilde = std::make_shared< gnuplot_output_object<double> >("phi_tilde.dat");
    for (size_t i = 0; i<phi_tilde.vertices.rows(); i++) {
        node = msh.points.at(i);
        test_phi_tilde->add_data(node,phi_tilde.vertices(i));
    }
    postoutput3.add_object(test_phi_tilde);
    postoutput3.write();
    */
    
    
    // Term R_i^n
    Matrix<T, Dynamic, 1> Emax_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    Matrix<T, Dynamic, 1> Emin_global = Eigen::Matrix<T, Dynamic, 1>::Ones(dim, 1);
    Emax_global *= -1e20;
    Emin_global *= 1e20;
    Matrix<T, Dynamic, 1> R_i = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    for( auto& cl: msh.cells )
    {
        r_i_calculator( msh , cl , E , phi_tilde , phi , dt , u ,S_i , Emax_global , Emin_global , R_i);
    }
    //std::cout<<"Emax_global - Emin_global = "<<'\n'<<Emax_global-Emin_global<<std::endl;
    R_i = R_i.cwiseQuotient( Emax_global - Emin_global );
    //std::cout<<"Ri = "<<'\n'<<R_i<<std::endl;
    
    
    
    // CONVOLUTION TERM
    //sum_Si(S_i, global_cij_x , flux0 , conv_global );
    //sum_Si(S_i, global_cij_y , flux1 , conv_global );
    averaged_sum_Si(S_i, global_cij_x , flux0 , conv_global );
    averaged_sum_Si(S_i, global_cij_y , flux1 , conv_global );
    // OLD IMPLEMENTATION
    //auto conv_global2 = global_cij_x*( flux0 ) + global_cij_y*( flux1 ) ;
    
    // Norm of c_ij
    moltiplication_Si( S_i , global_cij_x , global_cij_x , cij_square );
    moltiplication_Si( S_i , global_cij_y , global_cij_y , cij_square );
    Matrix<T, Dynamic, Dynamic> cij_norm = cij_square.cwiseSqrt();
    
    // Matrix n_ij
    division_Si( S_i , global_cij_x , cij_norm , nij.first );
    division_Si( S_i , global_cij_y , cij_norm , nij.second );
    // OLD IMPLEMENTATION
    //auto nij2 = std::make_pair( global_cij_x.cwiseQuotient(cij_norm) , (global_cij_y).cwiseQuotient( cij_norm ) ) ;
    
    // NOTICE:
    /// Normal Velocity calculation IS EQUIVALENT TO f', that is EQUIVALENT to lambda_r , lambda_l.  Moreover I don't need the calculation of the normal flux. Since then it has to be derived, it is enough to calculate f' = u dot n
     
    // Normal velocity
    Matrix<T, Dynamic, Dynamic> normal_vel = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    size_t counter_row = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            auto u_max_0 = std::max( u.vertices.first(counter_row),u.vertices.first(elem) );
            auto u_max_1 = std::max( u.vertices.second(counter_row),u.vertices.second(elem) );
            normal_vel(counter_row,elem) = u_max_0 * nij.first(counter_row,elem) + u_max_1*nij.second(counter_row,elem);
        }
        counter_row++;
    }
     
    
    
   
    // C_ji matrix
    std::pair<Matrix<T, Dynamic, Dynamic>,Matrix<T, Dynamic, Dynamic>> cji_global = std::make_pair(global_cij_x.adjoint() , global_cij_y.adjoint() );
    
    // Norm of c_ji -> i.e. c_ij transposed
    moltiplication_Si( S_i , cji_global.first , cji_global.first , cji_square );
    moltiplication_Si( S_i , cji_global.second , cji_global.second , cji_square );
    // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
    //moltiplication_Si_T( S_i , cji_global.first , cji_global.first , cji_square );
    //moltiplication_Si_T( S_i , cji_global.second , cji_global.second , cji_square );
    
    //C_ji Norm
    Matrix<T, Dynamic, Dynamic> cji_norm = cji_square.cwiseSqrt();
    //auto cji_square2 = (cji_global.first).cwiseProduct(cji_global.first) + (cji_global.second).cwiseProduct(cji_global.second);
    //Matrix<T, Dynamic, Dynamic> cji_norm2 = cji_square2.cwiseSqrt();
    
   
    // Matrix n_ij (TRANSPOSED)
    division_Si( S_i , cji_global.first , cji_norm , nij_transpose.first );
    division_Si( S_i , cji_global.second , cji_norm , nij_transpose.second );
     // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
    //division_Si_T( S_i , cji_global.first , cji_norm , nij_transpose.first );
    //division_Si_T( S_i , cji_global.second , cji_norm , nij_transpose.second );
    //auto  nij_transposex2 = cji_global.first.cwiseQuotient( cji_norm2 );
    //auto  nij_transposey2 = cji_global.second.cwiseQuotient( cji_norm2 );
    
    // Normal velocity (TRANSPOSED)
    Matrix<T, Dynamic, Dynamic> normal_vel_adj = Eigen::Matrix<T,Dynamic,Dynamic>::Zero(dim, dim);
    //Matrix<T, Dynamic, Dynamic> normal_vel_adj2 = Eigen::Matrix<T,Dynamic,Dynamic>::Zero(dim, dim);
    /*
    for(size_t i = 0 ; i< normal_vel_adj2.rows() ; i++)
    {
        for(size_t j = 0 ; j< normal_vel_adj2.cols() ; j++)
        {
            auto u_max_0 = std::max( u.vertices.first(i),u.vertices.first(j) );
            auto u_max_1 = std::max( u.vertices.second(i),u.vertices.second(j) );
            //normal_vel(counter_row,elem) = u_max_0 * nij.first(counter_row,elem) + u_max_1*nij.second(counter_row,elem);
            // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
            normal_vel_adj2(i,j) = u_max_0 * nij_transposex2(i,j) + u_max_1*nij_transposey2(i,j);
        }
    }
    
    */

    counter_row = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            auto u_max_0 = std::max( u.vertices.first(elem),u.vertices.first(counter_row) );
            auto u_max_1 = std::max( u.vertices.second(elem),u.vertices.second(counter_row) );
            //normal_vel_adj(elem,counter_row) = u_max_0 * nij.first(elem,counter_row) + u_max_1*nij.second(elem,counter_row);
            // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
            normal_vel_adj(counter_row , elem) = u_max_0 * nij_transpose.first(counter_row,elem) + u_max_1*nij_transpose.second(counter_row,elem);
        }
        counter_row++;
    }
    
    
    // Matrix Dij calculation
    Matrix<T, Dynamic, Dynamic> lambda_max = normal_vel.cwiseAbs() ;  // o lambda_max=normal_vel ?
    Matrix<T, Dynamic, Dynamic> tmp = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    moltiplication_Si( S_i , lambda_max , cij_norm , tmp );
    
    Matrix<T, Dynamic, Dynamic> lambda_max_adj=normal_vel_adj.cwiseAbs();// o lambda_max=normal_vel?
    Matrix<T, Dynamic, Dynamic> tmp_adj = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    moltiplication_Si( S_i , lambda_max_adj , cji_norm , tmp_adj );
    // CHECK IF WORKS BETTER NOW (  NO TRANSPOSE )
    //moltiplication_Si_T( S_i , lambda_max_adj , cji_norm , tmp_adj );
    //Matrix<T, Dynamic, Dynamic> lambda_max_adj2=normal_vel_adj2.cwiseAbs();
    //auto tmp_adj2 = lambda_max_adj.cwiseProduct( cji_norm2 ) ;
    
    Matrix<T, Dynamic, Dynamic> dij = tmp.cwiseMax(tmp_adj);
    
    //Matrix<T, Dynamic, Dynamic> dij2 = tmp.cwiseMax(tmp_adj2);
    
    //std::cout<<"Dij: "<<'\n'<<dij<<std::endl;
    
    /*
      /// The  error is < 1e-15, at max
    std::cout<<"D_ij check: "<<std::endl;
    for(size_t i = 0; i<tmp.rows();i++){
        for(size_t j = i+1; j<tmp.rows();j++){
            if(std::abs(dij(i,j)-dij(j,i))>1e-15)
            std::cout<<dij(i,j)-dij(j,i);
        }
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    */
   
    
    tc1.toc();
    std::cout << bold << yellow << "Solving time method original: " << tc1 << " seconds" << reset << std::endl;
    
    timecounter tc2;
    tc2.tic();
       
    

    
    // Minimum between d_ij and R_i
    T c_e = 1.;
    Matrix<T, Dynamic, Dynamic> dE_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    Matrix<T, Dynamic, Dynamic> phi_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    counter_row = 0;
   
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            if(elem!=counter_row)
            {
            auto R_i_j = c_e * std::max( std::abs( R_i(counter_row) ) , std::abs( R_i(elem) ) );
            dE_ij( counter_row , elem ) = std::min( dij( counter_row , elem ), R_i_j );
            //std::cout<<"R_i_j "<<R_i_j<<" and dij"<<dij( counter_row , elem )<<std::endl;
            }
            phi_ij( counter_row , elem ) = 0.5*( phi.vertices(counter_row) + phi.vertices(elem) );
        }
        counter_row++;
    }
    //std::cout<<"dE_ij: "<<'\n'<<dE_ij<<std::endl;
    //std::cout<<"phi_ij"<<'\n'<<phi_ij<<std::endl;
    T c_comp = 1.;
    Matrix<T, Dynamic, Dynamic> dC_ij = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    Matrix<T, Dynamic, Dynamic> tmp0 = phi_ij - phi_ij.cwiseProduct(phi_ij);
    //Matrix<T, Dynamic, Dynamic> tmp0 = (phi_ij-phi.phi_min*Mat_One).cwiseProduct(phi.phi_max*Mat_One-phi_ij);
    //std::cout<<"tmp0: "<<'\n'<<tmp0<<std::endl;
    positive_part( tmp0 );
    
    
   // Matrix<T, Dynamic, Dynamic> tmp_dC0 = positive_part( tmp0 );

   
    Matrix<T, Dynamic, Dynamic> tmp_dC1 =  Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim, dim);
    
    size_t counter = 0;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
           // std::cout<<"In ("<<counter<<" , "<<elem<<") tmp_dC0 = "<<tmp_dC0(counter,elem)<<" and phi.ver diff "<<std::abs(phi.vertices(counter) - phi.vertices(elem))<<std::endl;
            if( std::abs(phi.vertices(counter) - phi.vertices(elem))>1e-15 )
            tmp_dC1(counter,elem) = tmp0(counter,elem)/( std::abs(phi.vertices(counter) - phi.vertices(elem)) );
        }
        counter++;
    }
    //std::cout<<'\n'<<"tmp_dC1: "<<'\n'<<tmp_dC1<<std::endl;
    Matrix<T, Dynamic, Dynamic> tmp1 = Mat_One-c_comp*tmp_dC1;
    
    //std::cout<<'\n'<<"tmp1: "<<'\n'<<tmp1<<std::endl;
    positive_part( tmp1  );
    //std::cout<<'\n'<<"tmp1 POSTIVIVE: "<<'\n'<<tmp1<<std::endl;
    
    moltiplication_Si( S_i , dE_ij , tmp1 , dC_ij );
   
    
    for(size_t i = 0 ; i < dC_ij.rows() ; i++)
    {
        dC_ij(i,i) = 0;
        dij(i,i) = 0;
        dC_ij(i,i) = -dC_ij.row(i).sum();
        dij(i,i) = -dij.row(i).sum();
    }
    //dC_ij = dE_ij.cwiseProduct( positive_part( tmp1  ) );
    //   std::cout<<'\n'<<"dC_ij: "<<'\n'<<dC_ij<<std::endl;
    //   std::cout<<'\n'<<"dij: "<<'\n'<<dij<<std::endl;
    
   
    //std::cout<<"dC_ij: "<<'\n'<<dC_ij<<std::endl;
    
    
    /*
    counter_row = 0;
    std::cout<<"DC_ij check symmetry: "<<std::endl;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<dC_ij(counter_row,elem) - dC_ij(elem,counter_row)<<std::endl;
        }
        counter_row++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    
    
    counter_row = 0;
    std::cout<<"D_ij check symmetry "<<std::endl;
    for(auto& row_i:S_i)
    {
        for(auto& elem:row_i)
        {
            std::cout<<dij(counter_row,elem) - dij(elem,counter_row)<<std::endl;
        }
        counter_row++;
        std::cout<<'\n';
    }
    std::cout<<std::endl;
    */
    
    
    // Term D_ij(phi_j - phi_i )
    averaged_sum_Si(S_i, dC_ij , phi.vertices , term_dij );
    //std::cout<<"term_dij : "<<'\n'<<term_dij<<std::endl;
    
    
    //averaged_sum_Si(S_i, dE_ij , phi.vertices , term_dij );
    Matrix<T, Dynamic, 1> term_dij_no_entropy = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1 );
    // VERSION WITHOUT ENTROPY CORRECTION
    averaged_sum_Si(S_i, dij , phi.vertices , term_dij_no_entropy );
    
    //std::cout<<"term_dij_no_entropy : "<<'\n'<<term_dij_no_entropy<<std::endl;
    
    // M*phi^n
    Matrix<T, Dynamic, 1> mass_phi_old = Eigen::Matrix<T, Dynamic, 1>::Zero(dim, 1);
    sum_Si( S_i , global_mass , phi.vertices , mass_phi_old );
    
    
    
  /*
    // **************** OLD IMPLEMENTATION **************** //
    
   for (size_t i = 0; i<global_lumped_mass.rows(); i++) {
          global_lumped_mass(i) = global_mass.row(i).sum() ;
      }
   
   
    Matrix<T, Dynamic, Dynamic> normal_vel2 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim1, dim2);
    Matrix<T, Dynamic, Dynamic> normal_vel_adj2 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero(dim2, dim1);
    
    //normal velocity
    for (size_t i = 0; i < dim1; i++)
    {
        for (size_t j = 0; j < dim2; j++)
        {
            auto u_max_0 = std::max( u.vertices.first(i),u.vertices.first(j) );
            auto u_max_1 = std::max( u.vertices.second(i),u.vertices.second(j) );
            normal_vel2( i , j ) = u_max_0 * nij.first(i,j) + u_max_1*nij.second(i,j);
            //}
        }
           
    }
    
     //normal velocity (transposed)
    auto nij_transpose2 = std::make_pair( ( global_cij_x.transpose() ).cwiseQuotient(cji_norm) ,                                    ( global_cij_y. transpose() ).cwiseQuotient( cji_norm ) ) ;
    
    for (size_t i = 0; i < dim2; i++)
    {
        for (size_t j = 0; j < dim1; j++)
        {
            auto u_max_0 = std::max( u.vertices.first(i),u.vertices.first(j) );
            auto u_max_1 = std::max( u.vertices.second(i),u.vertices.second(j) );
            normal_vel_adj2( i , j ) = u_max_0 * nij_transpose2.first(i,j) + u_max_1*nij_transpose2.second(i,j);
        }
    }

    Matrix<T, Dynamic, Dynamic> lambda_max2 = normal_vel2.cwiseAbs() ;  // o lambda_max=normal_vel ?
    auto tmp2 = lambda_max2.cwiseProduct( cij_norm ) ;
    
    Matrix<T, Dynamic, Dynamic> lambda_max_adj2 = normal_vel_adj2.cwiseAbs() ;
    auto tmp_adj2 = lambda_max_adj2.cwiseProduct( cji_norm ) ;
  
    auto dij2 = tmp2.cwiseMax(tmp_adj2);
    Matrix<T, Dynamic, 1> term_dij2 = Eigen::Matrix<T, Dynamic, 1>::Zero(dim2, 1);
    /// CHECK OUT THE FUNCTION: make_dij_vector
    // auto phi_old = (phi.values_bis).col(counter);
    // auto last_term = make_dij_vector(msh , dij, phi_old );

    for (size_t i = 0; i<dij.rows(); i++)
    {
        auto diff =  phi.vertices - phi.vertices(i)*Vec_One;
        term_dij2(i) = ( dij2.row(i) ).dot( diff )  ;
    }
    std::cout<<"diff term dij: "<<term_dij-term_dij2<<std::endl;
   
    // ****************END OF OLD IMPLEMENTATION **************** //
    
 */
    
    tc2.toc();
    std::cout << bold << yellow << "Solving time method2: " << tc2 << " seconds" << reset << std::endl;
    
    
    timecounter tc3;
    tc3.tic();
       
    ///********* RESOLUTION OF THE SYSTEM: **********//
    //tc.tic();
    Matrix<T, Dynamic, 1> phi_L = solveFEM(global_lumped_mass , conv_global , term_dij_no_entropy , phi.vertices , dt , bdry_nodes); // VERSION WITHOUT ENTROPY CORRECTION
    tc3.toc();
    std::cout << bold << yellow << "Solving time ORIGINAL METHOD: " << tc3 << " seconds" << reset << std::endl;
    
    timecounter tc4;
    tc4.tic();

     Matrix<T, Dynamic, 1> phi_H = solveFEM_Entropic_FAST(llt , conv_global , term_dij , mass_phi_old , dt );
    
    //Matrix<T, Dynamic, 1> phi_H = solveFEM_Entropic(global_mass , conv_global , term_dij , mass_phi_old , dt );
     //std::cout<<'\n'<<"phi_H -phi_H_fast : "<<'\n'<<phi_H-phi_H_prova<<std::endl;
    tc4.toc();
    
    std::cout << bold << yellow << "Solving time CORRECTED METHOD: " << tc4 << " seconds" << reset << std::endl;
    
    
    
    // std::cout<<'\n'<<"phi_L : "<<'\n'<<phi_L<<std::endl;
    
    
    // CHECKING phi_h and phi_l
    Matrix<T, Dynamic, 1> zero_vec = Eigen::Matrix<T, Dynamic, 1>::Zero(phi.vertices.rows(), 1);
   // checking_phi_l( global_lumped_mass , global_cij_x , global_cij_y ,dij , phi.vertices ,phi_L,dt, S_i); // NO, DA MODIFICARE
    
    //checking_phi_lBIS( global_lumped_mass , conv_global , term_dij_no_entropy , phi.vertices , phi_L , dt );
    
    //checking_phi_h( global_mass , global_cij_x , global_cij_y , dC_ij , phi.vertices  , phi_H ,dt,S_i); // NO, DA MODIFICARE
    
    //checking_phi_hBIS( global_mass , conv_global , term_dij , phi.vertices , phi_H , dt );
    
   
    
    
    

    // EXTENSION: MAXIMUM PRINCIPLE PRESERVING
    
    Matrix<T, Dynamic, 1> delta_phi = phi_H - phi.vertices;
    //std::cout<<'\n'<<"delta_phi "<<'\n'<<delta_phi<<std::endl;
    
    Matrix<T, Dynamic, 1> f_i = f_ij_creator( global_lumped_mass , global_mass , delta_phi , dt , dij , dC_ij , phi.vertices , S_i );
    
    // CHECKING PHI
    auto check_phi = phi_H - phi_L - f_i.cwiseQuotient(global_lumped_mass) ;//+dt*Vec_One ;
    Matrix<T, Dynamic, 1>  phi_H2 = phi_L + f_i.cwiseQuotient(global_lumped_mass);
    //std::cout<<'\n'<<"check phi "<<'\n'<<check_phi<<std::endl;
   // std::cout<<'\n'<<"check phi_h: "<<'\n'<<phi_H - phi_H2<<std::endl;
    
    // CORRECTION TERM
    Matrix<T, Dynamic, 1> correction_fi = alfaf_ij_creator( global_lumped_mass , global_mass , delta_phi , phi_L , dt , dij , dC_ij , phi.vertices , S_i );
    Matrix<T, Dynamic, 1>  phi_new = phi_L + correction_fi.cwiseQuotient(global_lumped_mass);
    
    //T check_phi2 = ((phi_new - phi_L).cwiseProduct(global_lumped_mass)).sum();
   // std::cout<<'\n'<<"check PHI_NEW - PHI_L = "<<check_phi2<<std::endl;
    
    //T check_phi3 = ((phi_new - phi.vertices).cwiseProduct(global_lumped_mass)).sum();
   // std::cout<<'\n'<<"check PHI_NEW - PHI_OLD = "<<check_phi3<<std::endl;
    
   // T check_phi4 = ((phi_L - phi.vertices).cwiseProduct(global_lumped_mass)).sum();
   // std::cout<<'\n'<<"check PHI_L - PHI_OLD = "<<check_phi4<<std::endl;
    
    //T check_phi5 = ((phi_H - phi.vertices).cwiseProduct(global_lumped_mass)).sum();
    //std::cout<<'\n'<<"check PHI_H - PHI_OLD = "<<check_phi5<<std::endl;
    
    //T check_phi6 = ((phi_H - phi_L).cwiseProduct(global_lumped_mass)).sum();
    //std::cout<<'\n'<<"check PHI_H - PHI_L = "<<check_phi6<<std::endl;
    
    

    // BOUNDARY CONDITION IMPOSITION -->   NO ENTROPY
    for (auto& j : bdry_nodes )
        phi_L(j) = phi.vertices(j) ;
    
    // BOUNDARY CONDITION IMPOSITION -> ENTROPIC SOLUTION
    for (auto& j : bdry_nodes )
        phi_H(j) = phi.vertices(j) ;
    
    // BOUNDARY CONDITION IMPOSITION -> ENTROPIC AND MAX PRESERVING SOLUTION
    for (auto& j : bdry_nodes )
        phi_new(j) = phi.vertices(j) ;
    
    
    // SAVING AND UPLOAD phi_new  INTO CLASS projected_level_set
    phi.converting_into_HHO_formulation(phi_new);
    phi.vertices = phi_new;
    
    // INVERSE MAPPING BACK [0,1] --> [PHI_MIN,PHI_MAX]
    inverse_mapping_phi( phi , phi.phi_max , phi.phi_min );
    tc.toc();
    std::cout << bold << yellow << "FEM method, time resolution: " << tc << " seconds" << reset << std::endl;
    
    /// PLOTTING SOLUTION (GNUPLOT)
    /*
    postprocess_output<double> postoutput5;
    // auto test_phi_h = std::make_shared< gnuplot_output_object<double> >("phi_h.dat");
    // auto test_phi_l = std::make_shared< gnuplot_output_object<double> >("phi_l.dat");
    auto test_phi_new = std::make_shared< gnuplot_output_object<double> >("phi_new.dat");
    
    size_t iii = 0;
    for (auto pt : msh.points) {
      //  test_phi_h->add_data(pt,phi_H(iii) );
      //  test_phi_l->add_data(pt,phi_L(iii) );
        test_phi_new->add_data(pt,phi_new(iii) );
        iii++;
    }
    //   postoutput5.add_object(test_phi_h);
    //  postoutput5.add_object(test_phi_l);
    postoutput5.add_object(test_phi_new);
    postoutput5.write();
    */
    
    std::cout << bold << yellow << "!!ATTENCTION: I'M COMPUTING BOTH phi AND phi_entropic!!" << reset <<std::endl;
    
    
    
    /// COUT OF PHI IN CELLWISE NOTATION
/*
    for (size_t i = 0; i<phi.values_bis.rows(); i++) {
        for (size_t j = 0; j<phi.values_bis.cols(); j++) {
            std::cout<<phi.values_bis(i,j);
        }
        std::cout<<'\n';
    }
    std::cout<<std::endl;
*/
  
    

}

template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type ,typename VEC >
void Lp_space_Tfin_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree ,double p , VEC& error )
{
    T errorLp = 0.;
    for(auto& cl : msh.cells)
    {
        auto qps = integrate(msh, cl, 2*degree + 2 );
            for (auto& qp : qps )
            {
                auto diff_val = std::abs( level_set_final( qp.first , msh , cl ) - level_set_initial( qp.first , msh , cl ) );
                errorLp += qp.second * pow(diff_val , p) ;
            }
        //std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
    }
    //std::cout<<"The L^"<<p<<" error is "<< pow (errorLp , 1.0/p )<<std::endl;
    error.push_back(pow (errorLp , 1.0/p ));
}



template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type >
void Lp_space_Tfin_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree ,double p )
{
    T errorLp = 0.;
    for(auto& cl : msh.cells)
    {
        auto qps = integrate(msh, cl, 2*degree + 2 );
            for (auto& qp : qps )
            {
                auto diff_val = std::abs( level_set_final( qp.first , msh , cl ) - level_set_initial( qp.first , msh , cl ) );
                errorLp += qp.second * pow(diff_val , p) ;
            }
        //std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
    }
    std::cout<<"The L^"<<p<<" error is "<< pow (errorLp , 1.0/p )<<std::endl;
}

template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type >
T Lp_space_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree , double p )
{
    // L^p in space ; l^q in time
    T errorLp = 0.;
    for(auto& cl : msh.cells)
    {
        auto qps = integrate(msh, cl, 2*degree + 2 ); // what orders?
            for (auto& qp : qps )
            {
                auto diff_val = std::abs( level_set_final( qp.first , msh , cl ) - level_set_initial( qp.first , msh , cl ) );
                errorLp += qp.second * pow(diff_val,p) ;
            }
        //std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
    }
    errorLp = pow( errorLp , 1.0/ p );
    //std::cout<<"The L^2 error is "<<sqrt( errorL2 )<<std::endl;
    return errorLp ;
}

template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type >
T W1p_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree , double p )
{
    T errorH1 = 0.;
    for(auto& cl : msh.cells)
    {
        auto qps = integrate(msh, cl, 2*degree + 2 ); // what orders?
            for (auto& qp : qps )
            {
                auto diff_val0 = std::abs( level_set_final.gradient( qp.first , msh , cl )(0) - level_set_initial.gradient( qp.first , msh , cl )(0) );
                auto diff_val1 = std::abs( level_set_final.gradient( qp.first , msh , cl )(1) - level_set_initial.gradient( qp.first , msh , cl )(1) );
                errorH1 += qp.second * (pow(diff_val0,p) + pow(diff_val1,p)) ;
            }
        //std::cout<<"The L^2 error squared in cell "<<offset(msh,cl)<<" is "<< errorL2 <<std::endl;
    }
    errorH1 = pow( errorH1 , 1.0/p );
    //std::cout<<"The L^2 error is "<<sqrt( errorL2 )<<std::endl;
    return errorH1 ;
}



template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type >
T Linf_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree )
{
    T errorLinf = ((level_set_final.vertices - level_set_initial.vertices).cwiseAbs()).maxCoeff();
    return errorLinf;

}

template< typename Mesh , typename Fonction , typename T = typename Mesh::coordinate_type >
T W1inf_error_FEM( const Fonction& level_set_final , const Fonction& level_set_initial , const Mesh& msh , size_t degree )
{
    T errorLinf0 = 0 , errorLinf1 = 0;
    
    for(auto& cl : msh.cells)
    {
        auto pts = points(msh,cl);
        for (auto& pt : pts )
        {
            auto diff_val0 = std::abs( level_set_final.gradient( pt , msh , cl )(0) - level_set_initial.gradient( pt , msh , cl )(0) );
            errorLinf0 = std::max(errorLinf0 , diff_val0);
            auto diff_val1 = std::abs( level_set_final.gradient( pt , msh , cl )(1) - level_set_initial.gradient( pt , msh , cl )(1) );
            errorLinf1 = std::max(errorLinf1 , diff_val1);
        }
        
    }
    return errorLinf0 + errorLinf1;

}


template< typename T , typename VeloField>
T time_step_CFL( const VeloField& u , const mesh_init_params<T>& mip , T eps ){
    
    auto h_max = std::max(mip.hx() , mip.hy() );
    auto u_max = u.values_bis.first.template lpNorm<Infinity>() + u.values_bis.second.template lpNorm<Infinity>() ;
    if( std::abs(u_max) < 1e-15 )
        return 1e-8;
    else
        return eps*h_max/u_max ;
}

#if 0
int main(int argc, char **argv)
{
    using RealType = double;

    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t degree_FEM       = 0;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    RealType d = 0.5;
    /* k <deg>:     method degree
     * g<deg>:  method FEM degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:q:c:M:N:r:ifDAd")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;
                
            case 'c':
                d = 1./atoi(optarg);;
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;
            
            

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    
    
    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    typedef cuthho_poly_mesh<RealType> Mesh;
    offset_definition(msh);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
       
    
    
    
    /************** FINITE ELEMENT INITIALIZATION **************/
    auto fe_data = Finite_Element<RealType>( msh.nodes.size() , degree_FEM , mip ) ;
    
    
    /************** ANALYTIC LEVEL SET FUNCTION  **************/
    RealType radius = 1.0/3.0; // I PUT 8.0
    auto level_set_function_anal = circle_level_set<RealType>(radius, 0.5, 0.5);
    //std::cout << bold << yellow << "Initial Area: "<< M_PI*radius*radius  << reset << std::endl;
    
    // auto level_set_function_anal = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);//(0.11, 0.1, 0.1, 4, 0.04);
    //  auto level_set_function_anal = square_level_set<RealType>(0.6,0.4,0.4,0.6);
    
    typedef RealType T;
    typedef  circle_level_set<T> Fonction;
   // typedef  flower_level_set<T> Fonction; //DA RIMETTERE POI PER LA DISCRETA
    //typedef line_level_set<T> Fonction;
    // typedef square_level_set<T> Fonction;
    
       
    
    /************** ANALYTIC LEVEL SET FUNCTION, OLD IMPLEMENTATION **************/
    //auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    //auto level_set_function = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);
    // auto level_set_function = line_level_set<RealType>(0.5);


    /************** ANALYTIC VELOCITY FIELD   **************/
    auto u = linear_velocity_field<RealType>(0,3,0,0); // analytic velocity (0,0,0,0)
    typedef linear_velocity_field<RealType> Velocity;
   // auto u = rotational_velocity_field<RealType>(1);
   // typedef rotational_velocity_field<RealType> Velocity;
  
    auto u_projected = projection_velocity< T, Mesh,Velocity>(u, msh, degree_FEM , mip);
          
    /************** LEVEL SET FUNCTION DISCRETISATION **************/
    std::cout<<"degree FEM "<<degree_FEM<<std::endl;
    auto level_set_function = projected_level_set< T, Mesh,Fonction>(level_set_function_anal, msh, degree_FEM , mip);
     
    // T d = 0.5;
    level_set_function.cut_off(d);
    //testing_level_set(msh,level_set_function,level_set_function_anal);
     
     
    // ************** Initialization of meshes-storage **************
    auto msh_storage =  Current_Mesh<Mesh>(msh);
     
    //************ DO cutHHO MESH PROCESSING **************
    tc.tic();
    detect_node_position2(msh, level_set_function); // IN cuthho_geom->da modificare operator() ->problema che giro su n:nodes
    detect_cut_faces2(msh, level_set_function); // IN cuthho_geom->da modificare operator() ->problema che giro su fc:faces

        //std::cout<<"Number of cells "<<msh.cells.size()<<std::endl;
        //std::cout<<"Number of nodes "<<msh.nodes.size()<<std::endl;
        //std::cout<<"Number of faces "<<msh.faces.size()<<std::endl;
        //std::cout<<"Number of points "<<msh.points.size()<<std::endl;
    
   
    T area0 = 0. , mass0 = 0. , global_mass0 = 0. ;
    if (agglomeration)
    {
           
        detect_cut_cells2(msh, level_set_function); // IN cuthho_geom

        detect_cell_agglo_set(msh, level_set_function); // Non serve modificarla
            //std::cout << bold << yellow << "before make_neighbors_info_cartesian"<< reset << std::endl;
        make_neighbors_info_cartesian(msh); // Non serve modificarla
            /*
             for (auto&cl:msh.cells) {
             auto tmp =  cl.user_data.f_neighbors;
             auto tmp1 =  cl.user_data.d_neighbors;
             std::cout<<"Cella numero "<<offset(msh,cl)<<std::endl;
             for (auto& tm:tmp) {
                std::cout<<"Friend face neigh cells "<<tm;
             }
             for (auto& tm1:tmp1) {
                std::cout<<" , friend diag neigh cells "<<tm1;
             }
             std::cout<<std::endl;
               
             }
             */
           
        refine_interface2(msh, level_set_function, int_refsteps); // IN cuthho_geom->da modificare operator() ->problema se esco da cella
            //std::cout << bold << yellow << "after refine_interface2"<< reset << std::endl;
            //for (auto& cl : msh.cells)
            //     {
            //       check_negative_weights<T,Mesh>( msh, cl, int_refsteps);
                
            //  }
        
        for(auto& cl : msh.cells){
            if( location(msh, cl) == element_location::IN_NEGATIVE_SIDE || location(msh, cl) == element_location::ON_INTERFACE )
            {
                T partial_area = measure( msh, cl, element_location::IN_NEGATIVE_SIDE);
                area0 += partial_area;
                auto qps=integrate(msh,cl,2*degree_FEM+1,element_location::IN_NEGATIVE_SIDE);
                for(auto& qp:qps)
                    mass0 += qp.second * level_set_function(qp.first,msh,cl);
                if( location(msh, cl) == element_location::ON_INTERFACE )
                {
                    auto qps1 = integrate(msh,cl,2*degree_FEM+1,element_location::IN_POSITIVE_SIDE);
                    for(auto& qp1:qps1)
                        global_mass0 += qp1.second * level_set_function(qp1.first,msh,cl);
                }
                //std::cout<<"NEGATIVE area in cl"<<offset(msh,cl)<<" = "<<partial_area<<std::endl;
            }
            else
            {
                auto qps2 = integrate( msh, cl, 2*degree_FEM+1 ,element_location::IN_POSITIVE_SIDE);
                for(auto& qp2:qps2)
                    global_mass0 += qp2.second * level_set_function(qp2.first,msh,cl);
            }
        }
        global_mass0 += mass0;
        //std::cout<<"area is "<<area<<std::endl;
        std::cout << bold << yellow << "Area 0: "<< area0  << reset << std::endl;
        std::cout << bold << yellow << "Initial Mass : "<< mass0  << reset << std::endl;
        std::cout<<bold<<yellow << "Initial GLOBAL Mass : "<<global_mass0<< reset << std::endl;
        
        
        /************** FEM -  PROCESSING **************/
        
        T eps = 0.5 ; // factor to be inside CFL stability zone
        auto dt = time_step_CFL( u_projected , mip , eps );
        std::cout<<"dt is "<<dt<<std::endl;
        //for (size_t jjj = 0; jjj<3; jjj++) {
        // PASSO level_set_function o LS_cell??
        
        output_mesh_info2_pre_FEM(msh, level_set_function); // IN cuthho_export.. Points/Nodes doesn't change
        
        std::cout<<"CASO NO TEMPO!!!!!!!"<<std::endl;
        run_FEM_levelset( level_set_function.msh, degree_FEM, level_set_function ,u_projected , dt , mip);
        //}
        
        // Added to upload the mesh:
        detect_node_position2(msh, level_set_function); // IN cuthho_geom->da modificare operator() ->problema che giro su n:nodes
        detect_cut_faces2(msh, level_set_function); // IN cuthho_geom->da modificare operator()
        
        detect_cut_cells2(msh, level_set_function); // IN cuthho_geom

        detect_cell_agglo_set(msh, level_set_function); // Non serve modificarla
           
        make_neighbors_info_cartesian(msh); // Non serve modificarla
           
        refine_interface2(msh, level_set_function, int_refsteps);
        
        
        
        
        
        
            
        T  mass_fin = 0. , global_mass1 = 0. ;
        for(auto& cl : msh.cells){
            if( location(msh, cl) == element_location::IN_NEGATIVE_SIDE || location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto qps=integrate(msh,cl,2*degree_FEM+1,element_location::IN_NEGATIVE_SIDE);
                for(auto& qp:qps)
                    mass_fin += qp.second * level_set_function(qp.first,msh,cl);
                if( location(msh, cl) == element_location::ON_INTERFACE )
                {
                    auto qps1 = integrate(msh,cl,2*degree_FEM+1,element_location::IN_POSITIVE_SIDE);
                    for(auto& qp1:qps1)
                        global_mass1 += qp1.second * level_set_function(qp1.first,msh,cl);
                }
            //std::cout<<"NEGATIVE area in cl "<<offset(msh,cl)<<" = "<<partial_area<<std::endl;
            }
            else{
                auto qps2 = integrate(msh,cl,2*degree_FEM+1,element_location::IN_POSITIVE_SIDE);
                for(auto& qp2:qps2)
                    global_mass1 += qp2.second * level_set_function(qp2.first,msh,cl);
            }
        }
        global_mass1 += mass_fin;
        std::cout << bold << yellow << "Final Mass : "<< mass_fin  << reset << std::endl;
        std::cout << bold << yellow << "Final GLOBAL Mass : "<< global_mass1  << reset << std::endl;
        std::cout << bold << yellow << "Final MASS - Initial MASS : "<< mass_fin - mass0  << reset << std::endl;
        std::cout << bold << yellow << "GLOBAL Final - Initial MASS : "<< global_mass1 - global_mass0  << reset << std::endl;
         
        T area_fin = 0. ;
        for(auto& cl : msh.cells){
            if( location(msh, cl) == element_location::IN_NEGATIVE_SIDE || location(msh, cl) == element_location::ON_INTERFACE ){
                T partial_area = measure( msh, cl, element_location::IN_NEGATIVE_SIDE);
                area_fin += partial_area;
                   //std::cout<<"NEGATIVE area in cl "<<offset(msh,cl)<<" = "<<partial_area<<std::endl;
            }
        }
                   //std::cout<<"area is "<<area<<std::endl;
        std::cout << bold << yellow << "Final Area MESH NO AGGLO : "<< area_fin  << reset << std::endl;
        std::cout << bold << yellow << "Final Area - Initial Area (MESH NO AGGLO): "<< area_fin - area0  << reset << std::endl;
        
        testing_level_set(msh,level_set_function,level_set_function_anal);

     
        make_agglomeration(msh, level_set_function); // Non serve modificarla
        //std::cout << bold << yellow << "after make_agglomeration"<< reset << std::endl;
    }
    else
    {
        move_nodes(msh, level_set_function);
        detect_cut_faces2(msh, level_set_function); //do it again to update intersection points
        detect_cut_cells2(msh, level_set_function);
        refine_interface2(msh, level_set_function, int_refsteps);
    }
       
    tc.toc();
    std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

    // NON CREDO SI ENTRI MAI QUA DENTRO..
    if (dump_debug)
    {
        dump_mesh(msh);
        output_mesh_info(msh, level_set_function);
        test_projection(msh, level_set_function, degree);
    }
       
    output_mesh_info2(msh, level_set_function); // IN cuthho_export.. Points/Nodes doesn't change
   
    postprocess_output<double> postoutput3;
    auto cut_cell_disegno = std::make_shared< gnuplot_output_object<double> >("cut_cell_disegno.dat");
    for (auto cl:msh.cells) {
        if( is_cut(msh,cl) )
        {
        // std::cout<<" the cell in the new mesh num "<<offset(msh,cl)<<" is cut."<<std::endl;
            // std::cout<<" EQuivalent to the old cells numero: ";
            auto pts = points(msh,cl) ;
            for (auto pt:pts) {
                T value = 1.;
                cut_cell_disegno->add_data(pt,value);
            }
        }
    }

    postoutput3.add_object(cut_cell_disegno);
    
    postoutput3.write();
   
    T area_fin = 0. ;
    for(auto& cl : msh.cells){
        if( location(msh, cl) == element_location::IN_NEGATIVE_SIDE || location(msh, cl) == element_location::ON_INTERFACE ){
            T partial_area = measure( msh, cl, element_location::IN_NEGATIVE_SIDE);
            area_fin += partial_area;
            //std::cout<<"NEGATIVE area in cl "<<offset(msh,cl)<<" = "<<partial_area<<std::endl;
        }
    }
            //std::cout<<"area is "<<area<<std::endl;
    std::cout << bold << yellow << "Final Area : "<< area_fin  << reset << std::endl;
    std::cout << bold << yellow << "Final Area - Initial Area : "<< area_fin - area0  << reset << std::endl;
              
    // jumps sin_sin -> exp_cos
    // auto test_case = make_test_case_laplacian_jumps_1(msh, level_set_function);
    // jumps3 sin_sin -> sin_sin + pol
    
    msh_storage.set_agglomesh(msh);
    typedef projected_level_set< T, Mesh,Fonction> Level_Set;
    auto ls_cell = LS_cell< T , Mesh , Level_Set, Fonction >(level_set_function,msh);
    
    auto current_cell =  Current_Cell<Mesh>();
    
    //auto test_case = make_test_case_laplacian_jumps_3bis(msh, ls_cell,current_cell);
    auto test_case = make_test_case_laplacian_jumps_3(msh, ls_cell); // 2 possibilità.. 1) creo 3bis 2) anzichè passarle level_set_function le passo LS_cell

    // auto method = make_Nitsche_interface_method(msh, 1.0, test_case);
    // auto method = make_sym_gradrec_interface_method(msh, 1.0, test_case);
    // auto method = make_gradrec_interface_method(msh, 1.0, test_case);
    auto method = make_Nitsche_interface_method_2(msh, 1.0, test_case);
    //Matrix<RealType, Dynamic, 1> sol;
    if(solve_interface){
        run_cuthho_interface2(msh, degree, method,test_case, ls_cell);
    //run_cuthho_interface(msh, degree, method, test_case);
    }
    if (solve_fictdom)
        run_cuthho_fictdom(msh, degree, test_case);

    //sol =  run_cuthho_interface(msh, degree, method, test_case);
    return 0;

}
#endif






void convergence_test_time()
{
    using T = double;

    std::vector<size_t> mesh_sizes, pol_orders, FE_orders ;
    std::vector<T> time_steps;
    
    T cut_off_level = 1.0/27.0 ;
    // meshes
    mesh_sizes.push_back(12);
    mesh_sizes.push_back(24);
    mesh_sizes.push_back(48);
    //mesh_sizes.push_back(64);
    //mesh_sizes.push_back(128);
    // mesh_sizes.push_back(256);

    // polynomial orders
    //pol_orders.push_back(0);
    //pol_orders.push_back(1);
    pol_orders.push_back(2);
    //pol_orders.push_back(3);
    

    // Finite Element orders for Level Set
    //FE_orders.push_back(0);
    FE_orders.push_back(1);
    //FE_orders.push_back(2);
    //FE_orders.push_back(3);
    
    // Time steps to be analysed // HAS TO be smaller than ||u||/max(h) = 1/32 = 0.03
    //time_steps.push_back(1*1e-2);
    time_steps.push_back(1*1e-2);
    time_steps.push_back(5*1e-3);
    time_steps.push_back(2.5*1e-3);

    T final_T = 0.05;
    
    // export to files ...
    std::vector<std::string> files;
    files.push_back("./output/test_dt0.txt");
    files.push_back("./output/test_dt1.txt");
    files.push_back("./output/test_dt2.txt");
    files.push_back("./output/test_dt3.txt");
    files.push_back("./output/test_time_conv.txt");
    files.push_back("./output/test_time_errors.txt");
    
    Matrix<T, Dynamic, 4> error_mat = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( time_steps.size()*mesh_sizes.size() , 4 ) ;
    size_t which_time = 0;
    T previous_dt = 0;
    T dt = 0;
    
    std::vector<T> error_finT_L1 , error_finT_L2 ;
    
    for (std::vector<T>::iterator it = time_steps.begin(); it != time_steps.end(); it++)
    {
        T dt2 = *it;
        int T_N = final_T/dt2 - 1;
        std::cout << "start tests for dt = " << dt2 << " , number of steps: "<< T_N << std::endl;
        
         
        
        // init the file
        std::ofstream output;
        output.open (files.at(which_time), std::ios::in | std::ios::trunc);
        if (!output.is_open())
            throw std::logic_error("file not open");

        // output << "N\th\tH1\tordre1\tL2\tordre2" << std::endl;
        //output << "N\th\tH1\tordre1\tL2\tordre2\tcond" << std::endl;
        //output << "N\th\tdt\tL1\tordre1\tL2\tordre2" << std::endl;
        output << "N\th\tdt\tl1(L1)\tordre1\tl2(L1)\tordre21\tl1(L2)\tordre2\tl2(L2)\tordre22"
                << std::endl;
        // convergence tests
        T previous_l1_L1 = 0.0 , previous_l2_L1 = 0.0;
        T previous_l1_L2 = 0.0 , previous_l2_L2 = 0.0;
        T previous_h = 0.0;
        size_t which_mesh = 0;
        
        std::vector<std::vector<T>> L1_err_vec(mesh_sizes.size()) , L2_err_vec(mesh_sizes.size()) ;
        
        for (std::vector<size_t>::iterator it_msh = mesh_sizes.begin();
             it_msh != mesh_sizes.end(); it_msh++)
        {
            size_t N = *it_msh;
             std::cout << "Mesh Size is = " << N << std::endl;
            // init mesh (with agglomeration)
            mesh_init_params<T> mip;
            mip.Nx = N;
            mip.Ny = N;
            cuthho_poly_mesh<T> msh(mip);
            offset_definition(msh);
            size_t int_refsteps = 4; // CHECK IT!
            T radius = 1.0/9.0;
            T x_centre = 0.45;
            T y_centre = 0.5;
            auto level_set_function_anal = circle_level_set<T>(radius, x_centre, y_centre);
    
            

            // auto level_set_function = flower_level_set<T>(0.31, 0.5, 0.5, 4, 0.04);
            // auto level_set_function = circle_level_set<T>(radius, 0.5, 0.5);
            // auto level_set_function = square_level_set<T>(1.05, -0.05, -0.05, 1.05);
            // auto level_set_function = square_level_set<T>(1.0, -0.0, -0.0, 1.0);
            // auto level_set_function = square_level_set<T>(0.76, 0.24, 0.24, 0.76);
           // auto level_set_function = square_level_set<T>(0.751, 0.249, 0.249, 0.751);
            
            
           // typedef RealType T;
            typedef  circle_level_set<T> Fonction;
            typedef cuthho_poly_mesh<T> Mesh;
            //typedef  flower_level_set<T> Fonction; //DA RIMETTERE POI PER LA DISCRETA
            //typedef line_level_set<T> Fonction;

             /************** LEVEL SET FUNCTION DISCRETISATION **************/
            size_t degree_FEM = FE_orders[0];
            std::cout<<"degree FEM "<<degree_FEM<<std::endl;
            auto level_set_function = projected_level_set< T, Mesh,Fonction>(level_set_function_anal, msh, degree_FEM , mip);
            std::cout<<"Cut off activated "<<std::endl;
            //level_set_function.cut_off(cut_off_level);
            T C = 0.2;//level_set_function.vertices.maxCoeff();
            std::cout<<"C is "<<C<<std::endl;
            level_set_function.smooth_cut_off( C , x_centre , y_centre , radius );
            
            
            /************** ANALYTIC VELOCITY FIELD   **************/
            T u_0 = 1. ;
            T u_1 = 0. ;
            auto u = linear_velocity_field<T>(0,u_0,0,u_1); // analytic velocity (0,0,0,0)
            typedef linear_velocity_field<T> Velocity;
            //auto u = rotational_velocity_field<RealType>( x_centre , y_centre , 1.0);
            //typedef rotational_velocity_field<RealType> Velocity;
            auto u_projected = projection_velocity< T, Mesh,Velocity>(u, msh, degree_FEM , mip);
            //testing_velocity( msh , u_projected , u );
            
            
            auto crr_mesh =  Current_Mesh<Mesh>(msh);
            
            // Error Initialisation
            T l1_L1_error = 0 , l2_L1_error = 0 ;
            T l1_L2_error = 0 , l2_L2_error = 0 ;
    
            for (size_t time_step = 0; time_step<=T_N; time_step++)
            {
                std::cout<<"Beginning of time step "<<dt*time_step<<std::endl;
                // ************** Re-Initialization mesh **************
                /*
                crr_mesh.current_mesh = msh;
                Mesh msh_i =  crr_mesh.current_mesh;
                offset_definition(msh_i);

            
                detect_node_position2(msh_i, level_set_function); // In cuthho_geom
                detect_cut_faces2(msh_i, level_set_function); // In cuthho_geom

                if(1)  // AGGLOMERATION
                {
                    detect_cut_cells2(msh_i, level_set_function); // In cuthho_geom
                    detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
                    make_neighbors_info_cartesian(msh_i); // Non serve modificarla
                    refine_interface2(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
                    make_agglomeration(msh_i, level_set_function); // Non serve modificarla
                    //detect_cut_cells2(msh, level_set_function);
                    //detect_cell_agglo_set(msh, level_set_function);
                    //make_neighbors_info_cartesian(msh);
                    //refine_interface2(msh, level_sChamps-sur-Marneet_function, int_refsteps);
                    //make_agglomeration(msh, level_set_function);
                }
                else  // NODE DISPLACEMENT
                {
                    move_nodes(msh_i, level_set_function);
                    detect_cut_faces2(msh_i, level_set_function); //do it again to update intersection points
                    detect_cut_cells2(msh_i, level_set_function);
                    refine_interface2(msh_i, level_set_function, int_refsteps);
                }
                */
                /************** FEM -  PROCESSING **************/
                
                T eps = 1 ; // factor to be inside CFL stability zone
                T dt1 = time_step_CFL( u_projected , mip , eps );
                dt = std::min(dt1 , dt2);
                std::cout<<"Dt = "<<dt<<" . CFL is "<<dt1<<std::endl;
                run_FEM_levelset( level_set_function.msh , degree_FEM , level_set_function , u_projected , dt , mip ); //-> make faster
                
                // Uploading mesh data to check out differences in mass and areas
                /*
                crr_mesh.current_mesh = msh;
                Mesh msh_i2 =  crr_mesh.current_mesh;
                offset_definition(msh_i2);
                detect_node_position2(msh_i2, level_set_function); // In cuthho_geom
                detect_cut_faces2(msh_i2, level_set_function); // In cuthho_geom
                if (1)
                {
                    detect_cut_cells2(msh_i2, level_set_function); // In cuthho_geom
                    detect_cell_agglo_set(msh_i2, level_set_function); // Non serve modificarla
                    make_neighbors_info_cartesian(msh_i2); // Non serve modificarla
                    refine_interface2(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                    make_agglomeration(msh_i2, level_set_function); // Non serve modificarla
                }
                // Uploading level set
                ls_cell.level_set = level_set_function;
                ls_cell.agglo_msh = msh_i2;
                */
                
                //************ Computation of exact solution + errors  ****************//
                
                T x_deviation = u_0*(time_step+1)*dt;
                T y_deviation = u_1*(time_step+1)*dt;
                auto analytic_level_set_final = circle_level_set<T>(radius, x_centre + x_deviation, y_centre + y_deviation );
                auto level_set_final = projected_level_set< T, Mesh,Fonction > (analytic_level_set_final, msh, degree_FEM , mip);
                //level_set_final.cut_off(cut_off_level);
                //T C = 0.2;//level_set_function.vertices.maxCoeff();
                std::cout<<"C is "<<C<<std::endl;
                level_set_final.smooth_cut_off( C , x_centre + x_deviation, y_centre + y_deviation, radius );
                
                T L1_err_par = Lp_space_error_FEM( level_set_final,level_set_function,msh, degree_FEM , 1.0 );
                T L2_err_par = Lp_space_error_FEM( level_set_final , level_set_function,msh, degree_FEM , 2.0);
                
                // l^q in time: norm in time
                l1_L1_error += dt*L1_err_par;
                l2_L1_error += dt*L1_err_par*L1_err_par;
                l1_L2_error += dt*L2_err_par;
                l2_L2_error += dt*L2_err_par*L2_err_par;
                
                L1_err_vec[which_mesh].push_back(L1_err_par);
                L2_err_vec[which_mesh].push_back(L2_err_par);
                
            }
            l2_L1_error = sqrt(l2_L1_error);
            l2_L2_error = sqrt(l2_L2_error);
            
            /************** ANALYTIC SOLUTION FOR FINAL TIME LINEAR VELOCITY FIELD   **************/
            T x_deviation = u_0*(T_N+1)*dt;
            T y_deviation = u_1*(T_N+1)*dt;
            auto analytic_level_set_final = circle_level_set<T>(radius, x_centre + x_deviation, y_centre + y_deviation );
            auto level_set_final = projected_level_set< T, Mesh,Fonction>(analytic_level_set_final, msh, degree_FEM , mip);
            //level_set_final.cut_off(cut_off_level);
            std::cout<<"C is "<<C<<std::endl;
            level_set_final.smooth_cut_off( C , x_centre + x_deviation , y_centre + y_deviation, radius );
            
            Lp_space_Tfin_error_FEM( level_set_final , level_set_function , msh ,degree_FEM ,1.0 ,error_finT_L1);
            Lp_space_Tfin_error_FEM( level_set_final , level_set_function , msh ,degree_FEM ,2.0 ,error_finT_L2);
           /*
            if(1) // jumps sin_sin -> exp_cos
            {
                auto test_case = make_test_case_laplacian_jumps_1(msh, level_set_function);
                auto meth3 = make_gradrec_interface_method(msh, 1.0, test_case);
                TI = run_cuthho_interface(msh, k, meth3, test_case);
            }
           */


            // report info in the file
            T h = 1.0/N;
            if (it_msh == mesh_sizes.begin())
            {
                
                // output << N << "\t" << h << "\t" << TI.H1 << "\t" << "."
                //        << "\t" << TI.L2 << "\t" << "." << "\t" << "0.0"
                //        << std::endl;
                output << N << "\t" << h << "\t" << dt << "\t" << l1_L1_error << "\t" << "." << "\t"
                       << l2_L1_error << "\t" << "." << "\t" << l1_L2_error << "\t"
                       << "." << "\t" << l2_L2_error << "\t" << "." << std::endl;
                       
            }
            else
            {
                T orderl1_L1 = log(previous_l1_L1 / l1_L1_error) / log(previous_h / h);
                T orderl2_L1 = log(previous_l2_L1 / l2_L1_error) / log(previous_h / h);
                T orderl1_L2 = log(previous_l1_L2 / l1_L2_error) / log(previous_h / h);
                T orderl2_L2 = log(previous_l2_L2 / l2_L2_error) / log(previous_h / h);
                //T orderL = log(previous_L2 / TI.L2) / log(previous_h / h);
                // output << N << "\t" << h << "\t" << TI.H1 << "\t" << orderH
                //        << "\t" << TI.L2 << "\t" << orderL << "\t" << "0.0"
                //        << std::endl;
                output << "N\th\tdt\tl1(L1)\tordre1\tl2(L1)\tordre21\tl1(L2)\tordre2\tl2(L2)\tordre22"
                << std::endl;
                output << N << "\t" << h << "\t" << dt << "\t" << l1_L1_error << "\t" << orderl1_L1
                       << "\t" << l2_L1_error << "\t" << orderl2_L1 << "\t"
                       << l1_L2_error << "\t" << orderl1_L2 << "\t" << l2_L2_error << "\t" << orderl2_L2 << "\t" << std::endl;
            }
            previous_l1_L1 = l1_L1_error;
            previous_l2_L1 = l2_L1_error;
            previous_l1_L2 = l1_L2_error;
            previous_l2_L2 = l2_L2_error;
            //previous_L2 = TI.L2;
            previous_h = h;
            error_mat(which_time*mesh_sizes.size() +which_mesh,0) = l1_L1_error ;
            error_mat(which_time*mesh_sizes.size()+which_mesh,1) = l2_L1_error ;
            error_mat(which_time*mesh_sizes.size()+which_mesh,2) = l1_L2_error ;
            error_mat(which_time*mesh_sizes.size()+which_mesh,3) = l2_L2_error ;
            
           
            if(which_mesh == 0 )
                testing_level_set_mesh0(msh,level_set_function,level_set_final);
            if(which_mesh == 1 )
                testing_level_set_mesh1(msh,level_set_function,level_set_final);
            if(which_mesh == 2 )
                testing_level_set_mesh2(msh,level_set_function,level_set_final);
            
            
            
            which_mesh ++ ;
        }
        // close the file
        output.close();
        
       // std::cout<<"error_mat size: rows"<<error_mat.rows()<<" , cols "<<error_mat.cols()<<std::endl;
       // std::cout<<"error_mat"<<'\n'<<error_mat<<std::endl;
        
        // init the file
        std::ofstream output3;
        //output3.open (files.at(4), std::ios::in | std::ios::trunc);
        output3.open (files.at(5), std::ios::out  | std::ios::app);// | std::ios::trunc);
        if (!output3.is_open())
            throw std::logic_error("file not open");
        output3 << '\n';
        output3 << "tdt\terrorL1N8\terrorL2N8\terrorL1N16\terrorL2N16\terrorL1N32\terrorL2N32\ttime" << std::endl;
        for(size_t pos = 0 ; pos<L1_err_vec[0].size();pos++)
        {
            T time_pos = dt*(pos+1);
            output3  << dt ;
            size_t i_vec = 0;
            while( i_vec < mesh_sizes.size() )
            {
            output3 << "\t" << L1_err_vec.at(i_vec).at(pos) << "\t" << L2_err_vec.at(i_vec).at(pos);
            // << L1_err_vec.at(1).at(pos)  << "\t" << L2_err_vec.at(1).at(pos) << "\t" << L1_err_vec.at(2).at(pos)  << "\t" << L2_err_vec.at(2).at(pos) << "\t" << time_pos << std::endl;
                i_vec++;
            }
            output3 << std::endl;
            
        }
        output3 <<'\n'<<std::endl;
        output3.close();
        
        
        
        // init the file
        std::ofstream output2;
        output2.open (files.at(4), std::ios::out | std::ios::app);
           if (!output2.is_open())
               throw std::logic_error("file not open");

           // output << "N\th\tH1\tordre1\tL2\tordre2" << std::endl;
           //output << "N\th\tH1\tordre1\tL2\tordre2\tcond" << std::endl;
           //output << "N\th\tdt\tL1\tordre1\tL2\tordre2" << std::endl;
        
       
        if ( it == time_steps.begin())
        {
            output2 << "tdt\totimel1(L1)8\totimel2(L1)8\totimel1(L2)8\totimel2(L2)8\totimel1(L1)16\totimel2(L1)16\totimel1(L2)16\totimel2(L2)16\totimel1(L1)32\totimel2(L1)32\totimel1(L2)32\totimel2(L2)32" << std::endl;
           
            output2<< dt << "\t" << "." << "\t" << "." << "\t"
                    << "." << "\t" << "." << "\t" << "." << "\t" << "." << "\t"
                    << "." << "\t" << "." << "\t" << "." << "\t" << "." << "\t"
                    << "." << "\t" << "."  << std::endl;
                   
        }
        else
        {
            /*
            Matrix<T, 1, 4> error_conv8 = Eigen::Matrix<T, 1, 4>::Zero( 1 , 4 ) ;
            Matrix<T, 1, 4> error_conv16 = Eigen::Matrix<T, 1, 4>::Zero( 1 , 4 ) ;
            Matrix<T, 1, 4> error_conv32 = Eigen::Matrix<T, 1, 4>::Zero( 1 , 4 ) ;
             */
            int num_msh = mesh_sizes.size();
            int num_time = which_time-1;
            size_t i_conv = 0;
            output2 << dt ;
            while(i_conv < num_msh)
            {
                Matrix<T, 1, 4> error_conv_time_i = error_mat.row(num_time*num_msh+i_conv).cwiseQuotient(error_mat.row(num_time*num_msh+num_msh+i_conv));
                
                error_conv_time_i = ( error_conv_time_i.array().log() ) / log( previous_dt / dt );
                output2 << "\t" << error_conv_time_i(0,0) << "\t" << error_conv_time_i(0,1)  << "\t" << error_conv_time_i(0,2)  << "\t" << error_conv_time_i(0,3) ;
                i_conv++;
            }
           output2 << std::endl;
            /*
            error_conv8 = error_mat.row(num_time*num_msh).cwiseQuotient(error_mat.row(num_time*num_msh+num_msh));
            error_conv8 = (error_conv8.array().log() ) / log(previous_dt / dt);
            error_conv16= error_mat.row(num_time*num_msh+1).cwiseQuotient(error_mat.row(num_time*num_msh+num_msh+1));
            error_conv16= (error_conv16.array().log() ) / log(previous_dt / dt);
            error_conv32 = error_mat.row(num_time*num_msh+2).cwiseQuotient(error_mat.row(num_time*num_msh+num_msh+2));
            error_conv32 = (error_conv32.array().log() ) / log(previous_dt / dt);
             std::cout<<"I m here 3."<<std::endl;

          //  std::cout<<"Error conv"<<error_conv8<<std::endl;
          //  std::cout<<"Error conv"<<error_conv16<<std::endl;
          //  std::cout<<"Error conv"<<error_conv32<<std::endl;
            output2 << dt << "\t" << error_conv8(0,0) << "\t" << error_conv8(0,1)  << "\t" << error_conv8(0,2)  << "\t" << error_conv8(0,3)<< "\t" << error_conv16(0,0) << "\t" << error_conv16(0,1)  << "\t" << error_conv16(0,2)  << "\t" << error_conv16(0,3)  << "\t" << error_conv32(0,0) << "\t" << error_conv32(0,1) << "\t" << error_conv32(0,2) << "\t" << error_conv32(0,3) << std::endl;
            */
            output2.close();
        }
        previous_dt = dt;
        which_time ++ ;
        
        
       // if(it == time_steps.end()-1)
      //      testing_level_set2(msh,level_set_final,level_set_function);
        
        
    }
    
   
    
    // update the gnuplot curves
    system("gnuplot './output/gnuplot_script.txt'");

    // update the .pdf file
    system("pdflatex ./output/autom_tests.tex");

    // open the .pdf file
    system("xdg-open ./autom_tests.pdf");
}





// Velocity is analytical: test cases linear and rotational

#if 0
int main(int argc, char **argv)
{
    using RealType = double;

    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t degree_FEM       = 0;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    RealType d = 0.5;
    size_t T_N = 0;
    /* k <deg>:     method degree
     * g<deg>:  method FEM degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:q:c:M:N:r:T:ifDAd")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;
                
            case 'c':
                d = 1./atoi(optarg);
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;
            
            case 'T':
                T_N = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;
            
            

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    
    
    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    typedef cuthho_poly_mesh<RealType> Mesh;
    offset_definition(msh);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
       
    
    
    
    /************** FINITE ELEMENT INITIALIZATION **************/
    auto fe_data = Finite_Element<RealType>( msh.nodes.size() , degree_FEM , mip ) ;
    
    
    /************** ANALYTIC LEVEL SET FUNCTION  **************/
    RealType radius = 1.0/9.0; // I PUT 6.0, IT WAS 1.0/3.0
    RealType x_centre = 0.45;
    RealType y_centre = 0.5;
    auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre, y_centre);
    std::cout << bold << yellow << "Initial Analytic Area Circle: "<< M_PI*radius*radius << " , and circonference: "<< 2*M_PI*radius << reset << std::endl;
    
    
    /************** CELL BASIS PROVA  **************/
    testing_basis(msh , degree_FEM );
   
    
    // auto level_set_function_anal = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);//(0.11, 0.1, 0.1, 4, 0.04);
    //  auto level_set_function_anal = square_level_set<RealType>(0.6,0.4,0.4,0.6);
    
    typedef RealType T;
    typedef  circle_level_set<T> Fonction;
    //typedef  flower_level_set<T> Fonction; //DA RIMETTERE POI PER LA DISCRETA
    //typedef line_level_set<T> Fonction;
    // typedef square_level_set<T> Fonction;
    
       
    
    /************** ANALYTIC LEVEL SET FUNCTION, OLD IMPLEMENTATION **************/
    //auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    //auto level_set_function = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);
    // auto level_set_function = line_level_set<RealType>(0.5);


    /************** ANALYTIC VELOCITY FIELD   **************/
    T u_0 = 1. ;
    T u_1 = 0. ;
    //auto u = linear_velocity_field<RealType>(0,u_0,0,u_1); // analytic velocity (0,0,0,0)
    //typedef linear_velocity_field<RealType> Velocity;
    //auto u = rotational_velocity_field<RealType>( x_centre , y_centre , 1.0);
    //typedef rotational_velocity_field<RealType> Velocity;
    //auto u = stokes_field<RealType>();
    auto u = stokes_field<RealType>(4.0,2.0);
    //auto u = stokes_field<RealType>(4.0,2.0,2.0);
    typedef stokes_field<RealType> Velocity;
 //    std::cout<<"prob here 0  "<<std::endl;
    auto u_projected = projection_velocity< T, Mesh,Velocity>(u, msh, degree_FEM , mip);
//     std::cout<<"prob here 1 "<<std::endl;
    testing_velocity( msh , u_projected , u );

    
    
    /************** LEVEL SET FUNCTION DISCRETISATION **************/
    std::cout<<"degree FEM "<<degree_FEM<<std::endl;
    auto level_set_function = projected_level_set< T, Mesh,Fonction>(level_set_function_anal, msh, degree_FEM , mip);
     
    // T d = 0.5;
    //level_set_function.cut_off(d);
    T C = 0.2;//level_set_function.vertices.maxCoeff();
    std::cout<<"Smoothed solution, C  max value is "<< C <<std::endl;
    level_set_function.smooth_cut_off( C , x_centre , y_centre , radius );
    testing_level_set(msh,level_set_function,level_set_function_anal);
     
     
    auto crr_mesh =  Current_Mesh<Mesh>(msh);
    
    T area_previous_time = 0 , mass_previous_time = 0 , dt = 0 ;
    T diff_area = 0. , diff_mass = 0. , diff_global_mass = 0. ;
    T initial_area = 0. , initial_mass = 0.;
    T area0 = 0. , mass0 = 0. , global_mass0 = 0. ;
   
    T error_normal_global = 0. ;
    T error_normal_local = 0. ;
   
   
    /*
    T l1_L1_error = 0 , l2_L1_error = 0 , linf_L1_error = 0;
    T l1_L2_error = 0 ,  l2_L2_error = 0 , linf_L2_error = 0;
    T l1_Linf_error = 0 ,  l2_Linf_error = 0 , linf_Linf_error = 0;
    
    T l1_W11_error = 0 , l2_W11_error = 0 , linf_W11_error = 0;
    T l1_W12_error = 0 ,  l2_W12_error = 0 , linf_W12_error = 0;
    T l1_W1inf_error = 0 ,  l2_W1inf_error = 0 , linf_W1inf_error = 0;
    */
    
    for (size_t time_step = 0; time_step<=T_N; time_step++)
    {
    
        // ************** Re-Initialization mesh **************
        crr_mesh.current_mesh = msh;
        Mesh msh_i =  crr_mesh.current_mesh;
        offset_definition(msh_i);

        //************ DO cutHHO MESH PROCESSING **************
        tc.tic();
        detect_node_position2(msh_i, level_set_function); // In cuthho_geom
        detect_cut_faces2(msh_i, level_set_function); // In cuthho_geom
       
       
        if (agglomeration)
        {
            detect_cut_cells2(msh_i, level_set_function); // In cuthho_geom
            detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
            make_neighbors_info_cartesian(msh_i); // Non serve modificarla
            refine_interface2(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
            make_agglomeration(msh_i, level_set_function); // Non serve modificarla
        }
        else
        {
            move_nodes(msh_i, level_set_function);
            detect_cut_faces2(msh_i, level_set_function); //do it again to update intersection points
            detect_cut_cells2(msh_i, level_set_function);
            refine_interface2(msh_i, level_set_function, int_refsteps);
        }
       
        tc.toc();
        std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

        if (dump_debug)
        {
            dump_mesh(msh_i);
            output_mesh_info(msh_i, level_set_function);
            test_projection(msh_i, level_set_function, degree);
        }
       if(time_step == 0)
           output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export
        //output_mesh_info2(msh_i, level_set_function); // IN cuthho_export..Points/Nodes don't change
   
        typedef projected_level_set< T, Mesh,Fonction> Level_Set;
        auto ls_cell = LS_cell< T , Mesh , Level_Set, Fonction >(level_set_function,msh_i);
           
        
        // CALCULATION OF AREA AND MASS AT TIME STEP t^n
        for(auto& cl : msh_i.cells)
        {
            ls_cell.cell_assignment(cl);
            if( location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE || location(msh_i, cl) == element_location::ON_INTERFACE )
            {
                T partial_area = measure( msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area0 += partial_area;
                auto qps=integrate(msh_i,cl,2*degree_FEM+1,element_location::IN_NEGATIVE_SIDE);
                for(auto& qp:qps)
                    mass0 += qp.second * ls_cell(qp.first);
                if( location(msh_i, cl) == element_location::ON_INTERFACE )
                {
                    auto qps1 = integrate(msh_i,cl,2*degree_FEM+1,element_location::IN_POSITIVE_SIDE);
                    for(auto& qp1:qps1)
                        global_mass0 += qp1.second * ls_cell(qp1.first);
                }
            }
            else
            {
                auto qps2 = integrate(msh_i,cl,2*degree_FEM+1,element_location::IN_POSITIVE_SIDE);
                for(auto& qp2:qps2)
                    global_mass0 += qp2.second * ls_cell(qp2.first);
            }
        }
        global_mass0 += mass0;
        std::cout << bold << yellow << "Area at time step: "<<time_step*dt<<" is "<<reset<< area0  << reset << std::endl;
        std::cout << bold << yellow << "Internal mass at time step: "<<time_step*dt<<" is "<<reset<< mass0  << reset << std::endl;
        //std::cout<<bold<<yellow << "GLOBAL Mass at time step: "<<time_step<<" is "<<reset<<global_mass0<< reset << std::endl;
           
       
        
        
        if(time_step == 0 ){
            initial_area  = area0 ;
            initial_mass = mass0 ;
        }
       
        if(time_step > 0 )
        {
            diff_area = area0 - area_previous_time ;
            diff_mass = mass0 - mass_previous_time ;
            std::cout << bold << yellow << "Difference in Area (new - old) at time step: "<<time_step*dt<<" is "<<reset<< diff_area  << reset << std::endl;
            std::cout << bold << yellow << "Difference in internal MASS (new - old) at time step: "<<time_step*dt<<" is "<<reset<< diff_mass  << reset << std::endl;
            
           
        }
        
        /// Checking normal error
        T x_deviation = u_0*(time_step)*dt;
        T y_deviation = u_1*(time_step)*dt;
        auto analytic_level_set_final = circle_level_set<RealType>(radius, x_centre + x_deviation, y_centre + y_deviation );
        auto level_set_exact = projected_level_set< T, Mesh,Fonction>(analytic_level_set_final, msh, degree_FEM , mip);
        
        level_set_exact.smooth_cut_off( C , x_centre + x_deviation , y_centre +y_deviation,radius );
        
        auto ls_cell_exact = LS_cell< T , Mesh , Level_Set, Fonction >(level_set_exact,msh_i);
        
        // postprocess_output<double> postoutput_vec;
        //  auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("vec_normal_interface.dat");
        // auto vec_normal_exact = std::make_shared< gnuplot_output_object_vec<double> >("vec_normal_interface_EXACT.dat");
           
        T error_normal_global0 = 0. ;
        T diff_interface_normal = 0. ;
        size_t counter_interface_pts = 0;
        for(const auto & cl:msh_i.cells)
        {
            if(cl.user_data.location == element_location::ON_INTERFACE)
            {
                ls_cell.cell_assignment(cl);
                ls_cell_exact.cell_assignment(cl);
                auto qps=integrate(msh_i,cl,2*degree_FEM+1);
                for(auto& qp:qps){
                    Eigen::Matrix<T,2,1> diff_global_normal = (ls_cell.normal(qp.first) - ls_cell_exact.normal(qp.first) ).cwiseAbs();
                    error_normal_global0 += qp.second * (pow(diff_global_normal(0),2));
                    error_normal_global0 += qp.second * (pow(diff_global_normal(1),2));
                
                }
            
            /*
            for(auto& pt:points(msh_i,cl))
            {
                Eigen::Matrix<T,2,1> diff_global_normal = (ls_cell.normal(pt) - ls_cell_exact.normal(pt) ).cwiseAbs();
                   
                std::cout <<'\t' << diff_global_normal(0)<<" , "<< diff_global_normal(1) <<'\t';
            }
            
            std::cout<<std::endl;
            */
              
                // std::cout<<"Normal error in CUT CELL "<<offset(msh_i,cl)<<" is "<<'\n';
                
                for(auto& interface_point : cl.user_data.interface)
                {
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(interface_point);
                    Eigen::Matrix<T,2,1> normal_exact = ls_cell_exact.normal(interface_point);
                    
                    diff_interface_normal += (pow((normal(0)-normal_exact(0)),2) + pow((normal(1)-normal_exact(1)),2));
                    
                  //  std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                  //  vec_normal->add_data(interface_point,normal_vec);
                   // std::pair<T,T> normal_vec_exact = std::make_pair(normal(0),normal(1));
                   // vec_normal_exact->add_data(interface_point,normal_vec_exact);
                    
                    counter_interface_pts++;
                }
                std::cout<<std::endl;
            }
        }
      //  postoutput_vec.add_object(vec_normal);
      //  postoutput_vec.add_object(vec_normal_exact);
      //  postoutput_vec.write();
            
        error_normal_global0 = sqrt(error_normal_global0);
        
        std::cout<<"number of interface points is "<<counter_interface_pts<<std::endl;
        diff_interface_normal /= counter_interface_pts;
        diff_interface_normal = sqrt(diff_interface_normal);
        
        std::cout<<"The L2 error of normal N over the cut cells, at time "<< dt*time_step <<" is " << error_normal_global0 <<std::endl;
        std::cout<<"The l2 error of normal N over the INTERFACE, at time "<< dt*time_step <<" is " << diff_interface_normal <<std::endl;
        
        
        area_previous_time  = area0 ;
        mass_previous_time = mass0 ;
        
        error_normal_global += error_normal_global0*dt;
        error_normal_local += diff_interface_normal*dt;
         
        
        // CUT CELL PLOTTING
        /*
        postprocess_output<double> postoutput3;
        auto cut_cell_disegno = std::make_shared< gnuplot_output_object<double> >("cut_cell_disegno.dat");
        for (auto cl:msh_i.cells) {
            if( is_cut(msh_i,cl) )
            {
            // std::cout<<" the cell in the new mesh num "<<offset(msh,cl)<<" is cut."<<std::endl;
                // std::cout<<" EQuivalent to the old cells numero: ";
                auto pts = points(msh_i,cl) ;
                for (auto pt:pts) {
                    T value = 1.;
                    cut_cell_disegno->add_data(pt,value);
                }
            }
        }
        postoutput3.add_object(cut_cell_disegno);
        postoutput3.write();
        */
    
        // jumps sin_sin -> exp_cos
        // auto test_case = make_test_case_laplacian_jumps_1(msh, level_set_function);
        // jumps3 sin_sin -> sin_sin + pol
    
/*
        //auto test_case = make_test_case_laplacian_jumps_3bis(msh_i, ls_cell,current_cell);
        auto test_case = make_test_case_laplacian_jumps_3(msh_i, ls_cell);
        // 2 possibilità.. 1) creo 3bis 2) anzichè passarle level_set_function le passo LS_cell

        // auto method = make_Nitsche_interface_method(msh, 1.0, test_case);
        // auto method = make_sym_gradrec_interface_method(msh, 1.0, test_case);
        // auto method = make_gradrec_interface_method(msh, 1.0, test_case);
        auto method = make_Nitsche_interface_method_2(msh_i, 1.0, test_case);
        //Matrix<RealType, Dynamic, 1> sol;
        if(solve_interface){
            run_cuthho_interface2(msh_i, degree, method,test_case, ls_cell);
        //run_cuthho_interface(msh, degree, method, test_case);
        }
        if (solve_fictdom)
            run_cuthho_fictdom(msh_i, degree, test_case);
        
*/
        /************** FEM -  PROCESSING **************/
        
        T eps = 0.5 ; // factor to be inside CFL stability zone
        T dt1 = time_step_CFL( u_projected , mip , eps );
        //std::cout<<"dt1 is "<<dt1<<std::endl;
        T dt2 = 5*1e-3;
        dt = std::min(dt1 , dt2);
        std::cout<<"I USE dt = "<<dt<<" AND CFL IS "<<dt1<<std::endl;
        
     //   run_FEM_levelset( level_set_function.msh,degree_FEM,level_set_function,u_projected,dt,mip);
        run_FEM_levelset_SPARSE( level_set_function.msh ,degree_FEM, level_set_function,u_projected,dt,mip);
        
          // Uploading mesh data to check out differences in mass and areas at LAST TIME
        if( (T_N - time_step)==0 )
        {
            crr_mesh.current_mesh = msh;
            Mesh msh_i2 =  crr_mesh.current_mesh;
            offset_definition(msh_i2);
            tc.tic();
            detect_node_position2(msh_i2, level_set_function); // In cuthho_geom
            detect_cut_faces2(msh_i2, level_set_function); // In cuthho_geom
        
       
            if (agglomeration)
            {
           // std::cout<<"i m here 1"<<std::endl;
                detect_cut_cells2(msh_i2, level_set_function); // In cuthho_geom
           
                detect_cell_agglo_set(msh_i2, level_set_function); // Non serve modificarla
           
                make_neighbors_info_cartesian(msh_i2); // Non serve modificarla
                refine_interface2(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                make_agglomeration(msh_i2, level_set_function); // Non serve modificarla
            }
            else
            {
                move_nodes(msh_i2, level_set_function);
                detect_cut_faces2(msh_i2, level_set_function); //do it again to update intersection points
                detect_cut_cells2(msh_i2, level_set_function);
                refine_interface2(msh_i2, level_set_function, int_refsteps);
            }
        
            tc.toc();
            std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

            if (dump_debug)
            {
                dump_mesh(msh_i2);
                output_mesh_info(msh_i2, level_set_function);
                test_projection(msh_i2, level_set_function, degree);
            }
        
            output_mesh_info2(msh_i2, level_set_function); // IN cuthho_export..Points/Nodes don't change
        
            // Uploading level set
            ls_cell.level_set = level_set_function;
            ls_cell.agglo_msh = msh_i2;
        
       
            T  mass_fin = 0. , global_mass1 = 0. , area_fin = 0. ;
            for(auto& cl : msh_i2.cells)
            {
                ls_cell.cell_assignment(cl);
                if( location(msh_i2, cl) == element_location::IN_NEGATIVE_SIDE || location(msh_i2, cl) == element_location::ON_INTERFACE )
                {
                    T partial_area = measure( msh_i2, cl, element_location::IN_NEGATIVE_SIDE);
                    area_fin += partial_area;
                    auto qps=integrate(msh_i2,cl,2*degree_FEM+1,element_location::IN_NEGATIVE_SIDE);
                    for(auto& qp:qps)
                        mass_fin += qp.second * ls_cell(qp.first);
                    if( location(msh_i2, cl) == element_location::ON_INTERFACE )
                    {
                        auto qps1 = integrate(msh_i2,cl,2*degree_FEM+1,element_location::IN_POSITIVE_SIDE);
                        for(auto& qp1:qps1)
                            global_mass1 += qp1.second * ls_cell(qp1.first);
                    }
                }
                else
                {
                    auto qps2 = integrate(msh_i2,cl,2*degree_FEM+1,element_location::IN_POSITIVE_SIDE);
                    for(auto& qp2:qps2)
                        global_mass1 += qp2.second * ls_cell(qp2.first);
                }
            }
            global_mass1 += mass_fin;
            
            std::cout << bold << yellow << "Area at time step: " <<(time_step+1)*dt<<" is "<< reset<< area_fin   << std::endl;
           std::cout << bold << yellow << "Internal mass at time step: "<<(time_step+1)*dt<<" is "<<reset<< mass_fin  << reset << std::endl;
        //    std::cout<<bold<<yellow << "GLOBAL Mass at time step: "<<(time_step+1)<<" is "<<reset<<global_mass1<< reset << std::endl;
            
            std::cout << bold << yellow << "Difference in AREA between initial state and final TIME "<<(T_N+1)*(dt)<<" IS "<< reset<< area_fin - initial_area  << std::endl;
            std::cout << bold << yellow << "Difference in INTERNAL MASS between initial state and final TIME "<<(T_N+1)*(dt)<<" IS "<< reset<< mass_fin - initial_mass << std::endl;
          //  std::cout << bold << yellow << "Difference in GLOBAL MASS: "<< reset<< global_mass1 - global_mass0 << std::endl;
            
            
            
            
            
            /// Checking normal error
            x_deviation = u_0*(time_step+1)*dt;
            y_deviation = u_1*(time_step+1)*dt;
            
            
            
            auto analytic_level_set_final_exact = circle_level_set<RealType>(radius, x_centre + x_deviation, y_centre + y_deviation );
            auto level_set_exact_fin = projected_level_set < T, Mesh,Fonction > (analytic_level_set_final_exact, msh, degree_FEM , mip);
            
            level_set_exact_fin.smooth_cut_off( C , x_centre + x_deviation , y_centre + y_deviation , radius );
            
            auto ls_cell_exact_fin = LS_cell< T,Mesh,Level_Set,Fonction > (level_set_exact_fin,msh_i);
            
            
            postprocess_output<double> postoutput_vec;
            auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("vec_normal_interface.dat");
            auto vec_normal_exact = std::make_shared< gnuplot_output_object_vec<double> >("vec_normal_interface_EXACT.dat");
            error_normal_global0 = 0. ;
            diff_interface_normal = 0. ;
            counter_interface_pts = 0;
            for(const auto & cl:msh_i.cells)
            {
                if(cl.user_data.location == element_location::ON_INTERFACE)
                {
                    ls_cell.cell_assignment(cl);
                    ls_cell_exact_fin.cell_assignment(cl);
                    auto qps=integrate(msh_i,cl,2*degree_FEM+1);
                    for(auto& qp:qps){
                        Eigen::Matrix<T,2,1> diff_global_normal = (ls_cell.normal(qp.first) - ls_cell_exact_fin.normal(qp.first) ).cwiseAbs();
                        error_normal_global0 += qp.second * (pow(diff_global_normal(0),2));
                        error_normal_global0 += qp.second * (pow(diff_global_normal(1),2));
                    
                    }
            
                    for(auto& interface_point : cl.user_data.interface)
                    {
                        Eigen::Matrix<T,2,1> normal = ls_cell.normal(interface_point);
                        Eigen::Matrix<T,2,1> normal_exact = ls_cell_exact_fin.normal(interface_point);
                                
                        diff_interface_normal += (pow((normal(0)-normal_exact(0)),2) + pow((normal(1)-normal_exact(1)),2));
                                
                        std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                        vec_normal->add_data(interface_point,normal_vec);
                        std::pair<T,T> normal_vec_exact = std::make_pair(normal(0),normal(1));
                        vec_normal_exact->add_data(interface_point,normal_vec_exact);
                                
                        counter_interface_pts++;
                    }
                    std::cout<<std::endl;
                }
            }
            postoutput_vec.add_object(vec_normal);
            postoutput_vec.add_object(vec_normal_exact);
            postoutput_vec.write();
                        
            error_normal_global0 = sqrt(error_normal_global0);
                
            std::cout<<"number of interface points is "<<counter_interface_pts<<std::endl;
                    
            diff_interface_normal /= counter_interface_pts;
            diff_interface_normal = sqrt(diff_interface_normal);
                
            std::cout<<"The L2 error of normal N over the cut cells, at time "<< dt*(time_step+1) <<" is " << error_normal_global0 <<std::endl;
            std::cout<<"The l2 error of normal N over the INTERFACE, at time "<< dt*(time_step+1) <<" is " << diff_interface_normal <<std::endl;
                    
                    
                    
            error_normal_global += error_normal_global0*dt;
            error_normal_local += diff_interface_normal*dt;
                    
                
            std::cout << bold << yellow << "Error l1(L2) of the normal in the CUT CELLS: "<< reset<< error_normal_global  << std::endl;
            
            std::cout << bold << yellow << "Error l1(l2) of the normal in the INTERFACE POINTS: "<< reset<< error_normal_local  << std::endl;
           
        }
       // testing_level_set(msh,level_set_function,level_set_function_anal);
        /*
        T x_deviation = u_0*(time_step+1)*dt;
        T y_deviation = u_1*(time_step+1)*dt;
        auto analytic_level_set_final = circle_level_set<RealType>(radius, x_centre + x_deviation, y_centre + y_deviation );
        auto level_set_final = projected_level_set< T, Mesh,Fonction>(analytic_level_set_final, msh, degree_FEM , mip);
        
        
        
        level_set_final.smooth_cut_off( C , x_centre + x_deviation , y_centre +y_deviation , radius );
        //level_set_final.cut_off(d);
        testing_level_set2(msh,level_set_function,level_set_final);
            //testing_level_set(msh,level_set_function,analytic_level_set_final);
         
        T L1_err_par = Lp_space_error_FEM( level_set_final , level_set_function,msh,degree_FEM,1.0);
        
        T L2_err_par = Lp_space_error_FEM( level_set_final , level_set_function,msh,degree_FEM,2.0);
         
        T Linf_err_par = Linf_error_FEM( level_set_final , level_set_function , msh,degree_FEM);
         
        std::cout << bold << yellow << "L1-space error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< L1_err_par << std::endl;
        std::cout << bold << yellow << "L2-space error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< L2_err_par << std::endl;
        std::cout << bold << yellow << "Linf-space error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< Linf_err_par << std::endl;
        
        T W11_err_par =W1p_error_FEM( level_set_final , level_set_function , msh ,degree_FEM, 1.0);
        T W12_err_par =W1p_error_FEM( level_set_final , level_set_function , msh ,degree_FEM, 2.0);
        T W1inf_err_par =W1inf_error_FEM(level_set_final , level_set_function , msh,degree_FEM);
        
        // l^q in time: norm in time
        l1_L1_error += dt*L1_err_par;
        l2_L1_error += dt*L1_err_par*L1_err_par;
        linf_L1_error = std::max( L1_err_par , linf_L1_error );
        l1_L2_error += dt*L2_err_par;
        l2_L2_error += dt*L2_err_par*L2_err_par;
        linf_L2_error = std::max( L2_err_par , linf_L2_error );
        l1_Linf_error += dt*Linf_err_par;
        l2_Linf_error += dt*Linf_err_par*Linf_err_par;
        linf_Linf_error = std::max( Linf_err_par , linf_Linf_error );
        
        l1_W11_error += dt*W11_err_par ;
        l2_W11_error += dt*W11_err_par*W11_err_par ;
        linf_W11_error = std::max( W11_err_par , linf_W11_error );
        l1_W12_error += dt*W12_err_par ;
        l2_W12_error += dt*W12_err_par*W12_err_par ;
        linf_W12_error = std::max( W12_err_par , linf_W12_error );
        l1_W1inf_error += dt*W1inf_err_par;
        l2_W1inf_error += dt*W1inf_err_par*W1inf_err_par;
        linf_W1inf_error = std::max( W1inf_err_par , linf_W1inf_error );
        
        std::cout << bold << yellow << "W_1,1 error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< W11_err_par << std::endl;
        std::cout << bold << yellow << "W_1,2 error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< W12_err_par << std::endl;
        std::cout << bold << yellow << "W_1,inf error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< W1inf_err_par << std::endl;
        */
        
    } // End of the temporal loop
    /*
    l2_L1_error = sqrt(l2_L1_error);
    l2_L2_error = sqrt(l2_L2_error);
    l2_Linf_error = sqrt(l2_Linf_error);
    
    l2_W11_error = sqrt(l2_W11_error);
    l2_W12_error = sqrt(l2_W12_error);
    l2_W1inf_error = sqrt(l2_W1inf_error);
    std::cout<<'\n';
    std::cout << bold << yellow << "l1(L1) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_L1_error << std::endl;
    std::cout << bold << yellow << "l2(L1) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_L1_error << std::endl;
    std::cout << bold << yellow << "linf(L1) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_L1_error << std::endl;
    std::cout<<'\n';
    
    std::cout << bold << yellow << "l1(L2) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_L2_error << std::endl;
    std::cout << bold << yellow << "l2(L2) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_L2_error << std::endl;
    std::cout << bold << yellow << "linf(L2) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_L2_error << std::endl;
    std::cout<<'\n';
    std::cout << bold << yellow << "l1(Linf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_Linf_error << std::endl;
    std::cout << bold << yellow << "l2(Linf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_Linf_error << std::endl;
    std::cout << bold << yellow << "linf(Linf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_Linf_error << std::endl;
    std::cout<<'\n';
    
    std::cout << bold << yellow << "Seminorm l1(W11) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_W11_error << std::endl;
   
    std::cout << bold << yellow << "Seminorm l2(W11) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_W11_error << std::endl;
    std::cout << bold << yellow << "Seminorm linf(W11) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_W11_error << std::endl;
   
    std::cout<<'\n';
    std::cout << bold << yellow << "Seminorm l1(W12) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_W12_error << std::endl;
    
    std::cout << bold << yellow << "Seminorm l2(W12) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_W12_error << std::endl;
    std::cout << bold << yellow << "Seminorm linf(W12) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_W12_error << std::endl;
    
    std::cout<<'\n';
    std::cout << bold << yellow << "Seminorm l1(W1inf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_W1inf_error << std::endl;
    
    std::cout << bold << yellow << "Seminorm l2(W1inf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_W1inf_error << std::endl;
    std::cout << bold << yellow << "Seminorm linf(W1inf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_W1inf_error << std::endl;
    std::cout<<'\n';
    
    
    // ************** ANALYTIC SOLUTION FOR FINAL TIME LINEAR VELOCITY FIELD   **************
    T x_deviation = u_0*(T_N+1)*dt;
    T y_deviation = u_1*(T_N+1)*dt;
    auto analytic_level_set_final = circle_level_set<RealType>(radius, x_centre + x_deviation, y_centre + y_deviation );
    auto level_set_final = projected_level_set< T, Mesh,Fonction>(analytic_level_set_final, msh, degree_FEM , mip);
    

    level_set_final.smooth_cut_off( C , x_centre + x_deviation , y_centre + y_deviation , radius );
    //level_set_final.cut_off(d);
    testing_level_set2(msh,level_set_function,level_set_final);
    //testing_level_set(msh,level_set_function,analytic_level_set_final);
    Lp_space_Tfin_error_FEM( level_set_final , level_set_function , msh ,degree_FEM ,1.0 );
    Lp_space_Tfin_error_FEM( level_set_final , level_set_function , msh ,degree_FEM ,2.0 );
    */
    
    return 0;
}
#endif


// Convergence analysis for FEM
#if 0
int main(int argc, char **argv)
{
   
    convergence_test_time();
    return 1;
}
#endif


// Velocity is analytical: Stokes Case -> Time Evolution FAST - no area checking

#if 1
int main(int argc, char **argv)
{
    using RealType = double;

    size_t degree           = 0;
    size_t int_refsteps     = 4;
    size_t degree_FEM       = 0;

    bool dump_debug         = false;
    bool solve_interface    = false;
    bool solve_fictdom      = false;
    bool agglomeration      = false;

    mesh_init_params<RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;
    RealType d = 0.5;
    size_t T_N = 0;
    /* k <deg>:     method degree
     * g<deg>:  method FEM degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ( (ch = getopt(argc, argv, "k:q:c:M:N:r:T:ifDAd")) != -1 )
    {
        switch(ch)
        {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;
                
            case 'c':
                d = 1./atoi(optarg);
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;
            
            case 'T':
                T_N = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;
            
            

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;

    
    
    timecounter tc;

    /************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh<RealType> msh(mip);
    typedef cuthho_poly_mesh<RealType> Mesh;
    offset_definition(msh);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
       
    
    
    
    /************** FINITE ELEMENT INITIALIZATION **************/
    auto fe_data = Finite_Element<RealType>( msh.nodes.size() , degree_FEM , mip ) ;
    
    
    /************** ANALYTIC LEVEL SET FUNCTION  **************/
    RealType radius = 1.0/9.0; // I PUT 6.0, IT WAS 1.0/3.0
    RealType x_centre = 0.45;
    RealType y_centre = 0.5;
    auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre, y_centre);
    std::cout << bold << yellow << "Initial Analytic Area Circle: "<< M_PI*radius*radius << " , and circonference: "<< 2*M_PI*radius << reset << std::endl;
    
    
    /************** CELL BASIS PROVA  **************/
    testing_basis(msh , degree_FEM );
   
    
    // auto level_set_function_anal = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);//(0.11, 0.1, 0.1, 4, 0.04);
    //  auto level_set_function_anal = square_level_set<RealType>(0.6,0.4,0.4,0.6);
    
    typedef RealType T;
    typedef  circle_level_set<T> Fonction;
    //typedef  flower_level_set<T> Fonction; //DA RIMETTERE POI PER LA DISCRETA
    //typedef line_level_set<T> Fonction;
    // typedef square_level_set<T> Fonction;
    
       
    
    /************** ANALYTIC LEVEL SET FUNCTION, OLD IMPLEMENTATION **************/
    //auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
    //auto level_set_function = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);
    // auto level_set_function = line_level_set<RealType>(0.5);


    /************** ANALYTIC VELOCITY FIELD   **************/
    T u_0 = 1. ;
    T u_1 = 0. ;
    auto u = linear_velocity_field<RealType>(0,u_0,0,u_1); // analytic velocity (0,0,0,0)
    typedef linear_velocity_field<RealType> Velocity;
    //auto u = rotational_velocity_field<RealType>( x_centre , y_centre , 1.0);
    //typedef rotational_velocity_field<RealType> Velocity;
    //auto u = stokes_field<RealType>();
    //auto u = stokes_field<RealType>(4.0,2.0);
    //auto u = stokes_field<RealType>(4.0,2.0,2.0);
    //typedef stokes_field<RealType> Velocity;
 //    std::cout<<"prob here 0  "<<std::endl;
    auto u_projected = projection_velocity< T, Mesh,Velocity>(u, msh, degree_FEM , mip);
//     std::cout<<"prob here 1 "<<std::endl;
    testing_velocity( msh , u_projected , u );

    
    
    /************** LEVEL SET FUNCTION DISCRETISATION **************/
    std::cout<<"degree FEM "<<degree_FEM<<std::endl;
    auto level_set_function = projected_level_set< T, Mesh,Fonction>(level_set_function_anal, msh, degree_FEM , mip);
     
    // T d = 0.5;
    //level_set_function.cut_off(d);
    T C = 0.2;//level_set_function.vertices.maxCoeff();
    std::cout<<"Smoothed solution, C  max value is "<< C <<std::endl;
    level_set_function.smooth_cut_off( C , x_centre , y_centre , radius );
    testing_level_set(msh,level_set_function,level_set_function_anal);
     
     
    auto crr_mesh =  Current_Mesh<Mesh>(msh);
   
    T area_previous_time = 0 , mass_previous_time = 0 , dt = 0 ;
    T diff_area = 0. , diff_mass = 0. , diff_global_mass = 0. ;
    T initial_area = 0. , initial_mass = 0.;
    
   
    T error_normal_global = 0. ;
    T error_normal_local = 0. ;
   
   
    
    T l1_L1_error = 0 , l2_L1_error = 0 , linf_L1_error = 0;
    T l1_L2_error = 0 ,  l2_L2_error = 0 , linf_L2_error = 0;
    T l1_Linf_error = 0 ,  l2_Linf_error = 0 , linf_Linf_error = 0;
    
    T l1_W11_error = 0 , l2_W11_error = 0 , linf_W11_error = 0;
    T l1_W12_error = 0 ,  l2_W12_error = 0 , linf_W12_error = 0;
    T l1_W1inf_error = 0 ,  l2_W1inf_error = 0 , linf_W1inf_error = 0;
    
    
    for (size_t time_step = 0; time_step<=T_N; time_step++)
    {
    
        T area0 = 0. , mass0 = 0. , global_mass0 = 0. ;
        
        // ************** Re-Initialization mesh **************
        crr_mesh.current_mesh = msh;
        Mesh msh_i =  crr_mesh.current_mesh;
        offset_definition(msh_i);
        typedef projected_level_set< T, Mesh,Fonction> Level_Set;
        auto ls_cell = LS_cell< T , Mesh , Level_Set, Fonction >(level_set_function,msh_i);
           
    if(time_step == 0)
    {
        //************ DO cutHHO MESH PROCESSING **************
        tc.tic();
        detect_node_position2(msh_i, level_set_function); // In cuthho_geom
        detect_cut_faces2(msh_i, level_set_function); // In cuthho_geom
       
       
        if (agglomeration)
        {
            detect_cut_cells2(msh_i, level_set_function); // In cuthho_geom
            detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
            make_neighbors_info_cartesian(msh_i); // Non serve modificarla
            refine_interface2(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
            make_agglomeration(msh_i, level_set_function); // Non serve modificarla
        }
        else
        {
            move_nodes(msh_i, level_set_function);
            detect_cut_faces2(msh_i, level_set_function); //do it again to update intersection points
            detect_cut_cells2(msh_i, level_set_function);
            refine_interface2(msh_i, level_set_function, int_refsteps);
        }
       
        tc.toc();
        std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

        if (dump_debug)
        {
            dump_mesh(msh_i);
            output_mesh_info(msh_i, level_set_function);
            test_projection(msh_i, level_set_function, degree);
        }
       if(time_step == 0)
           output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export
        //output_mesh_info2(msh_i, level_set_function); // IN cuthho_export..Points/Nodes don't change
   
        
        
        // CALCULATION OF AREA AND MASS AT TIME STEP t^n
        for(auto& cl : msh_i.cells)
        {
            ls_cell.cell_assignment(cl);
            if( location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE || location(msh_i, cl) == element_location::ON_INTERFACE )
            {
                T partial_area = measure( msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                area0 += partial_area;
                auto qps=integrate(msh_i,cl,2*degree_FEM+1,element_location::IN_NEGATIVE_SIDE);
                for(auto& qp:qps)
                    mass0 += qp.second * ls_cell(qp.first);
                if( location(msh_i, cl) == element_location::ON_INTERFACE )
                {
                    auto qps1 = integrate(msh_i,cl,2*degree_FEM+1,element_location::IN_POSITIVE_SIDE);
                    for(auto& qp1:qps1)
                        global_mass0 += qp1.second * ls_cell(qp1.first);
                }
            }
            else
            {
                auto qps2 = integrate(msh_i,cl,2*degree_FEM+1,element_location::IN_POSITIVE_SIDE);
                for(auto& qp2:qps2)
                    global_mass0 += qp2.second * ls_cell(qp2.first);
            }
        }
        global_mass0 += mass0;
        std::cout << bold << yellow << "Area at time step: "<<time_step*dt<<" is "<<reset<< area0  << reset << std::endl;
        std::cout << bold << yellow << "Internal mass at time step: "<<time_step*dt<<" is "<<reset<< mass0  << reset << std::endl;
        //std::cout<<bold<<yellow << "GLOBAL Mass at time step: "<<time_step<<" is "<<reset<<global_mass0<< reset << std::endl;
           
    }
        
        
        if(time_step == 0 ){
            initial_area  = area0 ;
            initial_mass = mass0 ;
        }
       
        /*
        if(time_step > 0 )
        {
            diff_area = area0 - area_previous_time ;
            diff_mass = mass0 - mass_previous_time ;
            std::cout << bold << yellow << "Difference in Area (new - old) at time step: "<<time_step*dt<<" is "<<reset<< diff_area  << reset << std::endl;
            std::cout << bold << yellow << "Difference in internal MASS (new - old) at time step: "<<time_step*dt<<" is "<<reset<< diff_mass  << reset << std::endl;
            
           
        } */
        
        /// Checking normal error --> TOLTO PER VELOCIZZARE
    
        T x_deviation = u_0*(time_step)*dt;
        T y_deviation = u_1*(time_step)*dt;
        auto analytic_level_set_final = circle_level_set<RealType>(radius, x_centre + x_deviation, y_centre + y_deviation );
        auto level_set_exact = projected_level_set< T, Mesh,Fonction>(analytic_level_set_final, msh, degree_FEM , mip);
        
        level_set_exact.smooth_cut_off( C , x_centre + x_deviation , y_centre +y_deviation,radius );
        
        auto ls_cell_exact = LS_cell< T , Mesh , Level_Set, Fonction >(level_set_exact,msh_i);
        
        // postprocess_output<double> postoutput_vec;
        //  auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("vec_normal_interface.dat");
        // auto vec_normal_exact = std::make_shared< gnuplot_output_object_vec<double> >("vec_normal_interface_EXACT.dat");
           
        T error_normal_global0 = 0. ;
        T diff_interface_normal = 0. ;
        size_t counter_interface_pts = 0;
        for(const auto & cl:msh_i.cells)
        {
            if(cl.user_data.location == element_location::ON_INTERFACE)
            {
                ls_cell.cell_assignment(cl);
                ls_cell_exact.cell_assignment(cl);
                auto qps=integrate(msh_i,cl,2*degree_FEM+1);
                for(auto& qp:qps){
                    Eigen::Matrix<T,2,1> diff_global_normal = (ls_cell.normal(qp.first) - ls_cell_exact.normal(qp.first) ).cwiseAbs();
                    error_normal_global0 += qp.second * (pow(diff_global_normal(0),2));
                    error_normal_global0 += qp.second * (pow(diff_global_normal(1),2));
                
                }
        
            /*
            for(auto& pt:points(msh_i,cl))
            {
                Eigen::Matrix<T,2,1> diff_global_normal = (ls_cell.normal(pt) - ls_cell_exact.normal(pt) ).cwiseAbs();
                   
                std::cout <<'\t' << diff_global_normal(0)<<" , "<< diff_global_normal(1) <<'\t';
            }
            
            std::cout<<std::endl;
            */
              
                // std::cout<<"Normal error in CUT CELL "<<offset(msh_i,cl)<<" is "<<'\n';
        
                for(auto& interface_point : cl.user_data.interface)
                {
                    Eigen::Matrix<T,2,1> normal = ls_cell.normal(interface_point);
                    Eigen::Matrix<T,2,1> normal_exact = ls_cell_exact.normal(interface_point);
                    
                    diff_interface_normal += (pow((normal(0)-normal_exact(0)),2) + pow((normal(1)-normal_exact(1)),2));
                    
                  //  std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                  //  vec_normal->add_data(interface_point,normal_vec);
                   // std::pair<T,T> normal_vec_exact = std::make_pair(normal(0),normal(1));
                   // vec_normal_exact->add_data(interface_point,normal_vec_exact);
                    
                    counter_interface_pts++;
                }
                std::cout<<std::endl;
            }
        }

      //  postoutput_vec.add_object(vec_normal);
      //  postoutput_vec.add_object(vec_normal_exact);
      //  postoutput_vec.write();
            
        error_normal_global0 = sqrt(error_normal_global0);
        
        std::cout<<"number of interface points is "<<counter_interface_pts<<std::endl;
        diff_interface_normal /= counter_interface_pts;
        diff_interface_normal = sqrt(diff_interface_normal);
        
        std::cout<<"The L2 error of normal N over the cut cells, at time "<< dt*time_step <<" is " << error_normal_global0 <<std::endl;
        std::cout<<"The l2 error of normal N over the INTERFACE, at time "<< dt*time_step <<" is " << diff_interface_normal <<std::endl;
        
        
        area_previous_time  = area0 ;
        mass_previous_time = mass0 ;
        
        error_normal_global += error_normal_global0*dt;
        error_normal_local += diff_interface_normal*dt;

        
        // CUT CELL PLOTTING
        /*
        postprocess_output<double> postoutput3;
        auto cut_cell_disegno = std::make_shared< gnuplot_output_object<double> >("cut_cell_disegno.dat");
        for (auto cl:msh_i.cells) {
            if( is_cut(msh_i,cl) )
            {
            // std::cout<<" the cell in the new mesh num "<<offset(msh,cl)<<" is cut."<<std::endl;
                // std::cout<<" EQuivalent to the old cells numero: ";
                auto pts = points(msh_i,cl) ;
                for (auto pt:pts) {
                    T value = 1.;
                    cut_cell_disegno->add_data(pt,value);
                }
            }
        }
        postoutput3.add_object(cut_cell_disegno);
        postoutput3.write();
        */
    
        // jumps sin_sin -> exp_cos
        // auto test_case = make_test_case_laplacian_jumps_1(msh, level_set_function);
        // jumps3 sin_sin -> sin_sin + pol
    
/*
        //auto test_case = make_test_case_laplacian_jumps_3bis(msh_i, ls_cell,current_cell);
        auto test_case = make_test_case_laplacian_jumps_3(msh_i, ls_cell);
        // 2 possibilità.. 1) creo 3bis 2) anzichè passarle level_set_function le passo LS_cell

        // auto method = make_Nitsche_interface_method(msh, 1.0, test_case);
        // auto method = make_sym_gradrec_interface_method(msh, 1.0, test_case);
        // auto method = make_gradrec_interface_method(msh, 1.0, test_case);
        auto method = make_Nitsche_interface_method_2(msh_i, 1.0, test_case);
        //Matrix<RealType, Dynamic, 1> sol;
        if(solve_interface){
            run_cuthho_interface2(msh_i, degree, method,test_case, ls_cell);
        //run_cuthho_interface(msh, degree, method, test_case);
        }
        if (solve_fictdom)
            run_cuthho_fictdom(msh_i, degree, test_case);
        
*/
        /************** FEM -  PROCESSING **************/
        
        T eps = 0.5 ; // factor to be inside CFL stability zone
        T dt1 = time_step_CFL( u_projected , mip , eps );
        //std::cout<<"dt1 is "<<dt1<<std::endl;
        T dt2 = 2.5*1e-3;
        dt = std::min(dt1 , dt2);
        std::cout<<"I USE dt = "<<dt<<" AND CFL IS "<<dt1<<std::endl;
        
         std::cout<<"PRE FEM è"<<std::endl;
               for(auto& cl : msh.cells)
                   for(auto& pt : points(msh,cl)){
                       std::cout.precision(15);
                       std::cout<<level_set_function(pt,msh,cl)<<std::endl;
                   }
        
               std::cout<<"FINE FEM"<<std::endl;
        
        run_FEM_levelset( level_set_function.msh,degree_FEM,level_set_function,u_projected,dt,mip);
       //run_FEM_levelset_SPARSE( level_set_function.msh ,degree_FEM, level_set_function,u_projected,dt,mip);
        
      std::cout<<"LA SOL è"<<std::endl;
                for(auto& cl : msh.cells)
                    for(auto& pt : points(msh,cl)){
                        std::cout.precision(15);
                        std::cout<<level_set_function(pt,msh,cl)<<std::endl;
                    }
         std::cout<<"FINE SOL"<<std::endl;
        
          // Uploading mesh data to check out differences in mass and areas at LAST TIME
        if( (T_N - time_step)==0 )
        {
            crr_mesh.current_mesh = msh;
            Mesh msh_i2 =  crr_mesh.current_mesh;
            offset_definition(msh_i2);
            tc.tic();
            detect_node_position2(msh_i2, level_set_function); // In cuthho_geom
            detect_cut_faces2(msh_i2, level_set_function); // In cuthho_geom
        
       
            if (agglomeration)
            {
           // std::cout<<"i m here 1"<<std::endl;
                detect_cut_cells2(msh_i2, level_set_function); // In cuthho_geom
           
                detect_cell_agglo_set(msh_i2, level_set_function); // Non serve modificarla
           
                make_neighbors_info_cartesian(msh_i2); // Non serve modificarla
                refine_interface2(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
                make_agglomeration(msh_i2, level_set_function); // Non serve modificarla
            }
            else
            {
                move_nodes(msh_i2, level_set_function);
                detect_cut_faces2(msh_i2, level_set_function); //do it again to update intersection points
                detect_cut_cells2(msh_i2, level_set_function);
                refine_interface2(msh_i2, level_set_function, int_refsteps);
            }
        
            tc.toc();
            std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

            if (dump_debug)
            {
                dump_mesh(msh_i2);
                output_mesh_info(msh_i2, level_set_function);
                test_projection(msh_i2, level_set_function, degree);
            }
        
            output_mesh_info2(msh_i2, level_set_function); // IN cuthho_export..Points/Nodes don't change
        
            // Uploading level set
            ls_cell.level_set = level_set_function;
            ls_cell.agglo_msh = msh_i2;
        
       
            T  mass_fin = 0. , global_mass1 = 0. , area_fin = 0. ;
            for(auto& cl : msh_i2.cells)
            {
                ls_cell.cell_assignment(cl);
                if( location(msh_i2, cl) == element_location::IN_NEGATIVE_SIDE || location(msh_i2, cl) == element_location::ON_INTERFACE )
                {
                    T partial_area = measure( msh_i2, cl, element_location::IN_NEGATIVE_SIDE);
                    area_fin += partial_area;
                    auto qps=integrate(msh_i2,cl,2*degree_FEM+1,element_location::IN_NEGATIVE_SIDE);
                    for(auto& qp:qps)
                        mass_fin += qp.second * ls_cell(qp.first);
                    if( location(msh_i2, cl) == element_location::ON_INTERFACE )
                    {
                        auto qps1 = integrate(msh_i2,cl,2*degree_FEM+1,element_location::IN_POSITIVE_SIDE);
                        for(auto& qp1:qps1)
                            global_mass1 += qp1.second * ls_cell(qp1.first);
                    }
                }
                else
                {
                    auto qps2 = integrate(msh_i2,cl,2*degree_FEM+1,element_location::IN_POSITIVE_SIDE);
                    for(auto& qp2:qps2)
                        global_mass1 += qp2.second * ls_cell(qp2.first);
                }
            }
            global_mass1 += mass_fin;
            
            std::cout << bold << yellow << "Area at time step: " <<(time_step+1)*dt<<" is "<< reset<< area_fin   << std::endl;
           std::cout << bold << yellow << "Internal mass at time step: "<<(time_step+1)*dt<<" is "<<reset<< mass_fin  << reset << std::endl;
        //    std::cout<<bold<<yellow << "GLOBAL Mass at time step: "<<(time_step+1)<<" is "<<reset<<global_mass1<< reset << std::endl;
            
            std::cout << bold << yellow << "Difference in AREA between initial state and final TIME "<<(T_N+1)*(dt)<<" IS "<< reset<< area_fin - initial_area  << std::endl;
            std::cout << bold << yellow << "Difference in INTERNAL MASS between initial state and final TIME "<<(T_N+1)*(dt)<<" IS "<< reset<< mass_fin - initial_mass << std::endl;
          //  std::cout << bold << yellow << "Difference in GLOBAL MASS: "<< reset<< global_mass1 - global_mass0 << std::endl;
            
            
            
            
            
            /// Checking normal error
         
            x_deviation = u_0*(time_step+1)*dt;
            y_deviation = u_1*(time_step+1)*dt;
            
            
            
            auto analytic_level_set_final_exact = circle_level_set<RealType>(radius, x_centre + x_deviation, y_centre + y_deviation );
            auto level_set_exact_fin = projected_level_set < T, Mesh,Fonction > (analytic_level_set_final_exact, msh, degree_FEM , mip);
            
            level_set_exact_fin.smooth_cut_off( C , x_centre + x_deviation , y_centre + y_deviation , radius );
            
            auto ls_cell_exact_fin = LS_cell< T,Mesh,Level_Set,Fonction > (level_set_exact_fin,msh_i2);
            
            
            postprocess_output<double> postoutput_vec;
            auto vec_normal = std::make_shared< gnuplot_output_object_vec<double> >("vec_normal_interface.dat");
            auto vec_normal_exact = std::make_shared< gnuplot_output_object_vec<double> >("vec_normal_interface_EXACT.dat");
            error_normal_global0 = 0. ;
            diff_interface_normal = 0. ;
            counter_interface_pts = 0;
            for(const auto & cl:msh_i2.cells)
            {
                if(cl.user_data.location == element_location::ON_INTERFACE)
                {
                    ls_cell.cell_assignment(cl);
                    ls_cell_exact_fin.cell_assignment(cl);
                    auto qps=integrate(msh_i2,cl,2*degree_FEM+1);
                    for(auto& qp:qps){
                        Eigen::Matrix<T,2,1> diff_global_normal = (ls_cell.normal(qp.first) - ls_cell_exact_fin.normal(qp.first) ).cwiseAbs();
                        error_normal_global0 += qp.second * (pow(diff_global_normal(0),2));
                        error_normal_global0 += qp.second * (pow(diff_global_normal(1),2));
                    
                    }
            
                    for(auto& interface_point : cl.user_data.interface)
                    {
                        Eigen::Matrix<T,2,1> normal = ls_cell.normal(interface_point);
                        Eigen::Matrix<T,2,1> normal_exact = ls_cell_exact_fin.normal(interface_point);
                                
                        diff_interface_normal += (pow((normal(0)-normal_exact(0)),2) + pow((normal(1)-normal_exact(1)),2));
                                
                        std::pair<T,T> normal_vec = std::make_pair(normal(0),normal(1));
                        vec_normal->add_data(interface_point,normal_vec);
                        std::pair<T,T> normal_vec_exact = std::make_pair(normal(0),normal(1));
                        vec_normal_exact->add_data(interface_point,normal_vec_exact);
                                
                        counter_interface_pts++;
                    }
                    std::cout<<std::endl;
                }
            }
            postoutput_vec.add_object(vec_normal);
            postoutput_vec.add_object(vec_normal_exact);
            postoutput_vec.write();
                        
            error_normal_global0 = sqrt(error_normal_global0);
                
            std::cout<<"number of interface points is "<<counter_interface_pts<<std::endl;
                    
            diff_interface_normal /= counter_interface_pts;
            diff_interface_normal = sqrt(diff_interface_normal);
                
            std::cout<<"The L2 error of normal N over the cut cells, at time "<< dt*(time_step+1) <<" is " << error_normal_global0 <<std::endl;
            std::cout<<"The l2 error of normal N over the INTERFACE, at time "<< dt*(time_step+1) <<" is " << diff_interface_normal <<std::endl;
                    
                    
                    
            error_normal_global += error_normal_global0*dt;
            error_normal_local += diff_interface_normal*dt;
                    
                
            std::cout << bold << yellow << "Error l1(L2) of the normal in the CUT CELLS: "<< reset<< error_normal_global  << std::endl;
            
            std::cout << bold << yellow << "Error l1(l2) of the normal in the INTERFACE POINTS: "<< reset<< error_normal_local  << std::endl;
        
        }
       // testing_level_set(msh,level_set_function,level_set_function_anal);
        
        x_deviation = u_0*(time_step+1)*dt;
        y_deviation = u_1*(time_step+1)*dt;
        analytic_level_set_final = circle_level_set<RealType>(radius, x_centre + x_deviation, y_centre + y_deviation );
        auto level_set_final = projected_level_set< T, Mesh,Fonction>(analytic_level_set_final, msh, degree_FEM , mip);
        
        
        
        level_set_final.smooth_cut_off( C , x_centre + x_deviation , y_centre +y_deviation , radius );
        //level_set_final.cut_off(d);
        testing_level_set2(msh,level_set_function,level_set_final);
            //testing_level_set(msh,level_set_function,analytic_level_set_final);
         
        T L1_err_par = Lp_space_error_FEM( level_set_final , level_set_function,msh,degree_FEM,1.0);
        
        T L2_err_par = Lp_space_error_FEM( level_set_final , level_set_function,msh,degree_FEM,2.0);
         
        T Linf_err_par = Linf_error_FEM( level_set_final , level_set_function , msh,degree_FEM);
         
        std::cout << bold << yellow << "L1-space error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< L1_err_par << std::endl;
        std::cout << bold << yellow << "L2-space error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< L2_err_par << std::endl;
        std::cout << bold << yellow << "Linf-space error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< Linf_err_par << std::endl;
        
        T W11_err_par =W1p_error_FEM( level_set_final , level_set_function , msh ,degree_FEM, 1.0);
        T W12_err_par =W1p_error_FEM( level_set_final , level_set_function , msh ,degree_FEM, 2.0);
        T W1inf_err_par =W1inf_error_FEM(level_set_final , level_set_function , msh,degree_FEM);
        
        // l^q in time: norm in time
        l1_L1_error += dt*L1_err_par;
        l2_L1_error += dt*L1_err_par*L1_err_par;
        linf_L1_error = std::max( L1_err_par , linf_L1_error );
        l1_L2_error += dt*L2_err_par;
        l2_L2_error += dt*L2_err_par*L2_err_par;
        linf_L2_error = std::max( L2_err_par , linf_L2_error );
        l1_Linf_error += dt*Linf_err_par;
        l2_Linf_error += dt*Linf_err_par*Linf_err_par;
        linf_Linf_error = std::max( Linf_err_par , linf_Linf_error );
        
        l1_W11_error += dt*W11_err_par ;
        l2_W11_error += dt*W11_err_par*W11_err_par ;
        linf_W11_error = std::max( W11_err_par , linf_W11_error );
        l1_W12_error += dt*W12_err_par ;
        l2_W12_error += dt*W12_err_par*W12_err_par ;
        linf_W12_error = std::max( W12_err_par , linf_W12_error );
        l1_W1inf_error += dt*W1inf_err_par;
        l2_W1inf_error += dt*W1inf_err_par*W1inf_err_par;
        linf_W1inf_error = std::max( W1inf_err_par , linf_W1inf_error );
        
        std::cout << bold << yellow << "W_1,1 error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< W11_err_par << std::endl;
        std::cout << bold << yellow << "W_1,2 error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< W12_err_par << std::endl;
        std::cout << bold << yellow << "W_1,inf error at time "<<(time_step+1)*(dt)<<" IS "<< reset<< W1inf_err_par << std::endl;
        
        
    } // End of the temporal loop
    
    l2_L1_error = sqrt(l2_L1_error);
    l2_L2_error = sqrt(l2_L2_error);
    l2_Linf_error = sqrt(l2_Linf_error);
    
    l2_W11_error = sqrt(l2_W11_error);
    l2_W12_error = sqrt(l2_W12_error);
    l2_W1inf_error = sqrt(l2_W1inf_error);
    std::cout<<'\n';
    std::cout << bold << yellow << "l1(L1) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_L1_error << std::endl;
    std::cout << bold << yellow << "l2(L1) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_L1_error << std::endl;
    std::cout << bold << yellow << "linf(L1) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_L1_error << std::endl;
    std::cout<<'\n';
    
    std::cout << bold << yellow << "l1(L2) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_L2_error << std::endl;
    std::cout << bold << yellow << "l2(L2) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_L2_error << std::endl;
    std::cout << bold << yellow << "linf(L2) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_L2_error << std::endl;
    std::cout<<'\n';
    std::cout << bold << yellow << "l1(Linf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_Linf_error << std::endl;
    std::cout << bold << yellow << "l2(Linf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_Linf_error << std::endl;
    std::cout << bold << yellow << "linf(Linf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_Linf_error << std::endl;
    std::cout<<'\n';
    
    std::cout << bold << yellow << "Seminorm l1(W11) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_W11_error << std::endl;
   
    std::cout << bold << yellow << "Seminorm l2(W11) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_W11_error << std::endl;
    std::cout << bold << yellow << "Seminorm linf(W11) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_W11_error << std::endl;
   
    std::cout<<'\n';
    std::cout << bold << yellow << "Seminorm l1(W12) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_W12_error << std::endl;
    
    std::cout << bold << yellow << "Seminorm l2(W12) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_W12_error << std::endl;
    std::cout << bold << yellow << "Seminorm linf(W12) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_W12_error << std::endl;
    
    std::cout<<'\n';
    std::cout << bold << yellow << "Seminorm l1(W1inf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l1_W1inf_error << std::endl;
    
    std::cout << bold << yellow << "Seminorm l2(W1inf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< l2_W1inf_error << std::endl;
    std::cout << bold << yellow << "Seminorm linf(W1inf) error GLOBAL at final time "<<(T_N+1)*(dt)<<" IS "<< reset<< linf_W1inf_error << std::endl;
    std::cout<<'\n';
    
    
    // ************** ANALYTIC SOLUTION FOR FINAL TIME LINEAR VELOCITY FIELD   **************
    /*
    T x_deviation = u_0*(T_N+1)*dt;
    T y_deviation = u_1*(T_N+1)*dt;
    auto analytic_level_set_final = circle_level_set<RealType>(radius, x_centre + x_deviation, y_centre + y_deviation );
    auto level_set_final = projected_level_set< T, Mesh,Fonction>(analytic_level_set_final, msh, degree_FEM , mip);
    

    level_set_final.smooth_cut_off( C , x_centre + x_deviation , y_centre + y_deviation , radius );
    //level_set_final.cut_off(d);
    testing_level_set2(msh,level_set_function,level_set_final);
    //testing_level_set(msh,level_set_function,analytic_level_set_final);
    Lp_space_Tfin_error_FEM( level_set_final , level_set_function , msh ,degree_FEM ,1.0 );
    Lp_space_Tfin_error_FEM( level_set_final , level_set_function , msh ,degree_FEM ,2.0 );
    */
    
    return 0;
}
#endif
