/*
 *       /\        Matteo Cicuttin (C) 2017,2018
 *      /__\       matteo.cicuttin@enpc.fr
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

#pragma once

#include <array>
#include <set>
#include <cassert>

#include "core/core"

enum class element_location {
    IN_NEGATIVE_SIDE,
    IN_POSITIVE_SIDE,
    ON_INTERFACE,
    UNDEF
};

enum class cell_agglo_set {
    T_OK,
    T_KO_NEG,
    T_KO_POS,
    UNDEF
};



template<typename T, typename Cell >
struct integration_mesh_cl
{

    typedef point<T,2>          point_type;
    //typedef T                   coordinate_type;
  

    std::vector<point_type>     points;
    std::vector<Cell>      cells;
    size_t size_cls ;
    size_t degree_curve ;
   
    //Mesh msh ;
    //Cell cl;
    
    integration_mesh_cl(){}
    
    
    integration_mesh_cl(integration_mesh_cl& other ){
        points = other.points ;
        cells = other.cells ;
        size_cls = other.size_cls ;
        degree_curve = other.degree_curve ;
        
        
    }
    
    integration_mesh_cl( Cell& cl, T degree_curve ) : degree_curve(degree_curve), points(cl.userd_data.interface), size_cls((points.size()-1)/degree_curve)
    {
      
        size_t point_num = 0;
        //cell_basis_Lagrange_1d_reference <Mesh,T> cb(msh, cl, basis_degree);
        for(size_t pos = 0 ; pos < size_cls ; pos++ )
        {
            Cell m_cl;
            for (size_t i = 0; i <= degree_curve ; i++)
            {
                m_cl.ptids.push_back( point_num ) ;
                if( i < degree_curve )
                    point_num++ ;
            }
            cells.push_back(m_cl);
        }
                
        
            
        
    }
    
       
};

template<typename T, typename Cell >
struct cell_cuthho_info_bis
{
    element_location            location;
    cell_agglo_set              agglo_set;
    point<T,2>                  p0, p1;
    std::vector<point<T,2>>     interface;
    //integration_mesh<T>         integration_msh;
    
    integration_mesh_cl<T,Cell>         integration_msh;
    bool                        distorted;
    std::set<size_t>            f_neighbors; // face neighbors
    std::set<size_t>            d_neighbors; // diagonal neighbors
    
    std::vector< std::pair<point<T,2>, T> > integration_n; // composite integration rules
    std::vector< std::pair<point<T,2>, T> > integration_p;

    bool                        highlight; // for tests
    
    std::vector<size_t>         offset_subcells;
    size_t                      degree_curve ;
    
    cell_cuthho_info_bis() :
        location(element_location::UNDEF),
        agglo_set(cell_agglo_set::UNDEF),
        distorted(false),
        //integration_msh(),
        integration_msh(),
        highlight(false)
        
    {}
  
    
    
};


template<typename T>
struct cell_cuthho_info
{
    element_location            location;
    cell_agglo_set              agglo_set;
    point<T,2>                  p0, p1;
    std::vector<point<T,2>>     interface;
    integration_mesh<T>         integration_msh;
    
    //integration_mesh_cl<T>         integration_msh;
    bool                        distorted;
    std::set<size_t>            f_neighbors; // face neighbors
    std::set<size_t>            d_neighbors; // diagonal neighbors
    
    std::vector< std::pair<point<T,2>, T> > integration_n; // composite integration rules
    std::vector< std::pair<point<T,2>, T> > integration_p;

    bool                        highlight; // for tests
    
    std::vector<size_t>         offset_subcells;
    size_t                      degree_curve ;
    
    cell_cuthho_info() :
        location(element_location::UNDEF),
        agglo_set(cell_agglo_set::UNDEF),
        distorted(false),
        integration_msh(),
        highlight(false)
        
    {}
};


template<typename T>
struct face_cuthho_info
{
    element_location    location;
    size_t              node_inside; /* tells which is the node inside the area
                                        delimited by interface. Can be 0 or 1,
                                        nothing else. */
    point<T, 2>         intersection_point;

    face_cuthho_info() :
        location(element_location::UNDEF),
        node_inside(0)
    {}
};

template<typename T>
struct node_cuthho_info
{
    element_location    location;
    bool                displaced;
    point<T,2>          displacement;

    node_cuthho_info() :
        location(element_location::UNDEF),
        displaced(false)
    {}
};


template<typename T, size_t ET>
using cuthho_mesh = mesh<T, ET, cell_cuthho_info<T>, face_cuthho_info<T>, node_cuthho_info<T>>;

template<typename T>
using cuthho_quad_mesh = mesh<T, 4, cell_cuthho_info<T>, face_cuthho_info<T>, node_cuthho_info<T>>;

template<typename T>
using cuthho_poly_mesh = mesh<T, 0, cell_cuthho_info<T>, face_cuthho_info<T>, node_cuthho_info<T>>;


template<typename T >
using hho_mesh_integration = mesh<T, 1, cell_cuthho_info<T>, face_cuthho_info<T>, node_cuthho_info<T>>;
