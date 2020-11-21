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


#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <random>

#include "point.hpp"
//#include "quadratures.hpp"


/*****************************************************************************
 *   Mesh stuff
 *****************************************************************************/
template<typename UserData>
struct mesh_element
{
    UserData    user_data;
};

template<>
struct mesh_element<void>
{};

struct dynamic_storage;

template<typename UserData, size_t N>
struct cell : public mesh_element<UserData> {
    std::array<size_t, N>   ptids;

    cell()
    {}

    bool operator<(const cell& other) const
    {
        return (this->ptids < other.ptids);
    }

    bool operator==(const cell& other) const
    {
        return (this->ptids == other.ptids);
    }
};

template<typename UserData>
struct cell<UserData, 0> : public mesh_element<UserData> {
    std::vector<size_t>   ptids;

    cell()
    {}

    bool operator<(const cell& other) const
    {
        return (this->ptids < other.ptids);
    }

    bool operator==(const cell& other) const
    {
        return (this->ptids == other.ptids);
    }
};

template<typename UserData>
struct cell<UserData, 1> : public mesh_element<UserData> {
    std::vector<size_t>   ptids;

    cell()
    {}

    bool operator<(const cell& other) const
    {
        return (this->ptids < other.ptids);
    }

    bool operator==(const cell& other) const
    {
        return (this->ptids == other.ptids);
    }
};



template<typename UserData = void>
using poly_cell = cell<UserData, 0>;

template<typename UserData = void>
using quad_cell = cell<UserData, 4>;

template<typename UserData = void>
using tri_cell = cell<UserData, 3>;

template<typename UserData, size_t N>
std::ostream&
operator<<(std::ostream& os, const cell<UserData, N>& cl)
{
    os << "Cell: ";
    for (size_t i = 0; i < N; i++)
        os << cl.ptids[i] << " ";

    return os;
}

enum class boundary
{
    NONE,
    DIRICHLET,
    NEUMANN,
    ROBIN
};

template<typename UserData>
struct face : public mesh_element<UserData> {
    std::array<size_t, 2>   ptids;
    bool                    is_boundary;
    boundary                bndtype;

    face() :
        is_boundary(false),
        bndtype(boundary::NONE)
    {}

    bool operator<(const face& other) const
    {
        assert(ptids[0] < ptids[1]);
        assert(other.ptids[0] < other.ptids[1]);
        return (this->ptids < other.ptids);
    }

    bool operator==(const face& other) const
    {
        assert(ptids[0] < ptids[1]);
        assert(other.ptids[0] < other.ptids[1]);
        return (this->ptids == other.ptids);
    }
};

template<typename UserData>
std::ostream&
operator<<(std::ostream& os, const face<UserData>& fc)
{
    os << "Face: " << fc.ptids[0] << " " << fc.ptids[1];
    if (fc.is_boundary && fc.bndtype == boundary::NONE)
        os << " [Boundary, unspecified kind]";
    if (fc.is_boundary && fc.bndtype == boundary::DIRICHLET)
        os << " [Dirichlet boundary]";
    if (fc.is_boundary && fc.bndtype == boundary::NEUMANN)
        os << " [Neumann boundary]";
    if (fc.is_boundary && fc.bndtype == boundary::ROBIN)
        os << " [Robin boundary]";
    return os;
}

template<typename UserData>
struct node : public mesh_element<UserData> {
    size_t      ptid;

    bool operator<(const node& other) const
    {
        return (this->ptid < other.ptid);
    }

    bool operator==(const node& other) const
    {
        return (this->ptid == other.ptid);
    }
};

template<typename UserData>
std::ostream&
operator<<(std::ostream& os, const node<UserData>& nd)
{
    os << "Node: " << nd.ptid;
    return os;
}

template<typename T>
struct mesh_init_params {
    T           min_x, max_x;
    T           min_y, max_y;
    size_t      Nx, Ny;

    mesh_init_params()
        : min_x(0.0), max_x(1.0),
          min_y(0.0), max_y(1.0),
          Nx(4), Ny(4)
    {}

    T hx() const {
        return (max_x - min_x)/Nx;
    }

    T hy() const {
        return (max_y - min_y)/Ny;
    }
};





struct elem_simplex;
struct elem_quad;
struct elem_poly;

template<typename T, size_t ElemType, typename CellUD = void, typename FaceUD = void, typename NodeUD = void>
struct mesh_impl;


template<typename T, typename CellUD, typename FaceUD, typename NodeUD>
struct mesh_impl<T, 4, CellUD, FaceUD, NodeUD> {

    typedef point<T,2>          point_type;
    typedef cell<CellUD, 4>     cell_type;
    typedef face<FaceUD>        face_type;
    typedef node<NodeUD>        node_type;
    typedef CellUD              cell_ud_type;
    typedef FaceUD              face_ud_type;
    typedef NodeUD              node_ud_type;
    typedef T                   coordinate_type;

    std::vector<point_type>     points;
    std::vector<node_type>      nodes;
    std::vector<face_type>      faces;
    std::vector<cell_type>      cells;

    mesh_impl() : mesh_impl( mesh_init_params<T>() )
    {}

    mesh_impl(const mesh_init_params<T>& parms)
    {
        auto hx = parms.hx();
        auto hy = parms.hy();

        size_t numpoints = (parms.Nx + 1) * (parms.Ny + 1);
        points.reserve(numpoints);

        size_t point_num = 0;
        for (size_t j = 0; j < parms.Ny+1; j++)
        {
            for(size_t i = 0; i < parms.Nx+1; i++)
            {
                auto px = parms.min_x + i*hx;
                auto py = parms.min_y + j*hy;
                point_type pt(px, py);
                points.push_back(pt);
                node_type n;
                n.ptid = point_num++;
                nodes.push_back(n);
            }
        }

        for (size_t j = 0; j < parms.Ny; j++)
        {
            for (size_t i = 0; i < parms.Nx; i++)
            {
                auto pt0_idx = j*(parms.Nx+1) + i;
                auto pt1_idx = pt0_idx + 1;
                auto pt2_idx = pt0_idx + parms.Nx + 2;
                auto pt3_idx = pt0_idx + parms.Nx + 1;

                cell_type cl;
                cl.ptids = {{pt0_idx, pt1_idx, pt2_idx, pt3_idx}};
                cells.push_back(cl);

                face_type f0;
                f0.ptids = {pt0_idx, pt1_idx};
                if (j == 0) f0.is_boundary = true;
                faces.push_back(f0);

                face_type f1;
                f1.ptids = {pt1_idx, pt2_idx};
                if (i == parms.Nx-1) f1.is_boundary = true;
                faces.push_back(f1);

                face_type f2;
                f2.ptids = {pt3_idx, pt2_idx};
                if (j == parms.Ny-1) f2.is_boundary = true;
                faces.push_back(f2);

                face_type f3;
                f3.ptids = {pt0_idx, pt3_idx};
                if (i == 0) f3.is_boundary = true;
                faces.push_back(f3);

            }
        }

        std::sort(cells.begin(), cells.end());
        std::sort(faces.begin(), faces.end());
        faces.erase( std::unique(faces.begin(), faces.end()), faces.end() );

        for (auto& fc : faces)
        {
            if (fc.is_boundary)
                fc.bndtype = boundary::DIRICHLET;
        }
    }
};

template<typename T, typename CellUD, typename FaceUD, typename NodeUD>
struct mesh_impl<T, 0, CellUD, FaceUD, NodeUD> {

    typedef point<T,2>          point_type;
    typedef cell<CellUD, 0>     cell_type;
    typedef face<FaceUD>        face_type;
    typedef node<NodeUD>        node_type;
    typedef CellUD              cell_ud_type;
    typedef FaceUD              face_ud_type;
    typedef NodeUD              node_ud_type;
    typedef T                   coordinate_type;

    std::vector<point_type>     points;
    std::vector<node_type>      nodes;
    std::vector<face_type>      faces;
    std::vector<cell_type>      cells;
    

    mesh_impl() : mesh_impl( mesh_init_params<T>() )
    {}

    mesh_impl(const mesh_init_params<T>& parms)
    {
        auto hx = parms.hx();
        auto hy = parms.hy();

        size_t numpoints = (parms.Nx + 1) * (parms.Ny + 1);
        points.reserve(numpoints);

        //std::default_random_engine generator;
        //std::uniform_real_distribution<double> dist_x(-hx/10.0, hx/10.0);
        //std::uniform_real_distribution<double> dist_y(-hy/10.0, hy/10.0);
        //auto disp_x = std::bind( dist_x, generator );
        //auto disp_y = std::bind( dist_y, generator );

        size_t point_num = 0;
        for (size_t j = 0; j < parms.Ny+1; j++)
        {
            for(size_t i = 0; i < parms.Nx+1; i++)
            {
                double dx = 0.0;
                double dy = 0.0;
                //if ( i != 0 && i != parms.Nx )
                //    dx = disp_x();

                //if ( j != 0 && j != parms.Ny )
                //    dy = disp_y();

                auto px = parms.min_x + i*hx + dx;
                auto py = parms.min_y + j*hy + dy;
                point_type pt(px, py);
                points.push_back(pt);
                node_type n;
                n.ptid = point_num++;
                nodes.push_back(n);
            }
        }

        for (size_t j = 0; j < parms.Ny; j++)
        {
            for (size_t i = 0; i < parms.Nx; i++)
            {
                auto pt0_idx = j*(parms.Nx+1) + i;
                auto pt1_idx = pt0_idx + 1;
                auto pt2_idx = pt0_idx + parms.Nx + 2;
                auto pt3_idx = pt0_idx + parms.Nx + 1;

                cell_type cl;
                cl.ptids = {{pt0_idx, pt1_idx, pt2_idx, pt3_idx}};
                cells.push_back(cl);

                face_type f0;
                f0.ptids = {pt0_idx, pt1_idx};
                if (j == 0) f0.is_boundary = true;
                faces.push_back(f0);

                face_type f1;
                f1.ptids = {pt1_idx, pt2_idx};
                if (i == parms.Nx-1) f1.is_boundary = true;
                faces.push_back(f1);

                face_type f2;
                f2.ptids = {pt3_idx, pt2_idx};
                if (j == parms.Ny-1) f2.is_boundary = true;
                faces.push_back(f2);

                face_type f3;
                f3.ptids = {pt0_idx, pt3_idx};
                if (i == 0) f3.is_boundary = true;
                faces.push_back(f3);

            }
        }

        std::sort(cells.begin(), cells.end());
        std::sort(faces.begin(), faces.end());
        faces.erase( std::unique(faces.begin(), faces.end()), faces.end() );

        
        for (auto& fc : faces)
        {
            if (fc.is_boundary)
                fc.bndtype = boundary::DIRICHLET;
        }
    }
    
    
    
    
    
    mesh_impl(const std::string& filename)
    {
        std::ifstream ifs(filename);

        size_t num_elems, num_nodes, node, dummy;
        T x, y;

        ifs >> num_elems;
        for (size_t i = 0; i < num_elems; i++)
        {
            ifs >> x >> y;
            point_type pt({x, y});
            points.push_back(pt);

            node_type n;
            n.ptid = i;
            nodes.push_back(n);
        }

        ifs >> num_elems;
        for (size_t i = 0; i < num_elems; i++)
        {
            ifs >> num_nodes;
            ifs >> dummy;

            cell_type c;
            for (size_t j = 0; j < num_nodes; j++)
            {
                ifs >> node;
                c.ptids.push_back(node);
            }
            cells.push_back(c);

            auto npts = c.ptids.size();
            for (size_t j = 0; j < npts; j++)
            {
                face_type f;
                f.ptids[0] = c.ptids[j];
                f.ptids[1] = c.ptids[(j+1)%npts];

                if (f.ptids[1] < f.ptids[0])
                    std::swap(f.ptids[0], f.ptids[1]);

                faces.push_back(f);
            }
        }

        std::sort(cells.begin(), cells.end());
        std::sort(faces.begin(), faces.end());
        faces.erase( std::unique(faces.begin(), faces.end()), faces.end() );

        ifs >> num_elems;
        for (size_t i = 0; i < num_elems; i++)
        {
            ifs >> dummy;

            face_type f;
            ifs >> f.ptids[0];
            ifs >> f.ptids[1];

            
            auto af = std::lower_bound(faces.begin(), faces.end(), f);
            if (af == faces.end())
                throw std::invalid_argument("Invalid face");

            (*af).is_boundary = true;
            (*af).bndtype = boundary::DIRICHLET;
        }

        ifs.close();
    }
};

//, typename Mesh
    
    


    
    
template<typename T,  typename CellUD, typename FaceUD, typename NodeUD  >
struct mesh_impl<T, 1, CellUD, FaceUD, NodeUD> {

    typedef point<T,2>          point_type;
    typedef cell<CellUD, 0>     cell_type;
    typedef face<FaceUD>        face_type;
    typedef node<NodeUD>        node_type;
    typedef CellUD              cell_ud_type;
    typedef FaceUD              face_ud_type;
    typedef NodeUD              node_ud_type;
    typedef T                   coordinate_type;

    std::vector<point_type>     points;
    std::vector<node_type>      nodes;
    std::vector<cell_type>      cells;
    
    //cell_type cl ;
    std::vector<point_type>     interface_vertices; // just the vertices of the mesh
    size_t degree_curve ;
    
    
    
    mesh_impl(){}
   
    


    mesh_impl& operator=( const mesh_impl& other)
    {
         
        if (this != &other)
        {
            points = other.points ;
            cells = other.cells ;
            nodes= other.nodes ;
            degree_curve = other.degree_curve ;
            interface_vertices = other.interface_vertices ;
           
        
        }
        return *this;
        
    }
    
    mesh_impl(  const mesh_impl& other )
    {
        points = other.points ;
        cells = other.cells ;
        nodes= other.nodes ;
        degree_curve = other.degree_curve ;
        interface_vertices = other.interface_vertices ;
        
    
    }
    
    mesh_impl( size_t degree_curve ): degree_curve(degree_curve){}
    
    
    /*
    mesh_impl( cell_type& cl , size_t degree_curve ): cl(cl),points(cl.user_data.interface),degree_curve(degree_curve),size_cls((points.size()-1)/degree_curve)
    {
        //interface_pts = cl.user_data.interface ;
       
        //if ( !is_cut(msh, cl) )
        //    exit(9);
        //size_cls = (points.size()-1)/degree_curve ;
        
        size_t point_num = 0;
        //cell_basis_Lagrange_1d_reference <Mesh,T> cb(msh, cl, basis_degree);
        //auto size_pts = interface_pts.size() ;
        //size_cls = (size_pts-1)/basis_degree ;
    
        for(size_t pos = 0 ; pos < size_cls ; pos++ )
        {
            cell_type cl;
            for (size_t i = 0; i <= degree_curve ; i++)
            {
                cl.ptids.push_back( point_num ) ;
                if( i < degree_curve )
                    point_num++ ;
                //cl.ptids = {{pt0_idx, pt1_idx, pt2_idx, pt3_idx}};
            }
            cells.push_back(cl);
        }
    
    }
    */
    
    /*
    template<typename Cell>
    void
    set_cell(Cell& other_cl){
        interface_vertices = other_cl.user_data.interface ;
        auto size_cls = (interface_vertices.size()-1)/degree_curve ;
        size_t point_num = 0;
        for(size_t pos = 0 ; pos < size_cls ; pos++ )
        {
            cell_type cl;
            for (size_t i = 0; i <= degree_curve ; i++)
            {
                cl.ptids.push_back( point_num ) ;
                node_type n;
                n.ptid = point_num;
                nodes.push_back(n);
                
                if( i < degree_curve )
                    point_num++ ;
            }
            cells.push_back(cl);
        }
        
    }
    
    template<typename Cell>
    void
    set_cell(Cell& other_cl,size_t m_degree){
        degree_curve = m_degree;
        interface_vertices = other_cl.user_data.interface ;
        auto size_cls = (interface_vertices.size()-1)/degree_curve ;
        size_t point_num = 0;
        for(size_t pos = 0 ; pos < size_cls ; pos++ )
        {
            cell_type cl;
            for (size_t i = 0; i <= degree_curve ; i++)
            {
                cl.ptids.push_back( point_num ) ;
                node_type n;
                n.ptid = point_num;
                nodes.push_back(n);
                
                if( i < degree_curve )
                    point_num++ ;
            }
            cells.push_back(cl);
        }
        auto last = std::unique(nodes.begin(), nodes.end());
        nodes.erase(last, nodes.end());
        
    }
    */
    template<typename Cell>
    void
    set_cell_new(Cell& other_cl,size_t m_degree){
        
        points.clear();
        cells.clear();
        nodes.clear();
        interface_vertices.clear();
        
        degree_curve = m_degree;
        points = other_cl.user_data.interface ;
        
        auto size_cls = (points.size()-1)/degree_curve ;
        
        
        for(size_t point_num = 0; point_num < points.size() ; point_num++ )
        {
            node_type n;
            n.ptid = point_num;
            nodes.push_back(n);
            
        }
        
        //size_t vertices = degree_curve;
        for(size_t pos = 0 ; pos < size_cls ; pos++ )
        {
            cell_type cl;
            auto pt0 = 0 + pos*degree_curve ;
            auto pt1 = degree_curve + pos*degree_curve ;
            cl.ptids = {{ pt0 , pt1 }};
            
            for (size_t i = 1 ; i < degree_curve ; i++)
            {
                auto pt_high_order = i + pos*degree_curve ;
                cl.ptids.push_back( pt_high_order ) ;
            }
            cells.push_back(cl);
        }
        
        for(size_t j = 0 ; j < points.size(); j+= degree_curve )
        {
            interface_vertices.push_back(points[j]);
        }
        
     
                
    }
    /*
    template<typename Cell>
    void
    set_cell_new2(Cell& other_cl,size_t m_degree){
        degree_curve = m_degree;
        
        interface_vertices = other_cl.user_data.interface ;
        
        points.resize( interface_vertices.size() );
        points = interface_vertices ;
        points[0] = interface_vertices[0] ;
        points[1] = interface_vertices[0] ;
        auto size_cls = (points.size()-1)/degree_curve ;
        
        size_t point_num = 0;
        size_t pt0_cl = 0;
        size_t pt1_cl = 1;
        for(size_t pos = 0 ; pos < size_cls ; pos++ )
        {
            cell_type cl;
            node_type n;
            if(pos == 0){
                points[point_num] = interface_vertices[pos] ;
                n.ptid = point_num++;
                nodes.push_back(n);
            }
            auto pt1 = degree_curve + pos*degree_curve ;
            points[point_num] = interface_vertices[pt1] ;
            n.ptid = point_num++;
            nodes.push_back(n);
            
            cl.ptids = {{ pt0_cl , pt1_cl }};
           
            for (size_t i = 1 ; i < degree_curve ; i++)
            {
                auto pt_high_order = i + pos*degree_curve ;
                points[point_num] = interface_vertices[pt_high_order] ;
                n.ptid = point_num++;
                nodes.push_back(n);
                auto pt_high_order_cl = pt1_cl + i  ;
                cl.ptids.push_back( pt_high_order_cl ) ;
            }
            cells.push_back(cl);
            pt0_cl = pt1_cl ;
            pt1_cl += degree_curve ;
        }
        
                
    }
    */
    
};

    
    

template<typename T, size_t ElemType, typename CellUD = void, typename FaceUD = void, typename NodeUD = void>
using mesh = mesh_impl<T, ElemType, CellUD, FaceUD, NodeUD>;

template<typename T, typename CellUD = void, typename FaceUD = void, typename NodeUD = void>
using quad_mesh = mesh_impl<T, 4, CellUD, FaceUD, NodeUD>;

template<typename T, typename CellUD = void, typename FaceUD = void, typename NodeUD = void>
using poly_mesh = mesh_impl<T, 0, CellUD, FaceUD, NodeUD>;

template<typename T, typename CellUD = void, typename FaceUD = void, typename NodeUD = void >
using integration_mesh = mesh_impl<T, 1, CellUD, FaceUD, NodeUD>;

#if 0






template<typename T>
bool load_netgen_2d(const std::string& filename, mesh<T>& msh)
{
    std::ifstream ifs(filename);
    if (!ifs.is_open())
    {
        std::cout << "Problem opening input mesh" << std::endl;
        return false;
    }

    size_t count;

    ifs >> count;
    msh.points.reserve(count);

    for (size_t i = 0; i < count; i++)
    {
        T x, y;
        ifs >> x >> y;
        msh.points.push_back( typename mesh<T>::point_type(x,y) );
    }

    std::cout << "Points: " << msh.points.size() << std::endl;

    ifs >> count;
    msh.cells.reserve(count);

    for (size_t i = 0; i < count; i++)
    {
        size_t dom, p0, p1, p2;
        ifs >> dom >> p0 >> p1 >> p2;

        if (dom == 0 || p0 == 0 || p1 == 0 || p2 == 0 )
        {
            std::cout << "Indices in netgen file should be 1-based, found a 0.";
            std::cout << std::endl;
            return false;
        }

        polygon p;
        p.domain_id = dom-1;
        p.point_ids.push_back(p0-1);
        p.point_ids.push_back(p1-1);
        p.point_ids.push_back(p2-1);
        msh.cells.push_back(p);
    }

    std::cout << "Cells: " << msh.cells.size() << std::endl;
    std::sort(msh.cells.begin(), msh.cells.end());

    msh.faces.reserve( 3 * msh.cells.size() );

    for (auto& cl : msh.cells)
    {
        assert(cl.point_ids.size() == 3);

        for (size_t i = 0; i < 3; i++)
        {
            auto p0 = cl.point_ids[i];
            auto p1 = cl.point_ids[(i+1)%3];

            if (p1 < p0)
                std::swap(p0, p1);

            typename mesh<T>::face_type face;
            face.point_ids[0] = p0;
            face.point_ids[1] = p1;

            msh.faces.push_back(face);
        }
    }

    std::sort(msh.faces.begin(), msh.faces.end());
    msh.faces.erase(std::unique(msh.faces.begin(), msh.faces.end()), msh.faces.end());

    std::cout << "Faces: " << msh.faces.size() << std::endl;

    for (size_t i = 0; i < msh.points.size(); i++)
    {
        typename mesh<T>::node_type node;
        node.point_id = i;
        msh.nodes.push_back(node);
    }

    std::cout << "Nodes: " << msh.nodes.size() << std::endl;

    ifs.close();
    return true;
}




#endif
