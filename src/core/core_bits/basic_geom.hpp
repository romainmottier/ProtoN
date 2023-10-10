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
#include <stdexcept>

#include "point.hpp"

template<typename Mesh>
size_t
offset(const Mesh& msh, const typename Mesh::cell_type& cl)
{
    auto itor = std::lower_bound(msh.cells.begin(), msh.cells.end(), cl);
    if ( itor == msh.cells.end() )
        throw std::logic_error("Cell not found: this is likely a bug.");

    return std::distance(msh.cells.begin(), itor);
}

template<typename Mesh>
size_t
offset(const Mesh& msh, const typename Mesh::face_type& fc)
{
    auto itor = std::lower_bound(msh.faces.begin(), msh.faces.end(), fc);
    if ( itor == msh.faces.end() )
        throw std::logic_error("Face not found: this is likely a bug.");

    return std::distance(msh.faces.begin(), itor);
}

template<typename Mesh>
size_t
offset(const Mesh& msh, const typename Mesh::node_type& n)
{
    auto itor = std::lower_bound(msh.nodes.begin(), msh.nodes.end(), n);
    if ( itor == msh.nodes.end() )
        throw std::logic_error("Node not found: this is likely a bug.");

    return std::distance(msh.nodes.begin(), itor);
}


/**********************************************************************************/
/**********************************************************************************/
/**********************************************************************************/
template<typename T, size_t N, typename CellUD, typename FaceUD, typename NodeUD>
std::array< typename mesh<T, N, CellUD, FaceUD, NodeUD>::node_type, N >
nodes(const mesh<T, N, CellUD, FaceUD, NodeUD>& msh,
      const typename mesh<T, N, CellUD, FaceUD, NodeUD>::cell_type& cl)
{
    using Mesh = mesh<T, N, CellUD, FaceUD, NodeUD>;

    std::array< typename Mesh::node_type, N > ret;

    auto ptid2node = [&](size_t ptid) -> auto {
        return msh.nodes.at(ptid);
    };

    std::transform( cl.ptids.begin(), cl.ptids.end(), ret.begin(), ptid2node );

    return ret;
}

template<typename T, typename CellUD, typename FaceUD, typename NodeUD>
std::vector< typename mesh<T, 0, CellUD, FaceUD, NodeUD>::node_type >
nodes(const mesh<T, 0, CellUD, FaceUD, NodeUD>& msh,
      const typename mesh<T, 0, CellUD, FaceUD, NodeUD>::cell_type& cl)
{
    using Mesh = mesh<T, 0, CellUD, FaceUD, NodeUD>;

    std::vector< typename Mesh::node_type> ret;
    ret.resize( cl.ptids.size() );

    auto ptid2node = [&](size_t ptid) -> auto {
        return msh.nodes.at(ptid);
    };

    std::transform( cl.ptids.begin(), cl.ptids.end(), ret.begin(), ptid2node );

    return ret;
}


template<typename T, typename CellUD, typename FaceUD, typename NodeUD>
std::vector< typename mesh<T, 1, CellUD, FaceUD, NodeUD>::node_type >
nodes(const mesh<T, 1, CellUD, FaceUD, NodeUD>& msh,
      const typename mesh<T, 0, CellUD, FaceUD, NodeUD>::cell_type& cl)
{
    using Mesh = mesh<T, 1, CellUD, FaceUD, NodeUD>;

    std::vector< typename Mesh::node_type> ret;
    ret.resize( cl.ptids.size() );

    auto ptid2node = [&](size_t ptid) -> auto {
        return msh.nodes.at(ptid);
    };

    std::transform( cl.ptids.begin(), cl.ptids.end(), ret.begin(), ptid2node );

    return ret;
}


template<typename Mesh>
std::array< typename Mesh::node_type, 2 >
nodes(const Mesh& msh, const typename Mesh::face_type& fc)
{
    std::array< typename Mesh::node_type, 2 > ret;

    auto ptid2node = [&](size_t ptid) -> auto {
        return msh.nodes.at(ptid);
    };

    std::transform( fc.ptids.begin(), fc.ptids.end(), ret.begin(), ptid2node );

    return ret;
}

/**********************************************************************************/
/**********************************************************************************/
/**********************************************************************************/
template<typename T, size_t N, typename CellUD, typename FaceUD, typename NodeUD>
std::array< typename mesh<T, N, CellUD, FaceUD, NodeUD>::point_type, N >
points(const mesh<T, N, CellUD, FaceUD, NodeUD>& msh,
       const typename mesh<T, N, CellUD, FaceUD, NodeUD>::cell_type& cl)
{
    using Mesh = mesh<T, N, CellUD, FaceUD, NodeUD>;
    std::array< typename Mesh::point_type, N > ret;

    auto ptid2pt = [&](size_t ptid) -> auto {
        return msh.points.at(ptid);
    };

    std::transform( cl.ptids.begin(), cl.ptids.end(), ret.begin(), ptid2pt );

    return ret;
}

template<typename T, typename CellUD, typename FaceUD, typename NodeUD>
std::vector< typename mesh<T, 0, CellUD, FaceUD, NodeUD>::point_type >
points(const mesh<T, 0, CellUD, FaceUD, NodeUD>& msh,
      const typename mesh<T, 0, CellUD, FaceUD, NodeUD>::cell_type& cl)
{
    using Mesh = mesh<T, 0, CellUD, FaceUD, NodeUD>;

    std::vector< typename Mesh::point_type> ret;
    ret.resize( cl.ptids.size() );

    auto ptid2pt = [&](size_t ptid) -> auto {
        return msh.points.at(ptid);
    };

    std::transform( cl.ptids.begin(), cl.ptids.end(), ret.begin(), ptid2pt );

    return ret;
}

template<typename T, typename CellUD, typename FaceUD, typename NodeUD>
std::vector< typename mesh<T, 1, CellUD, FaceUD, NodeUD>::point_type >
points(const mesh<T, 1, CellUD, FaceUD, NodeUD>& msh,
      const typename mesh<T, 0, CellUD, FaceUD, NodeUD>::cell_type& cl)
{
    using Mesh = mesh<T, 1, CellUD, FaceUD, NodeUD>;

    std::vector< typename Mesh::point_type> ret;
    ret.resize( cl.ptids.size() );

    auto ptid2pt = [&](size_t ptid) -> auto {
        return msh.points.at(ptid);
    };

    std::transform( cl.ptids.begin(), cl.ptids.end(), ret.begin(), ptid2pt );

    return ret;
}


template<typename Mesh>
std::array< typename Mesh::point_type, 2 >
points(const Mesh& msh, const typename Mesh::face_type& fc)
{
    std::array< typename Mesh::point_type, 2 > ret;

    auto ptid2pt = [&](size_t ptid) -> auto {
        return msh.points.at(ptid);
    };

    std::transform( fc.ptids.begin(), fc.ptids.end(), ret.begin(), ptid2pt );

    return ret;
}

template<typename Mesh>
typename Mesh::point_type
points(const Mesh& msh, const typename Mesh::node_type& node)
{
    return msh.points.at( node.ptid );
}

/**********************************************************************************/
/**********************************************************************************/
/**********************************************************************************/
template<typename T, size_t N, typename CellUD, typename FaceUD, typename NodeUD>
std::array< typename mesh<T, N, CellUD, FaceUD, NodeUD>::face_type, N >
faces(const mesh<T, N, CellUD, FaceUD, NodeUD>& msh,
      const typename mesh<T, N, CellUD, FaceUD, NodeUD>::cell_type& cl)
{
    using Mesh = mesh<T, N, CellUD, FaceUD, NodeUD>;
    typedef typename Mesh::face_type face_type;
    std::array< typename Mesh::face_type, N > ret;

    auto numpts = cl.ptids.size();

    for (size_t i = 0; i < numpts; i++)
    {

        face_type f;

        f.ptids[0] = cl.ptids[i];
        f.ptids[1] = cl.ptids[(i+1)%numpts];

        if ( f.ptids[0] > f.ptids[1] )
            std::swap(f.ptids[0], f.ptids[1]);

        auto itor = std::lower_bound(msh.faces.begin(), msh.faces.end(), f);
        if (itor == msh.faces.end())
            throw std::logic_error("Face not found, this is likely a bug.");
        ret[i] = *itor;
    }

    return ret;
}

template<typename T, typename CellUD, typename FaceUD, typename NodeUD>
std::vector< typename mesh<T, 0, CellUD, FaceUD, NodeUD>::face_type >
faces(const mesh<T, 0, CellUD, FaceUD, NodeUD>& msh,
      const typename mesh<T, 0, CellUD, FaceUD, NodeUD>::cell_type& cl)
{
    using Mesh = mesh<T, 0, CellUD, FaceUD, NodeUD>;
    typedef typename Mesh::face_type face_type;
    std::vector< typename Mesh::face_type > ret;

    auto numpts = cl.ptids.size();

    ret.resize(numpts);

    for (size_t i = 0; i < numpts; i++)
    {

        face_type f;

        f.ptids[0] = cl.ptids[i];
        f.ptids[1] = cl.ptids[(i+1)%numpts];

        if ( f.ptids[0] > f.ptids[1] )
            std::swap(f.ptids[0], f.ptids[1]);

        auto itor = std::lower_bound(msh.faces.begin(), msh.faces.end(), f);
        if (itor == msh.faces.end())
            throw std::logic_error("Face not found, this is likely a bug.");
        ret[i] = *itor;
    }

    return ret;
}

template<typename Iterator>
auto
barycenter(Iterator begin, Iterator end)
{
    typedef typename std::iterator_traits<Iterator>::value_type point_type;
    typedef typename point_type::value_type T;

    point_type  ret;
    T           den = 0.0;

    auto numpts = std::distance(begin, end);
    auto p0 = *begin;

    for (size_t i = 2; i < numpts; i++)
    {
        auto pprev  = *std::next(begin, i-1) - p0;
        auto pcur   = *std::next(begin, i) - p0;
        auto d      = det(pprev, pcur) / 2.0;
        ret         = ret + (pprev + pcur) * d;
        den         += d;
    }

    return p0 + ret/(den*3);
}

template<typename Mesh>
typename Mesh::point_type
barycenter(const Mesh& msh, const typename Mesh::cell_type& cl)
{
    auto pts = points(msh, cl);
    return barycenter(pts.begin(), pts.end());
}

template<typename Mesh>
typename Mesh::point_type
barycenter(const Mesh& msh, const typename Mesh::face_type& fc)
{
    auto pts = points(msh, fc);
    return (pts[0] + pts[1])/2.0;
}

template<typename Mesh>
typename Mesh::coordinate_type
diameter(const Mesh& msh, const typename Mesh::cell_type& cl)
{
    auto diam = 0.0;

    for (size_t i = 0; i < cl.ptids.size(); i++)
    {
        for (size_t j = i+1; j < cl.ptids.size(); j++)
        {
            auto p0 = msh.points.at( cl.ptids[i] );
            auto p1 = msh.points.at( cl.ptids[j] );
            diam = std::max( diam, (p1-p0).to_vector().norm() );
        }
    }

    return diam;
}

template<typename Mesh>
typename Mesh::coordinate_type
diameter(const Mesh& msh, const typename Mesh::face_type& fc)
{
    auto p0 = msh.points.at( fc.ptids[0] );
    auto p1 = msh.points.at( fc.ptids[1] );

    return (p1-p0).to_vector().norm();
}

template<typename Mesh>
typename Mesh::coordinate_type
measure(const Mesh& msh, const typename Mesh::cell_type& cl)
{
    using T = typename Mesh::coordinate_type;

    auto pts = points(msh, cl);

    T acc{};
    for (size_t i = 1; i < pts.size() - 1; i++)
    {
        auto u = (pts.at(i) - pts.at(0)).to_vector();
        auto v = (pts.at(i+1) - pts.at(0)).to_vector();
        acc += (u(0)*v(1) - u(1)*v(0)) * 0.5;
        // acc is negative when the point are sorted clockwise
    }

    return acc;
}

template<typename Mesh>
typename Mesh::coordinate_type
measure(const Mesh& msh, const typename Mesh::face_type& fc)
{
    auto p0 = msh.points.at( fc.ptids[0] );
    auto p1 = msh.points.at( fc.ptids[1] );

    return (p1-p0).to_vector().norm();
}

/**********************************************************************************/
/**********************************************************************************/
/**********************************************************************************/
template<typename T, size_t N, typename CellUD, typename FaceUD, typename NodeUD>
std::array< Eigen::Matrix<typename mesh<T, N, CellUD, FaceUD, NodeUD>::coordinate_type, 2, 1>, N >
normals(const mesh<T, N, CellUD, FaceUD, NodeUD>& msh,
        const typename mesh<T, N, CellUD, FaceUD, NodeUD>::cell_type& cl)
{
    //using Mesh = mesh<T, N, CellUD, FaceUD, NodeUD>;
    std::array< Eigen::Matrix<T,2,1>, N >  ret;

    auto pts = points(msh, cl);
    auto numpts = pts.size();
    assert(numpts == N);

    for (size_t i = 0; i < numpts; i++)
    {
        auto v = pts[(i+1)%numpts] - pts[i];


        ret[i](0) = v.y();
        ret[i](1) = -v.x();
        ret[i] = ret[i]/ret[i].norm();
    }

    return ret;
}


template<typename T, typename CellUD, typename FaceUD, typename NodeUD>
std::vector< Eigen::Matrix<typename mesh<T, 0, CellUD, FaceUD, NodeUD>::coordinate_type, 2, 1> >
normals(const mesh<T, 0, CellUD, FaceUD, NodeUD>& msh,
        const typename mesh<T, 0, CellUD, FaceUD, NodeUD>::cell_type& cl)
{
    //using Mesh = mesh<T, 0, CellUD, FaceUD, NodeUD>;
    std::vector< Eigen::Matrix<T,2,1> > ret;

    auto pts = points(msh, cl);
    auto numpts = pts.size();

    ret.resize(numpts);

    for (size_t i = 0; i < numpts; i++)
    {
        auto v = pts[(i+1)%numpts] - pts[i];


        ret[i](0) = v.y();
        ret[i](1) = -v.x();
        ret[i] = ret[i]/ret[i].norm();
    }

    return ret;
}


#if 0

template<typename Mesh>
std::vector< typename Mesh::point_type >
make_test_points(const Mesh& msh, const typename Mesh::cell_type& cl)
{
    const size_t N = 10;

    auto trans = make_reference_transform(msh, cl);

    std::vector< typename Mesh::point_type > ret;

    auto cell_pts = points(msh, cl);

    auto min = -1.0;
    auto h = 2.0/N;

    for (size_t j = 0; j < N+1; j++)
    {
        for(size_t i = 0; i < N+1; i++)
        {
            auto p_xi = min + i*h;
            auto p_eta = min + j*h;
            typename Mesh::point_type pt(p_xi, p_eta);
            ret.push_back( trans.ref_to_phys(pt) );
        }
    }

    return ret;
}

template<typename Mesh>
std::vector< typename Mesh::point_type >
make_test_points(const Mesh& msh, const typename Mesh::face_type& fc)
{
    const size_t N = 10;

    std::vector< typename Mesh::point_type > ret;

    auto face_pts = points(msh, fc);

    auto min = face_pts[0];

    auto h = (face_pts[1] - face_pts[0])/N;

    for (size_t i = 0; i < N+1; i++)
    {
        auto p = min + i*h;
        ret.push_back(p);
    }

    return ret;
}

#endif

