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
#include <iterator>
#include "cuthho_mesh.hpp"

template<typename T, size_t ET>
bool
is_cut(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl) {
    assert(cl.user_data.location != element_location::UNDEF);
    return cl.user_data.location == element_location::ON_INTERFACE;
}

template<typename T, size_t ET>
bool
is_cut(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc) {
    assert(fc.user_data.location != element_location::UNDEF);
    return fc.user_data.location == element_location::ON_INTERFACE;
}

template<typename T, size_t ET>
element_location
location(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl) {
    assert(cl.user_data.location != element_location::UNDEF);
    return cl.user_data.location;
}

template<typename T, size_t ET>
element_location
location(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc) {
    assert(fc.user_data.location != element_location::UNDEF);
    return fc.user_data.location;
}

template<typename T, size_t ET>
element_location
location(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::node_type& n) {
    assert(n.user_data.location != element_location::UNDEF);
    return n.user_data.location;
}

template<typename T, typename Function>
point<T, 2>
find_zero_crossing(const point<T,2>& p0, const point<T,2>& p1, const Function& level_set_function, const T& threshold) {

    /* !!! We assume that the level set function *has* a zero crossing
     * between p0 and p1 !!! */
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
        auto la = level_set_function(pa);
        auto lb = level_set_function(pb);
        auto lm = level_set_function(pm);

        if ( (lb >= 0 && lm >= 0) || (lb < 0 && lm < 0) ) {   /* intersection is between pa and pm */
            pm_prev = pm;
            pb = pm;
            pm = (pa+pb)/2.0;
        }
        else {   /* intersection is between pm and pb */
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

template<typename T, size_t ET, typename Function>
void
detect_node_position(cuthho_mesh<T, ET>& msh, const Function& level_set_function) {
    for (auto& n : msh.nodes) {
        auto pt = points(msh, n);
        if ( level_set_function(pt) < 0 )
            n.user_data.location = element_location::IN_NEGATIVE_SIDE;
        else
            n.user_data.location = element_location::IN_POSITIVE_SIDE;
    }
}

template<typename T, size_t ET, typename Function>
void
detect_cut_faces(cuthho_mesh<T, ET>& msh, const Function& level_set_function) {

    for (auto& fc : msh.faces) {
        auto pts = points(msh, fc);
        auto l0 = level_set_function(pts[0]);
        auto l1 = level_set_function(pts[1]);
        if (l0 >= 0 && l1 >= 0) {
            fc.user_data.location = element_location::IN_POSITIVE_SIDE;
            continue;
        }

        if (l0 < 0 && l1 < 0) {
            fc.user_data.location = element_location::IN_NEGATIVE_SIDE;
            continue;
        }

        auto threshold = diameter(msh, fc) / 1e20;
        auto pm = find_zero_crossing(pts[0], pts[1], level_set_function, threshold);

        fc.user_data.node_inside = ( l0 < 0 ) ? 0 : 1;
        fc.user_data.location = element_location::ON_INTERFACE;
        fc.user_data.intersection_point = pm;
    }
}

template<typename T, size_t ET, typename Function>
void
detect_cut_cells(cuthho_mesh<T, ET>& msh, const Function& level_set_function) {

    typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;

    size_t cell_i = 0;
    for (auto& cl : msh.cells) {

        auto fcs = faces(msh, cl);
        std::array< std::pair<size_t, point_type>, 2 >  cut_faces;

        size_t k = 0;
        for (size_t i = 0; i < fcs.size(); i++) {
            bool face_is_cut_Q = is_cut(msh, fcs[i]);
            if ( face_is_cut_Q )
                cut_faces.at(k++) = std::make_pair(i, fcs[i].user_data.intersection_point);
        }

        /* If a face is cut, the cells that own the face are cut. Is this
         * unconditionally true? It should...fortunately this isn't avionics
         * software */

        if (k == 0) {
            auto is_positive = [&](const point_type& pt) -> bool {
                return level_set_function(pt) > 0;
            };

            auto pts = points(msh, cl);
            if ( std::all_of(pts.begin(), pts.end(), is_positive) )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
        }

        if (k == 2) {
            cl.user_data.location = element_location::ON_INTERFACE;
            auto p0 = cut_faces[0].second;
            auto p1 = cut_faces[1].second;
            auto pt = p1 - p0;
            auto pt_t = point<T,2>(-pt.y(), pt.x());
            auto pn = p0 + pt_t;

            if ( level_set_function(pn) >= 0 ) {
                cl.user_data.p0 = p1;
                cl.user_data.p1 = p0;
            }
            else {
                cl.user_data.p0 = p0;
                cl.user_data.p1 = p1;
            }

            cl.user_data.interface.push_back(cl.user_data.p0);
            cl.user_data.interface.push_back(cl.user_data.p1);
        }

        if ( k != 0 && k != 2 )
            throw std::logic_error("invalid number of cuts in cell");

        cell_i++;
    }
}

#ifdef USE_OLD_DISPLACEMENT
template<typename T, size_t ET, typename Function>
void
move_nodes(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;

    for (auto& fc : msh.faces)
    {
        if ( location(msh, fc) == element_location::ON_INTERFACE )
        {
            auto nds = nodes(msh, fc);
            auto pts = points(msh, fc);
            auto lf = (pts[1] - pts[0]).to_vector().norm();
            auto dp = (fc.user_data.intersection_point - pts[0]).to_vector().norm();

            auto closeness = dp/lf;

            typename cuthho_mesh<T, ET>::node_type ntc;

            if (closeness < 0.45) //pts[0] is too close
                ntc = nds[0];
            else if (closeness > 0.55) // pts[1] is too close
                ntc = nds[1];
            else // nothing to do
                continue;

            auto ofs = offset(msh, ntc);

            msh.nodes.at( ofs ).user_data.displaced = true;

            T displacement;

            if ( location(msh, ntc) == element_location::IN_NEGATIVE_SIDE )
                displacement = -std::abs(0.5-closeness)*lf*0.7;
            else
                displacement = std::abs(0.5-closeness)*lf*0.7;

            Matrix<T, 2, Dynamic> normal = level_set_function.normal(fc.user_data.intersection_point) * displacement;

            typename cuthho_mesh<T, ET>::point_type p({normal(0), normal(1)});

            msh.points.at(ofs) = msh.points.at(ofs) + p;
        }
    }

    for (auto& cl : msh.cells)
    {
        auto nds = nodes(msh, cl);
        for (auto& n : nds)
            if (n.user_data.displaced)
                cl.user_data.distorted = true;
    }

    for (auto& cl : msh.cells) /* Check we didn't generate concave cells */
    {
        if (!cl.user_data.distorted)
            continue;

        auto pts = points(msh, cl);

        if (pts.size() < 4)
            continue;

        for (size_t i = 0; i < pts.size(); i++)
        {
            auto pa = pts[i];
            auto pb = pts[(i+1)%pts.size()];
            auto pc = pts[(i+2)%pts.size()];
            auto v1 = (pb - pa);
            auto v2 = (pc - pb);
            auto cross = v1.x() * v2.y() - v2.x() * v1.y();

            if ( cross < 0 )
                std::cout << red << "[ !WARNING! ] A concave polygon was generated (cell " << offset(msh, cl) << ")." << nocolor << std::endl;
        }
    }
}

#else

template<typename T, size_t ET, typename Function>
void
move_nodes(cuthho_mesh<T, ET>& msh, const Function& level_set_function) {

    typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;

    T closeness_thresh = 0.4;

    for (auto& fc : msh.faces) {
        if ( location(msh, fc) == element_location::ON_INTERFACE ) {
            auto nds = nodes(msh, fc);
            auto pts = points(msh, fc);
            auto bar = (pts[1] + pts[0])/2.0;

            auto lf = (pts[1] - pts[0]).to_vector().norm();
            auto dp = (fc.user_data.intersection_point - pts[0]).to_vector().norm();

            auto closeness = dp/lf;

            typename cuthho_mesh<T, ET>::node_type ntc;

            if (closeness < closeness_thresh) //pts[0] is too close
                ntc = nds[0];
            else if (closeness > 1.0-closeness_thresh) // pts[1] is too close
                ntc = nds[1];
            else // nothing to do
                continue;

            auto ofs = offset(msh, ntc);
            auto delta = (bar - fc.user_data.intersection_point)/2;
            msh.nodes.at( ofs ).user_data.displacement = msh.nodes.at( ofs ).user_data.displacement - delta;
            msh.nodes.at( ofs ).user_data.displaced = true;
        }
    }

    for (auto& nd : msh.nodes)
        if (nd.user_data.displaced)
            msh.points.at( nd.ptid ) = msh.points.at( nd.ptid ) + nd.user_data.displacement;


    for (auto& cl : msh.cells) {
        auto nds = nodes(msh, cl);
        for (auto& n : nds)
            if (n.user_data.displaced)
                cl.user_data.distorted = true;
    }

    for (auto& cl : msh.cells) {
        if (!cl.user_data.distorted)
            continue;

        auto pts = points(msh, cl);

        if (pts.size() < 4)
            continue;

        for (size_t i = 0; i < pts.size(); i++) {
            auto pa = pts[i];
            auto pb = pts[(i+1)%pts.size()];
            auto pc = pts[(i+2)%pts.size()];
            auto v1 = (pb - pa);
            auto v2 = (pc - pb);
            auto cross = v1.x() * v2.y() - v2.x() * v1.y();
            if (cross < 0) {
                std::cout << red << "[ !WARNING! ] A concave polygon was generated (cell " << offset(msh, cl) << ")." << nocolor << std::endl;
                throw std::logic_error("concave poly");
            }
        }
    }
}

#endif

template<typename T, size_t ET>
std::array< typename cuthho_mesh<T, ET>::point_type, 2 >
points(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc, const element_location& where) {

    if ( location(msh, fc) != where && location(msh, fc) != element_location::ON_INTERFACE )
        throw std::invalid_argument("This face has no points where requested");

    auto pts = points(msh, fc);
    if ( !is_cut(msh, fc) )
        return pts;

    auto nds = nodes(msh, fc);
    if ( location(msh, nds[0]) == where && location(msh, nds[1]) != where )
        pts[1] = fc.user_data.intersection_point;
    else if ( location(msh, nds[0]) != where && location(msh, nds[1]) == where )
        pts[0] = fc.user_data.intersection_point;
    else
        throw std::logic_error("Invalid point configuration");

    return pts;
}

template<typename T>
auto
barycenter(const std::vector< point<T, 2> >& pts) {
    return barycenter(pts.begin(), pts.end());
}

template<typename T, size_t ET>
auto
barycenter(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, element_location where) {
    if ( !is_cut(msh, cl) )
        return barycenter(msh, cl);

    auto tp = collect_triangulation_points(msh, cl, where);
    auto bar = barycenter(tp);

    return bar;
}

template<typename T, size_t ET, typename Function>
void
refine_interface(cuthho_mesh<T, ET>& msh, typename cuthho_mesh<T, ET>::cell_type& cl, const Function& level_set_function, size_t min, size_t max) {
    
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

    auto lm = level_set_function(pm);
    auto ls1 = level_set_function(ps1);
    auto ls2 = level_set_function(ps2);

    point_type ip;

    if ( !((lm >= 0 && ls1 >= 0) || (lm < 0 && ls1 < 0)) ) {
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing(pm, ps1, level_set_function, threshold);
    }
    else if ( !((lm >= 0 && ls2 >= 0) || (lm < 0 && ls2 < 0)) ) {
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing(pm, ps2, level_set_function, threshold);
    }
    else
        throw std::logic_error("interface not found in search range");

    cl.user_data.interface.at(mid) = ip;

    refine_interface(msh, cl, level_set_function, min, mid);
    refine_interface(msh, cl, level_set_function, mid, max);
}

template<typename T, size_t ET, typename Function>
void
refine_interface(cuthho_mesh<T, ET>& msh, const Function& level_set_function, size_t levels) {

    if (levels == 0)
        return;

    size_t interface_points = iexp_pow(2, levels);

    for (auto& cl : msh.cells) {
        if ( !is_cut(msh, cl) )
            continue;

        cl.user_data.interface.resize(interface_points+1);
        cl.user_data.interface.at(0)                = cl.user_data.p0;
        cl.user_data.interface.at(interface_points) = cl.user_data.p1;
        refine_interface(msh, cl, level_set_function, 0, interface_points);

    }
}

template<typename T, size_t ET>
std::vector< typename cuthho_mesh<T, ET>::point_type >
collect_triangulation_points(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, const element_location& where) {

    typedef typename cuthho_mesh<T, ET>::point_type     point_type;
    typedef typename cuthho_mesh<T, ET>::node_type      node_type;

    assert( is_cut(msh, cl) );
    auto ns = nodes(msh, cl);

    std::vector< point_type > ret;

    auto node2pt = [&](const node_type& n) -> auto {
        return msh.points.at(n.ptid);
    };

    auto insert_interface = [&](void) -> void {
        if (where == element_location::IN_NEGATIVE_SIDE)
            ret.insert(ret.end(), cl.user_data.interface.begin(), cl.user_data.interface.end());
        else if (where == element_location::IN_POSITIVE_SIDE)
            ret.insert(ret.end(), cl.user_data.interface.rbegin(), cl.user_data.interface.rend());
        else
            throw std::logic_error("If you've got here there is some issue...");
    };

    bool case1 = location(msh, ns.front()) == where && location(msh, ns.back()) != where;
    bool case2 = location(msh, ns.front()) != where && location(msh, ns.back()) == where;
    bool case3 = location(msh, ns.front()) != where && location(msh, ns.back()) != where;
    //bool case4 = location(msh, ns.front()) == where && location(msh, ns.back()) == where;

    if ( case1 || case2 || case3 ) {
        for (size_t i = 0; i < ns.size(); i++)
            if ( location(msh, ns[i]) == where )
                ret.push_back( node2pt(ns[i]) );

        insert_interface();
    }
    else  {
        size_t i = 0;
        while ( i < ns.size() && location(msh, ns.at(i)) == where )
            ret.push_back( node2pt(ns.at(i++)) );
        insert_interface();
        while ( i < ns.size() && location(msh, ns.at(i)) != where )
            i++;
        while ( i < ns.size() && location(msh, ns.at(i)) == where )
            ret.push_back( node2pt(ns.at(i++)) );
    }

    return ret;
}

template<typename T>
struct temp_tri {

    std::array< point<T,2>, 3 > pts;

    T area() const {
        auto v1 = pts[1] - pts[0];
        auto v2 = pts[2] - pts[0];

        return ( v1.x()*v2.y() - v2.x()*v1.y() ) / 2.0;
        // can be negative
    }
};

template<typename T>
std::ostream&
operator<<(std::ostream& os, const temp_tri<T>& t) {
    os << "line([" << t.pts[0].x() << "," << t.pts[1].x() << "],[" << t.pts[0].y() << "," << t.pts[1].y() << "]);" << std::endl;
    os << "line([" << t.pts[1].x() << "," << t.pts[2].x() << "],[" << t.pts[1].y() << "," << t.pts[2].y() << "]);" << std::endl;
    os << "line([" << t.pts[2].x() << "," << t.pts[0].x() << "],[" << t.pts[2].y() << "," << t.pts[0].y() << "]);" << std::endl;
    return os;
}


template<typename T, size_t ET>
typename cuthho_mesh<T, ET>::point_type
tesselation_center(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, element_location where) {

    auto fcs = faces(msh, cl);
    auto pts = points(msh, cl);
    auto nds = nodes(msh, cl);

    if (fcs.size() != 4)
        throw std::invalid_argument("This works only on quads for now");

    if( !is_cut(msh, cl) )
        throw std::invalid_argument("No tesselation centers for uncut cells");

    // if two consecutive faces are cut
    // return either the common node or the opposite node
    for (size_t i = 0; i < fcs.size(); i++) {
        auto f1 = i;
        auto f2 = (i+1) % fcs.size();
        auto n = (i+1) % fcs.size();

        if ( is_cut(msh, fcs[f1]) && is_cut(msh, fcs[f2]) ) {
            if ( location(msh, nds[n]) == where )
                return pts[n];
            else
                return pts[(n+2)%4];
        }
    }

    // if two opposite faces are cut
    // return the center of one of the other faces
    for (size_t i = 0; i < 2; i++) {
        auto f1 = i;
        auto f2 = i+2;
        auto n = i+1;
        if ( is_cut(msh, fcs[f1]) && is_cut(msh, fcs[f2]) ) {
            if( location(msh, nds[n]) == where )
                return 0.5*(pts[n] + pts[n+1]);
            else
                return 0.5*(pts[(n+2)%4] + pts[(n+3)%4]);
        }
    }

    // normally the tesselation center is already found
    throw std::logic_error("we shouldn't arrive here !!");
}

template<typename T, size_t ET>
std::vector<temp_tri<T>>
triangulate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, element_location where) {

    assert( is_cut(msh, cl) );

    auto tp = collect_triangulation_points(msh, cl, where);
    // auto bar = barycenter(tp);
    auto bar = tesselation_center(msh, cl, where);

    std::vector<temp_tri<T>> tris;

    for (size_t i = 0; i < tp.size(); i++) {
        temp_tri<T> t;
        t.pts[0] = bar;
        t.pts[1] = tp[i];
        t.pts[2] = tp[(i+1)%tp.size()];
        tris.push_back(t);
    }

    return tris;
}

template<typename T, size_t ET>
T
measure(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, const element_location& where) {

    if ( !is_cut(msh, cl) ) /* Element is not cut, use std. integration */
        return measure(msh, cl);

    T totmeas = 0.0;
    auto qpsi = integrate(msh, cl, 0, where);
    for (auto& qp : qpsi) {
        totmeas += qp.second;
    }

    return totmeas;
}

template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
make_integrate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, size_t degree, const element_location& where) {

    std::vector< std::pair<point<T,2>, T> > ret;

    if ( location(msh, cl) != where && location(msh, cl) != element_location::ON_INTERFACE )
        return ret;

    if ( !is_cut(msh, cl) ) /* Element is not cut, use std. integration */
        return integrate(msh, cl, degree);

    auto tris = triangulate(msh, cl, where);
    for (auto& tri : tris) {
        auto qpts = triangle_quadrature(tri.pts[0], tri.pts[1], tri.pts[2], degree);
        ret.insert(ret.end(), qpts.begin(), qpts.end());
    }

    return ret;
}

template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
make_integrate_with_mapping(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, size_t degree, const element_location& where) {

    std::vector< std::pair<point<T,2>, T> > ret;

    if ( location(msh, cl) != where && location(msh, cl) != element_location::ON_INTERFACE )
        return ret;

    if ( !is_cut(msh, cl) ) /* Element is not cut, use std. integration */
        return integrate(msh, cl, degree);

    auto tris = triangulate(msh, cl, where);
    for (auto& tri : tris) {
        auto qpts = triangle_quadrature(tri.pts[0], tri.pts[1], tri.pts[2], degree);
        ret.insert(ret.end(), qpts.begin(), qpts.end());
    }

    return ret;
}

template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
integrate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, size_t degree, const element_location& where) {

    if( cl.user_data.integration_n.size() != 0 && where == element_location::IN_NEGATIVE_SIDE)
        return cl.user_data.integration_n;

    if( cl.user_data.integration_p.size() != 0 && where == element_location::IN_POSITIVE_SIDE)
        return cl.user_data.integration_p;

    return make_integrate(msh, cl, degree, where);
}

template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
integrate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc, size_t degree, const element_location& where) {

    std::vector< std::pair<point<T,2>, T> > ret;

    if ( location(msh, fc) != where && location(msh, fc) != element_location::ON_INTERFACE )
        return ret;

    if ( !is_cut(msh, fc) ) /* Element is not cut, use std. integration */
        return integrate(msh, fc, degree);

    auto pts = points(msh, fc, where);

    auto scale = pts[1] - pts[0];
    auto meas = scale.to_vector().norm();

    auto qps = edge_quadrature<T>(degree);

    for (auto itor = qps.begin(); itor != qps.end(); itor++) {
        auto qp = *itor;
        //auto p = qp.first.x() * scale + pts[0];
        auto t = qp.first.x();
        auto p = 0.5*(1-t)*pts[0] + 0.5*(1+t)*pts[1];
        auto w = qp.second * meas * 0.5;

        ret.push_back( std::make_pair(p, w) );
    }

    return ret;
}

template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
integrate_interface(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, size_t degree, element_location where) {

    assert( is_cut(msh, cl) );

    typedef typename cuthho_mesh<T, ET>::point_type point_type;

    std::vector< std::pair<point<T,2>, T> > ret;

    auto pa = cl.user_data.interface.at(0);
    auto pb = cl.user_data.interface.at(1);
    auto bar = barycenter(msh, cl, where);

    auto va = (pa - bar).to_vector();
    auto vb_temp = pb - pa;
    auto vb = point_type({vb_temp.y(), -vb_temp.x()}).to_vector();

    auto int_sign = va.dot(vb) < 0 ? -1.0 : +1.0;

    auto qps = edge_quadrature<T>(degree);

    for (size_t i = 1; i < cl.user_data.interface.size(); i++) {

        auto p0 = cl.user_data.interface.at(i-1);
        auto p1 = cl.user_data.interface.at(i);

        auto scale = p1 - p0;
        auto meas = scale.to_vector().norm();

        for (auto itor = qps.begin(); itor != qps.end(); itor++) {
            auto qp = *itor;
            //auto p = qp.first.x() * scale + p0;
            auto t = qp.first.x();
            auto p = 0.5*(1-t)*p0 + 0.5*(1+t)*p1;
            auto w = int_sign * qp.second * meas * 0.5;
            ret.push_back( std::make_pair(p, w) );
        }
    }

    return ret;
}

template<typename T, typename Function>
std::vector< typename cuthho_quad_mesh<T>::point_type >
make_test_points(const cuthho_quad_mesh<T>& msh, const typename cuthho_quad_mesh<T>::cell_type& cl, const Function& level_set_function, element_location where) {

    const size_t N = 10;

    auto trans = make_reference_transform(msh, cl);

    std::vector< typename cuthho_quad_mesh<T>::point_type > ret;

    auto cell_pts = points(msh, cl);

    auto min = -1.0;
    auto h = 2.0/N;

    for (size_t j = 0; j < N+1; j++) {
        for(size_t i = 0; i < N+1; i++) {
            auto p_xi = min + i*h;
            auto p_eta = min + j*h;
            typename cuthho_quad_mesh<T>::point_type pt(p_xi, p_eta);
            auto phys_pt = trans.ref_to_phys(pt);
            if ( level_set_function(phys_pt) < 0 && where == element_location::IN_NEGATIVE_SIDE)
              ret.push_back( phys_pt );
            else if ( level_set_function(phys_pt) > 0 && where == element_location::IN_POSITIVE_SIDE)
              ret.push_back( phys_pt );
        }
    }

    return ret;
}

// NE FONCTIONNE PAS EN AGGLO SI PAS DE MODIFICATION DE MERGE_CELL
template<typename T, size_t ET>
std::vector<Matrix<T,2,1>> get_discrete_normal(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, size_t degree, element_location where) {

    assert(is_cut(msh, cl));
    
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    std::vector<Matrix<T,2,1>> ret;
    
    auto pa = cl.user_data.interface.at(0);
    auto pb = cl.user_data.interface.at(1);
    auto bar = barycenter(msh, cl, where);
    auto va = (pa - bar).to_vector();
    auto vb_temp = pb - pa;
    auto vb = point_type({vb_temp.y(), -vb_temp.x()}).to_vector();
    auto int_sign = va.dot(vb) < 0 ? -1.0 : +1.0;
    auto qps = edge_quadrature<T>(degree);
    for (size_t i = 1; i < cl.user_data.interface.size(); i++) {
        auto p0 = cl.user_data.interface.at(i-1); 
        auto p1 = cl.user_data.interface.at(i);
        auto scale = p1 - p0;
        Matrix<T,2,1> ni;
        auto meas = scale.to_vector().norm();
        ni(0) =  scale[1] / meas;
        ni(1) = -scale[0] / meas;
        // std::cout << meas << std::endl << std::endl; 
        for (auto itor = qps.begin(); itor != qps.end(); itor++) 
            ret.push_back(ni);
    }

    return ret;

}

template<typename T, size_t ET>
void dump_mesh(const cuthho_mesh<T, ET>& msh) {
    std::ofstream ofs("mesh_dump.m");
    size_t i = 0;
    ofs << "hold on;" << std::endl;
    for (auto& fc : msh.faces) {
        auto pts = points(msh, fc);
        if (fc.is_boundary)
            ofs << "line([" << pts[0].x() << ", " << pts[1].x() << "], [" << pts[0].y() << ", " << pts[1].y() << "], 'Color', 'r');" << std::endl;
        else if ( is_cut(msh, fc) )
            ofs << "line([" << pts[0].x() << ", " << pts[1].x() << "], [" << pts[0].y() << ", " << pts[1].y() << "], 'Color', 'g');" << std::endl;
        else
            ofs << "line([" << pts[0].x() << ", " << pts[1].x() << "], [" << pts[0].y() << ", " << pts[1].y() << "], 'Color', 'k');" << std::endl;

        auto bar = barycenter(msh, fc);
        ofs << "text(" << bar.x() << ", " << bar.y() << ", '" << i << "');" << std::endl;

        i++;
    }

    i = 0;
    for (auto& cl : msh.cells) {
        auto bar = barycenter(msh, cl);
        ofs << "text(" << bar.x() << ", " << bar.y() << ", '" << i << "');" << std::endl;

        if ( is_cut(msh, cl) ) {
            auto p0 = cl.user_data.p0;
            auto p1 = cl.user_data.p1;
            auto q = p1 - p0;
            ofs << "quiver(" << p0.x() << ", " << p0.y() << ", " << q.x() << ", " << q.y() << ", 0)" << std::endl;

            for (size_t i = 1; i < cl.user_data.interface.size(); i++) {
                auto s = cl.user_data.interface.at(i-1);
                auto l = cl.user_data.interface.at(i) - s;
                ofs << "quiver(" << s.x() << ", " << s.y() << ", " << l.x() << ", " << l.y() << ", 0)" << std::endl;
            }

            for (auto& ip : cl.user_data.interface)
                ofs << "plot(" << ip.x() << ", " << ip.y() << ", '*k');" << std::endl;

            auto tpn = collect_triangulation_points(msh, cl, element_location::IN_NEGATIVE_SIDE);
            auto bn = barycenter(tpn);
            ofs << "plot(" << bn.x() << ", " << bn.y() << ", 'dr');" << std::endl;

            auto tpp = collect_triangulation_points(msh, cl, element_location::IN_POSITIVE_SIDE);
            auto bp = barycenter(tpp);
            ofs << "plot(" << bp.x() << ", " << bp.y() << ", 'db');" << std::endl;

        }
        i++;
    }

    ofs << "t = linspace(0,2*pi,1000);plot(sqrt(0.15)*cos(t)+0.5, sqrt(0.15)*sin(t)+0.5)" << std::endl;

    ofs.close();
}

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////                                            ////////////////////////////
////////////////////               CUT BASIS                    ////////////////////////////
////////////////////                                            ////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////  CELL CUT BASIS  ///////////////////////////
template<typename Mesh, typename VT>
class cut_cell_basis {

    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;
    point_type          cell_bar;
    coordinate_type     cell_h, cell_hx, cell_hy;
    size_t              basis_degree, basis_size;

public:

    cut_cell_basis(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, element_location where) {

        auto loc = location(msh, cl);
        if(!is_cut(msh, cl)) {
            cell_bar     = barycenter(msh, cl);
            cell_h       = diameter(msh, cl);
            cell_hx      = cell_h;
            cell_hy      = cell_h;
            basis_degree = degree;
            basis_size   = (basis_degree+2)*(basis_degree+1)/2;
        }
        else {
            cell_bar     = barycenter(msh, cl, where);
            cell_h       = diameter(msh, cl, where);
            // cell_hx      = cell_h;
            // cell_hy      = cell_h;
            cell_hx      = compute_hx_hy(msh, cl, where).first; 
            cell_hy      = compute_hx_hy(msh, cl, where).second; 
            // BASIS INFOS
            basis_degree = degree;
            basis_size   = (basis_degree+2)*(basis_degree+1)/2;
        }
    }

    Matrix<VT, Dynamic, 1> eval_basis(const point_type& pt) {

        Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size);

        auto bx = (pt.x() - cell_bar.x()) / (0.5*cell_hx);
        auto by = (pt.y() - cell_bar.y()) / (0.5*cell_hy);

#ifdef POWER_CACHE
        if ( power_cache.size() != (basis_degree+1)*2 )
          power_cache.resize( (basis_degree+1)*2);

        power_cache[0] = 1.0;
        power_cache[1] = 1.0;
        for (size_t i = 1; i <= basis_degree; i++)
        {
            power_cache[2*i]    = iexp_pow(bx, i);
            power_cache[2*i+1]  = iexp_pow(by, i);
        }
#endif

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++) {
            for (size_t i = 0; i <= k; i++) {
                auto pow_x = k-i;
                auto pow_y = i;

#ifdef POWER_CACHE
                auto bv = power_cache[2*pow_x] * power_cache[2*pow_y+1];
#else
                auto bv = iexp_pow(bx, pow_x) * iexp_pow(by, pow_y);
#endif
                ret(pos++) = bv;
            }
        }

        assert(pos == basis_size);

        return ret;
        
    }

    Matrix<VT, Dynamic, 2>
    eval_gradients(const point_type& pt) {

        Matrix<VT, Dynamic, 2> ret = Matrix<VT, Dynamic, 2>::Zero(basis_size, 2);
        auto bx = (pt.x() - cell_bar.x()) / (0.5*cell_hx);
        auto by = (pt.y() - cell_bar.y()) / (0.5*cell_hy);
        auto ihx = 2.0/cell_hx;
        auto ihy = 2.0/cell_hy;

#ifdef POWER_CACHE
        if ( power_cache.size() != (basis_degree+1)*2 )
          power_cache.resize( (basis_degree+1)*2);

        power_cache[0] = 1.0;
        power_cache[1] = 1.0;
        for (size_t i = 1; i <= basis_degree; i++) {
            power_cache[2*i]    = iexp_pow(bx, i);
            power_cache[2*i+1]  = iexp_pow(by, i);
        }
#endif

        size_t pos = 0;
        for (size_t k = 0; k <= basis_degree; k++) {
            for (size_t i = 0; i <= k; i++) {
                auto pow_x = k-i;
                auto pow_y = i;

#ifdef POWER_CACHE
                auto px = power_cache[2*pow_x];
                auto py = power_cache[2*pow_y+1];
                auto dx = (pow_x == 0) ? 0 : pow_x*ihx*power_cache[2*(pow_x-1)];
                auto dy = (pow_y == 0) ? 0 : pow_y*ihy*power_cache[2*(pow_y-1)+1];
#else
                auto px = iexp_pow(bx, pow_x);
                auto py = iexp_pow(by, pow_y);
                auto dx = (pow_x == 0) ? 0 : pow_x*ihx*iexp_pow(bx, pow_x-1);
                auto dy = (pow_y == 0) ? 0 : pow_y*ihy*iexp_pow(by, pow_y-1);
#endif
                ret(pos,0) = dx*py;
                ret(pos,1) = px*dy;
                pos++;
            }
        }
        assert(pos == basis_size);

        return ret;
    }

    size_t size() const {
        return basis_size;
    }

    size_t degree() const {
        return basis_degree;
    }
    
    static size_t size(size_t degree) {
        return (degree+2)*(degree+1)/2;
    }

};

template<typename Mesh, typename VT>
class cut_vector_cell_basis {

    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;
    typedef Matrix<VT, 2, 2>                gradient_type;
    typedef Matrix<VT, Dynamic, 2>          function_type;

    point_type          cell_bar;
    size_t              basis_degree, basis_size;

    cut_cell_basis<Mesh,VT> scalar_basis;
    
#ifdef POWER_CACHE
    std::vector<coordinate_type>  power_cache;
#endif

public:

    cut_vector_cell_basis(const Mesh& msh, const typename Mesh::cell_type& cl, size_t degree, element_location where) : scalar_basis(msh, cl, degree, where) {

        auto loc = location(msh, cl);
        if( loc != where && loc != element_location::ON_INTERFACE) {
            basis_degree = degree;
            basis_size   = 2*(basis_degree+2)*(basis_degree+1)/2;
        }
        else {
            basis_degree = degree;
            basis_size   = 2*(basis_degree+2)*(basis_degree+1)/2;
        }

    }

    Matrix<VT, Dynamic, 2>
    eval_basis(const point_type& pt) {

        Matrix<VT, Dynamic, 2> ret = Matrix<VT, Dynamic, 2>::Zero(basis_size, 2);
        
        const auto phi = scalar_basis.eval_basis(pt);

        for(size_t i = 0; i < scalar_basis.size(); i++) {
            ret(2 * i, 0)     = phi(i);
            ret(2 * i + 1, 1) = phi(i);
        }
        
        assert(2 * scalar_basis.size() == basis_size);

        return ret;
    }

    std::vector<gradient_type>
    eval_gradients(const point_type& pt) {

        std::vector<gradient_type> ret;
        ret.reserve(basis_size);
        const function_type dphi = scalar_basis.eval_gradients(pt);

        for (size_t i = 0; i < scalar_basis.size(); i++) {

            const Matrix<VT, 1, 2> dphi_i = dphi.row(i);
            gradient_type g;
            g = gradient_type::Zero();
            g.row(0) = dphi_i;
            ret.push_back(g);

            g = gradient_type::Zero();
            g.row(1) = dphi_i;
            ret.push_back(g);
        
        }
        
        assert(ret.size() == basis_size);

        return ret;
    }

    size_t size() const {
        return basis_size;
    }

    size_t degree() const {
        return basis_degree;
    }

    static size_t size(size_t degree) {
        return 2*(degree+2)*(degree+1)/2;
    }

};

template<typename T, size_t ET>
auto diameter(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, element_location where) {

    if (!is_cut(msh, cl))
        return diameter(msh, cl);

    auto tp = collect_triangulation_points(msh, cl, where);

    auto diam = 0.0;
    for (size_t i = 0; i < tp.size(); i++) {
        for (size_t j = i+1; j < tp.size(); j++) {
            diam = std::max(diam, (tp[i]-tp[j]).to_vector().norm() );
        }
    }

    return diam;
}

template<typename T, size_t ET>
auto compute_hx_hy(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, element_location where) {
    
    if (!is_cut(msh, cl))
        return std::make_pair(diameter(msh, cl),diameter(msh, cl));

    auto tp = collect_triangulation_points(msh, cl, where);

    T min_x = tp[0].x(), max_x = tp[0].x();
    T min_y = tp[0].y(), max_y = tp[0].y();

    for (const auto& p : tp) {
        min_x = std::min(min_x, p.x());
        max_x = std::max(max_x, p.x());
        min_y = std::min(min_y, p.y());
        max_y = std::max(max_y, p.y());
    }

    return std::make_pair(max_x - min_x, max_y - min_y);
}

//////////////////////////  FACE CUT BASIS  ///////////////////////////
template<typename Mesh, typename VT>
class cut_face_basis {

    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          face_bar;
    point_type          base;
    coordinate_type     face_h;
    size_t              basis_degree, basis_size;

public:
    cut_face_basis(const Mesh& msh, const typename Mesh::face_type& fc, size_t degree, element_location where) {

        auto loc = location(msh,fc);
        if( loc != where && loc != element_location::ON_INTERFACE) {
            face_bar        = barycenter(msh, fc);
            face_h          = diameter(msh, fc);
            basis_degree    = degree;
            basis_size      = degree+1;
            auto pts = points(msh, fc);
            base = face_bar - pts[0];
        }
        else {
            face_bar        = barycenter(msh, fc, where);
            face_h          = diameter(msh, fc, where);
            basis_degree    = degree;
            basis_size      = degree+1;
            auto pts = points(msh, fc, where);
            base = face_bar - pts[0];
        }
    }
    Matrix<VT, Dynamic, 1>
    eval_basis(const point_type& pt) {
        Matrix<VT, Dynamic, 1> ret = Matrix<VT, Dynamic, 1>::Zero(basis_size);
        auto v = base.to_vector();
        auto t = (pt - face_bar).to_vector();
        auto dot = v.dot(t);
        auto ep = 4.0*dot/(face_h*face_h);
        coordinate_type coeff = sqrt(face_h / 2.0);
        ret(0) = sqrt(1.0 / 2.0) / coeff;
        if( basis_degree == 0)
            return ret;
        ret(1) = ep * sqrt(3.0 / 2.0) / coeff;
        if( basis_degree == 1)
            return ret;
        ret(2) = (3*ep*ep - 1) * sqrt(5.0/8.0) / coeff;
        if( basis_degree == 2)
            return ret;
        ret(3) = (5*ep*ep*ep - 3*ep) * sqrt(7.0 / 8.0) / coeff;
        if( basis_degree == 3)
            return ret;
        throw std::logic_error("bases : we shouldn't be here");
    }

    size_t size() const {
        return basis_size;
    }

    size_t degree() const {
        return basis_degree;
    }

    static size_t size(size_t degree) {
        return degree+1;
    }
};

template<typename Mesh, typename VT>
class cut_vector_face_basis {

    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;
    point_type          face_bar;
    point_type          base;
    coordinate_type     face_h;
    size_t              basis_degree, basis_size;
    cut_face_basis<Mesh,VT> scalar_basis;

public:
    cut_vector_face_basis(const Mesh& msh, const typename Mesh::face_type& fc, size_t degree,
        element_location where) :
        scalar_basis(msh, fc, degree, where)
    {
        auto loc = location(msh,fc);
        if( loc != where && loc != element_location::ON_INTERFACE)
        {
            face_bar        = barycenter(msh, fc);
            face_h          = diameter(msh, fc);
            basis_degree    = degree;
            basis_size      = 2*(degree+1);

            auto pts = points(msh, fc);
            base = face_bar - pts[0];
        }
        else
        {
            face_bar        = barycenter(msh, fc, where);
            face_h          = diameter(msh, fc, where);
            basis_degree    = degree;
            basis_size      = 2*(degree+1);

            auto pts = points(msh, fc, where);
            base = face_bar - pts[0];
        }
    }

    Matrix<VT, Dynamic, 2>
    eval_basis(const point_type& pt)
    {
        Matrix<VT, Dynamic, 2> ret = Matrix<VT, Dynamic, 2>::Zero(basis_size, 2);

        const auto psi = scalar_basis.eval_basis(pt);

        for(size_t i = 0; i < scalar_basis.size(); i++)
        {
            ret(2 * i, 0)     = psi(i);
            ret(2 * i + 1, 1) = psi(i);
        }

        assert(2 * scalar_basis.size() == basis_size);

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
        return 2*(degree+1);
    }
};

////////////////////////////////////////////////

template<typename T, size_t ET>
auto
barycenter(const cuthho_mesh<T, ET>& msh,
           const typename cuthho_mesh<T, ET>::face_type& fc,
           element_location where)
{
    if ( !is_cut(msh, fc) )
        return barycenter(msh, fc);

    auto tp = points(msh, fc, where);

    point<T, 2> bar;
    bar[0] = 0.5 * (tp[0][0] + tp[1][0]);
    bar[1] = 0.5 * (tp[0][1] + tp[1][1]);

    return bar;
}


template<typename T, size_t ET>
auto
diameter(const cuthho_mesh<T, ET>& msh,
           const typename cuthho_mesh<T, ET>::face_type& fc,
           element_location where)
{
    if ( !is_cut(msh, fc) )
        return diameter(msh, fc);

    auto tp = points(msh, fc, where);
    auto vect = (tp[0] - tp[1]).to_vector();
    T ret = vect.norm();
    return ret;
}

