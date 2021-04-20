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
#include <boost/range/adaptor/reversed.hpp>


size_t
degree_det_jacobian(size_t degree)
{
    return  2*degree ;
}

size_t
jacobian_interface(size_t deg)
{
    return  deg - 1  ;
}



template<typename T, size_t ET>
bool
is_cut(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl)
{
    assert(cl.user_data.location != element_location::UNDEF);
    return cl.user_data.location == element_location::ON_INTERFACE;
}

template<typename T, size_t ET>
bool
is_cut(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc)
{
    assert(fc.user_data.location != element_location::UNDEF);
    return fc.user_data.location == element_location::ON_INTERFACE;
}

template<typename T, size_t ET>
element_location
location(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl)
{
    assert(cl.user_data.location != element_location::UNDEF);
    return cl.user_data.location;
}

template<typename T, size_t ET>
element_location
location(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc)
{
    assert(fc.user_data.location != element_location::UNDEF);
    return fc.user_data.location;
}

template<typename T, size_t ET>
element_location
location(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::node_type& n)
{
    assert(n.user_data.location != element_location::UNDEF);
    return n.user_data.location;
}



template<typename T, typename Function>
point<T, 2>
find_zero_crossing(const point<T,2>& p0, const point<T,2>& p1, const Function& level_set_function,
                   const T& threshold)
{
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
        //auto la = level_set_function(pa);
        auto lb = level_set_function(pb);
        auto lm = level_set_function(pm);

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



template<typename T, size_t ET, typename Function>
void
detect_node_position(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    for (auto& n : msh.nodes)
    {
        auto pt = points(msh, n); //deleted by Stefano
        if ( level_set_function(pt) < 0 ) //deleted by Stefano
        //if ( level_set_function(n) < 0 ) // add by Stefano
            n.user_data.location = element_location::IN_NEGATIVE_SIDE;
        else
            n.user_data.location = element_location::IN_POSITIVE_SIDE;
    }
}

template<typename T, size_t ET, typename Function>
void
detect_cut_faces(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    for (auto& fc : msh.faces)
    {
        auto pts = points(msh, fc);
        auto l0 = level_set_function(pts[0]);      //deleted by Stefano
        auto l1 = level_set_function(pts[1]);       //deleted by Stefano
        
        //auto l0 = level_set_function(pts[0],msh,fc);      // add by Stefano
        //auto l1 = level_set_function(pts[1],msh,fc);       // add by Stefano
        
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
        auto pm = find_zero_crossing(pts[0], pts[1], level_set_function, threshold);
        //auto pm = find_zero_crossing(pts[0], pts[1], level_set_function, threshold,msh,fc);

        /* If node 0 is in the negative region, mark it as node inside, otherwise mark node 1 */
        fc.user_data.node_inside = ( l0 < 0 ) ? 0 : 1;
        fc.user_data.location = element_location::ON_INTERFACE;
        fc.user_data.intersection_point = pm;
    }
}

template<typename T, size_t ET, typename Function>
void
detect_cut_cells(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    //typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
     //typedef typename cuthho_mesh<T, ET>::cell_type cell_type;

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
           /*
            auto is_positive = [&](const point_type& pt, const cuthho_mesh<T, ET> & msh, const cell_type& cl) -> bool {
                return level_set_function(pt,msh,cl) > 0;
                */
            };

            auto pts = points(msh, cl);
            if ( std::all_of(pts.begin(), pts.end(), is_positive) )
                cl.user_data.location = element_location::IN_POSITIVE_SIDE;
            else
                cl.user_data.location = element_location::IN_NEGATIVE_SIDE;
        }

        if (k == 2)
        {
            cl.user_data.location = element_location::ON_INTERFACE;
            auto p0 = cut_faces[0].second;
            auto p1 = cut_faces[1].second;
            auto pt = p1 - p0;
            auto pn = p0 + point<T,2>(-pt.y(), pt.x());

            if ( level_set_function(pn) >= 0 )
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

        if ( k != 0 && k != 2 )
            throw std::logic_error("invalid number of cuts in cell");

        cell_i++;
    }
}

//#define USE_OLD_DISPLACEMENT

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
move_nodes(cuthho_mesh<T, ET>& msh, const Function& level_set_function)
{
    //typedef typename cuthho_mesh<T, ET>::face_type  face_type;
    //typedef typename cuthho_mesh<T, ET>::point_type point_type;

    T closeness_thresh = 0.4;

    for (auto& fc : msh.faces)
    {
        if ( location(msh, fc) == element_location::ON_INTERFACE )
        {
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
            {
                std::cout << red << "[ !WARNING! ] A concave polygon was generated (cell " << offset(msh, cl) << ")." << nocolor << std::endl;
                throw std::logic_error("concave poly");
            }
        }
    }
}

#endif

template<typename T, size_t ET>
std::array< typename cuthho_mesh<T, ET>::point_type, 2 >
points(const cuthho_mesh<T, ET>& msh,
       const typename cuthho_mesh<T, ET>::face_type& fc,
       const element_location& where)
{
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
    else{
        if(location(msh, nds[0]) == element_location::ON_INTERFACE)
            std::cout<<"IN INTERFACE POINT nds[0] = "<<nds[0] <<" , fc.user_data.intersection_point = "<<fc.user_data.intersection_point<<std::endl;
        if(location(msh, nds[1]) == element_location::ON_INTERFACE)
            std::cout<<"IN INTERFACE POINT nds[1] = "<<nds[1] <<" , fc.user_data.intersection_point = "<<fc.user_data.intersection_point<<std::endl;
        
        if(location(msh, nds[0]) == element_location::IN_POSITIVE_SIDE)
            std::cout<<"IN POSITIVE POINT nds[0] = "<<nds[0] <<" , fc.user_data.intersection_point = "<<fc.user_data.intersection_point<<std::endl;
        if(location(msh, nds[1]) == element_location::IN_POSITIVE_SIDE)
            std::cout<<"IN POSITIVE POINT nds[1] = "<<nds[1] <<" , fc.user_data.intersection_point = "<<fc.user_data.intersection_point<<std::endl;
        
        if(location(msh, nds[0]) == element_location::IN_NEGATIVE_SIDE)
            std::cout<<"IN NEGATIVE POINT nds[0] = "<<nds[0] <<" , fc.user_data.intersection_point = "<<fc.user_data.intersection_point<<std::endl;
        if(location(msh, nds[1]) == element_location::IN_NEGATIVE_SIDE)
            std::cout<<"IN NEGATIVE POINT nds[1] = "<<nds[1] <<" , fc.user_data.intersection_point = "<<fc.user_data.intersection_point<<std::endl;
        std::cout<<"STOP SINCE ONE OF THE NODES OF THE CELL IS AN INTERFACE POINT!"<<std::endl;
        throw std::logic_error("Invalid point configuration");
    }

    return pts;
}

template<typename T>
auto
barycenter(const std::vector< point<T, 2> >& pts)
{
    /*
    point<T, 2> ret;

    T den = 0.0;

    for (size_t i = 2; i < pts.size(); i++)
    {
        auto pprev  = pts[i-1]-pts[0];
        auto pcur   = pts[i]-pts[0];
        auto d      = det(pprev, pcur) / 2.0;
        ret         = ret + (pprev + pcur) * d;
        den         += d;
    }

    return pts[0] + ret/(den*3);
    */
    return barycenter(pts.begin(), pts.end());
}

template<typename T, size_t ET>
auto
barycenter(const cuthho_mesh<T, ET>& msh,
           const typename cuthho_mesh<T, ET>::cell_type& cl,
           element_location where)
{
    if ( !is_cut(msh, cl) )
        return barycenter(msh, cl);

    auto tp = collect_triangulation_points(msh, cl, where);
    /*
    for(auto& t : tp)
        std::cout<<t<<" , ";
    std::cout<<std::endl;
    */
    auto bar = barycenter(tp);

    return bar;
}

template<typename T, size_t ET, typename Function>
void
refine_interface(cuthho_mesh<T, ET>& msh, typename cuthho_mesh<T, ET>::cell_type& cl,
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

    auto lm = level_set_function(pm);
    auto ls1 = level_set_function(ps1);
    auto ls2 = level_set_function(ps2);

    point_type ip;

    if ( !((lm >= 0 && ls1 >= 0) || (lm < 0 && ls1 < 0)) )
    {
        auto threshold = diameter(msh, cl) / 1e20;
        ip = find_zero_crossing(pm, ps1, level_set_function, threshold);
    }
    else if ( !((lm >= 0 && ls2 >= 0) || (lm < 0 && ls2 < 0)) )
    {
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
refine_interface(cuthho_mesh<T, ET>& msh, const Function& level_set_function, size_t levels)
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

        refine_interface(msh, cl, level_set_function, 0, interface_points);
    }
}

template<typename T, size_t ET>
std::vector< typename cuthho_mesh<T, ET>::point_type >
collect_triangulation_points(const cuthho_mesh<T, ET>& msh,
                             const typename cuthho_mesh<T, ET>::cell_type& cl,
                             const element_location& where)
{
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

    if ( case1 || case2 || case3 )
    {
        for (size_t i = 0; i < ns.size(); i++)
            if ( location(msh, ns[i]) == where )
                ret.push_back( node2pt(ns[i]) );

        insert_interface();
    }
    else // case4, the only possible remaining
    {
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
struct temp_tri
{
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
operator<<(std::ostream& os, const temp_tri<T>& t)
{
    os << "line([" << t.pts[0].x() << "," << t.pts[1].x() << "],[" << t.pts[0].y() << "," << t.pts[1].y() << "]);" << std::endl;
    os << "line([" << t.pts[1].x() << "," << t.pts[2].x() << "],[" << t.pts[1].y() << "," << t.pts[2].y() << "]);" << std::endl;
    os << "line([" << t.pts[2].x() << "," << t.pts[0].x() << "],[" << t.pts[2].y() << "," << t.pts[0].y() << "]);" << std::endl;
    return os;
}


template<typename T, size_t ET>
typename cuthho_mesh<T, ET>::point_type
tesselation_center(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                    element_location where)
{
    auto fcs = faces(msh, cl);
    auto pts = points(msh, cl);
    auto nds = nodes(msh, cl);

    if (fcs.size() != 4)
        throw std::invalid_argument("This works only on quads for now");

    if( !is_cut(msh, cl) )
        throw std::invalid_argument("No tesselation centers for uncut cells");

    // if two consecutive faces are cut
    // return either the common node or the opposite node
    for (size_t i = 0; i < fcs.size(); i++)
    {
        auto f1 = i;
        auto f2 = (i+1) % fcs.size();
        auto n = (i+1) % fcs.size();

        if ( is_cut(msh, fcs[f1]) && is_cut(msh, fcs[f2]) )
        {
            if ( location(msh, nds[n]) == where )
                return pts[n];
            else
                return pts[(n+2)%4];
        }
    }

    // if two opposite faces are cut
    // return the center of one of the other faces
    for (size_t i = 0; i < 2; i++)
    {
        auto f1 = i;
        auto f2 = i+2;
        auto n = i+1;

        if ( is_cut(msh, fcs[f1]) && is_cut(msh, fcs[f2]) )
        {
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
triangulate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
            element_location where)
{
    assert( is_cut(msh, cl) );

    auto tp = collect_triangulation_points(msh, cl, where);
    // auto bar = barycenter(tp);
    auto bar = tesselation_center(msh, cl, where);

    std::vector<temp_tri<T>> tris;

    for (size_t i = 0; i < tp.size(); i++)
    {
        temp_tri<T> t;
        t.pts[0] = bar;
        t.pts[1] = tp[i];
        t.pts[2] = tp[(i+1)%tp.size()];
        //std::cout<<"bar = "<<bar <<" , tp[i] = "<<tp[i]<<" , tp[(i+1)%tp.size()] = "<<tp[(i+1)%tp.size()]<<std::endl;
        tris.push_back(t);
    }

    return tris;
}

template<typename T, size_t ET>
T
measure(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
        const element_location& where)
{
    if ( !is_cut(msh, cl) ) /* Element is not cut, use std. integration */
        return measure(msh, cl);

    T totmeas = 0.0;
    auto qpsi = integrate(msh, cl, 0, where);
    for (auto& qp : qpsi)
    {
        totmeas += qp.second;
        //std::cout<<"qpsi.first = "<<qp.first<<" , qpsi.second = "<<qp.second<<std::endl;
    }

    return totmeas;
}

template<typename T, size_t ET>
T
measure_old(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
        const element_location& where)
{
    if ( !is_cut(msh, cl) ) /* Element is not cut, use std. integration */
        return measure(msh, cl);

    T totmeas = 0.0;
    auto qpsi = integrate_old(msh, cl, 0, where);
    for (auto& qp : qpsi)
    {
        totmeas += qp.second;
        //std::cout<<"qpsi.first = "<<qp.first<<" , qpsi.second = "<<qp.second<<std::endl;
    }

    return totmeas;
}



    
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}





struct Interface_parametrisation_mesh1d
{
    
    size_t basis_degree , basis_size;
    size_t degree_det ;
   
    
    Interface_parametrisation_mesh1d( size_t degree ) : basis_degree(degree), basis_size(degree+1) ,degree_det( jacobian_interface(degree) ){
//        size_t n_dof =  DA DEFINIRE
//        Matrix<T, Dynamic, 1> RHS =  Matrix<T, Dynamic, 1>::Zero( n_dof );
//        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
//        solver_global_mass.compute(method.Global_Mass);
//
//        for(auto cl : msh.cells )
//        {
//            auto local_RHS = make_bernstein_local_RHS_VEC( msh , cl , degree_FEM , *this );
//            size_t cell_offset = offset(msh,cl);
//            for (size_t i = 0; i < local_dim; i++)
//            {
//                size_t asm_map_i = connectivity_matrix[cell_offset][i].first ;
//                RHS1(asm_map_i) += local_RHS.first(i) ;
//            }
//
//        }
//        sol_FEM.first = solver_global_mass.solve(RHS1);
//
//        for(size_t counter_bis = 0 ; counter_bis < n_cls ;counter_bis++)
//        {
//            for (size_t i = 0; i < local_dim; i++){
//                size_t asm_map =  connectivity_matrix[counter_bis][i].first ;
//                sol_HHO.first(i,counter_bis) = sol_FEM.first( asm_map );
//            }
//
//        }
//
//
    }
    
    

    
    Interface_parametrisation_mesh1d(){}

    
    template<typename T>
    Matrix<T, 2, 1>
    operator()( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        size_t basis_size = basis_degree + 1 ;
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_basis_1d(pt) ;
        
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            ret(0) += basis(i)*physical_pts[i].x();
            ret(1) += basis(i)*physical_pts[i].y();
        }
        return ret;
        
        
    }
    
    template<typename T>
    Matrix<T, 2, 1>
    operator()( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_basis_1d(pt) ;
        
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            ret(0) += basis(i)*physical_pts[i].x();
            ret(1) += basis(i)*physical_pts[i].y();
        }
        return ret;
        
        
    }
    
    
    template<typename T>
    Matrix<T, 2, 1>
    derivative( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        size_t basis_size = basis_degree + 1 ;
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_gradients_1d(pt) ;
       
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            ret(0) += basis(i)*physical_pts[i].x();
            ret(1) += basis(i)*physical_pts[i].y();
        }
        return ret;
        
        
    }
    
    template<typename T>
    Matrix<T, 2, 1>
    derivative( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_gradients_1d(pt) ;
       
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            ret(0) += basis(i)*physical_pts[i].x();
            ret(1) += basis(i)*physical_pts[i].y();
        }
        return ret;
        
        
    }
    
  
    
    
    template<typename T>
    T
    jacobian( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {
        return (this->derivative(pt,physical_pts,basis_degree)).norm();
    }
    
    template<typename T>
    T
    jacobian( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {
        return (this->derivative(pt,physical_pts)).norm();
    }
    
    
    template<typename T>
    Matrix<T, 2, 1>
    tangent( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {

        auto der = this->derivative(pt,physical_pts,basis_degree) ;
        return der/der.norm();
       
        
    }
    
    template<typename T>
    Matrix<T, 2, 1>
    tangent( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {

        auto der = this->derivative(pt,physical_pts) ;
        return der/der.norm();
       
        
    }
    
    template<typename T>
    Matrix<T, 2, 1>
    normal( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        auto tan_pt = (this->tangent(pt, physical_pts , basis_degree)) ;
        ret(0) = tan_pt(1);
        ret(1) = -tan_pt(0);
        return ret;
           
    }
    
   
   
    
    template<typename T>
    Matrix<T, 2, 1>
    normal( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        auto tan_pt = (this->tangent(pt, physical_pts)) ;
        ret(0) = tan_pt(1);
        ret(1) = -tan_pt(0);
        return ret;
           
    }
    
    template<typename T>
    T
    curvature( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        Matrix<T, 2, 1> curv_double_der = Matrix<T, 2, 1>::Zero(2, 1);
        
        size_t basis_size = basis_degree + 1 ;
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_double_derivative_1d(pt) ;
    
        auto curv_der = (this->derivative( pt, physical_pts , basis_degree )) ;
        auto curv_der_norm = curv_der.norm() ;
       
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            curv_double_der(0) += basis(i)*physical_pts[i].x();
            curv_double_der(1) += basis(i)*physical_pts[i].y();
        }
    
        T coeff = curv_der(0)*curv_double_der(0) + curv_der(1)*curv_double_der(1) ;
        ret(0) = curv_double_der(0)/curv_der_norm - curv_der(0)/pow(curv_der_norm,3)*coeff;
        ret(1) = curv_double_der(1)/curv_der_norm - curv_der(1)/pow(curv_der_norm,3)*coeff;
        int sign_curv ;
        auto n = this->normal( pt, physical_pts , basis_degree );
        
        if( (sgn( n(0) ) == sgn( ret(0) ) ) && (sgn( n(1) ) == sgn( ret(1) ) ) )
            sign_curv = 1.0 ;
        else
            sign_curv =  -1.0 ;
       
        return sign_curv * ret.norm()/curv_der_norm ;
           
    }
    
    template<typename T>
    T
    curvature( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        Matrix<T, 2, 1> curv_double_der = Matrix<T, 2, 1>::Zero(2, 1);
        
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_double_derivative_1d(pt) ;
    
        auto curv_der = (this->derivative( pt, physical_pts , basis_degree )) ;
        auto curv_der_norm = curv_der.norm() ;
       
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            curv_double_der(0) += basis(i)*physical_pts[i].x();
            curv_double_der(1) += basis(i)*physical_pts[i].y();
        }
    
        T coeff = curv_der(0)*curv_double_der(0) + curv_der(1)*curv_double_der(1) ;
        ret(0) = curv_double_der(0)/curv_der_norm - curv_der(0)/pow(curv_der_norm,3)*coeff;
        ret(1) = curv_double_der(1)/curv_der_norm - curv_der(1)/pow(curv_der_norm,3)*coeff;
        return ret.norm()/curv_der_norm ;
           
    }
    
    
};


// ---------- FUNCTIONS TO PRESCRIBE CONTINUITY IN NORMAL / CURAVATURE



template< typename T = double >
Matrix<T, Dynamic, Dynamic>
make_local_mass_matrix_curve( size_t degree, size_t basis_degree, size_t degree_det, const std::vector<point<T,2>>& pts, size_t di = 0)
{
    
    cell_basis_Lagrange_1d_reference_new <T> cb(degree);
    Interface_parametrisation_mesh1d curve(basis_degree) ;
    auto cbs = cb.size();
    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    size_t int_deg = degree_det + 2*degree +di;
//    if (degree == 0)
//        int_deg = 2;
    auto qps = edge_quadrature<T>( int_deg );
    
    for(auto& qp:qps)
    {
        auto t = 0.5 * qp.first.x() + 0.5;
        auto w = 0.5 * qp.second * curve.jacobian( t , pts , basis_degree ) ;
        auto phi = cb.eval_basis_1d(t);
        ret += w * phi * phi.transpose();
        

    }
    
    
    return ret;
}


template< typename T , typename Level_Set , typename Mesh >
std::pair< Matrix<T, Dynamic, 1> , Matrix<T, Dynamic, 1> >
make_local_RHS_normal( size_t dev_degree , size_t degree  , size_t degree_det, const std::vector<point<T,2>>& pts  , Level_Set& ls_cell, const Mesh& msh , size_t di = 0)
{

    cell_basis_Lagrange_1d_reference_new <T> cb(dev_degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret0 = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
    Matrix<T, Dynamic, 1> ret1 = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;

    Interface_parametrisation_mesh1d curve(degree);
    size_t deg_ls = ls_cell.level_set.degree_FEM ;

    size_t int_deg = degree_det + deg_ls + dev_degree +di;
    

    auto qps = edge_quadrature<T>( int_deg );

    for (auto& qp : qps)
    {
        auto t = 0.5 * qp.first.x() + 0.5;
        auto w = 0.5 * qp.second * curve.jacobian( t , pts , degree ) ;
        
        auto b = cb.eval_basis_1d(t);
    
        auto p = curve(t , pts ) ;
        point<T,2> pt = typename Mesh::point_type( p(0) , p(1) ) ;
        
        Matrix<T, 2, 1> normal_val = ls_cell.normal(pt) ;

        ret0 += w * normal_val(0) * b.transpose() ;
        ret1 += w * normal_val(1) * b.transpose() ;
    }

    return std::make_pair(ret0,ret1) ;
}


template< typename T , typename Level_Set , typename Mesh >
std::pair< Matrix<T, Dynamic, 1> , Matrix<T, Dynamic, 1> >
make_local_RHS_normal_disc( size_t dev_degree , size_t degree , size_t degree_det , const std::vector<point<T,2>>& pts  , Level_Set& ls_cell, const Mesh& msh , size_t di = 0)
{

    cell_basis_Lagrange_1d_reference_new <T> cb(dev_degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret0 = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
    Matrix<T, Dynamic, 1> ret1 = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;

    Interface_parametrisation_mesh1d curve(degree);
    size_t deg_ls = ls_cell.level_set.degree_FEM ;

    size_t int_deg = degree_det + deg_ls + dev_degree +di;
    

    auto qps = edge_quadrature<T>( int_deg );

    for (auto& qp : qps)
    {
        auto t = 0.5 * qp.first.x() + 0.5;
        auto w = 0.5 * qp.second * curve.jacobian( t , pts , degree ) ;
        
        auto b = cb.eval_basis_1d(t);
    
        auto p = curve(t , pts ) ;
        point<T,2> pt = typename Mesh::point_type( p(0) , p(1) ) ;
        
        Matrix<T, 2, 1> normal_val = ls_cell.normal_disc(pt) ;

        ret0 += w * normal_val(0) * b.transpose() ;
        ret1 += w * normal_val(1) * b.transpose() ;
    }

    return std::make_pair(ret0,ret1) ;
}


template< typename T >
std::pair< Matrix<T, Dynamic, 1> , Matrix<T, Dynamic, 1> >
make_local_RHS_derivative( size_t dev_degree , size_t degree ,size_t degree_det , const std::vector<point<T,2>>& pts  , size_t di = 0)
{

    cell_basis_Lagrange_1d_reference_new <T> cb(dev_degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret0 = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
    Matrix<T, Dynamic, 1> ret1 = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;

    Interface_parametrisation_mesh1d curve(degree);
    
    size_t int_deg = degree + dev_degree + degree_det +di ;
    

    auto qps = edge_quadrature<T>( int_deg );

    for (auto& qp : qps)
    {
        auto t = 0.5 * qp.first.x() + 0.5;
        auto w = 0.5 * qp.second * curve.jacobian( t , pts , degree ) ;
        
        auto b = cb.eval_basis_1d(t);
    
        Matrix<T, 2, 1> der_val = curve.derivative(t,pts,degree) ;

        ret0 += w * der_val(0) * b.transpose() ;
        ret1 += w * der_val(1) * b.transpose() ;
    }

    return std::make_pair(ret0,ret1) ;
}

template< typename T >
Matrix<T, Dynamic, 1>
make_local_RHS_curvature(size_t dev_degree ,  size_t degree , size_t degree_det , const std::vector<point<T,2>>& pts  , size_t di = 0)
{

    cell_basis_Lagrange_1d_reference_new <T> cb(dev_degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;

    Interface_parametrisation_mesh1d curve(degree);

    size_t int_deg = degree + dev_degree + degree_det + di ;
    

    auto qps = edge_quadrature<T>(int_deg) ; //(   2*degree + dev_degree );

    for (auto& qp : qps)
    {
        auto t = 0.5 * qp.first.x() + 0.5;
        auto w = 0.5 * qp.second * curve.jacobian( t , pts , degree ) ;
        
        auto b = cb.eval_basis_1d(t);
//        std::cout<<"t = "<<t<<" , b = "<<b<<std::endl;
        auto curvature_val = curve.curvature(t,pts,degree) ;
//        std::cout<<"w = "<<w<<std::endl;
//        std::cout<<"qp.second = "<<qp.second<<std::endl;
        ret += w * curvature_val * b.transpose() ;
//        std::cout<<"ret = "<<ret<<std::endl;
    }

    return ret;
}



template< typename T , typename Level_Set , typename Mesh >
Matrix<T, Dynamic, 1>
make_local_RHS_curvature(size_t dev_degree ,  size_t degree ,size_t degree_det , const std::vector<point<T,2>>& pts , Level_Set& ls_cell , const Mesh& msh , size_t di = 0)
{

    cell_basis_Lagrange_1d_reference_new <T> cb(dev_degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
    size_t deg_ls = ls_cell.level_set.degree_FEM ;
    Interface_parametrisation_mesh1d curve(degree);

    size_t int_deg = degree_det + deg_ls + dev_degree +di;
   

    auto qps = edge_quadrature<T>(int_deg) ; //(   2*degree + dev_degree );

    for (auto& qp : qps)
    {
        auto t = 0.5 * qp.first.x() + 0.5;
        auto w = 0.5 * qp.second * curve.jacobian( t , pts , degree ) ;
        
        auto b = cb.eval_basis_1d(t);
//        std::cout<<"t = "<<t<<" , b = "<<b<<std::endl;
        auto p = curve(t , pts ) ;
        point<T,2> pt = typename Mesh::point_type( p(0) , p(1) ) ;
        
        auto curvature_val = ls_cell.divergence(pt) ;
//        std::cout<<"w = "<<w<<std::endl;
//        std::cout<<"qp.second = "<<qp.second<<std::endl;
        ret += w * curvature_val * b.transpose() ;
//        std::cout<<"ret = "<<ret<<std::endl;
    }

    return ret;
}

template< typename T , typename Level_Set , typename Mesh >
Matrix<T, Dynamic, 1>
make_local_RHS_curvature_disc(size_t dev_degree ,  size_t degree ,size_t degree_det , const std::vector<point<T,2>>& pts , Level_Set& ls_cell , const Mesh& msh , size_t di = 0)
{

    cell_basis_Lagrange_1d_reference_new <T> cb(dev_degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs, 1) ;
    size_t deg_ls = ls_cell.level_set.degree_FEM ;
    Interface_parametrisation_mesh1d curve(degree);

    size_t int_deg = degree_det + deg_ls + dev_degree + di ;
   

    auto qps = edge_quadrature<T>(int_deg) ; //(   2*degree + dev_degree );

    for (auto& qp : qps)
    {
        auto t = 0.5 * qp.first.x() + 0.5;
        auto w = 0.5 * qp.second * curve.jacobian( t , pts , degree ) ;
        
        auto b = cb.eval_basis_1d(t);
//        std::cout<<"t = "<<t<<" , b = "<<b<<std::endl;
        auto p = curve(t , pts ) ;
        point<T,2> pt = typename Mesh::point_type( p(0) , p(1) ) ;
        
        auto curvature_val = ls_cell.divergence_disc(pt) ;
//        std::cout<<"w = "<<w<<std::endl;
//        std::cout<<"qp.second = "<<qp.second<<std::endl;
        ret += w * curvature_val * b.transpose() ;
//        std::cout<<"ret = "<<ret<<std::endl;
    }

    return ret;
}






template<typename Mesh, typename T = typename Mesh::coordinate_type >
struct Interface_parametrisation_mesh1d_global
{
    
    
    size_t degree_det ;
    
    // ---- FINITE SPACE FOR THE INTERFACE
    std::vector< std::vector<size_t> > connectivity_cells ;
    std::vector< std::vector<size_t> > connectivity_matrix ;
    size_t ndof ;
    size_t counter_subcls = 0;
    size_t basis_degree , basis_size;
    
    // ---- FINITE SPACE FOR THE INTERFACE DERIVATIVE
    std::vector< std::vector<size_t> > connectivity_matrix_der ;
    //    std::vector< std::vector<size_t> > connectivity_cells_der ;
    size_t ndof_der ;
//    size_t counter_subcls_der = 0;
    size_t der_degree ;
    size_t der_size ;
    
    // ---- FINITE SPACE FOR THE INTERFACE CURVATURE
    std::vector< std::vector<size_t> > connectivity_matrix_dd ;
//    std::vector< std::vector<size_t> > connectivity_cells_dd ;
    size_t ndof_dd ;
//    size_t counter_subcls_dd = 0;
    size_t dd_degree ;
    size_t dd_size ;
    
    
    Eigen::Matrix<T, Dynamic, Dynamic> interface0 ;
    Eigen::Matrix<T, Dynamic, Dynamic> interface1 ;
    
//    Eigen::Matrix<T, Dynamic, 1> derivative_FE0 ;
//    Eigen::Matrix<T, Dynamic, 1> derivative_FE1 ;

    Eigen::Matrix<T, Dynamic, Dynamic> derivative0 ;
    Eigen::Matrix<T, Dynamic, Dynamic> derivative1 ;
    
    Eigen::Matrix<T, Dynamic, Dynamic> normal0 ;
    Eigen::Matrix<T, Dynamic, Dynamic> normal1 ;
    
//    Eigen::Matrix<T, Dynamic, 1> curvature_FE ;
    Eigen::Matrix<T, Dynamic, Dynamic> curvature_field ;
    
   
    
    Interface_parametrisation_mesh1d_global();

    
    Interface_parametrisation_mesh1d_global(const Interface_parametrisation_mesh1d_global& other)
    {
        degree_det = other.degree_det;
        connectivity_cells = other.connectivity_cells ;
        connectivity_matrix = other.connectivity_matrix;
        ndof = other.ndof;
        counter_subcls = other.counter_subcls;
        basis_degree = other.basis_degree;
        basis_size = other.basis_size;
        connectivity_matrix_der = other.connectivity_matrix_der;
        ndof_der = other.ndof_der;
        
        der_degree = other.der_degree;
        der_size = other.der_size;
            
        connectivity_matrix_dd = other.connectivity_matrix_dd;
        
        ndof_dd = other.ndof_dd;
      
        dd_degree = other.dd_degree;
        dd_size = other.dd_size;
            
            
        interface0 = other.interface0;
        interface1 = other.interface1;
    

        derivative0 = other.derivative0;
        derivative1 = other.derivative1;
           
        normal0 = other.normal0;
        normal1 = other.normal1;
        curvature_field = other.curvature_field;
        
       
    }
    
//    Interface_parametrisation_mesh1d_global(const Mesh& msh , size_t degree ) : basis_degree(degree), basis_size(degree+1) ,degree_det(2*degree),connectivity_cells(msh.cells.size()),der_degree(degree),der_size(degree+1)
//    {
//        // DEFINITION OF GLOBAL connectivity_matrix
//        space_definition_derivative(msh ,der_degree, der_size );
//
//    }
    
    Interface_parametrisation_mesh1d_global(const Mesh& msh , size_t degree ) : basis_degree(degree), basis_size(degree+1) ,degree_det( jacobian_interface(degree) ),connectivity_cells(msh.cells.size()), der_degree(degree-1), der_size(degree), dd_degree(degree-1), dd_size(degree)
    {
      
//        std::cout<<'\n'<<"CI VUOLE UN COPY CONSTUCTORRRR"<<std::endl;
        std::cout<<'\n'<<"Parametric interface: local order = "<<basis_degree<<std::endl;
        space_definition_interface(msh ,basis_degree, basis_size );
        
        std::cout<<'\n'<<"Parametric DERIVATIVE: local order = "<<der_degree<<std::endl;
    
        space_definition_derivative(msh ,der_degree, der_size );
        std::cout<<'\n'<<"Parametric CURVATURE: local order = "<<dd_degree<<std::endl;
        space_definition_curvature(msh ,dd_degree, dd_size );
        std::cout<<'\n'<<std::endl;
    }
    
    Interface_parametrisation_mesh1d_global(const Mesh& msh , size_t degree_curve , size_t degree_curvature ) : basis_degree(degree_curve) ,degree_det( jacobian_interface(degree_curve) ), basis_size(degree_curve+1), connectivity_cells(msh.cells.size()), der_degree(degree_curvature), der_size(degree_curvature+1), dd_degree(degree_curvature), dd_size(degree_curvature+1)
        {
          
    //        std::cout<<'\n'<<"CI VUOLE UN COPY CONSTUCTORRRR"<<std::endl;
            std::cout<<'\n'<<"Parametric interface: local order = "<<basis_degree<<std::endl;
            std::cout<<"Degree jacobian along interface = "<<degree_det<<std::endl;
//            space_definition_interface_OLD(msh ,basis_degree, basis_size );
            
            space_definition_interface(msh ,basis_degree, basis_size );
            
            std::cout<<"Parametric DERIVATIVE: local order = "<<der_degree<<std::endl;
        
//            space_definition_derivative_OLD(msh ,der_degree, der_size );
            
            space_definition_derivative(msh ,der_degree, der_size );
            
            std::cout<<"Parametric CURVATURE: local order = "<<dd_degree<<std::endl;
//            space_definition_curvature_OLD(msh ,dd_degree, dd_size );
            
            space_definition_curvature(msh ,dd_degree, dd_size );
            std::cout<<'\n'<<std::endl;
        }

    
    template< typename Level_Set >
    void
    updating_parametric_interface(const Mesh& msh, Level_Set& ls_cell,bool l2proj,bool avg,bool l2proj_para,bool disc)
    {
        std::cout<<"Parametric interface updating."<<std::endl;
        space_definition_interface(msh ,basis_degree, basis_size );
        space_definition_derivative(msh ,der_degree, der_size );
        space_definition_curvature(msh ,dd_degree, dd_size );
        this->make_L2_proj_para_derivative(msh);
        //---------------------------- L2 global Normal from LS  ----------------------- //
        if(l2proj){
            if(!disc)
                this->make_L2_proj_para_normal(msh,ls_cell);
            else
                this->make_L2_proj_para_normal_disc(msh,ls_cell);
        }
                              
        //---------------------------- Avg Normal from LS  ---------------------------- //
        if(avg){
            if(!disc)
                this->make_avg_L2_local_proj_para_normal(msh, ls_cell);
            else
                this->make_avg_L2_local_proj_para_normal_disc(msh, ls_cell);
        }
                       
        // *********************** CURVATURE PARA *************************//
                       
        //------------- L2 cont curvature from parametric interface  r ---------- //
        if(l2proj_para)
            this->make_L2_proj_para_curvature(msh);

                        
        //---------------------------- L2 global Curvature from LS  ----------------------- //
        if(l2proj){
            if(!disc)
                this->make_L2_proj_para_curvature(msh,ls_cell);
            else
                this->make_L2_proj_para_curvature_disc(msh,ls_cell);
        }
                      
        //---------------------------- Avg Curvature from LS  ---------------------------- //
        if(avg){
            if(!disc)
                this->make_avg_L2_local_proj_para_curvature(msh, ls_cell);
            else
                this->make_avg_L2_local_proj_para_curvature_disc(msh, ls_cell);
                 
        }
        
        
    }
    
    

    
    
    std::vector<size_t>
    get_global_cells_interface(const Mesh& msh , const typename Mesh::cell_type& cl) const
    {
        size_t k_cl = offset(msh,cl);
        return connectivity_cells[k_cl] ;
    }
    
    
    void
    space_definition_interface_OLD(const Mesh& msh , size_t deg , size_t deg_size)
    {
        size_t counter_cut_cls = 0;
        size_t counter = 0;
        for(auto& cl : msh.cells)
        {
            if(location(msh, cl) == element_location::ON_INTERFACE )
            {
                counter_cut_cls++;
                for (size_t i_cell = 0; i_cell < cl.user_data.integration_msh.cells.size(); i_cell++)
                {
    //                   connectivity_cells[counter].push_back(counter_subcls);
                    counter_subcls++;
                }
                    
            }
            counter++;
        }
        std::cout<<"Amount of cut cells = "<<counter_cut_cls<<std::endl;
        std::cout<<"Amount of subcells = "<<counter_subcls<<std::endl;
            
        if (deg == 0)
            ndof = counter_subcls ;
        else
            ndof = counter_subcls * deg ;
            
        connectivity_matrix.resize( counter_subcls , std::vector<size_t>(deg_size) ) ;
             
        size_t i_cl_global = 0 ;
        size_t i_inner = 2 ;
        point<T,2> first_point ;
        point<T,2> cell_end_point ;
        bool first_cut_cell_found = FALSE ;
    
//        size_t last_k_offset ;
//        bool come_back = false ;
        if( deg == 0 )
        {
            for(auto& cl : msh.cells)
            {
                if(location(msh, cl) == element_location::ON_INTERFACE )
                {
                    size_t k_offset = offset(msh,cl); // ADD ORA
                    auto msh_int = cl.user_data.integration_msh ;
                        
                    if(!first_cut_cell_found)
                    {
                    
                        for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                        {
                            connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
                            connectivity_matrix[i_cl_global][0] = i_cl_global;
                            i_cl_global++;
                            
                        }
                        first_cut_cell_found = TRUE;
                        first_point = *cl.user_data.interface.begin() ;
                        cell_end_point = *(cl.user_data.interface.end() -1) ;
                    }
                    else if( first_cut_cell_found && !( first_point == cell_end_point  ) )
                    {
                        for(auto& cl : msh.cells)
                        {
                            if((cl.user_data.location == element_location::ON_INTERFACE)&& (cell_end_point == *cl.user_data.interface.begin() ) && !( first_point == cell_end_point)  )
                            {
                                size_t k_offset = offset(msh,cl); // ADD ORA
                                auto msh_int = cl.user_data.integration_msh ;
                                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                                {
                                    connectivity_cells[k_offset].push_back(i_cl_global);
                                    connectivity_matrix[i_cl_global][0] = i_cl_global;
                                    i_cl_global++;
                                }
                                cell_end_point = *(cl.user_data.interface.end() -1) ;
                                         
                            }
                        }
                        
                    }
                    else
                        break;
                             
                     
                }
            }
                
        }
            
        else
        {
            for(auto& cl : msh.cells)
            {
                if(location(msh, cl) == element_location::ON_INTERFACE )
                {
                    size_t k_offset = offset(msh,cl); // ADD ORA
                    auto msh_int = cl.user_data.integration_msh ;
                    if(!first_cut_cell_found)
                    {
                        for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                        {
                            if(i_local == 0 || i_local == 1)
                                connectivity_matrix[0][i_local] = i_local;
                            else
                                connectivity_matrix[0][i_local] = i_inner++;
                        }
                        connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
//                        last_k_offset = k_offset;
                        i_cl_global++;
                        for (size_t i_cell = 1; i_cell < msh_int.cells.size(); i_cell++)
                        {
                            for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                            {
                                if(i_local == 0)
                                    connectivity_matrix[i_cl_global][0] = connectivity_matrix[i_cl_global - 1][1];
                                else if(i_local == 1)
                                    connectivity_matrix[i_cl_global][1] = i_inner++ ;
                                else
                                    connectivity_matrix[i_cl_global][i_local] = i_inner++;
                            }
                            connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
                            i_cl_global++;
                        }
                        first_cut_cell_found = TRUE;
                        first_point = *cl.user_data.interface.begin() ;
                        cell_end_point = *(cl.user_data.interface.end() -1) ;
                    }
                    else if( first_cut_cell_found && !( first_point == cell_end_point  ) )
                    {
                        
                        for(auto& cl : msh.cells)
                        {
                            if((cl.user_data.location == element_location::ON_INTERFACE)&& (cell_end_point == *cl.user_data.interface.begin() )   )
                            //&& !( first_point == cell_end_point)
                            {
                                size_t k_offset = offset(msh,cl); // ADD ORA
                               
                                auto msh_int = cl.user_data.integration_msh ;
                                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                                {
                                    for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                                    {
                                        if(i_local == 0)
                                            connectivity_matrix[i_cl_global][0] = connectivity_matrix[i_cl_global - 1][1];
                                        else if(i_local == 1)
                                                connectivity_matrix[i_cl_global][1] = i_inner++ ;
                                        else
                                            connectivity_matrix[i_cl_global][i_local] = i_inner++;
                                    }
                                    connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
                                    i_cl_global++;
                                }
                                cell_end_point = *(cl.user_data.interface.end() -1) ;
                                     
                            }
                        }
                        
                        
                         
                    }
                    else
                        break;
                         
                 
                }
                else if( first_cut_cell_found && !( first_point == cell_end_point  ) )
                {
//                    if(!come_back)
//                    {
                        for(auto& cl : msh.cells)
                        {
                            if((cl.user_data.location == element_location::ON_INTERFACE)&& (cell_end_point == *cl.user_data.interface.begin() )   )
                                        //&& !( first_point == cell_end_point)
                            {
                                size_t k_offset = offset(msh,cl); // ADD ORA
//                                last_k_offset = k_offset;
//                                if( last_k_offset > k_offset ){
//                                    std::cout<<"I am coming back! last_k_offset = "<<last_k_offset<<" , k_offset = "<<k_offset<<std::endl;
//                                    come_back = true ;
//
//                                }
                                auto msh_int = cl.user_data.integration_msh ;
                                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                                {
                                    for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                                    {
                                        if(i_local == 0)
                                            connectivity_matrix[i_cl_global][0] = connectivity_matrix[i_cl_global - 1][1];
                                        else if(i_local == 1)
                                            connectivity_matrix[i_cl_global][1] = i_inner++ ;
                                        else
                                            connectivity_matrix[i_cl_global][i_local] = i_inner++;
                                    }
                                    connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
                                    i_cl_global++;
                                }
                                cell_end_point = *(cl.user_data.interface.end() -1) ;
                                                
                            }
                        }
//                    }
//                    else{
//                        for(auto& cl : boost::adaptors::reverse(msh.cells) )
//                        {
//                            std::cout<<"DOING REVERSE LOOP (FASTER?)"<<std::endl;
//                            if((cl.user_data.location == element_location::ON_INTERFACE)&& (cell_end_point == *cl.user_data.interface.begin() )   )
//                                        //&& !( first_point == cell_end_point)
//                            {
//                                size_t k_offset = offset(msh,cl); // ADD ORA
//                                last_k_offset = k_offset;
//
//                                auto msh_int = cl.user_data.integration_msh ;
//                                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
//                                {
//                                    for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
//                                    {
//                                        if(i_local == 0)
//                                            connectivity_matrix[i_cl_global][0] = connectivity_matrix[i_cl_global - 1][1];
//                                        else if(i_local == 1)
//                                            connectivity_matrix[i_cl_global][1] = i_inner++ ;
//                                        else
//                                            connectivity_matrix[i_cl_global][i_local] = i_inner++;
//                                    }
//                                    connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
//                                    i_cl_global++;
//                                }
//                                cell_end_point = *(cl.user_data.interface.end() -1) ;
//
//                            }
//                        }
//                    }
                                    
                }
                else if( first_cut_cell_found && first_point == cell_end_point  )
                    break;
            }
            connectivity_matrix[counter_subcls-1][1] = 0 ;
            for( size_t i_local = 2 ; i_local < deg_size ; i_local++)
                connectivity_matrix[counter_subcls-1][i_local] -= 1 ;
                

                
        }
            
            
        interface0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);
        interface1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);

        Interface_parametrisation_mesh1d curve(deg);
        auto ref_nodes = reference_nodes_ordered_01<T>(deg);
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
               
                auto T_cells =  get_global_cells_interface(msh , cl);
                size_t loc_counter = 0 ;
                for(auto& Tj : T_cells )
                {
                    auto pts = points(msh_int,msh_int.cells[loc_counter]);
                    for (size_t i = 0; i < deg_size; i++){
                        auto interface = curve(ref_nodes[i].x(),pts) ;
                        interface0(i,Tj) = interface(0) ;
                        interface1(i,Tj) = interface(1) ;
                    }
                    
                    loc_counter++;
                }
            }
        }
        
    }
    
    void
    space_definition_interface(const Mesh& msh , size_t deg , size_t deg_size)
    {
        std::vector<size_t> cut_cell_cointainer ;
        size_t counter_cut_cls = 0;
        size_t counter = 0;
        counter_subcls=0 ; 
        for(auto& cl : msh.cells)
        {
            if(location(msh, cl) == element_location::ON_INTERFACE )
            {
                cut_cell_cointainer.push_back(offset(msh,cl));
                counter_cut_cls++;
                for (size_t i_cell = 0; i_cell < cl.user_data.integration_msh.cells.size(); i_cell++)
                {
    //                   connectivity_cells[counter].push_back(counter_subcls);
                    counter_subcls++;
                }
                    
            }
            counter++;
        }
        std::cout<<"Amount of cut cells = "<<counter_cut_cls<<std::endl;
        std::cout<<"Amount of subcells = "<<counter_subcls<<std::endl;
            
        if (deg == 0)
            throw std::logic_error("error, order zero for interface too slow, at least order 1.");
            
        else
            ndof = counter_subcls * deg ;
            
        connectivity_matrix.resize( counter_subcls , std::vector<size_t>(deg_size) ) ;
        connectivity_cells.clear();
        connectivity_cells.resize(msh.cells.size()) ;
        size_t i_cl_global = 0 ;
        size_t i_inner = 2 ;
        point<T,2> first_point ;
        point<T,2> cell_end_point ;
        bool first_cut_cell_found = FALSE ;
        
       
    
        size_t cl_i = 0;
        while(cut_cell_cointainer.size()>0)
        {
            if(cl_i > cut_cell_cointainer.size()-1 )
                std::cout<<"stop: first_pt = "<<first_point<<" pt_to_find = "<<cell_end_point<<std::endl;
            size_t k_offset =  cut_cell_cointainer[cl_i];
            auto cl = msh.cells[k_offset];
            auto msh_int = cl.user_data.integration_msh ;
            
            if(!first_cut_cell_found)
            {
                cut_cell_cointainer.erase(cut_cell_cointainer.begin() ); //pop_front();
                for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                {
                    if(i_local == 0 || i_local == 1)
                        connectivity_matrix[0][i_local] = i_local;
                    else
                        connectivity_matrix[0][i_local] = i_inner++;
                }
                connectivity_cells[k_offset].push_back(i_cl_global);
                i_cl_global++;
                for (size_t i_cell = 1; i_cell < msh_int.cells.size(); i_cell++)
                {
                    for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                    {
                        if(i_local == 0)
                            connectivity_matrix[i_cl_global][0] = connectivity_matrix[i_cl_global - 1][1];
                        else if(i_local == 1)
                            connectivity_matrix[i_cl_global][1] = i_inner++ ;
                        else
                            connectivity_matrix[i_cl_global][i_local] = i_inner++;
                    }
                    connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
                    i_cl_global++;
                }
                    
                first_cut_cell_found = TRUE;
                first_point = *cl.user_data.interface.begin() ;
                cell_end_point = *(cl.user_data.interface.end() -1) ;
                cl_i = 0;
            }
            else if( first_cut_cell_found && cell_end_point == *cl.user_data.interface.begin() && !( first_point == cell_end_point  ) )
            {
                cut_cell_cointainer.erase(cut_cell_cointainer.begin() + cl_i );
//                cut_cell_cointainer.pop_front();
                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                {
                    for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                    {
                        if(i_local == 0)
                            connectivity_matrix[i_cl_global][0] = connectivity_matrix[i_cl_global - 1][1];
                        else if(i_local == 1)
                                connectivity_matrix[i_cl_global][1] = i_inner++ ;
                        else
                            connectivity_matrix[i_cl_global][i_local] = i_inner++;
                    }
                    connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
                    i_cl_global++;
                }
                cell_end_point = *(cl.user_data.interface.end() -1) ;
                cl_i = 0;
  
            }
            else if (first_point == cell_end_point)
                break;
            else
                cl_i++;
                         
        }
        connectivity_matrix[counter_subcls-1][1] = 0 ;
        for( size_t i_local = 2 ; i_local < deg_size ; i_local++)
            connectivity_matrix[counter_subcls-1][i_local] -= 1 ;
                

                
        
            
            
        interface0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);
        interface1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);

        Interface_parametrisation_mesh1d curve(deg);
        auto ref_nodes = reference_nodes_ordered_01<T>(deg);
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
               
                auto T_cells =  get_global_cells_interface(msh , cl);
                size_t loc_counter = 0 ;
                for(auto& Tj : T_cells )
                {
                    auto pts = points(msh_int,msh_int.cells[loc_counter]);
                    for (size_t i = 0; i < deg_size; i++){
                        auto interface = curve(ref_nodes[i].x(),pts) ;
                        interface0(i,Tj) = interface(0) ;
                        interface1(i,Tj) = interface(1) ;
                    }
                    
                    loc_counter++;
                }
            }
        }
        
    }
    
    
    void
    space_definition_derivative_OLD(const Mesh& msh , size_t deg , size_t deg_size)
    {
//        size_t counter_cut_cls = 0;
//        size_t counter = 0;
//        for(auto& cl : msh.cells)
//        {
//            if(location(msh, cl) == element_location::ON_INTERFACE )
//            {
//                counter_cut_cls++;
//                for (size_t i_cell = 0; i_cell < cl.user_data.integration_msh.cells.size(); i_cell++)
//                {
////                    connectivity_cells[counter].push_back(counter_subcls);
//                    counter_subcls++;
//                }
//
//            }
//            counter++;
//        }
//        std::cout<<"Amount of cut cells = "<<counter_cut_cls<<std::endl;
//        std::cout<<"Amount of subcells = "<<counter_subcls<<std::endl;
        
        if (deg == 0)
            ndof_der = counter_subcls ;
        else
            ndof_der = counter_subcls * deg ;
        
        connectivity_matrix_der.resize( counter_subcls , std::vector<size_t>(deg_size) ) ;
         
        size_t i_cl_global = 0 ;
        size_t i_inner = 2 ;
        point<T,2> first_point ;
        point<T,2> cell_end_point ;
        bool first_cut_cell_found = FALSE ;
        
        if( deg == 0 )
        {
            for(auto& cl : msh.cells)
            {
                if(location(msh, cl) == element_location::ON_INTERFACE )
                {
//                    size_t k_offset = offset(msh,cl); // ADD ORA
                    auto msh_int = cl.user_data.integration_msh ;
                    
                    if(!first_cut_cell_found)
                    {
                
                        for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                        {
//                            connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
                            connectivity_matrix_der[i_cl_global][0] = i_cl_global;
                            i_cl_global++;
                        
                        }
                        first_cut_cell_found = TRUE;
                        first_point = *cl.user_data.interface.begin() ;
                        cell_end_point = *(cl.user_data.interface.end() -1) ;
                    }
                    else if( first_cut_cell_found && !( first_point == cell_end_point  ) )
                    {
                        for(auto& cl : msh.cells)
                        {
                            if((cl.user_data.location == element_location::ON_INTERFACE)&& (cell_end_point == *cl.user_data.interface.begin() ) && !( first_point == cell_end_point)  )
                            {
//                                size_t k_offset = offset(msh,cl); // ADD ORA
                                auto msh_int = cl.user_data.integration_msh ;
                                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                                {
//                                    connectivity_cells[k_offset].push_back(i_cl_global);
                                    connectivity_matrix_der[i_cl_global][0] = i_cl_global;
                                    i_cl_global++;
                                }
                                cell_end_point = *(cl.user_data.interface.end() -1) ;
                                     
                            }
                        }
                        
                    }
                    else
                        break;
                         
                 
                }
            }
            
        }
        
        else
        {
            for(auto& cl : msh.cells)
            {
                if(location(msh, cl) == element_location::ON_INTERFACE )
                {
//                    size_t k_offset = offset(msh,cl); // ADD ORA
                    auto msh_int = cl.user_data.integration_msh ;
                    if(!first_cut_cell_found)
                    {
                        for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                        {
                            if(i_local == 0 || i_local == 1)
                                connectivity_matrix_der[0][i_local] = i_local;
                            else
                                connectivity_matrix_der[0][i_local] = i_inner++;
                        }
//                        connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
                        i_cl_global++;
                        for (size_t i_cell = 1; i_cell < msh_int.cells.size(); i_cell++)
                        {
                            for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                            {
                                if(i_local == 0)
                                    connectivity_matrix_der[i_cl_global][0] = connectivity_matrix_der[i_cl_global - 1][1];
                                else if(i_local == 1)
                                    connectivity_matrix_der[i_cl_global][1] = i_inner++ ;
                                else
                                    connectivity_matrix_der[i_cl_global][i_local] = i_inner++;
                            }
//                            connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
                            i_cl_global++;
                        }
                        first_cut_cell_found = TRUE;
                        first_point = *cl.user_data.interface.begin() ;
                        cell_end_point = *(cl.user_data.interface.end() -1) ;
                    }
                    else if( first_cut_cell_found && !( first_point == cell_end_point  ) )
                    {
                        for(auto& cl : msh.cells)
                        {
                            if((cl.user_data.location == element_location::ON_INTERFACE)&& (cell_end_point == *cl.user_data.interface.begin() ) && !( first_point == cell_end_point)  )
                            {
//                                size_t k_offset = offset(msh,cl); // ADD ORA
                                auto msh_int = cl.user_data.integration_msh ;
                                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                                {
                                    for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                                    {
                                        if(i_local == 0)
                                            connectivity_matrix_der[i_cl_global][0] = connectivity_matrix_der[i_cl_global - 1][1];
                                        else if(i_local == 1)
                                             connectivity_matrix_der[i_cl_global][1] = i_inner++ ;
                                        else
                                            connectivity_matrix_der[i_cl_global][i_local] = i_inner++;
                                    }
//                                    connectivity_cells[k_offset].push_back(i_cl_global); // ADD ORA
                                    i_cl_global++;
                                }
                                cell_end_point = *(cl.user_data.interface.end() -1) ;
                                 
                            }
                        }
                     
                    }
                    else
                        break;
                     
             
                }
            }
            connectivity_matrix_der[counter_subcls-1][1] = 0 ;
            for( size_t i_local = 2 ; i_local < deg_size ; i_local++)
                connectivity_matrix_der[counter_subcls-1][i_local] -= 1 ;
            

            
        }
        
        
//        std::cout<<"Beginning of connectivity matrix"<<std::endl;
//        for(size_t i = 0 ; i< counter_subcls ; i++)
//        {
//            for(size_t j = 0 ; j< deg_size ; j++)
//            {
//                std::cout<<" ( "<<connectivity_matrix[i][j] << " ) " ;
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<std::endl;
//        std::cout<<"End of connectivity matrix"<<std::endl;
//        std::cout<<"Beginning of connectivity cell"<<std::endl;
//        for(size_t i = 0 ; i< msh.cells.size() ; i++)
//        {
//            std::cout<<"Cell = "<<offset(msh, msh.cells[i])<<" : ";
//            for(size_t j = 0 ; j< connectivity_cells[i].size() ; j++)
//            {
//                std::cout<<" ( "<<connectivity_cells[i][j] << " ) " ;
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<std::endl;
//        std::cout<<"End of connectivity cell"<<std::endl;

        // DEFINITION OF THE MASS MATRIX -> USEFUL FOR L2 PROJ
//        derivative_FE0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof );
//        derivative_FE1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof );
        
        derivative0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);
        derivative1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);
        
        normal0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);
        normal1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);

    }
    
    void
    space_definition_derivative(const Mesh& msh , size_t deg , size_t deg_size)
    {

        std::vector<size_t> cut_cell_cointainer ;
        for(auto& cl : msh.cells)
        {
            if(location(msh, cl) == element_location::ON_INTERFACE )
            {
                cut_cell_cointainer.push_back(offset(msh,cl));
                    
            }
           
        }
 
        
        if (deg == 0)
            ndof_der = counter_subcls ;
        else
            ndof_der = counter_subcls * deg ;
        
        connectivity_matrix_der.resize( counter_subcls , std::vector<size_t>(deg_size) ) ;
         
        size_t i_cl_global = 0 ;
        size_t i_inner = 2 ;
        point<T,2> first_point ;
        point<T,2> cell_end_point ;
        bool first_cut_cell_found = FALSE ;
        
        if( deg == 0 )
        {
            size_t cl_i = 0;
            while(cut_cell_cointainer.size()>0)
            {
                size_t k_offset =  cut_cell_cointainer[cl_i];
                auto cl = msh.cells[k_offset];
                auto msh_int = cl.user_data.integration_msh ;
                    
                if(!first_cut_cell_found)
                {
                    cut_cell_cointainer.erase(cut_cell_cointainer.begin() );
                    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                    {
                        connectivity_matrix_der[i_cl_global][0] = i_cl_global;
                        i_cl_global++;
                        
                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin() ;
                    cell_end_point = *(cl.user_data.interface.end() -1) ;
                    cl_i = 0;
                }
                else if( first_cut_cell_found && cell_end_point == *cl.user_data.interface.begin() && !( first_point == cell_end_point  ) )
                {
                    cut_cell_cointainer.erase(cut_cell_cointainer.begin() + cl_i );
                           
                    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                    {
                        connectivity_matrix_der[i_cl_global][0] = i_cl_global;
                        i_cl_global++;
                    }
                    cell_end_point = *(cl.user_data.interface.end() -1) ;
                    cl_i = 0;
                }
                else if (first_point == cell_end_point)
                    break;
                else
                    cl_i++;
            }
         
            
        }
        
        else
        {
            size_t cl_i = 0;
            while(cut_cell_cointainer.size()>0)
            {
                size_t k_offset =  cut_cell_cointainer[cl_i];
                auto cl = msh.cells[k_offset];
                auto msh_int = cl.user_data.integration_msh ;
                if(!first_cut_cell_found)
                {
                    cut_cell_cointainer.erase(cut_cell_cointainer.begin() );
                    for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                    {
                        if(i_local == 0 || i_local == 1)
                            connectivity_matrix_der[0][i_local] = i_local;
                        else
                            connectivity_matrix_der[0][i_local] = i_inner++;
                    }
                    i_cl_global++;
                    for (size_t i_cell = 1; i_cell < msh_int.cells.size(); i_cell++)
                    {
                        for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                        {
                            if(i_local == 0)
                                connectivity_matrix_der[i_cl_global][0] = connectivity_matrix_der[i_cl_global - 1][1];
                            else if(i_local == 1)
                                connectivity_matrix_der[i_cl_global][1] = i_inner++ ;
                            else
                                connectivity_matrix_der[i_cl_global][i_local] = i_inner++;
                        }
                        i_cl_global++;
                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin() ;
                    cell_end_point = *(cl.user_data.interface.end() -1) ;
                    cl_i = 0;
                }
                else if( first_cut_cell_found && cell_end_point == *cl.user_data.interface.begin() && !( first_point == cell_end_point  ) )
                {
                    cut_cell_cointainer.erase(cut_cell_cointainer.begin() + cl_i );
                    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                    {
                        for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                        {
                            if(i_local == 0)
                                connectivity_matrix_der[i_cl_global][0] = connectivity_matrix_der[i_cl_global - 1][1];
                            else if(i_local == 1)
                                connectivity_matrix_der[i_cl_global][1] = i_inner++ ;
                            else
                                connectivity_matrix_der[i_cl_global][i_local] = i_inner++;
                        }
                        i_cl_global++;
                    }
                    cell_end_point = *(cl.user_data.interface.end() -1) ;
                    cl_i = 0;
                }
                else if (first_point == cell_end_point)
                    break;
                else
                    cl_i++;
                     
             
            }
            connectivity_matrix_der[counter_subcls-1][1] = 0 ;
            for( size_t i_local = 2 ; i_local < deg_size ; i_local++)
                connectivity_matrix_der[counter_subcls-1][i_local] -= 1 ;
            
        }
        
       
            

       
        
        derivative0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);
        derivative1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);
        
        normal0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);
        normal1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);

    }
    
    void
    space_definition_curvature_OLD(const Mesh& msh , size_t deg , size_t deg_size)
    {
//        size_t counter_cut_cls = 0;
//        size_t counter = 0;
//        for(auto& cl : msh.cells)
//        {
//            if(location(msh, cl) == element_location::ON_INTERFACE )
//            {
//                counter_cut_cls++;
//                for (size_t i_cell = 0; i_cell < cl.user_data.integration_msh.cells.size(); i_cell++)
//                {
//    //                    connectivity_cells[counter].push_back(counter_subcls);
//                    counter_subcls_dd++;
//                }
//
//            }
//            counter++;
//        }
//        std::cout<<"Amount of cut cells = "<<counter_cut_cls<<std::endl;
//        std::cout<<"Amount of subcells = "<<counter_subcls_dd<<std::endl;
//
        
        if (deg == 0)
            ndof_dd = counter_subcls ;
        else
            ndof_dd = counter_subcls * deg ;
        
        connectivity_matrix_dd.resize( counter_subcls , std::vector<size_t>(deg_size) ) ;
             
        size_t i_cl_global = 0 ;
        size_t i_inner = 2 ;
        point<T,2> first_point ;
        point<T,2> cell_end_point ;
        bool first_cut_cell_found = FALSE ;
        
        if( deg == 0 )
        {
            for(auto& cl : msh.cells)
            {
                if(location(msh, cl) == element_location::ON_INTERFACE )
                {
//                    size_t k_offset = offset(msh,cl); // ADD ORA
                    auto msh_int = cl.user_data.integration_msh ;
                    
                    if(!first_cut_cell_found)
                    {
                
                        for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                        {
//                            connectivity_cells_dd[k_offset].push_back(i_cl_global); // ADD ORA
                            connectivity_matrix_dd[i_cl_global][0] = i_cl_global;
                            i_cl_global++;
                        
                        }
                        first_cut_cell_found = TRUE;
                        first_point = *cl.user_data.interface.begin() ;
                        cell_end_point = *(cl.user_data.interface.end() -1) ;
                    }
                    else if( first_cut_cell_found && !( first_point == cell_end_point  ) )
                    {
                        for(auto& cl : msh.cells)
                        {
                            if((cl.user_data.location == element_location::ON_INTERFACE)&& (cell_end_point == *cl.user_data.interface.begin() ) && !( first_point == cell_end_point)  )
                            {
//                                size_t k_offset = offset(msh,cl); // ADD ORA
                                auto msh_int = cl.user_data.integration_msh ;
                                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                                {
//                                    connectivity_cells_dd[k_offset].push_back(i_cl_global);
                                    connectivity_matrix_dd[i_cl_global][0] = i_cl_global;
                                    i_cl_global++;
                                }
                                cell_end_point = *(cl.user_data.interface.end() -1) ;
                                     
                            }
                        }
                        
                    }
                    else
                        break;
                         
                 
                }
            }
            
        }
        
        else
        {
            for(auto& cl : msh.cells)
            {
                if(location(msh, cl) == element_location::ON_INTERFACE )
                {
//                    size_t k_offset = offset(msh,cl); // ADD ORA
                    auto msh_int = cl.user_data.integration_msh ;
                    if(!first_cut_cell_found)
                    {
                        for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                        {
                            if(i_local == 0 || i_local == 1)
                                connectivity_matrix_dd[0][i_local] = i_local;
                            else
                                connectivity_matrix_dd[0][i_local] = i_inner++;
                        }
//                        connectivity_cells_dd[k_offset].push_back(i_cl_global); // ADD ORA
                        i_cl_global++;
                        for (size_t i_cell = 1; i_cell < msh_int.cells.size(); i_cell++)
                        {
                            for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                            {
                                if(i_local == 0)
                                    connectivity_matrix_dd[i_cl_global][0] = connectivity_matrix_dd[i_cl_global - 1][1];
                                else if(i_local == 1)
                                    connectivity_matrix_dd[i_cl_global][1] = i_inner++ ;
                                else
                                    connectivity_matrix_dd[i_cl_global][i_local] = i_inner++;
                            }
//                            connectivity_cells_dd[k_offset].push_back(i_cl_global); // ADD ORA
                            i_cl_global++;
                        }
                        first_cut_cell_found = TRUE;
                        first_point = *cl.user_data.interface.begin() ;
                        cell_end_point = *(cl.user_data.interface.end() -1) ;
                    }
                    else if( first_cut_cell_found && !( first_point == cell_end_point  ) )
                    {
                        for(auto& cl : msh.cells)
                        {
                            if((cl.user_data.location == element_location::ON_INTERFACE)&& (cell_end_point == *cl.user_data.interface.begin() ) && !( first_point == cell_end_point)  )
                            {
//                                size_t k_offset = offset(msh,cl); // ADD ORA
                                auto msh_int = cl.user_data.integration_msh ;
                                for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                                {
                                    for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                                    {
                                        if(i_local == 0)
                                            connectivity_matrix_dd[i_cl_global][0] = connectivity_matrix_dd[i_cl_global - 1][1];
                                        else if(i_local == 1)
                                                connectivity_matrix_dd[i_cl_global][1] = i_inner++ ;
                                        else
                                            connectivity_matrix_dd[i_cl_global][i_local] = i_inner++;
                                    }
//                                    connectivity_cells_dd[k_offset].push_back(i_cl_global); // ADD ORA
                                    i_cl_global++;
                                }
                                cell_end_point = *(cl.user_data.interface.end() -1) ;
                                     
                            }
                        }
                        
                    }
                    else
                        break;
                         
                 
                }
            }
            connectivity_matrix_dd[counter_subcls-1][1] = 0 ;
            for( size_t i_local = 2 ; i_local < deg_size ; i_local++)
                connectivity_matrix_dd[counter_subcls-1][i_local] -= 1 ;

            
        }
                    
            
//        std::cout<<"Beginning of connectivity matrix"<<std::endl;
//        for(size_t i = 0 ; i< counter_subcls ; i++)
//        {
//            for(size_t j = 0 ; j< deg_size ; j++)
//            {
//                std::cout<<" ( "<<connectivity_matrix_dd[i][j] << " ) " ;
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<std::endl;
//        std::cout<<"End of connectivity matrix"<<std::endl;
//        std::cout<<"Beginning of connectivity cell"<<std::endl;
//        for(size_t i = 0 ; i< msh.cells.size() ; i++)
//        {
//            std::cout<<"Cell = "<<offset(msh, msh.cells[i])<<" : ";
//            for(size_t j = 0 ; j< connectivity_cells_dd[i].size() ; j++)
//            {
//                std::cout<<" ( "<<connectivity_cells_dd[i][j] << " ) " ;
//            }
//            std::cout<<std::endl;
//        }
//        std::cout<<std::endl;
//        std::cout<<"End of connectivity cell"<<std::endl;

        // DEFINITION OF THE MASS MATRIX -> USEFUL FOR L2 PROJ

//        curvature_FE = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_dd );
            
        curvature_field = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);

    }
    
    
    void
    space_definition_curvature(const Mesh& msh , size_t deg , size_t deg_size)
    {

        std::vector<size_t> cut_cell_cointainer ;
        for(auto& cl : msh.cells)
        {
            if(location(msh, cl) == element_location::ON_INTERFACE )
            {
                cut_cell_cointainer.push_back(offset(msh,cl));
                    
            }
           
        }
        
        if (deg == 0)
            ndof_dd = counter_subcls ;
        else
            ndof_dd = counter_subcls * deg ;
        
        connectivity_matrix_dd.resize( counter_subcls , std::vector<size_t>(deg_size) ) ;
             
        size_t i_cl_global = 0 ;
        size_t i_inner = 2 ;
        point<T,2> first_point ;
        point<T,2> cell_end_point ;
        bool first_cut_cell_found = FALSE ;
        
        if( deg == 0 )
        {
            size_t cl_i = 0;
            while(cut_cell_cointainer.size()>0)
            {
                size_t k_offset =  cut_cell_cointainer[cl_i];
                auto cl = msh.cells[k_offset];
                auto msh_int = cl.user_data.integration_msh ;
                    
                if(!first_cut_cell_found)
                {
                    cut_cell_cointainer.erase(cut_cell_cointainer.begin() );
                    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                    {
                        connectivity_matrix_dd[i_cl_global][0] = i_cl_global;
                        i_cl_global++;
                        
                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin() ;
                    cell_end_point = *(cl.user_data.interface.end() -1) ;
                    cl_i = 0;
                }
                else if( first_cut_cell_found && cell_end_point == *cl.user_data.interface.begin() && !( first_point == cell_end_point  ) )
                {
                    cut_cell_cointainer.erase(cut_cell_cointainer.begin() + cl_i );
                    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                    {
                        connectivity_matrix_dd[i_cl_global][0] = i_cl_global;
                        i_cl_global++;
                    }
                    cell_end_point = *(cl.user_data.interface.end() -1) ;
                    cl_i = 0;
                }
                else if (first_point == cell_end_point)
                    break;
                else
                    cl_i++;
                        
            
            }
                        
        }
        else
        {
            size_t cl_i = 0;
            while(cut_cell_cointainer.size()>0)
            {
                size_t k_offset =  cut_cell_cointainer[cl_i];
                auto cl = msh.cells[k_offset];
                auto msh_int = cl.user_data.integration_msh ;
                if(!first_cut_cell_found)
                {
                    cut_cell_cointainer.erase(cut_cell_cointainer.begin() );
                    for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                    {
                        if(i_local == 0 || i_local == 1)
                            connectivity_matrix_dd[0][i_local] = i_local;
                        else
                            connectivity_matrix_dd[0][i_local] = i_inner++;
                    }
                    i_cl_global++;
                    for (size_t i_cell = 1; i_cell < msh_int.cells.size(); i_cell++)
                    {
                        for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                        {
                            if(i_local == 0)
                                connectivity_matrix_dd[i_cl_global][0] = connectivity_matrix_dd[i_cl_global - 1][1];
                            else if(i_local == 1)
                                connectivity_matrix_dd[i_cl_global][1] = i_inner++ ;
                            else
                                connectivity_matrix_dd[i_cl_global][i_local] = i_inner++;
                        }
//                            connectivity_cells_dd[k_offset].push_back(i_cl_global); // ADD ORA
                            i_cl_global++;
                    }
                    first_cut_cell_found = TRUE;
                    first_point = *cl.user_data.interface.begin() ;
                    cell_end_point = *(cl.user_data.interface.end() -1) ;
                    cl_i = 0;
                }
           
                
                else if( first_cut_cell_found && cell_end_point == *cl.user_data.interface.begin() && !( first_point == cell_end_point  ) )
                {
                    cut_cell_cointainer.erase(cut_cell_cointainer.begin() + cl_i );
                    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
                    {
                        for( size_t i_local = 0 ; i_local < deg_size ; i_local++)
                        {
                            if(i_local == 0)
                                connectivity_matrix_dd[i_cl_global][0] = connectivity_matrix_dd[i_cl_global - 1][1];
                            else if(i_local == 1)
                                connectivity_matrix_dd[i_cl_global][1] = i_inner++ ;
                            else
                                connectivity_matrix_dd[i_cl_global][i_local] = i_inner++;
                        }
 
                        i_cl_global++;
                    }
                    
                    cell_end_point = *(cl.user_data.interface.end() -1) ;
                    cl_i = 0;
                            
                }
                else if (first_point == cell_end_point)
                    break;
                else
                    cl_i++;
                        
            }
    
            connectivity_matrix_dd[counter_subcls-1][1] = 0 ;
            for( size_t i_local = 2 ; i_local < deg_size ; i_local++)
                connectivity_matrix_dd[counter_subcls-1][i_local] -= 1 ;

            
        }
                    
 
            
        curvature_field = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( deg_size, counter_subcls);

    }
    
    
    
    
    void
    make_L2_proj_para_derivative( const Mesh& msh , size_t di = 1)
    {
        
        SparseMatrix<T> Global_Mass = SparseMatrix<T>( ndof_der , ndof_der );
        std::vector< Triplet<T> >       triplets;
        Eigen::Matrix<T, Dynamic, 1> derivative_FE0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_der );
        Eigen::Matrix<T, Dynamic, 1> derivative_FE1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_der );
        Matrix<T, Dynamic, 1>  RHS0 = Matrix<T, Dynamic, 1>::Zero( ndof_der , 1);
        Matrix<T, Dynamic, 1>  RHS1 = Matrix<T, Dynamic, 1>::Zero( ndof_der , 1);
        size_t counter = 0;
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
                for (size_t i_cl = 0; i_cl < msh_int.cells.size(); i_cl++)
                {
                    auto pts = points(msh_int,msh_int.cells[i_cl]);
                    auto local_mass = make_local_mass_matrix_curve( der_degree ,basis_degree,degree_det, pts );
//                    std::cout<<"local_mass = "<<local_mass<<std::endl;
                    auto local_RHS = make_local_RHS_derivative( der_degree , basis_degree ,degree_det, pts );
//                    std::cout<<"local_RHS[0] = "<<local_RHS.first<<std::endl;
//                    std::cout<<"local_RHS[1] = "<<local_RHS.second<<std::endl;
                    size_t global_i_cl = connectivity_cells[counter][i_cl] ;
//                    std::cout<<"i_cl = "<<i_cl<<" , global i_cl = "<<global_i_cl<<std::endl;
                    for (size_t i = 0; i < der_size ; i++)
                    {
                        size_t asm_map_i = connectivity_matrix_der[global_i_cl][i];
//                        std::cout<<"i = "<<i<<" , asm_map_i = "<<asm_map_i<<std::endl;
                        for (size_t j = 0; j < der_size ; j++)
                        {
                            size_t asm_map_j = connectivity_matrix_der[global_i_cl][j] ;
//                            std::cout<<"j = "<<j<<" , asm_map_j = "<<asm_map_j<<std::endl;
                            triplets.push_back(Triplet<T>(asm_map_i,asm_map_j, local_mass(i,j)));
                        }
                        RHS0(asm_map_i) += local_RHS.first(i) ;
                        RHS1(asm_map_i) += local_RHS.second(i) ;

                    }
                }
            }
            counter++;
        }

        Global_Mass.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
//        std::cout<<"RHS0 = "<<RHS0<<std::endl;
//        std::cout<<"RHS1 = "<<RHS1<<std::endl;
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        derivative_FE0 = solver_global_mass.solve(RHS0);
        derivative_FE1 = solver_global_mass.solve(RHS1);
        
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            for (size_t i = 0; i < der_size; i++)
            {
                size_t asm_map =  connectivity_matrix_der[counter_bis][i] ;
                derivative0(i,counter_bis) = derivative_FE0( asm_map ) ;
                derivative1(i,counter_bis) = derivative_FE1( asm_map ) ;
                
//                std::cout<<"asm_map = "<<asm_map<<" , i = "<<i<<" , counter_bis = "<<counter_bis <<std::endl;
//                std::cout<< "derivative0(i,counter_bis) = "<<derivative0(i,counter_bis)<<" , derivative1(i,counter_bis) = "<<derivative1(i,counter_bis) <<std::endl;
            }

        }
    
                                            
    }
    
    
    template< typename Level_Set >
    void
    make_L2_proj_para_normal( const Mesh& msh , Level_Set& ls_cell , size_t di = 1)
    {
            
        SparseMatrix<T> Global_Mass = SparseMatrix<T>( ndof_der , ndof_der );
        std::vector< Triplet<T> >       triplets;
        Eigen::Matrix<T, Dynamic, 1> derivative_FE0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_der );
        Eigen::Matrix<T, Dynamic, 1> derivative_FE1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_der );
        Matrix<T, Dynamic, 1>  RHS0 = Matrix<T, Dynamic, 1>::Zero( ndof_der , 1);
        Matrix<T, Dynamic, 1>  RHS1 = Matrix<T, Dynamic, 1>::Zero( ndof_der , 1);
        size_t counter = 0;
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
                for (size_t i_cl = 0; i_cl < msh_int.cells.size(); i_cl++)
                {
                    auto pts = points(msh_int,msh_int.cells[i_cl]);
                    auto local_mass = make_local_mass_matrix_curve( der_degree ,basis_degree,degree_det, pts );
    //                    std::cout<<"local_mass = "<<local_mass<<std::endl;
                    ls_cell.cell_assignment(cl);
                    auto local_RHS = make_local_RHS_normal( der_degree , basis_degree ,degree_det, pts , ls_cell, msh);
                    
    //                    std::cout<<"local_RHS[0] = "<<local_RHS.first<<std::endl;
    //                    std::cout<<"local_RHS[1] = "<<local_RHS.second<<std::endl;
                    size_t global_i_cl = connectivity_cells[counter][i_cl] ;
    //                    std::cout<<"i_cl = "<<i_cl<<" , global i_cl = "<<global_i_cl<<std::endl;
                    for (size_t i = 0; i < der_size ; i++)
                    {
                        size_t asm_map_i = connectivity_matrix_der[global_i_cl][i];
    //                        std::cout<<"i = "<<i<<" , asm_map_i = "<<asm_map_i<<std::endl;
                        for (size_t j = 0; j < der_size ; j++)
                        {
                            size_t asm_map_j = connectivity_matrix_der[global_i_cl][j] ;
    //                            std::cout<<"j = "<<j<<" , asm_map_j = "<<asm_map_j<<std::endl;
                            triplets.push_back(Triplet<T>(asm_map_i,asm_map_j, local_mass(i,j)));
                        }
                        RHS0(asm_map_i) += local_RHS.first(i) ;
                        RHS1(asm_map_i) += local_RHS.second(i) ;

                    }
                }
            }
            counter++;
        }

        Global_Mass.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    //        std::cout<<"RHS0 = "<<RHS0<<std::endl;
    //        std::cout<<"RHS1 = "<<RHS1<<std::endl;
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        derivative_FE0 = solver_global_mass.solve(RHS0);
        derivative_FE1 = solver_global_mass.solve(RHS1);
            
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            for (size_t i = 0; i < der_size; i++)
            {
                size_t asm_map =  connectivity_matrix_der[counter_bis][i] ;
                normal0(i,counter_bis) = derivative_FE0( asm_map ) ;
                normal1(i,counter_bis) = derivative_FE1( asm_map ) ;
                    
    //                std::cout<<"asm_map = "<<asm_map<<" , i = "<<i<<" , counter_bis = "<<counter_bis <<std::endl;
    //                std::cout<< "derivative0(i,counter_bis) = "<<derivative0(i,counter_bis)<<" , derivative1(i,counter_bis) = "<<derivative1(i,counter_bis) <<std::endl;
            }
            
        }
        
    }
    
    
    template< typename Level_Set >
    void
    make_L2_proj_para_normal_disc( const Mesh& msh , Level_Set& ls_cell , size_t di = 1)
    {
            
        SparseMatrix<T> Global_Mass = SparseMatrix<T>( ndof_der , ndof_der );
        std::vector< Triplet<T> >       triplets;
        Eigen::Matrix<T, Dynamic, 1> derivative_FE0 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_der );
        Eigen::Matrix<T, Dynamic, 1> derivative_FE1 = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_der );
        Matrix<T, Dynamic, 1>  RHS0 = Matrix<T, Dynamic, 1>::Zero( ndof_der , 1);
        Matrix<T, Dynamic, 1>  RHS1 = Matrix<T, Dynamic, 1>::Zero( ndof_der , 1);
        size_t counter = 0;
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
                for (size_t i_cl = 0; i_cl < msh_int.cells.size(); i_cl++)
                {
                    auto pts = points(msh_int,msh_int.cells[i_cl]);
                    auto local_mass = make_local_mass_matrix_curve( der_degree ,basis_degree,degree_det, pts );
    //                    std::cout<<"local_mass = "<<local_mass<<std::endl;
                    ls_cell.cell_assignment(cl);
                    auto local_RHS = make_local_RHS_normal_disc( der_degree , basis_degree ,degree_det, pts , ls_cell, msh);
                    
    //                    std::cout<<"local_RHS[0] = "<<local_RHS.first<<std::endl;
    //                    std::cout<<"local_RHS[1] = "<<local_RHS.second<<std::endl;
                    size_t global_i_cl = connectivity_cells[counter][i_cl] ;
    //                    std::cout<<"i_cl = "<<i_cl<<" , global i_cl = "<<global_i_cl<<std::endl;
                    for (size_t i = 0; i < der_size ; i++)
                    {
                        size_t asm_map_i = connectivity_matrix_der[global_i_cl][i];
    //                        std::cout<<"i = "<<i<<" , asm_map_i = "<<asm_map_i<<std::endl;
                        for (size_t j = 0; j < der_size ; j++)
                        {
                            size_t asm_map_j = connectivity_matrix_der[global_i_cl][j] ;
    //                            std::cout<<"j = "<<j<<" , asm_map_j = "<<asm_map_j<<std::endl;
                            triplets.push_back(Triplet<T>(asm_map_i,asm_map_j, local_mass(i,j)));
                        }
                        RHS0(asm_map_i) += local_RHS.first(i) ;
                        RHS1(asm_map_i) += local_RHS.second(i) ;

                    }
                }
            }
            counter++;
        }

        Global_Mass.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    //        std::cout<<"RHS0 = "<<RHS0<<std::endl;
    //        std::cout<<"RHS1 = "<<RHS1<<std::endl;
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        derivative_FE0 = solver_global_mass.solve(RHS0);
        derivative_FE1 = solver_global_mass.solve(RHS1);
            
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            for (size_t i = 0; i < der_size; i++)
            {
                size_t asm_map =  connectivity_matrix_der[counter_bis][i] ;
                normal0(i,counter_bis) = derivative_FE0( asm_map ) ;
                normal1(i,counter_bis) = derivative_FE1( asm_map ) ;
                    
    //                std::cout<<"asm_map = "<<asm_map<<" , i = "<<i<<" , counter_bis = "<<counter_bis <<std::endl;
    //                std::cout<< "derivative0(i,counter_bis) = "<<derivative0(i,counter_bis)<<" , derivative1(i,counter_bis) = "<<derivative1(i,counter_bis) <<std::endl;
            }
            
        }
        
    }
    
    

    
    template<typename Level_Set>
    void
    make_avg_L2_local_proj_para_normal_bis( const Mesh& msh , Level_Set& ls_cell, size_t di = 1)
    {
        std::cout<<"---> Parametric normal smooth: CONTINUITY BY AVG."<<std::endl;
        // new_HHO contains the discontinuous curvature field
        Matrix<T, Dynamic, Dynamic>  new_HHO0 = Matrix<T, Dynamic, Dynamic>::Zero( der_size, counter_subcls );
        Matrix<T, Dynamic, Dynamic>  new_HHO1 = Matrix<T, Dynamic, Dynamic>::Zero( der_size, counter_subcls );
        
        Interface_parametrisation_mesh1d curve(basis_degree);
        auto ref_nodes = reference_nodes_ordered_01<T>(der_degree);
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
                ls_cell.cell_assignment(cl);
                auto T_cells =  get_global_cells_interface(msh , cl);
                size_t loc_counter = 0 ;
                for(auto& Tj : T_cells )
                {
                    auto pts = points(msh_int,msh_int.cells[loc_counter]);
                    for (size_t i = 0; i < der_size; i++)
                    {
                        auto p = curve(ref_nodes[i].x() , pts ) ;
                        point<T,2> pt = typename Mesh::point_type( p(0) , p(1) ) ;
                        
                        std::vector< size_t> offset_cells ;
                        if( i == 0 || i == 1 )
                            offset_cells = pt_in_skeleton(ls_cell.level_set.msh,pt);
//                        else
//                            new_HHO(i,Tj) = ls_cell.divergence(pt) ;
                                                
                                                                                    
                        if(offset_cells.size() == 0 || offset_cells.size() == 1){
                            auto val = ls_cell.normal(pt) ;
                            new_HHO0(i,Tj) = val(0) ;
                            new_HHO1(i,Tj) = val(1) ;
                    
                        }
                                                
                        else if(offset_cells.size() == 2)
                        {
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            if(Tj == 0 && loc_counter == 0 )
                            {
                                if( i == 0 ){
                                    auto valcl0 = ls_cell.normal(pt, subcl0) ;
                                    new_HHO0(1,counter_subcls - 1) = valcl0(0) ;
                                    new_HHO1(1,counter_subcls - 1) = valcl0(1) ;
                                    
                                    auto valcl1 = ls_cell.normal(pt, subcl1) ;
                                    new_HHO0(0,Tj) = valcl1(0) ;
                                    new_HHO1(0,Tj) = valcl1(1) ;
//                                    new_HHO(1,counter_subcls - 1) = ls_cell.divergence( pt , subcl0);
//                                    new_HHO(0,Tj) = ls_cell.divergence( pt , subcl1 );
                                                       
                                }
//                                if(i == 1 && loc_counter == msh_int.cells.size()-1 ){
//                                    auto valcl0 = ls_cell.normal(pt, subcl0) ;
//                                    new_HHO0(0,Tj + 1) = valcl0(0) ;
//                                    new_HHO1(0,Tj + 1) = valcl0(1) ;
//
//                                    auto valcl1 = ls_cell.normal(pt, subcl1) ;
//                                    new_HHO0(1,Tj) = valcl1(0) ;
//                                    new_HHO1(1,Tj) = valcl1(1) ;
////                                    new_HHO(0,Tj + 1) = ls_cell.divergence( pt , subcl0 );
////                                    new_HHO(1,Tj) = ls_cell.divergence( pt , subcl1 );
//                                }
                                                        
                            }
                            else if(Tj == counter_subcls - 1 && loc_counter == msh_int.cells.size()-1 && i == 1 ){
//                                if(i == 0 )
//                                 new_HHO(1,Tj - 1) = ls_cell.divergence( pt , subcl0 );
//                                 new_HHO(0,Tj) = ls_cell.divergence( pt , subcl1 );
//
//                                 }
                                                        
                                // Already made in Tj = 0
//                                 if(i == 1 && loc_counter == msh_int.cells.size()-1 ){
//                                 new_HHO(0,0) = ls_cell.divergence( pt , subcl1 );
//                                 new_HHO(1,Tj) = ls_cell.divergence( pt , subcl0 );
//                                 }

                                                        
                            
                            }
                                                    
                        
                            else
                            {
                                if(i == 0 ){
                                    auto valcl0 = ls_cell.normal(pt, subcl0) ;
                                    new_HHO0(1,Tj - 1) = valcl0(0) ;
                                    new_HHO1(1,Tj - 1) = valcl0(1) ;
                                    
                                    auto valcl1 = ls_cell.normal(pt, subcl1) ;
                                    new_HHO0(0,Tj) = valcl1(0) ;
                                    new_HHO1(0,Tj) = valcl1(1) ;
                                    
//                                    new_HHO(1,Tj - 1) = ls_cell.divergence( pt , subcl0 );
//                                    new_HHO(0,Tj) = ls_cell.divergence( pt , subcl1 );
                                }
                                if(i == 1 ){
                                    auto valcl0 = ls_cell.normal(pt, subcl0) ;
                                    new_HHO0(1,Tj) = valcl0(0) ;
                                    new_HHO1(1,Tj) = valcl0(1) ;
                                    
                                    auto valcl1 = ls_cell.normal(pt, subcl1) ;
                                    new_HHO0(0,Tj+1) = valcl1(0) ;
                                    new_HHO1(0,Tj+1) = valcl1(1) ;
                                    
//                                    new_HHO(1,Tj) = ls_cell.divergence( pt , subcl0 );
//                                    new_HHO(0,Tj+1) = ls_cell.divergence( pt , subcl1 );
                                }
                            }
                                                    
                        }
                        else
                        {
                            std::cout<<"Error pt = "<<pt<<" between "<<offset_cells.size()<<" cells: particularly: ";
                            for(auto& offset_cell : offset_cells)
                                std::cout<<" cell = "<< offset_cell << " , ";
                            std::cout<<std::endl;
                            exit(9);
                        }
                                                
//                        new_HHO(i,Tj) = ls_cell.divergence(pt) ;
//                        std::cout<<"Cell Tj = "<<Tj<<" , node i = "<<i <<" , pt = "<<pt <<" -> val = "<<new_HHO(i,Tj)<<std::endl;
                        
                        
                        
//                        std::cout<<"Cell Tj = "<<Tj<<" , node i = "<<i <<" , pt = "<<pt <<" -> val = "<<new_HHO(i,Tj)<<std::endl;
                    }
                    loc_counter++;
                }
            }
        }
                
        size_t counter_old , counter_next ;
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            if(counter_bis == 0){
                counter_old = counter_subcls - 1;
                counter_next = counter_bis + 1;
            }
            else if(counter_bis == counter_subcls - 1){
                counter_old = counter_bis - 1;
                counter_next = 0;
            }
            else{
                counter_old = counter_bis - 1;
                counter_next = counter_bis + 1;
            }
                    
    //            std::cout<<"Cell Tj = "<<counter_bis<<" , cell old = "<<counter_old<<std::endl;
            for (size_t i = 0; i < der_size; i++)
            {
    //                size_t asm_map =  connectivity_matrix_dd[counter_bis][i] ;
                if( i == 0 ){
                    normal0(0,counter_bis) = ( new_HHO0(0,counter_bis) + new_HHO0(1,counter_old)) *0.5;
                    normal1(0,counter_bis) = ( new_HHO1(0,counter_bis) + new_HHO1(1,counter_old)) *0.5;
                }
                else if( i == 1 ){
                    normal0(1,counter_bis) = ( new_HHO0(1,counter_bis) + new_HHO0(0,counter_next)) *0.5;
                    normal1(1,counter_bis) = ( new_HHO1(1,counter_bis) + new_HHO1(0,counter_next)) *0.5;
                }
                else{
                    normal0(i,counter_bis) = new_HHO0(i,counter_bis) ;
                    normal1(i,counter_bis) = new_HHO1(i,counter_bis) ;
                        
                    
                //                curvature_FE( asm_map ) = curvature_field(i,counter_bis);
                }
   
                                   
    //                std::cout<<"Node i = "<<i<<" , curvature_field(0,counter_bis)= "<<curvature_field(i,counter_bis)<<std::endl;
            }
        }
                                               
    }
    
    template<typename Level_Set>
    void
    make_avg_L2_local_proj_para_normal( const Mesh& msh , Level_Set& ls_cell, size_t di = 1)
    {
        std::cout<<"---> Parametric normal smooth: CONTINUITY BY AVG."<<std::endl;
        // new_HHO contains the discontinuous curvature field
        Matrix<T, Dynamic, Dynamic>  new_HHO0 = Matrix<T, Dynamic, Dynamic>::Zero( der_size, counter_subcls );
        Matrix<T, Dynamic, Dynamic>  new_HHO1 = Matrix<T, Dynamic, Dynamic>::Zero( der_size, counter_subcls );
            
        Interface_parametrisation_mesh1d curve(basis_degree);
        auto ref_nodes = reference_nodes_ordered_01<T>(der_degree);
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
                ls_cell.cell_assignment(cl);
                auto T_cells =  get_global_cells_interface(msh , cl);
                size_t loc_counter = 0 ;
                for(auto& Tj : T_cells )
                {
                    auto pts = points(msh_int,msh_int.cells[loc_counter]);
                    for (size_t i = 0; i < der_size; i++)
                    {
                        auto p = curve(ref_nodes[i].x() , pts ) ;
                        point<T,2> pt = typename Mesh::point_type( p(0) , p(1) ) ;
                            
                        std::vector< size_t> offset_cells ;
                        if( i == 0 || i == 1 )
                            offset_cells = pt_in_skeleton(ls_cell.level_set.msh,pt);
    //                        else
    //                            new_HHO(i,Tj) = ls_cell.divergence(pt) ;
                                                    
                                                                                        
                        if(offset_cells.size() == 0 || offset_cells.size() == 1){
                            auto val = ls_cell.normal(pt) ;
                            new_HHO0(i,Tj) = val(0) ;
                            new_HHO1(i,Tj) = val(1) ;
                        
                        }
                                                    
                        if(offset_cells.size() == 2)
                        {
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            if(Tj == 0 && loc_counter == 0 )
                            {
                                if( i == 0 )
                                {
                                    auto valcl0 = ls_cell.normal(pt, subcl0) ;
                                    new_HHO0(1,counter_subcls - 1) = valcl0(0) ;
                                    new_HHO1(1,counter_subcls - 1) = valcl0(1) ;
                                        
                                    auto valcl1 = ls_cell.normal(pt, subcl1) ;
                                    new_HHO0(0,Tj) = valcl1(0) ;
                                    new_HHO1(0,Tj) = valcl1(1) ;
  
                                }
   
                                                            
                            }
                            else if ( i == 0 )
                            {
                                
                                auto valcl0 = ls_cell.normal(pt, subcl0) ;
                                new_HHO0(1,Tj - 1) = valcl0(0) ;
                                new_HHO1(1,Tj - 1) = valcl0(1) ;
                                        
                                auto valcl1 = ls_cell.normal(pt, subcl1) ;
                                new_HHO0(0,Tj) = valcl1(0) ;
                                new_HHO1(0,Tj) = valcl1(1) ;
                                        
                                    
                            }
                                                        
                        }
                        
                    }
                    loc_counter++;
                    
                }
                
            }
            
        }
                    
        size_t counter_old , counter_next ;
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            if(counter_bis == 0){
                counter_old = counter_subcls - 1;
                counter_next = counter_bis + 1;
            }
            else if(counter_bis == counter_subcls - 1){
                counter_old = counter_bis - 1;
                counter_next = 0;
            }
            else{
                counter_old = counter_bis - 1;
                counter_next = counter_bis + 1;
            }
                        
        //            std::cout<<"Cell Tj = "<<counter_bis<<" , cell old = "<<counter_old<<std::endl;
            for (size_t i = 0; i < der_size; i++)
            {
        //                size_t asm_map =  connectivity_matrix_dd[counter_bis][i] ;
                if( i == 0 ){
                    normal0(0,counter_bis) = ( new_HHO0(0,counter_bis) + new_HHO0(1,counter_old)) *0.5;
                    normal1(0,counter_bis) = ( new_HHO1(0,counter_bis) + new_HHO1(1,counter_old)) *0.5;
                    }
                else if( i == 1 ){
                    normal0(1,counter_bis) = ( new_HHO0(1,counter_bis) + new_HHO0(0,counter_next)) *0.5;
                    normal1(1,counter_bis) = ( new_HHO1(1,counter_bis) + new_HHO1(0,counter_next)) *0.5;
                }
                else{
                    normal0(i,counter_bis) = new_HHO0(i,counter_bis) ;
                    normal1(i,counter_bis) = new_HHO1(i,counter_bis) ;
                            
                        
                    //                curvature_FE( asm_map ) = curvature_field(i,counter_bis);
                }
       
                                       
        //                std::cout<<"Node i = "<<i<<" , curvature_field(0,counter_bis)= "<<curvature_field(i,counter_bis)<<std::endl;
            }
        }
                                                
    }
    
    template<typename Level_Set>
    void
    make_avg_L2_local_proj_para_normal_disc( const Mesh& msh , Level_Set& ls_cell, size_t di = 1)
    {
        std::cout<<"---> Parametric normal smooth: CONTINUITY BY AVG."<<std::endl;
          // new_HHO contains the discontinuous curvature field
          Matrix<T, Dynamic, Dynamic>  new_HHO0 = Matrix<T, Dynamic, Dynamic>::Zero( der_size, counter_subcls );
          Matrix<T, Dynamic, Dynamic>  new_HHO1 = Matrix<T, Dynamic, Dynamic>::Zero( der_size, counter_subcls );
              
          Interface_parametrisation_mesh1d curve(basis_degree);
          auto ref_nodes = reference_nodes_ordered_01<T>(der_degree);
          for( const auto& cl : msh.cells )
          {
              if( location(msh, cl) == element_location::ON_INTERFACE )
              {
                  auto msh_int = cl.user_data.integration_msh ;
                  ls_cell.cell_assignment(cl);
                  auto T_cells =  get_global_cells_interface(msh , cl);
                  size_t loc_counter = 0 ;
                  for(auto& Tj : T_cells )
                  {
                      auto pts = points(msh_int,msh_int.cells[loc_counter]);
                      for (size_t i = 0; i < der_size; i++)
                      {
                          auto p = curve(ref_nodes[i].x() , pts ) ;
                          point<T,2> pt = typename Mesh::point_type( p(0) , p(1) ) ;
                              
                          std::vector< size_t> offset_cells ;
                          if( i == 0 || i == 1 )
                              offset_cells = pt_in_skeleton(ls_cell.level_set.msh,pt);
      //                        else
      //                            new_HHO(i,Tj) = ls_cell.divergence(pt) ;
                                                      
                                                                                          
                          if(offset_cells.size() == 0 || offset_cells.size() == 1){
                              auto val = ls_cell.normal_disc(pt) ;
                              new_HHO0(i,Tj) = val(0) ;
                              new_HHO1(i,Tj) = val(1) ;
                          
                          }
                                                      
                          if(offset_cells.size() == 2)
                          {
                              auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                              auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                              if(Tj == 0 && loc_counter == 0 )
                              {
                                  if( i == 0 )
                                  {
                                      auto valcl0 = ls_cell.normal_disc(pt, subcl0) ;
                                      new_HHO0(1,counter_subcls - 1) = valcl0(0) ;
                                      new_HHO1(1,counter_subcls - 1) = valcl0(1) ;
                                          
                                      auto valcl1 = ls_cell.normal_disc(pt, subcl1) ;
                                      new_HHO0(0,Tj) = valcl1(0) ;
                                      new_HHO1(0,Tj) = valcl1(1) ;
    
                                  }
     
                                                              
                              }
                              else if ( i == 0 )
                              {
                                  
                                  auto valcl0 = ls_cell.normal_disc(pt, subcl0) ;
                                  new_HHO0(1,Tj - 1) = valcl0(0) ;
                                  new_HHO1(1,Tj - 1) = valcl0(1) ;
                                          
                                  auto valcl1 = ls_cell.normal_disc(pt, subcl1) ;
                                  new_HHO0(0,Tj) = valcl1(0) ;
                                  new_HHO1(0,Tj) = valcl1(1) ;
                                          
                                      
                              }
                                                          
                          }
                          
                      }
                      loc_counter++;
                      
                  }
                  
              }
              
          }
                      
          size_t counter_old , counter_next ;
          for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
          {
              if(counter_bis == 0){
                  counter_old = counter_subcls - 1;
                  counter_next = counter_bis + 1;
              }
              else if(counter_bis == counter_subcls - 1){
                  counter_old = counter_bis - 1;
                  counter_next = 0;
              }
              else{
                  counter_old = counter_bis - 1;
                  counter_next = counter_bis + 1;
              }
                          
          //            std::cout<<"Cell Tj = "<<counter_bis<<" , cell old = "<<counter_old<<std::endl;
              for (size_t i = 0; i < der_size; i++)
              {
          //                size_t asm_map =  connectivity_matrix_dd[counter_bis][i] ;
                  if( i == 0 ){
                      normal0(0,counter_bis) = ( new_HHO0(0,counter_bis) + new_HHO0(1,counter_old)) *0.5;
                      normal1(0,counter_bis) = ( new_HHO1(0,counter_bis) + new_HHO1(1,counter_old)) *0.5;
                      }
                  else if( i == 1 ){
                      normal0(1,counter_bis) = ( new_HHO0(1,counter_bis) + new_HHO0(0,counter_next)) *0.5;
                      normal1(1,counter_bis) = ( new_HHO1(1,counter_bis) + new_HHO1(0,counter_next)) *0.5;
                  }
                  else{
                      normal0(i,counter_bis) = new_HHO0(i,counter_bis) ;
                      normal1(i,counter_bis) = new_HHO1(i,counter_bis) ;
                              
                          
                      //                curvature_FE( asm_map ) = curvature_field(i,counter_bis);
                  }
         
                                         
          //                std::cout<<"Node i = "<<i<<" , curvature_field(0,counter_bis)= "<<curvature_field(i,counter_bis)<<std::endl;
              }
          }
                                                  
      }
    
    
    
    void
    make_L2_proj_para_curvature( const Mesh& msh , size_t di = 1)
    {

        std::cout<<"---> Parametric CURVATURE smooth: CONTINUITY BY L2 proj of parametric curvature."<<std::endl;
        SparseMatrix<T> Global_Mass = SparseMatrix<T>( ndof_dd, ndof_dd );
        std::vector< Triplet<T> >       triplets;
        Matrix<T, Dynamic, 1> curvature_FE = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_dd );
        Matrix<T, Dynamic, 1>  RHS = Matrix<T, Dynamic, 1>::Zero( ndof_dd , 1);
        size_t counter = 0;
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
                for (size_t i_cl = 0; i_cl < msh_int.cells.size(); i_cl++)
                {
                    auto pts = points(msh_int,msh_int.cells[i_cl]);
                    auto local_mass = make_local_mass_matrix_curve( dd_degree,basis_degree , degree_det,pts );
//                    std::cout<<"local_mass = "<<local_mass<<std::endl;
                   
                    auto local_RHS = make_local_RHS_curvature( dd_degree , basis_degree ,degree_det, pts );
//                    std::cout<<"local_RHS = "<<local_RHS<<std::endl;
                    size_t global_i_cl = connectivity_cells[counter][i_cl] ;
//                    std::cout<<"i_cl = "<<i_cl<<" , global i_cl = "<<global_i_cl<<std::endl;
                    for (size_t i = 0; i < dd_size ; i++)
                    {
                        size_t asm_map_i = connectivity_matrix_dd[global_i_cl][i];
//                        std::cout<<"i = "<<i<<" , asm_map_i = "<<asm_map_i<<std::endl;
                        for (size_t j = 0; j < dd_size ; j++)
                        {
                            size_t asm_map_j = connectivity_matrix_dd[global_i_cl][j] ;
//                            std::cout<<"j = "<<j<<" , asm_map_j = "<<asm_map_j<<std::endl;
                            triplets.push_back(Triplet<T>(asm_map_i,asm_map_j, local_mass(i,j)));
                        }
                        RHS(asm_map_i) += local_RHS(i) ;

                    }
                }
            }
            counter++;
        }

        Global_Mass.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
//        std::cout<<"RHS = "<<RHS<<std::endl;
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        curvature_FE = solver_global_mass.solve(RHS);
        
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            for (size_t i = 0; i < dd_size; i++)
            {
                size_t asm_map =  connectivity_matrix_dd[counter_bis][i] ;
                curvature_field(i,counter_bis) = curvature_FE( asm_map ) ;
                
//                std::cout<<"asm_map = "<<asm_map<<" , i = "<<i<<" , counter_bis = "<<counter_bis <<std::endl;
//                 std::cout<< "curvature_field(i,counter_bis) = "<<curvature_field(i,counter_bis) <<std::endl;
            }

        }
    
                                            
    }
    
    
    template<typename Level_Set>
    void
    make_L2_proj_para_curvature( const Mesh& msh , Level_Set& ls_cell, size_t di = 1)
    {
        std::cout<<"---> Parametric CURVATURE smooth: CONTINUITY BY L2 proj of LS discrete curvature."<<std::endl;
        SparseMatrix<T> Global_Mass = SparseMatrix<T>( ndof_dd, ndof_dd );
        std::vector< Triplet<T> >       triplets;
        Matrix<T, Dynamic, 1> curvature_FE = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_dd );
        Matrix<T, Dynamic, 1>  RHS = Matrix<T, Dynamic, 1>::Zero( ndof_dd , 1);
        size_t counter = 0;
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
                for (size_t i_cl = 0; i_cl < msh_int.cells.size(); i_cl++)
                {
                    auto pts = points(msh_int,msh_int.cells[i_cl]);
                    auto local_mass = make_local_mass_matrix_curve( dd_degree,basis_degree ,degree_det, pts );
    //                    std::cout<<"local_mass = "<<local_mass<<std::endl;
                    ls_cell.cell_assignment(cl);
                    auto local_RHS = make_local_RHS_curvature( dd_degree , basis_degree ,degree_det, pts , ls_cell, msh);
    //                    std::cout<<"local_RHS = "<<local_RHS<<std::endl;
                    size_t global_i_cl = connectivity_cells[counter][i_cl] ;
    //                    std::cout<<"i_cl = "<<i_cl<<" , global i_cl = "<<global_i_cl<<std::endl;
                    for (size_t i = 0; i < dd_size ; i++)
                    {
                        size_t asm_map_i = connectivity_matrix_dd[global_i_cl][i];
    //                        std::cout<<"i = "<<i<<" , asm_map_i = "<<asm_map_i<<std::endl;
                        for (size_t j = 0; j < dd_size ; j++)
                        {
                            size_t asm_map_j = connectivity_matrix_dd[global_i_cl][j] ;
    //                            std::cout<<"j = "<<j<<" , asm_map_j = "<<asm_map_j<<std::endl;
                            triplets.push_back(Triplet<T>(asm_map_i,asm_map_j, local_mass(i,j)));
                        }
                        RHS(asm_map_i) += local_RHS(i) ;

                    }
                }
            }
            counter++;
        }

        Global_Mass.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    //        std::cout<<"RHS = "<<RHS<<std::endl;
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        curvature_FE = solver_global_mass.solve(RHS);
            
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            for (size_t i = 0; i < dd_size; i++)
            {
                size_t asm_map =  connectivity_matrix_dd[counter_bis][i] ;
                curvature_field(i,counter_bis) = curvature_FE( asm_map ) ;
                    
    //                std::cout<<"asm_map = "<<asm_map<<" , i = "<<i<<" , counter_bis = "<<counter_bis <<std::endl;
    //                 std::cout<< "curvature_field(i,counter_bis) = "<<curvature_field(i,counter_bis) <<std::endl;
            }

        }
        
                                                
    }
    
    template<typename Level_Set>
    void
    make_L2_proj_para_curvature_disc( const Mesh& msh , Level_Set& ls_cell, size_t di = 1)
    {
        std::cout<<"---> Parametric CURVATURE smooth: CONTINUITY BY L2 proj of LS discrete curvature."<<std::endl;
        SparseMatrix<T> Global_Mass = SparseMatrix<T>( ndof_dd, ndof_dd );
        std::vector< Triplet<T> >       triplets;
        Matrix<T, Dynamic, 1> curvature_FE = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_dd );
        Matrix<T, Dynamic, 1>  RHS = Matrix<T, Dynamic, 1>::Zero( ndof_dd , 1);
        size_t counter = 0;
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
                for (size_t i_cl = 0; i_cl < msh_int.cells.size(); i_cl++)
                {
                    auto pts = points(msh_int,msh_int.cells[i_cl]);
                    auto local_mass = make_local_mass_matrix_curve( dd_degree,basis_degree ,degree_det, pts );
    //                    std::cout<<"local_mass = "<<local_mass<<std::endl;
                    ls_cell.cell_assignment(cl);
                    auto local_RHS = make_local_RHS_curvature_disc( dd_degree , basis_degree ,degree_det, pts , ls_cell, msh);
    //                    std::cout<<"local_RHS = "<<local_RHS<<std::endl;
                    size_t global_i_cl = connectivity_cells[counter][i_cl] ;
    //                    std::cout<<"i_cl = "<<i_cl<<" , global i_cl = "<<global_i_cl<<std::endl;
                    for (size_t i = 0; i < dd_size ; i++)
                    {
                        size_t asm_map_i = connectivity_matrix_dd[global_i_cl][i];
    //                        std::cout<<"i = "<<i<<" , asm_map_i = "<<asm_map_i<<std::endl;
                        for (size_t j = 0; j < dd_size ; j++)
                        {
                            size_t asm_map_j = connectivity_matrix_dd[global_i_cl][j] ;
    //                            std::cout<<"j = "<<j<<" , asm_map_j = "<<asm_map_j<<std::endl;
                            triplets.push_back(Triplet<T>(asm_map_i,asm_map_j, local_mass(i,j)));
                        }
                        RHS(asm_map_i) += local_RHS(i) ;

                    }
                }
            }
            counter++;
        }

        Global_Mass.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    //        std::cout<<"RHS = "<<RHS<<std::endl;
        SimplicialLLT<SparseMatrix<T> >solver_global_mass;
        solver_global_mass.compute(Global_Mass);
        curvature_FE = solver_global_mass.solve(RHS);
            
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            for (size_t i = 0; i < dd_size; i++)
            {
                size_t asm_map =  connectivity_matrix_dd[counter_bis][i] ;
                curvature_field(i,counter_bis) = curvature_FE( asm_map ) ;
                    
    //                std::cout<<"asm_map = "<<asm_map<<" , i = "<<i<<" , counter_bis = "<<counter_bis <<std::endl;
    //                 std::cout<< "curvature_field(i,counter_bis) = "<<curvature_field(i,counter_bis) <<std::endl;
            }

        }
        
                                                
    }
    
    
    
    void
    make_avg_L2_local_proj_para_curvature( const Mesh& msh , size_t di = 1)
    {
//        Matrix<T, Dynamic, 1> curvature_FE = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_dd );
        std::cout<<"---> Parametric CURVATURE smooth: CONTINUITY BY AVG of LS discrete curvature."<<std::endl;
       
        // new_HHO contains the discontinuous curvature field
//        Matrix<T, Dynamic, 1>  new_FEM = Matrix<T, Dynamic, 1>::Zero( ndof_dd , 1);
        Matrix<T, Dynamic, Dynamic>  new_HHO = Matrix<T, Dynamic, Dynamic>::Zero( dd_size, counter_subcls );
        Interface_parametrisation_mesh1d curve(basis_degree);
        auto ref_nodes = reference_nodes_ordered_01<T>(basis_degree);
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
               
                auto T_cells =  get_global_cells_interface(msh , cl);
                size_t loc_counter = 0 ;
                for(auto& Tj : T_cells )
                {
                    auto pts = points(msh_int,msh_int.cells[loc_counter]);
                    for (size_t i = 0; i < dd_size; i++)
                        new_HHO(i,Tj) = curve.curvature(ref_nodes[i],pts) ;
                    loc_counter++;
                }
            }
        }
        
        size_t counter_old , counter_next ;
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            if(counter_bis == 0){
                counter_old = counter_subcls - 1;
                counter_next = counter_bis + 1;
            }
            else if(counter_bis == counter_subcls - 1){
                counter_old = counter_bis - 1;
                counter_next = 0;
            }
            else{
                counter_old = counter_bis - 1;
                counter_next = counter_bis - 1;
            }
            
            for (size_t i = 0; i < dd_size; i++)
            {
//                size_t asm_map =  connectivity_matrix_dd[counter_bis][i] ;
                if( i == 0 )
                    curvature_field(0,counter_bis) = ( new_HHO(0,counter_bis) + new_HHO(1,counter_old)) *0.5;
                else if( i == 1 )
                    curvature_field(1,counter_bis) = ( new_HHO(1,counter_bis) + new_HHO(0,counter_next)) *0.5;
                else
                    curvature_field(i,counter_bis) = new_HHO(i,counter_bis) ;
                
//                curvature_FE( asm_map ) = curvature_field(i,counter_bis);
                           
            }
        }
       
                                                
    }
    
    template<typename Level_Set>
    void
    make_avg_L2_local_proj_para_curvature_bis( const Mesh& msh , Level_Set& ls_cell, size_t di = 1)
    {
        std::cout<<"---> Parametric CURVATURE smooth: CONTINUITY BY AVG of parametric curvature."<<std::endl;
//        Matrix<T, Dynamic, 1> curvature_FE = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_dd );
       
           
        // new_HHO contains the discontinuous curvature field
        Matrix<T, Dynamic, Dynamic>  new_HHO = Matrix<T, Dynamic, Dynamic>::Zero( dd_size, counter_subcls );
        Interface_parametrisation_mesh1d curve(basis_degree);
        auto ref_nodes = reference_nodes_ordered_01<T>(dd_degree);
       
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
                ls_cell.cell_assignment(cl);
                auto T_cells =  get_global_cells_interface(msh , cl);
                size_t loc_counter = 0 ;
                for(auto& Tj : T_cells )
                {
                    auto pts = points(msh_int,msh_int.cells[loc_counter]);
                    for (size_t i = 0; i < dd_size; i++){
                        auto p = curve(ref_nodes[i].x() , pts ) ;
                        point<T,2> pt = typename Mesh::point_type( p(0) , p(1) ) ;
                        std::vector< size_t> offset_cells ;
                        
                        if( i == 0 || i == 1 )
                            offset_cells = pt_in_skeleton(ls_cell.level_set.msh,pt);
//                        else
//                            new_HHO(i,Tj) = ls_cell.divergence(pt) ;
                        
                        
                        if(offset_cells.size() == 0 || offset_cells.size() == 1){
                            new_HHO(i,Tj) = ls_cell.divergence(pt) ;
                        }
                        
                        else if(offset_cells.size() == 2)
                        {
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            if(Tj == 0 && loc_counter == 0 )
                            {
                                if( i == 0 ){
                                new_HHO(1,counter_subcls - 1) = ls_cell.divergence( pt , subcl0 );
                                new_HHO(0,Tj) = ls_cell.divergence( pt , subcl1 );
                                }
                                if(i == 1 && loc_counter == msh_int.cells.size()-1 ){
                                new_HHO(0,Tj + 1) = ls_cell.divergence( pt , subcl0 );
                                new_HHO(1,Tj) = ls_cell.divergence( pt , subcl1 );
                                }
                                
                            }
                            else if(Tj == counter_subcls - 1 && loc_counter == msh_int.cells.size()-1 && i == 1 ){
//                                if(i == 0 ){
//                                 new_HHO(1,Tj - 1) = ls_cell.divergence( pt , subcl0 );
//                                 new_HHO(0,Tj) = ls_cell.divergence( pt , subcl1 );
//
//                                 }
                                
                                // Already made in Tj = 0
//                                 if(i == 1 && loc_counter == msh_int.cells.size()-1 ){
//                                 new_HHO(0,0) = ls_cell.divergence( pt , subcl1 );
//                                 new_HHO(1,Tj) = ls_cell.divergence( pt , subcl0 );
//                                 }

                                
                            }
                            
                            else
                            {
                                if(i == 0 ){
                                new_HHO(1,Tj - 1) = ls_cell.divergence( pt , subcl0 );
                                new_HHO(0,Tj) = ls_cell.divergence( pt , subcl1 );
                                }
                                if(i == 1 ){
                                new_HHO(1,Tj) = ls_cell.divergence( pt , subcl0 );
                                new_HHO(0,Tj+1) = ls_cell.divergence( pt , subcl1 );
                                }
                            }
                            
                        }
                        else{
                            std::cout<<"Error pt = "<<pt<<" between "<<offset_cells.size()<<" cells: particularly: ";
                            for(auto& offset_cell : offset_cells)
                                std::cout<<" cell = "<< offset_cell << " , ";
                            std::cout<<std::endl;
                            exit(9);
                        }
                        
//                        new_HHO(i,Tj) = ls_cell.divergence(pt) ;
//                        std::cout<<"Cell Tj = "<<Tj<<" , node i = "<<i <<" , pt = "<<pt <<" -> val = "<<new_HHO(i,Tj)<<std::endl;
                    }
                    loc_counter++;
                   
                }
            }
        }
            
        size_t counter_old , counter_next ;
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            if(counter_bis == 0){
                counter_old = counter_subcls - 1;
                counter_next = counter_bis + 1;
            }
            else if(counter_bis == counter_subcls - 1){
                counter_old = counter_bis - 1;
                counter_next = 0;
            }
            else{
                counter_old = counter_bis - 1;
                counter_next = counter_bis + 1;
            }
                
//            std::cout<<"Cell Tj = "<<counter_bis<<" , cell old = "<<counter_old<<std::endl;
            for (size_t i = 0; i < dd_size; i++)
            {
//                size_t asm_map =  connectivity_matrix_dd[counter_bis][i] ;
                if( i == 0 )
                    curvature_field(0,counter_bis) = ( new_HHO(0,counter_bis) + new_HHO(1,counter_old)) *0.5;
                else if( i == 1 )
                    curvature_field(1,counter_bis) = ( new_HHO(1,counter_bis) + new_HHO(0,counter_next)) *0.5;
                else
                    curvature_field(i,counter_bis) = new_HHO(i,counter_bis) ;
                    
//                curvature_FE( asm_map ) = curvature_field(i,counter_bis);
                
//                std::cout<<"Node i = "<<i<<" , curvature_field(i,counter_bis)= "<<curvature_field(i,counter_bis)<<" , new_HHO = "<<new_HHO(i,counter_bis)<<std::endl;
//                if(i == 0 )
//                std::cout<<"new_HHO(0,counter_bis) = "<<new_HHO(0,counter_bis)<<" , new_HHO(1,counter_old) = "<<new_HHO(1,counter_old)<<std::endl;
//                if(i == 1 )
//                std::cout<<"new_HHO(1,counter_bis) = "<<new_HHO(1,counter_bis)<<" , new_HHO(0,counter_next)) = "<<new_HHO(0,counter_next)<<std::endl;
            }
        }
                                           
    }
    
    

    template<typename Level_Set>
    void
    make_avg_L2_local_proj_para_curvature( const Mesh& msh , Level_Set& ls_cell, size_t di = 1)
    {
        std::cout<<"---> Parametric CURVATURE smooth: CONTINUITY BY AVG proj of parametric curvature."<<std::endl;
    //        Matrix<T, Dynamic, 1> curvature_FE = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_dd );
           
               
            // new_HHO contains the discontinuous curvature field
        Matrix<T, Dynamic, Dynamic>  new_HHO = Matrix<T, Dynamic, Dynamic>::Zero( dd_size, counter_subcls );
        Interface_parametrisation_mesh1d curve(basis_degree);
        auto ref_nodes = reference_nodes_ordered_01<T>(dd_degree);
           
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
                ls_cell.cell_assignment(cl);
                auto T_cells =  get_global_cells_interface(msh , cl);
                size_t loc_counter = 0 ;
                for(auto& Tj : T_cells )
                {
                    auto pts = points(msh_int,msh_int.cells[loc_counter]);
                    for (size_t i = 0; i < dd_size; i++){
                        auto p = curve(ref_nodes[i].x() , pts ) ;
                        point<T,2> pt = typename Mesh::point_type( p(0) , p(1) ) ;
                        std::vector< size_t> offset_cells ;
                            
                        if( i == 0 || i == 1 )
                            offset_cells = pt_in_skeleton(ls_cell.level_set.msh,pt);
    //                        else
    //                            new_HHO(i,Tj) = ls_cell.divergence(pt) ;
                            
                            
                        if(offset_cells.size() == 0 || offset_cells.size() == 1){
                            new_HHO(i,Tj) = ls_cell.divergence(pt) ;
                        }
                            
                        if(offset_cells.size() == 2)
                        {
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            if(Tj == 0 && loc_counter == 0 )
                            {
                                if( i == 0 ){
                                new_HHO(1,counter_subcls - 1) = ls_cell.divergence( pt , subcl0 );
                                new_HHO(0,Tj) = ls_cell.divergence( pt , subcl1 );
                                }
                               
                                    
                            }
                            else if(i == 0 )
                            {
                                new_HHO(1,Tj - 1) = ls_cell.divergence( pt , subcl0 );
                                new_HHO(0,Tj) = ls_cell.divergence( pt , subcl1 );
                                   
                            }
                            
                        }
    //                        new_HHO(i,Tj) = ls_cell.divergence(pt) ;
    //                        std::cout<<"Cell Tj = "<<Tj<<" , node i = "<<i <<" , pt = "<<pt <<" -> val = "<<new_HHO(i,Tj)<<std::endl;
                    }
                    loc_counter++;
                       
                }
            }
        }
                
        size_t counter_old , counter_next ;
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            if(counter_bis == 0){
                counter_old = counter_subcls - 1;
                counter_next = counter_bis + 1;
            }
            else if(counter_bis == counter_subcls - 1){
                counter_old = counter_bis - 1;
                counter_next = 0;
            }
            else{
                counter_old = counter_bis - 1;
                counter_next = counter_bis + 1;
            }
                    
//            std::cout<<"Cell Tj = "<<counter_bis<<" , cell old = "<<counter_old<<std::endl;
            for (size_t i = 0; i < dd_size; i++)
            {
    //                size_t asm_map =  connectivity_matrix_dd[counter_bis][i] ;
                if( i == 0 )
                    curvature_field(0,counter_bis) = ( new_HHO(0,counter_bis) + new_HHO(1,counter_old)) *0.5;
                else if( i == 1 )
                    curvature_field(1,counter_bis) = ( new_HHO(1,counter_bis) + new_HHO(0,counter_next)) *0.5;
                else
                    curvature_field(i,counter_bis) = new_HHO(i,counter_bis) ;
                        
    //                curvature_FE( asm_map ) = curvature_field(i,counter_bis);
                    
//                std::cout<<"Node i = "<<i<<" , curvature_field(i,counter_bis)= "<<curvature_field(i,counter_bis)<<" , new_HHO = "<<new_HHO(i,counter_bis)<<std::endl;
//                if(i == 0 )
//                std::cout<<"new_HHO(0,counter_bis) = "<<new_HHO(0,counter_bis)<<" , new_HHO(1,counter_old) = "<<new_HHO(1,counter_old)<<std::endl;
//                if(i == 1 )
//                std::cout<<"new_HHO(1,counter_bis) = "<<new_HHO(1,counter_bis)<<" , new_HHO(0,counter_next)) = "<<new_HHO(0,counter_next)<<std::endl;
            }
        }
                                               
    }
    
    
    template<typename Level_Set>
    void
    make_avg_L2_local_proj_para_curvature_disc( const Mesh& msh , Level_Set& ls_cell, size_t di = 1)
    {
        std::cout<<"---> Parametric CURVATURE smooth: CONTINUITY BY AVG of parametric curvature."<<std::endl;
    //        Matrix<T, Dynamic, 1> curvature_FE = Eigen::Matrix<T, Dynamic, 1>::Zero( ndof_dd );
           
               
            // new_HHO contains the discontinuous curvature field
        Matrix<T, Dynamic, Dynamic>  new_HHO = Matrix<T, Dynamic, Dynamic>::Zero( dd_size, counter_subcls );
        Interface_parametrisation_mesh1d curve(basis_degree);
        auto ref_nodes = reference_nodes_ordered_01<T>(dd_degree);
           
        for( const auto& cl : msh.cells )
        {
            if( location(msh, cl) == element_location::ON_INTERFACE )
            {
                auto msh_int = cl.user_data.integration_msh ;
                ls_cell.cell_assignment(cl);
                auto T_cells =  get_global_cells_interface(msh , cl);
                size_t loc_counter = 0 ;
                for(auto& Tj : T_cells )
                {
                    auto pts = points(msh_int,msh_int.cells[loc_counter]);
                    for (size_t i = 0; i < dd_size; i++){
                        auto p = curve(ref_nodes[i].x() , pts ) ;
                        point<T,2> pt = typename Mesh::point_type( p(0) , p(1) ) ;
                        std::vector< size_t> offset_cells ;
                            
                        if( i == 0 || i == 1 )
                            offset_cells = pt_in_skeleton(ls_cell.level_set.msh,pt);
    //                        else
    //                            new_HHO(i,Tj) = ls_cell.divergence(pt) ;
                            
                            
                        if(offset_cells.size() == 0 || offset_cells.size() == 1){
                            new_HHO(i,Tj) = ls_cell.divergence_disc(pt) ;
                        }
                            
                        if(offset_cells.size() == 2)
                        {
                            auto subcl0 = ls_cell.level_set.msh.cells[offset_cells[0]];
                            auto subcl1 = ls_cell.level_set.msh.cells[offset_cells[1]];
                            if(Tj == 0 && loc_counter == 0 )
                            {
                                if( i == 0 ){
                                new_HHO(1,counter_subcls - 1) = ls_cell.divergence_disc( pt , subcl0);
                                new_HHO(0,Tj) = ls_cell.divergence_disc( pt , subcl1 );
                                }
                               
                                    
                            }
                            else if(i == 0 )
                            {
                                new_HHO(1,Tj - 1) = ls_cell.divergence_disc( pt , subcl0 );
                                new_HHO(0,Tj) = ls_cell.divergence_disc( pt , subcl1 );
                                   
                            }
                            
                        }
    //                        new_HHO(i,Tj) = ls_cell.divergence(pt) ;
    //                        std::cout<<"Cell Tj = "<<Tj<<" , node i = "<<i <<" , pt = "<<pt <<" -> val = "<<new_HHO(i,Tj)<<std::endl;
                    }
                    loc_counter++;
                       
                }
            }
        }
                
        size_t counter_old , counter_next ;
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            if(counter_bis == 0){
                counter_old = counter_subcls - 1;
                counter_next = counter_bis + 1;
            }
            else if(counter_bis == counter_subcls - 1){
                counter_old = counter_bis - 1;
                counter_next = 0;
            }
            else{
                counter_old = counter_bis - 1;
                counter_next = counter_bis + 1;
            }
                    
//            std::cout<<"Cell Tj = "<<counter_bis<<" , cell old = "<<counter_old<<std::endl;
            for (size_t i = 0; i < dd_size; i++)
            {
    //                size_t asm_map =  connectivity_matrix_dd[counter_bis][i] ;
                if( i == 0 )
                    curvature_field(0,counter_bis) = ( new_HHO(0,counter_bis) + new_HHO(1,counter_old)) *0.5;
                else if( i == 1 )
                    curvature_field(1,counter_bis) = ( new_HHO(1,counter_bis) + new_HHO(0,counter_next)) *0.5;
                else
                    curvature_field(i,counter_bis) = new_HHO(i,counter_bis) ;
                        
    //                curvature_FE( asm_map ) = curvature_field(i,counter_bis);
                    
//                std::cout<<"Node i = "<<i<<" , curvature_field(i,counter_bis)= "<<curvature_field(i,counter_bis)<<" , new_HHO = "<<new_HHO(i,counter_bis)<<std::endl;
//                if(i == 0 )
//                std::cout<<"new_HHO(0,counter_bis) = "<<new_HHO(0,counter_bis)<<" , new_HHO(1,counter_old) = "<<new_HHO(1,counter_old)<<std::endl;
//                if(i == 1 )
//                std::cout<<"new_HHO(1,counter_bis) = "<<new_HHO(1,counter_bis)<<" , new_HHO(0,counter_next)) = "<<new_HHO(0,counter_next)<<std::endl;
            }
        }
                                               
    }
    
    

    void
    make_smooth_filter_curvature()
    {
        std::cout<<"Filtering of curvature."<<std::endl;
        std::cout<<"---> Parametric CURVATURE FILTER: AVG (3 elements) of parametric curvature values."<<std::endl;
               
        // new_HHO contains the discontinuous curvature field
        Matrix<T, Dynamic, Dynamic>  new_HHO = Matrix<T, Dynamic, Dynamic>::Zero( dd_size, counter_subcls );
        Interface_parametrisation_mesh1d curve(basis_degree);
        auto ref_nodes = reference_nodes_ordered_01<T>(dd_degree);
           
                
        size_t counter_old , counter_next ;
        for(size_t counter_bis = 0 ; counter_bis < counter_subcls ;counter_bis++)
        {
            if(counter_bis == 0){
                counter_old = counter_subcls - 1;
                counter_next = counter_bis + 1;
            }
            else if(counter_bis == counter_subcls - 1){
                counter_old = counter_bis - 1;
                counter_next = 0;
            }
            else{
                counter_old = counter_bis - 1;
                counter_next = counter_bis + 1;
            }
                    
//            std::cout<<"Cell Tj = "<<counter_bis<<" , cell old = "<<counter_old<<std::endl;
            if(dd_size < 2)
                throw std::logic_error("error, order zero for interface too slow, at least order 1.");
            else if(dd_size == 2)
            {
                curvature_field(0,counter_bis) =  0.5*curvature_field(0,counter_bis) + 0.25*curvature_field(1,counter_bis) + 0.25*curvature_field(0,counter_old);
                
                curvature_field(1,counter_old) = curvature_field(0,counter_bis) ;
//                curvature_field(1,counter_bis) =  0.5*curvature_field(1,counter_bis) + 0.25*curvature_field(0,counter_bis) + 0.25*curvature_field(1,counter_next);
                
                
            }
            
            else if(dd_size == 3)
            {
                curvature_field(0,counter_bis) =  0.5*curvature_field(0,counter_bis) + 0.25*curvature_field(2,counter_bis) + 0.25*curvature_field(2,counter_old);
                curvature_field(2,counter_bis) =  0.5*curvature_field(2,counter_bis) + 0.25*curvature_field(0,counter_bis) + 0.25*curvature_field(1,counter_bis);
                curvature_field(1,counter_old) = curvature_field(0,counter_bis) ;
//                curvature_field(1,counter_bis) =  0.5*curvature_field(1,counter_bis) + 0.25*curvature_field(2,counter_bis) + 0.25*curvature_field(2,counter_next);
            }
            else
            {
                curvature_field(0,counter_bis) =  0.5*curvature_field(0,counter_bis) + 0.25*curvature_field(2,counter_bis) + 0.25*curvature_field(2,counter_old);
                curvature_field(2,counter_bis) =  0.5*curvature_field(2,counter_bis) + 0.25*curvature_field(0,counter_bis) + 0.25*curvature_field(3,counter_bis);
                curvature_field(1,counter_old) = curvature_field(0,counter_bis) ;
                
                for (size_t i = 3; i < dd_size - 1 ; i++)
                {
                    curvature_field(i,counter_bis) =  0.5*curvature_field(i,counter_bis) + 0.25*curvature_field(i-1,counter_bis) + 0.25*curvature_field(i+1,counter_bis);
                    
                }
                curvature_field(dd_size - 1,counter_bis) =  0.5*curvature_field(dd_size - 1,counter_bis) + 0.25*curvature_field(dd_size - 2,counter_bis) + 0.25*curvature_field(1,counter_bis);
                
            }

            
        }
                                               
    }
    
    
    Matrix<T, 2, 1>
    operator()( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        size_t basis_size = basis_degree + 1 ;
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_basis_1d(pt) ;
        
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            ret(0) += basis(i)*physical_pts[i].x();
            ret(1) += basis(i)*physical_pts[i].y();
        }
        return ret;
        
        
    }
    
    
    Matrix<T, 2, 1>
    operator()( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_basis_1d(pt) ;
        
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            ret(0) += basis(i)*physical_pts[i].x();
            ret(1) += basis(i)*physical_pts[i].y();
        }
        return ret;
        
        
    }
    
    
   
    Matrix<T, 2, 1>
    derivative( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        size_t basis_size = basis_degree + 1 ;
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_gradients_1d(pt) ;
       
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            ret(0) += basis(i)*physical_pts[i].x();
            ret(1) += basis(i)*physical_pts[i].y();
        }
        return ret;
        
        
    }
    
   
    
    Matrix<T, 2, 1>
    derivative( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_gradients_1d(pt) ;
       
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            ret(0) += basis(i)*physical_pts[i].x();
            ret(1) += basis(i)*physical_pts[i].y();
        }
        return ret;
        
        
    }
    
  
    
   
    T
    jacobian( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {
        return (this->derivative(pt,physical_pts,basis_degree)).norm();
    }
    
   
    T
    jacobian( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {
        return (this->derivative(pt,physical_pts)).norm();
    }
    
    
    
    Matrix<T, 2, 1>
    tangent( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {

        auto der = this->derivative(pt,physical_pts,basis_degree) ;
        return der/der.norm();
       
        
    }
    
   
    Matrix<T, 2, 1>
    tangent( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {

        auto der = this->derivative(pt,physical_pts) ;
        return der/der.norm();
       
        
    }
    
   
    Matrix<T, 2, 1>
    normal( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        auto tan_pt = (this->tangent(pt, physical_pts , basis_degree)) ;
        ret(0) = tan_pt(1);
        ret(1) = -tan_pt(0);
        return ret;
           
    }
    
   
   
    
  
    Matrix<T, 2, 1>
    normal( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        auto tan_pt = (this->tangent(pt, physical_pts)) ;
        ret(0) = tan_pt(1);
        ret(1) = -tan_pt(0);
        return ret;
           
    }
    
   
    T
    curvature( const T& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        Matrix<T, 2, 1> curv_double_der = Matrix<T, 2, 1>::Zero(2, 1);
        
        size_t basis_size = basis_degree + 1 ;
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_double_derivative_1d(pt) ;
    
        auto curv_der = (this->derivative( pt, physical_pts , basis_degree )) ;
        auto curv_der_norm = curv_der.norm() ;
       
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            curv_double_der(0) += basis(i)*physical_pts[i].x();
            curv_double_der(1) += basis(i)*physical_pts[i].y();
        }
    
        T coeff = curv_der(0)*curv_double_der(0) + curv_der(1)*curv_double_der(1) ;
        ret(0) = curv_double_der(0)/curv_der_norm - curv_der(0)/pow(curv_der_norm,3)*coeff;
        ret(1) = curv_double_der(1)/curv_der_norm - curv_der(1)/pow(curv_der_norm,3)*coeff;
        int sign_curv ;
        auto n = this->normal( pt, physical_pts , basis_degree );
        
        if( (sgn( n(0) ) == sgn( ret(0) ) ) && (sgn( n(1) ) == sgn( ret(1) ) ) )
            sign_curv = 1.0 ;
        else
            sign_curv =  -1.0 ;
       
        return sign_curv * ret.norm()/curv_der_norm ;
           
    }
    
    
    T
    curvature( const T& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        Matrix<T, 2, 1> curv_double_der = Matrix<T, 2, 1>::Zero(2, 1);
        
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis = cb.eval_double_derivative_1d(pt) ;
    
        auto curv_der = (this->derivative( pt, physical_pts , basis_degree )) ;
        auto curv_der_norm = curv_der.norm() ;
       
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            curv_double_der(0) += basis(i)*physical_pts[i].x();
            curv_double_der(1) += basis(i)*physical_pts[i].y();
        }
    
        T coeff = curv_der(0)*curv_double_der(0) + curv_der(1)*curv_double_der(1) ;
        ret(0) = curv_double_der(0)/curv_der_norm - curv_der(0)/pow(curv_der_norm,3)*coeff;
        ret(1) = curv_double_der(1)/curv_der_norm - curv_der(1)/pow(curv_der_norm,3)*coeff;
        return ret.norm()/curv_der_norm ;
           
    }
    
    Matrix<T, 2, 1>
    operator()( const T& pt , const size_t& i_cl ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        
        auto basis_eval = cb.eval_basis_1d(pt) ;
        auto values_cell0 = interface0.col(i_cl);
        auto values_cell1 = interface1.col(i_cl);
        
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        
        return ret;
        
        
    }
    
    Matrix<T, 2, 1>
    derivative( const T& pt , const size_t& i_cl ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        
        auto basis_eval = cb.eval_gradients_1d(pt) ;
        auto values_cell0 = interface0.col(i_cl);
        auto values_cell1 = interface1.col(i_cl);
        
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        
        return ret;
        
        
    }
    
    
    Matrix<T, 2, 1>
    derivative_cont( const T& pt , size_t i_cl ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrange_1d_reference_new <T> cb(der_degree);
        auto basis_eval = cb.eval_basis_1d(pt) ;
           
        auto values_cell0 = derivative0.col(i_cl);
        auto values_cell1 = derivative1.col(i_cl);
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
           
           
        return ret;
           
           
    }
    
    
    
    
//    T
//    jacobian_cont( const T& pt , size_t i_cl ) const
//    {
//        return (this->derivative_cont(pt,i_cl)).norm();
//    }
    
    // THIS CASE IS DISCONTINUOS, BUT I USE IT HERE, FINALLY FIX ALL THESE STUFF
    T
    jacobian_cont( const T& pt , size_t i_cl ) const
    {
        return (this->derivative(pt,i_cl)).norm();
    }
    
    
    Matrix<T, 2, 1>
    tangent_cont( const T& pt , size_t i_cl ) const
    {

        auto der = this->derivative_cont(pt,i_cl) ;
        return der/der.norm();
       
        
    }
    
    
    
    Matrix<T, 2, 1>
    normal_der_cont( const T& pt , size_t i_cl ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        auto tan_pt = (this->tangent_cont(pt, i_cl)) ;
        ret(0) = tan_pt(1);
        ret(1) = -tan_pt(0);
        return ret;
           
    }
    
    Matrix<T, 2, 1>
    normal_cont( const T& pt , size_t i_cl ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrange_1d_reference_new <T> cb(der_degree);
        auto basis_eval = cb.eval_basis_1d(pt) ;
           
        auto values_cell0 = normal0.col(i_cl);
        auto values_cell1 = normal1.col(i_cl);
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
           
           
        return ret;
           
    }
    

    
    
    T
    curvature_der_cont( const T& pt , size_t i_cl ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        Matrix<T, 2, 1> curv_double_der = Matrix<T, 2, 1>::Zero(2, 1);
        
        cell_basis_Lagrange_1d_reference_new <T> cb(der_degree);
        auto basis_eval = cb.eval_gradients_1d(pt) ;
    
        auto curv_der = (this->derivative_cont( pt, i_cl ) ) ;
        auto curv_der_norm = curv_der.norm() ;
        
        auto values_cell0 = derivative0.col(i_cl);
        auto values_cell1 = derivative1.col(i_cl);
        
        curv_double_der(0) = values_cell0.dot( basis_eval );
        curv_double_der(1) = values_cell1.dot( basis_eval );
       
        T coeff = curv_der(0)*curv_double_der(0) + curv_der(1)*curv_double_der(1) ;
       
        ret(0) = curv_double_der(0)/curv_der_norm - curv_der(0)/pow(curv_der_norm,3)*coeff;
        ret(1) = curv_double_der(1)/curv_der_norm - curv_der(1)/pow(curv_der_norm,3)*coeff;
        int sign_curv ;
        auto n = this->normal_cont( pt, i_cl );
        
        if( (sgn( n(0) ) == sgn( ret(0) ) ) && (sgn( n(1) ) == sgn( ret(1) ) ) )
            sign_curv = 1.0 ;
        else
            sign_curv =  -1.0 ;
       
        return sign_curv * ret.norm()/curv_der_norm ;
           
    }
    
    T
    curvature_cont( const T& pt , size_t i_cl ) const
    {
        cell_basis_Lagrange_1d_reference_new <T> cb(dd_degree);
        return (curvature_field.col(i_cl) ).dot( cb.eval_basis_1d(pt) ) ;
    }
    
    
};


template <typename Mesh, typename T >
std::vector< std::pair< size_t , size_t > >
mapping_S( const Mesh& msh_next, const Mesh& msh_last , const typename Mesh::cell_type&  cl , const mesh_init_params<T>& mip )
{
   // cl is in the agglo_mesh at time t+1 -> msh_next
    //auto i = offset(msh,cl) ;
    //auto S_i = cl.user_data.offset_subcells ;
    
    size_t Nx =  mip.Nx ;
    size_t Ny = mip.Ny ;
    
    
    std::vector< std::pair< size_t , size_t > > ret ;
    size_t i_cl_next = offset(msh_next , cl );
    // For each cell of the agglo mesh t^{N+1}, loop the original subcells K_i of mesh t^0
    for(auto& i: cl.user_data.offset_subcells)
    {
        
        //std::cout<<"i = "<<i<<std::endl;
        if( (cl.user_data.offset_subcells[0] == cl.user_data.offset_subcells[1] ) )
        {
            size_t posp = i_cl_next+1 ;
            size_t pospp = i_cl_next+Nx-1 ;
            int posn = i_cl_next ;
            int posnn = i_cl_next-Nx+1 ;
            int limit_inf = i_cl_next-Nx+1 ;
            int limit_sup = i_cl_next+Nx-1 ;
            //for( auto& cl_last : msh_last.cells )
            while( posn>limit_inf || pospp<msh_last.cells.size() || posnn>=0 || posp<limit_sup )
            {
                if(ret.size()==1)
                    break;
                
                //std::cout<<"posn = "<<posn<<" and i - Nx + 1 = "<<i - Nx + 1<<std::endl;
                if( posn > limit_inf )
                {
                    auto cl_lastn = msh_last.cells[posn];
                    for(auto& i_last : cl_lastn.user_data.offset_subcells )
                    {
                        //std::cout<<"i_last_n = "<<i_last<<std::endl;
                        if( i_last == i ){
                            ret.push_back( std::make_pair(offset(msh_last, cl_lastn),i) ) ;
                            break;
                        }
                    }
                }
                if(ret.size()==1)
                    break;
                
                if( posnn >= 0 )
                {
                    auto cl_lastn = msh_last.cells[posnn];
                    for(auto& i_last : cl_lastn.user_data.offset_subcells )
                    {
                        //std::cout<<"i_last_nn = "<<i_last<<std::endl;
                        if( i_last == i ){
                            ret.push_back( std::make_pair(offset(msh_last, cl_lastn),i) ) ;
                            break;
                        }
                    }
                }
                if(ret.size()==1)
                    break;

               
                if( posp < limit_sup )
                {
                    auto cl_lastp = msh_last.cells[posp];
                    for(auto& i_last : cl_lastp.user_data.offset_subcells )
                    {
                        //std::cout<<"i_last_p = "<<i_last<<std::endl;
                        if( i_last == i ){
                            ret.push_back( std::make_pair(offset(msh_last, cl_lastp) , i) ) ;
                            break;
                        }
                    }
                }
                if(ret.size()==1)
                    break;
                
                if( pospp < msh_last.cells.size() )
                {
                    auto cl_lastp = msh_last.cells[pospp];
                    for(auto& i_last : cl_lastp.user_data.offset_subcells )
                    {
                        //std::cout<<"i_last_pp = "<<i_last<<std::endl;
                        if( i_last == i ){
                            ret.push_back( std::make_pair(offset(msh_last, cl_lastp) , i) ) ;
                            break;
                        }
                    }
                }
                if(ret.size()==1)
                    break;
                
                
             
                posp ++;
                posn --;
                pospp ++;
                posnn --;
            }
            break ;
            
        }
        else
        {
            //for( auto& cl_last : msh_last.cells )
            size_t posp = i_cl_next+1 ;
            size_t pospp = i_cl_next+Nx-1 ;
            int posn = i_cl_next ;
            int posnn = i_cl_next-Nx+1 ;
            int limit_inf = i_cl_next-Nx+1 ;
            int limit_sup = i_cl_next+Nx-1 ;
            
            
            while( posn>limit_inf || pospp<msh_last.cells.size() || posnn>=0 || posp<limit_sup )
            {
                bool cell_n_found = FALSE ;
                if( ret.size () == cl.user_data.offset_subcells.size() ) // || ret ==  cl.user_data.offset_subcells )
                    break;
                
                
                if(posn > limit_inf )
                {
                    auto cl_lastn = msh_last.cells[posn];
                    for(auto& i_last : cl_lastn.user_data.offset_subcells )
                    {
                        //std::cout<<"AGGLO: i_last_n = "<<i_last<<std::endl;
                        if( i_last == i )
                        {
                            ret.push_back( std::make_pair(offset(msh_last, cl_lastn),i) ) ;
                            cell_n_found = TRUE ;
                            break;
                        }
                    }
                }
                if( cell_n_found == TRUE || ret.size () == cl.user_data.offset_subcells.size() )
                    break;
                

                if(posnn >= 0 )
                {
                    auto cl_lastn = msh_last.cells[posnn];
                    for(auto& i_last : cl_lastn.user_data.offset_subcells )
                    {
                        //std::cout<<"AGGLO: i_last_nn = "<<i_last<<std::endl;
                        if( i_last == i )
                        {
                            ret.push_back( std::make_pair(offset(msh_last, cl_lastn),i) ) ;
                            cell_n_found = TRUE ;
                            break;
                        }
                    }
                }
                if( cell_n_found == TRUE || ret.size () == cl.user_data.offset_subcells.size() )
                    break;

                
                if( posp < limit_sup )
                {
                    auto cl_lastp = msh_last.cells[posp];
                    for(auto& i_last : cl_lastp.user_data.offset_subcells )
                    {
                        //std::cout<<"AGGLO: i_last_p = "<<i_last<<std::endl;
                        if( i_last == i )
                        {
                            ret.push_back( std::make_pair(offset(msh_last, cl_lastp),i) ) ;
                            cell_n_found = TRUE ;
                            break;
                        }
                    }
                }
                if( cell_n_found == TRUE || ret.size () == cl.user_data.offset_subcells.size() )
                    break;
                
                if(pospp< msh_last.cells.size())
                {
                    auto cl_lastp = msh_last.cells[pospp];
                    for(auto& i_last : cl_lastp.user_data.offset_subcells )
                    {
                        //std::cout<<"AGGLO: i_last_pp = "<<i_last<<std::endl;
                        if( i_last == i )
                        {
                            ret.push_back( std::make_pair(offset(msh_last, cl_lastp),i) ) ;
                            cell_n_found = TRUE ;
                            break;
                        }
                    }
                }
                if( cell_n_found == TRUE || ret.size () == cl.user_data.offset_subcells.size() )
                    break;
                
                
                posp ++;
                posn --;
                pospp ++;
                posnn --;
            }
            
            
        }
            
            
        
    }
    return ret ;
}




template<typename Mesh, typename T = typename Mesh::coordinate_type >
struct Parametric_Interface
{
    
    typedef typename Mesh::point_type       point_type;
//    Mesh original_mesh , agglomerated_mesh ;
    size_t degree_det ;
    size_t basis_degree , basis_size;
    
    T radius_a , radius_b ;
    size_t N_cells ; // amount of cells
    size_t N_pts ; // amount of cells
    size_t N_interface ; // amount of cells
//    std::vector<std::set<size_t>> connectivity_HHO_cells_this ;
    std::vector<std::map< size_t , std::pair<T,T> > > connectivity_HHO_cells_this ;
    std::vector<std::vector<size_t>> connectivity_this_HHO_cells ;
    
    
    std::vector<std::map< size_t , std::pair<T,T> > > connectivity_agglo_HHO_cells_this ;
    std::vector<std::vector<size_t>> connectivity_this_agglo_HHO_cells ;
    
    Eigen::Matrix<T, Dynamic, Dynamic> interface0 ;
    Eigen::Matrix<T, Dynamic, Dynamic> interface1 ;
    
    // ---- FINITE SPACE FOR THE INTERFACE
//    std::vector< std::vector<size_t> > connectivity_cells ;
//    std::vector< std::vector<size_t> > connectivity_matrix ;
//    size_t ndof ;
//    size_t counter_subcls = 0;
    
    
    // ---- FINITE SPACE FOR THE INTERFACE DERIVATIVE
//    std::vector< std::vector<size_t> > connectivity_matrix_der ;
//    size_t ndof_der ;
    size_t der_degree ;
    size_t der_size ;
    
    // ---- FINITE SPACE FOR THE INTERFACE CURVATURE
//    std::vector< std::vector<size_t> > connectivity_matrix_dd ;
//    size_t ndof_dd ;
    size_t dd_degree ;
    size_t dd_size ;
    
    
    
    

//    Eigen::Matrix<T, Dynamic, Dynamic> derivative0 ;
//    Eigen::Matrix<T, Dynamic, Dynamic> derivative1 ;
//
//    Eigen::Matrix<T, Dynamic, Dynamic> normal0 ;
//    Eigen::Matrix<T, Dynamic, Dynamic> normal1 ;
//
//    Eigen::Matrix<T, Dynamic, Dynamic> curvature_field ;
    
   
    
    Parametric_Interface();

    
    Parametric_Interface(const Parametric_Interface& other)
    {
        degree_det = other.degree_det;
//        connectivity_cells = other.connectivity_cells ;
//        connectivity_matrix = other.connectivity_matrix;
//        ndof = other.ndof;
//        counter_subcls = other.counter_subcls;
        basis_degree = other.basis_degree;
        basis_size = other.basis_size;
//        connectivity_matrix_der = other.connectivity_matrix_der;
//        ndof_der = other.ndof_der;
//
        der_degree = other.der_degree;
        der_size = other.der_size;
//
//        connectivity_matrix_dd = other.connectivity_matrix_dd;
//
//        ndof_dd = other.ndof_dd;
//
        dd_degree = other.dd_degree;
        dd_size = other.dd_size;
            
            
        interface0 = other.interface0;
        interface1 = other.interface1;
//        original_mesh = other.original_mesh ;

//        derivative0 = other.derivative0;
//        derivative1 = other.derivative1;
//
//        normal0 = other.normal0;
//        normal1 = other.normal1;
//        curvature_field = other.curvature_field;
        
        radius_a = other.radius_a ;
        radius_b = other.radius_b ;
        N_cells = other.N_cells ;
        N_pts = other.N_pts ;
        N_interface = other.N_interface ;
        connectivity_HHO_cells_this = other.connectivity_HHO_cells_this ;
        connectivity_this_HHO_cells = other.connectivity_this_HHO_cells ;
        
        connectivity_agglo_HHO_cells_this = other.connectivity_agglo_HHO_cells_this ;
        connectivity_this_agglo_HHO_cells = other.connectivity_this_agglo_HHO_cells ;
        
        
       
    }
    
    
    
    
    Parametric_Interface(const Mesh& msh , T x_c , T y_c , T Ra, T Rb , size_t degree_curve , size_t N ):  basis_degree(degree_curve) , basis_size(degree_curve+1),der_degree(degree_curve-1) , der_size(degree_curve),dd_degree(degree_curve-1) , dd_size(degree_curve), radius_a(Ra), radius_b (Rb), degree_det(jacobian_interface(degree_curve)), N_cells(N),N_pts(basis_degree*N), connectivity_this_HHO_cells(N),  connectivity_HHO_cells_this(msh.cells.size())
    {
        interface0 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( basis_size, N);
        interface1 = Eigen::Matrix<T, Dynamic, Dynamic>::Zero( basis_size, N);
        
        T step = 2.0*M_PI/N ;
        T sub_step = step/basis_degree;
        size_t counter_cell = 0;
        for( T theta = 0.0 ; theta < 2.0*M_PI ; theta += step)
        {
            auto pt0 = x_c + radius_a * cos(theta) ; // point_type(,  );
            auto pt1 = y_c + radius_b * sin(theta) ;
            interface0(0,counter_cell) = pt0;
            interface1(0,counter_cell) = pt1;
            pt0 = x_c + radius_a * cos(theta + step) ; // point_type(,  );
            pt1 = y_c + radius_b * sin(theta + step) ;
            
            interface0(1,counter_cell) = pt0;
            interface1(1,counter_cell) = pt1;
            
            
            for(size_t i = 2 ; i < basis_size ; i++){
                pt0 = x_c + radius_a * cos(theta + sub_step*(i-1)) ; // point_type(,  );
                pt1 = y_c + radius_b * sin(theta + sub_step*(i-1)) ;
                
                interface0(i,counter_cell) = pt0;
                interface1(i,counter_cell) = pt1;
                
            }
                
            counter_cell++;
           
            
        }
        
    }
    
    void
    re_initialisation(const Mesh& msh)
    {
        connectivity_HHO_cells_this.clear() ;
        connectivity_this_HHO_cells.clear() ;
        connectivity_this_HHO_cells.resize(N_cells);
        connectivity_HHO_cells_this.resize(msh.cells.size());
    }
    
    
    void
    interface_mesh_agglomeration_update(const Mesh& msh_new ,const Mesh& msh_past ,const mesh_init_params<T>& mip)
    {
        connectivity_agglo_HHO_cells_this.resize(msh_new.cells.size())  ;
        connectivity_this_agglo_HHO_cells.resize(N_cells)  ;
        
        for(auto& agglo_cl : msh_new.cells)
        {
            if(agglo_cl.user_data.location == element_location::ON_INTERFACE)
            {
//                auto cl_n_offsets = mapping_S( msh_new, msh_past , agglo_cl , mip );
                size_t agglo_offset = offset(msh_new,agglo_cl) ;
                
                
                auto subcells = agglo_cl.user_data.offset_subcells ;
                std::cout<<"For agglo cell: "<<agglo_offset<<std::endl;
                for(auto& i : subcells)
                    std::cout<<i<<" -- ";
                std::cout<<std::endl;
            
                
                for(auto& i_cl_orig : subcells)
                {
                    auto cl_i = connectivity_HHO_cells_this[i_cl_orig];
                    for(auto& loc : cl_i)
                    {
                            
                        auto cell_para_i = loc.first ;
                        auto cell_para_limits = loc.second ;
                        connectivity_agglo_HHO_cells_this[agglo_offset].insert( {cell_para_i,std::make_pair(1.0,0.0)} ) ;
                        if(cell_para_limits.first < connectivity_agglo_HHO_cells_this[agglo_offset][cell_para_i].first)
                            connectivity_agglo_HHO_cells_this[agglo_offset][cell_para_i].first = cell_para_limits.first;
                        if(cell_para_limits.second > connectivity_agglo_HHO_cells_this[agglo_offset][cell_para_i].second)
                            connectivity_agglo_HHO_cells_this[agglo_offset][cell_para_i].second = cell_para_limits.second;
                    
                    
//                        connectivity_this_agglo_HHO_cells[cell_para_i].push_back(agglo_offset) ;
                  
                    } // loop on parametric functions into cell i_cl_orig
                } // loop on sub_cells of agglomerated function cl
          
                
            } // if agglomerated
            
            
        }
        
//        for(size_t cell_i = 0 ; cell_i < N_cells ; cell_i++)
//        {
//            std::vector<size_t>::iterator it = std::unique (connectivity_this_agglo_HHO_cells[cell_i].begin(), connectivity_this_agglo_HHO_cells[cell_i].end());
//
//            connectivity_this_agglo_HHO_cells[cell_i].resize( std::distance(connectivity_this_agglo_HHO_cells[cell_i].begin(),it) );
//
//        }
        
        
        
        
        
        
//        std::cout<<'\n'<<"Case ORIGINAL"<<std::endl;
//
//            size_t counter = 0;
//            for(auto& elem : connectivity_this_HHO_cells)
//            {
//
//                std::cout<<'\n'<<"In cl PARA num "<<counter<<" , HHO cells: "<<'\n';
//                for(auto& sub_elem : elem)
//                    std::cout<<sub_elem<<" , ";
//                counter++;
//            }
//            std::cout<<std::endl;
//
//        std::cout<<'\n'<<"Case AGGLOMERATED NOT ORDERED"<<std::endl;
//            counter = 0;
//            for(auto& elem : connectivity_this_agglo_HHO_cells)
//            {
//
//                std::cout<<'\n'<<"In cl PARA num "<<counter<<" , HHO cells: "<<'\n';
//                for(auto& sub_elem : elem)
//                    std::cout<<sub_elem<<" , ";
//                counter++;
//            }
//            std::cout<<std::endl;
        
//        std::cout<<'\n'<<"Case ORIGINAL"<<std::endl;
//
//            counter = 0;
//            for(auto& elem : connectivity_HHO_cells_this)
//            {
//                if(msh_past.cells[counter].user_data.location == element_location::ON_INTERFACE)
//                {
//                std::cout<<'\n'<<"In cl HHO num "<<counter<<" , PARA cells: "<<'\n';
//                for(auto& sub_elem : elem)
//                    std::cout<<sub_elem.first<<" , ( "<<sub_elem.second.first<<" , "<<sub_elem.second.second<<" ) "<<'\n';
//                }
//                counter++;
//            }
//            std::cout<<std::endl;
//        std::cout<<'\n'<<"Case AGGLOMERATED"<<std::endl;
//
//        counter = 0;
//        for(auto& elem : connectivity_agglo_HHO_cells_this)
//        {
//            if(msh_new.cells[counter].user_data.location == element_location::ON_INTERFACE)
//            {
//            std::cout<<'\n'<<"In cl HHO num "<<counter<<" , PARA cells: "<<'\n';
//            for(auto& sub_elem : elem)
//                std::cout<<sub_elem.first<<" , ( "<<sub_elem.second.first<<" , "<<sub_elem.second.second<<" ) "<<'\n';
//            }
//            counter++;
//        }
//        std::cout<<std::endl;

        
        
        
//        std::vector<std::vector<size_t>> tmp_this_HHO_cells(N_cells) ;

        typedef typename Mesh::point_type point_type;
        size_t old_i = 0;
        bool first_cut_cell = FALSE ;
        size_t Nx =  mip.Nx ;
        size_t Ny = mip.Ny ;
        T eps = 1e-5;
        T  h = std::min(mip.hx() , mip.hy() ) ;


        size_t first_cell , last_cell ;
        for(size_t cell_i = 0 ; cell_i < this->N_cells ; cell_i++)
        {

            auto pt0 = this->operator()(0,cell_i);
            auto pt1 = this->operator()(1,cell_i);
            T dist = (pt0 - pt1).norm();
            T N_pts = 10.0*ceil(dist/h) + 1.0 ;
            T N_half = (N_pts-1.0)/2.0;

            for(T pt_i = 1.0; pt_i < N_pts ; pt_i++)
            {
                T pos = pt_i / N_pts;


                auto pt_vec = this->operator()(pos,cell_i) ;
                auto pt = point_type(pt_vec(0) , pt_vec(1));

                if(!first_cut_cell)
                {
                    for(auto& agglo_cl :msh_new.cells)
                    {
                        auto offset_vec = agglo_cl.user_data.offset_subcells;
                        auto offset_agglo = offset(msh_new,agglo_cl);
                        for(auto& offset : offset_vec)
                        {
                            auto cl = msh_past.cells[offset];
                            if( pt_in_cell(msh_past, pt, cl) )
                            {

                                first_cell = offset_agglo;
                                connectivity_this_agglo_HHO_cells[cell_i].push_back(offset_agglo) ;
                                first_cut_cell = TRUE ;
                                break;
                            }
                            if(first_cut_cell)
                                break;
                        }



                    }

                }
                else
                {

                    bool found_cut_cell = FALSE ;
                    for(auto& agglo_cl :msh_new.cells)
                    {
                        auto offset_vec = agglo_cl.user_data.offset_subcells;
                        auto offset_agglo = offset(msh_new,agglo_cl);
                        for(auto& offset : offset_vec)
                        {
                            auto cl = msh_past.cells[offset];
                            if( pt_in_cell(msh_past, pt, cl) )
                            {

                                first_cell = offset_agglo;
                                connectivity_this_agglo_HHO_cells[cell_i].push_back(offset_agglo) ;
                                found_cut_cell = TRUE;
                                break;
                            }
                            if(found_cut_cell)
                                break;
                        }



                    }

                }


            }


            std::vector<size_t>::iterator it = std::unique (connectivity_this_agglo_HHO_cells[cell_i].begin(), connectivity_this_agglo_HHO_cells[cell_i].end());

            connectivity_this_agglo_HHO_cells[cell_i].resize( std::distance(connectivity_this_agglo_HHO_cells[cell_i].begin(),it) );
        }

        
        
//        std::cout<<'\n'<<"Case AGGLOMERATED BIS "<<std::endl;
//        counter = 0;
//        for(auto& elem : connectivity_this_agglo_HHO_cells)
//        {
//
//            std::cout<<'\n'<<"In cl PARA num "<<counter<<" , HHO cells: "<<'\n';
//            for(auto& sub_elem : elem)
//                std::cout<<sub_elem<<" , ";
//            counter++;
//        }
//        std::cout<<std::endl;
        
       
       



    }
  
    std::map< size_t , std::pair<T,T> >
    get_global_interface(bool agglo ,const Mesh& msh, const typename Mesh::cell_type& cl) const
    {
        size_t k_offset = offset(msh,cl) ;
        if(agglo)
            return connectivity_agglo_HHO_cells_this[k_offset] ;
        
        else
            return connectivity_HHO_cells_this[k_offset] ;
        
        
    }
    
    Matrix<T, 2, 1>
    operator()( const T& pt , const size_t& i_cl ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        
        auto basis_eval = cb.eval_basis_1d(pt) ;
        auto values_cell0 = interface0.col(i_cl);
        auto values_cell1 = interface1.col(i_cl);
        
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        
        return ret;
        
        
    }
    
  
    
    
    
    
    
    Matrix<T, 2, 1>
    derivative( const T& pt , const size_t& i_cl ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        
        auto basis_eval = cb.eval_gradients_1d(pt) ;
        auto values_cell0 = interface0.col(i_cl);
        auto values_cell1 = interface1.col(i_cl);
        
        ret(0) = values_cell0.dot( basis_eval );
        ret(1) = values_cell1.dot( basis_eval );
        
        return ret;
        
        
    }
    
    
    T
    jacobian( const T& pt , size_t i_cl ) const
    {
        return (this->derivative(pt,i_cl)).norm();
    }
    
    
    Matrix<T, 2, 1>
    tangent( const T& pt , size_t i_cl ) const
    {

        auto der = this->derivative(pt,i_cl) ;
        return der/der.norm();
       
        
    }
    
    
    
    Matrix<T, 2, 1>
    normal( const T& pt , size_t i_cl ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        auto tan_pt = (this->tangent(pt, i_cl)) ;
        ret(0) = tan_pt(1);
        ret(1) = -tan_pt(0);
        return ret;
           
    }
    
    
    T
    curvature( const T& pt , size_t i_cl ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        Matrix<T, 2, 1> curv_double_der = Matrix<T, 2, 1>::Zero(2, 1);
        
        cell_basis_Lagrange_1d_reference_new <T> cb(basis_degree);
        auto basis_eval = cb.eval_double_derivative_1d(pt) ;
    
        auto curv_der = (this->derivative( pt, i_cl ) ) ;
        auto curv_der_norm = curv_der.norm() ;
        
        auto values_cell0 = interface0.col(i_cl);
        auto values_cell1 = interface1.col(i_cl);
        
        curv_double_der(0) = values_cell0.dot( basis_eval );
        curv_double_der(1) = values_cell1.dot( basis_eval );
       
        T coeff = curv_der(0)*curv_double_der(0) + curv_der(1)*curv_double_der(1) ;
       
        ret(0) = curv_double_der(0)/curv_der_norm - curv_der(0)/pow(curv_der_norm,3)*coeff;
        ret(1) = curv_double_der(1)/curv_der_norm - curv_der(1)/pow(curv_der_norm,3)*coeff;
        int sign_curv ;
        auto n = this->normal( pt, i_cl );
        
        if( (sgn( n(0) ) == sgn( ret(0) ) ) && (sgn( n(1) ) == sgn( ret(1) ) ) )
            sign_curv = 1.0 ;
        else
            sign_curv =  -1.0 ;
       
        return sign_curv * ret.norm()/curv_der_norm ;
           
    }
    

    
    

    
    
};

template<typename Mesh , typename Curve, typename T >
void
set_cut_cellOLD(Mesh& msh, const mesh_init_params<T>& mip, Curve& parametric_interface)
{
    typedef typename Mesh::point_type point_type;
    size_t old_i = 0;
    bool first_cut_cell = FALSE ;
    size_t Nx =  mip.Nx ;
    size_t Ny = mip.Ny ;
    T eps = 1e-5;
    T  h = std::min(mip.hx() , mip.hy() ) ;
    
    
    size_t first_cell , last_cell ;
    for(size_t cell_i = 0 ; cell_i < parametric_interface.N_cells ; cell_i++)
    {
        
        auto pt0 = parametric_interface(0,cell_i);
        auto pt1 = parametric_interface(1,cell_i);
        T dist = (pt0 - pt1).norm();
        T N_pts = 10.0*ceil(dist/h) + 1.0 ;
        T N_half = (N_pts-1.0)/2.0;
        
        for(T pt_i = 1.0; pt_i < N_pts ; pt_i++)
        {
            T pos = pt_i / N_pts;
//            if( pt_i == 0.0 )
//                pos = pt_i / N_pts + eps;
//            if( pt_i == N_pts )
//                pos = pt_i / N_pts - eps;
            
            
            auto pt_vec = parametric_interface(pos,cell_i) ;
            auto pt = point_type(pt_vec(0) , pt_vec(1));
            
            
            
            if(!first_cut_cell)
            {
                for(auto& cl :msh.cells)
                {
                    if( pt_in_cell(msh, pt, cl) )
                    {
                        cl.user_data.location = element_location::ON_INTERFACE;
                        old_i = offset(msh,cl);
                        first_cell = old_i;
//                        parametric_interface.connectivity_HHO_cells_this[old_i].push_back(cell_i) ;
                        parametric_interface.connectivity_this_HHO_cells[cell_i].push_back(old_i) ;
//                        if(pt_i == 0)
//                        {
//                            cl.user_data.interface.push_back(pt);
//
//                        }
//
//                        if(pt_i == N_half)
//                        {
//                            cl.user_data.interface.push_back(pt);
//                        }
//
//                        if(pt_i == N_pts - 1)
//                        {
//                            cl.user_data.interface.push_back(pt);
//                        }
                        
                        first_cut_cell = TRUE ;
                        break;
                    }
                    
                    if(first_cut_cell)
                        break;
                    
                }
                
            }
            else
            {
//                if(pt_in_cell(msh, pt, msh.cells[old_i]))
//                    continue;
                bool found_cut_cell = FALSE ;
                for(int i = -1 ; i < 2 ; i++)
                {
                    for(int j = -1 ; j < 2 ; j++)
                    {
                        size_t cl_HHO_i = old_i + i + j * Ny ; // Notice it cannot be on the cell on the boundary
                        auto cl = msh.cells[cl_HHO_i];
                        
                        if(parametric_interface.connectivity_this_HHO_cells[cell_i].size() == 0){
                            size_t last_elem = parametric_interface.connectivity_this_HHO_cells[cell_i-1].size() -1;
                            parametric_interface.connectivity_this_HHO_cells[cell_i].push_back(parametric_interface.connectivity_this_HHO_cells[cell_i-1][last_elem]) ;
//                            size_t last_elem_HHO = parametric_interface.connectivity_HHO_cells_this[old_i].size() -1;
//                            parametric_interface.connectivity_HHO_cells_this[cl_HHO_i].push_back(parametric_interface.connectivity_HHO_cells_this[old_i][last_elem_HHO]) ;
                            
                        }
                            
                        if( cl.user_data.location != element_location::ON_INTERFACE && pt_in_cell(msh, pt, cl) )
                        {
                            msh.cells[cl_HHO_i].user_data.location = element_location::ON_INTERFACE;
                            old_i = offset(msh,cl);
                            
                            
                            
//                            parametric_interface.connectivity_HHO_cells_this[old_i].push_back(cell_i) ;
                            parametric_interface.connectivity_this_HHO_cells[cell_i].push_back(old_i) ;
                            
                            
//                            if(pt_i == 0)
//                            {
//                                msh.cells[cl_HHO_i].user_data.interface.push_back(pt);
//
//                            }
//
//                            if(pt_i == N_half)
//                            {
//                                msh.cells[cl_HHO_i].user_data.interface.push_back(pt);
//                            }
//
//                            if(pt_i == N_pts - 1)
//                            {
//                                msh.cells[cl_HHO_i].user_data.interface.push_back(pt);
//                            }
                            found_cut_cell = TRUE;
                            
                        }
                        else if(cl_HHO_i == first_cell && pt_in_cell(msh, pt, cl))
                        {
                            size_t last_elem = parametric_interface.connectivity_this_HHO_cells[cell_i].size();
                            if(last_elem>0 && parametric_interface.connectivity_this_HHO_cells[cell_i][last_elem-1] != first_cell){
//                                parametric_interface.connectivity_HHO_cells_this[first_cell].push_back(cell_i) ;
                                parametric_interface.connectivity_this_HHO_cells[cell_i].push_back(first_cell) ;
                                
                                
                            }
                            if(last_elem == 0)
                            {
//                                parametric_interface.connectivity_HHO_cells_this[first_cell].push_back(cell_i) ;
                                parametric_interface.connectivity_this_HHO_cells[cell_i].push_back(first_cell) ;
                            }
                        }
                                
                        if( found_cut_cell )
                            break;
                    }
                    if( found_cut_cell )
                        break;
                    
                }
                
            }
            
            
        }
    }
}




template<typename Mesh , typename Curve, typename T >
void
set_cut_cell(Mesh& msh, const mesh_init_params<T>& mip, Curve& parametric_interface)
{
    typedef typename Mesh::point_type point_type;
    size_t old_i = 0;
    bool first_cut_cell = FALSE ;
    size_t Nx =  mip.Nx ;
    size_t Ny = mip.Ny ;
    T eps = 1e-5;
    T  h = std::min(mip.hx() , mip.hy() ) ;
    
    
    size_t first_cell , last_cell ;
    for(size_t cell_i = 0 ; cell_i < parametric_interface.N_cells ; cell_i++)
    {
        
        auto pt0 = parametric_interface(0,cell_i);
        auto pt1 = parametric_interface(1,cell_i);
        T dist = (pt0 - pt1).norm();
        T N_pts = 10.0*ceil(dist/h) + 1.0 ;
        T N_half = (N_pts-1.0)/2.0;
        
        for(T pt_i = 1.0; pt_i < N_pts ; pt_i++)
        {
            T pos = pt_i / N_pts;
//            if( pt_i == 0.0 )
//                pos = pt_i / N_pts + eps;
//            if( pt_i == N_pts )
//                pos = pt_i / N_pts - eps;
            
            
            auto pt_vec = parametric_interface(pos,cell_i) ;
            auto pt = point_type(pt_vec(0) , pt_vec(1));
            
            
            
            if(!first_cut_cell)
            {
                for(auto& cl :msh.cells)
                {
                    if( pt_in_cell(msh, pt, cl) )
                    {
                        cl.user_data.location = element_location::ON_INTERFACE;
                        old_i = offset(msh,cl);
                        first_cell = old_i;
                        parametric_interface.connectivity_this_HHO_cells[cell_i].push_back(old_i) ;
                        first_cut_cell = TRUE ;
                        break;
                    }
                    
                    if(first_cut_cell)
                        break;
                    
                }
                
            }
            else
            {

                bool found_cut_cell = FALSE ;
                for(int i = -1 ; i < 2 ; i++)
                {
                    for(int j = -1 ; j < 2 ; j++)
                    {
                        size_t cl_HHO_i = old_i + i + j * Ny ; // Notice it cannot be on the cell on the boundary
                        auto cl = msh.cells[cl_HHO_i];
                        

                        if( pt_in_cell(msh, pt, cl) )
                        {
                            msh.cells[cl_HHO_i].user_data.location = element_location::ON_INTERFACE;
                            old_i = offset(msh,cl);
                            
                            parametric_interface.connectivity_this_HHO_cells[cell_i].push_back(old_i) ;
                          
                            found_cut_cell = TRUE;
                            
                        }

                                
                        if( found_cut_cell )
                            break;
                    }
                    if( found_cut_cell )
                        break;
                    
                }
                
            }
            
            
        }
        

        std::vector<size_t>::iterator it = std::unique (parametric_interface.connectivity_this_HHO_cells[cell_i].begin(), parametric_interface.connectivity_this_HHO_cells[cell_i].end());

        parametric_interface.connectivity_this_HHO_cells[cell_i].resize( std::distance(parametric_interface.connectivity_this_HHO_cells[cell_i].begin(),it) );
    }
}

template<typename Mesh , typename Curve, typename T >
void
set_cut_faces(Mesh& msh, const mesh_init_params<T>& mip, Curve& parametric_interface)
{
    typedef typename Mesh::point_type point_type;
    size_t Ny = mip.Ny ;
    size_t N_cells = parametric_interface.N_cells;
    size_t last_term = parametric_interface.connectivity_this_HHO_cells[N_cells-1].size() -1 ;
    int elem_old = parametric_interface.connectivity_this_HHO_cells[N_cells-1][last_term] ;
    
    auto& fc_loop = msh.faces ;
    
   
    for(size_t cell_i = 0 ; cell_i < parametric_interface.N_cells ; cell_i++)
    {
        
        
//        T prev_ref_x = 0.0 ;
        size_t counter_para_cell = 0 ;
        size_t amount_cut_cells = parametric_interface.connectivity_this_HHO_cells[cell_i].size() ;
        
        std::pair< Matrix<T, 2, 1> , T > pt_on_skeleton ;
        
        for(const int& elem : parametric_interface.connectivity_this_HHO_cells[cell_i])
        {
            
            parametric_interface.connectivity_HHO_cells_this[elem].insert( {cell_i,std::make_pair(0.0,1.0)} ) ;
            auto& cl = msh.cells[elem] ;
            bool face_found = true ;
            bool face0 = false , face1 = false , face2 = false , face3 = false ;
            int which_fc = elem - elem_old ;
            T pt_to_find ;
            auto fcs = faces(msh, cl);
            size_t fc_counter = 0;
            
            if( which_fc == 1 )
            {
                face1 = true ;
                while ( face_found )
                {
                    
                    if( fcs[3] == fc_loop[fc_counter] ){
                        face_found = false;
                        fc_loop[fc_counter].user_data.location = element_location::ON_INTERFACE;
//                        std::cout<<"fc_loop[fc_counter] "<<fc_loop[fc_counter]<<" is cut!"<<std::endl;
                        msh.nodes[fc_loop[fc_counter].ptids[1]].user_data.location = element_location::IN_NEGATIVE_SIDE;
                        msh.nodes[fc_loop[fc_counter].ptids[0]].user_data.location = element_location::IN_POSITIVE_SIDE;
                        
                        /* If node 0 is in the negative region, mark it as node inside, otherwise mark node 1 */
                        fc_loop[fc_counter].user_data.node_inside = 1;
                        
                        pt_to_find = points(msh,msh.nodes[fc_loop[fc_counter].ptids[1]]).x() ;
//                        std::cout<<"fc_loop[fc_counter].ptids[1] = "<<fc_loop[fc_counter].ptids[1]<<std::endl;
//                        std::cout<<"node1 = "<<msh.nodes[fc_loop[fc_counter].ptids[1]]<<std::endl;
//                        std::cout<<"node1bis = "<<points(msh,msh.nodes[fc_loop[fc_counter].ptids[1]])<<std::endl;
//
//                        std::cout<<"fc_loop[fc_counter].ptids[0] = "<<fc_loop[fc_counter].ptids[0]<<std::endl;
//                        std::cout<<"node0 = "<<msh.nodes[fc_loop[fc_counter].ptids[0]]<<std::endl;
//                        std::cout<<"node0bis = "<<points(msh,msh.nodes[fc_loop[fc_counter].ptids[0]])<<std::endl;
//                        std::cout<<"fc_loop[fc_counter] "<<fc_loop[fc_counter]<<" is cut!"<<std::endl;
                    }
                    fc_counter++;
                    
                }
                fc_counter--;

                
            }
            else if ( which_fc == -1 )
            {
                face3 = true ;
                while ( face_found )
                {
                    
                    if( fcs[1] == fc_loop[fc_counter] ){
                        face_found = false;
                        fc_loop[fc_counter].user_data.location = element_location::ON_INTERFACE;
//                        std::cout<<"fc_loop[fc_counter] "<<fc_loop[fc_counter]<<" is cut!"<<std::endl;
                        msh.nodes[fc_loop[fc_counter].ptids[0]].user_data.location = element_location::IN_NEGATIVE_SIDE;
                        msh.nodes[fc_loop[fc_counter].ptids[1]].user_data.location = element_location::IN_POSITIVE_SIDE;
                        
                        /* If node 0 is in the negative region, mark it as node inside, otherwise mark node 1 */
                        fc_loop[fc_counter].user_data.node_inside = 0;
                        
                        pt_to_find = points(msh,msh.nodes[fc_loop[fc_counter].ptids[1]]).x() ;
                        
//                        std::cout<<"fc_loop[fc_counter].ptids[1] = "<<fc_loop[fc_counter].ptids[1]<<std::endl;
//                        std::cout<<"node1 = "<<msh.nodes[fc_loop[fc_counter].ptids[1]]<<std::endl;
//                        std::cout<<"node1bis = "<<points(msh,msh.nodes[fc_loop[fc_counter].ptids[1]])<<std::endl;
//                        std::cout<<"fc_loop[fc_counter].ptids[0] = "<<fc_loop[fc_counter].ptids[0]<<std::endl;
//                        std::cout<<"node0 = "<<msh.nodes[fc_loop[fc_counter].ptids[0]]<<std::endl;
//                        std::cout<<"node0bis = "<<points(msh,msh.nodes[fc_loop[fc_counter].ptids[0]])<<std::endl;
//                        std::cout<<"fc_loop[fc_counter] "<<fc_loop[fc_counter]<<" is cut!"<<std::endl;
                    }
                    
                    fc_counter++;
                }
                fc_counter--;

            }
            else if ( which_fc == Ny )
            {
                face0 = true ;
                while ( face_found )
                {

                    if( fcs[0] == fc_loop[fc_counter] ){
                        face_found = false;
                        fc_loop[fc_counter].user_data.location = element_location::ON_INTERFACE;
//                        std::cout<<"fc_loop[fc_counter] "<<fc_loop[fc_counter]<<" is cut!"<<std::endl;
                        msh.nodes[fc_loop[fc_counter].ptids[0]].user_data.location = element_location::IN_NEGATIVE_SIDE;
                        msh.nodes[fc_loop[fc_counter].ptids[1]].user_data.location = element_location::IN_POSITIVE_SIDE;
                        /* If node 0 is in the negative region, mark it as node inside, otherwise mark node 1 */
                        fc_loop[fc_counter].user_data.node_inside = 0;
                        
                        pt_to_find = points(msh,msh.nodes[fc_loop[fc_counter].ptids[1]]).y() ;
                        
//                        std::cout<<"fc_loop[fc_counter].ptids[1] = "<<fc_loop[fc_counter].ptids[1]<<std::endl;
//                        std::cout<<"node1 = "<<msh.nodes[fc_loop[fc_counter].ptids[1]]<<std::endl;
//                        std::cout<<"node1bis = "<<points(msh,msh.nodes[fc_loop[fc_counter].ptids[1]])<<std::endl;
//                        std::cout<<"fc_loop[fc_counter].ptids[0] = "<<fc_loop[fc_counter].ptids[0]<<std::endl;
//                        std::cout<<"node0 = "<<msh.nodes[fc_loop[fc_counter].ptids[0]]<<std::endl;
//                        std::cout<<"node0bis = "<<points(msh,msh.nodes[fc_loop[fc_counter].ptids[0]])<<std::endl;
//                        std::cout<<"fc_loop[fc_counter] "<<fc_loop[fc_counter]<<" is cut!"<<std::endl;
                    }
                    fc_counter++;
                }
                fc_counter--;

            }
            else if ( which_fc == -Ny )
            {
                face2 = true ;
                while ( face_found )
                {

                    if( fcs[2] == fc_loop[fc_counter] ){
                        face_found = false;
                        fc_loop[fc_counter].user_data.location = element_location::ON_INTERFACE;
//                        std::cout<<"fc_loop[fc_counter] "<<fc_loop[fc_counter]<<" is cut!"<<std::endl;
                        msh.nodes[fc_loop[fc_counter].ptids[1]].user_data.location = element_location::IN_NEGATIVE_SIDE;
                        msh.nodes[fc_loop[fc_counter].ptids[0]].user_data.location = element_location::IN_POSITIVE_SIDE ;
                        
                        /* If node 0 is in the negative region, mark it as node inside, otherwise mark node 1 */
                        fc_loop[fc_counter].user_data.node_inside = 1;
                        
                        pt_to_find = points(msh,msh.nodes[fc_loop[fc_counter].ptids[1]]).y() ;
                        
//                        std::cout<<"fc_loop[fc_counter].ptids[1] = "<<fc_loop[fc_counter].ptids[1]<<std::endl;
//                        std::cout<<"node1 = "<<msh.nodes[fc_loop[fc_counter].ptids[1]]<<std::endl;
//                        std::cout<<"node1bis = "<<points(msh,msh.nodes[fc_loop[fc_counter].ptids[1]])<<std::endl;
//                        std::cout<<"fc_loop[fc_counter].ptids[0] = "<<fc_loop[fc_counter].ptids[0]<<std::endl;
//                        std::cout<<"node0 = "<<msh.nodes[fc_loop[fc_counter].ptids[0]]<<std::endl;
//                        std::cout<<"node0bis = "<<points(msh,msh.nodes[fc_loop[fc_counter].ptids[0]])<<std::endl;
//                        std::cout<<"fc_loop[fc_counter] "<<fc_loop[fc_counter]<<" is cut!"<<std::endl;
                    }
                    fc_counter++;
                    
                }
                fc_counter--;

                
            }
            
            
            
            int axis = 0 ;
            if( face0 || face2 )
                axis = 1 ;
            
            if( ! face_found )
            {
                auto threshold = diameter(msh, fc_loop[fc_counter]) / 1e20;
                pt_on_skeleton = find_crossing_on_cut_face(pt_to_find, parametric_interface, cell_i , threshold , axis );
                point_type pt_interface = point_type(pt_on_skeleton.first(0),pt_on_skeleton.first(1));
                fc_loop[fc_counter].user_data.intersection_point = pt_interface ;
                
                msh.cells[elem_old].user_data.p1 = pt_interface ;
                msh.cells[elem].user_data.p0 = pt_interface ;
                
            }
            
            if(counter_para_cell == 0)
                parametric_interface.connectivity_HHO_cells_this[elem][cell_i] = std::make_pair( 0.0 , 1.0 ) ;
            
            else
            {
                parametric_interface.connectivity_HHO_cells_this[elem_old][cell_i].second = pt_on_skeleton.second ;
                parametric_interface.connectivity_HHO_cells_this[elem][cell_i] = std::make_pair( pt_on_skeleton.second , 1.0 ) ;
            }
                
            
           
            elem_old = elem;
            
            counter_para_cell++;
            
        }
        
        
        
    }
    
    
   
    
   
}


template<typename T, typename Function>
std::pair< Matrix<T, 2, 1> , T >
find_crossing_on_cut_face(T pt ,Function& parametric_interface, size_t cell_i , const T& threshold , int axis )
{
//    typedef typename Mesh::point_type point_type ;
    size_t max_iter = 100;
    
    T pm = 0.5;
    T pa = 0.0;
    T pb = 1.0;
    auto pm_prev = pm;
    
    T diff_sq ;
    
    do {
        auto lm = parametric_interface(pm,cell_i);
        auto lm_axis = lm(axis) ;
//        auto pt_a = curve(pm,cell_i);
        auto lb = parametric_interface(pb,cell_i);
        auto lb_axis = lb(axis) ;
        
        if ( (lb_axis >= pt && lm_axis >= pt) || (lb_axis < pt && lm_axis < pt) )
        {
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
        
        auto lm_prev = parametric_interface(pm_prev,cell_i);
        auto lm_prev_axis = lm_prev(axis) ;
        diff_sq = (lm_prev_axis - pt) * (lm_prev_axis - pt) ;
    } while ( (sqrt( diff_sq ) > threshold) && max_iter-- );
    
    
    return std::make_pair(parametric_interface(pm_prev,cell_i) , pm_prev);
}


template<typename Mesh , typename Curve, typename T >
void
set_domains_properties(Mesh& msh, const mesh_init_params<T>& mip, Curve& parametric_interface)
{
//    std::set<size_t> closed_undef_cells ;
    for(auto& cl : msh.cells)
    {
        for(auto& fc: faces(msh,cl) )
        {
            if(fc.is_boundary)
            {
                cl.user_data.location = element_location::IN_POSITIVE_SIDE ;
                auto nodes_cl = nodes(msh,cl) ;
                for(auto& nd_cl : nodes_cl )
                {
                    if( nd_cl.user_data.location == element_location::UNDEF )
                        msh.nodes[nd_cl.ptid].user_data.location = element_location::IN_POSITIVE_SIDE ;
//                    else
//                        std::cout<<"ANALYSE WELL THIS, msh.nodes[nd_cl.ptid] = "<<msh.nodes[nd_cl.ptid]<<std::endl;
                }
                
                auto fcs_cl = faces(msh,cl) ;
                for(auto& fc_cl : fcs_cl )
                {
                    for(auto& fc : msh.faces)
                    {
                        if(fc == fc_cl){
                            if( fc.user_data.location == element_location::UNDEF )
                                fc.user_data.location = element_location::IN_POSITIVE_SIDE ;
//                            else
//                                std::cout<<"ANALYSE WELL THIS, fc = "<<fc<<std::endl;
                        }
                    }
                    
                }
                
                break;
            }
        }
        
        if( !(cl.user_data.location == element_location::UNDEF) )
            continue ;
        
        auto f_neigh = cl.user_data.f_neighbors ;
        auto d_neigh = cl.user_data.d_neighbors ;
        std::set<size_t> neigh;
        std::merge(f_neigh.begin(), f_neigh.end(),
                    d_neigh.begin(), d_neigh.end(),
                    std::inserter(neigh, neigh.begin()));
        
        size_t offset_cl = offset(msh,cl);
//        closed_undef_cells.insert(offset_cl);
        for(auto& cl_i : neigh)
        {
//            if(msh.cells[cl_i].user_data.location == element_location::IN_NEGATIVE_SIDE)
//            {
//
//                for(auto& other_cls : closed_undef_cells)
//                {
//                    if( (cl.user_data.location == element_location::UNDEF) )
//                        msh.cells[other_cls].user_data.location = element_location::IN_NEGATIVE_SIDE ;
//                    // SET NODES AND FACES NEGATIVE
//                }
//                closed_undef_cells.erase(closed_undef_cells.begin(),closed_undef_cells.end());
//            }
            if(msh.cells[cl_i].user_data.location == element_location::IN_POSITIVE_SIDE)
            {
                cl.user_data.location = element_location::IN_POSITIVE_SIDE ;
                auto nodes_cl = nodes(msh,cl) ;
                for(auto& nd_cl : nodes_cl )
                {
                    if( nd_cl.user_data.location == element_location::UNDEF )
                        msh.nodes[nd_cl.ptid].user_data.location = element_location::IN_POSITIVE_SIDE ;
//                    else
//                        std::cout<<"ANALYSE WELL THIS, msh.nodes[nd_cl.ptid] = "<<msh.nodes[nd_cl.ptid]<<std::endl;
                }
                
                auto fcs_cl = faces(msh,cl) ;
                for(auto& fc_cl : fcs_cl )
                {
                    for(auto& fc : msh.faces)
                    {
                        if(fc == fc_cl){
                            if( fc.user_data.location == element_location::UNDEF )
                                fc.user_data.location = element_location::IN_POSITIVE_SIDE ;
//                            else
//                                std::cout<<"ANALYSE WELL THIS, fc = "<<fc<<std::endl;
                        }
                    }
                    
                }
                break;
            }
//            else if(msh.cells[cl_i].user_data.location == element_location::ON_INTERFACE)
//            {
//                auto nds = nodes(msh , msh.cells[cl_i]);
//
//
//                closed_undef_cells.erase(closed_undef_cells.begin(),closed_undef_cells.end());
//            }
            else if(msh.cells[cl_i].user_data.location == element_location::UNDEF)
            {
                bool bdry_found = false;
                for(auto& fc: faces(msh,msh.cells[cl_i]))
                {
                    if(fc.is_boundary)
                    {
                        cl.user_data.location = element_location::IN_POSITIVE_SIDE ;
                        auto nodes_cl = nodes(msh,cl) ;
                        for(auto& nd_cl : nodes_cl )
                        {
                            if( nd_cl.user_data.location == element_location::UNDEF )
                                msh.nodes[nd_cl.ptid].user_data.location = element_location::IN_POSITIVE_SIDE ;
                            else
                                std::cout<<"ANALYSE WELL THIS, msh.nodes[nd_cl.ptid] = "<<msh.nodes[nd_cl.ptid]<<std::endl;
                        }
                        
                        auto fcs_cl = faces(msh,cl) ;
                        for(auto& fc_cl : fcs_cl )
                        {
                            for(auto& fc : msh.faces)
                            {
                                if(fc == fc_cl){
                                    if( fc.user_data.location == element_location::UNDEF )
                                        fc.user_data.location = element_location::IN_POSITIVE_SIDE ;
                                    else
                                        std::cout<<"ANALYSE WELL THIS, fc = "<<fc<<std::endl;
                                }
                            }
                            
                        }
                        
                        bdry_found = true;
                        break;
                    }
                }
                
//                if(!bdry_found)
//                    closed_undef_cells.insert(cl_i);
                
            }
        }
        
    }
    
    
    for(auto& cl : msh.cells)
    {
        if( cl.user_data.location == element_location::UNDEF )
        {
            cl.user_data.location = element_location::IN_NEGATIVE_SIDE ;
            auto nodes_cl = nodes(msh,cl) ;
            for(auto& nd_cl : nodes_cl )
            {
                if( nd_cl.user_data.location == element_location::UNDEF )
                    msh.nodes[nd_cl.ptid].user_data.location = element_location::IN_NEGATIVE_SIDE ;
            }
            
            auto fcs_cl = faces(msh,cl) ;
            for(auto& fc_cl : fcs_cl )
            {
                for(auto& fc : msh.faces)
                {
                    if(fc == fc_cl){
                        if( fc.user_data.location == element_location::UNDEF )
                            fc.user_data.location = element_location::IN_NEGATIVE_SIDE ;
                    }
                }
                
            }
        }
        
        else if( cl.user_data.location == element_location::ON_INTERFACE )
        {
            auto fcs_cl = faces(msh,cl) ;
            for(auto& fc_cl : fcs_cl )
            {
                for(auto& fc : msh.faces)
                {
                    if(fc == fc_cl){
                        if( fc.user_data.location == element_location::UNDEF ){
                            auto nods = fc.ptids ;
                            bool neg0 = !(msh.nodes[nods[0]].user_data.location == element_location::IN_POSITIVE_SIDE );
                            bool neg1 = !(msh.nodes[nods[1]].user_data.location == element_location::IN_POSITIVE_SIDE);
                            
                            if(neg0 && neg1 )
                                fc.user_data.location = element_location::IN_NEGATIVE_SIDE ;
                            else if(! neg0 && ! neg1 )
                                fc.user_data.location = element_location::IN_POSITIVE_SIDE ;
                            else
                                std::cout<<"ERROR, IT SHOULDN'T BE CUT!"<<std::endl;
                            
                        }
                    }
                }
                
            }
            
//            auto nodes_cl = nodes(msh,cl) ;
//            size_t counter_p=0 , counter_n = 0;
//            for(auto& nd_cl : nodes_cl )
//            {
//                if( nd_cl.user_data.location == element_location::UNDEF )
//                    msh.nodes[nd_cl.ptid].user_data.location = element_location::IN_POSITIVE_SIDE ;
//            }
        }
        
        
        
        
//        if ( location(msh, cl) == element_location::IN_POSITIVE_SIDE )
//            std::cout<<"Cell = "<<offset(msh,cl)<<" is IN POSITIVE SIDE."<<std::endl;
//        else if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
//            std::cout<<"Cell = "<<offset(msh,cl)<<" is IN NEGATIVE SIDE."<<std::endl;
//        else if ( location(msh, cl) == element_location::ON_INTERFACE )
//            std::cout<<"Cell = "<<offset(msh,cl)<<" is ON INTERFACE."<<std::endl;
//        else
//            std::cout<<"Cell = "<<offset(msh,cl)<<" error is UNDEF."<<std::endl;
        

    }
    
    
    
    
//    for(auto& nd : msh.nodes)
//    {
//        if ( location(msh, nd) == element_location::IN_POSITIVE_SIDE )
//            std::cout<<"Node = "<<nd<<" is IN POSITIVE SIDE."<<std::endl;
//        else if ( location(msh, nd) == element_location::IN_NEGATIVE_SIDE )
//            std::cout<<"Node = "<<nd<<" is IN NEGATIVE SIDE."<<std::endl;
//        else if ( location(msh, nd) == element_location::ON_INTERFACE )
//            std::cout<<"Node = "<<nd<<" is ON INTERFACE."<<std::endl;
//        else
//            std::cout<<"Node = "<<nd<<" error is UNDEF."<<std::endl;
//    }
//
//    for(auto& fc : msh.faces)
//    {
//        if ( location(msh, fc) == element_location::IN_POSITIVE_SIDE )
//            std::cout<<"Face = "<<fc<<" is IN POSITIVE SIDE."<<std::endl;
//        else if ( location(msh, fc) == element_location::IN_NEGATIVE_SIDE )
//            std::cout<<"Face = "<<fc<<" is IN NEGATIVE SIDE."<<std::endl;
//        else if ( location(msh, fc) == element_location::ON_INTERFACE )
//            std::cout<<"Face = "<<fc<<" is ON INTERFACE."<<std::endl;
//        else{
//            std::cout<<"Face = "<<fc<<" error is UNDEF."<<std::endl;
//            auto nods = fc.ptids ;
//            for(auto& cl : msh.cells)
//            {
//                for(auto& nd_cl : nodes(msh,cl))
//                {
//                    if(nods[0] == nd_cl.ptid)
//                        std::cout<<"Node0 = "<<nd_cl.ptid<<" in cell = "<<offset(msh,cl)<<std::endl;
//                    if(nods[1] == nd_cl.ptid)
//                        std::cout<<"Node1 = "<<nd_cl.ptid<<" in cell = "<<offset(msh,cl)<<std::endl;
//                }
//            }
//        }
//    }
        
}


template<typename Mesh , typename Curve, typename T >
void
refine_intereface_HHO(Mesh& msh, const mesh_init_params<T>& mip, Curve& parametric_interface,size_t levels)
{
    typedef typename Mesh::point_type point_type ;
    size_t interface_points;
    size_t degree_curve = parametric_interface.basis_degree;

    if (levels == 0 && degree_curve == 1)
        return;

    interface_points = iexp_pow(2, levels)*degree_curve;

    for (auto& cl : msh.cells)
    {
        if ( !is_cut(msh, cl) )
            continue;


        cl.user_data.interface.resize(interface_points+1);
        cl.user_data.interface.at(0)                = cl.user_data.p0;
        cl.user_data.interface.at(interface_points) = cl.user_data.p1;

        size_t elem = offset(msh,cl) ;
        
        auto local_para_functions = parametric_interface.connectivity_HHO_cells_this[elem] ;
        T distance = 0.0 ;
        T pos ;
        size_t counter = 0;
        for(auto& local_func : local_para_functions)
        {
            distance += (local_func.second.second - local_func.second.first);
            
            if(counter == 0)
                pos = local_func.second.first;
            
            counter++;
        }
        
        T step = distance / interface_points ;
//        std::cout<<"step = "<<step<<std::endl;
        
        size_t counter_pts = 1;
        
        
        for(auto& local_func : local_para_functions)
        {
            auto  vec_init_end = local_func.second ;
            
            while(counter_pts < interface_points)
            {
                
            
                T loc_fin = vec_init_end.second - step ;
            
                if( pos < loc_fin )
                {
                    pos += step ;
                    size_t cell_i = local_func.first ;
                    auto vec_pt = parametric_interface(pos,cell_i) ;
                    point_type interface_pt = point_type(vec_pt[0],vec_pt[1]) ;
                    cl.user_data.interface.at(counter_pts) = interface_pt ;
                    counter_pts++;
                }
                else
                {
                    T diff = vec_init_end.second - pos  ;
                    pos = 0.0 - diff ;
                    break;
                }
            }
            
        }
        
//        std::cout<<"In cell "<<offset(msh,cl)<<'\n';
//        for(auto& pt : cl.user_data.interface)
//            std::cout<<pt<<" , ";
//
//        std::cout<<'\n'<<std::endl;
       
    }

}


struct triangular_parametrisation_curve
{
    
    size_t basis_degree , basis_size;
    
   
    
    triangular_parametrisation_curve( size_t degree ) : basis_degree(degree), basis_size((basis_degree+1.0)*(basis_degree+2.0)/2.0) {}
    
    triangular_parametrisation_curve(){}

    
    template<typename T>
    Matrix<T, 2, 1>
    operator()( const point<T, 2>& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {
        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        size_t basis_size = (basis_degree+1.0)*(basis_degree+2.0)/2.0 ;
        cell_basis_triangle_Lagrange<T> cb(basis_degree);
        auto basis = cb.eval_basis(pt) ;
        
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            ret(0) += basis(i)*physical_pts[i].x();
            ret(1) += basis(i)*physical_pts[i].y();
        }
        return ret;
        
    }
    
    template<typename T>
    Matrix<T, 2, 1>
    operator()( const point<T, 2>& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {

        Matrix<T, 2, 1> ret = Matrix<T, 2, 1>::Zero(2, 1);
        cell_basis_triangle_Lagrange<T> cb(basis_degree);
        auto basis = cb.eval_basis(pt) ;
//        std::cout<<'\n'<<"basis"<<'\n'<<basis<<'\n'<<std::endl;
        for(size_t i = 0 ; i < basis_size ; i++)
        {
//            std::cout<<"basis(i,0) = "<<basis(i)<<std::endl;
//            std::cout<<"physical_pts = ("<<physical_pts[i].x()<<" , "<<physical_pts[i].y()<<" ) "<<std::endl;
            ret(0) += basis(i)*physical_pts[i].x();
            ret(1) += basis(i)*physical_pts[i].y();
            
        }
//        std::cout<<"ret = "<<ret<<std::endl;
        return ret;
        
        
    }
    
    
    template<typename T>
    Matrix<T, 2, 2>
    gradient( const point<T, 2>& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {

        Matrix<T, 2, 2> ret = Matrix<T, 2, 2>::Zero(2, 2);
        size_t basis_size = (basis_degree+1.0)*(basis_degree+2.0)/2.0 ;
        cell_basis_triangle_Lagrange <T> cb(basis_degree);
        auto basis = cb.eval_gradients(pt) ;
        
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            ret(0,0) += basis(i,0)*physical_pts[i].x();
            ret(0,1) += basis(i,1)*physical_pts[i].x();
            ret(1,0) += basis(i,0)*physical_pts[i].y();
            ret(1,1) += basis(i,1)*physical_pts[i].y();
        }
        
        return ret;
        
        
    }
    
    template<typename T>
    Matrix<T, 2, 2>
    gradient( const point<T, 2>& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {

        Matrix<T, 2, 2> ret = Matrix<T, 2, 2>::Zero(2, 2);
        cell_basis_triangle_Lagrange <T> cb(basis_degree);
        auto basis = cb.eval_gradients(pt) ;
//        std::cout<<'\n'<<"basis"<<'\n'<<basis<<'\n'<<std::endl;
        
        for(size_t i = 0 ; i < basis_size ; i++)
        {
            ret(0,0) += basis(i,0)*physical_pts[i].x();
//            std::cout<<"basis(i,0) = "<<basis(i,0)<<std::endl;
//            std::cout<<"physical_pts = "<<physical_pts[i].x()<<std::endl;
            ret(0,1) += basis(i,1)*physical_pts[i].x();
            ret(1,0) += basis(i,0)*physical_pts[i].y();
            ret(1,1) += basis(i,1)*physical_pts[i].y();
        }
        return ret;
        
        
    }
    
    
    template<typename T>
    T
    jacobian( const point<T, 2>& pt , const std::vector<point<T, 2>>& physical_pts , size_t basis_degree ) const
    {
        auto grad = this->gradient(pt,physical_pts,basis_degree) ;
        return std::abs( grad(0,0)*grad(1,1) - grad(0,1)*grad(1,0) );
        //return (this->gradient(pt,physical_pts,basis_degree)).norm();
    }
    
    template<typename T>
    T
    jacobian( const point<T, 2>& pt , const std::vector<point<T, 2>>& physical_pts ) const
    {
        auto grad = this->gradient(pt,physical_pts,basis_degree) ;
        return std::abs( grad(0,0)*grad(1,1) - grad(0,1)*grad(1,0) );
        //return (this->gradient(pt,physical_pts)).norm();
    }
    
    
    
    
    
};


/*
template<typename T, size_t ET>
std::vector< typename cuthho_mesh<T, ET>::point_type >
collect_triangulation_points_curve(const cuthho_mesh<T, ET>& msh,
                             const typename cuthho_mesh<T, ET>::cell_type& cl,
                             const element_location& where)
{
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
            ret.insert(ret.end(), cl.user_data.integration_msh.interface_vertices.begin(), cl.user_data.integration_msh.interface_vertices.end());
        else if (where == element_location::IN_POSITIVE_SIDE)
            ret.insert(ret.end(), cl.user_data.integration_msh.interface_vertices.rbegin(), cl.user_data.integration_msh.interface_vertices.rend());
        else
            throw std::logic_error("If you've got here there is some issue...");
    };

    bool case1 = location(msh, ns.front()) == where && location(msh, ns.back()) != where;
    bool case2 = location(msh, ns.front()) != where && location(msh, ns.back()) == where;
    bool case3 = location(msh, ns.front()) != where && location(msh, ns.back()) != where;
    //bool case4 = location(msh, ns.front()) == where && location(msh, ns.back()) == where;

    if ( case1 || case2 || case3 )
    {
        for (size_t i = 0; i < ns.size(); i++)
            if ( location(msh, ns[i]) == where )
                ret.push_back( node2pt(ns[i]) );

        insert_interface();
    }
    else // case4, the only possible remaining
    {
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

*/



template<typename T, size_t ET>
std::vector< typename cuthho_mesh<T, ET>::point_type >
collect_triangulation_points_curve_old(const cuthho_mesh<T, ET>& msh,
                             const typename cuthho_mesh<T, ET>::cell_type& cl, typename cuthho_mesh<T, ET>::point_type& bar ,
                             const element_location& where)
{
    typedef typename cuthho_mesh<T, ET>::point_type     point_type;
    typedef typename cuthho_mesh<T, ET>::node_type      node_type;

    assert( is_cut(msh, cl) );
    auto ns = nodes(msh, cl);
    std::vector< node_type > n_where;
    
    auto node2pt = [&](const node_type& n) -> auto {
        return msh.points.at(n.ptid);
    };
    
    size_t counter = 0;
    size_t first_node ;
    //bool case_odd = false ;
    //bool first_n = false ;
    for(auto& n : ns ){
        if(location(msh, n) == where ){
            if( bar == node2pt(n) ){
                first_node = counter++;
                //case_odd = true ;
            }
            else
            {
                n_where.push_back(n);
                counter++;
                //if(first_n == false){
                //    first_n = true ;
                 //   n_where.push_back(n);
                //}
                //else
                //{
                //    counter++;
                //    n_where.push_back(n);
                //}
               
            }
           
        }
    }
    
    std::vector< point_type > ret;

    

    auto insert_interface = [&](void) -> void {
        if (where == element_location::IN_NEGATIVE_SIDE)
            ret.insert(ret.end(), cl.user_data.integration_msh.interface_vertices.begin(), cl.user_data.integration_msh.interface_vertices.end());
        else if (where == element_location::IN_POSITIVE_SIDE)
            ret.insert(ret.end(), cl.user_data.integration_msh.interface_vertices.rbegin(), cl.user_data.integration_msh.interface_vertices.rend());
        else
            throw std::logic_error("If you've got here there is some issue...");
    };
    
    
    
    if( counter == 1 )
        insert_interface();

    //if( case_odd == false && counter == 2 )
    if( counter == 2 )
    {
        ret.push_back( node2pt(n_where[1]) );
        insert_interface();
        ret.push_back( node2pt(n_where[0]) );
        
    }
    
    //if( case_odd == true && counter == 2 )
    if( counter == 3 )
    {
        if( first_node == 2 )
            first_node = 0;
        ret.push_back( node2pt(n_where[first_node]) );
        insert_interface();
        ret.push_back( node2pt(n_where[1-first_node]) );
    }
    

    return ret;
}
    
    



    
    
template<typename T, size_t ET>
std::vector< typename cuthho_mesh<T, ET>::point_type >
collect_triangulation_points_curve(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh< T , ET>::cell_type& cl, typename cuthho_mesh<T, ET>::point_type& bar , const element_location& where )
 {
        
    typedef typename cuthho_mesh<T, ET>::point_type     point_type;
    typedef typename cuthho_mesh<T, ET>::node_type      node_type;

    assert( is_cut(msh, cl) );
    
    
    
        
    auto node2pt = [&](const node_type& n) -> auto {
        return msh.points.at(n.ptid);
    };
        
   
    
    std::vector< point_type > ret;

    auto insert_interface = [&](void) -> void {
        if (where == element_location::IN_NEGATIVE_SIDE)
            ret.insert(ret.end(), cl.user_data.integration_msh.interface_vertices.begin(), cl.user_data.integration_msh.interface_vertices.end());
        else if (where == element_location::IN_POSITIVE_SIDE)
            ret.insert(ret.end(), cl.user_data.integration_msh.interface_vertices.rbegin(), cl.user_data.integration_msh.interface_vertices.rend());
        else
            throw std::logic_error("If you've got here there is some issue...");
    };
        

       
    std::vector<size_t> fc_numb ;
    node_type n_first , n_second ;
    
    for(auto& fc: faces(msh, cl))
    {
        if( fc.user_data.location == element_location::ON_INTERFACE && fc.user_data.intersection_point == cl.user_data.integration_msh.interface_vertices.front() ){
            if (where == element_location::IN_NEGATIVE_SIDE)
                n_first = nodes(msh, fc)[fc.user_data.node_inside] ;
            else
                n_second = nodes(msh, fc)[1-fc.user_data.node_inside] ;
        }
    
        if( fc.user_data.location == element_location::ON_INTERFACE && fc.user_data.intersection_point == cl.user_data.integration_msh.interface_vertices.back() )
        {
            if (where == element_location::IN_NEGATIVE_SIDE)
                n_second = nodes(msh, fc)[fc.user_data.node_inside] ;
            else
                n_first = nodes(msh, fc)[1-fc.user_data.node_inside] ;
    
        }

    }
    
    
    if( n_first == n_second )
        insert_interface();

    else
    {
        ret.push_back( node2pt(n_first) );
        insert_interface();
        ret.push_back( node2pt(n_second) );
            
    }

    return ret;
    
}
    
    

template<typename T>
struct temp_tri_curve
{
    //std::array< point<T,2>, 3 > pts;
    std::vector< point<T,2> > pts;
    size_t degree ;
    size_t size;
    
    temp_tri_curve(size_t degree_int):degree(degree_int),size((degree_int+1)*(degree_int+2)/2.0){
        pts.resize( size ) ;
    }

    /*
    T area() const {
        auto v1 = pts[1] - pts[0];
        auto v2 = pts[2] - pts[0];

        return ( v1.x()*v2.y() - v2.x()*v1.y() ) / 2.0;
        // can be negative
    }
    */
};


template<typename T, size_t ET>
std::vector<std::vector< point<T,2> > >  // std::vector<temp_tri_curve<T>>
triangulate_curve(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
            element_location where)
{
    assert( is_cut(msh, cl) );

    auto integration_msh = cl.user_data.integration_msh ;
    size_t degree_int = integration_msh.degree_curve ;
    auto cl_vertices = integration_msh.interface_vertices ;
    size_t size_vert = cl_vertices.size() - 1 ; 
    //for(auto& vertex : cl_vertices)
    //    std::cout<<" , vertex = "<<vertex;
    //std::cout<<std::endl;
    
    T degree_int_num = degree_int*1.0;
    T degree_size_triangle = (degree_int + 1.0 )*(degree_int +2.0 )/2.0 ;
    //std::vector<temp_tri_curve<T> > tris_high_order ;
    std::vector<std::vector< point<T,2> >> tris_high_order ;
    
    auto bar = tesselation_center(msh, cl, where);
    auto tp = collect_triangulation_points_curve(msh, cl,bar, where);
    
    
    
//    for(auto& p : tp)
//        std::cout<<" , point = "<<p;
//    std::cout<<std::endl;
//    std::cout<<"bar = "<<bar<<std::endl;
//

    if( degree_int == 1)
    {
        for (size_t i = 0; i < tp.size() - 1 ; i++)
        {
            std::vector< point<T,2> > t(degree_size_triangle);
            t[0] = bar ;
            t[1] = tp[i];
            t[2] = tp[(i+1)%tp.size()];
            //temp_tri_curve<T> t(degree_int);
            //t.pts[0] = bar;
            //t.pts[1] = tp[i];
            //t.pts[2] = tp[(i+1)%tp.size()];
            tris_high_order.push_back(t);
        }
        return tris_high_order;
    }
    else
    {
            size_t counter_vert = 0;
//            if(where == element_location::IN_POSITIVE_SIDE)
//            {
//               std::reverse(cl_vertices.begin(), cl_vertices.end());
//            }
            
            for (size_t i = 0; i < tp.size() - 1 ; i++)
            {
                size_t pos = 0;
                std::vector< point<T,2> > t(degree_size_triangle);
                //temp_tri_curve<T> t(degree_int);
                //t.pts[pos++] = bar;
                bool interface_side = false ;
                t[pos++] = bar ;
                auto tp1 = tp[i];
                auto tp2 = tp[(i+1)%tp.size()];
                t[pos++] = tp1;
                t[pos++] = tp2;
                //t.pts[pos++] = tp1;
                //t.pts[pos++] = tp2;
            
                for(size_t i = 2 ; i < degree_int + 1 ; i++ )
                {
               
                    t[pos++] = bar + (i-1.0)/(degree_int_num)*(tp1-bar); // high order point
                    //t.pts[pos++] = bar + (i-1.0)/(degree_int_num)*(tp1-bar); // high order point
                   // && counter_vert < cl_vertices.size() && ERA COSI
                }
                for(size_t i = 2 ; i < degree_int + 1 ; i++ )
                {
                    // && tp2 == cl_vertices[(counter_vert+1)%cl_vertices.size() ERA COSI
                    if( where == element_location::IN_NEGATIVE_SIDE && counter_vert < size_vert &&  tp1 == cl_vertices[counter_vert] && tp2 == cl_vertices[(counter_vert+1)] )
                    {
                    // If I'm in the curvilinear side
                        auto pts =  points( integration_msh,integration_msh.cells[counter_vert] ) ;
                        //t.pts[pos++] = pts[i]; // high order point
                        t[pos++] = pts[i]; // high order point
                        interface_side = true;
//                        counter_vert ++ ;
                    }
                    else if( where == element_location::IN_POSITIVE_SIDE && counter_vert < size_vert &&  tp1 == cl_vertices[size_vert-counter_vert] && tp2 == cl_vertices[size_vert-(counter_vert+1)] )
                    {
                    // If I'm in the curvilinear side
                        auto pts =  points( integration_msh,integration_msh.cells[size_vert-1-counter_vert] ) ;
                        //t.pts[pos++] = pts[i]; // high order point
                        t[pos++] = pts[pts.size()+1-i]; // high order point
                        interface_side = true;
//                        counter_vert ++ ;
                    }
                    else
                        t[pos++] = tp1 + (i-1.0)/(degree_int_num)*(tp2-tp1); // high order point
                        //t.pts[pos++] = tp1 + (i-1.0)/(degree_int_num)*(tp2-tp1); // high order point
                }
                for(size_t i = 2 ; i < degree_int + 1 ; i++ )
                {
                    t[pos++] = tp2 + (i-1.0)/(degree_int_num)*(bar-tp2); // high order point
                    //t.pts[pos++] = tp2 + (i-1.0)/(degree_int_num)*(bar-tp2); // high order point
                    
                
                }
                if( degree_int == 3 )
                    t[pos++] = t[7] + 0.5*(t[4]-t[7]);
                    //t[pos++] = { bar.x() + (2.0)/(degree_int_num)*(t[1].x()-bar.x()) ,  bar.y() + (2.0)/(degree_int_num)*(tp2.y()-bar.y()) } ;
                    if( degree_int == 4 )
                    {
//                        std::cout<<" -------------> ANCORA SBAGLIATO CREDO, DA CONTROLLARE!!!"<<std::endl;
                         t[pos++] = t[10] + 0.5*(t[4]-t[10]);
                        t[pos++] = t[9] + 2.0/3.0*(t[5]-t[9]);
                        t[pos++] = t[9] + 1.0/3.0*(t[5]-t[9]);
                        //t[pos++] = { bar.x() + (1.0)/(degree_int_num)*(tp1.x()-bar.x()) ,  bar.y() + (1.0)/(degree_int_num)*(tp2.y()-bar.y()) } ;
                        //t[pos++] = { bar.x() + (2.0)/(degree_int_num)*(tp1.x()-bar.x()) ,  bar.y() + (1.0)/(degree_int_num)*(tp2.y()-bar.y()) } ;
                        //t[pos++] = { bar.x() + (1.0)/(degree_int_num)*(tp1.x()-bar.x()) ,  bar.y() + (2.0)/(degree_int_num)*(tp2.y()-bar.y()) } ;
                        
                    }
                if(interface_side)
                    counter_vert ++ ;
                
                tris_high_order.push_back(t);
//                if(where == element_location::IN_POSITIVE_SIDE)
//                {
//                for(auto& tt : t)
//                    std::cout<< std::setprecision(std::numeric_limits<long double>::digits10 + 1)<<" , tt = "<<tt;
//                std::cout<<'\n'<<std::endl;
//                }
            }
        

        return tris_high_order;
    }
}

    
    
    
template<typename T, typename Cell >
std::vector<std::pair<point<T,2>, T>> // temp_tri_curve<T> tri
triangle_quadrature_curve(const std::vector< point<T,2> >& tri, const Cell& cl, size_t deg)
{
    //typedef typename Cell::point_type     point_type;
    if (deg == 0)
        deg = 1;
    
    if (deg == 10)
           deg = 11;
    
    if (deg == 14 || deg == 15)
        deg = 16;
    
    if (deg == 17)
        deg = 18;
    


    if (deg > 18){
        std::cout<<"Integration degree demanded : "<<deg<<std::endl;
        throw std::invalid_argument("Quadrature order too high");
    }
    size_t degree_int = cl.user_data.integration_msh.degree_curve;
    //cell_basis_triangle_Lagrange<T> cb_tri(degree_int);
    triangular_parametrisation_curve curved_tri_para( degree_int );
    

    using namespace dunavant_quadratures;

    std::vector<std::pair<point<T,2>, T>> ret;

    ret.reserve( rules[deg].num_points );

    for (size_t i = 0; i < rules[deg].num_points; i++)
    {

        //point_type p = point_type(rules[deg].data[i][1] , rules[deg].data[i][2] );
        point<T,2> qp =  { rules[deg].data[i][1] , rules[deg].data[i][2] };   // p;
        auto p = curved_tri_para( qp , tri ) ;
        point<T,2> pt = {p(0) , p(1)} ;
//        std::cout<<"qp = "<<qp<<" , pt = "<<pt<<std::endl;
//        // QUESTION: these points are good also for curved triangles??
//        std::cout<<"Grad( "<< qp <<" ) = "<<curved_tri_para.gradient( qp, tri )<<std::endl;
//        std::cout<<"curved_tri_para.jacobian( qp, tri ) = "<<curved_tri_para.jacobian( qp, tri ) <<std::endl;
        T qw          = 0.5*curved_tri_para.jacobian( qp, tri ) * rules[deg].data[i][3];
        // 0.5 = Area of reference triangle

        ret.push_back( std::make_pair(pt, qw) );
    }

    return ret;
}


template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
make_integrate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, size_t degree, const element_location& where , bool used_from_integrate_fc = false)
{
    std::vector< std::pair<point<T,2>, T> > ret;

    if ( location(msh, cl) != where && location(msh, cl) != element_location::ON_INTERFACE )
        return ret;

    if ( !is_cut(msh, cl) ) /* Element is not cut, use std. integration */
        return integrate(msh, cl, degree);
    
    // (1) TOLTO PER CONTROLLARE CASE LINEARE
    if(!used_from_integrate_fc){
        size_t degree_int = cl.user_data.integration_msh.degree_curve;
        degree += degree_det_jacobian(degree_int) ;
    }
        
  
    auto tris = triangulate_curve(msh, cl, where);
    //size_t degree_int = cl.user_data.integration_msh.degree;
    //cell_basis_triangle_Lagrange<T> cb_tri(degree_int);
   // std::cout<<"the size of tris is "<<tris.size()<<std::endl;
    for (auto& tri : tris)
    {
        // fatto io da qua
        /*
        auto v0 = tri.pts[1] - tri.pts[0];
        auto v1 = tri.pts[2] - tri.pts[0];
        auto area = (v0.x() * v1.y() - v0.y() * v1.x()) / 2.0;
        auto counter = offset(msh,cl);
        if(area < 0){
               for( auto& r : cl.user_data.interface ){
                  std::cout<<"In cella num "<<counter<<std::endl;
                   std::cout<<"Point r: x is "<<r.x()<<", y is "<<r.y()<<std::endl;
               }
           
           }
         */
        // a qua
    
        auto qpts = triangle_quadrature_curve(tri,cl, degree);
        ret.insert(ret.end(), qpts.begin(), qpts.end());
    }
    
    return ret;
}







template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> > // HO INVERTITO I NOMI PER FARE LA PROVA
integrate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
          size_t degree, const element_location& where)
{
    // (2) TOLTO PER CONTROLLARE CASE LINEARE
    if ( is_cut(msh, cl) ){
        size_t degree_int = cl.user_data.integration_msh.degree_curve;
        degree += degree_det_jacobian(degree_int) ;
    }
    
    if( cl.user_data.integration_n.size() != 0 && where == element_location::IN_NEGATIVE_SIDE)
        return cl.user_data.integration_n;

    if( cl.user_data.integration_p.size() != 0 && where == element_location::IN_POSITIVE_SIDE)
        return cl.user_data.integration_p;

    return make_integrate(msh, cl, degree, where,true);
}



template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
make_integrate_old(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
               size_t degree, const element_location& where)
{
    std::vector< std::pair<point<T,2>, T> > ret;

    if ( location(msh, cl) != where && location(msh, cl) != element_location::ON_INTERFACE )
        return ret;

    if ( !is_cut(msh, cl) ) /* Element is not cut, use std. integration */
        return integrate(msh, cl, degree);
    
  
    auto tris = triangulate(msh, cl, where);
   // std::cout<<"the size of tris is "<<tris.size()<<std::endl;
    for (auto& tri : tris)
    {
        // fatto io da qua
        /*
        auto v0 = tri.pts[1] - tri.pts[0];
        auto v1 = tri.pts[2] - tri.pts[0];
        auto area = (v0.x() * v1.y() - v0.y() * v1.x()) / 2.0;
        auto counter = offset(msh,cl);
        if(area < 0){
               for( auto& r : cl.user_data.interface ){
                  std::cout<<"In cella num "<<counter<<std::endl;
                   std::cout<<"Point r: x is "<<r.x()<<", y is "<<r.y()<<std::endl;
               }
           
           }
         */
        // a qua
        auto qpts = triangle_quadrature(tri.pts[0], tri.pts[1], tri.pts[2], degree);
        ret.insert(ret.end(), qpts.begin(), qpts.end());
    }
    
    return ret;
}


template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
integrate_old(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
          size_t degree, const element_location& where)
{
    
    if( cl.user_data.integration_n.size() != 0 && where == element_location::IN_NEGATIVE_SIDE)
        return cl.user_data.integration_n;

    if( cl.user_data.integration_p.size() != 0 && where == element_location::IN_POSITIVE_SIDE)
        return cl.user_data.integration_p;

    return make_integrate(msh, cl, degree, where);
}


template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
integrate(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc,
          size_t degree, const element_location& where)
{
    std::vector< std::pair<point<T,2>, T> > ret;

    if ( location(msh, fc) != where && location(msh, fc) != element_location::ON_INTERFACE )
        return ret;

    if ( !is_cut(msh, fc) ) /* Element is not cut, use std. integration */
        return integrate(msh, fc, degree);

    auto pts = points(msh, fc, where);

    auto scale = pts[1] - pts[0];
    auto meas = scale.to_vector().norm();

    auto qps = edge_quadrature<T>(degree);

    for (auto itor = qps.begin(); itor != qps.end(); itor++)
    {
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
integrate_interface_old(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                    size_t degree, element_location where)
{
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

    for (size_t i = 1; i < cl.user_data.interface.size(); i++)
    {
        auto p0 = cl.user_data.interface.at(i-1);
        auto p1 = cl.user_data.interface.at(i);

        auto scale = p1 - p0;
        auto meas = scale.to_vector().norm();

        for (auto itor = qps.begin(); itor != qps.end(); itor++)
        {
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



template<typename T, size_t ET>
std::vector< std::pair<point<T,2>, T> >
integrate_interface(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, size_t degree, element_location where )
{
    assert( is_cut(msh, cl) );
    typedef typename cuthho_mesh<T, ET>::point_type point_type ;
    std::vector< std::pair<point<T,2>, T> > ret;
    
    
    auto integration_msh = cl.user_data.integration_msh ;
    size_t degree_int = integration_msh.degree_curve ;
    auto para_curve = Interface_parametrisation_mesh1d(degree_int) ;
    size_t degree_jacobian = para_curve.degree_det;
    
    degree += (degree_jacobian) ;
    auto qps = edge_quadrature<T>(degree); // Gauss Legendre integration points/weights [-1,1]
    
    
    
    for (size_t i = 0; i < integration_msh.cells.size(); i++)
    {
        auto pts = points(integration_msh,integration_msh.cells[i]);
        
        for (auto itor = qps.begin(); itor != qps.end(); itor++)
        {
            auto qp = *itor;
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = para_curve(t , pts , degree_int ) ;
            point<T,2> pt = point_type( p(0) , p(1) ) ;
            T jacobian = para_curve.jacobian( t , pts , degree_int ) ;
            auto w = 0.5 * qp.second * jacobian ;
            //std::cout<<"t = "<<t << " , pt = "<<pt <<std::endl;
            // change of variable of g-l quadrature' nodes to [0,1]
            ret.push_back( std::make_pair(pt, w) );
        }
    }
    
    return ret;
}

template<typename T, size_t ET>
T
measure_interface(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, element_location where )
{
    if ( !is_cut(msh, cl) ) /* Element is not cut, use std. integration */
        exit(9);

    T totmeas = 0.0;
    auto qpsi = integrate_interface(msh, cl, 0, where);
    for (auto& qp : qpsi)
    {
        totmeas += qp.second;
        
    }

    return totmeas;
}
    




template<typename T, size_t ET , typename Curve>
std::vector< std::pair<point<T,2>, T> >
integrate_interface(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, const Curve& curve , size_t degree, element_location where )
{
    assert( is_cut(msh, cl) );
    typedef typename cuthho_mesh<T, ET>::point_type point_type ;
    std::vector< std::pair<point<T,2>, T> > ret;
    
    
    
    degree += curve.degree_det ;
    auto qps = edge_quadrature<T>(degree); // Gauss Legendre integration points/weights [-1,1]
    
    
    size_t elem = offset(msh,cl);
    auto local_para_functions = curve.connectivity_HHO_cells_this[elem] ;
    
    for(auto& local_func : local_para_functions)
    {
        size_t cell_i = local_func.first ;
        T a = local_func.second.first;
        T b = local_func.second.second;
        
        for(auto& qp:qps)
        {
            auto t = 0.5*(qp.first.x()+1.0)*(b-a) + a ;
            T jacobian = curve.jacobian( t , cell_i ) ;
            auto w = 0.5 * qp.second * jacobian *(b-a) ;

            auto p = curve(t , cell_i ) ;
            point<T,2> pt = point_type( p(0) , p(1) ) ;
         
            ret.push_back( std::make_pair(pt, w) );
        }
        
    }
    
    
    return ret;
}

template<typename T, size_t ET, typename Curve>
T
measure_interface(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,const Curve& curve , element_location where )
{
    if ( !is_cut(msh, cl) ) /* Element is not cut, use std. integration */
        exit(9);

    T totmeas = 0.0;
    auto qpsi = integrate_interface(msh, cl,curve, 0, where);
    for (auto& qp : qpsi)
    {
        totmeas += qp.second;
        
    }

    return totmeas;
}
    



template<typename T, typename Function>
std::vector< typename cuthho_quad_mesh<T>::point_type >
make_test_points(const cuthho_quad_mesh<T>& msh, const typename cuthho_quad_mesh<T>::cell_type& cl,
                 const Function& level_set_function, element_location where)
{
    const size_t N = 10;

    auto trans = make_reference_transform(msh, cl);

    std::vector< typename cuthho_quad_mesh<T>::point_type > ret;

    auto cell_pts = points(msh, cl);

    auto min = -1.0;
    auto h = 2.0/N;

    for (size_t j = 0; j < N+1; j++)
    {
        for(size_t i = 0; i < N+1; i++)
        {
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




template<typename T, size_t ET>
void dump_mesh(const cuthho_mesh<T, ET>& msh)
{
    std::ofstream ofs("mesh_dump.m");
    size_t i = 0;
    ofs << "hold on;" << std::endl;
    for (auto& fc : msh.faces)
    {
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
    for (auto& cl : msh.cells)
    {
        auto bar = barycenter(msh, cl);
        ofs << "text(" << bar.x() << ", " << bar.y() << ", '" << i << "');" << std::endl;

        if ( is_cut(msh, cl) )
        {
            auto p0 = cl.user_data.p0;
            auto p1 = cl.user_data.p1;
            auto q = p1 - p0;
            ofs << "quiver(" << p0.x() << ", " << p0.y() << ", " << q.x() << ", " << q.y() << ", 0)" << std::endl;

            for (size_t i = 1; i < cl.user_data.interface.size(); i++)
            {
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


#if 0

template<typename T, typename ET>
class assembler<cuthho_mesh<T, ET>>
{
    std::vector<size_t>                 cell_compress_table, face_compress_table;
    std::vector<size_t>                 cell_expand_table, face_expand_table;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

    size_t num_all_cells, num_asm_cells, num_notasm_cells;
    size_t num_all_faces, num_asm_faces, num_notasm_faces;

    class assembly_index
    {
        size_t  idx;
        bool    assem;

    public:
        assembly_index(size_t i, bool as)
            : idx(i), assem(as)
        {}

        operator size_t() const
        {
            if (!assem)
                throw std::logic_error("Invalid assembly_index");

            return idx;
        }

        bool assemble() const
        {
            return assem;
        }

        friend std::ostream& operator<<(std::ostream& os, const assembly_index& as)
        {
            os << "(" << as.idx << "," << as.assem << ")";
            return os;
        }
    };

    bool cell_needs_assembly(const cuthho_mesh<T, ET>& msh,
                             const typename cuthho_mesh<T, ET>::cell_type& cl)
    {
        return location(msh, cl) == element_location::IN_NEGATIVE_SIDE ||
               location(msh, cl) == element_location::ON_INTERFACE;
    }

    bool face_needs_assembly(const cuthho_mesh<T, ET>& msh,
                             const typename cuthho_mesh<T, ET>::face_type& fc)
    {
        return location(msh, fc) == element_location::IN_NEGATIVE_SIDE ||
               location(msh, fc) == element_location::ON_INTERFACE;
    }

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    assembler(const cuthho_mesh<T, ET>& msh, hho_degree_info hdi)
        : di(hdi)
    {
        auto fna = [&](const typename cuthho_mesh<T, ET>::face_type& fc) -> bool {
            return face_needs_assembly(msh, fc);
        };

        num_all_faces = msh.faces.size();
        num_asm_faces = std::count_if(msh.faces.begin(), msh.faces.end(), fna);
        num_notasm_faces = num_all_faces - num_asm_faces;

        auto cna = [&](const typename cuthho_mesh<T, ET>::cell_type& cl) -> bool {
            return cell_needs_assembly(msh, cl);
        }
        ;
        num_all_cells = msh.cells.size();
        num_asm_cells = std::count_if(msh.cells.begin(), msh.cells.end(), cna);
        num_notasm_cells = num_all_cells - num_asm_cells;

        cell_compress_table.resize( num_all_cells );
        cell_expand_table.resize( num_asm_cells );
        face_compress_table.resize( num_all_faces );
        face_expand_table.resize( num_asm_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_cells; i++)
        {
            auto cl = msh.cells[i];
            if ( cell_needs_assembly(msh, cl) )
            {
                cell_compress_table.at(i) = compressed_offset;
                cell_expand_table.at(compressed_offset) = i;
                compressed_offset++;
            }
        }

        compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( face_needs_assembly(msh, fc) )
            {
                face_compress_table.at(i) = compressed_offset;
                face_expand_table.at(compressed_offset) = i;
                compressed_offset++;
            }
        }

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
        auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

        auto system_size = cbs * num_asm_cells + fbs * num_asm_faces;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void dump_tables() const
    {
        std::cout << "Compress table: " << std::endl;
        for (size_t i = 0; i < cell_compress_table.size(); i++)
            std::cout << i << " -> " << cell_compress_table.at(i) << std::endl;
    }

    template<typename Function>
    void
    assemble(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs,
             const Function& dirichlet_bf)
    {
        if ( !cell_needs_assembly(msh, cl) )
            return;

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
        auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs + 4*fbs);

        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset    = cell_compress_table.at(cell_offset) * cbs;

        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        auto fcs = faces(msh, cl);
        for (size_t face_i = 0; face_i < 4; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = cbs * num_asm_cells + face_compress_table.at(face_offset)*fbs;

            bool asm_face = face_needs_assembly(msh, fc);

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, asm_face) );
        }

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
        }

        RHS.block(cell_LHS_offset, 0, cbs, 1) += rhs.block(0, 0, cbs, 1);
    } // assemble()

    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_local_data(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
    const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
        auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

        if ( !cell_needs_assembly(msh, cl) )
        {
            Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + 4*fbs);
            return ret;
        }

        auto cell_offset        = offset(msh, cl);
        auto cell_SOL_offset    = cell_compress_table.at(cell_offset) * cbs;

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + 4*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);

        auto fcs = faces(msh, cl);
        for (size_t face_i = 0; face_i < 4; face_i++)
        {
            auto fc = fcs[face_i];

            if ( face_needs_assembly(msh, fc) )
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = cbs * num_asm_cells + face_compress_table.at(face_offset)*fbs;
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }

    Matrix<T, Dynamic, 1>
    expand_solution(const cuthho_mesh<T, ET>& msh,
                    const Matrix<T, Dynamic, 1>& solution)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
        auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

        auto solsize = msh.cells.size() * cbs + msh.faces.size() * fbs;

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(solsize);

        for (size_t i = 0; i < num_asm_cells; i++)
        {
            auto exp_index = cell_expand_table.at(i);
            ret.block(exp_index*cbs, 0, cbs, 1) = solution.block(i*cbs, 0, cbs, 1);
        }

        return ret;
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};


template<typename T, typename ET>
auto make_assembler(const cuthho_mesh<T, ET>& msh, hho_degree_info hdi)
{
    return assembler<cuthho_mesh<T, ET>>(msh, hdi);
}

#endif


//////////////////////////  CUT BASIS  ///////////////////////////


template<typename Mesh, typename VT>
class cut_face_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          face_bar;
    point_type          base;
    coordinate_type     face_h;
    size_t              basis_degree, basis_size;

public:
    cut_face_basis(const Mesh& msh, const typename Mesh::face_type& fc, size_t degree,
                   element_location where)
    {
        auto loc = location(msh,fc);
        if( loc != where && loc != element_location::ON_INTERFACE)
        {
            face_bar        = barycenter(msh, fc);
            face_h          = diameter(msh, fc);
            basis_degree    = degree;
            basis_size      = degree+1;

            auto pts = points(msh, fc);
            base = face_bar - pts[0];
        }
        else
        {
            face_bar        = barycenter(msh, fc, where);
            face_h          = diameter(msh, fc, where);
            basis_degree    = degree;
            basis_size      = degree+1;

            auto pts = points(msh, fc, where);
            base = face_bar - pts[0];
        }
    }
    Matrix<VT, Dynamic, 1>
    eval_basis(const point_type& pt)
    {
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
        return degree+1;
    }
};

/////////////////////////////////////////

template<typename Mesh, typename VT>
class cut_vector_face_basis
{
    typedef typename Mesh::coordinate_type  coordinate_type;
    typedef typename Mesh::point_type       point_type;

    point_type          face_bar;
    point_type          base;
    coordinate_type     face_h;
    size_t              basis_degree, basis_size;


    cut_face_basis<Mesh,VT>          scalar_basis;

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
