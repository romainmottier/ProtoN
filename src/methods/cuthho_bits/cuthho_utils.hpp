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

#pragma once

template<typename T>
struct params
{
    T kappa_1, kappa_2;
    T c_1, c_2;
    params() : kappa_1(1.0), kappa_2(1.0), c_1(1.0), c_2(1.0) {}
    params(T kap1, T kap2) : kappa_1(kap1), kappa_2(kap2) {
        c_1 = 1.0;
        c_2 = 1.0;
    }
};

//////////////////////////////////////// MAKE MASS MATRIX ////////////////////////////////////////
template<typename T, size_t ET>
Matrix<T, Dynamic, Dynamic>
make_mass_matrix(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, size_t degree, element_location where)
{

    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);

    auto qps = integrate(msh, cl, 2*degree, where);
    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi * phi.transpose();
    }

    return ret;
}

template<typename T, size_t ET>
Matrix<T, Dynamic, Dynamic>
make_mass_matrix(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc,
                 size_t degree, element_location where)
{
    cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, degree, where);
    auto fbs = fb.size();

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(fbs, fbs);

    auto qps = integrate(msh, fc, 2*degree, where);
    for (auto& qp : qps)
    {
        auto phi = fb.eval_basis(qp.first);
        ret += qp.second * phi * phi.transpose();
    }

    return ret;
}

template<typename T, size_t ET>
Matrix<T, Dynamic, Dynamic>
make_vec_mass_matrix(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                 hho_degree_info di, element_location where = element_location::UNDEF)
{
    size_t celdeg = di.cell_degree();
    size_t gradeg = di.grad_degree();
    size_t facdeg = di.face_degree();
    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, di.grad_degree());
    auto cbs = cb.size();
    
    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(cbs, cbs);
    if (element_location::UNDEF == where) {
        auto qps = integrate(msh, cl, celdeg - 1 + facdeg);
        for (auto& qp : qps)
        {
            auto phi = cb.eval_basis(qp.first);
            ret += qp.second * phi * phi.transpose();
        }
    }else{
        auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
        for (auto& qp : qps)
        {
            auto phi = cb.eval_basis(qp.first);
            ret += qp.second * phi * phi.transpose();
        }
    }

    return ret;
}


//////////////////////////////////////// MAKE_HHO_LAPLACIAN ////////////////////////////////////////
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, Dynamic>  >
make_hho_laplacian(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                   const Function& level_set_function, hho_degree_info& di,
                   element_location where)
{

    if ( !is_cut(msh, cl) )
        return make_hho_laplacian(msh, cl, di);

    std::cout<<"make_hho_laplacian: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    auto recdeg = di.reconstruction_degree();
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    cell_basis<cuthho_mesh<T, ET>,T>     cb(msh, cl, recdeg);

    auto rbs = cell_basis<cuthho_mesh<T, ET>,T>::size(recdeg);
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> stiff = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);
    Matrix<T, Dynamic, Dynamic> gr_lhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);
    Matrix<T, Dynamic, Dynamic> gr_rhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs, cbs + num_faces*fbs);

    /* Cell term (cut) */
    auto qps = integrate(msh, cl, 2*recdeg, where);
    for (auto& qp : qps)
    {
        auto dphi = cb.eval_gradients(qp.first);
        stiff += qp.second * dphi * dphi.transpose();
    }

    auto hT = diameter(msh, cl);

    /* Interface term */
    auto iqps = integrate_interface(msh, cl, 2*recdeg, where);
    for (auto& qp : iqps)
    {
        auto phi    = cb.eval_basis(qp.first);
        auto dphi   = cb.eval_gradients(qp.first);
        Matrix<T,2,1> n      = level_set_function.normal(qp.first);

        stiff -= qp.second * phi * (dphi * n).transpose();
        stiff -= qp.second * (dphi * n) * phi.transpose();
        stiff += qp.second * phi * phi.transpose() * cell_eta(msh, cl) / hT;
    }

    gr_lhs.block(0, 0, rbs, rbs) = stiff;
    gr_rhs.block(0, 0, rbs, cbs) = stiff.block(0, 0, rbs, cbs);

    auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        auto fc = fcs[i];
        auto n = ns[i];

        // face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);
        /* Terms on faces */
        auto qps = integrate(msh, fc, 2*recdeg, where);
        for (auto& qp : qps)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);
            auto r_dphi_tmp = cb.eval_gradients(qp.first);
            auto r_dphi = r_dphi_tmp.block(0, 0, rbs, 2);
            gr_rhs.block(0, cbs+i*fbs, rbs, fbs) += qp.second * (r_dphi * n) * f_phi.transpose();
            gr_rhs.block(0, 0, rbs, cbs) -= qp.second * (r_dphi * n) * c_phi.transpose();
        }
    }

    Matrix<T, Dynamic, Dynamic> oper = gr_lhs.llt().solve(gr_rhs);
    Matrix<T, Dynamic, Dynamic> data = gr_rhs.transpose() * oper;
    return std::make_pair(oper, data);
}

template<typename T, size_t ET, typename Function>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, Dynamic>  >
make_hho_laplacian_interface(const cuthho_mesh<T, ET>& msh,
    const typename cuthho_mesh<T, ET>::cell_type& cl,
    const Function& level_set_function, hho_degree_info& di, const params<T>& parms = params<T>())
{

    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    auto recdeg = di.reconstruction_degree();
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    std::cout<<"make_hho_laplacian_interface: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    cell_basis<cuthho_mesh<T, ET>,T>     cb(msh, cl, recdeg);

    auto rbs = cell_basis<cuthho_mesh<T, ET>,T>::size(recdeg);
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> stiff = Matrix<T, Dynamic, Dynamic>::Zero(2*rbs, 2*rbs);
    Matrix<T, Dynamic, Dynamic> gr_lhs = Matrix<T, Dynamic, Dynamic>::Zero(2*rbs, 2*rbs);
    Matrix<T, Dynamic, Dynamic> gr_rhs = Matrix<T, Dynamic, Dynamic>::Zero(2*rbs, 2*(cbs + num_faces*fbs));

    /* Cell term (cut) */

    auto qps_n = integrate(msh, cl, 2*recdeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : qps_n)
    {
        auto dphi = cb.eval_gradients(qp.first);
        stiff.block(0,0,rbs,rbs) += parms.kappa_1 * qp.second * dphi * dphi.transpose();
    }

    auto qps_p = integrate(msh, cl, 2*recdeg, element_location::IN_POSITIVE_SIDE);
    for (auto& qp : qps_p)
    {
        auto dphi = cb.eval_gradients(qp.first);
        stiff.block(rbs,rbs,rbs,rbs) += parms.kappa_2 * qp.second * dphi * dphi.transpose();
    }

    auto hT = diameter(msh, cl);

    /* Interface term */
    auto iqps = integrate_interface(msh, cl, 2*recdeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        auto phi        = cb.eval_basis(qp.first);
        auto dphi       = cb.eval_gradients(qp.first);
        Matrix<T,2,1> n = level_set_function.normal(qp.first);

        Matrix<T, Dynamic, Dynamic> a = parms.kappa_1 * qp.second * phi * (dphi * n).transpose();
        Matrix<T, Dynamic, Dynamic> b = parms.kappa_1 * qp.second * (dphi * n) * phi.transpose();
        Matrix<T, Dynamic, Dynamic> c = parms.kappa_1 * qp.second * phi * phi.transpose() * cell_eta(msh, cl) / hT;

        stiff.block(  0,   0, rbs, rbs) -= a;
        stiff.block(rbs,   0, rbs, rbs) += a;

        stiff.block(  0,   0, rbs, rbs) -= b;
        stiff.block(  0, rbs, rbs, rbs) += b;

        stiff.block(  0,   0, rbs, rbs) += c;
        stiff.block(  0, rbs, rbs, rbs) -= c;
        stiff.block(rbs,   0, rbs, rbs) -= c;
        stiff.block(rbs, rbs, rbs, rbs) += c;

    }

    gr_lhs = stiff;
    gr_rhs.block(0,   0, 2*rbs, cbs) = stiff.block(0,   0, 2*rbs, cbs);
    gr_rhs.block(0, cbs, 2*rbs, cbs) = stiff.block(0, rbs, 2*rbs, cbs);

    auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        auto fc = fcs[i];
        auto n = ns[i];

        // face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        cut_face_basis<cuthho_mesh<T, ET>,T> fb_n(msh, fc, facdeg,
                                                  element_location::IN_NEGATIVE_SIDE);
        cut_face_basis<cuthho_mesh<T, ET>,T> fb_p(msh, fc, facdeg,
                                                  element_location::IN_POSITIVE_SIDE);
        /* Terms on faces */
        auto qps_n = integrate(msh, fc, 2*recdeg, element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : qps_n)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb_n.eval_basis(qp.first);
            auto r_dphi = cb.eval_gradients(qp.first);

            gr_rhs.block(0, 0, rbs, cbs) -= parms.kappa_1 * qp.second * (r_dphi * n) * c_phi.transpose();
            size_t col_ofs = 2*cbs + i*fbs;
            gr_rhs.block(0, col_ofs, rbs, fbs) += parms.kappa_1 * qp.second * (r_dphi * n) * f_phi.transpose();
        }

        auto qps_p = integrate(msh, fc, 2*recdeg, element_location::IN_POSITIVE_SIDE);
        for (auto& qp : qps_p)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb_p.eval_basis(qp.first);
            auto r_dphi = cb.eval_gradients(qp.first);

            gr_rhs.block(rbs, cbs, rbs, cbs) -= parms.kappa_2 * qp.second * (r_dphi * n) * c_phi.transpose();
            size_t col_ofs = 2*cbs + fbs*fcs.size() + i*fbs;
            gr_rhs.block(rbs, col_ofs, rbs, fbs) += parms.kappa_2 * qp.second * (r_dphi * n) * f_phi.transpose();
        }
    }

    Matrix<T, Dynamic, Dynamic> oper = gr_lhs.ldlt().solve(gr_rhs);
    Matrix<T, Dynamic, Dynamic> data = gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}

////////////////////////////////////////////////// PARAMETRIC INTERFACE
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, Dynamic>  >
make_hho_laplacian_ref_pts(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                   const Function& level_set_function, hho_degree_info& di,
                   element_location where)
{

    if ( !is_cut(msh, cl) )
        return make_hho_laplacian(msh, cl, di);

    auto recdeg = di.reconstruction_degree();
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    cell_basis<cuthho_mesh<T, ET>,T>     cb(msh, cl, recdeg);

    auto rbs = cell_basis<cuthho_mesh<T, ET>,T>::size(recdeg);
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> stiff = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);
    Matrix<T, Dynamic, Dynamic> gr_lhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);
    Matrix<T, Dynamic, Dynamic> gr_rhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs, cbs + num_faces*fbs);

    /* Cell term (cut) */
    auto qps = integrate(msh, cl, 2*recdeg, where);
    for (auto& qp : qps)
    {
        auto dphi = cb.eval_gradients(qp.first);
        stiff += qp.second * dphi * dphi.transpose();
    }

    auto hT = diameter(msh, cl);

    /* Interface term */
    auto msh_int = cl.user_data.integration_msh ;
    size_t degree_curve = msh_int.degree_curve ;
    Interface_parametrisation_mesh1d curve(degree_curve) ;
    size_t degree = 2*recdeg;
    //    degree += 2*degree_curve ; // TO BE CHECKED
    auto qpsi = edge_quadrature<T>(degree);
        
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
        auto pts = points(msh_int,msh_int.cells[i_cell]);
        for(auto& qp:qpsi)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = curve(t , pts , degree_curve ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = curve.jacobian( t , pts , degree_curve ) ;
            auto w = 0.5 * qp.second * jacobian ;
                
                
                
            auto phi    = cb.eval_basis(pt);
            auto dphi   = cb.eval_gradients(pt);

            Matrix<T,2,1> n = curve.normal(t , pts , degree_curve ) ;
//            Matrix<T,2,1> n      = level_set_function.normal(qp.first);

            stiff -= w * phi * (dphi * n).transpose();
            stiff -= w * (dphi * n) * phi.transpose();
            stiff += w * phi * phi.transpose() * cell_eta(msh, cl) / hT;
                
        }
    }
    
    gr_lhs.block(0, 0, rbs, rbs) = stiff;
    gr_rhs.block(0, 0, rbs, cbs) = stiff.block(0, 0, rbs, cbs);

    auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        auto fc = fcs[i];
        auto n = ns[i];

        // face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);
        /* Terms on faces */
        auto qps = integrate(msh, fc, 2*recdeg, where);
        for (auto& qp : qps)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);
            auto r_dphi_tmp = cb.eval_gradients(qp.first);
            auto r_dphi = r_dphi_tmp.block(0, 0, rbs, 2);
            gr_rhs.block(0, cbs+i*fbs, rbs, fbs) += qp.second * (r_dphi * n) * f_phi.transpose();
            gr_rhs.block(0, 0, rbs, cbs) -= qp.second * (r_dphi * n) * c_phi.transpose();
        }
    }

    Matrix<T, Dynamic, Dynamic> oper = gr_lhs.llt().solve(gr_rhs);
    Matrix<T, Dynamic, Dynamic> data = gr_rhs.transpose() * oper;
    return std::make_pair(oper, data);
}

template<typename T, size_t ET, typename Function>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, Dynamic>  >
make_hho_laplacian_ref_pts_cont(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, const Function& parametric_interface, hho_degree_info& di,
                   element_location where)
{

    if ( !is_cut(msh, cl) )
        return make_hho_laplacian(msh, cl, di);

    auto recdeg = di.reconstruction_degree();
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    cell_basis<cuthho_mesh<T, ET>,T>     cb(msh, cl, recdeg);

    auto rbs = cell_basis<cuthho_mesh<T, ET>,T>::size(recdeg);
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> stiff = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);
    Matrix<T, Dynamic, Dynamic> gr_lhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);
    Matrix<T, Dynamic, Dynamic> gr_rhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs, cbs + num_faces*fbs);

    /* Cell term (cut) */
    auto qps = integrate(msh, cl, 2*recdeg, where);
    for (auto& qp : qps)
    {
        auto dphi = cb.eval_gradients(qp.first);
        stiff += qp.second * dphi * dphi.transpose();
    }

    auto hT = diameter(msh, cl);

    /* Interface term */
    auto global_cells_i = parametric_interface.get_global_cells_interface(msh , cl);
    auto msh_int = cl.user_data.integration_msh ;
//    size_t degree_curvature = parametric_interface.der_degree ;

    size_t degree = 2*recdeg;
    degree += (parametric_interface.der_degree + parametric_interface.degree_det ); // TO BE CHECKED
    auto qpsi = edge_quadrature<T>(degree);
        
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
//        auto pts = points(msh_int,msh_int.cells[i_cell]);
        size_t global_cl_i = global_cells_i[i_cell] ;
        for(auto& qp:qpsi)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = parametric_interface(t , global_cl_i ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = parametric_interface.jacobian_cont( t , global_cl_i ) ;
            auto w = 0.5 * qp.second * jacobian ;
                
                
                
            auto phi    = cb.eval_basis(pt);
            auto dphi   = cb.eval_gradients(pt);

            Matrix<T,2,1> n = parametric_interface.normal_cont(t , global_cl_i ) ;


            stiff -= w * phi * (dphi * n).transpose();
            stiff -= w * (dphi * n) * phi.transpose();
            stiff += w * phi * phi.transpose() * cell_eta(msh, cl) / hT;
                
        }
    }
    
    gr_lhs.block(0, 0, rbs, rbs) = stiff;
    gr_rhs.block(0, 0, rbs, cbs) = stiff.block(0, 0, rbs, cbs);

    auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        auto fc = fcs[i];
        auto n = ns[i];

        // face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);
        /* Terms on faces */
        auto qps = integrate(msh, fc, 2*recdeg, where);
        for (auto& qp : qps)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);
            auto r_dphi_tmp = cb.eval_gradients(qp.first);
            auto r_dphi = r_dphi_tmp.block(0, 0, rbs, 2);
            gr_rhs.block(0, cbs+i*fbs, rbs, fbs) += qp.second * (r_dphi * n) * f_phi.transpose();
            gr_rhs.block(0, 0, rbs, cbs) -= qp.second * (r_dphi * n) * c_phi.transpose();
        }
    }

    Matrix<T, Dynamic, Dynamic> oper = gr_lhs.llt().solve(gr_rhs);
    Matrix<T, Dynamic, Dynamic> data = gr_rhs.transpose() * oper;
    return std::make_pair(oper, data);
}

//////////////////////////////////////// STABILIZATION ////////////////////////////////////////
template<typename T, size_t ET, typename Function>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_stabilization_interface(const cuthho_mesh<T, ET>& msh,
                                 const typename cuthho_mesh<T, ET>::cell_type& cl,
                                 const Function& level_set_function,
                                 const hho_degree_info& di,
                                 const params<T>& parms = params<T>(), bool scaled_Q = true)
{
    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut ...");

    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> data
        = Matrix<T, Dynamic, Dynamic>::Zero(2*cbs+2*num_faces*fbs, 2*cbs+2*num_faces*fbs);
    // Matrix<T, Dynamic, Dynamic> If = Matrix<T, Dynamic, Dynamic>::Identity(fbs, fbs);

    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);

    //auto hT = diameter(msh, cl);


    const auto stab_n = make_hho_cut_stabilization(msh, cl, di,element_location::IN_NEGATIVE_SIDE);
    const auto stab_p = make_hho_cut_stabilization(msh, cl, di,element_location::IN_POSITIVE_SIDE);

    // cells--cells
    data.block(0, 0, cbs, cbs) += parms.kappa_1 * stab_n.block(0, 0, cbs, cbs);
    data.block(cbs, cbs, cbs, cbs) += parms.kappa_2 * stab_p.block(0, 0, cbs, cbs);
    // cells--faces
    data.block(0, 2*cbs, cbs, num_faces*fbs)
        += parms.kappa_1 * stab_n.block(0, cbs, cbs, num_faces*fbs);
    data.block(cbs, 2*cbs + num_faces*fbs, cbs, num_faces*fbs)
        += parms.kappa_2 * stab_p.block(0, cbs, cbs, num_faces*fbs);
    // faces--cells
    data.block(2*cbs, 0, num_faces*fbs, cbs)
        += parms.kappa_1 * stab_n.block(cbs, 0, num_faces*fbs, cbs);
    data.block(2*cbs + num_faces*fbs, cbs, num_faces*fbs, cbs)
        += parms.kappa_2 * stab_p.block(cbs, 0, num_faces*fbs, cbs);
    // faces--faces
    data.block(2*cbs, 2*cbs, num_faces*fbs, num_faces*fbs)
        += parms.kappa_1 * stab_n.block(cbs, cbs, num_faces*fbs, num_faces*fbs);
    data.block(2*cbs + num_faces*fbs, 2*cbs + num_faces*fbs, num_faces*fbs, num_faces*fbs)
        += parms.kappa_2 * stab_p.block(cbs, cbs, num_faces*fbs, num_faces*fbs);

    return data;
}

template<typename T, size_t ET>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_cut_stabilization(const cuthho_mesh<T, ET>& msh,
                           const typename cuthho_mesh<T, ET>::cell_type& cl,
                           const hho_degree_info& di, element_location where, bool scaled_Q = true) {
    if ( !is_cut(msh, cl) )
        return make_hho_naive_stabilization(msh, cl, di);

    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> data = Matrix<T, Dynamic, Dynamic>::Zero(cbs+num_faces*fbs, cbs+num_faces*fbs);
    Matrix<T, Dynamic, Dynamic>   If = Matrix<T, Dynamic, Dynamic>::Identity(fbs, fbs);

    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    auto hT = diameter(msh, cl);

    for (size_t i = 0; i < num_faces; i++) {
        auto fc = fcs[i];
        // face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        Matrix<T, Dynamic, Dynamic> oper  = Matrix<T, Dynamic, Dynamic>::Zero(fbs, cbs+num_faces*fbs);
        Matrix<T, Dynamic, Dynamic> mass  = Matrix<T, Dynamic, Dynamic>::Zero(fbs, fbs);
        Matrix<T, Dynamic, Dynamic> trace = Matrix<T, Dynamic, Dynamic>::Zero(fbs, cbs);

        oper.block(0, cbs+i*fbs, fbs, fbs) = -If;

        auto qps = integrate(msh, fc, facdeg + celdeg, where);
        for (auto& qp : qps) {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);
            mass += qp.second * f_phi * f_phi.transpose();
            trace += qp.second * f_phi * c_phi.transpose();
        }

        if (qps.size() == 0) /* Avoid to invert a zero matrix */
            continue;


        oper.block(0, 0, fbs, cbs) = mass.ldlt().solve(trace);
        
        if (scaled_Q) {
            data += oper.transpose() * mass * oper * (1./hT);
        }
        else {
            data += oper.transpose() * mass * oper;
        }
    }

    return data;
}

////////////////////////////////////////////////// POLYNOMIAL EXTENSION
// STABILIZATION s° + s^\Gamma
template<typename T, size_t ET>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_stabilization(const cuthho_mesh<T, ET>& msh, std::tuple<double,element_location,std::vector<double>>& PAIRE, 
                                      const hho_degree_info& di, const params<T>& parms = params<T>(),  bool scaled_Q = true) {
    // SUB-CELL INFOS
    auto cell_index = std::get<0>(PAIRE);
    auto loc = std::get<1>(PAIRE);
    auto cl = msh.cells[cell_index];
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg); 
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();
    auto current_dofs = cbs + num_faces*fbs;
    if (is_cut(msh,cl)) 
        current_dofs = 2*current_dofs;
    auto extended_dofs = 2*(cbs + num_faces*fbs);
    auto nb_dp_cells = std::get<2>(PAIRE).size();
    auto local_dofs = current_dofs + nb_dp_cells*extended_dofs; 

    Matrix<T, Dynamic, Dynamic> data = Matrix<T, Dynamic, Dynamic>::Zero(local_dofs, local_dofs);
    Matrix<T, Dynamic, Dynamic> If   = Matrix<T, Dynamic, Dynamic>::Identity(fbs, fbs);

    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    auto h = diameter(msh, cl);

    for (size_t i = 0; i < fcs.size(); i++) {
        auto fc = fcs[i];
        cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, loc);
        Matrix<T, Dynamic, Dynamic> oper = Matrix<T, Dynamic, Dynamic>::Zero(fbs, local_dofs);
        Matrix<T, Dynamic, Dynamic> mass = Matrix<T, Dynamic, Dynamic>::Zero(fbs, fbs);
        Matrix<T, Dynamic, Dynamic> trace = Matrix<T, Dynamic, Dynamic>::Zero(fbs, cbs);

        // FACE UNKNOWNS
        oper.block(0, cbs+i*fbs, fbs, fbs) = -If;

        // TRACE OF CELL UNKNOWNS
        auto qps = integrate(msh, fc, 2*facdeg + 1);
        for (auto& qp : qps) {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);
            mass  += qp.second * f_phi * f_phi.transpose();
            trace += qp.second * f_phi * c_phi.transpose();
        }
        oper.block(0, 0, fbs, cbs) = mass.llt().solve(trace);

        // CONSTRUCTION OF STAB BLOCKS
        if (scaled_Q) {
            data += oper.transpose() * mass * oper * (1.0/h);
        }
        else {
            data += oper.transpose() * mass * oper;
        }
    }

    // AJOUTER LES 4 BLOCKS 
    if (is_cut(msh,cl))
        data.block(0,0,cbs,cbs) += make_hho_cut_interface_penalty(msh, cl, di, 1.0).block(0, 0, cbs, cbs);

    return data;
}

// STABILIZATION s^N
template<typename T, size_t ET>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_ill_dofs_stabilization(const cuthho_mesh<T, ET>& msh, std::tuple<double,element_location,std::vector<double>>& PAIRE, 
                                      const hho_degree_info& di, const params<T>& parms = params<T>(),  bool scaled_Q = true) {

    // SUB-CELL INFOS
    auto cell_index = std::get<0>(PAIRE);
    auto loc = std::get<1>(PAIRE);
    auto dp_cells = std::get<2>(PAIRE);
    auto cl = msh.cells[cell_index];
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg); 
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    auto current_dofs = cbs + num_faces*fbs;
    if (is_cut(msh,cl)) 
        current_dofs = 2*current_dofs;
    auto extended_dofs = 2*(cbs + num_faces*fbs);
    auto nb_dp_cells = dp_cells.size();
    auto local_dofs = current_dofs + nb_dp_cells*extended_dofs; 

    Matrix<T, Dynamic, Dynamic> data = Matrix<T, Dynamic, Dynamic>::Zero(local_dofs, local_dofs);
    Matrix<T, Dynamic, Dynamic> If   = Matrix<T, Dynamic, Dynamic>::Identity(fbs, fbs);
    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);

    // LOOP OVER DEPENDENT CELLS 
    size_t offset_extended_dofs = current_dofs;  
    for (auto &dp_cl : dp_cells) {

        auto dp_cell = msh.cells[dp_cl];
        auto h = diameter(msh, dp_cell);
        fcs = faces(msh, dp_cell);
        num_faces = fcs.size();

        cell_basis<cuthho_mesh<T, ET>,T> cb_dp(msh, dp_cell, celdeg);

        for (size_t i = 0; i < fcs.size(); i++) {

            auto fc = fcs[i];
            cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, loc);
            // Matrix<T, Dynamic, Dynamic> oper = Matrix<T, Dynamic, Dynamic>::Zero(fbs, local_dofs);
            // Matrix<T, Dynamic, Dynamic> mass = Matrix<T, Dynamic, Dynamic>::Zero(fbs, fbs);
            // Matrix<T, Dynamic, Dynamic> trace = Matrix<T, Dynamic, Dynamic>::Zero(fbs, cbs);
            // Matrix<T, Dynamic, Dynamic> trace_dp = Matrix<T, Dynamic, Dynamic>::Zero(fbs, cbs);
                        
            // TRACE OF CELL UNKNOWNS
            auto qps = integrate(msh, fc, 2*celdeg, loc);
            for (auto& qp : qps) {
                auto c_phi    = cb.eval_basis(qp.first);
                auto c_phi_dp = cb_dp.eval_basis(qp.first);
                auto f_phi = fb.eval_basis(qp.first);
                // // // // // // // // // data += ... AJOUTER LES CONTRIB ICI (4 contribs à ajouter)
                // mass  += qp.second * f_phi * f_phi.transpose();
                // trace_loc += qp.second * c_phi * c_phi.transpose();
                // trace_ext -= qp.second * c_phi * c_phi_dp.transpose();
            }

            // // ADDING CONTRIBUTIONS OF THE LOCAL DOFS
            // if (is_cut(msh,cl)) {
            //     if (loc == element_location::IN_NEGATIVE_SIDE) 
            //         data.block(0, 0, fbs, cbs) = mass.llt().solve(trace_loc);
            //     else 
            //         data.block(0, cbs, fbs, cbs) = mass.llt().solve(trace_loc);
            // }
            // else 
            //     data.block(0, 0, fbs, cbs) = mass.llt().solve(trace_loc);
            
            // // ADDING CONTRIBUTION OF THE EXTENDED DOFS
            // if (loc == element_location::IN_NEGATIVE_SIDE) 
            //     oper.block(0, offset_extended_dofs, fbs, cbs) = mass.llt().solve(trace_ext);
            // else 
            //     oper.block(0, offset_extended_dofs+cbs, fbs, cbs) = mass.llt().solve(trace_ext);
                    
            // CONSTRUCTION OF STAB BLOCKS
            // if (scaled_Q)
            //     data += oper.transpose() * mass * oper * (1.0/h);
            // else
            //     data += oper.transpose() * mass * oper;

        }

        // UPDATING OFFSET 
        offset_extended_dofs += extended_dofs;

    }

    return data;

}

////////////////////////////////////////////////// PARAMETRIC INTERFACE
template<typename T, size_t ET , typename INTERFACE>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_cut_parametric_interface_penalty(const cuthho_mesh<T, ET>& msh,
                               const typename cuthho_mesh<T, ET>::cell_type& cl, const INTERFACE& parametric_interface,
                               const hho_degree_info& di, const T eta)
{
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto num_faces = faces(msh, cl).size();

    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);

    Matrix<T, Dynamic, Dynamic> data = Matrix<T, Dynamic, Dynamic>::Zero(cbs+num_faces*fbs, cbs+num_faces*fbs);

    auto hT = diameter(msh, cl);

    auto iqps = integrate_interface(msh, cl,parametric_interface, 2*celdeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto c_phi  = cb.eval_basis(qp.first);

        data.block(0, 0, cbs, cbs) += qp.second * c_phi * c_phi.transpose() * eta / hT;
    }

    return data;
}

template<typename T, size_t ET>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_stabilization_parametric_interface(const cuthho_mesh<T, ET>& msh,
                                 const typename cuthho_mesh<T, ET>::cell_type& cl,
                                 const hho_degree_info& di,
                                 const params<T>& parms = params<T>())
{
    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut ...");

    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, Dynamic> data
        = Matrix<T, Dynamic, Dynamic>::Zero(2*cbs+2*num_faces*fbs, 2*cbs+2*num_faces*fbs);
    // Matrix<T, Dynamic, Dynamic> If = Matrix<T, Dynamic, Dynamic>::Identity(fbs, fbs);

    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);

    //auto hT = diameter(msh, cl);


    const auto stab_n = make_hho_cut_stabilization(msh, cl, di,element_location::IN_NEGATIVE_SIDE);
    const auto stab_p = make_hho_cut_stabilization(msh, cl, di,element_location::IN_POSITIVE_SIDE);

    // cells--cells
    data.block(0, 0, cbs, cbs) += parms.kappa_1 * stab_n.block(0, 0, cbs, cbs);
    data.block(cbs, cbs, cbs, cbs) += parms.kappa_2 * stab_p.block(0, 0, cbs, cbs);
    // cells--faces
    data.block(0, 2*cbs, cbs, num_faces*fbs)
        += parms.kappa_1 * stab_n.block(0, cbs, cbs, num_faces*fbs);
    data.block(cbs, 2*cbs + num_faces*fbs, cbs, num_faces*fbs)
        += parms.kappa_2 * stab_p.block(0, cbs, cbs, num_faces*fbs);
    // faces--cells
    data.block(2*cbs, 0, num_faces*fbs, cbs)
        += parms.kappa_1 * stab_n.block(cbs, 0, num_faces*fbs, cbs);
    data.block(2*cbs + num_faces*fbs, cbs, num_faces*fbs, cbs)
        += parms.kappa_2 * stab_p.block(cbs, 0, num_faces*fbs, cbs);
    // faces--faces
    data.block(2*cbs, 2*cbs, num_faces*fbs, num_faces*fbs)
        += parms.kappa_1 * stab_n.block(cbs, cbs, num_faces*fbs, num_faces*fbs);
    data.block(2*cbs + num_faces*fbs, 2*cbs + num_faces*fbs, num_faces*fbs, num_faces*fbs)
        += parms.kappa_2 * stab_p.block(cbs, cbs, num_faces*fbs, num_faces*fbs);

    return data;
}

//////////////////////////////////////// PENALTY TERM ////////////////////////////////////////
template<typename T, size_t ET>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_cut_interface_penalty(const cuthho_mesh<T, ET>& msh,
                               const typename cuthho_mesh<T, ET>::cell_type& cl,
                               const hho_degree_info& di, const T eta, bool scaled_Q = true)
{
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto num_faces = faces(msh, cl).size();

    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);

    Matrix<T, Dynamic, Dynamic> data = Matrix<T, Dynamic, Dynamic>::Zero(cbs+num_faces*fbs, cbs+num_faces*fbs);

    auto hT = diameter(msh, cl);

    auto iqps = integrate_interface(msh, cl, 2*celdeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps) {
        const auto c_phi  = cb.eval_basis(qp.first);
        if (scaled_Q)
            data.block(0, 0, cbs, cbs) += qp.second * c_phi * c_phi.transpose() * eta / hT;
        else 
            data.block(0, 0, cbs, cbs) += qp.second * c_phi * c_phi.transpose() * eta;
    }

    return data;
}

//////////////////////////////////////// GRADREC ////////////////////////////////////////
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_vector(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, const Function& level_set_function, const hho_degree_info& di, element_location where, const T coeff) {

    if ( !is_cut(msh, cl) )
        return make_hho_gradrec_vector(msh, cl, di);

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    cell_basis<cuthho_mesh<T, ET>,T>        cb(msh, cl, celdeg);
    vector_cell_basis<cuthho_mesh<T, ET>,T> gb(msh, cl, graddeg);


    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
    for (auto& qp : qps)
    {
        const auto c_dphi = cb.eval_gradients(qp.first);
        const auto g_phi  = gb.eval_basis(qp.first);

        gr_lhs.block(0, 0, gbs, gbs) += qp.second * g_phi * g_phi.transpose();
        gr_rhs.block(0, 0, gbs, cbs) += qp.second * g_phi * c_dphi.transpose();
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        // face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const vector_type c_phi      = cb.eval_basis(qp.first);
            const vector_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const vector_type qp_g_phi_n = qp.second * g_phi * n;

            gr_rhs.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            gr_rhs.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }


    matrix_type interface_term = matrix_type::Zero(gbs, cbs);
    const auto iqps = integrate_interface(msh, cl, celdeg + graddeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps) {
        const auto c_phi        = cb.eval_basis(qp.first);
        const auto g_phi        = gb.eval_basis(qp.first);

        Matrix<T,2,1> n = level_set_function.normal(qp.first);
        const vector_type qp_g_phi_n = qp.second * g_phi * n;

        interface_term -= qp_g_phi_n * c_phi.transpose();
    }
    gr_rhs.block(0, 0, gbs, cbs) += coeff * interface_term;

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}

template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_vector_interface(const cuthho_mesh<T, ET>& msh,
                                  const typename cuthho_mesh<T, ET>::cell_type& cl,
                                  const Function& level_set_function, const hho_degree_info& di,
                                  element_location where, T coeff)
{

    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    // std::cout<<"make_hho_gradrec_vector_interface: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    cell_basis<cuthho_mesh<T, ET>,T>            cb(msh, cl, celdeg);
    vector_cell_basis<cuthho_mesh<T, ET>,T>     gb(msh, cl, graddeg);
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type       rhs_tmp = matrix_type::Zero(gbs, cbs + num_faces * fbs);
    matrix_type        gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type        gr_rhs = matrix_type::Zero(gbs, 2*cbs + 2*num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
    for (auto& qp : qps)
    {
        const auto c_dphi = cb.eval_gradients(qp.first);
        const auto g_phi  = gb.eval_basis(qp.first);

        gr_lhs.block(0, 0, gbs, gbs) += qp.second * g_phi * g_phi.transpose();
        rhs_tmp.block(0, 0, gbs, cbs) += qp.second * g_phi * c_dphi.transpose();
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++) {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const vector_type c_phi      = cb.eval_basis(qp.first);
            const vector_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const vector_type qp_g_phi_n = qp.second * g_phi * n;

            rhs_tmp.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            rhs_tmp.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }

    // term on the interface
    matrix_type        interface_term = matrix_type::Zero(gbs, 2*cbs);
    const auto iqps = integrate_interface(msh, cl, celdeg + graddeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps) {
        const auto c_phi        = cb.eval_basis(qp.first);
        const auto g_phi        = gb.eval_basis(qp.first);

        Matrix<T,2,1> n = level_set_function.normal(qp.first);
        const vector_type qp_g_phi_n = qp.second * g_phi * n;

        interface_term.block(0 , 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        interface_term.block(0 , cbs, gbs, cbs) += qp_g_phi_n * c_phi.transpose();
    }
    gr_rhs.block(0, 0, gbs, 2*cbs) += coeff * interface_term;

    // other terms
    if(where == element_location::IN_NEGATIVE_SIDE) {
        gr_rhs.block(0, 0, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs, gbs, num_faces*fbs)
            += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }
    else if( where == element_location::IN_POSITIVE_SIDE) {
        gr_rhs.block(0, cbs, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs + num_faces*fbs, gbs, num_faces*fbs)
                     += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}

template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_mixed_vector_interface(const cuthho_mesh<T, ET>& msh,
                                  const typename cuthho_mesh<T, ET>::cell_type& cl,
                                  const Function& level_set_function, const hho_degree_info& di,
                                  element_location where, T coeff) {

    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    cell_basis<cuthho_mesh<T, ET>,T>            cb(msh, cl, celdeg);
    vector_cell_basis<cuthho_mesh<T, ET>,T>     gb(msh, cl, graddeg);


    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type rhs_tmp = matrix_type::Zero(gbs, cbs + num_faces * fbs);
    matrix_type gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type gr_rhs = matrix_type::Zero(gbs, 2*cbs + 2*num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
    for (auto& qp : qps) {
        const auto c_dphi = cb.eval_gradients(qp.first);
        const auto g_phi  = gb.eval_basis(qp.first);
        gr_lhs.block(0, 0, gbs, gbs) += qp.second * g_phi * g_phi.transpose();
        rhs_tmp.block(0, 0, gbs, cbs) += qp.second * g_phi * c_dphi.transpose();
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++) {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        // face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg);
        cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);
        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f) {
            const vector_type c_phi      = cb.eval_basis(qp.first);
            const vector_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const vector_type qp_g_phi_n = qp.second * g_phi * n;

            rhs_tmp.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            rhs_tmp.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }

    // term on the interface
    matrix_type interface_term = matrix_type::Zero(gbs, 2*cbs);
    const auto iqps = integrate_interface(msh, cl, celdeg + graddeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps) {
        const auto c_phi        = cb.eval_basis(qp.first);
        const auto g_phi        = gb.eval_basis(qp.first);
        Matrix<T,2,1> n = level_set_function.normal(qp.first);
        const vector_type qp_g_phi_n = qp.second * g_phi * n;
        interface_term.block(0 , 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        interface_term.block(0 , cbs, gbs, cbs) += qp_g_phi_n * c_phi.transpose();
    }
    gr_rhs.block(0, 0, gbs, 2*cbs) += coeff * interface_term;

    // other terms
    if(where == element_location::IN_NEGATIVE_SIDE) {
        gr_rhs.block(0, 0, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs, gbs, num_faces*fbs)
            += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }
    else if( where == element_location::IN_POSITIVE_SIDE) {
        gr_rhs.block(0, cbs, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs + num_faces*fbs, gbs, num_faces*fbs)
                     += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }
    
    auto vec_cell_size = 2*gbs;
    auto nrows = gr_rhs.cols()+vec_cell_size;
    auto ncols = gr_rhs.cols()+vec_cell_size;
    
    // Shrinking data
    matrix_type data_mixed = matrix_type::Zero(nrows,ncols);
    if (where == element_location::IN_NEGATIVE_SIDE) {
        data_mixed.block(0, vec_cell_size, gbs, ncols-vec_cell_size) = -gr_rhs;
        data_mixed.block(vec_cell_size, 0, nrows-vec_cell_size, gbs) = gr_rhs.transpose();
    }
    else if ( where == element_location::IN_POSITIVE_SIDE) {
        data_mixed.block(gbs, vec_cell_size, gbs, ncols-vec_cell_size) = -gr_rhs;
        data_mixed.block(vec_cell_size, gbs, nrows-vec_cell_size, gbs) = gr_rhs.transpose();
    }

    matrix_type oper = gr_lhs.llt().solve(gr_rhs);
    return std::make_pair(oper, data_mixed);
    
}

////////////////////////////////////////////////// POLYNOMIAL EXTENSION
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_vector_POK(const cuthho_mesh<T, ET>& msh, std::tuple<double,element_location,std::vector<double>>& P_OK, const hho_degree_info& di, const Function& level_set_function) {

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    // SUB-CELL INFOS
    auto cell_index = std::get<0>(P_OK);
    auto cl = msh.cells[cell_index];
    auto loc = std::get<1>(P_OK);

    // DISCRETIZATION INFOS
    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();
    cell_basis<cuthho_mesh<T, ET>,T>        cb(msh, cl, celdeg);
    vector_cell_basis<cuthho_mesh<T, ET>,T> gb(msh, cl, graddeg);
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);
    auto fcs = faces(msh, cl);
    auto ns  = normals(msh, cl);
    auto num_faces = fcs.size();
    auto uncut_dofs = cbs + num_faces*fbs;
    auto current_dofs = cbs + num_faces*fbs;
    if (is_cut(msh,cl)) 
        current_dofs = 2*current_dofs;
    auto extended_dofs = 2*(cbs + num_faces*fbs);
    auto dp_cells = std::get<2>(P_OK);
    auto local_dofs = current_dofs + dp_cells.size()*extended_dofs; 
    
    matrix_type rhs_tmp = matrix_type::Zero(gbs, uncut_dofs);
    matrix_type gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type gr_rhs = matrix_type::Zero(gbs, local_dofs);

    // CELLS CONTRIBUTION
    if (celdeg > 0) {
        const auto qps = integrate(msh, cl, celdeg-1 + facdeg, loc);
        for (auto& qp : qps) {
            const auto c_dphi = cb.eval_gradients(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);
            gr_lhs.block(0, 0, gbs, gbs)  += qp.second*g_phi*g_phi.transpose();  // Mass matrix
            rhs_tmp.block(0, 0, gbs, cbs) += qp.second*g_phi*c_dphi.transpose(); // Current cell unknown of a given side
        }
    }

    // FACES CONTRIBUTION
    for (size_t i = 0; i < fcs.size(); i++) {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, loc);
        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), loc);
        for (auto& qp : qps_f) {
            const vector_type c_phi      = cb.eval_basis(qp.first);
            const vector_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const vector_type qp_g_phi_n = qp.second*g_phi*n;
            rhs_tmp.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n*f_phi.transpose();
            rhs_tmp.block(0, 0, gbs, cbs) -= qp_g_phi_n*c_phi.transpose();
        }
    }

    // ADDING CONTRIBUTIONS OF CURRENT CELL  
    if (is_cut(msh,cl)) {
        if (loc == element_location::IN_NEGATIVE_SIDE) {
            gr_rhs.block(0, 0, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
            gr_rhs.block(0, 2*cbs, gbs, num_faces*fbs) += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
        }
        else {
            gr_rhs.block(0, cbs, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
            gr_rhs.block(0, 2*cbs + num_faces*fbs, gbs, num_faces*fbs) += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
        }
    }
    else {
        gr_rhs.block(0, 0, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, cbs, gbs, num_faces*fbs) += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }

    // INTERFACE TERM CURRENT CELL
    if (is_cut(msh,cl) && loc == element_location::IN_NEGATIVE_SIDE) {
        matrix_type interface_term = matrix_type::Zero(gbs, 2*cbs);
        const auto iqps = integrate_interface(msh, cl, celdeg+graddeg, element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : iqps) {
            const auto c_phi = cb.eval_basis(qp.first);
            const auto g_phi = gb.eval_basis(qp.first);
            Matrix<T,2,1> n = level_set_function.normal(qp.first);
            const vector_type qp_g_phi_n = qp.second*g_phi*n;
            interface_term.block(0 , 0, gbs, cbs)   -= qp_g_phi_n*c_phi.transpose();
            interface_term.block(0 , cbs, gbs, cbs) += qp_g_phi_n*c_phi.transpose();
        }
        gr_rhs.block(0, 0, gbs, cbs)   += interface_term.block(0, 0, gbs, cbs);   // NEG CELL DOFS
        gr_rhs.block(0, cbs, gbs, cbs) += interface_term.block(0, cbs, gbs, cbs); // POS CELL DOFS
    }

    // LOOP OVER DEPENDENT CELLS 
    size_t offset_extended_dofs = current_dofs;  
    for (auto &dp_cl : dp_cells) {
        
        auto dp_cell = msh.cells[dp_cl];
        rhs_tmp = matrix_type::Zero(gbs, uncut_dofs);  
        cell_basis<cuthho_mesh<T, ET>,T> cb_dp(msh, dp_cell, celdeg);

        // CELL CONTRIBUTION
        if(celdeg > 0) {
            auto qps = integrate(msh, dp_cell, celdeg-1 + facdeg, loc);
            for (auto& qp : qps) {
                const auto c_dphi = cb.eval_gradients(qp.first);
                const auto g_phi  = gb.eval_basis(qp.first);
                rhs_tmp.block(0, 0, gbs, cbs) += qp.second * g_phi * c_dphi.transpose(); // CURRENT CELL UNKNOWNS
            }
        }

        // FACE CONTRIBUTION
        fcs = faces(msh, dp_cell);
        ns  = normals(msh, dp_cell);
        num_faces = fcs.size();
        for (size_t i=0; i < num_faces; i++) {
            const auto fc = fcs[i];
            const auto n  = ns[i];
            cut_face_basis<cuthho_mesh<T, ET>,T> fb_dp(msh, fc, facdeg, loc);
            const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), loc);
            for (auto& qp : qps_f) {
                const vector_type c_phi       = cb_dp.eval_basis(qp.first);
                const vector_type f_phi       = fb_dp.eval_basis(qp.first);
                const auto        g_phi       = gb.eval_basis(qp.first);
                const vector_type qp_g_phi_n  = qp.second * g_phi * n;
                rhs_tmp.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();           // EXTENDED CELL UNKNOWNS
                rhs_tmp.block(0, cbs + i*fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose(); // EXTENDED FACE UNKNOWNS
            }
        }

        // ADDING EXTENDED DOFS CONTRIBUTION
        if (loc == element_location::IN_NEGATIVE_SIDE) {
            gr_rhs.block(0, offset_extended_dofs        , gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
            gr_rhs.block(0, offset_extended_dofs + 2*cbs, gbs, num_faces*fbs) += rhs_tmp.block(0, 0, gbs, num_faces*fbs);
        }
        if (loc == element_location::IN_POSITIVE_SIDE) {
            gr_rhs.block(0, offset_extended_dofs + cbs                  , gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
            gr_rhs.block(0, offset_extended_dofs + 2*cbs + num_faces*fbs, gbs, num_faces*fbs) += rhs_tmp.block(0, 0, gbs, num_faces*fbs);
        }

        // INTERFACE TERMS OF THE EXTENDED CELLS
        if (dp_cell.user_data.location == element_location::IN_NEGATIVE_SIDE) {
            matrix_type interface_term = matrix_type::Zero(gbs, 2*cbs);
            const auto iqps = integrate_interface(msh, dp_cell, celdeg+graddeg, element_location::IN_NEGATIVE_SIDE);
            for (auto& qp : iqps) {
                const auto c_phi = cb_dp.eval_basis(qp.first);
                const auto g_phi = gb.eval_basis(qp.first);
                Matrix<T,2,1> n = level_set_function.normal(qp.first);
                const vector_type qp_g_phi_n = qp.second*g_phi*n;
                interface_term.block(0 , 0, gbs, cbs)   -= qp_g_phi_n*c_phi.transpose();
                interface_term.block(0 , cbs, gbs, cbs) += qp_g_phi_n*c_phi.transpose();
            }
            gr_rhs.block(0, offset_extended_dofs      , gbs, cbs) += interface_term.block(0, 0, gbs, cbs);   
            gr_rhs.block(0, offset_extended_dofs + cbs, gbs, cbs) += interface_term.block(0, cbs, gbs, cbs); 
        }

        // UPDATING THE OFFSET OF DOFS OF THE EXTENDED CELLS
        offset_extended_dofs += extended_dofs;

    }

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = gr_rhs.transpose()*oper;

    return std::make_pair(oper, data);

}

template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_vector_PKO(const cuthho_mesh<T, ET>& msh, std::tuple<double,element_location,std::vector<double>>& P_KO, const hho_degree_info& di, const Function& level_set_function) {

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    // SUB-CELL INFOS
    auto cell_index = std::get<0>(P_KO);
    auto cl = msh.cells[cell_index];
    auto loc = std::get<1>(P_KO);

    // DISCRETIZATION INFOS
    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();
    cell_basis<cuthho_mesh<T, ET>,T>        cb(msh, cl, celdeg);
    vector_cell_basis<cuthho_mesh<T, ET>,T> gb(msh, cl, graddeg);
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);
    auto fcs = faces(msh, cl);
    auto ns  = normals(msh, cl);
    auto num_faces = fcs.size();
    auto uncut_dofs = cbs + num_faces*fbs;
    auto current_dofs = cbs + num_faces*fbs;
    if (is_cut(msh,cl)) 
        current_dofs = 2*current_dofs;
    auto extended_dofs = 2*(cbs + num_faces*fbs);
    auto dp_cells = std::get<2>(P_KO);
    auto local_dofs = current_dofs + dp_cells.size()*extended_dofs; 
    
    matrix_type rhs_tmp = matrix_type::Zero(gbs, uncut_dofs);
    matrix_type gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type gr_rhs = matrix_type::Zero(gbs, local_dofs);

    // CELLS CONTRIBUTION
    if (celdeg > 0) {
        const auto qps = integrate(msh, cl, celdeg-1 + facdeg, loc);
        for (auto& qp : qps) {
            const auto c_dphi = cb.eval_gradients(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);
            gr_lhs.block(0, 0, gbs, gbs)  += qp.second*g_phi*g_phi.transpose();  // Mass matrix
            rhs_tmp.block(0, 0, gbs, cbs) += qp.second*g_phi*c_dphi.transpose(); // Current cell unknown of a given side
        }
    }

    // ADDING CONTRIBUTIONS OF CURRENT CELL  
    if (loc == element_location::IN_NEGATIVE_SIDE) {
        gr_rhs.block(0, 0, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs, gbs, num_faces*fbs) += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }
    else {
        gr_rhs.block(0, cbs, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs + num_faces*fbs, gbs, num_faces*fbs) += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = gr_rhs.transpose()*oper;

    return std::make_pair(oper, data);

}

//////////////////////////////////////// NITSCHE TERMS ////////////////////////////////////////
template<typename T, size_t ET, typename Function>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_Nitsche(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
             const Function& level_set_function, hho_degree_info& di)
{
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();
    std::cout<<"make_Nitsche: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    auto cbs = cb.size();
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    auto size_tot = cbs + num_faces * fbs;

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(size_tot, size_tot);

    if( !is_cut(msh, cl) )
        return ret;

    auto iqps = integrate_interface(msh, cl, 2*celdeg-1, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto c_phi  = cb.eval_basis(qp.first);
        const auto c_dphi  = cb.eval_gradients(qp.first);
        const Matrix<T,2,1> n = level_set_function.normal(qp.first);
        const auto c_dphi_n = c_dphi * n;

        ret.block(0, 0, cbs, cbs) -= qp.second * c_phi * c_dphi_n.transpose();
        ret.block(0, 0, cbs, cbs) -= qp.second * c_dphi_n * c_phi.transpose();
    }

    return ret;
}

// NON SYMMETRIC NITSCHE
template<typename T, size_t ET, typename Function>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_NS_Nitsche(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                const Function& level_set_function, hho_degree_info& di)
{
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();
    std::cout<<"make_NS_Nitsche: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    auto cbs = cb.size();
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    auto size_tot = cbs + num_faces * fbs;

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero(size_tot, size_tot);

    if( !is_cut(msh, cl) )
        return ret;

    auto iqps = integrate_interface(msh, cl, 2*celdeg-1, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto c_phi  = cb.eval_basis(qp.first);
        const auto c_dphi  = cb.eval_gradients(qp.first);
        const Matrix<T,2,1> n = level_set_function.normal(qp.first);
        const auto c_dphi_n = c_dphi * n;

        ret.block(0, 0, cbs, cbs) += qp.second * c_phi * c_dphi_n.transpose();
    }

    return ret;
}

//////////////////////////////////////// RHS TERMS ////////////////////////////////////////
// MAKE VOLUMIC RHS
template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
         size_t degree, const F1& f, const element_location where)
{
    if ( location(msh, cl) == where )
        return make_rhs(msh, cl, degree, f);

    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
    auto qps = integrate(msh, cl, 2*degree, where);
    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi * f(qp.first);
    }
    return ret;
}

// MAKE RHS ON FACES
template<typename T, size_t ET, typename Function>
Matrix<T, Dynamic, 1>
make_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc,
         size_t degree, element_location where, const Function& f)
{
    cut_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, degree, where);
    auto fbs = fb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(fbs);

    auto qps = integrate(msh, fc, 2*degree, where);

    for (auto& qp : qps)
    {
        auto phi = fb.eval_basis(qp.first);
        ret += qp.second * phi * f(qp.first);
    }

    return ret;
}

// MAKE RHS PENALTY
template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_rhs_penalty(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                 size_t degree, const F1& g, T eta)
{
    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    auto hT = diameter(msh, cl);

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
    auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : qpsi)
    {
        const auto phi = cb.eval_basis(qp.first);

        ret += qp.second * g(qp.first) * phi * eta / hT;
    }

    // return (g , v_T)_{Gamma} * eta/h_T
    return ret;
}


// MAKE GR RHS
template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_GR_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
            size_t degree, const F1& g, const F2& level_set_function,
            Matrix<T, Dynamic, Dynamic> GR)
{
    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    std::cout<<"make_GR_rhs: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    vector_cell_basis<cuthho_mesh<T, ET>,T> gb(msh, cl, degree-1);
    auto gbs = gb.size();

    auto hT = diameter(msh, cl);

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    Matrix<T, Dynamic, 1> source_vect = Matrix<T, Dynamic, 1>::Zero(gbs);

    auto qpsi = integrate_interface(msh, cl, 2*degree-1, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : qpsi)
    {
        const auto n = level_set_function.normal(qp.first);
        const auto g_phi  = gb.eval_basis(qp.first);

        source_vect += qp.second * g(qp.first) * g_phi * n;
    }

    // return -(g , GR(v_T) . n)_{Gamma}
    return -GR.transpose() * source_vect;
}

// MAKE RHS FOR THE FILE CUTHHO_SQUARE.CPP
template<typename T, size_t ET, typename F1, typename F2, typename F3>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
         size_t degree, const F1& f, const element_location where, const F2& level_set_function, const F3& bcs)
{
    if ( location(msh, cl) == where ) {
        return make_rhs(msh, cl, degree, f);
    }
    else if ( location(msh, cl) == element_location::ON_INTERFACE ) {

        cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
        auto cbs = cb.size();
        std::cout<<"make_rhs: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
        auto hT = diameter(msh, cl);

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

        auto qps = integrate(msh, cl, 2*degree, where);
        for (auto& qp : qps)
        {
            auto phi = cb.eval_basis(qp.first);
            ret += qp.second * phi * f(qp.first);
        }


        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            auto dphi = cb.eval_gradients(qp.first);
            auto n = level_set_function.normal(qp.first);

            ret += qp.second * bcs(qp.first) * ( phi * cell_eta(msh, cl)/hT - dphi*n);
        }


        return ret;
    }
    else
    {
        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(degree);
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
        return ret;
    }
}

// MAKE DIRICHLET JUMP
template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_Dirichlet_jump(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                   size_t degree, const element_location where, const F1& level_set_function, 
                    const F2& dir_jump, T eta)
{
    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
    std::cout<<"make_Dirichlet_jump: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    if( location(msh, cl) != element_location::ON_INTERFACE )
        return ret;

    auto hT = diameter(msh, cl);

    if(where == element_location::IN_NEGATIVE_SIDE) {
        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE );

        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            auto dphi = cb.eval_gradients(qp.first);
            auto n = level_set_function.normal(qp.first);

            ret += qp.second * dir_jump(qp.first) * ( phi * eta / hT - dphi*n);
        }
    }
    else if(where == element_location::IN_POSITIVE_SIDE) {
        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE );

        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            ret -= qp.second * dir_jump(qp.first) * phi * eta / hT;
        }
    }
    return ret;
}

template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_flux_jump(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
               size_t degree, const element_location where, const F1& flux_jump)
{
    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
    std::cout<<"make_flux_jump: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    if( location(msh, cl) != element_location::ON_INTERFACE )
        return ret;

    auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE);

    for (auto& qp : qpsi)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * flux_jump(qp.first) * phi;
    }

    return ret;
}

//////////////////////////////////////// MISCELLANEOUS ////////////////////////////////////////
template<typename T, size_t ET, typename Function>
Matrix<T, Dynamic, 1>
project_function(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                 hho_degree_info& hdi, element_location where, const Function& f)
{

    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(hdi.cell_degree());
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(hdi.face_degree());
    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs+num_faces*fbs);

    if (location(msh, cl)!=element_location::ON_INTERFACE && location(msh, cl) != where)
        return ret;

    Matrix<T, Dynamic, Dynamic> cell_mm = make_mass_matrix(msh, cl, hdi.cell_degree(), where);
    Matrix<T, Dynamic, 1> cell_rhs = make_rhs(msh, cl, hdi.cell_degree(), f, where);
    
    ret.block(0, 0, cbs, 1) = cell_mm.llt().solve(cell_rhs);

    for (size_t i = 0; i < num_faces; i++)
    {
        auto fc = fcs[i];

        if ( location(msh, fc) != element_location::ON_INTERFACE &&
             location(msh, fc) != where )
        {
            ret.block(cbs+i*fbs, 0, fbs, 1) = Matrix<T, Dynamic, 1>::Zero(fbs);
        }
        else
        {
            Matrix<T, Dynamic, Dynamic> face_mm = make_mass_matrix(msh, fc, hdi.face_degree(), where);
            Matrix<T, Dynamic, 1> face_rhs = make_rhs(msh, fc, hdi.face_degree(), where, f);
            ret.block(cbs+i*fbs, 0, fbs, 1) = face_mm.llt().solve(face_rhs);
        }
    }

    return ret;
}

////////////////////////////////////////////////// POLYNOMIAL EXTENSION
template<typename T, size_t ET, typename Function>
Matrix<T, Dynamic, 1>
project_function_TOK(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                 hho_degree_info hdi, const Function& f)
{

    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(hdi.cell_degree());
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(hdi.face_degree());
    auto num_faces = faces(msh, cl).size();
    auto local_dofs = cbs+num_faces*fbs;

    auto dp_cells_neg = cl.user_data.dependent_cells_neg;
    auto dp_cells_pos = cl.user_data.dependent_cells_pos;
    auto nb_dp_cells = dp_cells_neg.size() + dp_cells_pos.size();
    size_t extended_dofs = nb_dp_cells*local_dofs;
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(2*local_dofs+extended_dofs);

    //////////////////////////////////////// Projection on local dofs 
    ret.block(0, 0, local_dofs, 1)          = project_function(msh, cl, hdi, element_location::IN_NEGATIVE_SIDE, f);
    ret.block(local_dofs, 0, local_dofs, 1) = project_function(msh, cl, hdi, element_location::IN_POSITIVE_SIDE, f);

    //////////////////////////////////////// Projection on extended dofs 
    size_t i = 1; //cpt
    // Projection on negative dependant cells 
    for (auto &dp_cl : dp_cells_neg) {
        // std::cout << "HELLLLOOOOOO" << std::endl;
        ret.block(i*local_dofs, 0, local_dofs, 1) = project_function(msh, msh.cells[dp_cl], hdi, element_location::IN_NEGATIVE_SIDE, f);
        i = i+1;
    }
    // Projection on positive dependant cells 
    for (auto &dp_cl : cl.user_data.dependent_cells_pos) {
        // std::cout << "COUCOUCOUCOUCOUC" << std::endl;
        ret.block(i*local_dofs, 0, local_dofs, 1) = project_function(msh, msh.cells[dp_cl], hdi, element_location::IN_POSITIVE_SIDE, f);
        i = i+1;
    }

    return ret;
}

template<typename T, size_t ET, typename Function>
Matrix<T, Dynamic, 1>
project_function_TKOi(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                 hho_degree_info hdi, element_location where, const Function& f)
{
    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(hdi.cell_degree());
    auto dp_cell = cl.user_data.paired_cells;
    auto dp_cl = msh.cells[dp_cell];

    // if (where == element_location::IN_NEGATIVE_SIDE) { 
    //     auto offset_cl = offset(msh, cl);
    //     std::cout << "Projection on TKO NEG cell: " << offset_cl << std::endl;
    //     std::cout << "Using of the paired cell info: " << dp_cell << std::endl;
    // }
    // else { 
    //     auto offset_cl = offset(msh, cl);
    //     std::cout << "Projection on TKO POS cell: " << offset_cl << std::endl;
    //     std::cout << "Using of the paired cell info: " << dp_cell << std::endl;
    // }

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    if ( location(msh, cl) != element_location::ON_INTERFACE &&
         location(msh, cl) != where )
        return ret;

    // Projection on cell
    Matrix<T, Dynamic, Dynamic> cell_mm = make_mass_matrix(msh, dp_cl, hdi.cell_degree(), where);
    Matrix<T, Dynamic, 1> cell_rhs = make_rhs(msh, dp_cl, hdi.cell_degree(), f, where);
    ret.block(0, 0, cbs, 1) = cell_mm.llt().solve(cell_rhs);

    return ret;
}


template<typename T, size_t ET, typename Function>
Matrix<T, Dynamic, 1>
project_function_TKOibar(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                 hho_degree_info hdi, element_location where, const Function& f)
{

    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(hdi.cell_degree());
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(hdi.face_degree());
    auto num_faces = faces(msh, cl).size();
    auto local_dofs = cbs+num_faces*fbs;

    auto dp_cells = cl.user_data.dependent_cells_neg;
    if (where == element_location::IN_POSITIVE_SIDE) {
        dp_cells = cl.user_data.dependent_cells_pos;
    }
    auto nb_dp_cells = dp_cells.size();
    size_t extended_dofs = nb_dp_cells*local_dofs;
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(local_dofs+extended_dofs);

    //////////////////////////////////////// Projection on local dofs 
    ret.block(0, 0, local_dofs, 1) = project_function(msh, cl, hdi, where, f);

    //////////////////////////////////////// Projection on extended dofs 
    size_t i = 1; //cpt
    // Projection on dependant cells 
    for (auto &dp_cl : dp_cells) {
        ret.block(i*local_dofs, 0, local_dofs, 1) = project_function(msh, msh.cells[dp_cl], hdi, where, f);
        i = i+1;
    }

    return ret;
}

template<typename T, size_t ET, typename Function>
Matrix<T, Dynamic, 1>
project_function_uncut(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                 hho_degree_info hdi, element_location where, const Function& f)
{
    
    size_t di = 0;

    auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(hdi.cell_degree());
    auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(hdi.face_degree());
    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();
    auto local_dofs = cbs+num_faces*fbs;

    auto dp_cells = cl.user_data.dependent_cells_neg;
    if (where == element_location::IN_POSITIVE_SIDE) {
        dp_cells = cl.user_data.dependent_cells_pos;
    }
    auto nb_dp_cells = dp_cells.size();
    size_t extended_dofs = nb_dp_cells*local_dofs;

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(local_dofs+extended_dofs);

    //////////////////////////////////////// Projection on local dofs 
    // Projection on cell
    Matrix<T, Dynamic, Dynamic> cell_mm = make_mass_matrix(msh, cl, hdi.cell_degree(), di);
    Matrix<T, Dynamic, 1> cell_rhs = make_rhs(msh, cl, hdi.cell_degree(), f, di);
    ret.block(0, 0, cbs, 1) = cell_mm.llt().solve(cell_rhs);
    // Projection on faces
    for (size_t i = 0; i < num_faces; i++)
    {
        auto fc = fcs[i];
        Matrix<T, Dynamic, Dynamic> face_mm = make_mass_matrix(msh, fc, hdi.face_degree(), di);
        Matrix<T, Dynamic, 1> face_rhs = make_rhs(msh, fc, hdi.face_degree(), f, di);
        ret.block(cbs+i*fbs, 0, fbs, 1) = face_mm.llt().solve(face_rhs);
    }

    //////////////////////////////////////// Projection on extended dofs 
    size_t i = 1; //cpt
    for (auto &dp_cl : dp_cells) {
        // Projection on dependant cells 
        // std::cout << "dimension ret: " << 
        ret.block(i*local_dofs, 0, local_dofs, 1) = project_function(msh, msh.cells[dp_cl], hdi, where, f);
        i = i+1;
    }

    return ret;
}

template<typename Mesh>
class cut_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 compress_table;
    std::vector<size_t>                 expand_table;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

    element_location loc_zone;

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

public:

    SparseMatrix<T>         LHS;
    Matrix<T, Dynamic, 1>   RHS;

    cut_assembler(const Mesh& msh, hho_degree_info& hdi, element_location where)
        : di(hdi), loc_zone(where)
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
        };

        auto num_all_faces = msh.faces.size();
        auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
        auto num_other_faces = num_all_faces - num_dirichlet_faces;

        compress_table.resize( num_all_faces );
        expand_table.resize( num_other_faces );
        //dirichlet_data.resize( num_dirichlet_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( !is_dirichlet(fc) )
            {
                compress_table.at(i) = compressed_offset;
                expand_table.at(compressed_offset) = i;
                compressed_offset++;
            }
        }

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto system_size = cbs * msh.cells.size() + fbs * num_other_faces;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void dump_tables() const
    {
        std::cout << "Compress table: " << std::endl;
        for (size_t i = 0; i < compress_table.size(); i++)
            std::cout << i << " -> " << compress_table.at(i) << std::endl;
    }

    template<typename Function>
    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs,
             const Function& dirichlet_bf)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs + num_faces*fbs);

        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset    = cell_offset * cbs;

        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset = cbs * msh.cells.size() + compress_table.at(face_offset)*fbs;

            element_location loc_fc = location(msh, fc);
            bool in_dom = (loc_fc == element_location::ON_INTERFACE ||
                           loc_fc == loc_zone);

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET
                && in_dom;

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg, loc_zone);
                Matrix<T, Dynamic, 1> rhs_ = make_rhs(msh, fc, facdeg, loc_zone, dirichlet_bf);
                dirichlet_data.block(cbs+face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs_);
            }
        }

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
                else
                    RHS[asm_map[i]] -= lhs(i,j)*dirichlet_data(j);
            }
        }

        RHS.block(cell_LHS_offset, 0, cbs, 1) += rhs.block(0, 0, cbs, 1);
        if ( rhs.rows() > cbs )
        {
            for (size_t face_i = 0; face_i < num_faces; face_i++)
            {
                auto fc = fcs[face_i];
                auto face_offset = offset(msh, fc);
                auto face_LHS_offset = cbs * msh.cells.size() + compress_table.at(face_offset)*fbs;

                RHS.block(face_LHS_offset, 0, fbs, 1) += rhs.block(cbs+face_i*fbs, 0, fbs, 1);
            }
        }
    } // assemble()

    template<typename Function>
    Matrix<T, Dynamic, 1>
    take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl,
    const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto cell_offset        = offset(msh, cl);
        auto cell_SOL_offset    = cell_offset * cbs;

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dirichlet_bf);
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = cbs * msh.cells.size() + compress_table.at(face_offset)*fbs;
                ret.block(cbs+face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};

template<typename Mesh>
auto make_cut_assembler(const Mesh& msh, hho_degree_info& hdi, element_location where)
{
    return cut_assembler<Mesh>(msh, hdi, where);
}

/******************************************************************************************/
/*******************                                               ************************/
/*******************               VECTOR  LAPLACIAN               ************************/
/*******************                                               ************************/
/******************************************************************************************/

// make_hho_cut_interface_vector_penalty
// eta is the penalty (Nitsche's) parameter
// return eta h_T^{-1} (u_T , v_T)_{Gamma}
// we use the definition of the vector basis
template<typename T, size_t ET>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_cut_interface_vector_penalty(const cuthho_mesh<T, ET>& msh,
                                      const typename cuthho_mesh<T, ET>::cell_type& cl,
                                      const hho_degree_info& di, const T eta)
{
    auto scalar_penalty = make_hho_cut_interface_penalty(msh, cl, di, eta);

    return vector_assembly(scalar_penalty);
}

template<typename T, size_t ET , typename INTERFACE>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_cut_parametric_interface_vector_penalty(const cuthho_mesh<T, ET>& msh,
                                      const typename cuthho_mesh<T, ET>::cell_type& cl, const INTERFACE& parametric_interface,
                                      const hho_degree_info& di, const T eta)
{
    auto scalar_penalty = make_hho_cut_parametric_interface_penalty(msh, cl,parametric_interface, di, eta);

    return vector_assembly(scalar_penalty);
}

/////  GRADREC
template<typename T, size_t ET, typename Function>
std::pair< Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, Dynamic>  >
make_hho_gradrec_matrix
    (const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
     const Function& level_set_function, const hho_degree_info& di,
     element_location where, const T coeff)
{
    auto gradrec_vector = make_hho_gradrec_vector(msh, cl, level_set_function, di, where, coeff);
    auto oper = vector_assembly(gradrec_vector.first);
    auto data = vector_assembly(gradrec_vector.second);

    return std::make_pair(oper, data);
}

template<typename T, size_t ET, typename Function>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, Dynamic>  >
make_hho_gradrec_matrix_interface(const cuthho_mesh<T, ET>& msh,
                                  const typename cuthho_mesh<T, ET>::cell_type& cl,
                                  const Function& level_set_function, const hho_degree_info& di,
                                  element_location where, T coeff)
{
    auto gradrec_vector = make_hho_gradrec_vector_interface(msh, cl, level_set_function, di, where, coeff);
    auto oper = vector_assembly(gradrec_vector.first);
    auto data = vector_assembly(gradrec_vector.second);

    return std::make_pair(oper, data);
}

/////// STAB
template<typename T, size_t ET>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_vector_cut_stabilization(const cuthho_mesh<T, ET>& msh,
                                  const typename cuthho_mesh<T, ET>::cell_type& cl,
                                  const hho_degree_info& di, element_location where)
{
    auto scalar_stab = make_hho_cut_stabilization(msh, cl, di, where);

    return vector_assembly(scalar_stab);
}

template<typename T, size_t ET, typename Function>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_vector_stabilization_interface(const cuthho_mesh<T, ET>& msh,
                                        const typename cuthho_mesh<T, ET>::cell_type& cl,
                                        const Function& level_set_function,
                                        const hho_degree_info& di,
                                        const params<T>& parms = params<T>())
{
    auto scalar_stab = make_hho_stabilization_interface(msh, cl, level_set_function, di, parms);

    return vector_assembly(scalar_stab);
}

template<typename T, size_t ET>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_hho_vector_stabilization_parametric_interface(const cuthho_mesh<T, ET>& msh,
                                        const typename cuthho_mesh<T, ET>::cell_type& cl,
                                        const hho_degree_info& di,
                                        const params<T>& parms = params<T>())
{
    auto scalar_stab = make_hho_stabilization_parametric_interface(msh, cl, di, parms);

    return vector_assembly(scalar_stab);
}

/////////   RHS
// make volumic rhs
template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_vector_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                size_t degree, const F1& f, const element_location where)
{
    if ( location(msh, cl) == where )
        return make_vector_rhs(msh, cl, degree, f);

    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
    auto qps = integrate(msh, cl, 2*degree, where);
    for (auto& qp : qps)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi * f(qp.first);
    }
    return ret;
}

// make_vector_rhs_penalty
// return (g , v_T)_{Gamma} * eta/h_T
template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_vector_rhs_penalty(const cuthho_mesh<T, ET>& msh,
                        const typename cuthho_mesh<T, ET>::cell_type& cl, size_t degree,
                        const F1& g, T eta)
{
    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();

    auto hT = diameter(msh, cl);

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
    auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : qpsi)
    {
        const auto phi = cb.eval_basis(qp.first);

        ret += qp.second * phi * g(qp.first) * eta / hT;
    }
    return ret;
}

// make_vector_GR_rhs
// return -(g , GR(v_T) . n)_{Gamma}
template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_vector_GR_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                   size_t degree, const F1& g, const F2& level_set_function,
                   Matrix<T, Dynamic, Dynamic> GR, bool sym_grad = false)
{
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
//    std::cout<<"make_vector_GR_rhs: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    size_t gbs;
    if( sym_grad )
        gbs = sym_matrix_cell_basis<cuthho_mesh<T, ET>,T>::size(degree-1);
    else
        gbs = matrix_cell_basis<cuthho_mesh<T, ET>,T>::size(degree-1);

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    Matrix<T, Dynamic, 1> source_vect = Matrix<T, Dynamic, 1>::Zero(gbs);

    if( sym_grad )
    {
        sym_matrix_cell_basis<cuthho_mesh<T, ET>,T> gb(msh, cl, degree-1);
        auto qpsi = integrate_interface(msh, cl, 2*degree-1, element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : qpsi)
        {
            const auto n = level_set_function.normal(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);
            const matrix_type     qp_g_phi_n = qp.second * outer_product(g_phi, n);

            source_vect += 2 * qp_g_phi_n * g(qp.first);
        }
    }
    else
    {
        matrix_cell_basis<cuthho_mesh<T, ET>,T> gb(msh, cl, degree-1);
        auto qpsi = integrate_interface(msh, cl, 2*degree-1, element_location::IN_NEGATIVE_SIDE);
        for (auto& qp : qpsi)
        {
            const auto n = level_set_function.normal(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);
            const matrix_type     qp_g_phi_n = qp.second * outer_product(g_phi, n);

            source_vect += qp_g_phi_n * g(qp.first);
        }
    }

    return -GR.transpose() * source_vect;
}

template<typename T, size_t ET, typename Function>
Matrix<T, Dynamic, 1>
make_vector_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::face_type& fc,
                size_t degree, element_location where, const Function& f)
{
    cut_vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, degree, where);
    auto fbs = fb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(fbs);

    auto qps = integrate(msh, fc, 2*degree, where);

    for (auto& qp : qps)
    {
        auto phi = fb.eval_basis(qp.first);
        ret += qp.second * phi * f(qp.first);
    }

    return ret;
}

// make_vector_Dirichlet_jump
template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_vector_Dirichlet_jump(const cuthho_mesh<T, ET>& msh,
                           const typename cuthho_mesh<T, ET>::cell_type& cl,
                           size_t degree, const element_location where,
                           const F1& level_set_function, const F2& dir_jump, T eta)
{
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
    std::cout<<"make_vector_Dirichlet_jump: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    if( location(msh, cl) != element_location::ON_INTERFACE )
        return ret;

    auto hT = diameter(msh, cl);

    if(where == element_location::IN_NEGATIVE_SIDE) {
        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE );

        for (auto& qp : qpsi)
        {
            const auto phi = cb.eval_basis(qp.first);
            const auto dphi = cb.eval_gradients(qp.first);
            const auto n = level_set_function.normal(qp.first);
            const matrix_type     dphi_n = outer_product(dphi, n);

            ret += qp.second * ( phi * eta / hT - dphi_n) * dir_jump(qp.first);
        }
    }
    else if(where == element_location::IN_POSITIVE_SIDE) {
        auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE );

        for (auto& qp : qpsi)
        {
            auto phi = cb.eval_basis(qp.first);
            ret -= qp.second * eta / hT * phi * dir_jump(qp.first);
        }
    }
    return ret;
}

template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_vector_flux_jump_reference_pts(const cuthho_mesh<T, ET>& msh,
                      const typename cuthho_mesh<T, ET>::cell_type& cl,
                      size_t degree, const element_location where, const F1& flux_jump)
{
    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    if( location(msh, cl) != element_location::ON_INTERFACE )
        return ret;
    
   
    auto msh_int = cl.user_data.integration_msh ;
    size_t degree_curve = msh_int.degree_curve ;
    Interface_parametrisation_mesh1d curve(degree_curve) ;
    degree += 3*degree_curve -4 ; // 2*degree_curve ; // TO BE CHECKED
    auto qps = edge_quadrature<T>(degree);
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
        auto pts = points(msh_int,msh_int.cells[i_cell]);
        for(auto& qp:qps)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = curve(t , pts , degree_curve ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = curve.jacobian( t , pts , degree_curve ) ;
            auto w = 0.5 * qp.second * jacobian ;
            
            auto phi = cb.eval_basis(pt);
            ret += w * phi * flux_jump(t , pts );
            
        }
    }
            

    return ret;
}

template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_vector_flux_jump_reference_pts_cont(const cuthho_mesh<T, ET>& msh,
                      const typename cuthho_mesh<T, ET>::cell_type& cl,
                      size_t degree, const element_location where, const F1& flux_jump , const F2& parametric_interface)
{
    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    if( location(msh, cl) != element_location::ON_INTERFACE )
        return ret;
    
   
    auto msh_int = cl.user_data.integration_msh ;
    auto global_cells_i = parametric_interface.get_global_cells_interface(msh , cl);
    
//    size_t degree_curvature = parametric_interface.der_degree ;
    
    degree += (parametric_interface.der_degree + parametric_interface.dd_degree + parametric_interface.degree_det) ; //3*degree_curvature vs 4*degree_curvature -3  TO BE CHECKED
    auto qps = edge_quadrature<T>(degree);
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
        size_t global_cl_i = global_cells_i[i_cell] ;
        for(auto& qp:qps)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = parametric_interface(t , global_cl_i ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = parametric_interface.jacobian_cont( t , global_cl_i ) ;
            auto w = 0.5 * qp.second * jacobian ;
            
            auto phi = cb.eval_basis(pt);
            ret += w * phi * flux_jump(t , global_cl_i );
            
        }
    }
            

    return ret;
}

template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_vector_flux_jump_parametric_interface(const cuthho_mesh<T, ET>& msh,
                      const typename cuthho_mesh<T, ET>::cell_type& cl,
                      size_t degree, const element_location where, const F1& flux_jump , const F2& parametric_interface)
{
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    if( location(msh, cl) != element_location::ON_INTERFACE )
        return ret;
    
   
    auto para_cells_i = parametric_interface.get_global_interface(true ,msh, cl);
    
    
    degree += (parametric_interface.der_degree + parametric_interface.dd_degree + parametric_interface.degree_det) ;
    
    auto qps = edge_quadrature<T>(degree);
    for(auto& local_func : para_cells_i)
    {
        size_t cell_i = local_func.first ;
        T a = local_func.second.first;
        T b = local_func.second.second;
        for(auto& qp:qps)
        {
            auto t = 0.5*(qp.first.x()+1.0)*(b-a) + a ;
            T jacobian = parametric_interface.jacobian( t , cell_i ) ;
            auto w = 0.5 * qp.second * jacobian *(b-a) ;

            auto p = parametric_interface(t , cell_i ) ;
            point<T,2> pt = point_type( p(0) , p(1) ) ;
            
            auto phi = cb.eval_basis(pt);
            ret += w * phi * flux_jump(t , cell_i );
            
        }
    }
            

    return ret;
}

template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_vector_flux_jump(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl, size_t degree, const element_location where, const F1& flux_jump)
{
    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
//    std::cout<<"make_vector_flux_jump: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    if( location(msh, cl) != element_location::ON_INTERFACE )
            return ret;
    
    
    auto qpsi = integrate_interface(msh, cl, 3*degree , element_location::IN_NEGATIVE_SIDE);
    
    for (auto& qp : qpsi)
    {
        auto phi = cb.eval_basis(qp.first);
        ret += qp.second * phi * flux_jump(qp.first);
    }
    return ret;
}

/////////  MASS MATRIX
template<typename T, size_t ET>
Matrix<T, Dynamic, Dynamic>
make_vector_mass_matrix(const cuthho_mesh<T, ET>& msh,
                        const typename cuthho_mesh<T, ET>::face_type& fc,
                        size_t degree, element_location where)
{
    auto scalar_matrix = make_mass_matrix(msh, fc, degree, where);

    return vector_assembly(scalar_matrix);
}

/******************************************************************************************/
/*******************                                               ************************/
/*******************                STOKES PROBLEM                 ************************/
/*******************                                               ************************/
/******************************************************************************************/

///////////////////////   DIVERGENCE RECONSTRUCTION  ///////////////////////////
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<T, Dynamic, Dynamic>, Matrix<T, Dynamic, Dynamic>  >
make_hho_divergence_reconstruction(const cuthho_mesh<T, ET>& msh,
                                   const typename cuthho_mesh<T, ET>::cell_type& cl,
                                   const Function& level_set_function, const hho_degree_info& di,
                                   element_location where, const T coeff)
{
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    if ( !is_cut(msh, cl) )
        return make_hho_divergence_reconstruction(msh, cl, di);

//    std::cout<<"make_hho_divergence_reconstruction: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    const auto celdeg = di.cell_degree();
    const auto facdeg = di.face_degree();
    const auto pdeg = di.face_degree();

    cell_basis<cuthho_mesh<T, ET>,T>                   pb(msh, cl, pdeg);
    vector_cell_basis<cuthho_mesh<T, ET>,T>            cb(msh, cl, celdeg);

    auto pbs = cell_basis<cuthho_mesh<T, ET>,T>::size(pdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);

    const auto fcs = faces(msh, cl);
    const auto num_faces = fcs.size();
    const auto ns = normals(msh, cl);

    matrix_type dr_lhs = matrix_type::Zero(pbs, pbs);
    matrix_type dr_rhs = matrix_type::Zero(pbs, cbs + num_faces*fbs);


    const auto qps = integrate(msh, cl, celdeg + pdeg - 1, where);
    for (auto& qp : qps)
    {
        const auto s_phi  = pb.eval_basis(qp.first);
        const auto s_dphi = pb.eval_gradients(qp.first);
        const auto v_phi  = cb.eval_basis(qp.first);

        dr_lhs += qp.second * s_phi * s_phi.transpose();
        dr_rhs.block(0, 0, pbs, cbs) -= qp.second * s_dphi * v_phi.transpose();
    }


    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc     = fcs[i];
        const auto n      = ns[i];
        cut_vector_face_basis<cuthho_mesh<T, ET>,T>            fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, facdeg + pdeg, where);
        for (auto& qp : qps_f)
        {
            const auto p_phi = pb.eval_basis(qp.first);
            const auto f_phi = fb.eval_basis(qp.first);

            const Matrix<T, Dynamic, 2> p_phi_n = (p_phi * n.transpose());
            dr_rhs.block(0, cbs + i * fbs, pbs, fbs) += qp.second * p_phi_n * f_phi.transpose();
        }
    }


    // interface term
    auto iqp = integrate_interface(msh, cl, celdeg + pdeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqp)
    {
        const auto v_phi = cb.eval_basis(qp.first);
        const auto s_phi = pb.eval_basis(qp.first);
        const auto n = level_set_function.normal(qp.first);

        const Matrix<T, Dynamic, 2> p_phi_n = (s_phi * n.transpose());
        dr_rhs.block(0, 0, pbs, cbs) += (1.0 - coeff) * qp.second * p_phi_n * v_phi.transpose();
    }


    assert(dr_lhs.rows() == pbs && dr_lhs.cols() == pbs);
    assert(dr_rhs.rows() == pbs && dr_rhs.cols() == cbs + num_faces * fbs);

    matrix_type oper = dr_lhs.ldlt().solve(dr_rhs);
    matrix_type data = dr_rhs;

    return std::make_pair(oper, data);
}

//// make_hho_divergence_reconstruction_interface
// return the divergence reconstruction for the interface pb
// coeff -> scales the interface term
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_divergence_reconstruction_interface
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const Function& level_set_function, const hho_degree_info& di, element_location where, T coeff)
{

    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    //typedef Matrix<T, Dynamic, 1>       vector_type;
//    std::cout<<"make_hho_divergence_reconstruction_interface: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto pdeg    = facdeg;

    vector_cell_basis<cuthho_mesh<T, ET>,T>   cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>          pb(msh, cl, pdeg);


    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto pbs = cell_basis<cuthho_mesh<T, ET>,T>::size(pdeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type       rhs_tmp = matrix_type::Zero(pbs, cbs + num_faces * fbs);
    matrix_type        dr_lhs = matrix_type::Zero(pbs, pbs);
    matrix_type        dr_rhs = matrix_type::Zero(pbs, 2*cbs + 2*num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + pdeg, where);
    for (auto& qp : qps)
    {
        const auto v_phi     = cb.eval_basis(qp.first);
        const auto s_phi   = pb.eval_basis(qp.first);
        const auto s_dphi  = pb.eval_gradients(qp.first);
        // integration pour parties de la divergence
        dr_lhs += qp.second * s_phi * s_phi.transpose();
        rhs_tmp.block(0, 0, pbs, cbs) -= qp.second * s_dphi * v_phi.transpose();
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        cut_vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, pdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const matrix_type f_phi      = fb.eval_basis(qp.first);
            const auto        s_phi      = pb.eval_basis(qp.first);
            const matrix_type qp_s_phi_n = qp.second * s_phi * n.transpose();
            rhs_tmp.block(0, cbs + i * fbs, pbs, fbs) += qp_s_phi_n * f_phi.transpose();
        }
    }

    // term on the interface
    matrix_type        interface_term = matrix_type::Zero(pbs, cbs);
    size_t phi_n_deg = level_set_function.grad_deg;
    const auto iqps = integrate_interface(msh, cl, celdeg + pdeg + phi_n_deg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto v_phi        = cb.eval_basis(qp.first);
        const auto s_phi        = pb.eval_basis(qp.first);

        Matrix<T,2,1> n = level_set_function.normal(qp.first);
        const matrix_type qp_s_phi_n = qp.second * s_phi * n.transpose();

        interface_term += qp_s_phi_n * v_phi.transpose();
    }

    // finalize rhs
    if(where == element_location::IN_NEGATIVE_SIDE)
    {
        dr_rhs.block(0, 0, pbs, cbs) += rhs_tmp.block(0, 0, pbs, cbs);
        dr_rhs.block(0, 2*cbs, pbs, num_faces*fbs)
            += rhs_tmp.block(0, cbs, pbs, num_faces*fbs);
        // coeff -> interface jump + 1.0 -> derivé par l'integration pour parties de la DIV
        dr_rhs.block(0, 0, pbs, cbs) += (1.0 - coeff) * interface_term;
        dr_rhs.block(0, cbs, pbs, cbs) += coeff * interface_term;
        // Stefano implementation alfai dependent
//        dr_rhs.block(0, 0, pbs, cbs) += coeff * interface_term;
//        dr_rhs.block(0, cbs, pbs, cbs) += (1.0 - coeff) * interface_term;
    }
    else if( where == element_location::IN_POSITIVE_SIDE)
    {
        dr_rhs.block(0, cbs, pbs, cbs) += rhs_tmp.block(0, 0, pbs, cbs);
        dr_rhs.block(0, 2*cbs + num_faces*fbs, pbs, num_faces*fbs)
                     += rhs_tmp.block(0, cbs, pbs, num_faces*fbs);
        // coeff -> interface jump + 1.0 -> derivé par l'integration pour parties de la DIV
        dr_rhs.block(0, 0, pbs, cbs) -= coeff * interface_term;
        dr_rhs.block(0, cbs, pbs, cbs) += (coeff-1.0) * interface_term;
        // Stefano implementation alfai dependent
//        dr_rhs.block(0, 0, pbs, cbs) += (coeff-1.0) * interface_term;
//        dr_rhs.block(0, cbs, pbs, cbs) -= coeff * interface_term;
    }

    matrix_type oper = dr_lhs.ldlt().solve(dr_rhs);

    return std::make_pair(oper, dr_rhs);
}

//// make_hho_divergence_reconstruction_interface_ref_pts
// return the divergence reconstruction for the interface pb
// coeff -> scales the interface term
//Parametric interface to handlethe normal
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_divergence_reconstruction_interface_ref_pts
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const Function& level_set_function, const hho_degree_info& di, element_location where, T coeff)
{

    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    //typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto pdeg    = facdeg;

    vector_cell_basis<cuthho_mesh<T, ET>,T>   cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>          pb(msh, cl, pdeg);


    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto pbs = cell_basis<cuthho_mesh<T, ET>,T>::size(pdeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type       rhs_tmp = matrix_type::Zero(pbs, cbs + num_faces * fbs);
    matrix_type        dr_lhs = matrix_type::Zero(pbs, pbs);
    matrix_type        dr_rhs = matrix_type::Zero(pbs, 2*cbs + 2*num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + pdeg, where);
    for (auto& qp : qps)
    {
        const auto v_phi     = cb.eval_basis(qp.first);
        const auto s_phi   = pb.eval_basis(qp.first);
        const auto s_dphi  = pb.eval_gradients(qp.first);

        dr_lhs += qp.second * s_phi * s_phi.transpose();
        rhs_tmp.block(0, 0, pbs, cbs) -= qp.second * s_dphi * v_phi.transpose();
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        cut_vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, pdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const matrix_type f_phi      = fb.eval_basis(qp.first);
            const auto        s_phi      = pb.eval_basis(qp.first);
            const matrix_type qp_s_phi_n = qp.second * s_phi * n.transpose();

            rhs_tmp.block(0, cbs + i * fbs, pbs, fbs) += qp_s_phi_n * f_phi.transpose();
        }
    }

    // term on the interface
    matrix_type        interface_term = matrix_type::Zero(pbs, cbs);
    auto msh_int = cl.user_data.integration_msh ;
    size_t degree_curve = msh_int.degree_curve ;
    Interface_parametrisation_mesh1d curve(degree_curve) ;
    size_t degree =  celdeg + pdeg;
    degree += + 2*degree_curve - 2 ; // 2*degree_curve ; // TO BE CHECKED
    auto qpsi = edge_quadrature<T>(degree);
        
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
        auto pts = points(msh_int,msh_int.cells[i_cell]);
        for(auto& qp:qpsi)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = curve(t , pts , degree_curve ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = curve.jacobian( t , pts , degree_curve ) ;
            auto w = 0.5 * qp.second * jacobian ;
            
            const auto v_phi        = cb.eval_basis(pt);
            const auto s_phi        = pb.eval_basis(pt);

//            Matrix<T,2,1> n = level_set_function.normal(qp.first);
            Matrix<T,2,1> n = curve.normal(t , pts , degree_curve ) ;
            const matrix_type qp_s_phi_n = w * s_phi * n.transpose();

            interface_term += qp_s_phi_n * v_phi.transpose();
                
            
               
            }
        }
    

    // finalize rhs
    if(where == element_location::IN_NEGATIVE_SIDE)
    {
        dr_rhs.block(0, 0, pbs, cbs) += rhs_tmp.block(0, 0, pbs, cbs);
        dr_rhs.block(0, 2*cbs, pbs, num_faces*fbs)
            += rhs_tmp.block(0, cbs, pbs, num_faces*fbs);
        dr_rhs.block(0, 0, pbs, cbs) += (1.0 - coeff) * interface_term;
        dr_rhs.block(0, cbs, pbs, cbs) += coeff * interface_term;
    }
    else if( where == element_location::IN_POSITIVE_SIDE)
    {
        dr_rhs.block(0, cbs, pbs, cbs) += rhs_tmp.block(0, 0, pbs, cbs);
        dr_rhs.block(0, 2*cbs + num_faces*fbs, pbs, num_faces*fbs)
                     += rhs_tmp.block(0, cbs, pbs, num_faces*fbs);
        dr_rhs.block(0, 0, pbs, cbs) -= coeff * interface_term;
        dr_rhs.block(0, cbs, pbs, cbs) += (coeff-1.0) * interface_term;
    }

    matrix_type oper = dr_lhs.ldlt().solve(dr_rhs);

    return std::make_pair(oper, dr_rhs);
}

//// make_hho_divergence_reconstruction_interface_ref_pts
// return the divergence reconstruction for the interface pb
// coeff -> scales the interface term
//Parametric interface to handlethe normal
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_divergence_reconstruction_interface_ref_pts_cont
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const Function& parametric_interface, const hho_degree_info& di, element_location where, T coeff)
{

    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    //typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto pdeg    = facdeg;

    vector_cell_basis<cuthho_mesh<T, ET>,T>   cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>          pb(msh, cl, pdeg);


    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto pbs = cell_basis<cuthho_mesh<T, ET>,T>::size(pdeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type       rhs_tmp = matrix_type::Zero(pbs, cbs + num_faces * fbs);
    matrix_type        dr_lhs = matrix_type::Zero(pbs, pbs);
    matrix_type        dr_rhs = matrix_type::Zero(pbs, 2*cbs + 2*num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + pdeg, where);
    for (auto& qp : qps)
    {
        const auto v_phi     = cb.eval_basis(qp.first);
        const auto s_phi   = pb.eval_basis(qp.first);
        const auto s_dphi  = pb.eval_gradients(qp.first);

        dr_lhs += qp.second * s_phi * s_phi.transpose();
        rhs_tmp.block(0, 0, pbs, cbs) -= qp.second * s_dphi * v_phi.transpose();
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        cut_vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, pdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const matrix_type f_phi      = fb.eval_basis(qp.first);
            const auto        s_phi      = pb.eval_basis(qp.first);
            const matrix_type qp_s_phi_n = qp.second * s_phi * n.transpose();

            rhs_tmp.block(0, cbs + i * fbs, pbs, fbs) += qp_s_phi_n * f_phi.transpose();
        }
    }

    // term on the interface
    matrix_type        interface_term = matrix_type::Zero(pbs, cbs);
    auto global_cells_i = parametric_interface.get_global_cells_interface(msh , cl);
    auto msh_int = cl.user_data.integration_msh ;
//    size_t degree_curvature = parametric_interface.der_degree ;
    
    size_t degree =  celdeg + pdeg;
    degree += (parametric_interface.der_degree + parametric_interface.degree_det)  ; //2*degree_curvature vs  3*degree_curve - 2 IT WAS
    auto qpsi = edge_quadrature<T>(degree);
        
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
        size_t global_cl_i = global_cells_i[i_cell] ;
        for(auto& qp:qpsi)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = parametric_interface(t , global_cl_i ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = parametric_interface.jacobian_cont( t , global_cl_i ) ;
            auto w = 0.5 * qp.second * jacobian ;
            
            const auto v_phi        = cb.eval_basis(pt);
            const auto s_phi        = pb.eval_basis(pt);

            Matrix<T,2,1> n = parametric_interface.normal_cont(t , global_cl_i ) ;
            const matrix_type qp_s_phi_n = w * s_phi * n.transpose();

            interface_term += qp_s_phi_n * v_phi.transpose();
                
            
               
            }
        }
    

    // finalize rhs
    if(where == element_location::IN_NEGATIVE_SIDE)
    {
        dr_rhs.block(0, 0, pbs, cbs) += rhs_tmp.block(0, 0, pbs, cbs);
        dr_rhs.block(0, 2*cbs, pbs, num_faces*fbs)
            += rhs_tmp.block(0, cbs, pbs, num_faces*fbs);
        dr_rhs.block(0, 0, pbs, cbs) += (1.0 - coeff) * interface_term;
        dr_rhs.block(0, cbs, pbs, cbs) += coeff * interface_term;
    }
    else if( where == element_location::IN_POSITIVE_SIDE)
    {
        dr_rhs.block(0, cbs, pbs, cbs) += rhs_tmp.block(0, 0, pbs, cbs);
        dr_rhs.block(0, 2*cbs + num_faces*fbs, pbs, num_faces*fbs)
                     += rhs_tmp.block(0, cbs, pbs, num_faces*fbs);
        dr_rhs.block(0, 0, pbs, cbs) -= coeff * interface_term;
        dr_rhs.block(0, cbs, pbs, cbs) += (coeff-1.0) * interface_term;
    }

    matrix_type oper = dr_lhs.ldlt().solve(dr_rhs);

    return std::make_pair(oper, dr_rhs);
}

//// make_hho_divergence_reconstruction_interface_ref_pts
// return the divergence reconstruction for the interface pb
// coeff -> scales the interface term
//Parametric interface to handlethe normal
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_divergence_reconstruction_parametric_interface
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const Function& parametric_interface, const hho_degree_info& di, element_location where, T coeff)
{

    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    //typedef Matrix<T, Dynamic, 1>       vector_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto pdeg    = facdeg;

    vector_cell_basis<cuthho_mesh<T, ET>,T>   cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>          pb(msh, cl, pdeg);


    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto pbs = cell_basis<cuthho_mesh<T, ET>,T>::size(pdeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type       rhs_tmp = matrix_type::Zero(pbs, cbs + num_faces * fbs);
    matrix_type        dr_lhs = matrix_type::Zero(pbs, pbs);
    matrix_type        dr_rhs = matrix_type::Zero(pbs, 2*cbs + 2*num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + pdeg, where);
    for (auto& qp : qps)
    {
        const auto v_phi     = cb.eval_basis(qp.first);
        const auto s_phi   = pb.eval_basis(qp.first);
        const auto s_dphi  = pb.eval_gradients(qp.first);

        dr_lhs += qp.second * s_phi * s_phi.transpose();
        rhs_tmp.block(0, 0, pbs, cbs) -= qp.second * s_dphi * v_phi.transpose();
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        cut_vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, pdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const matrix_type f_phi      = fb.eval_basis(qp.first);
            const auto        s_phi      = pb.eval_basis(qp.first);
            const matrix_type qp_s_phi_n = qp.second * s_phi * n.transpose();

            rhs_tmp.block(0, cbs + i * fbs, pbs, fbs) += qp_s_phi_n * f_phi.transpose();
        }
    }

    // term on the interface
    size_t degree =  celdeg + pdeg;
    matrix_type        interface_term = matrix_type::Zero(pbs, cbs);
    
    auto para_cells_i = parametric_interface.get_global_interface(true ,msh, cl);
    

    degree += (parametric_interface.der_degree + parametric_interface.degree_det) ;
    auto qpsi = edge_quadrature<T>(degree);
        
    for(auto& local_func : para_cells_i)
    {
        size_t cell_i = local_func.first ;
        T a = local_func.second.first;
        T b = local_func.second.second;
        for(auto& qp:qpsi)
        {
            auto t = 0.5*(qp.first.x()+1.0)*(b-a) + a ;
            T jacobian = parametric_interface.jacobian( t , cell_i ) ;
            auto w = 0.5 * qp.second * jacobian *(b-a) ;

            auto p = parametric_interface(t , cell_i ) ;
            point<T,2> pt = point_type( p(0) , p(1) ) ;
            
            const auto v_phi        = cb.eval_basis(pt);
            const auto s_phi        = pb.eval_basis(pt);

            Matrix<T,2,1> n = parametric_interface.normal(t , cell_i ) ;
            const matrix_type qp_s_phi_n = w * s_phi * n.transpose();

            interface_term += qp_s_phi_n * v_phi.transpose();
                
            
               
            }
        }
    

    // finalize rhs
    if(where == element_location::IN_NEGATIVE_SIDE)
    {
        dr_rhs.block(0, 0, pbs, cbs) += rhs_tmp.block(0, 0, pbs, cbs);
        dr_rhs.block(0, 2*cbs, pbs, num_faces*fbs)
            += rhs_tmp.block(0, cbs, pbs, num_faces*fbs);
        dr_rhs.block(0, 0, pbs, cbs) += (1.0 - coeff) * interface_term;
        dr_rhs.block(0, cbs, pbs, cbs) += coeff * interface_term;
    }
    else if( where == element_location::IN_POSITIVE_SIDE)
    {
        dr_rhs.block(0, cbs, pbs, cbs) += rhs_tmp.block(0, 0, pbs, cbs);
        dr_rhs.block(0, 2*cbs + num_faces*fbs, pbs, num_faces*fbs)
                     += rhs_tmp.block(0, cbs, pbs, num_faces*fbs);
        dr_rhs.block(0, 0, pbs, cbs) -= coeff * interface_term;
        dr_rhs.block(0, cbs, pbs, cbs) += (coeff-1.0) * interface_term;
    }

    matrix_type oper = dr_lhs.ldlt().solve(dr_rhs);

    return std::make_pair(oper, dr_rhs);
}

///////////////////////   Stokes Stabilization
template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_stokes_interface_stabilization
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const hho_degree_info& di, const F1& level_set_function)
{
    auto hT = diameter(msh, cl);

    auto celdeg = di.cell_degree();
    auto pdeg = di.face_degree();
//    std::cout<<"make_stokes_interface_stabilization: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>        pb(msh, cl, pdeg);

    auto cbs = cb.size();
    auto pbs = pb.size();

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero( 2*(cbs+pbs), 2*(cbs+pbs));
    Matrix<T, Dynamic, Dynamic> ret_temp = Matrix<T, Dynamic, Dynamic>::Zero( cbs+pbs, cbs+pbs  );

    size_t phi_n_deg = level_set_function.grad_deg;
    auto qpsi = integrate_interface(msh, cl, celdeg - 1 + pdeg + 2*phi_n_deg ,
                                    element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : qpsi)
    {
        const auto v_dphi = cb.eval_gradients(qp.first);
        const auto s_phi  = pb.eval_basis(qp.first);
        const auto n = level_set_function.normal(qp.first);
        const Matrix<T, Dynamic, Dynamic> v_dphi_n = outer_product(v_dphi, n);
        const Matrix<T, Dynamic, Dynamic> s_phi_n = s_phi * n.transpose();

        ret_temp.block(0, 0, cbs, cbs) += qp.second * v_dphi_n * v_dphi_n.transpose();
        ret_temp.block(cbs, 0, pbs, cbs) += qp.second * s_phi_n * v_dphi_n.transpose();
        ret_temp.block(cbs, cbs, pbs, pbs) += qp.second * s_phi_n * s_phi_n.transpose();
    }

    // vel -- vel
    ret.block(0, 0, cbs, cbs)     += hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(0, cbs, cbs, cbs)   -= hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(cbs, 0, cbs, cbs)   -= hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(cbs, cbs, cbs, cbs) += hT * ret_temp.block(0, 0, cbs, cbs);

    // vel -- p
    ret.block(0, 2*cbs, cbs, pbs)       -= hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(0, 2*cbs + pbs, cbs, pbs) += hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(cbs, 2*cbs, cbs, pbs)     += hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(cbs, 2*cbs+pbs, cbs, pbs) -= hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();

    // p -- vel
    ret.block(2*cbs, 0, pbs, cbs)       -= hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs, cbs, pbs, cbs)     += hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs+pbs, 0, pbs, cbs)   += hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs+pbs, cbs, pbs, cbs) -= hT * ret_temp.block(cbs, 0, pbs, cbs);

    // p -- p
    ret.block(2*cbs, 2*cbs, pbs, pbs)         += hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs, 2*cbs+pbs, pbs, pbs)     -= hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs+pbs, 2*cbs, pbs, pbs)     -= hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs+pbs, 2*cbs+pbs, pbs, pbs) += hT * ret_temp.block(cbs, cbs, pbs, pbs);

    return ret;
}

template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_stokes_interface_stabilization_ref_pts
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const hho_degree_info& di, const F1& level_set_function)
{
    auto hT = diameter(msh, cl);

    auto celdeg = di.cell_degree();
    auto pdeg = di.face_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>        pb(msh, cl, pdeg);

    auto cbs = cb.size();
    auto pbs = pb.size();

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero( 2*(cbs+pbs), 2*(cbs+pbs));
    Matrix<T, Dynamic, Dynamic> ret_temp = Matrix<T, Dynamic, Dynamic>::Zero( cbs+pbs, cbs+pbs  );

    
    auto msh_int = cl.user_data.integration_msh ;
    size_t degree_curve = msh_int.degree_curve ;
    Interface_parametrisation_mesh1d curve(degree_curve) ;
    size_t degree = celdeg - 1 + pdeg;
    if(degree_curve == 1)
        degree += 2;
    else
        degree += 3*degree_curve - 4 ; //  2*degree_curve ; // TO BE CHECKED
    
    auto qpsi = edge_quadrature<T>(degree);
        
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
        auto pts = points(msh_int,msh_int.cells[i_cell]);
        for(auto& qp:qpsi)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = curve(t , pts , degree_curve ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = curve.jacobian( t , pts , degree_curve ) ;
            auto w = 0.5 * qp.second * jacobian ;
                
                
            Matrix<T,2,1> n = curve.normal(t , pts , degree_curve ) ;
            
            const auto v_dphi = cb.eval_gradients(pt);
            const auto s_phi  = pb.eval_basis(pt);
//           const auto n = level_set_function.normal(pt);
            const Matrix<T, Dynamic, Dynamic> v_dphi_n = outer_product(v_dphi, n);
            const Matrix<T, Dynamic, Dynamic> s_phi_n = s_phi * n.transpose();

            ret_temp.block(0, 0, cbs, cbs) += w * v_dphi_n * v_dphi_n.transpose();
            ret_temp.block(cbs, 0, pbs, cbs) += w * s_phi_n * v_dphi_n.transpose();
            ret_temp.block(cbs, cbs, pbs, pbs) += w * s_phi_n * s_phi_n.transpose();
                
        }
    }
    

    // vel -- vel
    ret.block(0, 0, cbs, cbs)     += hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(0, cbs, cbs, cbs)   -= hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(cbs, 0, cbs, cbs)   -= hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(cbs, cbs, cbs, cbs) += hT * ret_temp.block(0, 0, cbs, cbs);

    // vel -- p
    ret.block(0, 2*cbs, cbs, pbs)       -= hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(0, 2*cbs + pbs, cbs, pbs) += hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(cbs, 2*cbs, cbs, pbs)     += hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(cbs, 2*cbs+pbs, cbs, pbs) -= hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();

    // p -- vel
    ret.block(2*cbs, 0, pbs, cbs)       -= hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs, cbs, pbs, cbs)     += hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs+pbs, 0, pbs, cbs)   += hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs+pbs, cbs, pbs, cbs) -= hT * ret_temp.block(cbs, 0, pbs, cbs);

    // p -- p
    ret.block(2*cbs, 2*cbs, pbs, pbs)         += hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs, 2*cbs+pbs, pbs, pbs)     -= hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs+pbs, 2*cbs, pbs, pbs)     -= hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs+pbs, 2*cbs+pbs, pbs, pbs) += hT * ret_temp.block(cbs, cbs, pbs, pbs);

    return ret;
}

template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_stokes_interface_stabilization_ref_pts_cont
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const hho_degree_info& di, const F1& parametric_interface)
{
    auto hT = diameter(msh, cl);

    auto celdeg = di.cell_degree();
    auto pdeg = di.face_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>        pb(msh, cl, pdeg);

    auto cbs = cb.size();
    auto pbs = pb.size();

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero( 2*(cbs+pbs), 2*(cbs+pbs));
    Matrix<T, Dynamic, Dynamic> ret_temp = Matrix<T, Dynamic, Dynamic>::Zero( cbs+pbs, cbs+pbs  );

    
    auto msh_int = cl.user_data.integration_msh ;
    auto global_cells_i = parametric_interface.get_global_cells_interface(msh , cl);
//    size_t degree_curvature = parametric_interface.der_degree ;
  
    size_t degree = celdeg - 1 + pdeg;
    
    
    degree += (2 * parametric_interface.der_degree + parametric_interface.degree_det ); // 3 * degree_curvature vs 4*degree_curvature - 3 TO BE CHECKED
    
    auto qpsi = edge_quadrature<T>(degree);
        
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
        size_t global_cl_i = global_cells_i[i_cell] ;
        for(auto& qp:qpsi)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = parametric_interface(t , global_cl_i ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = parametric_interface.jacobian_cont( t , global_cl_i ) ;
            auto w = 0.5 * qp.second * jacobian ;
                
                
            Matrix<T,2,1> n = parametric_interface.normal_cont(t , global_cl_i ) ;
            
            const auto v_dphi = cb.eval_gradients(pt);
            const auto s_phi  = pb.eval_basis(pt);
//           const auto n = level_set_function.normal(pt);
            const Matrix<T, Dynamic, Dynamic> v_dphi_n = outer_product(v_dphi, n);
            const Matrix<T, Dynamic, Dynamic> s_phi_n = s_phi * n.transpose();

            ret_temp.block(0, 0, cbs, cbs) += w * v_dphi_n * v_dphi_n.transpose();
            ret_temp.block(cbs, 0, pbs, cbs) += w * s_phi_n * v_dphi_n.transpose();
            ret_temp.block(cbs, cbs, pbs, pbs) += w * s_phi_n * s_phi_n.transpose();
                
        }
    }
    

    // vel -- vel
    ret.block(0, 0, cbs, cbs)     += hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(0, cbs, cbs, cbs)   -= hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(cbs, 0, cbs, cbs)   -= hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(cbs, cbs, cbs, cbs) += hT * ret_temp.block(0, 0, cbs, cbs);

    // vel -- p
    ret.block(0, 2*cbs, cbs, pbs)       -= hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(0, 2*cbs + pbs, cbs, pbs) += hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(cbs, 2*cbs, cbs, pbs)     += hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(cbs, 2*cbs+pbs, cbs, pbs) -= hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();

    // p -- vel
    ret.block(2*cbs, 0, pbs, cbs)       -= hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs, cbs, pbs, cbs)     += hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs+pbs, 0, pbs, cbs)   += hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs+pbs, cbs, pbs, cbs) -= hT * ret_temp.block(cbs, 0, pbs, cbs);

    // p -- p
    ret.block(2*cbs, 2*cbs, pbs, pbs)         += hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs, 2*cbs+pbs, pbs, pbs)     -= hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs+pbs, 2*cbs, pbs, pbs)     -= hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs+pbs, 2*cbs+pbs, pbs, pbs) += hT * ret_temp.block(cbs, cbs, pbs, pbs);

    return ret;
}

template<typename T, size_t ET, typename F1>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>
make_stokes_parametric_interface_stabilization
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const hho_degree_info& di, const F1& parametric_interface)
{
    auto hT = diameter(msh, cl);
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    auto celdeg = di.cell_degree();
    auto pdeg = di.face_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>        pb(msh, cl, pdeg);

    auto cbs = cb.size();
    auto pbs = pb.size();

    Matrix<T, Dynamic, Dynamic> ret = Matrix<T, Dynamic, Dynamic>::Zero( 2*(cbs+pbs), 2*(cbs+pbs));
    Matrix<T, Dynamic, Dynamic> ret_temp = Matrix<T, Dynamic, Dynamic>::Zero( cbs+pbs, cbs+pbs  );

    
    auto para_cells_i = parametric_interface.get_global_interface(true ,msh, cl);
  
    size_t degree = celdeg - 1 + pdeg;
    degree += (2 * parametric_interface.der_degree + parametric_interface.degree_det );
    
    auto qpsi = edge_quadrature<T>(degree);
        
    for(auto& local_func : para_cells_i)
    {
        size_t cell_i = local_func.first ;
        T a = local_func.second.first;
        T b = local_func.second.second;
        for(auto& qp:qpsi)
        {
            auto t = 0.5*(qp.first.x()+1.0)*(b-a) + a ;
            T jacobian = parametric_interface.jacobian( t , cell_i ) ;
            auto w = 0.5 * qp.second * jacobian *(b-a) ;
            auto p = parametric_interface(t , cell_i ) ;
            point<T,2> pt = point_type( p(0) , p(1) ) ;
                
                
            Matrix<T,2,1> n = parametric_interface.normal(t , cell_i ) ;
            
            const auto v_dphi = cb.eval_gradients(pt);
            const auto s_phi  = pb.eval_basis(pt);
//           const auto n = level_set_function.normal(pt);
            const Matrix<T, Dynamic, Dynamic> v_dphi_n = outer_product(v_dphi, n);
            const Matrix<T, Dynamic, Dynamic> s_phi_n = s_phi * n.transpose();

            ret_temp.block(0, 0, cbs, cbs) += w * v_dphi_n * v_dphi_n.transpose();
            ret_temp.block(cbs, 0, pbs, cbs) += w * s_phi_n * v_dphi_n.transpose();
            ret_temp.block(cbs, cbs, pbs, pbs) += w * s_phi_n * s_phi_n.transpose();
                
        }
    }
    

    // vel -- vel
    ret.block(0, 0, cbs, cbs)     += hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(0, cbs, cbs, cbs)   -= hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(cbs, 0, cbs, cbs)   -= hT * ret_temp.block(0, 0, cbs, cbs);
    ret.block(cbs, cbs, cbs, cbs) += hT * ret_temp.block(0, 0, cbs, cbs);

    // vel -- p
    ret.block(0, 2*cbs, cbs, pbs)       -= hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(0, 2*cbs + pbs, cbs, pbs) += hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(cbs, 2*cbs, cbs, pbs)     += hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();
    ret.block(cbs, 2*cbs+pbs, cbs, pbs) -= hT * ret_temp.block(cbs, 0, pbs, cbs).transpose();

    // p -- vel
    ret.block(2*cbs, 0, pbs, cbs)       -= hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs, cbs, pbs, cbs)     += hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs+pbs, 0, pbs, cbs)   += hT * ret_temp.block(cbs, 0, pbs, cbs);
    ret.block(2*cbs+pbs, cbs, pbs, cbs) -= hT * ret_temp.block(cbs, 0, pbs, cbs);

    // p -- p
    ret.block(2*cbs, 2*cbs, pbs, pbs)         += hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs, 2*cbs+pbs, pbs, pbs)     -= hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs+pbs, 2*cbs, pbs, pbs)     -= hT * ret_temp.block(cbs, cbs, pbs, pbs);
    ret.block(2*cbs+pbs, 2*cbs+pbs, pbs, pbs) += hT * ret_temp.block(cbs, cbs, pbs, pbs);

    return ret;
}

template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_stokes_interface_stabilization_RHS
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const hho_degree_info& di, const F1& level_set_function, const F2& neumann_jump)
{
    auto hT = diameter(msh, cl);

    auto celdeg = di.cell_degree();
    auto pdeg = di.face_degree();
//    std::cout<<"make_stokes_interface_stabilization_RHS: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>        pb(msh, cl, pdeg);

    auto cbs = cb.size();
    auto pbs = pb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero( 2*(cbs + 2*pbs) );
    size_t phi_n_deg = level_set_function.grad_deg;
    size_t phi_H_deg = level_set_function.div_deg;
    auto qpsi = integrate_interface(msh, cl, celdeg + 2*phi_n_deg + phi_H_deg, element_location::IN_NEGATIVE_SIDE );
    for (auto& qp : qpsi)
    {
        auto v_dphi = cb.eval_gradients(qp.first);
        auto s_phi  = pb.eval_basis(qp.first);
        auto n = level_set_function.normal(qp.first);
        const Matrix<T, Dynamic, Dynamic> v_dphi_n = outer_product(v_dphi, n);
        const Matrix<T, Dynamic, Dynamic> s_phi_n = s_phi * n.transpose();

        ret.head(cbs) += hT * qp.second * v_dphi_n * neumann_jump(qp.first);
        ret.tail(pbs) -= hT * qp.second * s_phi_n  * neumann_jump(qp.first);
    }

    return ret;
}

template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_stokes_interface_stabilization_RHS_ref_pts
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const hho_degree_info& di, const F1& level_set_function, const F2& neumann_jump)
{
    auto hT = diameter(msh, cl);

    auto celdeg = di.cell_degree();
    auto pdeg = di.face_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>        pb(msh, cl, pdeg);

    auto cbs = cb.size();
    auto pbs = pb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero( 2*(cbs + 2*pbs) );
    
    auto msh_int = cl.user_data.integration_msh ;
    size_t degree_curve = msh_int.degree_curve ;
    Interface_parametrisation_mesh1d curve(degree_curve) ;
    size_t degree = celdeg ; // 2*celdeg ;
    if(degree_curve == 1)
        degree += 2;
    else
        degree += 4 * degree_curve -6 ;// 2*degree_curve ; // TO BE CHECKED (3* degree_curve - 4) ?
    auto qps = edge_quadrature<T>(degree);
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
        auto pts = points(msh_int,msh_int.cells[i_cell]);
        for(auto& qp:qps)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = curve(t , pts , degree_curve ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = curve.jacobian( t , pts , degree_curve ) ;
            auto w = 0.5 * qp.second * jacobian ;
            
            auto v_dphi = cb.eval_gradients(pt);
            auto s_phi  = pb.eval_basis(pt);
            
//            auto n = level_set_function.normal(pt);
            auto n = curve.normal(t , pts , degree_curve ) ;
            
            const Matrix<T, Dynamic, Dynamic> v_dphi_n = outer_product(v_dphi, n);
            const Matrix<T, Dynamic, Dynamic> s_phi_n = s_phi * n.transpose();

            ret.head(cbs) += hT * w * v_dphi_n * neumann_jump(t , pts);
            ret.tail(pbs) -= hT * w * s_phi_n  * neumann_jump(t , pts);
            
        }
    }
       
    

    return ret;
}

template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_stokes_interface_stabilization_RHS_ref_pts_cont
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const hho_degree_info& di, const F1& parametric_interface, const F2& neumann_jump)
{
    auto hT = diameter(msh, cl);

    auto celdeg = di.cell_degree();
    auto pdeg = di.face_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>        pb(msh, cl, pdeg);

    auto cbs = cb.size();
    auto pbs = pb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero( 2*(cbs + 2*pbs) );
    
    auto msh_int = cl.user_data.integration_msh ;
    auto global_cells_i = parametric_interface.get_global_cells_interface(msh , cl);
    
    
//    size_t degree_curvature = parametric_interface.der_degree ;
    
    size_t degree = celdeg ; // 2*celdeg ;
    
    
    degree += (2*parametric_interface.der_degree + parametric_interface.dd_degree + parametric_interface.degree_det ); // 4*degree_curvature  (5 * degree_curve -4) ;
    
    auto qps = edge_quadrature<T>(degree);
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
        size_t global_cl_i = global_cells_i[i_cell] ;
        for(auto& qp:qps)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = parametric_interface(t , global_cl_i ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = parametric_interface.jacobian_cont( t , global_cl_i ) ;
            auto w = 0.5 * qp.second * jacobian ;
            
            auto v_dphi = cb.eval_gradients(pt);
            auto s_phi  = pb.eval_basis(pt);
            
            Matrix<T,2,1> n = parametric_interface.normal_cont(t , global_cl_i ) ;
            
            const Matrix<T, Dynamic, Dynamic> v_dphi_n = outer_product(v_dphi, n);
            const Matrix<T, Dynamic, Dynamic> s_phi_n = s_phi * n.transpose();

            ret.head(cbs) += hT * w * v_dphi_n * neumann_jump(t , global_cl_i);
            ret.tail(pbs) -= hT * w * s_phi_n  * neumann_jump(t , global_cl_i);
            
        }
    }
       
    

    return ret;
}

template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_stokes_parametric_interface_stabilization_RHS
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const hho_degree_info& di, const F1& parametric_interface, const F2& neumann_jump)
{
    auto hT = diameter(msh, cl);
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    auto celdeg = di.cell_degree();
    auto pdeg = di.face_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, celdeg);
    cell_basis<cuthho_mesh<T, ET>,T>        pb(msh, cl, pdeg);

    auto cbs = cb.size();
    auto pbs = pb.size();

    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero( 2*(cbs + 2*pbs) );
    
    auto para_cells_i = parametric_interface.get_global_interface(true ,msh, cl);
    
    size_t degree = celdeg ; // 2*celdeg ;
    
    
    degree += (2*parametric_interface.der_degree + parametric_interface.dd_degree + parametric_interface.degree_det ); // 4*degree_curvature  (5 * degree_curve -4) ;
    
    auto qps = edge_quadrature<T>(degree);
    for(auto& local_func : para_cells_i)
    {
        size_t cell_i = local_func.first ;
        T a = local_func.second.first;
        T b = local_func.second.second;
        for(auto& qp:qps)
        {
            auto t = 0.5*(qp.first.x()+1.0)*(b-a) + a ;
            T jacobian = parametric_interface.jacobian( t , cell_i ) ;
            auto w = 0.5 * qp.second * jacobian *(b-a) ;
            auto p = parametric_interface(t , cell_i ) ;
            point<T,2> pt = point_type( p(0) , p(1) ) ;
            
            auto v_dphi = cb.eval_gradients(pt);
            auto s_phi  = pb.eval_basis(pt);
            
            Matrix<T,2,1> n = parametric_interface.normal(t , cell_i ) ;
            
            const Matrix<T, Dynamic, Dynamic> v_dphi_n = outer_product(v_dphi, n);
            const Matrix<T, Dynamic, Dynamic> s_phi_n = s_phi * n.transpose();

            ret.head(cbs) += hT * w * v_dphi_n * neumann_jump(t , cell_i);
            ret.tail(pbs) -= hT * w * s_phi_n  * neumann_jump(t , cell_i);
            
        }
    }
       
    

    return ret;
}

///////////////////////   RHS  pressure
template<typename T, size_t ET, typename F1, typename F2>
Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, 1>
make_pressure_rhs(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
                  size_t degree, const element_location where,
                  const F1& level_set_function, const F2& bcs_fun)
{
    if( location(msh, cl) != element_location::ON_INTERFACE )
    {
        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(degree);
        Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);
        return ret;
    }
//    std::cout<<"make_pressure_rhs: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    cell_basis<cuthho_mesh<T, ET>,T> cb(msh, cl, degree);
    auto cbs = cb.size();
    Matrix<T, Dynamic, 1> ret = Matrix<T, Dynamic, 1>::Zero(cbs);

    auto qpsi = integrate_interface(msh, cl, 2*degree, element_location::IN_NEGATIVE_SIDE );
    for (auto& qp : qpsi)
    {
        auto phi = cb.eval_basis(qp.first);
        auto n = level_set_function.normal(qp.first);

        ret -= qp.second * bcs_fun(qp.first).dot(n) * phi;
    }

    return ret;
}

////////////////////////  SYMMETRICAL GRADIENT RECONSTRUCTIONS
// coeff scales the interface term
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_sym_matrix
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const Function& level_set_function, const hho_degree_info& di, element_location where,
 const T coeff)
{
    if ( !is_cut(msh, cl) )
        return make_hho_gradrec_sym_matrix(msh, cl, di);

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    //typedef Matrix<T, Dynamic, 1>       vector_type;
    std::cout<<"make_hho_gradrec_sym_matrix: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T>         cb(msh, cl, celdeg);
    sym_matrix_cell_basis<cuthho_mesh<T, ET>,T>     gb(msh, cl, graddeg);


    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = sym_matrix_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type        gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type        gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
    for (auto& qp : qps)
    {
        const auto c_dphi = cb.eval_gradients(qp.first);
        const auto g_phi  = gb.eval_basis(qp.first);

        gr_lhs.block(0, 0, gbs, gbs) += qp.second * inner_product(g_phi, g_phi);
        // we use here the symmetry of the basis gb
        gr_rhs.block(0, 0, gbs, cbs) += qp.second * inner_product(g_phi, c_dphi);
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        cut_vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const matrix_type c_phi      = cb.eval_basis(qp.first);
            const matrix_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const matrix_type qp_g_phi_n = qp.second * outer_product(g_phi, n);

            gr_rhs.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            gr_rhs.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }


    // interface term (scaled by coeff)
    matrix_type    interface_term = matrix_type::Zero(gbs, cbs);
    const auto iqps = integrate_interface(msh, cl, celdeg + graddeg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto c_phi        = cb.eval_basis(qp.first);
        const auto g_phi        = gb.eval_basis(qp.first);

        Matrix<T,2,1> n = level_set_function.normal(qp.first);
        const matrix_type qp_g_phi_n = qp.second * outer_product(g_phi, n);

        interface_term -= qp_g_phi_n * c_phi.transpose();
    }
    gr_rhs.block(0, 0, gbs, cbs) += coeff * interface_term;

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = 2.0 * gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}

//// make_hho_gradrec_sym_matrix_interface
// return the gradient reconstruction for the interface pb
// coeff -> scales the interface term
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_sym_matrix_interface
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const Function& level_set_function, const hho_degree_info& di, element_location where, T coeff)
{
    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    //typedef Matrix<T, Dynamic, 1>       vector_type;
//    std::cout<<"make_hho_gradrec_sym_matrix_interface: Old implementation: normal calculated via level set and not parametric interface."<<std::endl;
    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T>        cb(msh, cl, celdeg);
    sym_matrix_cell_basis<cuthho_mesh<T, ET>,T>    gb(msh, cl, graddeg);


    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = sym_matrix_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type       rhs_tmp = matrix_type::Zero(gbs, cbs + num_faces * fbs);
    matrix_type        gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type        gr_rhs = matrix_type::Zero(gbs, 2*cbs + 2*num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
    for (auto& qp : qps)
    {
        const auto c_dphi = cb.eval_gradients(qp.first);
        const auto g_phi  = gb.eval_basis(qp.first);

        gr_lhs.block(0, 0, gbs, gbs) += qp.second * inner_product(g_phi, g_phi);
        rhs_tmp.block(0, 0, gbs, cbs) += qp.second * inner_product(g_phi, c_dphi);
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        cut_vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const matrix_type c_phi      = cb.eval_basis(qp.first);
            const matrix_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const matrix_type qp_g_phi_n = qp.second * outer_product(g_phi, n);

            rhs_tmp.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            rhs_tmp.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }

    // term on the interface
    size_t phi_n_deg = level_set_function.grad_deg ;
    matrix_type        interface_term = matrix_type::Zero(gbs, 2*cbs);
    const auto iqps = integrate_interface(msh, cl, celdeg + graddeg + phi_n_deg, element_location::IN_NEGATIVE_SIDE);
    for (auto& qp : iqps)
    {
        const auto c_phi        = cb.eval_basis(qp.first);
        const auto g_phi        = gb.eval_basis(qp.first);

        Matrix<T,2,1> n = level_set_function.normal(qp.first);
        const matrix_type qp_g_phi_n = qp.second * outer_product(g_phi, n);

        interface_term.block(0 , 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        interface_term.block(0 , cbs, gbs, cbs) += qp_g_phi_n * c_phi.transpose();
    }
    gr_rhs.block(0, 0, gbs, 2*cbs) += coeff * interface_term;

    // other terms
    if(where == element_location::IN_NEGATIVE_SIDE)
    {
        gr_rhs.block(0, 0, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs, gbs, num_faces*fbs)
            += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }
    else if( where == element_location::IN_POSITIVE_SIDE)
    {
        gr_rhs.block(0, cbs, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs + num_faces*fbs, gbs, num_faces*fbs)
                     += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = 2 * gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}

//// make_hho_gradrec_sym_matrix_interface_ref_pts
// return the gradient reconstruction for the interface pb
// coeff -> scales the interface term
// Use the normal provided by the parametric curve
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_sym_matrix_interface_ref_pts
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const Function& level_set_function, const hho_degree_info& di, element_location where, T coeff)
{
    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    //typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T>        cb(msh, cl, celdeg);
    sym_matrix_cell_basis<cuthho_mesh<T, ET>,T>    gb(msh, cl, graddeg);


    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = sym_matrix_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type       rhs_tmp = matrix_type::Zero(gbs, cbs + num_faces * fbs);
    matrix_type        gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type        gr_rhs = matrix_type::Zero(gbs, 2*cbs + 2*num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
    for (auto& qp : qps)
    {
        const auto c_dphi = cb.eval_gradients(qp.first);
        const auto g_phi  = gb.eval_basis(qp.first);

        gr_lhs.block(0, 0, gbs, gbs) += qp.second * inner_product(g_phi, g_phi);
        rhs_tmp.block(0, 0, gbs, cbs) += qp.second * inner_product(g_phi, c_dphi);
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        cut_vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const matrix_type c_phi      = cb.eval_basis(qp.first);
            const matrix_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const matrix_type qp_g_phi_n = qp.second * outer_product(g_phi, n);

            rhs_tmp.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            rhs_tmp.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }

    // term on the interface
    matrix_type        interface_term = matrix_type::Zero(gbs, 2*cbs);
    
    auto msh_int = cl.user_data.integration_msh ;
    size_t degree_curve = msh_int.degree_curve ;
    Interface_parametrisation_mesh1d curve(degree_curve) ;
    size_t degree = celdeg + graddeg;
    degree += 2*degree_curve - 2 ; //  2*degree_curve ; // TO BE CHECKED
    auto qpsi = edge_quadrature<T>(degree);
    
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
        auto pts = points(msh_int,msh_int.cells[i_cell]);
        for(auto& qp:qpsi)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = curve(t , pts , degree_curve ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = curve.jacobian( t , pts , degree_curve ) ;
            auto w = 0.5 * qp.second * jacobian ;
            
            
            
            const auto c_phi        = cb.eval_basis(pt);
            const auto g_phi        = gb.eval_basis(pt);

//            Matrix<T,2,1> n = level_set_function.normal(qp.first);
            Matrix<T,2,1> n = curve.normal(t , pts , degree_curve ) ;
            const matrix_type qp_g_phi_n = w * outer_product(g_phi, n);

            interface_term.block(0 , 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
            interface_term.block(0 , cbs, gbs, cbs) += qp_g_phi_n * c_phi.transpose();
            
        }
    }
    
    gr_rhs.block(0, 0, gbs, 2*cbs) += coeff * interface_term;

    // other terms
    if(where == element_location::IN_NEGATIVE_SIDE)
    {
        gr_rhs.block(0, 0, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs, gbs, num_faces*fbs)
            += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }
    else if( where == element_location::IN_POSITIVE_SIDE)
    {
        gr_rhs.block(0, cbs, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs + num_faces*fbs, gbs, num_faces*fbs)
                     += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = 2 * gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}

//// make_hho_gradrec_sym_matrix_interface_ref_pts
// return the gradient reconstruction for the interface pb
// coeff -> scales the interface term
// Use the normal provided by the parametric curve
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_sym_matrix_parametric_interface
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const Function& parametric_interface, const hho_degree_info& di, element_location where, T coeff)
{
    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef typename cuthho_mesh<T, ET>::point_type point_type;
    //typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T>        cb(msh, cl, celdeg);
    sym_matrix_cell_basis<cuthho_mesh<T, ET>,T>    gb(msh, cl, graddeg);


    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = sym_matrix_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type       rhs_tmp = matrix_type::Zero(gbs, cbs + num_faces * fbs);
    matrix_type        gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type        gr_rhs = matrix_type::Zero(gbs, 2*cbs + 2*num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
    for (auto& qp : qps)
    {
        const auto c_dphi = cb.eval_gradients(qp.first);
        const auto g_phi  = gb.eval_basis(qp.first);

        gr_lhs.block(0, 0, gbs, gbs) += qp.second * inner_product(g_phi, g_phi);
        rhs_tmp.block(0, 0, gbs, cbs) += qp.second * inner_product(g_phi, c_dphi);
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        cut_vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const matrix_type c_phi      = cb.eval_basis(qp.first);
            const matrix_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const matrix_type qp_g_phi_n = qp.second * outer_product(g_phi, n);

            rhs_tmp.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            rhs_tmp.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }

    // term on the interface
    size_t degree = celdeg + graddeg;
    matrix_type        interface_term = matrix_type::Zero(gbs, 2*cbs);
    
    
    auto para_cells_i = parametric_interface.get_global_interface(true ,msh, cl);
    

    degree += (parametric_interface.der_degree + parametric_interface.degree_det) ;
    auto qpsi = edge_quadrature<T>(degree);
    
    for(auto& local_func : para_cells_i)
    {
        size_t cell_i = local_func.first ;
        T a = local_func.second.first;
        T b = local_func.second.second;
        for(auto& qp:qpsi)
        {
            auto t = 0.5*(qp.first.x()+1.0)*(b-a) + a ;
            T jacobian = parametric_interface.jacobian( t , cell_i ) ;
            auto w = 0.5 * qp.second * jacobian *(b-a) ;

            auto p = parametric_interface(t , cell_i ) ;
            point<T,2> pt = point_type( p(0) , p(1) ) ;
            
            const auto c_phi        = cb.eval_basis(pt);
            const auto g_phi        = gb.eval_basis(pt);

            Matrix<T,2,1> n = parametric_interface.normal(t , cell_i ) ;
            const matrix_type qp_g_phi_n = w * outer_product(g_phi, n);

            interface_term.block(0 , 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
            interface_term.block(0 , cbs, gbs, cbs) += qp_g_phi_n * c_phi.transpose();
        
            
        }
    }
   
    
    gr_rhs.block(0, 0, gbs, 2*cbs) += coeff * interface_term;

    // other terms
    if(where == element_location::IN_NEGATIVE_SIDE)
    {
        gr_rhs.block(0, 0, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs, gbs, num_faces*fbs)
            += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }
    else if( where == element_location::IN_POSITIVE_SIDE)
    {
        gr_rhs.block(0, cbs, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs + num_faces*fbs, gbs, num_faces*fbs)
                     += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = 2 * gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}

//// make_hho_gradrec_sym_matrix_interface_ref_pts
// return the gradient reconstruction for the interface pb
// coeff -> scales the interface term
// Use the normal provided by the parametric curve
template<typename T, size_t ET, typename Function>
std::pair<   Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename cuthho_mesh<T, ET>::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_sym_matrix_interface_ref_pts_cont
(const cuthho_mesh<T, ET>& msh, const typename cuthho_mesh<T, ET>::cell_type& cl,
 const Function& parametric_interface, const hho_degree_info& di, element_location where, T coeff)
{
    if ( !is_cut(msh, cl) )
        throw std::invalid_argument("The cell is not cut");

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    //typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    vector_cell_basis<cuthho_mesh<T, ET>,T>        cb(msh, cl, celdeg);
    sym_matrix_cell_basis<cuthho_mesh<T, ET>,T>    gb(msh, cl, graddeg);


    auto cbs = vector_cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
    auto fbs = vector_face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
    auto gbs = sym_matrix_cell_basis<cuthho_mesh<T, ET>,T>::size(graddeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type       rhs_tmp = matrix_type::Zero(gbs, cbs + num_faces * fbs);
    matrix_type        gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type        gr_rhs = matrix_type::Zero(gbs, 2*cbs + 2*num_faces * fbs);

    const auto qps = integrate(msh, cl, celdeg - 1 + facdeg, where);
    for (auto& qp : qps)
    {
        const auto c_dphi = cb.eval_gradients(qp.first);
        const auto g_phi  = gb.eval_basis(qp.first);

        gr_lhs.block(0, 0, gbs, gbs) += qp.second * inner_product(g_phi, g_phi);
        rhs_tmp.block(0, 0, gbs, cbs) += qp.second * inner_product(g_phi, c_dphi);
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        cut_vector_face_basis<cuthho_mesh<T, ET>,T> fb(msh, fc, facdeg, where);

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg), where);
        for (auto& qp : qps_f)
        {
            const matrix_type c_phi      = cb.eval_basis(qp.first);
            const matrix_type f_phi      = fb.eval_basis(qp.first);
            const auto        g_phi      = gb.eval_basis(qp.first);
            const matrix_type qp_g_phi_n = qp.second * outer_product(g_phi, n);

            rhs_tmp.block(0, cbs + i * fbs, gbs, fbs) += qp_g_phi_n * f_phi.transpose();
            rhs_tmp.block(0, 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
        }
    }

    // term on the interface
    matrix_type        interface_term = matrix_type::Zero(gbs, 2*cbs);
    
    auto global_cells_i = parametric_interface.get_global_cells_interface(msh , cl);
    auto msh_int = cl.user_data.integration_msh ;
//    size_t degree_normal = parametric_interface.der_degree ;
    size_t degree = celdeg + graddeg;
    degree += (parametric_interface.der_degree + parametric_interface.degree_det) ; // IT WAS 3* degree_curvature -2
    auto qpsi = edge_quadrature<T>(degree);
    
   
    for (size_t i_cell = 0; i_cell < msh_int.cells.size(); i_cell++)
    {
//        auto pts = points(msh_int,msh_int.cells[i_cell]);
        size_t global_cl_i = global_cells_i[i_cell] ;
        for(auto& qp:qpsi)
        {
            auto t = 0.5 * qp.first.x() + 0.5;
            auto p = parametric_interface(t , global_cl_i ) ;
            point<T,2> pt = typename cuthho_mesh<T, ET>::point_type( p(0) , p(1) ) ;
            T jacobian = parametric_interface.jacobian_cont( t , global_cl_i ) ;
            auto w = 0.5 * qp.second * jacobian ;
            
            
            
            const auto c_phi        = cb.eval_basis(pt);
            const auto g_phi        = gb.eval_basis(pt);

            Matrix<T,2,1> n = parametric_interface.normal_cont(t , global_cl_i ) ;
            const matrix_type qp_g_phi_n = w * outer_product(g_phi, n);

            interface_term.block(0 , 0, gbs, cbs) -= qp_g_phi_n * c_phi.transpose();
            interface_term.block(0 , cbs, gbs, cbs) += qp_g_phi_n * c_phi.transpose();
            
        }
    }
    
    gr_rhs.block(0, 0, gbs, 2*cbs) += coeff * interface_term;

    // other terms
    if(where == element_location::IN_NEGATIVE_SIDE)
    {
        gr_rhs.block(0, 0, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs, gbs, num_faces*fbs)
            += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }
    else if( where == element_location::IN_POSITIVE_SIDE)
    {
        gr_rhs.block(0, cbs, gbs, cbs) += rhs_tmp.block(0, 0, gbs, cbs);
        gr_rhs.block(0, 2*cbs + num_faces*fbs, gbs, num_faces*fbs)
                     += rhs_tmp.block(0, cbs, gbs, num_faces*fbs);
    }

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = 2 * gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}

