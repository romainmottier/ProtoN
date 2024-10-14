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

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "core/core"

//#include "src/methods/cuthho_bits/cuthho_geom.hpp"
//#include "src/methods/cuthho_bits/cuthho_utils.hpp"
#include "src/methods/cuthho"

//////////////////////////////////////// LAPLACIAN ////////////////////////////////////////
template<typename Mesh>
std::pair<   Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>  >
make_hho_laplacian(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;

    auto recdeg = di.reconstruction_degree();
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    cell_basis<Mesh,T>     cb(msh, cl, recdeg);

    auto rbs = cell_basis<Mesh,T>::size(recdeg);
    auto cbs = cell_basis<Mesh,T>::size(celdeg);
    auto fbs = face_basis<Mesh,T>::size(facdeg);

    auto fcs = faces(msh, cl);

    Matrix<T, Dynamic, Dynamic> stiff = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);
    Matrix<T, Dynamic, Dynamic> gr_lhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs-1, rbs-1);
    Matrix<T, Dynamic, Dynamic> gr_rhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs-1, cbs + fcs.size()*fbs);

    auto qps = integrate(msh, cl, 2*recdeg);

    for (auto& qp : qps)
    {
        auto dphi = cb.eval_gradients(qp.first);
        stiff += qp.second * dphi * dphi.transpose();
    }

    gr_lhs = stiff.block(1, 1, rbs-1, rbs-1);
    gr_rhs.block(0, 0, rbs-1, cbs) = stiff.block(1, 0, rbs-1, cbs);

    auto ns = normals(msh, cl);

    for (size_t i = 0; i < fcs.size(); i++)
    {
        auto fc = fcs[i];
        auto n = ns[i];
        face_basis<Mesh,T> fb(msh, fc, facdeg);

        auto qps_f = integrate(msh, fc, 2*facdeg);
        for (auto& qp : qps_f)
        {
            Matrix<T, Dynamic, 1> c_phi_tmp = cb.eval_basis(qp.first);
            Matrix<T, Dynamic, 1> c_phi = c_phi_tmp.head(cbs);
            Matrix<T, Dynamic, 2> c_dphi_tmp = cb.eval_gradients(qp.first);
            Matrix<T, Dynamic, 2> c_dphi = c_dphi_tmp.block(1, 0, rbs-1, 2);
            Matrix<T, Dynamic, 1> f_phi = fb.eval_basis(qp.first);
            gr_rhs.block(0, cbs+i*fbs, rbs-1, fbs) += qp.second * (c_dphi * n) * f_phi.transpose();
            gr_rhs.block(0, 0, rbs-1, cbs) -= qp.second * (c_dphi * n) * c_phi.transpose();
        }
    }

    //std::cout << cyan << gr_lhs << nocolor << std::endl;
    //std::cout << "Determinant is " << gr_lhs.determinant() << std::endl;
    //std::cout << "Condition number is " << condition_number(gr_lhs) << std::endl << std::endl;
    //std::cout << magenta << gr_rhs << nocolor << std::endl << std::endl;

    Matrix<T, Dynamic, Dynamic> oper = gr_lhs.llt().solve(gr_rhs);
    Matrix<T, Dynamic, Dynamic> data = gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}

template<typename Mesh>
std::pair<   Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>  >
make_hho_mixed_laplacian(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;

    auto recdeg = di.reconstruction_degree();
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    cell_basis<Mesh,T>     cb(msh, cl, recdeg);

    auto rbs = cell_basis<Mesh,T>::size(recdeg);
    auto cbs = cell_basis<Mesh,T>::size(celdeg);
    auto fbs = face_basis<Mesh,T>::size(facdeg);

    auto fcs = faces(msh, cl);

    Matrix<T, Dynamic, Dynamic> stiff = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);
    Matrix<T, Dynamic, Dynamic> gr_lhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs-1, rbs-1);
    Matrix<T, Dynamic, Dynamic> gr_rhs = Matrix<T, Dynamic, Dynamic>::Zero(rbs-1, cbs + fcs.size()*fbs);

    auto qps = integrate(msh, cl, 2*recdeg);

    for (auto& qp : qps)
    {
        auto dphi = cb.eval_gradients(qp.first);
        stiff += qp.second * dphi * dphi.transpose();
    }

    gr_lhs = stiff.block(1, 1, rbs-1, rbs-1);
    gr_rhs.block(0, 0, rbs-1, cbs) = stiff.block(1, 0, rbs-1, cbs);

    auto ns = normals(msh, cl);

    for (size_t i = 0; i < fcs.size(); i++)
    {
        auto fc = fcs[i];
        auto n = ns[i];
        face_basis<Mesh,T> fb(msh, fc, facdeg);

        auto qps_f = integrate(msh, fc, 2*facdeg);
        for (auto& qp : qps_f)
        {
            Matrix<T, Dynamic, 1> c_phi_tmp = cb.eval_basis(qp.first);
            Matrix<T, Dynamic, 1> c_phi = c_phi_tmp.head(cbs);
            Matrix<T, Dynamic, 2> c_dphi_tmp = cb.eval_gradients(qp.first);
            Matrix<T, Dynamic, 2> c_dphi = c_dphi_tmp.block(1, 0, rbs-1, 2);
            Matrix<T, Dynamic, 1> f_phi = fb.eval_basis(qp.first);
            gr_rhs.block(0, cbs+i*fbs, rbs-1, fbs) += qp.second * (c_dphi * n) * f_phi.transpose();
            gr_rhs.block(0, 0, rbs-1, cbs) -= qp.second * (c_dphi * n) * c_phi.transpose();
        }
    }

    auto vec_cell_size = gr_lhs.cols();
    auto nrows = gr_rhs.cols()+vec_cell_size;
    auto ncols = gr_rhs.cols()+vec_cell_size;
    
    // Shrinking data
    Matrix<T, Dynamic, Dynamic> data_mixed = Matrix<T, Dynamic, Dynamic>::Zero(nrows,ncols);
    data_mixed.block(0, 0, vec_cell_size, vec_cell_size) = gr_lhs;
    data_mixed.block(0, vec_cell_size, vec_cell_size, ncols-vec_cell_size) = -gr_rhs;
    data_mixed.block(vec_cell_size, 0, nrows-vec_cell_size, vec_cell_size) = gr_rhs.transpose();
    
    Matrix<T, Dynamic, Dynamic> oper = gr_lhs.llt().solve(gr_rhs);
    Matrix<T, Dynamic, Dynamic> data = gr_rhs.transpose() * oper; // Candidate for deletion
    return std::make_pair(oper, data_mixed);
}

//////////////////////////////////////// STABILIZATION ////////////////////////////////////////
template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_hho_naive_stabilization(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info& di, bool scaled_Q = true)
{
    using T = typename Mesh::coordinate_type;

    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    auto cbs = cell_basis<Mesh,T>::size(celdeg);
    auto fbs = face_basis<Mesh,T>::size(facdeg);

    auto fcs = faces(msh, cl);

    size_t msize = cbs+fcs.size()*fbs;
    Matrix<T, Dynamic, Dynamic> data = Matrix<T, Dynamic, Dynamic>::Zero(msize, msize);
    Matrix<T, Dynamic, Dynamic> If = Matrix<T, Dynamic, Dynamic>::Identity(fbs, fbs);

    cell_basis<Mesh,T> cb(msh, cl, celdeg);

    // auto h = measure(msh, cl);
    auto h = diameter(msh, cl);

    for (size_t i = 0; i < fcs.size(); i++) {
        auto fc = fcs[i];
        face_basis<Mesh,T> fb(msh, fc, facdeg);

        Matrix<T, Dynamic, Dynamic> oper = Matrix<T, Dynamic, Dynamic>::Zero(fbs, msize);
        Matrix<T, Dynamic, Dynamic> mass = Matrix<T, Dynamic, Dynamic>::Zero(fbs, fbs);
        Matrix<T, Dynamic, Dynamic> trace = Matrix<T, Dynamic, Dynamic>::Zero(fbs, cbs);

        oper.block(0, cbs+i*fbs, fbs, fbs) = -If;

        auto qps = integrate(msh, fc, 2*facdeg + 1);
        for (auto& qp : qps)
        {
            auto c_phi = cb.eval_basis(qp.first);
            auto f_phi = fb.eval_basis(qp.first);

            mass  += qp.second * f_phi * f_phi.transpose();
            trace += qp.second * f_phi * c_phi.transpose();
        }

        oper.block(0, 0, fbs, cbs) = mass.llt().solve(trace);
        
        if (scaled_Q) {
            data += oper.transpose() * mass * oper * (1.0/h);
        }
        else {
            data += oper.transpose() * mass * oper;
        }
    }

    return data;
}

template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_hho_fancy_stabilization(const Mesh& msh, const typename Mesh::cell_type& cl,
                             const Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic> reconstruction,
                             const hho_degree_info& di, bool scaled_Q = true)
{
    using T = typename Mesh::coordinate_type;

    auto recdeg = di.reconstruction_degree();
    auto celdeg = di.cell_degree();
    auto facdeg = di.face_degree();

    auto rbs = cell_basis<Mesh,T>::size(recdeg);
    auto cbs = cell_basis<Mesh,T>::size(celdeg);
    auto fbs = face_basis<Mesh,T>::size(facdeg);

    cell_basis<Mesh,T> cb(msh, cl, recdeg);

    Matrix<T, Dynamic, Dynamic> mass_mat = Matrix<T, Dynamic, Dynamic>::Zero(rbs, rbs);
    auto cell_quadpoints = integrate(msh, cl, 2*recdeg);
    for (auto& qp : cell_quadpoints)
    {
        auto c_phi = cb.eval_basis(qp.first);
        mass_mat += qp.second * c_phi * c_phi.transpose();
    }

    // Build \pi_F^k (v_F - P_T^K v) equations (21) and (22)

    //Step 1: compute \pi_T^k p_T^k v (third term).
    Matrix<T, Dynamic, Dynamic> M1 = mass_mat.block(0, 0, cbs, cbs);
    Matrix<T, Dynamic, Dynamic> M2 = mass_mat.block(0, 1, cbs, rbs-1);
    Matrix<T, Dynamic, Dynamic> proj1 = -M1.llt().solve(M2*reconstruction);

    //Step 2: v_T - \pi_T^k p_T^k v (first term minus third term)
    Matrix<T, Dynamic, Dynamic> I_T = Matrix<T, Dynamic, Dynamic>::Identity(cbs, cbs);
    proj1.block(0, 0, cbs, cbs) += I_T;

    auto fcs = faces(msh, cl);
    auto num_faces = fcs.size();
    size_t msize = cbs+num_faces*fbs;

    Matrix<T, Dynamic, Dynamic> data = Matrix<T, Dynamic, Dynamic>::Zero(msize, msize);

    // Step 3: project on faces (eqn. 21)
    for (size_t face_i = 0; face_i < num_faces; face_i++)
    {
        auto h = diameter(msh, /*fcs[face_i]*/cl);
        auto fc = fcs[face_i];
        face_basis<Mesh,T> fb(msh, fc, facdeg);

        Matrix<T, Dynamic, Dynamic> face_mass_matrix    = Matrix<T, Dynamic, Dynamic>::Zero(fbs, fbs);
        Matrix<T, Dynamic, Dynamic> face_trace_matrix   = Matrix<T, Dynamic, Dynamic>::Zero(fbs, rbs);

        auto face_quadpoints = integrate(msh, fc, 2*facdeg);
        for (auto& qp : face_quadpoints)
        {
            auto f_phi = fb.eval_basis(qp.first);
            auto c_phi = cb.eval_basis(qp.first);
            auto q_f_phi = qp.second * f_phi;
            face_mass_matrix += q_f_phi * f_phi.transpose();
            face_trace_matrix += q_f_phi * c_phi.transpose();
        }

        LLT<Matrix<T, Dynamic, Dynamic>> piKF;
        piKF.compute(face_mass_matrix);

        // Step 3a: \pi_F^k( v_F - p_T^k v )
        Matrix<T, Dynamic, Dynamic> MR1 = face_trace_matrix.block(0, 1, fbs, rbs-1);

        Matrix<T, Dynamic, Dynamic> proj2 = piKF.solve(MR1*reconstruction);
        Matrix<T, Dynamic, Dynamic> I_F = Matrix<T, Dynamic, Dynamic>::Identity(fbs, fbs);
        proj2.block(0, cbs+face_i*fbs, fbs, fbs) -= I_F;

        // Step 3b: \pi_F^k( v_T - \pi_T^k p_T^k v )
        Matrix<T, Dynamic, Dynamic> MR2 = face_trace_matrix.block(0, 0, fbs, cbs);
        Matrix<T, Dynamic, Dynamic> proj3 = piKF.solve(MR2*proj1);
        Matrix<T, Dynamic, Dynamic> BRF = proj2 + proj3;

        if (scaled_Q) {
            data += BRF.transpose() * face_mass_matrix * BRF / h;
        }else{
            data += BRF.transpose() * face_mass_matrix * BRF;
        }
        
    }

    return data;
}

//////////////////////////////////////// GRADIENT RECONSTRUCTION ////////////////////////////////////////
template<typename Mesh>
std::pair<   Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_vector(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info& di) {
    
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    cell_basis<Mesh,T>        cb(msh, cl, celdeg);
    vector_cell_basis<Mesh,T> gb(msh, cl, graddeg);

    auto cbs = cell_basis<Mesh,T>::size(celdeg);
    auto fbs = face_basis<Mesh,T>::size(facdeg);
    auto gbs = vector_cell_basis<Mesh,T>::size(graddeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);

    if(celdeg > 0) {
        const auto qps = integrate(msh, cl, celdeg - 1 + facdeg);
        for (auto& qp : qps) {
            const auto c_dphi = cb.eval_gradients(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);
            gr_lhs.block(0, 0, gbs, gbs) += qp.second * g_phi * g_phi.transpose();
            gr_rhs.block(0, 0, gbs, cbs) += qp.second * g_phi * c_dphi.transpose();
        }
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++) {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        face_basis<Mesh,T> fb(msh, fc, facdeg);
        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg));
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

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}

template<typename Mesh>
std::pair<   Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_mixed_vector(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    cell_basis<Mesh,T>            cb(msh, cl, celdeg);
    vector_cell_basis<Mesh,T>     gb(msh, cl, graddeg);

    auto cbs = cell_basis<Mesh,T>::size(celdeg);
    auto fbs = face_basis<Mesh,T>::size(facdeg);
    auto gbs = vector_cell_basis<Mesh,T>::size(graddeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);

    if(celdeg > 0)
    {
        const auto qps = integrate(msh, cl, celdeg - 1 + facdeg);
        for (auto& qp : qps)
        {
            const auto c_dphi = cb.eval_gradients(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);

            gr_lhs.block(0, 0, gbs, gbs) += qp.second * g_phi * g_phi.transpose();
            gr_rhs.block(0, 0, gbs, cbs) += qp.second * g_phi * c_dphi.transpose();
        }
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        face_basis<Mesh,T> fb(msh, fc, facdeg);

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg));
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
    
    auto vec_cell_size = gr_lhs.cols();
    auto nrows = gr_rhs.cols()+vec_cell_size;
    auto ncols = gr_rhs.cols()+vec_cell_size;
    
    // Shrinking data
    matrix_type data_mixed = matrix_type::Zero(nrows,ncols);
    data_mixed.block(0, vec_cell_size, vec_cell_size, ncols-vec_cell_size) = -gr_rhs;
    data_mixed.block(vec_cell_size, 0, nrows-vec_cell_size, vec_cell_size) = gr_rhs.transpose();
    
    matrix_type oper = gr_lhs.llt().solve(gr_rhs);
    return std::make_pair(oper, data_mixed);

}

/////////////////////////  ASSEMBLERS  ////////////////////////////
template<typename Mesh>
class assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 compress_table;
    std::vector<size_t>                 expand_table;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

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

    assembler(const Mesh& msh, hho_degree_info hdi)
        : di(hdi)
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
            
    assembler(const Mesh& msh, hho_degree_info hdi, bool mixed_Q)
        : di(hdi)
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

        auto recdeg = di.reconstruction_degree();
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg)+(cell_basis<Mesh,T>::size(recdeg)-1);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto system_size = cbs * msh.cells.size() + fbs * num_other_faces;

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }
            
    std::vector<size_t> & get_compress_table() {
            return compress_table;
    }
            
    std::vector<size_t> & get_expand_table() {
            return expand_table;
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

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dirichlet_bf);
                dirichlet_data.block(cbs+face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
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
    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl, const Matrix<T, Dynamic, 1>& rhs,
             const Function& dirichlet_bf)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset    = cell_offset * cbs;
        
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
    void
    assemble_mixed(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs,
             const Function& dirichlet_bf)
    {
        auto recdeg = di.reconstruction_degree();
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto rbs = cell_basis<Mesh,T>::size(recdeg)-1;
        auto cbs = cell_basis<Mesh,T>::size(celdeg) + rbs;
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

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dirichlet_bf);
                dirichlet_data.block(cbs+face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
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
    void
    assemble_mixed_RHS(const Mesh& msh, const typename Mesh::cell_type& cl,
    const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs,
    const Function& dirichlet_bf)
    {
        auto recdeg = di.reconstruction_degree();
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto rbs = cell_basis<Mesh,T>::size(recdeg)-1;
        auto cbs = cell_basis<Mesh,T>::size(celdeg) + rbs;
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

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dirichlet_bf);
                dirichlet_data.block(cbs+face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
            }
        }

        assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
            if ( asm_map[j].assemble() ){
//                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
            }else{
                    RHS[asm_map[i]] -= lhs(i,j)*dirichlet_data(j);
            }
            
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
auto make_assembler(const Mesh& msh, hho_degree_info hdi)
{
    return assembler<Mesh>(msh, hdi);
}

template<typename Mesh>
auto make_assembler(const Mesh& msh, hho_degree_info hdi, bool mixed_Q)
{
    return assembler<Mesh>(msh, hdi, mixed_Q);
}

/* Assembler for the obstacle problem (see "Bubbles enriched quadratic finite
 * element method for the 3D-elliptic obstacle problem - S. Gaddam, T. Gudi",
 * eqn. 5.1 onwards) */
template<typename Mesh>
class obstacle_assembler
{
    using T = typename Mesh::coordinate_type;
    std::vector<size_t>                 face_ct, A_ct, B_ct; //compress tables
    std::vector<size_t>                 face_et, A_et, B_et; //expand tables
    std::vector<bool>                   is_in_set_A;

    hho_degree_info                     di;

    std::vector< Triplet<T> >           triplets;

    size_t      num_all_faces, num_dirichlet_faces, num_other_faces;
    size_t      num_all_cells, num_A_cells, num_I_cells;

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

    obstacle_assembler(const Mesh& msh, const std::vector<bool>& in_A, hho_degree_info hdi)
        : di(hdi), is_in_set_A(in_A)
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
        };

        num_all_faces = msh.faces.size();
        num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
        num_other_faces = num_all_faces - num_dirichlet_faces;

        assert( is_in_set_A.size() == msh.cells.size() );

        num_all_cells = msh.cells.size();
        num_A_cells = std::count(is_in_set_A.begin(), is_in_set_A.end(), true);
        num_I_cells = std::count(is_in_set_A.begin(), is_in_set_A.end(), false);

        /* Make A tables: keep the unknowns of cells in set I */
        A_ct.resize( num_all_cells );
        A_et.resize( num_I_cells );
        for (size_t i = 0, co = 0; i < num_all_cells; i++)
        {
            auto cl = msh.cells[i];
            if ( !is_in_set_A.at(i) )
            {
                A_ct.at(i) = co;
                A_et.at(co) = i;
                co++;
            }
        }

        /* Make face tables */
        face_ct.resize( num_all_faces );
        face_et.resize( num_other_faces );
        for (size_t i = 0, co = 0; i < num_all_faces; i++)
        {
            auto fc = msh.faces[i];
            if ( !is_dirichlet(fc) )
            {
                face_ct.at(i) = co;
                face_et.at(co) = i;
                co++;
            }
        }

        /* Make B tables: keep the unknowns of cells in set A */
        B_ct.resize( num_all_cells );
        B_et.resize( num_A_cells );
        for (size_t i = 0, co = 0; i < num_all_cells; i++)
        {
            auto cl = msh.cells[i];
            if ( is_in_set_A.at(i) )
            {
                B_ct.at(i) = co;
                B_et.at(co) = i;
                co++;
            }
        }

        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        auto system_size = cbs * (num_I_cells + num_A_cells) + fbs * num_other_faces;

        assert( system_size == cbs * msh.cells.size() + fbs * num_other_faces );

        LHS = SparseMatrix<T>( system_size, system_size );
        RHS = Matrix<T, Dynamic, 1>::Zero( system_size );
    }

    void dump_tables() const
    {
        std::cout << "Compress table for A: " << std::endl;
        for (size_t i = 0; i < A_ct.size(); i++)
            std::cout << i << " -> " << A_ct.at(i) << std::endl;

        std::cout << "Compress table for faces: " << std::endl;
        for (size_t i = 0; i < face_ct.size(); i++)
            std::cout << i << " -> " << face_ct.at(i) << std::endl;

        std::cout << "Compress table for B: " << std::endl;
        for (size_t i = 0; i < B_ct.size(); i++)
            std::cout << i << " -> " << B_ct.at(i) << std::endl;
    }

    template<typename Function>
    void
    assemble(const Mesh& msh, const typename Mesh::cell_type& cl,
             const Matrix<T, Dynamic, Dynamic>& lhs, const Matrix<T, Dynamic, 1>& rhs,
             const Matrix<T, Dynamic, 1>& gamma, const Function& dirichlet_bf)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        std::vector<assembly_index> asm_map_row, asm_map_col;

        /* Cell dofs local to global */
        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset    = A_ct.at(cell_offset) * cbs;
        bool cell_needs_asm_A   = !is_in_set_A.at(cell_offset);
        bool cell_needs_asm_B   = is_in_set_A.at(cell_offset);

        for (size_t i = 0; i < cbs; i++)
        {
            asm_map_row.push_back( assembly_index(cell_offset+i, true) );
            asm_map_col.push_back( assembly_index(cell_LHS_offset+i, cell_needs_asm_A) );
        }

        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();

        /* Face dofs local to global */
        Matrix<T, Dynamic, 1> dirichlet_data = Matrix<T, Dynamic, 1>::Zero(cbs + num_faces*fbs);
        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_LHS_offset_rows = cbs * num_all_cells + face_ct.at(face_offset)*fbs;
            auto face_LHS_offset_cols = cbs * num_I_cells + face_ct.at(face_offset)*fbs;

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            for (size_t i = 0; i < fbs; i++)
            {
                asm_map_row.push_back( assembly_index(face_LHS_offset_rows+i, !dirichlet) );
                asm_map_col.push_back( assembly_index(face_LHS_offset_cols+i, !dirichlet) );
            }

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dirichlet_bf);
                dirichlet_data.block(cbs+face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
            }
        }

        //assert( asm_map.size() == lhs.rows() && asm_map.size() == lhs.cols() );

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if ( !asm_map_row[i].assemble() )
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( asm_map_col[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map_row[i], asm_map_col[j], lhs(i,j)) );
                else
                {
                    if (j < cbs)
                        RHS[asm_map_row[i]] -= lhs(i,j)*gamma(cell_offset);
                    else
                        RHS[asm_map_row[i]] -= lhs(i,j)*dirichlet_data(j);
                }
            }
        }


        /* Needed both in case A and I */
        RHS.block(cell_offset, 0, cbs, 1) += rhs.block(0, 0, cbs, 1);

        if (cell_needs_asm_B)
        {
            auto offset_row = cell_offset * cbs;
            auto offset_col = num_I_cells * cbs + num_other_faces * fbs + B_ct.at(cell_offset);
            triplets.push_back( Triplet<T>(offset_row, offset_col, 1.0) );
        }

    } // assemble_A()


    template<typename Function>
    void
    expand_solution(const Mesh& msh,
    const Matrix<T, Dynamic, 1>& solution, const Function& dirichlet_bf,
    const Matrix<T, Dynamic, 1>& gamma,
    Matrix<T, Dynamic, 1>& alpha, Matrix<T, Dynamic, 1>& beta)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();

        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto fbs = face_basis<Mesh,T>::size(facdeg);

        alpha.block(0, 0, num_all_cells*cbs, 1) = gamma;
        for (size_t i = 0; i < num_I_cells; i++)
        {
            auto exp_ofs = A_et.at(i);
            alpha.block(exp_ofs*cbs, 0, cbs, 1) = solution.block(i*cbs, 0, cbs, 1);
        }

        beta = Matrix<T, Dynamic, 1>::Zero(msh.cells.size());
        for (size_t i = 0; i < num_A_cells; i++)
        {
            auto exp_ofs = B_et.at(i);
            beta.block(exp_ofs*cbs, 0, cbs, 1) = solution.block(num_I_cells*cbs + num_other_faces*fbs + i*cbs, 0, cbs, 1);
        }

        for (auto& fc : msh.faces)
        {
            auto face_ofs = offset(msh, fc);

            bool dirichlet = fc.is_boundary && fc.bndtype == boundary::DIRICHLET;

            if (dirichlet)
            {
                Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
                Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dirichlet_bf);
                alpha.block(num_all_cells*cbs + face_ofs*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
            }
            else
            {
                auto face_offset = offset(msh, fc);
                auto face_SOL_offset = cbs * num_I_cells + face_ct.at(face_offset)*fbs;
                alpha.block(num_all_cells*cbs + face_ofs*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();
    }
};

template<typename T, typename Mesh>
Matrix<T, Dynamic, 1>
take_local_data(const Mesh& msh, const typename Mesh::cell_type& cl, hho_degree_info di,
                    const Matrix<T, Dynamic, 1>& expanded_solution)
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
        ret.block(0, 0, cbs, 1) = expanded_solution.block(cell_SOL_offset, 0, cbs, 1);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];
            auto face_offset = offset(msh, fc);
            auto face_SOL_offset = cbs * msh.cells.size() + face_offset*fbs;
            ret.block(cbs+face_i*fbs, 0, fbs, 1) = expanded_solution.block(face_SOL_offset, 0, fbs, 1);
        }

        return ret;
    }

template<typename Mesh>
auto make_obstacle_assembler(const Mesh& msh, const std::vector<bool>& in_A, hho_degree_info hdi)
{
    return obstacle_assembler<Mesh>(msh, in_A, hdi);
}

/******************************************************************************************/
/*******************                                               ************************/
/*******************               VECTOR  LAPLACIAN               ************************/
/*******************                                               ************************/
/******************************************************************************************/

template<typename Mesh>
std::pair<   Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_matrix(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info& di)
{
    auto gradrec_vector = make_hho_gradrec_vector(msh, cl, di);
    auto oper = vector_assembly(gradrec_vector.first);
    auto data = vector_assembly(gradrec_vector.second);

    return std::make_pair(oper, data);
}

template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_hho_vector_naive_stabilization(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info& di)
{
    auto scalar_stab = make_hho_naive_stabilization(msh, cl, di);

    return vector_assembly(scalar_stab);
}
           
template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_vector_hho_stabilization(const Mesh&                                                     msh,
                              const typename Mesh::cell_type&                                 cl,
                              const Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>& reconstruction,
                              const hho_degree_info&                                          di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    const auto recdeg = di.reconstruction_degree();
    const auto celdeg = di.cell_degree();
    const auto facdeg = di.face_degree();

    const auto rbs = vector_basis_size(recdeg, Mesh::dimension, Mesh::dimension);
    const auto cbs = vector_basis_size(celdeg, Mesh::dimension, Mesh::dimension);
    const auto fbs = vector_basis_size(facdeg, Mesh::dimension - 1, Mesh::dimension);

    const size_t N = Mesh::dimension;

    auto cb = make_vector_monomial_basis(msh, cl, recdeg);

    const matrix_type mass_mat = make_mass_matrix(msh, cl, cb);

    // Build \pi_F^k (v_F - P_T^K v) equations (21) and (22)

    // Step 1: compute \pi_T^k p_T^k v (third term).
    const matrix_type M1    = mass_mat.block(0, 0, cbs, cbs);
    const matrix_type M2    = mass_mat.block(0, N, cbs, rbs - N);
    matrix_type       proj1 = -M1.ldlt().solve(M2 * reconstruction);

    assert(M2.cols() == reconstruction.rows());

    // Step 2: v_T - \pi_T^k p_T^k v (first term minus third term)
    proj1.block(0, 0, cbs, cbs) += matrix_type::Identity(cbs, cbs);

    const auto fcs       = faces(msh, cl);
    const auto num_faces = fcs.size();

    matrix_type data = matrix_type::Zero(cbs + num_faces * fbs, cbs + num_faces * fbs);

    // Step 3: project on faces (eqn. 21)
    for (size_t face_i = 0; face_i < num_faces; face_i++)
    {
        const auto fc = fcs[face_i];
        const auto hf = diameter(msh, fc);
        auto       fb = make_vector_monomial_basis(msh, fc, facdeg);

        matrix_type face_mass_matrix  = make_mass_matrix(msh, fc, fb);
        matrix_type face_trace_matrix = matrix_type::Zero(fbs, rbs);

        const auto face_quadpoints = integrate(msh, fc, recdeg + facdeg);
        for (auto& qp : face_quadpoints)
        {
            const auto        f_phi   = fb.eval_functions(qp.point());
            const auto        c_phi   = cb.eval_functions(qp.point());
            face_trace_matrix += outer_product(inner_product(qp.weight(), f_phi), c_phi);
        }

        LLT<matrix_type> piKF;
        piKF.compute(face_mass_matrix);

        // Step 3a: \pi_F^k( v_F - p_T^k v )
        const matrix_type MR1 = face_trace_matrix.block(0, N, fbs, rbs - N);

        matrix_type       proj2 = piKF.solve(MR1 * reconstruction);
        proj2.block(0, cbs + face_i * fbs, fbs, fbs) -= matrix_type::Identity(fbs, fbs);

        // Step 3b: \pi_F^k( v_T - \pi_T^k p_T^k v )
        const matrix_type MR2   = face_trace_matrix.block(0, 0, fbs, cbs);
        const matrix_type proj3 = piKF.solve(MR2 * proj1);
        const matrix_type BRF   = proj2 + proj3;

        data += BRF.transpose() * face_mass_matrix * BRF / hf;
    }

    return data;
}


/******************************************************************************************/
/*******************                                               ************************/
/*******************                STOKES PROBLEM                 ************************/
/*******************                                               ************************/
/******************************************************************************************/

template<typename Mesh>
std::pair<   Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>  >
make_hho_divergence_reconstruction(const Mesh& msh, const typename Mesh::cell_type& cl,
                                   const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    const auto celdeg = di.cell_degree();
    const auto facdeg = di.face_degree();
    const auto pdeg = di.face_degree();

    cell_basis<Mesh,T>                   pb(msh, cl, pdeg);
    vector_cell_basis<Mesh,T>            cb(msh, cl, celdeg);

    auto pbs = cell_basis<Mesh,T>::size(pdeg);
    auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
    auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);

    const auto fcs = faces(msh, cl);
    const auto num_faces = fcs.size();
    const auto ns = normals(msh, cl);

    matrix_type dr_lhs = matrix_type::Zero(pbs, pbs);
    matrix_type dr_rhs = matrix_type::Zero(pbs, cbs + num_faces*fbs);


    const auto qps = integrate(msh, cl, celdeg + pdeg - 1);
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
        vector_face_basis<Mesh,T>            fb(msh, fc, facdeg);

        const auto qps_f = integrate(msh, fc, facdeg + pdeg);
        for (auto& qp : qps_f)
        {
            const auto p_phi = pb.eval_basis(qp.first);
            const auto f_phi = fb.eval_basis(qp.first);

            const Matrix<T, Dynamic, 2> p_phi_n = (p_phi * n.transpose());
            dr_rhs.block(0, cbs + i * fbs, pbs, fbs) += qp.second * p_phi_n * f_phi.transpose();
        }
    }


    assert(dr_lhs.rows() == pbs && dr_lhs.cols() == pbs);
    assert(dr_rhs.rows() == pbs && dr_rhs.cols() == cbs + num_faces * fbs);

    matrix_type oper = dr_lhs.ldlt().solve(dr_rhs);
    matrix_type data = dr_rhs;

    return std::make_pair(oper, data);
}

///////  SYMMETRICAL GRADIENT RECONSTRUCTIONS

template<typename Mesh>
std::pair<   Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>,
             Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>  >
make_hho_gradrec_sym_matrix
(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    //typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    vector_cell_basis<Mesh,T>     cb(msh, cl, celdeg);
    sym_matrix_cell_basis<Mesh,T>     gb(msh, cl, graddeg);

    auto cbs = vector_cell_basis<Mesh,T>::size(celdeg);
    auto fbs = vector_face_basis<Mesh,T>::size(facdeg);
    auto gbs = sym_matrix_cell_basis<Mesh,T>::size(graddeg);

    const auto num_faces = faces(msh, cl).size();

    matrix_type         gr_lhs = matrix_type::Zero(gbs, gbs);
    matrix_type         gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);

    if(celdeg > 0)
    {
        const auto qps = integrate(msh, cl, celdeg - 1 + facdeg);
        for (auto& qp : qps)
        {
            const auto c_dphi = cb.eval_gradients(qp.first);
            const auto g_phi  = gb.eval_basis(qp.first);

            gr_lhs.block(0, 0, gbs, gbs) += qp.second * inner_product(g_phi, g_phi);
            // we use here the symmetry of the basis gb
            gr_rhs.block(0, 0, gbs, cbs) += qp.second * inner_product(g_phi, c_dphi);
        }
    }

    const auto fcs = faces(msh, cl);
    const auto ns = normals(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = ns[i];
        vector_face_basis<Mesh,T> fb(msh, fc, facdeg);

        const auto qps_f = integrate(msh, fc, facdeg + std::max(facdeg, celdeg));
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

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = 2.0 * gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}
