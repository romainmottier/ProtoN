
#ifndef mixed_interface_method_hpp
#define mixed_interface_method_hpp

template<typename T, size_t ET, typename testType>
class mixed_interface_method
{
    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

protected:
    mixed_interface_method(){}

    virtual std::pair<Mat, Vect>
    make_contrib_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi) {
    }
    
    virtual Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi) {
    }

public:
    std::pair<Mat, Vect>
    make_contrib_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case) {
        T rho, vp;
        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE ) {
          rho = test_case.parms.kappa_1;
          vp = test_case.parms.c_1;
        }
        else {
          rho = test_case.parms.kappa_2;
          vp = test_case.parms.c_2;
        }

        auto gr = make_hho_gradrec_mixed_vector(msh, cl, hdi);
        
        Matrix<T, Dynamic, Dynamic> R_operator = gr.second;
        auto n_rows = R_operator.rows();
        auto n_cols = R_operator.cols();
        
        Matrix<T, Dynamic, Dynamic> S_operator = Matrix<T, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        auto stabilization_operator    = make_hho_naive_stabilization(msh, cl, hdi, scaled_stab_Q);
        auto n_s_rows = stabilization_operator.rows();
        auto n_s_cols = stabilization_operator.cols();
        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;

        Mat lc = R_operator + ((1.0)/(vp*rho))*S_operator;
        Mat f = make_mixed_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
            
        return std::make_pair(lc, f);
    }
    
    Vect
    make_contrib_rhs_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case)
    {
        Mat f = make_mixed_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        return f;
    }

    std::pair<Mat, Vect>
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType &test_case, const hho_degree_info hdi)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_uncut(msh, cl, hdi, test_case);
        else // on interface
            return make_contrib_cut(msh, cl, test_case, hdi);
    }
    
    Vect
    make_contrib_rhs(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType &test_case, const hho_degree_info hdi)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_rhs_uncut(msh, cl, hdi, test_case);
        else // on interface
            return make_contrib_rhs_cut(msh, cl, test_case, hdi);
    }
    
    Mat
    make_contrib_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType &test_case, const hho_degree_info hdi, bool add_scalar_mass_Q = true)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_uncut_mass(msh, cl, hdi, test_case, add_scalar_mass_Q);
        else // on interface
            return make_contrib_cut_mass(msh, cl, hdi, test_case, add_scalar_mass_Q);
    }
    Mat uncut_vec_mass_matrix(const Mesh& msh, const typename Mesh::cell_type& cl,
    const hho_degree_info hdi)
    {
         typedef Matrix<T, Dynamic, Dynamic> matrix_type;
         typedef Matrix<T, Dynamic, 1>       vector_type;

         const auto celdeg  = hdi.cell_degree();
         const auto facdeg  = hdi.face_degree();
         const auto graddeg = hdi.grad_degree();

         cell_basis<Mesh,T>            cb(msh, cl, celdeg);
         vector_cell_basis<Mesh,T>     gb(msh, cl, graddeg);

         auto cbs = cell_basis<Mesh,T>::size(celdeg);
         auto fbs = face_basis<Mesh,T>::size(facdeg);
         auto gbs = vector_cell_basis<Mesh,T>::size(graddeg);

         matrix_type         mass = matrix_type::Zero(gbs, gbs);
         const auto qps = integrate(msh, cl, celdeg - 1 + facdeg);
         for (auto& qp : qps)
         {
             const auto g_phi  = gb.eval_basis(qp.first);

             mass += qp.second * g_phi * g_phi.transpose();
         }
        return mass;
    }
    
    Mat
    make_contrib_uncut_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case, bool add_scalar_mass_Q = true)
    {
        
        T rho, vp;
        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
        {
            rho = test_case.parms.kappa_1;
            vp = test_case.parms.c_1;
        }
        else{
            rho = test_case.parms.kappa_2;
            vp = test_case.parms.c_2;
        }
        
        const auto celdeg  = hdi.cell_degree();
        const auto graddeg = hdi.grad_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);
        auto gbs = vector_cell_basis<Mesh,T>::size(graddeg);
        
        Mat mass_sigma = uncut_vec_mass_matrix(msh, cl, hdi);
        Mat mass_v = make_mass_matrix(msh, cl, celdeg);
        
        size_t n_data = mass_sigma.rows() + mass_v.rows();
        Mat mass = Mat::Zero(cbs+gbs,cbs+gbs);
        mass.block(0,0,gbs,gbs) = (1.0/rho)*mass_sigma;
        if (add_scalar_mass_Q) {
            mass.block(gbs,gbs,cbs,cbs) = (1.0/(rho*vp*vp))*mass_v;
        }
        return mass;
    }
    
    Mat
    make_contrib_cut_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case, bool add_scalar_mass_Q = true)
    {

        Mat mass_sigma_neg = make_vec_mass_matrix(msh, cl,
                                        hdi, element_location::IN_NEGATIVE_SIDE);
        Mat mass_sigma_pos = make_vec_mass_matrix(msh, cl,
                                        hdi, element_location::IN_POSITIVE_SIDE);
        
        Mat mass_v_neg = make_mass_matrix(msh, cl,
                                        hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
        Mat mass_v_pos = make_mass_matrix(msh, cl,
                                        hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
        mass_v_neg *= (1.0/(test_case.parms.kappa_1));
        mass_v_pos *= (1.0/(test_case.parms.kappa_2));
        
        size_t n_s_data_neg = mass_sigma_neg.rows();
        size_t n_s_data_pos = mass_sigma_pos.rows();
        size_t n_s_data = n_s_data_neg + n_s_data_pos;
        
        size_t n_data_neg = mass_v_neg.rows();
        size_t n_data_pos = mass_v_pos.rows();
        size_t n_v_data = n_data_neg + n_data_pos;
        
        size_t n_data = n_s_data + n_v_data;
        Mat mass = Mat::Zero(n_data,n_data);
        
        mass.block(0,0,n_s_data_neg,n_s_data_neg) = mass_sigma_neg;
        mass.block(n_s_data_neg,n_s_data_neg,n_s_data_pos,n_s_data_pos) = mass_sigma_pos;
        if (add_scalar_mass_Q) {
            T rho_1, vp_1, rho_2, vp_2;
            rho_1 = test_case.parms.kappa_1;
            rho_2 = test_case.parms.kappa_2;
            vp_1 = test_case.parms.c_1;
            vp_2 = test_case.parms.c_2;
            
            mass.block(n_s_data,n_s_data,n_data_neg,n_data_neg) = (1.0/(rho_1*vp_1*vp_1))*mass_v_neg;
            mass.block(n_s_data+n_data_neg,n_s_data+n_data_neg,n_data_pos,n_data_pos) = (1.0/(rho_2*vp_2*vp_2))*mass_v_pos;
        }
        return mass;
    }
    
};

#endif
