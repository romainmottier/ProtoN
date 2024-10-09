
#ifndef interface_method_hpp
#define interface_method_hpp

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
                     const testType &test_case, const hho_degree_info hdi) = 0;

    virtual Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi) = 0;

public:

    std::pair<Mat, Vect>
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType &test_case, const hho_degree_info hdi) {

        if (location(msh, cl) != element_location::ON_INTERFACE) {
            return make_contrib_uncut(msh, cl, hdi, test_case);
        }

        else { // on interface
            return make_contrib_cut(msh, cl, test_case, hdi);
        }
    }

    std::pair<Mat, Vect>
    make_contrib_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case) {

        // PARAMETERS
        T kappa;
        if (location(msh, cl) == element_location::IN_NEGATIVE_SIDE)
            kappa = test_case.parms.kappa_1;
        else
            kappa = test_case.parms.kappa_2;
        auto stab_parms = test_case.parms;
        auto level_set_function = test_case.level_set_;

        // OPERATORS
        auto gr  = make_hho_gradrec_vector_extended(msh, cl, hdi, level_set_function);
        Mat stab = make_hho_naive_stabilization_extended(msh, cl, hdi, stab_parms);
        Mat lc   = kappa * (gr.second + stab);    
        Mat f    = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);

        //  std::cout << "r = " << gr.second << std::endl;
        //  std::cout << "s = " << stab << std::endl;
        //  std::cout << "f = " << f << std::endl;
        
        return std::make_pair(lc, f);
    }
    
    Vect
    make_contrib_rhs_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case)
    {
        Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        return f;
    }
    
    Vect
    make_contrib_rhs(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType &test_case, const hho_degree_info hdi) {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_rhs_uncut(msh, cl, hdi, test_case);
        else // on interface
            return make_contrib_rhs_cut(msh, cl, test_case, hdi);
    }
    
    Mat
    make_contrib_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType &test_case, const hho_degree_info hdi) {

        if (location(msh, cl) != element_location::ON_INTERFACE)
            return make_contrib_uncut_mass(msh, cl, hdi, test_case);

        else // on interface
            return make_contrib_cut_mass(msh, cl, hdi, test_case);

    }

    Mat
    make_contrib_uncut_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case) {
        T c;
        if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
            c = test_case.parms.c_1;
        else
            c = test_case.parms.c_2;
        Mat mass = make_mass_matrix(msh, cl, hdi.cell_degree());
        mass *= (1.0/(c*c*test_case.parms.kappa_1));
        return mass;
    }
    
    Mat
    make_contrib_cut_mass(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case) {

        Mat mass_neg = make_mass_matrix(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
        Mat mass_pos = make_mass_matrix(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
        mass_neg *= (1.0/(test_case.parms.c_1*test_case.parms.c_1*test_case.parms.kappa_1));
        mass_pos *= (1.0/(test_case.parms.c_2*test_case.parms.c_2*test_case.parms.kappa_2));
        
        size_t n_data_neg = mass_neg.rows();
        size_t n_data_pos = mass_pos.rows();
        size_t n_data = n_data_neg + n_data_pos;
        
        Mat mass = Mat::Zero(n_data,n_data);
        mass.block(0,0,n_data_neg,n_data_neg) = mass_neg;
        mass.block(n_data_neg,n_data_neg,n_data_pos,n_data_pos) = mass_pos;

        return mass;
    }
    
};

#endif
