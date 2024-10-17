
#ifndef interface_method_hpp
#define interface_method_hpp

template<typename T, size_t ET, typename testType>
class interface_method
{
    using Mat  = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;
    using Tuple = std::tuple<double,element_location,std::vector<double>>;

protected:
    interface_method(){}

public:

    std::pair<Mat, Vect>
    make_contrib_POK(const Mesh& msh, Tuple P_OK, const hho_degree_info hdi, const testType &test_case) {

        // CELL INFOS & PARAMETERS
        T kappa;
        auto stab_parms = test_case.parms;
        if (std::get<1>(P_OK) == element_location::IN_NEGATIVE_SIDE)
            kappa = stab_parms.kappa_1;
        else 
            kappa = stab_parms.kappa_2; 
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        // OPERATORS
        auto gr = make_hho_gradrec_vector_POK(msh, P_OK, hdi, level_set_function); // renvoie les contribution des coté i et ibar 
        auto stab_o = make_hho_stabilization(msh, P_OK, hdi, stab_parms);          // renvoie les contribtions des cotés i et ibar 
        auto stab_ill_dofs = make_hho_ill_dofs_stabilization(msh, P_OK, hdi, stab_parms);
        auto stab = stab_o + stab_ill_dofs;

        Mat lc = kappa * (gr.second + stab); 
        Mat f  = Mat::Zero(3,1);
        // Mat f  = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);

        return std::make_pair(lc, f);
    }
    
    std::pair<Mat, Vect>
    make_contrib_PKO(const Mesh& msh, Tuple P_KO, const hho_degree_info hdi, const testType &test_case) {

        // CELL INFOS & PARAMETERS
        T kappa;
        auto stab_parms = test_case.parms;
        if (std::get<1>(P_KO) == element_location::IN_NEGATIVE_SIDE)
            kappa = stab_parms.kappa_1;
        else 
            kappa = stab_parms.kappa_2; 
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        // OPERATORS
        auto gr = make_hho_gradrec_vector_PKO(msh, P_KO, hdi, level_set_function);
        auto stab = make_hho_stabilization(msh, P_KO, hdi, stab_parms);

        Mat lc = kappa * (gr.second + stab);    
        Mat f  = Mat::Zero(3,1);
        // Mat f  = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);

        return std::make_pair(lc, f);
    }

    Mat
    make_contrib_mass(const Mesh& msh, Tuple P, const testType &test_case, const hho_degree_info hdi) {

        // CELL INFOS
        auto cell_index = std::get<0>(P);
        auto cl = msh.cells[cell_index];
        auto loc = std::get<1>(P);      
        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg); 
        
        auto local_dofs = cbs;
        if (is_cut(msh,cl))
            local_dofs = 2*cbs;

        Mat cell_mass = Mat::Zero(local_dofs,local_dofs);
        Mat mass = make_mass_matrix(msh, cl, hdi.cell_degree(), loc);
        if (loc == element_location::IN_NEGATIVE_SIDE) {
            mass *= (1.0/(test_case.parms.c_1*test_case.parms.c_1*test_case.parms.kappa_1));
            cell_mass.block(0,0,cbs,cbs) = mass;
        }
        else {
            mass *= (1.0/(test_case.parms.c_2*test_case.parms.c_2*test_case.parms.kappa_2));
            if (is_cut(msh,cl)) {
                cell_mass.block(cbs,cbs,cbs,cbs) = mass;
            }
            else {
                cell_mass.block(0,0,cbs,cbs) = mass;
            }
        }

        return cell_mass;

    }

    // Mat
    // make_contrib_mass(const Mesh& msh, Tuple P, const testType &test_case, const hho_degree_info hdi) {

    //     if (location(msh, cl) != element_location::ON_INTERFACE)
    //         return make_contrib_uncut_mass(msh, P, hdi, test_case);

    //     else // on interface
    //         return make_contrib_cut_mass(msh, P, hdi, test_case);

    // }

    // Mat
    // make_contrib_uncut_mass(const Mesh& msh, Tuple P, const hho_degree_info hdi, const testType &test_case) {
        
    //     T c;
    //     if ( location(msh, cl) == element_location::IN_NEGATIVE_SIDE )
    //         c = test_case.parms.c_1;
    //     else
    //         c = test_case.parms.c_2;
    //     Mat mass = make_mass_matrix(msh, cl, hdi.cell_degree());
    //     mass *= (1.0/(c*c*test_case.parms.kappa_1));
    //     return mass;
    // }
    
    // Mat
    // make_contrib_cut_mass(const Mesh& msh, Tuple P, const hho_degree_info hdi, const testType &test_case) {

    //     Mat mass_neg = make_mass_matrix(msh, cl, hdi.cell_degree(), element_location::IN_NEGATIVE_SIDE);
    //     Mat mass_pos = make_mass_matrix(msh, cl, hdi.cell_degree(), element_location::IN_POSITIVE_SIDE);
    //     mass_neg *= (1.0/(test_case.parms.c_1*test_case.parms.c_1*test_case.parms.kappa_1));
    //     mass_pos *= (1.0/(test_case.parms.c_2*test_case.parms.c_2*test_case.parms.kappa_2));
        
    //     size_t n_data_neg = mass_neg.rows();
    //     size_t n_data_pos = mass_pos.rows();
    //     size_t n_data = n_data_neg + n_data_pos;
        
    //     Mat mass = Mat::Zero(n_data,n_data);
    //     mass.block(0,0,n_data_neg,n_data_neg) = mass_neg;
    //     mass.block(n_data_neg,n_data_neg,n_data_pos,n_data_pos) = mass_pos;

    //     return mass;
    // }
    
    // Vect
    // make_contrib_rhs(const Mesh& msh, const typename Mesh::cell_type& cl,
    //              const testType &test_case, const hho_degree_info hdi) {
    //     if( location(msh, cl) != element_location::ON_INTERFACE )
    //         return make_contrib_rhs_uncut(msh, cl, hdi, test_case);
    //     else // on interface
    //         return make_contrib_rhs_cut(msh, cl, test_case, hdi);
    // }
    
    // Vect
    // make_contrib_rhs_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
    //                    const hho_degree_info hdi, const testType &test_case)
    // {
    //     Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
    //     return f;
    // }
    
    // std::pair<Mat, Vect>
    // make_contrib_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
    //                    const hho_degree_info hdi, const testType &test_case) {

    //     // PARAMETERS
    //     T kappa;
    //     if (location(msh, cl) == element_location::IN_NEGATIVE_SIDE){
    //         kappa = test_case.parms.kappa_1;
    //     }
    //     else {
    //         kappa = test_case.parms.kappa_2;
    //     }
    //     auto stab_parms = test_case.parms;
    //     auto level_set_function = test_case.level_set_;

    //     // OPERATORS
    //     auto gr  = make_hho_gradrec_vector_extended(msh, cl, hdi, level_set_function);
    //     Mat stab = make_hho_naive_stabilization_extended(msh, cl, hdi, stab_parms);
    //     Mat lc   = kappa * (gr.second + stab);    
    //     Mat f    = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);

    //      std::cout << "r = " << gr.second << std::endl;
    //      std::cout << "s = " << stab << std::endl;
    //      std::cout << "f = " << f << std::endl;
        
    //     return std::make_pair(lc, f);
    // }

};

#endif
