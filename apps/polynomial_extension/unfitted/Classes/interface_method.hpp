
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

    // std::pair<Mat, Vect>
    // make_contrib_POK(const Mesh& msh, Tuple P_OK, const hho_degree_info hdi, const testType &test_case) {

    //     // CELL INFOS & PARAMETERS
    //     T kappa;
    //     auto stab_parms = test_case.parms;
    //     if (std::get<1>(P_OK) == element_location::IN_NEGATIVE_SIDE)
    //         kappa = stab_parms.kappa_1;
    //     else 
    //         kappa = stab_parms.kappa_2; 
    //     auto level_set_function = test_case.level_set_;
    //     auto dir_jump = test_case.dirichlet_jump;

    //     // SUB-CELL INFOS
    //     auto cell_index = std::get<0>(P_OK);
    //     auto cl = msh.cells[cell_index];
    //     auto celdeg = hdi.cell_degree();
    //     auto cbs = cell_basis<Mesh,T>::size(celdeg);

    //     // OPERATORS
    //     auto gr = make_hho_gradrec_vector_POK(msh, P_OK, hdi, level_set_function); 
    //     auto stab_usual = make_hho_stabilization(msh, P_OK, hdi, stab_parms);          
    //     auto stab_ill_dofs = make_hho_ill_dofs_stabilization(msh, P_OK, hdi, stab_parms);
    //     auto stab = stab_usual + stab_ill_dofs;

    //     Mat lc = kappa * (gr.second + stab); 

    //     // AJOUTER SECOND MEMBRE CORRECTEMENT 
    //     Mat f  = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);

    //     return std::make_pair(lc, f);
    // }
    
    // std::pair<Mat, Vect>
    // make_contrib_PKO(const Mesh& msh, Tuple P_KO, const hho_degree_info hdi, const testType &test_case) {

    //     // CELL INFOS & PARAMETERS
    //     auto cell_index = std::get<0>(P_KO);
    //     auto cl = msh.cells[cell_index];
    //     auto celdeg = hdi.cell_degree();
    //     auto cbs = cell_basis<Mesh,T>::size(celdeg);
    //     T kappa;
    //     auto stab_parms = test_case.parms;
    //     if (std::get<1>(P_KO) == element_location::IN_NEGATIVE_SIDE)
    //         kappa = stab_parms.kappa_1;
    //     else 
    //         kappa = stab_parms.kappa_2; 
    //     auto level_set_function = test_case.level_set_;
    //     auto dir_jump = test_case.dirichlet_jump;

    //     // OPERATORS
    //     auto gr = make_hho_gradrec_vector_PKO(msh, P_KO, hdi, level_set_function);
    //     auto stab = make_hho_stabilization(msh, P_KO, hdi, stab_parms);

    //     Mat lc = kappa * (gr.second + stab);  
          
    //     // AJOUTER SECOND MEMBRE CORRECTEMENT 
    //     Mat f  = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);

    //     return std::make_pair(lc, f);
    // }

    Mat
    make_contrib_mass(const Mesh& msh, Tuple P, const testType &test_case, const hho_degree_info hdi) {

        // CELL INFOS
        auto cell_index = std::get<0>(P);
        auto cl = msh.cells[cell_index];
        auto loc = std::get<1>(P);      
        
        // DISCRETIZATION INFOS
        const auto celdeg  = hdi.cell_degree();
        const auto facdeg  = hdi.face_degree();
        const auto graddeg = hdi.grad_degree();
        cell_basis<cuthho_mesh<T, ET>,T>        cb(msh, cl, celdeg);
        auto cbs = cell_basis<cuthho_mesh<T, ET>,T>::size(celdeg);
        auto fbs = face_basis<cuthho_mesh<T, ET>,T>::size(facdeg);
        auto fcs = faces(msh, cl);
        auto num_faces = fcs.size();
        auto current_dofs = cbs + num_faces*fbs;
        if (is_cut(msh,cl)) 
            current_dofs = 2*current_dofs;
        auto extended_dofs = 2*(cbs + num_faces*fbs);
        auto nb_dp_cells = std::get<2>(P).size();
        auto local_dofs = current_dofs + nb_dp_cells*extended_dofs; 

        Mat cell_mass = Mat::Zero(local_dofs,local_dofs);
        Mat mass = make_mass_matrix(msh, cl, hdi.cell_degree(), loc);
        if (loc == element_location::IN_NEGATIVE_SIDE) {
            mass *= (1.0/(test_case.parms.c_1*test_case.parms.c_1*test_case.parms.kappa_1));
            cell_mass.block(0,0,cbs,cbs) = mass;
        }
        else {
            mass *= (1.0/(test_case.parms.c_2*test_case.parms.c_2*test_case.parms.kappa_2));
            if (is_cut(msh,cl))
                cell_mass.block(cbs,cbs,cbs,cbs) = mass;
            else
                cell_mass.block(0,0,cbs,cbs) = mass;
        }

        return cell_mass;

    }
    
    Vect
    make_contrib_rhs(const Mesh& msh, const typename Mesh::cell_type& cl,
                 const testType &test_case, const hho_degree_info hdi) {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return make_contrib_rhs_uncut(msh, cl, hdi, test_case);
        else // on interface
            return make_contrib_rhs_cut(msh, cl, test_case, hdi);
    }
    
    Vect
    make_contrib_rhs_uncut(const Mesh& msh, const typename Mesh::cell_type& cl,
                       const hho_degree_info hdi, const testType &test_case)
    {
        Mat f = make_rhs(msh, cl, hdi.cell_degree(), test_case.rhs_fun);
        return f;
    }

    Vect
    make_contrib_rhs_cut(const Mesh& msh, const typename Mesh::cell_type& cl,
                     const testType &test_case, const hho_degree_info hdi) {

        auto parms = test_case.parms;
        auto level_set_function = test_case.level_set_;
        auto dir_jump = test_case.dirichlet_jump;

        auto celdeg = hdi.cell_degree();
        auto cbs = cell_basis<Mesh,T>::size(celdeg);

        ///////////////    RHS
        Vect f = Vect::Zero(cbs*2);
        // neg part
        f.block(0, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_NEGATIVE_SIDE);
        // pos part
        f.block(cbs, 0, cbs, 1) += make_rhs(msh, cl, celdeg, test_case.rhs_fun, element_location::IN_POSITIVE_SIDE);

        return f;
    }


};

#endif
