
#ifndef methods_hpp
#define methods_hpp

#include "cut_methods.hpp"
#include "uncut_methods.hpp"

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>>
assembly_poly_extension(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg);

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>>
assembly_poly_extension(const Mesh& msh, hho_degree_info & hdi, meth &method, testType & test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg){
    
    using RealType = typename Mesh::coordinate_type;
    using VecTuple = std::vector<std::tuple<double,element_location,std::vector<double>>>;
    
    auto level_set_function = test_case.level_set_;
    auto rhs_fun = test_case.rhs_fun;
    auto sol_fun = test_case.sol_fun;   
    auto sol_grad = test_case.sol_grad;
    auto bcs_fun = test_case.bcs_fun;
    auto dirichlet_jump = test_case.dirichlet_jump;
    auto neumann_jump = test_case.neumann_jump;
    struct params<RealType> parms = test_case.parms;
    
    timecounter tc;
    
    tc.tic();
    auto assembler = make_one_field_interface_assembler(msh, bcs_fun, hdi);
    std::vector<std::pair<size_t,size_t>> cell_basis_data = assembler.compute_cell_basis_data(msh);
    std::pair<VecTuple,VecTuple> Pairs = make_pair_KO_pair_OK(msh);

    for (auto& pair : Pairs.first) { // Loop on POK subcells 
        auto cl = msh.cells[std::get<0>(pair)];
        auto contrib = method.make_contrib_POK(msh, pair, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        auto cell_mass = method.make_contrib_mass(msh, pair, test_case, hdi);      
        size_t n_dof = assembler.n_dof(msh, cl);
        Matrix<RealType, Dynamic, Dynamic> mass = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof, n_dof);
        mass.block(0,0,cell_mass.rows(), cell_mass.cols()) = cell_mass;
        assembler.assemble_extended(msh, pair, lc, f);  
        assembler.assemble_mass(msh, cl, mass);
    } 
    for (auto& pair : Pairs.second) { // Loop on PKO subcells 
        auto cl = msh.cells[std::get<0>(pair)];
        auto contrib = method.make_contrib_PKO(msh, pair, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        auto cell_mass = method.make_contrib_mass(msh, pair, test_case, hdi);      
        size_t n_dof = assembler.n_dof(msh, cl);
        Matrix<RealType, Dynamic, Dynamic> mass = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof, n_dof);
        mass.block(0,0,cell_mass.rows(), cell_mass.cols()) = cell_mass;
        assembler.assemble_extended(msh, pair, lc, f);  
        assembler.assemble_mass(msh, cl, mass);
    } 
    assembler.finalize();
    
    tc.toc();
    std::cout << bold << yellow << "         Matrix assembly: " << tc << " seconds" << reset << std::endl;

    Kg = assembler.LHS;
    Mg = assembler.MASS;

    return cell_basis_data;

}

template<typename T, size_t ET, typename testType>
class call_methods : public uncut_method<T, ET, testType>, cut_method<T, ET, testType> {

    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;
    using Tuple = std::tuple<double,element_location,std::vector<double>>;

public:

    T eta;

    call_methods(T eta_) : uncut_method<T,ET,testType>(), cut_method<T,ET,testType>(eta_), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib_POK(const Mesh& msh, Tuple P, const testType &test_case, const hho_degree_info hdi) {
            return cut_method<T, ET, testType>::make_contrib_POK(msh, P, test_case, hdi);
    }
    
    std::pair<Mat, Vect>
    make_contrib_PKO(const Mesh& msh, Tuple P, const testType &test_case, const hho_degree_info hdi) {
            return cut_method<T, ET, testType>::make_contrib_cut_PKO(msh, P, test_case, hdi);
    }

    Vect
    make_contrib_rhs(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi) {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return uncut_method<T, ET, testType>::make_contrib_rhs_uncut(msh, cl, hdi, test_case);
        else 
            return cut_method<T, ET, testType>::make_contrib_rhs_cut(msh, cl, test_case, hdi);
    }

    Mat
    make_contrib_mass(const Mesh& msh, Tuple P, const testType &test_case, const hho_degree_info hdi) {
        return cut_method<T, ET, testType>::make_contrib_cut_mass(msh, P, hdi, test_case);
    }

};

template<typename T, size_t ET, typename testType>
auto make_call_methods(const cuthho_mesh<T, ET>& msh, const T eta_, testType test_case) {
    
    return call_methods<T, ET, testType>(eta_);

}

template<typename Mesh>
std::pair<std::vector< std::tuple<double,element_location,std::vector<double>>>,
          std::vector< std::tuple<double,element_location,std::vector<double>>>> 
make_pair_KO_pair_OK(Mesh& msh) {

    std::vector< std::tuple<double,element_location,std::vector<double>>> PairOK;
    std::vector< std::tuple<double,element_location,std::vector<double>>> PairKO;

    for (auto &cl : msh.cells) {
        auto offset_cl = offset(msh,cl);
        if (cl.user_data.location != element_location::ON_INTERFACE) { 
            std::vector<double> dp_cells;
            element_location loc;
            if (cl.user_data.location == element_location::IN_NEGATIVE_SIDE) {
                loc = element_location::IN_NEGATIVE_SIDE;
                for (auto& dp_cl: cl.user_data.dependent_cells_neg) 
                    dp_cells.push_back(dp_cl);
            }
            else {
                loc = element_location::IN_POSITIVE_SIDE;
                for (auto& dp_cl: cl.user_data.dependent_cells_pos) 
                    dp_cells.push_back(dp_cl);
            }
            auto tuple = std::make_tuple(offset_cl, loc, dp_cells);
            PairOK.push_back(tuple);
        }
        else if (cl.user_data.agglo_set == cell_agglo_set::T_OK) { 
            std::vector<double> dp_cells_neg;
            std::vector<double> dp_cells_pos;
            for (auto& dp_cl: cl.user_data.dependent_cells_neg) 
                dp_cells_neg.push_back(dp_cl);
            for (auto& dp_cl: cl.user_data.dependent_cells_pos) 
                dp_cells_pos.push_back(dp_cl);
            auto tuple_neg = std::make_tuple(offset_cl, element_location::IN_NEGATIVE_SIDE, dp_cells_neg);
            auto tuple_pos = std::make_tuple(offset_cl, element_location::IN_POSITIVE_SIDE, dp_cells_pos);
            PairOK.push_back(tuple_neg);
            PairOK.push_back(tuple_pos);
        }
        else if (cl.user_data.agglo_set == cell_agglo_set::T_KO_NEG) { 
            std::vector<double> dp_cells_neg;
            std::vector<double> dp_cells_pos;
            for (auto& dp_cl: cl.user_data.dependent_cells_pos) 
                dp_cells_pos.push_back(dp_cl);
            auto tuple_neg = std::make_tuple(offset_cl, element_location::IN_NEGATIVE_SIDE, dp_cells_neg);
            auto tuple_pos = std::make_tuple(offset_cl, element_location::IN_POSITIVE_SIDE, dp_cells_pos);
            PairKO.push_back(tuple_neg);
            PairOK.push_back(tuple_pos);
        }
        else if (cl.user_data.agglo_set == cell_agglo_set::T_KO_POS) {
            std::vector<double> dp_cells_neg;
            std::vector<double> dp_cells_pos;
            for (auto& dp_cl: cl.user_data.dependent_cells_neg) 
                dp_cells_neg.push_back(dp_cl);
            auto tuple_neg = std::make_tuple(offset_cl, element_location::IN_NEGATIVE_SIDE, dp_cells_neg);
            auto tuple_pos = std::make_tuple(offset_cl, element_location::IN_POSITIVE_SIDE, dp_cells_pos);
            PairOK.push_back(tuple_neg);
            PairKO.push_back(tuple_pos);
        }
    }   

    // Debug 
    // std::cout << std::endl << "Paires OK:   " << std::endl;
    // for (auto& pair : PairOK) {
    //     if (std::get<1>(pair) == element_location::IN_NEGATIVE_SIDE) {
    //         std::cout << "(" << std::get<0>(pair) << ", " << "NEGATIVE SIDE";
    //         for (auto& dp_cl : std::get<2>(pair)) 
    //             std::cout << ", " << dp_cl;
    //         std::cout << ")" << std::endl;
    //     }
    //     if (std::get<1>(pair) == element_location::IN_POSITIVE_SIDE) {
    //         std::cout << "(" << std::get<0>(pair) << ", " << "POSITIVE SIDE";
    //         for (auto& dp_cl : std::get<2>(pair)) 
    //             std::cout << ", " << dp_cl;
    //         std::cout << ")" << std::endl;
    //     }   
    // }
    // std::cout << std::endl << "Paires KO:   " << std::endl;
    // for (auto& pair : PairKO) {
    //     if (std::get<1>(pair) == element_location::IN_NEGATIVE_SIDE) {
    //         std::cout << "(" << std::get<0>(pair) << ", " << "NEGATIVE SIDE";
    //         for (auto& dp_cl : std::get<2>(pair)) 
    //             std::cout << ", " << dp_cl;
    //         std::cout << ")" << std::endl;
    //     }
    //     if (std::get<1>(pair) == element_location::IN_POSITIVE_SIDE) {
    //         std::cout << "(" << std::get<0>(pair) << ", " << "POSITIVE SIDE";
    //         for (auto& dp_cl : std::get<2>(pair)) 
    //             std::cout << ", " << dp_cl;
    //         std::cout << ")" << std::endl;
    //     }   
    // }
    
    auto Pair_OK_KO = std::make_pair(PairOK,PairKO);
    return Pair_OK_KO;

}



#endif
