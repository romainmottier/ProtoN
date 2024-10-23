
#ifndef methods_hpp
#define methods_hpp

#include "cut_methods.hpp"
#include "uncut_methods.hpp"

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>>
create_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth &method, testType &test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg);

template<typename Mesh, typename testType, typename meth>
std::vector<std::pair<size_t,size_t>>
create_kg_and_mg_cuthho_interface(const Mesh& msh, hho_degree_info & hdi, meth &method, testType & test_case, SparseMatrix<typename Mesh::coordinate_type> & Kg, SparseMatrix<typename Mesh::coordinate_type> & Mg){
    
    using RealType = typename Mesh::coordinate_type;

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
    size_t cell_ind = 0;
    for (auto& cell : msh.cells) {
        auto contrib = method.make_contrib(msh, cell, test_case, hdi);
        auto lc = contrib.first;
        auto f = contrib.second;
        auto cell_mass = method.make_contrib_mass(msh, cell, test_case, hdi);
        size_t n_dof = assembler.n_dof(msh,cell);
        Matrix<RealType, Dynamic, Dynamic> mass = Matrix<RealType, Dynamic, Dynamic>::Zero(n_dof,n_dof);
        mass.block(0,0,cell_mass.rows(),cell_mass.cols()) = cell_mass;
        assembler.assemble(msh, cell, lc, f);
        assembler.assemble_mass(msh, cell, mass);
        cell_ind++;
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

public:

    T eta;

    call_methods(T eta_) : uncut_method<T,ET,testType>(), cut_method<T,ET,testType>(eta_), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return uncut_method<T, ET, testType>::make_contrib_uncut(msh, cl, hdi, test_case);
        else 
            return cut_method<T, ET, testType>::make_contrib_cut(msh, cl, test_case, hdi);
    }

    Vect
    make_contrib_rhs(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi)
    {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return uncut_method<T, ET, testType>::make_contrib_rhs_uncut(msh, cl, hdi, test_case);
        else 
            return cut_method<T, ET, testType>::make_contrib_rhs_cut(msh, cl, test_case, hdi);
    }

    Mat
    make_contrib_mass(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi) {
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return uncut_method<T, ET, testType>::make_contrib_uncut_mass(msh, cl, hdi, test_case);
        else 
            return cut_method<T, ET, testType>::make_contrib_cut_mass(msh, cl, hdi, test_case);
    }

};

template<typename T, size_t ET, typename testType>
auto make_call_methods(const cuthho_mesh<T, ET>& msh, const T eta_, testType test_case) {
    
    return call_methods<T, ET, testType>(eta_);

}


#endif
