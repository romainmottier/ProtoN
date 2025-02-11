
#ifndef methods_hpp
#define methods_hpp

// #include "cut_methods.hpp"
// #include "uncut_methods.hpp"

template<typename T, size_t ET, typename testType>
class call_methods : public uncut_method<T, ET, testType>, cut_method<T, ET, testType> {

    using Mat = Matrix<T, Dynamic, Dynamic>;
    using Vect = Matrix<T, Dynamic, 1>;
    using Mesh = cuthho_mesh<T, ET>;

public:

    T eta;

    call_methods(T eta_) : uncut_method<T,ET,testType>(), cut_method<T,ET,testType>(eta_), eta(eta_) {}

    std::pair<Mat, Vect>
    make_contrib(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi) {
        
        if( location(msh, cl) != element_location::ON_INTERFACE )
            return uncut_method<T, ET, testType>::make_contrib_uncut(msh, cl, hdi, test_case);
        else 
            return cut_method<T, ET, testType>::make_contrib_cut(msh, cl, test_case, hdi);
    
    }
    
    Vect
    make_contrib_rhs(const Mesh& msh, const typename Mesh::cell_type& cl, const testType &test_case, const hho_degree_info hdi) {
        
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
