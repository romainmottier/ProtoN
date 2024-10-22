#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cmath>
#include <memory>
#include <sstream>
#include <list>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Eigen/Eigenvalues>
using namespace Eigen;

#include "core/core"
#include "core/solvers"
#include "dataio/silo_io.hpp"
#include "methods/hho"
#include "methods/cuthho"


#include "../common/preprocessor.hpp"
#include "../common/postprocessor.hpp"
#include "../common/newmark_hho_scheme.hpp"
#include "../common/dirk_hho_scheme.hpp"
#include "../common/dirk_butcher_tableau.hpp"
#include "../common/erk_hho_scheme.hpp"
#include "../common/erk_butcher_tableau.hpp"
#include "../common/analytical_functions.hpp"

#define scaled_stab_Q 0

// ----- common data types ------------------------------
using RealType = double;
typedef cuthho_poly_mesh<RealType>  mesh_type;

// Classes interface methods 
#include "Classes/interface_method.hpp"
#include "Classes/cut_interface_method.hpp"
#include "Classes/mixed_interface_method.hpp"
#include "Classes/gradrec_mixed_interface_method.hpp"

// Classes Test cases Laplacian
#include "Classes/test_case_laplacian_conv.hpp"
#include "Classes/test_case_laplacian_waves.hpp"
#include "Classes/test_case_laplacian_waves_mixed.hpp"
#include "Classes/test_case_laplacian_waves_scatter.hpp"

// Temp Schemes 
#include "Temp_Schemes/newmark_step_cuthho_interface.hpp"
#include "Temp_Schemes/newmark_step_cuthho_interface_scatter.hpp"
#include "Temp_Schemes/erk_step_cuthho_interface.hpp"
#include "Temp_Schemes/erk_step_cuthho_interface_cfl.hpp"
#include "Temp_Schemes/erk_step_cuthho_interface_scatter.hpp"
#include "Temp_Schemes/sdirk_step_cuthho_interface.hpp"
#include "Temp_Schemes/sdirk_step_cuthho_interface_scatter.hpp"

// Pre Process
#include "Pre_Process/CutMesh.hpp"
#include "Pre_Process/SquareCutMesh.hpp"
#include "Pre_Process/SquareGar6moreCutMesh.hpp"

// Outputs
#include "Outputs/PrintIntegrationRule.hpp"
#include "Outputs/PrintAgglomeratedCells.hpp"

// Matrix Assemblers
#include "Matrix_Assemblers/create_kg_and_mg_cuthho_interface.hpp"
#include "Matrix_Assemblers/create_mixed_kg_and_mg_cuthho_interface.hpp"

// Prototype sources 
#include "Prototypes/CutHHOSecondOrderConvTest.hpp"
#include "Prototypes/CutHHOFirstOrderConvTest.hpp"
// #include "Prototypes/ICutHHOSecondOrder.hpp"
// #include "Prototypes/ICutHHOFirstOrder.hpp"
// #include "Prototypes/ECutHHOFirstOrder.hpp"
// #include "Prototypes/ECutHHOFirstOrderCFL.hpp"
// #include "Prototypes/ECutHHOFirstOrderEigenCFL.hpp"
// #include "Prototypes/HeterogeneousGar6moreICutHHOSecondOrder.hpp"
// #include "Prototypes/HeterogeneousGar6moreICutHHOFirstOrder.hpp"
// #include "Prototypes/HeterogeneousFlowerICutHHOSecondOrder.hpp"
// #include "Prototypes/HeterogeneousFlowerICutHHOFirstOrder.hpp"
// #include "Prototypes/HeterogeneousFlowerECutHHOFirstOrder.hpp"

// source /opt/intel/oneapi/setvars.sh intel64
// ./unfitted....

int main(int argc, char **argv)
{
    // Steady diffusion problem
    CutHHOSecondOrderConvTest(argc, argv); 			// Fonctionne avec source... 
    // CutHHOFirstOrderConvTest(argc, argv);  			// Fonctionne avec source...

    // Newmark - schemes / homogeneous cases 
    // ICutHHOSecondOrder(argc, argv);				// Fonctionne avec source...
    
    // SDIRK - schemes / homogeneous cases
    // ICutHHOFirstOrder(argc, argv);				// Fonctionne avec source...
    // ECutHHOFirstOrder(argc, argv);				// Fonctionne 
    // ECutHHOFirstOrderCFL(argc, argv);			// Fonctionne avec source
    // ECutHHOFirstOrderEigenCFL(argc, argv);			// Fonctionne pas
    
    // Newmark - schemes / heterogenous cases 
    // HeterogeneousFlowerICutHHOSecondOrder(argc, argv);	// Fonctionne  
    
    // SDIRK - Schemes / heterogeneous cases 
    // HeterogeneousFlowerICutHHOFirstOrder(argc, argv);	// Fonctionne
    //  HeterogeneousFlowerECutHHOFirstOrder(argc, argv);	// Fonctionne 
    
    // Gar6more / heterogeneous cases 
    // HeterogeneousGar6moreICutHHOSecondOrder(argc, argv);	// Fonctionne pas
    // HeterogeneousGar6moreICutHHOFirstOrder(argc, argv);	// Fonctionne pas

    return 0;
    
}


