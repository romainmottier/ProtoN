/*
 *       /\        Omar Duran 2019 
 *      /__\       omar.duran@enpc.fr
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
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

using namespace Eigen;

#include "core/core"
#include "core/solvers"
#include "dataio/silo_io.hpp"
#include "methods/hho"
#include "methods/cuthho"


// All these defines are going to be delete. 
// However for the case of unfitted implementation it is not required.
#define fancy_stabilization_Q
#define compute_energy_Q
#define spatial_errors_Q
//#define InhomogeneousQ
#define contrast 10.0
#define n_terms 100

// Class sources 
#include "Classes/TAnalyticalFunction.hpp"
#include "Classes/TDIRKSchemes.hpp"
#include "Classes/TSSPRKSchemes.hpp"
#include "Classes/TSSPRKHHOAnalyses.hpp"
#include "Classes/TDIRKHHOAnalyses.hpp"

// Post Process
#include "Post_Process/Compute_Energy/ComputeEnergyFirstOrder.hpp"
#include "Post_Process/Compute_Energy/ComputeEnergySecondOrder.hpp"
#include "Post_Process/Silo/Silo_Scalar_Field.hpp"
#include "Post_Process/Silo/Silo_Two_Field.hpp"
#include "Post_Process/Error/ErrorSingleField.hpp"
#include "Post_Process/Error/ErrorTwoFields.hpp"

// Matrix Assemblers
#include "Matrix_Assemblers/ComputeFG.hpp"
#include "Matrix_Assemblers/ComputeInhomogeneousKGFG.hpp"
#include "Matrix_Assemblers/ComputeKGFG.hpp"
#include "Matrix_Assemblers/ComputeKGFGSecondOrder.hpp"
#include "Matrix_Assemblers/FaceDoFUpdate.hpp"

// Temp Schemes
#include "Temp_Schemes/DIRKStep.hpp"
#include "Temp_Schemes/DIRKStepOpt.hpp"
#include "Temp_Schemes/ERKWeight.hpp"
#include "Temp_Schemes/ERKWeightOpt.hpp"
#include "Temp_Schemes/IRKWeight.hpp"
#include "Temp_Schemes/IRKWeightOpt.hpp"
#include "Temp_Schemes/SSPRKStep.hpp"
#include "Temp_Schemes/SSPRKStepOpt.hpp"


// Prototype sources 
#include "Prototypes/HHO_Steady_First_Order.hpp"

#include "Prototypes/IHHOSecondOrder.hpp"
#include "Prototypes/IHHOFirstOrder.hpp"
#include "Prototypes/HHOFirstOrderExample.hpp"
#include "Prototypes/EHHOFirstOrder.hpp"
#include "Prototypes/HeterogeneousIHHOFirstOrder.hpp"
#include "Prototypes/HeterogeneousIHHOSecondOrder.hpp"


int main(int argc, char **argv)
{

// Steady diffusion problem
 HHOSteadyFirstOrder(argc, argv);

// First order 
// HHOFirstOrderExample(argc, argv);	
// IHHOFirstOrder(argc, argv); 		
// EHHOFirstOrder(argc, argv);		
 
// Second order
// IHHOSecondOrder(argc, argv); 		 
  
// Heterogeneous simulations
// HeterogeneousIHHOFirstOrder(argc, argv);    
// HeterogeneousIHHOSecondOrder(argc, argv);	  
   
   return 0;
}




































