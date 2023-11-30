/*
 *       /\        Guillaume Delay 2018,2019
 *      /__\       guillaume.delay@enpc.fr
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

#include<iostream>
#include<fstream>
#include<vector>
#include<array>
#include<algorithm>
#include<numeric>
#include<cassert>
#include<cmath>
#include<memory>
#include<sstream>
#include<list>
#include<map>
#include<iomanip>
#include<stdbool.h>

#include<Eigen/Dense>
#include<Eigen/SparseCore>
#include<Eigen/SparseLU>
#include<unsupported/Eigen/SparseExtra>
#include<Spectra/SymEigsSolver.h>
#include<Spectra/MatOp/SparseSymMatProd.h>
#include<unsupported/Eigen/MatrixFunctions> // ADD BY STEFANO

#include"dataio/silo_io.hpp"
#include"core/core"
#include"core/solvers"
#include"methods/hho"
#include"methods/cuthho"

//#include "tbb/tbb.h"
//#define HAVE_INTEL_TBB
//#include "/usr/local/Cellar/tbb/2020_U2/include/tbb/tbb.h"
//#include "/opt/intel/compilers_and_libraries_2020.1.216⁩/mac/tbb/include/tbb/tbb.h"
//using namespace tbb;

using namespace Eigen;
using namespace computationQuantities;
using namespace random_functions;
using namespace level_set_transport;
using namespace stokes_info;
using namespace unsteady_stokes_terms;
using namespace pre_processing;



//-------------     MAIN     -------------


// Interface Stokes Problem: Two-fluid problem STATIONARY

#if 0

int main(int argc, char **argv) {
    using RealType = double;
    RealType sizeBox = 2;
    size_t degree = 0;
    size_t int_refsteps = 4;
    size_t degree_FEM = 0;
    size_t degree_curve = 2;
    size_t degree_curvature = 1; // degree_curve -1 ;
    bool dump_debug = false;
    bool solve_interface = false;
    bool solve_fictdom = false;
    bool agglomeration = false;

    bool high_order = false; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool entropic = false; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool entropic_mass_consistent = false;
    bool compressed = false; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool cut_off_active = false; // IF FALSE IS SMOOTH, IF TRUE  CUT_OFF

    mesh_init_params <RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;

    mip.min_x = -sizeBox;
    mip.min_y = -sizeBox;
    mip.max_x = sizeBox;
    mip.max_y = sizeBox;
    size_t T_N = 0;
    int ch;
    while ((ch = getopt(argc, argv, "k:q:M:N:r:T:l:p:ifDAdhesgc")) != -1) {
        switch (ch) {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;

            case 'T':
                T_N = atoi(optarg);
                break;

            case 'l':
                degree_curve = atoi(optarg);
                break;

            case 'p':
                degree_curvature = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;

            case 'h':
                high_order = true;
                break;

            case 'e':
                entropic = true;
                break;

            case 's':
                entropic_mass_consistent = true;
                break;

            case 'g':
                compressed = true;
                break;

            case 'c':
                cut_off_active = true;
                break;


            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;


    timecounter tc;

    timecounter tc_tot;
    tc_tot.tic();


/************** BUILD MESH **************/

    cuthho_poly_mesh <RealType> msh(mip);
    typedef cuthho_poly_mesh <RealType> Mesh;
    typedef RealType T;
//    typedef typename Mesh::point_type point_type;
    offset_definition(msh);

    std::cout << "Mesh size = " << mip.Nx << "x" << mip.Ny << std::endl;
    std::cout << "Number of refine interface points: r = " << int_refsteps << std::endl;

    if (high_order == true)
        std::cout << bold << green << "Transport problem high order with limiting (no inlet). " << reset << std::endl;
    else if (entropic == true)
        std::cout << bold << green << "Transport problem entropic mass lumped (no inlet). " << reset << std::endl;
    else if (entropic_mass_consistent == true)
        std::cout << bold << green << "Transport problem entropic mass consistent (no inlet). " << reset << std::endl;
    else if (compressed == true)
        std::cout << bold << green << "Transport problem high order compressed with limiting (no inlet). " << reset
                  << std::endl;
    else
        std::cout << bold << green << "Transport problem low order (no inlet).  " << reset << std::endl;

/************** FINITE ELEMENT INITIALIZATION **************/
    auto fe_data = Finite_Element<RealType, Mesh>(msh, degree_FEM, mip);
    typedef Finite_Element<RealType, Mesh> FiniteSpace;
    std::cout << "Level Set (finite element approximation): Bernstein basis in space Q^{k_phi},  k_phi = " << degree_FEM
              << std::endl;

/**************************************TRANSPORT PROBLEM METHOD *****************************************/
    auto method_transport_pb = Transport_problem_method<Mesh, FiniteSpace>(fe_data, msh);
//typedef  Transport_problem_method<Mesh, FiniteSpace> Method_Transport;

    size_t degree_gradient = degree_FEM;
    size_t degree_div = degree_FEM;
    std::cout << "Finite element space for gradient and divergence of the LS: grad deg = " << degree_gradient
              << " , div deg = " << degree_div << std::endl;

    auto fe_data_gradient = Finite_Element<RealType, Mesh>(msh, degree_gradient, mip);
    auto method_transport_pb_grad = Transport_problem_method<Mesh, FiniteSpace>(fe_data_gradient, msh);
    auto fe_data_div = Finite_Element<RealType, Mesh>(msh, degree_div, mip);
    auto method_transport_pb_div = Transport_problem_method<Mesh, FiniteSpace>(fe_data_div, msh);


/************** ANALYTIC LEVEL SET FUNCTION  **************/


    bool circle = false, ellipse = true;
    bool flower = false;
    RealType radius_a, radius_b, radius;
    RealType x_centre = 0.0; // 0.5
    RealType y_centre = 0.0;
//T h = std::max( fe_data.hx , fe_data.hy) ;
    if (circle) {
        radius = 1.0 / 6.0; // I ALWAYS USED 1.0/9.0
    }

    if (ellipse) {
        radius = 1.0 / 3.0;
//        radius_a = 1.0/12.0; // 6, 12, 18 varying the domain size (1, x2, x3)
//        radius_b = 1.0/24.0; // 12, 24, 36 varying the domain size (1, x2, x3)
//        radius_a = 1.0/12.0;
//        radius_b = 1.0/24.0;
        radius_a = 1.0 / 3.0;
        radius_b = 1.0 / 6.0;
//        T eps_circ = 1e-4;
//        radius_a = 1.0/3.0-eps_circ;
//        radius_b = 1.0/3.0+eps_circ;
        std::cout << bold << yellow << "Initial Analytic Area of the ELLIPSE: " << M_PI * radius_a * radius_b
                  << std::endl;
        radius = sqrt(radius_a * radius_b);
        std::cout << bold << yellow << "Final radius expected of the circle : " << radius << reset << std::endl;
    }




///---------->!!!!!!!!  THIS DATA BELOW HAS TO BE UPLOAD DEPENDING ON THE PROBLEM:

// ------------------------------------ CIRCLE LEVEL SET ------------------------------------
//    std::cout<<"Initial interface: CIRCLE"<<std::endl;
//    auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre, y_centre );
//    typedef  circle_level_set<T> Fonction;

//    std::cout<<"Initial interface: CIRCLE"<<std::endl;
//    auto level_set_function_anal = circle_level_set_signed_distance<RealType>(radius, x_centre, y_centre , 0.1 ); // , 0.01 --> eps to smooth gradient
//    typedef  circle_level_set_signed_distance<T> Fonction;

// ------------------------------------ FLOWER LEVEL SET ------------------------------------

//    radius = 0.31 ;
//    std::cout<<"Initial interface: FLOWER"<<std::endl;
////    auto level_set_function_anal = flower_level_set<T>(radius, x_centre, y_centre, 4, 0.04); //0.11
////    typedef  flower_level_set<T> Fonction;
//    flower = true ;
//
//    auto level_set_function_anal = flower_level_set_signed_distance<T>(radius, x_centre, y_centre, 4, 0.04); //0.11
//    typedef  flower_level_set_signed_distance<T> Fonction;



// ------------------------------------ ELLIPTIC LEVEL SET -----------------------------------
    std::cout << "Initial interface: ELLIPSE" << std::endl;
    auto level_set_function_anal = elliptic_level_set<RealType>(radius_a, radius_b, x_centre, y_centre);
    typedef elliptic_level_set <T> Fonction;

// ------------> OLD STUFF IMPLEMENTATION
//auto level_set_function_anal = elliptic_distance_ls<RealType>( radius_a, radius_b, x_centre, y_centre , h);
//typedef  elliptic_distance_ls<T> Fonction;
//auto level_set_function_anal = circle_distance_ls<RealType>(radius, x_centre, y_centre ,2*h );
//typedef  circle_distance_ls<T> Fonction;


    T curvature_anal = 1.0 / radius;

/**************  VELOCITY FIELD  INITIALISATION  **************/


    size_t degree_velocity = degree_FEM; // std::max(degree + 1 , degree_FEM) ;


// **************** --------> STORING OF LAGRANGIAN NODES
    nodes_Lagrangian_cell_definition(msh, degree_velocity);


    auto fe_data_Lagrange = Finite_Element<RealType, Mesh>(msh, degree_velocity, mip);

    std::cout << "Velocity field: high order Lagrange basis: degree = " << degree_velocity << std::endl;
    auto u_projected = velocity_high_order<Mesh, FiniteSpace, T>(fe_data_Lagrange, msh);



/************** LEVEL SET FUNCTION DISCRETISATION **************/


//    if(high_order)
//        std::cout<<"--------------------> USING phi^H - HIGH ORDER TRANSPORT PB "<<std::endl;
//    else
//        std::cout<<"--------------------> USING phi^L - LOW ORDER TRANSPORT PB "<<std::endl;

//    std::cout<<"Level set: high order Berstein x interpolated."<<std::endl;

//-------->  NEW FORMULATION
    auto level_set_function = Level_set_berstein<Mesh, Fonction, FiniteSpace, T>(fe_data, level_set_function_anal, msh,
                                                                                 fe_data_gradient, fe_data_div);
    typedef Level_set_berstein<Mesh, Fonction, FiniteSpace, T> Level_Set;
// ------------------  IF GRADIENT CONTINUOUS --------------
    level_set_function.gradient_continuous_setting(method_transport_pb_grad);
//  ------------------ IF DIVERGENCE CONTINUOUS  ------------------
//    level_set_function.divergence_continuous_setting(method_transport_pb_div ) ;

//-------->  OLD FORMULATION GRAD CONT
//    auto level_set_function = Level_set_berstein_high_order_interpolation_grad_cont< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
//    level_set_function.gradient_continuous_setting() ;
//    typedef Level_set_berstein_high_order_interpolation_grad_cont< Mesh , Fonction , FiniteSpace , T > Level_Set;


    std::cout << "Parametric interface: degree_curve = " << degree_curve << std::endl;
    auto curve = Interface_parametrisation_mesh1d(degree_curve);
    size_t degree_det_jac_curve = curve.degree_det; // 2*degree_curve INUTILE PER ORA
// integration CUT CELL degree += 2*degree_curve
// integration INTERFACE degree += degree_curve-1

//auto level_set_function = Level_set_berstein_high_order_interpolation_grad_cont_fast< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
//auto level_set_function = Level_set_berstein_high_order_interpolation< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);





/************** MESH INITIALISATION FOR ROUTINE  **************/


    auto crr_mesh = Current_Mesh<Mesh>(msh);
    crr_mesh.current_mesh = msh;
    Mesh msh_i = crr_mesh.current_mesh;      // MESH at t=t^n (FOR THE PROCESSING)
    offset_definition(msh_i);


/************** INITIAL DATA INITIALISATION (t = 0) **************/
    T dt = 0.;
    T initial_area = 0., initial_mass = 0.;
    T d_a = 0.;
    T perimeter_initial = 0.;
    T centre_mass_x_inital = 0., centre_mass_y_inital = 0.;
    T max_u_n_val_old = 1e+6, max_u_n_val_new = 1e+5;
    T check = 10.0;
    T tot_time = 0.;

/************** BOUNDARY CONDITIONS **************/
    std::cout << yellow << bold << "INLET BDRY: UP AND DOWN FOR DIRCIRCHLET_eps FP" << reset << std::endl;
    bool bdry_bottom = true, bdry_up = true;
    bool bdry_left = false, bdry_right = false;
    check_inlet(msh, fe_data, bdry_bottom, bdry_right, bdry_up, bdry_left, 1e-14);


//************ DO cutHHO MESH PROCESSING **************
    tc.tic();
    detect_node_position3(msh_i, level_set_function); // In cuthho_geom
//detect_node_position3_parallel(msh_i, level_set_function); // In cuthho_geom
    detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom

    if (agglomeration) {
        detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom
//detect_cut_cells3_parallelized(msh_i, level_set_function); // In cuthho_geom
        refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
        set_integration_mesh(msh_i, degree_curve);
        detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
        make_neighbors_info_cartesian(msh_i); // Non serve modificarla
//refine_interface_pro3(msh_i, level_set_function, int_refsteps);

        make_agglomeration_no_double_points(msh_i, level_set_function, degree_det_jac_curve);
        set_integration_mesh(msh_i, degree_curve); // TOLTO PER IL MOMENTO SENNO RADDOPPIO


    } else {
        detect_cut_cells3(msh_i, level_set_function);
        refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
        set_integration_mesh(msh_i, degree_curve);
    }

    tc.toc();
    std::cout << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << '\n' << std::endl;
    std::cout << "Agglomerated amount cells: " << msh_i.cells.size() << std::endl;
//    if (dump_debug)
//    {
//        dump_mesh(msh_i);
//        output_mesh_info(msh_i, level_set_function);
//    }



// IN cuthho_export..Points/Nodes don't change-> it's fast
    output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export

/************** UPDATING  LEVEL SET  AND VELOCITY  **************/
//    level_set_function.gradient_continuous_setting() ;
//    // IF GRADIENT CONTINUOUS
    level_set_function.gradient_continuous_setting(method_transport_pb_grad);
//    // IF DIVERGENCE CONTINUOUS
//    level_set_function.divergence_continuous_setting(method_transport_pb_div) ;


// --------------------- LS_CELL: CHOICE OF DISC/CONT ------------------------------- \\

// IF grad cont -> normal cont -> (divergence disc) -> divergence cont
//auto ls_cell = LS_cell_high_order_curvature_cont< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);

// IF grad cont -> normal cont -> divergence disc
    auto ls_cell = LS_cell_high_order_grad_cont_div_disc<T, Mesh, Level_Set, Fonction, FiniteSpace>(level_set_function,
                                                                                                    msh_i);

// IF grad disc -> normal disc -> divergence disc
//    auto ls_cell = LS_cell_high_order_grad_disc_div_disc< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);
// IF grad disc -> normal disc -> divergence disc -> normal and grad cont
//    auto ls_cell = LS_cell_high_order_div_disc_grad_n_cont< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);

//-------------------------- OLD CASE LS_CELL --------------------------
//    auto ls_cell = LS_cell_high_order_grad_cont< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i );



    ls_cell.radius = radius;
    u_projected.set_agglo_mesh(msh_i);

    timecounter tc_initial;
    tc_initial.tic();



/************** PLOTTINGS + GOAL QUANTITIES  **************/
    std::vector <T> max_val_u_n_time_para, L1_err_u_n_time_para, l1_err_u_n_time_para, linf_err_u_n_time_para, L1_err_u_n_time;
    std::vector <T> area_time, l1_err_u_n_time, linf_err_u_n_time, time_vec;
    std::vector <T> linf_der_time_interface, eccentricity_vec;
    std::vector <T> max_val_u_n_time, l1_err_curvature_time, linf_err_curvature_time;
    std::vector <T> circularity_time, circularity_time2, flux_interface_time, perimeter_time;
    std::vector <std::pair<T, T>> centre_mass_err_time, rise_velocity_time, min_max_vec;
    T circularity_ref = 0.0, perim_ref = 0.0, area_ref = 0.0;
    T l1_divergence_error = 0., l2_divergence_error = 0.;
    T linf_divergence_error = -10.;
    T L1_divergence_error = 0.;

    check_disc_curvature(msh_i, ls_cell, curvature_anal, degree_FEM);


    check_goal_quantities(msh_i, ls_cell, perimeter_initial, d_a, initial_area, centre_mass_x_inital,
                          centre_mass_y_inital, degree_FEM, initial_mass, flower, l1_divergence_error,
                          l2_divergence_error, linf_divergence_error, radius, L1_divergence_error, ellipse,
                          degree_curve, int_refsteps);



//------------------------ CHECK REFERENCE QUANTITIES ---------------------------//
//    reference_quantities_computation(perim_ref,area_ref,circularity_ref,radius, x_centre, y_centre,fe_data , msh, degree_curve,perimeter_initial , initial_area,int_refsteps , degree_det_jac_curve);

    T perimeter_anal = 2.0 * M_PI * sqrt((radius_a * radius_a + radius_b * radius_b) / 2.0); //   2.0*M_PI*radius ;
    std::cout << "Error( perimetre_anal - perimeter_initial ) = " << perimeter_anal - perimeter_initial << std::endl;
    T area_anal = M_PI * radius_a * radius_b; // M_PI*radius*radius ;
    std::cout << "Error( area_anal - initial_area ) = " << area_anal - initial_area << std::endl;

//    plot_curvature_normal_vs_curv_abscisse(msh_i, ls_cell, degree_curve,int_refsteps , 0 );

    plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(msh_i, ls_cell, degree_curve, int_refsteps, 0,
                                                      degree_curvature); // NO FILTER


    tc_initial.toc();
//    std::cout << "Time Machine for checking INITAL GOAL QUANTITIES: " << tc_initial << " seconds" << std::endl;

    circularity_time.push_back(M_PI * d_a / perimeter_initial);
    circularity_time2.push_back(4.0 * M_PI * initial_area / (perimeter_initial * perimeter_initial));
    perimeter_time.push_back(perimeter_initial);
    centre_mass_err_time.push_back(
            std::make_pair(centre_mass_x_inital / initial_area, centre_mass_y_inital / initial_area));
    time_vec.push_back(0);
    area_time.push_back(initial_area);

    min_max_vec.push_back(std::make_pair(level_set_function.phi_min, level_set_function.phi_max));

    l1_err_curvature_time.push_back(l1_divergence_error);
    linf_err_curvature_time.push_back(linf_divergence_error);

    T dt_M;
    T R_phi = radius;

    size_t tot_amount_transport_routine = 0;


    bool l2proj_para = false;

    bool l2proj = true;
    bool avg = false;
    bool disc = false;
    bool filter = false;

    Interface_parametrisation_mesh1d_global <Mesh> para_curve_cont(msh_i, degree_curve, degree_curvature);

// *********************** DERIVATIVE / NORMAL PARA *************************//
//------------- L2 cont curvature from parametric interface  r ---------- //
    para_curve_cont.make_L2_proj_para_derivative(msh_i);

//---------------------------- L2 global Normal from LS  ----------------------- //
    if (l2proj) {
        if (!disc)
            para_curve_cont.make_L2_proj_para_normal(msh_i, ls_cell);
        else
            para_curve_cont.make_L2_proj_para_normal_disc(msh_i, ls_cell);
    }
//---------------------------- Avg Normal from LS  ---------------------------- //
    if (avg) {
        if (!disc)
            para_curve_cont.make_avg_L2_local_proj_para_normal(msh_i, ls_cell);
        else
            para_curve_cont.make_avg_L2_local_proj_para_normal_disc(msh_i, ls_cell);
    }


// *********************** CURVATURE PARA *************************//

//------------- L2 cont curvature from parametric interface  r ---------- //
    if (l2proj_para)
        para_curve_cont.make_L2_proj_para_curvature(msh_i);



//---------------------------- L2 global Curvature from LS  ----------------------- //
    if (l2proj) {
        if (!disc)
            para_curve_cont.make_L2_proj_para_curvature(msh_i, ls_cell);
        else
            para_curve_cont.make_L2_proj_para_curvature_disc(msh_i, ls_cell);
    }
//---------------------------- Avg Curvature from LS  ---------------------------- //
    if (avg) {
        if (!disc)
            para_curve_cont.make_avg_L2_local_proj_para_curvature(msh_i, ls_cell);
        else
            para_curve_cont.make_avg_L2_local_proj_para_curvature_disc(msh_i, ls_cell);

    }
    if (filter) {
        std::cout << bold << yellow << "CURVATURE ANALYSIS PRE FILTER:" << '\n';
        para_curvature_error(msh_i, para_curve_cont, curvature_anal);
        para_curve_cont.make_smooth_filter_curvature();
        std::cout << bold << yellow << "CURVATURE ANALYSIS POST FILTER:" << '\n';
        para_curvature_error(msh_i, para_curve_cont, curvature_anal);

    }
    if (!filter) {
        std::cout << bold << yellow << "CURVATURE ANALYSIS PARA:" << '\n';
        para_curvature_error(msh_i, para_curve_cont, curvature_anal);
    }

// ******** TO FASTER THE SIMULATION, ERASED THE PLOTTINGS
    plotting_para_curvature_cont_time_fast(msh_i, para_curve_cont, degree_curve, degree_FEM, radius, 0, int_refsteps);

// output_mesh_info_ls_l_n(msh_i, para_curve_cont);

    T eps_dirichlet_cond = 1.0; //1.0/1.91008 ; // 0.01 -->  0.1


    std::cout << "Dirichlet eps Cond = " << eps_dirichlet_cond << std::endl;

// -----------------------------------------------------------------------------------------
// ----------------- RESOLUTION OF THE STOKES PROBLEM (HHO) ------------------
// -----------------------------------------------------------------------------------------

    bool sym_grad = TRUE;
    auto prm = params<T>();
    prm.kappa_1 = 0.1;
    prm.kappa_2 = 1.0;
    T gamma = 0.0; // 0.05
// kappa_1 is negative domain (inner)
// kappa_2 is positive domain (outer)

    std::cout << "gamma = " << gamma << std::endl;


    std::cout << '\n' << bold << yellow << "HHO flow resolution." << reset << '\n' << std::endl;


// ------------------ OLD VERSIONS ------------------
//auto test_case = make_test_case_eshelby(msh_i, ls_cell,  prm , sym_grad);
// Non serve modificare Gamma = 1/2
//auto test_case = make_test_case_eshelby_2(msh_i, ls_cell,  prm , sym_grad );
//auto test_case = make_test_case_eshelby_analytic(msh_i, ls_cell,  prm , sym_grad , radius);
// ------------- OLD GUILLAUME VERSIONS --------------
// auto test_case = make_test_case_stokes_1(msh, level_set_function);
// auto test_case = make_test_case_stokes_2(msh, ls_cell); //level_set_function);

// ----------------- ESHELBY VERSION - CORRECT BUT PRESSURE ------------------
//auto test_case_prova = make_test_case_eshelby_2_prova(msh_i, ls_cell,  prm , sym_grad );
// PRESSURE SIGN NOT CORRECT
// ---------------------- ESHELBY VERSION LEVEL SET - CORRECT ------------------------
//        auto test_case = make_test_case_eshelby_correct(msh_i, ls_cell,  prm , sym_grad,gamma);
// PRESSURE SIGN NOT CORRECT
// -------------------- ESHELBY VERSION PARAMETRIC (DISC) - CORRECT -------------------
//        auto test_case = make_test_case_eshelby_correct_parametric(msh_i, ls_cell,  prm , sym_grad,gamma);
// PRESSURE SIGN NOT CORRECT




// -------------------- ESHELBY VERSION PARAMETRIC (CONT) - CORRECT -------------------
// ---> THE OLD ONE FOR DIRICHLET =0
//        auto test_case_prova = make_test_case_eshelby_correct_parametric_cont( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma ); // SIGN OF PRESSURE HAS TO BE CHANGED CONFORMING TO THE CHANGE  OF SIGN OF THE CURVATURE. HERE DONE: CORRECT 25/01/2021

// PARAMETRIC normal and curvature
//        auto test_case_prova = make_test_case_eshelby_correct_parametric_cont_DIRICHLET_eps( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond); // SIGN OF PRESSURE HAS TO BE CHANGED CONFORMING TO THE CHANGE  OF SIGN OF THE CURVATURE.

// LS normal and curvature

    auto test_case = make_test_case_eshelby_LS_eps_DIR_domSym(msh_i, ls_cell, para_curve_cont, prm, sym_grad, gamma,
                                                              eps_dirichlet_cond, sizeBox);
// auto test_case = make_test_case_eshelby_LS_eps_DIR( msh_i, ls_cell ,para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond);

// ------------------------ HHO METHOD FOR LEVEL SET  ---------------------------
//        auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case, sym_grad);
    T alfa1 = prm.kappa_2 / (prm.kappa_1 + prm.kappa_2);
    T alfa2 = prm.kappa_1 / (prm.kappa_1 + prm.kappa_2);
// alfa1 is negative domain (inner)
// alfa2 is positive domain (outer)
    auto method = make_sym_gradrec_stokes_interface_method_alfai(msh_i, 1.0, 0.0, test_case, sym_grad, alfa1, alfa2);
// -------------------- HHO METHOD FOR DISC PARAMETRIC INTERFACE  -----------------------
//         auto method = make_sym_gradrec_stokes_interface_method_ref_pts(msh_i, 1.0, 0.0, test_case, sym_grad);

// -------------------- HHO METHOD FOR CONT PARAMETRIC INTERFACE  -----------------------
//        auto method_prova = make_sym_gradrec_stokes_interface_method_ref_pts_cont(msh_i, 1.0, 0.0, test_case_prova, sym_grad); // WITH H_p  I use this!!!!




//  ******************** - HHO RESOLUTION - ********************
    if (solve_interface) {
// ----------------- HHO RESOLUTION OLD CASE  --------------------------
//            TI = run_cuthho_interface_numerical_ls(msh_i, degree, method, test_case_prova , ls_cell ,  normal_analysis );
//            run_cuthho_interface_velocity_parallel(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad );

// ----------------- HHO RESOLUTION LS / PARAMETRIC DISC  ---------------------
//            run_cuthho_interface_velocity_prova(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad , time_step); // THE ONE CORRECT THAT I'M USING NOW

// ----------------- HHO RESOLUTION PARAMETRIC CONT  --------------------------
//             run_cuthho_interface_velocity_new(msh_i, degree, method_prova,test_case_prova, ls_cell , u_projected ,sym_grad , time_step); // THE ONE CORRECT THAT I'M USING NOW
//            run_cuthho_interface_velocity_new_post_processing(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad , 0); // FOR PARAMETRIC H_p
//            run_cuthho_interface_velocity_fast(msh_i, degree, method_prova,test_case_prova, ls_cell , u_projected ,sym_grad , time_step); // CORRECT BUT DOEST NOT COMPUTE ERRORS

        run_cuthho_interface_velocity_new_post_processingLS(msh_i, degree, method, test_case, ls_cell, u_projected,
                                                            sym_grad, 0, 1); // FOR LS H
    }
//        testing_velocity_field(msh , u_projected) ;
/************************************ FEM -  PRE-PROCESSING ******************************************/
// ----------------- PROJECTION OF THE VELOCITY FIELD ------------------
    if (0)
        std::cout << bold << green
                  << "CASE WITH VELOCITY DISCONTINUOUS: ho solo sol_HHO, sol_FEM non salvato, va cambiato il transport pb!!!"
                  << reset << std::endl;

    if (1) //1 FIRST RESULT WITH THIS
    {
        std::cout << '\n' << "Smoothing operator from velocity HHO to FE (continuity imposed): geometrical average."
                  << std::endl;
        u_projected.smooth_converting_into_FE_formulation(u_projected.sol_HHO);
    }
    if (0) {
        std::cout << '\n' << "------------------>>>> NOTICE: NON SMOOTH OPERATOR FROM HHO TO FEM." << std::endl;
        u_projected.converting_into_FE_formulation(u_projected.sol_HHO);
    }
    if (0) {
        std::cout << '\n' << "------------------>>>>NOTICE: L^2 PROJECTION FROM HHO TO FEM." << std::endl;
        u_projected.L2_proj_into_FE_formulation(level_set_function, msh, method_transport_pb);
    }


//testing_velocity_field(msh , u_projected) ;
//auto u_prova = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
//u_prova.sol_HHO = u_projected.sol_HHO ;
//u_prova.L2_proj_into_FE_formulation( level_set_function , msh );
//testing_velocity_field_L2projected(msh , u_prova) ;

    T rise_vel0 = 0.0, rise_vel1 = 0.0;
    T flux_interface = 0.0;


    T max_u_n_val_initial = 0.0;
    T max_u_n_val_abs_initial = 0.0;
    T l1_normal_interface_status_initial = 0.;

    T L1_normal_interface_status_initial = 0.;

    size_t counter_interface_pts_initial = 0;
    for (auto &cl: msh_i.cells) {
        if (cl.user_data.location == element_location::ON_INTERFACE) {
            ls_cell.cell_assignment(cl);
            u_projected.cell_assignment(cl);

            auto qps = integrate_interface(msh_i, cl, degree_FEM + degree_velocity, element_location::ON_INTERFACE);
            for (auto &qp: qps) {
                auto u_pt = u_projected(qp.first);

                auto ls_n_pt = ls_cell.normal(qp.first);
                T u_n_val = u_pt.first * ls_n_pt(0) + u_pt.second * ls_n_pt(1);
                L1_normal_interface_status_initial += qp.second * std::abs(u_n_val);
                max_u_n_val_abs_initial = std::max(max_u_n_val_abs_initial, std::abs(u_n_val));
                if (std::abs(u_n_val) == max_u_n_val_abs_initial)
                    max_u_n_val_initial = u_n_val;

                l1_normal_interface_status_initial += std::abs(u_n_val);
                counter_interface_pts_initial++;


            }

        }

    }
    l1_normal_interface_status_initial /= counter_interface_pts_initial;
    l1_err_u_n_time.push_back(l1_normal_interface_status_initial);
    linf_err_u_n_time.push_back(max_u_n_val_abs_initial);
//            linf_der_time_interface.push_back(0) ;
    max_val_u_n_time.push_back(max_u_n_val_initial);
    L1_err_u_n_time.push_back(L1_normal_interface_status_initial);

    std::cout << "------> The l1 error of u*n along the INTERFACE at INITIAL TIME is "
              << l1_normal_interface_status_initial << std::endl;

    std::cout << bold << green << "------> The linf error of u*n along the INTERFACE at INITIAL TIME is "
              << max_u_n_val_abs_initial << reset << std::endl;

    std::cout << "------> The L1 error of u*n along the INTERFACE at INITIAL TIME is "
              << L1_normal_interface_status_initial << std::endl;


    size_t degree_jacobian = para_curve_cont.degree_det;

    T L1_normal_interface_para = 0.0;
    T linf_u_n_para = 0.0;
    T max_u_n_val_para = 0.0;
    T l1_normal_interface_para = 0.0;
    size_t counter_interface_pts_para = 0;


    T area_para = 0.0;

    for (auto &cl: msh_i.cells) {

        if ((location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE) ||
            (location(msh_i, cl) == element_location::ON_INTERFACE)) {
            u_projected.cell_assignment(cl);
            T partial_area = measure(msh_i, cl, element_location::IN_NEGATIVE_SIDE);
            area_para += partial_area;

            size_t max_deg = std::max(degree_velocity, degree_FEM);
            auto qps_fin = integrate(msh_i, cl, max_deg, element_location::IN_NEGATIVE_SIDE);


            for (auto &qp: qps_fin) {
                auto u_pt = u_projected(qp.first);
                rise_vel0 += qp.second * u_pt.first;
                rise_vel1 += qp.second * u_pt.second;
            }

        }


        if (cl.user_data.location == element_location::ON_INTERFACE) {
            u_projected.cell_assignment(cl);
            auto global_cells_i = para_curve_cont.get_global_cells_interface(msh_i, cl);
            auto integration_msh = cl.user_data.integration_msh;
//                    auto degree_int = degree_curvature + degree_jacobian ;


            auto qps_un = edge_quadrature<T>(degree_jacobian + degree_curvature + degree_velocity);

            for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                size_t global_cl_i = global_cells_i[i_cell];

                for (auto &qp: qps_un) {
                    auto t = 0.5 * qp.first.x() + 0.5;

                    T jacobian = para_curve_cont.jacobian_cont(t, global_cl_i);
                    auto w = 0.5 * qp.second * jacobian;
                    auto p = para_curve_cont(t, global_cl_i);
                    auto pt = typename Mesh::point_type(p(0), p(1));
                    auto u_pt = u_projected(pt);
                    auto curve_n_pt = para_curve_cont.normal_cont(t, global_cl_i);
                    T flux = u_pt.first * curve_n_pt(0) + u_pt.second * curve_n_pt(1);
                    flux_interface += w * flux;

                    L1_normal_interface_para += w * std::abs(flux);
                    linf_u_n_para = std::max(linf_u_n_para, std::abs(flux));
                    if (std::abs(flux) == linf_u_n_para)
                        max_u_n_val_para = flux;

                    l1_normal_interface_para += std::abs(flux);
                    counter_interface_pts_para++;

                }
            }
        }
    }
    l1_normal_interface_para /= counter_interface_pts_para;
    l1_err_u_n_time_para.push_back(l1_normal_interface_para);
    linf_err_u_n_time_para.push_back(linf_u_n_para);
    L1_err_u_n_time_para.push_back(L1_normal_interface_para);
    max_val_u_n_time_para.push_back(max_u_n_val_para);
    flux_interface_time.push_back(flux_interface);
    rise_velocity_time.push_back(std::make_pair(rise_vel0 / area_para, rise_vel1 / area_para));

    std::cout << "------> The l1 error of PARA_CONT u*n along the INTERFACE at INITIAL TIME is "
              << l1_normal_interface_para << std::endl;

    std::cout << bold << yellow << "------> The linf error of PARA_CONT u*n along the INTERFACE at INITIAL TIME is "
              << linf_u_n_para << reset << std::endl;

    std::cout << "------> The L1 error of PARA_CONT u*n along the INTERFACE at INITIAL TIME is "
              << L1_normal_interface_para << std::endl;


    tc_tot.toc();
    std::cout << "Simulation machine time t = " << tc_tot << std::endl;

    return 0;
}

#endif




// Interface Stokes Problem: Two-fluid problem STATIONARY
// Loop varying \mu and ellipse radii

#if 1

int main(int argc, char **argv) {
    using RealType = double;
    RealType sizeBox = 2.0;
    std::string folder = "simulation_fullGradient";
    size_t degree = 1;
    size_t int_refsteps = 4;
    size_t degree_FEM = 0;
    size_t degree_curve = 2;
    size_t degree_curvature = 1; // degree_curve -1 ;
    bool dump_debug = false;
    bool solve_interface = false;
    bool solve_fictdom = false;
    bool agglomeration = false;

    bool high_order = false; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool entropic = false; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool entropic_mass_consistent = false;
    bool compressed = false; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool cut_off_active = false; // IF FALSE IS SMOOTH, IF TRUE  CUT_OFF

    mesh_init_params <RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;

    mip.min_x = -sizeBox;
    mip.min_y = -sizeBox;
    mip.max_x = sizeBox;
    mip.max_y = sizeBox;
    size_t T_N = 0;
    int ch;
    while ((ch = getopt(argc, argv, "k:q:M:N:r:T:l:p:ifDAdhesgc")) != -1) {
        switch (ch) {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;

            case 'T':
                T_N = atoi(optarg);
                break;

            case 'l':
                degree_curve = atoi(optarg);
                break;

            case 'p':
                degree_curvature = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;

            case 'h':
                high_order = true;
                break;

            case 'e':
                entropic = true;
                break;

            case 's':
                entropic_mass_consistent = true;
                break;

            case 'g':
                compressed = true;
                break;

            case 'c':
                cut_off_active = true;
                break;


            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;


    timecounter tc;

    timecounter tc_tot;
    tc_tot.tic();


    std::vector <RealType> int_refstepsVec{0};
    std::vector <RealType> degree_curveVec{4};
    std::vector <RealType> mu_vec{1.0}; // 0.01, 0.1, 1.0, 10.0, 100.0
    std::vector <RealType> radius_a_vec{1.0 / 3.0}; // , 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    std::vector <RealType> radius_b_vec{1.0 / 6.0}; // , 2.0 / 9.0, 4.0 / 15.0, 10.0 / 57.0, 10.0 / 33.0, 4.0 / 21.0


    int nOfRadii = radius_a_vec.size();

    std::vector <RealType> testCaseGamma{0.0, 1.0}; // {1.0};
    std::vector <RealType> testCaseEps{1.0, 0.0}; // {0.0};

    int nOfTestCases = testCaseGamma.size();
    size_t counter_tests = 0;
    for (auto mu: mu_vec) // for (auto int_refsteps1 : int_refstepsVec) //  for (auto mu : mu_vec)
    {
// auto mu = mu_vec.at(0);
// int R_i = 0 ;
        int_refsteps = int_refstepsVec.at(0); // int_refsteps1;
        for (int iTestCase = 0; iTestCase < nOfTestCases; iTestCase++) {
            RealType gamma = testCaseGamma.at(iTestCase);
            RealType eps_dirichlet_cond = testCaseEps.at(iTestCase);

            for (int R_i = 0; R_i <
                              nOfRadii; R_i++) // for (auto degree_curve1 : degree_curveVec) // for (int R_i = 0 ; R_i < nOfRadii ; R_i++)
            {
                degree_curve = degree_curveVec.at(0); //  degree_curve1;
/************** BUILD MESH **************/

                cuthho_poly_mesh <RealType> msh(mip);
                typedef cuthho_poly_mesh <RealType> Mesh;
                typedef RealType T;
                offset_definition(msh);
                std::cout << "Mesh size = " << mip.Nx << "x" << mip.Ny << std::endl;
                std::cout << "Number of refine interface points: r = " << int_refsteps << std::endl;
/************** FINITE ELEMENT INITIALIZATION **************/
                auto fe_data = Finite_Element<RealType, Mesh>(msh, degree_FEM, mip);
                typedef Finite_Element<RealType, Mesh> FiniteSpace;
                std::cout << "Level Set (finite element approximation): Bernstein basis in space Q^{k_phi},  k_phi = "
                          << degree_FEM << std::endl;

/**************************************TRANSPORT PROBLEM METHOD *****************************************/
// auto method_transport_pb = Transport_problem_method<Mesh, FiniteSpace>(fe_data, msh) ;
                size_t degree_gradient = degree_FEM;
                size_t degree_div = degree_FEM;
                std::cout << "Finite element space for gradient and divergence of the LS: grad deg = "
                          << degree_gradient << " , div deg = " << degree_div << std::endl;

                auto fe_data_gradient = Finite_Element<RealType, Mesh>(msh, degree_gradient, mip);
                auto method_transport_pb_grad = Transport_problem_method<Mesh, FiniteSpace>(fe_data_gradient, msh);
                auto fe_data_div = Finite_Element<RealType, Mesh>(msh, degree_div, mip);
// auto method_transport_pb_div = Transport_problem_method<Mesh, FiniteSpace>(fe_data_div, msh) ;

/************** ANALYTIC LEVEL SET FUNCTION  **************/
                bool circle = false, ellipse = true;
                bool flower = false;
                RealType radius_a, radius_b, radius;
                RealType x_centre = 0.0; // 0.5
                RealType y_centre = 0.0;
                if (circle) {
                    radius = 1.0 / 6.0;
                }
                if (ellipse) {

                    radius_a = radius_a_vec.at(R_i); // 1.0/3.0;
                    radius_b = radius_b_vec.at(R_i);
                    radius = sqrt(radius_a * radius_b);
                }

// -------------------------- ELLIPTIC LEVEL SET -----------------------------------
                std::cout << "Initial interface: ELLIPSE" << std::endl;
                auto level_set_function_anal = elliptic_level_set<RealType>(radius_a, radius_b, x_centre, y_centre);
                typedef elliptic_level_set <T> Fonction;
//std::cout<<"mu = "<<mu<<std::endl;
//std::cout<<"radius_a = "<<radius_a<<"radius_b = "<<radius_b<<std::endl;
                T curvature_anal = 1.0 / radius;

/**************  VELOCITY FIELD  INITIALISATION  **************/

                size_t degree_velocity = degree_FEM;
// **************** --------> STORING OF LAGRANGIAN NODES
                nodes_Lagrangian_cell_definition(msh, degree_velocity);


                auto fe_data_Lagrange = Finite_Element<RealType, Mesh>(msh, degree_velocity, mip);

                auto u_projected = velocity_high_order<Mesh, FiniteSpace, T>(fe_data_Lagrange, msh);

/************** LEVEL SET FUNCTION DISCRETISATION **************/
                auto level_set_function = Level_set_berstein<Mesh, Fonction, FiniteSpace, T>(fe_data,
                                                                                             level_set_function_anal,
                                                                                             msh, fe_data_gradient,
                                                                                             fe_data_div);
                typedef Level_set_berstein<Mesh, Fonction, FiniteSpace, T> Level_Set;
// ------------------  IF GRADIENT CONTINUOUS --------------
                level_set_function.gradient_continuous_setting(method_transport_pb_grad);
                auto curve = Interface_parametrisation_mesh1d(degree_curve);
                size_t degree_det_jac_curve = curve.degree_det;

/************** MESH INITIALISATION FOR ROUTINE  **************/


                auto crr_mesh = Current_Mesh<Mesh>(msh);
                crr_mesh.current_mesh = msh;
                Mesh msh_i = crr_mesh.current_mesh;      // MESH at t=t^n (FOR THE PROCESSING)
                offset_definition(msh_i);

/************** INITIAL DATA INITIALISATION (t = 0) **************/
                T dt = 0.;
                T initial_area = 0., initial_mass = 0.;
                T d_a = 0.;
                T perimeter_initial = 0.;
                T centre_mass_x_inital = 0., centre_mass_y_inital = 0.;
                T max_u_n_val_old = 1e+6, max_u_n_val_new = 1e+5;
                T check = 10.0;
                T tot_time = 0.;

/************** BOUNDARY CONDITIONS **************/
                std::cout << yellow << bold << "INLET BDRY: UP AND DOWN FOR DIRCIRCHLET_eps FP" << reset << std::endl;
                bool bdry_bottom = true, bdry_up = true;
                bool bdry_left = false, bdry_right = false;
                check_inlet(msh, fe_data, bdry_bottom, bdry_right, bdry_up, bdry_left, 1e-14);


//************ DO cutHHO MESH PROCESSING **************
                tc.tic();
                detect_node_position3(msh_i, level_set_function); // In cuthho_geom
                detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom

                if (agglomeration) {
                    detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom
                    refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
                    set_integration_mesh(msh_i, degree_curve);
                    detect_cell_agglo_set(msh_i, level_set_function);
                    make_neighbors_info_cartesian(msh_i);

                    make_agglomeration_no_double_points(msh_i, level_set_function, degree_det_jac_curve);
                    set_integration_mesh(msh_i, degree_curve);


                } else {
                    detect_cut_cells3(msh_i, level_set_function);
                    refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
                    set_integration_mesh(msh_i, degree_curve);
                }

                tc.toc();
                std::cout << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << '\n' << std::endl;
                std::cout << "Agglomerated amount cells: " << msh_i.cells.size() << std::endl;
                output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export

/************** UPDATING  LEVEL SET  AND VELOCITY  **************/
                level_set_function.gradient_continuous_setting(method_transport_pb_grad);
// -------------- LS_CELL: CHOICE OF DISC/CONT ----------------------------- \\
                // IF grad cont -> normal cont -> divergence disc
                auto ls_cell = LS_cell_high_order_grad_cont_div_disc<T, Mesh, Level_Set, Fonction, FiniteSpace>(
                        level_set_function, msh_i);
                ls_cell.radius = radius;
                u_projected.set_agglo_mesh(msh_i);

                timecounter tc_initial;
                tc_initial.tic();



/************** PLOTTINGS + GOAL QUANTITIES  **************/
                std::vector <T> max_val_u_n_time_para, L1_err_u_n_time_para, l1_err_u_n_time_para, linf_err_u_n_time_para, L1_err_u_n_time;
                std::vector <T> area_time, l1_err_u_n_time, linf_err_u_n_time, time_vec;
                std::vector <T> linf_der_time_interface, eccentricity_vec;
                std::vector <T> max_val_u_n_time, l1_err_curvature_time, linf_err_curvature_time;
                std::vector <T> circularity_time, circularity_time2, flux_interface_time, perimeter_time;
                std::vector <std::pair<T, T>> centre_mass_err_time, rise_velocity_time, min_max_vec;
                T circularity_ref = 0.0, perim_ref = 0.0, area_ref = 0.0;
                T l1_divergence_error = 0., l2_divergence_error = 0.;
                T linf_divergence_error = -10.;
                T L1_divergence_error = 0.;

                bool l2proj_para = false;

                bool l2proj = true;
                bool avg = false;
                bool disc = false;
                bool filter = false;

                Interface_parametrisation_mesh1d_global <Mesh> para_curve_cont(msh_i, degree_curve, degree_curvature);

// *********************** DERIVATIVE / NORMAL PARA *************************//
//------------- L2 cont curvature from parametric interface  r ---------- //
                para_curve_cont.make_L2_proj_para_derivative(msh_i);

//---------------------------- L2 global Normal from LS  ----------------------- //
                if (l2proj) {
                    if (!disc)
                        para_curve_cont.make_L2_proj_para_normal(msh_i, ls_cell);
                    else
                        para_curve_cont.make_L2_proj_para_normal_disc(msh_i, ls_cell);
                }
//---------------------------- Avg Normal from LS  ---------------------------- //
                if (avg) {
                    if (!disc)
                        para_curve_cont.make_avg_L2_local_proj_para_normal(msh_i, ls_cell);
                    else
                        para_curve_cont.make_avg_L2_local_proj_para_normal_disc(msh_i, ls_cell);
                }


// *********************** CURVATURE PARA *************************//

//------------- L2 cont curvature from parametric interface  r ---------- //
                if (l2proj_para)
                    para_curve_cont.make_L2_proj_para_curvature(msh_i);



//---------------------------- L2 global Curvature from LS  ----------------------- //
                if (l2proj) {
                    if (!disc)
                        para_curve_cont.make_L2_proj_para_curvature(msh_i, ls_cell);
                    else
                        para_curve_cont.make_L2_proj_para_curvature_disc(msh_i, ls_cell);
                }
//---------------------------- Avg Curvature from LS  ---------------------------- //
                if (avg) {
                    if (!disc)
                        para_curve_cont.make_avg_L2_local_proj_para_curvature(msh_i, ls_cell);
                    else
                        para_curve_cont.make_avg_L2_local_proj_para_curvature_disc(msh_i, ls_cell);

                }
                if (filter) {
                    std::cout << bold << yellow << "CURVATURE ANALYSIS PRE FILTER:" << '\n';
                    para_curvature_error(msh_i, para_curve_cont, curvature_anal);
                    para_curve_cont.make_smooth_filter_curvature();
                    std::cout << bold << yellow << "CURVATURE ANALYSIS POST FILTER:" << '\n';
                    para_curvature_error(msh_i, para_curve_cont, curvature_anal);

                }
                if (!filter) {
                    std::cout << bold << yellow << "CURVATURE ANALYSIS PARA:" << '\n';
                    para_curvature_error(msh_i, para_curve_cont, curvature_anal);
                }



// T eps_dirichlet_cond = 0.0 ;
                std::cout << "Dirichlet eps Cond = " << eps_dirichlet_cond << std::endl;

// -----------------------------------------------------------------------------------------
// ----------------- RESOLUTION OF THE STOKES PROBLEM (HHO) ------------------
// -----------------------------------------------------------------------------------------

                bool sym_grad = FALSE; //
                auto prm = params<T>();
                prm.kappa_1 = mu; // 100.0;
                prm.kappa_2 = 1.0;
// T gamma =  1.0;
// kappa_1 is negative domain (inner)
// kappa_2 is positive domain (outer)

                std::cout << "gamma = " << gamma << std::endl;
                std::cout << '\n' << bold << yellow << "HHO flow resolution." << reset << '\n' << std::endl;

// -------------- ESHELBY VERSION PARAMETRIC (CONT) -------------------
// LS normal and curvature
                auto test_case = make_test_case_eshelby_LS_eps_DIR_domSym(msh_i, ls_cell, para_curve_cont, prm,
                                                                          sym_grad, gamma, eps_dirichlet_cond, sizeBox);
// ------------------------ HHO METHOD FOR LEVEL SET  ---------------------------
                T alfa1 = prm.kappa_2 / (prm.kappa_1 + prm.kappa_2); // alfa1 is negative domain (inner)
                T alfa2 = prm.kappa_1 / (prm.kappa_1 + prm.kappa_2); // alfa2 is positive domain (outer)
                auto method = make_sym_gradrec_stokes_interface_method_alfai(msh_i, 1.0, 0.0, test_case, sym_grad,
                                                                             alfa1, alfa2);




//  ******************** - HHO RESOLUTION - ********************
// run_cuthho_interface_velocity_new_post_processingLS(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad , counter_tests, 1);
// new - 27 07 2023 - calculation tangential velocity at interface + separation of pos and neg component
                run_cuthho_interface_velocity_complete(msh_i, degree, method, test_case, ls_cell, u_projected, sym_grad,
                                                       counter_tests, 1, folder);

                counter_tests++;
                std::cout << "Test number = " << counter_tests << std::endl;
                std::cout << "Case: mu = " << mu << ", gamma = " << gamma << ", eps = " << eps_dirichlet_cond
                          << std::endl;
                std::cout << "Ra = " << radius_a << ", Rb = " << radius_b << std::endl;
            } // R_i loop


        } // test case gamma vs eps

    } // mu loop





    tc_tot.toc();
    std::cout << "Simulation machine time t = " << tc_tot << std::endl;

    return 0;
}

#endif







// Interface Stokes Problem: Two-fluid problem with FIXED POINT
// -------- Code paper: interface evolution under shear flow - perturbed flow - null flow

#if 0

int main(int argc, char **argv) {
    using RealType = double;
    RealType sizeBox = 0.5; // domain is a box (-sizeBox,sizeBox)^2
    std::string folder = "simu_weightCircle_"; // folder to save results
    int time_gap = 20; // every time_gap the results are saved


    size_t degree = 1; // HHO degre
    size_t int_refsteps = 0; // Number of refine interface points for each mesh cell
    size_t degree_FEM = 2; // LS degre
    size_t degree_curve = 2; // Degree of the interface approximation
    size_t degree_curvature = 1; // Degree of the interface curvature (after smoothing projection)
    bool dump_debug = false;
    bool solve_interface = true;
    bool solve_fictdom = false;
    bool agglomeration = true;

    bool high_order = false; // If false is low-order LS , if true  high-order LS
    bool entropic = false; // If false is low-order LS , if true  entropic LS
    bool entropic_mass_consistent = false;
    bool compressed = false;
    bool cut_off_active = false; // If false is smooth LS , if true  cut-off LS

    mesh_init_params <RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;

    mip.min_x = -sizeBox;
    mip.min_y = -sizeBox;
    mip.max_x = sizeBox;
    mip.max_y = sizeBox;


    size_t T_N = 10000; // max number of time step
    int ch;
    while ((ch = getopt(argc, argv, "k:q:M:N:r:T:l:p:ifDAdhesgc")) != -1) {
        switch (ch) {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;

            case 'T':
                T_N = atoi(optarg);
                break;

            case 'l':
                degree_curve = atoi(optarg);
                break;

            case 'p':
                degree_curvature = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;

            case 'h':
                high_order = true;
                break;

            case 'e':
                entropic = true;
                break;

            case 's':
                entropic_mass_consistent = true;
                break;

            case 'g':
                compressed = true;
                break;

            case 'c':
                cut_off_active = true;
                break;


            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;


    timecounter tc;

    timecounter tc_tot;
    tc_tot.tic();


/************** BUILD MESH **************/

    cuthho_poly_mesh <RealType> msh(mip);
    typedef cuthho_poly_mesh <RealType> Mesh;
    typedef RealType T;
    offset_definition(msh);

    std::cout << "Mesh size = " << mip.Nx << "x" << mip.Ny << std::endl;
    std::cout << "Number of refine interface points: r = " << int_refsteps << std::endl;


    folder = folder + std::to_string(mip.Nx) + "/";

    if (high_order == true)
        std::cout << bold << green << "Transport problem high order with limiting (no inlet). " << reset << std::endl;
    else if (entropic == true)
        std::cout << bold << green << "Transport problem entropic mass lumped (no inlet). " << reset << std::endl;
    else if (entropic_mass_consistent == true)
        std::cout << bold << green << "Transport problem entropic mass consistent (no inlet). " << reset << std::endl;
    else if (compressed == true)
        std::cout << bold << green << "Transport problem high order compressed with limiting (no inlet). " << reset
                  << std::endl;
    else
        std::cout << bold << green << "Transport problem low order (no inlet).  " << reset << std::endl;

/************** FINITE ELEMENT INITIALIZATION **************/
    auto fe_data = Finite_Element<RealType, Mesh>(msh, degree_FEM, mip);
    typedef Finite_Element <RealType, Mesh> FiniteSpace;
    std::cout << "Level Set (finite element approximation): Bernstein basis in space Q^{k_phi},  k_phi = " << degree_FEM
              << std::endl;

/************** TRANSPORT PROBLEM METHOD **************/
    auto method_transport_pb = Transport_problem_method<Mesh, FiniteSpace>(fe_data, msh);

    size_t degree_gradient = degree_FEM;
    size_t degree_div = degree_FEM;
    std::cout << "Finite element space for gradient and divergence of the LS: grad deg = " << degree_gradient
              << " , div deg = " << degree_div << std::endl;

    auto fe_data_gradient = Finite_Element<RealType, Mesh>(msh, degree_gradient, mip);
    auto method_transport_pb_grad = Transport_problem_method<Mesh, FiniteSpace>(fe_data_gradient, msh);
    auto fe_data_div = Finite_Element<RealType, Mesh>(msh, degree_div, mip);
    auto method_transport_pb_div = Transport_problem_method<Mesh, FiniteSpace>(fe_data_div, msh);


/************** ANALYTIC LEVEL SET FUNCTION  **************/
// This data have to be updated depending on the starting shape of the bubble
    constexpr bool circle = true;
    constexpr bool ellipse = false;
    constexpr bool flower = false;
    RealType radius_a, radius_b, radius;
    RealType x_centre = 0.0;
    RealType y_centre = 0.0;
    T oscillation = 0.0;

    using Fonction = std::conditional_t<circle, circle_level_set<T>,
            std::conditional_t<ellipse, elliptic_level_set<T>, flower_level_set<T>>>;

    std::unique_ptr<Fonction> level_set_function_anal_ptr;

    if constexpr (circle) {
        radius = 1.0 / 3.0;
        std::cout<<"Initial interface: CIRCLE"<<std::endl;
        level_set_function_anal_ptr = std::make_unique<Fonction>(radius, x_centre, y_centre );
    }
    if constexpr (ellipse) {
        radius_a = 1.0 / 12.0;
        radius_b = 1.0 / 24.0;
        radius = sqrt(radius_a * radius_b);
        level_set_function_anal_ptr = std::make_unique<Fonction>( radius_a, radius_b, x_centre, y_centre);
    }

    if constexpr (flower) {
        oscillation = 0.04;
        radius = 1.0 / 3.0;
        level_set_function_anal_ptr = std::make_unique<Fonction>(radius, x_centre, y_centre, 4, oscillation);
    }

    Fonction& level_set_function_anal = *level_set_function_anal_ptr;



/**************  VELOCITY FIELD  INITIALISATION  **************/


    size_t degree_velocity = degree_FEM; // std::max(degree + 1 , degree_FEM) ;


// **************** --------> STORING OF LAGRANGIAN NODES
    nodes_Lagrangian_cell_definition(msh, degree_velocity);


    auto fe_data_Lagrange = Finite_Element<RealType, Mesh>(msh, degree_velocity, mip);

    std::cout << "Velocity field: high order Lagrange basis: degree = " << degree_velocity << std::endl;
    auto u_projected = velocity_high_order<Mesh, FiniteSpace, T>(fe_data_Lagrange, msh);



/************** LEVEL SET FUNCTION DISCRETISATION **************/


//    if(high_order)
//        std::cout<<"--------------------> USING phi^H - HIGH ORDER TRANSPORT PB "<<std::endl;
//    else
//        std::cout<<"--------------------> USING phi^L - LOW ORDER TRANSPORT PB "<<std::endl;

//    std::cout<<"Level set: high order Berstein x interpolated."<<std::endl;

//-------->  NEW FORMULATION
    auto level_set_function = Level_set_berstein<Mesh, Fonction, FiniteSpace, T>(fe_data, level_set_function_anal, msh,
                                                                                 fe_data_gradient, fe_data_div);
    typedef Level_set_berstein <Mesh, Fonction, FiniteSpace, T> Level_Set;
// ------------------  IF GRADIENT CONTINUOUS --------------
    level_set_function.gradient_continuous_setting(method_transport_pb_grad);
//  ------------------ IF DIVERGENCE CONTINUOUS  ------------------
//    level_set_function.divergence_continuous_setting(method_transport_pb_div ) ;

//-------->  OLD FORMULATION GRAD CONT
//    auto level_set_function = Level_set_berstein_high_order_interpolation_grad_cont< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
//    level_set_function.gradient_continuous_setting() ;
//    typedef Level_set_berstein_high_order_interpolation_grad_cont< Mesh , Fonction , FiniteSpace , T > Level_Set;




    std::cout << "Parametric interface: degree_curve = " << degree_curve << std::endl;
    auto curve = Interface_parametrisation_mesh1d(degree_curve);
    size_t degree_det_jac_curve = curve.degree_det; // 2*degree_curve INUTILE PER ORA
// integration CUT CELL degree += 2*degree_curve
// integration INTERFACE degree += degree_curve-1

//auto level_set_function = Level_set_berstein_high_order_interpolation_grad_cont_fast< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
//auto level_set_function = Level_set_berstein_high_order_interpolation< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);





/************** MESH INITIALISATION FOR ROUTINE  **************/


    auto crr_mesh = Current_Mesh<Mesh>(msh);
    crr_mesh.current_mesh = msh;
    Mesh msh_i = crr_mesh.current_mesh;      // MESH at t=t^n (FOR THE PROCESSING)
    offset_definition(msh_i);


/************** INITIAL DATA INITIALISATION (t = 0) **************/
    T dt = 0.;
    T initial_area = 0., initial_mass = 0.;
    T d_a = 0.;
    T perimeter_initial = 0.;
    T centre_mass_x_inital = 0., centre_mass_y_inital = 0.;
    T max_u_n_val_old = 1e+6, max_u_n_val_new = 1e+5;
    T check = 10.0;
    T tot_time = 0.;

/************** BOUNDARY CONDITIONS **************/
    std::cout << yellow << bold << "INLET BDRY: UP AND DOWN FOR DIRCIRCHLET_eps FP" << reset << std::endl;
    bool bdry_bottom = false, bdry_up = false;
    bool bdry_left = false, bdry_right = false;
    check_inlet(msh, fe_data, bdry_bottom, bdry_right, bdry_up, bdry_left, 1e-14);


//************ DO cutHHO MESH PROCESSING **************
    tc.tic();
    detect_node_position3(msh_i, level_set_function); // In cuthho_geom
//detect_node_position3_parallel(msh_i, level_set_function); // In cuthho_geom
    detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom

    if (agglomeration) {
        detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom
//detect_cut_cells3_parallelized(msh_i, level_set_function); // In cuthho_geom
        refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
        set_integration_mesh(msh_i, degree_curve);
        detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
        make_neighbors_info_cartesian(msh_i); // Non serve modificarla
//refine_interface_pro3(msh_i, level_set_function, int_refsteps);

        make_agglomeration_no_double_points(msh_i, level_set_function, degree_det_jac_curve);
        set_integration_mesh(msh_i, degree_curve); // TOLTO PER IL MOMENTO SENNO RADDOPPIO
//        make_agglomeration(msh_i, level_set_function); // Non serve modificarla

    } else {
        detect_cut_cells3(msh_i, level_set_function);
//refine_interface_pro3(msh_i, level_set_function, int_refsteps);
        refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
    }

    tc.toc();
    std::cout << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << '\n' << std::endl;

    if (dump_debug) {
        dump_mesh(msh_i);
        output_mesh_info(msh_i, level_set_function);
    }



// IN cuthho_export..Points/Nodes don't change-> it's fast
    output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export

/************** UPDATING  LEVEL SET  AND VELOCITY  **************/
//    level_set_function.gradient_continuous_setting() ;
//    // IF GRADIENT CONTINUOUS
    level_set_function.gradient_continuous_setting(method_transport_pb_grad);
//    // IF DIVERGENCE CONTINUOUS
//    level_set_function.divergence_continuous_setting(method_transport_pb_div) ;


// --------------------- LS_CELL: CHOICE OF DISC/CONT ------------------------------- \\

// IF grad cont -> normal cont -> (divergence disc) -> divergence cont
//auto ls_cell = LS_cell_high_order_curvature_cont< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);

// IF grad cont -> normal cont -> divergence disc
    auto ls_cell = LS_cell_high_order_grad_cont_div_disc<T, Mesh, Level_Set, Fonction, FiniteSpace>(level_set_function,
                                                                                                    msh_i);

// IF grad disc -> normal disc -> divergence disc
//    auto ls_cell = LS_cell_high_order_grad_disc_div_disc< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);
// IF grad disc -> normal disc -> divergence disc -> normal and grad cont
//    auto ls_cell = LS_cell_high_order_div_disc_grad_n_cont< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);

//-------------------------- OLD CASE LS_CELL --------------------------
//    auto ls_cell = LS_cell_high_order_grad_cont< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i );



    ls_cell.radius = radius;
    u_projected.set_agglo_mesh(msh_i);

    timecounter tc_initial;
    tc_initial.tic();



/************** PLOTTINGS + GOAL QUANTITIES  **************/
    std::vector <T> max_val_u_n_time_para, L1_err_u_n_time_para, l1_err_u_n_time_para, linf_err_u_n_time_para, L1_err_u_n_time;
    std::vector <T> area_time, l1_err_u_n_time, linf_err_u_n_time, time_vec;
    std::vector <T> linf_der_time_interface, eccentricity_vec;
    std::vector <T> max_val_u_n_time, l1_err_curvature_time, linf_err_curvature_time;
    std::vector <T> circularity_time, circularity_time2, flux_interface_time, perimeter_time;
    std::vector <std::pair<T, T>> centre_mass_err_time, rise_velocity_time, min_max_vec;
    T circularity_ref = 0.0, perim_ref = 0.0, area_ref = 0.0;
    T l1_divergence_error = 0., l2_divergence_error = 0.;
    T linf_divergence_error = -10.;
    T L1_divergence_error = 0.;

    check_goal_quantities(msh_i, ls_cell, perimeter_initial, d_a, initial_area, centre_mass_x_inital,
                          centre_mass_y_inital, degree_FEM, initial_mass, flower, l1_divergence_error,
                          l2_divergence_error, linf_divergence_error, radius, L1_divergence_error, ellipse,
                          degree_curve, int_refsteps);



//------------------------ CHECK REFERENCE QUANTITIES ---------------------------//
    reference_quantities_computation(perim_ref, area_ref, circularity_ref, radius, x_centre, y_centre, fe_data, msh,
                                     degree_curve, perimeter_initial, initial_area, int_refsteps, degree_det_jac_curve);

//    plot_curvature_normal_vs_curv_abscisse(msh_i, ls_cell, degree_curve,int_refsteps , 0 );

    plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(msh_i, ls_cell, degree_curve, int_refsteps, 0, degree_curvature);


    tc_initial.toc();
//    std::cout << "Time Machine for checking INITAL GOAL QUANTITIES: " << tc_initial << " seconds" << std::endl;

    circularity_time.push_back(M_PI * d_a / perimeter_initial);
    circularity_time2.push_back(4.0 * M_PI * initial_area / (perimeter_initial * perimeter_initial));
    perimeter_time.push_back(perimeter_initial);
    centre_mass_err_time.push_back(
            std::make_pair(centre_mass_x_inital / initial_area, centre_mass_y_inital / initial_area));
    time_vec.push_back(0);
    area_time.push_back(initial_area);

    min_max_vec.push_back(std::make_pair(level_set_function.phi_min, level_set_function.phi_max));

    l1_err_curvature_time.push_back(l1_divergence_error);
    linf_err_curvature_time.push_back(linf_divergence_error);

    T dt_M;
    T R_phi = radius;

    size_t tot_amount_transport_routine = 0;


    bool l2proj_para = false;

    bool l2proj = true;
    bool avg = false;
    bool disc = false;
    bool filter = false;

    Interface_parametrisation_mesh1d_global <Mesh> para_curve_cont(msh_i, degree_curve, degree_curvature);

// *********************** DERIVATIVE / NORMAL PARA *************************//
//------------- L2 cont curvature from parametric interface  r ---------- //
    para_curve_cont.make_L2_proj_para_derivative(msh_i);

//---------------------------- L2 global Normal from LS  ----------------------- //
    if (l2proj) {
        if (!disc)
            para_curve_cont.make_L2_proj_para_normal(msh_i, ls_cell);
        else
            para_curve_cont.make_L2_proj_para_normal_disc(msh_i, ls_cell);
    }
//---------------------------- Avg Normal from LS  ---------------------------- //
    if (avg) {
        if (!disc)
            para_curve_cont.make_avg_L2_local_proj_para_normal(msh_i, ls_cell);
        else
            para_curve_cont.make_avg_L2_local_proj_para_normal_disc(msh_i, ls_cell);
    }


// *********************** CURVATURE PARA *************************//

//------------- L2 cont curvature from parametric interface  r ---------- //
    if (l2proj_para)
        para_curve_cont.make_L2_proj_para_curvature(msh_i);



//---------------------------- L2 global Curvature from LS  ----------------------- //
    if (l2proj) {
        if (!disc)
            para_curve_cont.make_L2_proj_para_curvature(msh_i, ls_cell);
        else
            para_curve_cont.make_L2_proj_para_curvature_disc(msh_i, ls_cell);
    }
//---------------------------- Avg Curvature from LS  ---------------------------- //
    if (avg) {
        if (!disc)
            para_curve_cont.make_avg_L2_local_proj_para_curvature(msh_i, ls_cell);
        else
            para_curve_cont.make_avg_L2_local_proj_para_curvature_disc(msh_i, ls_cell);

    }
    if (filter)
        para_curve_cont.make_smooth_filter_curvature();

// ******** TO FASTER THE SIMULATION, ERASED THE PLOTTINGS
    plotting_para_curvature_cont_time_fast(msh_i, para_curve_cont, degree_curve, degree_FEM, radius, 0, int_refsteps);

    T final_time = 8.0;

    T eps_dirichlet_cond = 0.0; //0.26; 0.52 ; // 0.01 -->  0.1

    for (size_t time_step = 0; time_step <= T_N; time_step++) {
        timecounter tc_iteration;
        tc_iteration.tic();
        std::cout << '\n' << bold << yellow << "Starting iteration numero  = " << time_step << " --> time t = "
                  << tot_time << reset << std::endl;
        std::cout << "Dirichlet eps Cond = " << eps_dirichlet_cond << std::endl;

// -----------------------------------------------------------------------------------------
// ----------------- RESOLUTION OF THE STOKES PROBLEM (HHO) ------------------
// -----------------------------------------------------------------------------------------

        bool sym_grad = TRUE;
        auto prm = params<T>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        T gamma = 1.0; // 0.05;


        savingVelocityLevelSet(level_set_function, u_projected);


        std::cout << '\n' << bold << yellow << "HHO flow resolution." << reset << '\n' << std::endl;


// ------------------ OLD VERSIONS ------------------
//auto test_case = make_test_case_eshelby(msh_i, ls_cell,  prm , sym_grad);
// Non serve modificare Gamma = 1/2
//auto test_case = make_test_case_eshelby_2(msh_i, ls_cell,  prm , sym_grad );
//auto test_case = make_test_case_eshelby_analytic(msh_i, ls_cell,  prm , sym_grad , radius);
// ------------- OLD GUILLAUME VERSIONS --------------
// auto test_case = make_test_case_stokes_1(msh, level_set_function);
// auto test_case = make_test_case_stokes_2(msh, ls_cell); //level_set_function);

// ----------------- ESHELBY VERSION - CORRECT BUT PRESSURE ------------------
//auto test_case_prova = make_test_case_eshelby_2_prova(msh_i, ls_cell,  prm , sym_grad );
// PRESSURE SIGN NOT CORRECT
// ---------------------- ESHELBY VERSION LEVEL SET - CORRECT ------------------------
//        auto test_case = make_test_case_eshelby_correct(msh_i, ls_cell,  prm , sym_grad,gamma);
// PRESSURE SIGN NOT CORRECT
// -------------------- ESHELBY VERSION PARAMETRIC (DISC) - CORRECT -------------------
//        auto test_case = make_test_case_eshelby_correct_parametric(msh_i, ls_cell,  prm , sym_grad,gamma);
// PRESSURE SIGN NOT CORRECT




// -------------------- ESHELBY VERSION PARAMETRIC (CONT) - CORRECT -------------------
// domain  (0,1)^2 - null flow
//        auto test_case_prova = make_test_case_eshelby_correct_parametric_cont( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma );

// domain  (-a,a)^2 - shear flow
        auto test_case_prova = make_test_case_eshelby_parametric_cont_eps_DIR_domSym(msh_i, ls_cell, para_curve_cont,
                                                                                     prm, sym_grad, gamma,
                                                                                     eps_dirichlet_cond); // sizeBox

// domain  (0,1)^2 - shear flow

//        auto test_case_prova = make_test_case_eshelby_correct_parametric_cont_DIRICHLET_eps( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond);

// domain  (-a,a)^2 - new test case perturbated
// T perturbation = 0.5;
// auto test_case_prova = make_test_case_eshelby_parametric_cont_eps_perturbated_DIR_domSym( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond,perturbation );


// domain  (0,1)^2 - new test case perturbated
//     auto test_case_prova = make_test_case_shear_flow_perturbated( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond,perturbation );
//        auto test_case_prova = make_test_case_shear_y( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond,perturbation );
// New test case TGV - fixed point
//        auto test_case_prova = make_test_case_TGV_FPscheme( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond );



//        auto test_case_prova = make_test_case_eshelby_correct_parametric_cont_TGV_source( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma );
// ------------------------ HHO METHOD FOR LEVEL SET  ---------------------------
//        auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case, sym_grad);
// -------------------- HHO METHOD FOR DISC PARAMETRIC INTERFACE  -----------------------
//         auto method = make_sym_gradrec_stokes_interface_method_perturbationref_pts(msh_i, 1.0, 0.0, test_case, sym_grad);


// -------------------- HHO METHOD FOR CONT PARAMETRIC INTERFACE  -----------------------
        auto method_prova = make_sym_gradrec_stokes_interface_method_ref_pts_cont(msh_i, 1.0, 0.0, test_case_prova,
                                                                                  sym_grad);




//  ******************** - HHO RESOLUTION - ********************
        if (solve_interface) {
// ----------------- HHO RESOLUTION OLD CASE  --------------------------
//            TI = run_cuthho_interface_numerical_ls(msh_i, degree, method, test_case_prova , ls_cell ,  normal_analysis );
//            run_cuthho_interface_velocity_parallel(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad );

// ----------------- HHO RESOLUTION LS / PARAMETRIC DISC  ---------------------
//            run_cuthho_interface_velocity_prova(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad , time_step); // THE ONE CORRECT THAT I'M USING NOW

// ----------------- HHO RESOLUTION PARAMETRIC CONT  --------------------------
//             run_cuthho_interface_velocity_new(msh_i, degree, method_prova,test_case_prova, ls_cell , u_projected ,sym_grad , time_step); // THE ONE CORRECT THAT I'M USING NOW
// run_cuthho_interface_velocity_fast(msh_i, degree, method_prova,test_case_prova, ls_cell , u_projected ,sym_grad , time_step); // CORRECT BUT DOEST NOT COMPUTE ERRORS


            run_cuthho_interface_velocity_complete(msh_i, degree, method_prova, test_case_prova, ls_cell, u_projected,
                                                   sym_grad, time_step, time_gap, folder);


        }
//        testing_velocity_field(msh , u_projected) ;
/************************************ FEM -  PRE-PROCESSING ******************************************/
// ----------------- PROJECTION OF THE VELOCITY FIELD ------------------
        if (0)
            std::cout << bold << green
                      << "CASE WITH VELOCITY DISCONTINUOUS: ho solo sol_HHO, sol_FEM non salvato, va cambiato il transport pb!!!"
                      << reset << std::endl;

        if (0) // method used in the paper - algebraic average
        {
            std::cout << '\n' << "Smoothing operator from velocity HHO to FE (continuity imposed): geometrical average."
                      << std::endl;
            u_projected.smooth_converting_into_FE_formulation(u_projected.sol_HHO);
        }
        if (1) // method used for the reviewers - weigth average
        {
            std::cout << '\n' << "Smoothing operator from velocity HHO to FE (continuity imposed): geometrical average."
                      << std::endl;
            u_projected.weight_converting_into_FE_formulation(u_projected.sol_HHO);
        }
        if (0) {
            std::cout << '\n' << "------------------>>>> NOTICE: NON SMOOTH OPERATOR FROM HHO TO FEM." << std::endl;
            u_projected.converting_into_FE_formulation(u_projected.sol_HHO);
        }
        if (0) {
            std::cout << '\n' << "------------------>>>>NOTICE: L^2 PROJECTION FROM HHO TO FEM." << std::endl;
            u_projected.L2_proj_into_FE_formulation(level_set_function, msh, method_transport_pb);
        }


//testing_velocity_field(msh , u_projected) ;
//auto u_prova = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
//u_prova.sol_HHO = u_projected.sol_HHO ;
//u_prova.L2_proj_into_FE_formulation( level_set_function , msh );
//testing_velocity_field_L2projected(msh , u_prova) ;

        T rise_vel0 = 0.0, rise_vel1 = 0.0;
        T flux_interface = 0.0;

        if (time_step == 0) {
            T max_u_n_val_initial = 0.0;
            T max_u_n_val_abs_initial = 0.0;
            T l1_normal_interface_status_initial = 0.;

            T L1_normal_interface_status_initial = 0.;

            size_t counter_interface_pts_initial = 0;
            for (auto &cl: msh_i.cells) {
                if (cl.user_data.location == element_location::ON_INTERFACE) {
                    ls_cell.cell_assignment(cl);
                    u_projected.cell_assignment(cl);

                    auto qps = integrate_interface(msh_i, cl, degree_FEM + degree_velocity,
                                                   element_location::ON_INTERFACE);
                    for (auto &qp: qps) {
                        auto u_pt = u_projected(qp.first);

                        auto ls_n_pt = ls_cell.normal(qp.first);
                        T u_n_val = u_pt.first * ls_n_pt(0) + u_pt.second * ls_n_pt(1);
                        L1_normal_interface_status_initial += qp.second * std::abs(u_n_val);
                        max_u_n_val_abs_initial = std::max(max_u_n_val_abs_initial, std::abs(u_n_val));
                        if (std::abs(u_n_val) == max_u_n_val_abs_initial)
                            max_u_n_val_initial = u_n_val;

                        l1_normal_interface_status_initial += std::abs(u_n_val);
                        counter_interface_pts_initial++;


                    }

                }

            }
            l1_normal_interface_status_initial /= counter_interface_pts_initial;
            l1_err_u_n_time.push_back(l1_normal_interface_status_initial);
            linf_err_u_n_time.push_back(max_u_n_val_abs_initial);
//            linf_der_time_interface.push_back(0) ;
            max_val_u_n_time.push_back(max_u_n_val_initial);
            L1_err_u_n_time.push_back(L1_normal_interface_status_initial);

            std::cout << "------> The l1 error of u*n along the INTERFACE at INITIAL TIME is "
                      << l1_normal_interface_status_initial << std::endl;

            std::cout << bold << green << "------> The linf error of u*n along the INTERFACE at INITIAL TIME is "
                      << max_u_n_val_abs_initial << reset << std::endl;

            std::cout << "------> The L1 error of u*n along the INTERFACE at INITIAL TIME is "
                      << L1_normal_interface_status_initial << std::endl;


            size_t degree_curvature = para_curve_cont.dd_degree;
            size_t degree_jacobian = para_curve_cont.degree_det;

            T L1_normal_interface_para = 0.0;
            T linf_u_n_para = 0.0;
            T max_u_n_val_para = 0.0;
            T l1_normal_interface_para = 0.0;
            size_t counter_interface_pts_para = 0;


            T area_para = 0.0;

            for (auto &cl: msh_i.cells) {

                if ((location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE) ||
                    (location(msh_i, cl) == element_location::ON_INTERFACE)) {
                    u_projected.cell_assignment(cl);
                    T partial_area = measure(msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                    area_para += partial_area;

                    size_t max_deg = std::max(degree_velocity, degree_FEM);
                    auto qps_fin = integrate(msh_i, cl, max_deg, element_location::IN_NEGATIVE_SIDE);


                    for (auto &qp: qps_fin) {
                        auto u_pt = u_projected(qp.first);
                        rise_vel0 += qp.second * u_pt.first;
                        rise_vel1 += qp.second * u_pt.second;
                    }

                }


                if (cl.user_data.location == element_location::ON_INTERFACE) {
                    u_projected.cell_assignment(cl);
                    auto global_cells_i = para_curve_cont.get_global_cells_interface(msh_i, cl);
                    auto integration_msh = cl.user_data.integration_msh;
//                    auto degree_int = degree_curvature + degree_jacobian ;


                    auto qps_un = edge_quadrature<T>(degree_jacobian + degree_curvature + degree_velocity);

                    for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                        auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                        size_t global_cl_i = global_cells_i[i_cell];

                        for (auto &qp: qps_un) {
                            auto t = 0.5 * qp.first.x() + 0.5;

                            T jacobian = para_curve_cont.jacobian_cont(t, global_cl_i);
                            auto w = 0.5 * qp.second * jacobian;
                            auto p = para_curve_cont(t, global_cl_i);
                            auto pt = typename Mesh::point_type(p(0), p(1));
                            auto u_pt = u_projected(pt);
                            auto curve_n_pt = para_curve_cont.normal_cont(t, global_cl_i);
                            T flux = u_pt.first * curve_n_pt(0) + u_pt.second * curve_n_pt(1);
                            flux_interface += w * flux;

                            L1_normal_interface_para += w * std::abs(flux);
                            linf_u_n_para = std::max(linf_u_n_para, std::abs(flux));
                            if (std::abs(flux) == linf_u_n_para)
                                max_u_n_val_para = flux;

                            l1_normal_interface_para += std::abs(flux);
                            counter_interface_pts_para++;

                        }
                    }
                }
            }
            l1_normal_interface_para /= counter_interface_pts_para;
            l1_err_u_n_time_para.push_back(l1_normal_interface_para);
            linf_err_u_n_time_para.push_back(linf_u_n_para);
            L1_err_u_n_time_para.push_back(L1_normal_interface_para);
            max_val_u_n_time_para.push_back(max_u_n_val_para);
            flux_interface_time.push_back(flux_interface);
            rise_velocity_time.push_back(std::make_pair(rise_vel0 / area_para, rise_vel1 / area_para));

            std::cout << "------> The l1 error of PARA_CONT u*n along the INTERFACE at INITIAL TIME is "
                      << l1_normal_interface_para << std::endl;

            std::cout << bold << yellow
                      << "------> The linf error of PARA_CONT u*n along the INTERFACE at INITIAL TIME is "
                      << linf_u_n_para << reset << std::endl;

            std::cout << "------> The L1 error of PARA_CONT u*n along the INTERFACE at INITIAL TIME is "
                      << L1_normal_interface_para << std::endl;
        }
// -----------------------------------------------------------------------------------------
// ----------------- TIME EVOLUTION (u^n,phi^n) (FEM) ------------------
// -----------------------------------------------------------------------------------------


//------------- MACRO TIME STEP dt -------------
//(before checking if it is too big, I need to find the new interface)
        T dt_M_max = 0.1; // 0.1 ; //5e-2 ; //5e-2; // phi_h 1e-3 e dt_m^MAX = 1e-4 not enought (oscillation )

//if(tot_time < 0.75)
        T eps = 0.05; //1.0 ; // 0.05 ; // factor to be inside CFL stability zone IT WAS 0.4
        T dt_one_cell = time_step_CFL_new(u_projected, mip, eps);

// Stokes time step maximum
        T c2 = 2.0; // 2.0; // it was 1
        T eta = std::min(prm.kappa_1, prm.kappa_2);
        T dt_STK = time_step_STK(eta, gamma, mip, c2);
        dt_M = std::min(dt_one_cell, dt_STK);

// dt_M =  dt_M * 0.1 ; // add 23/09/2021

// ------> ADAPTIVE SCHEME
//        dt_M = dt_one_cell * 0.1 ;
//        dt_M = floorf(dt_M * 1000) / 1000;

//        T dt_M_max ;
//        if( std::max(mip.hx() , mip.hy() ) <= 16 )
//            dt_M_max = 0.1;
//        else if( std::max(mip.hx() , mip.hy() ) == 32 )
//            dt_M_max = 5e-2;
//        else
//            dt_M_max = 5e-2;



        dt = std::min(dt_M_max, dt_M); //5e-2 for mesh 32x32, 0.1 for  mesh 16x16, 8x8

        if (final_time < (dt + tot_time)) {
            std::cout << "Last step, dt_M changed to arrive at T." << std::endl;
            dt = final_time - tot_time;

        }
        int N_sub = 10; //N_sub = 20 for BOH, N_sub = 10 for  mesh 16x16, 8x8, 32x32
// ------> FIXED  SCHEME
//        dt_M =  8*1e-3; // FOR LS low phi_L order-> 8*1e-3; // FOR LS high order phi_H ->  2*1e-3;
//        dt = std::min(dt_one_cell , dt_M);
//        T N_sub = 20 ;  // N_sub = 20 for mesh 32x32, N_sub = 10 for  mesh 16x16,  N_sub = 40 for phi_H
//std::cout<<"dt1 is "<<dt1<<std::endl;


        std::cout << '\n' << "Macro time step dt_M = " << dt << ", dt_h = " << dt_one_cell << ", dt_STK = " << dt_STK
                  << std::endl;

// I can create a sub-time. I solve several time the FEM problem, given a Stokes field. The amount of time is s.t. at maximum there is a displacement of a cell of the interface and no more than a maximum T
        T sub_time = 0.;
        T sub_dt = dt / N_sub; //std::min(4*1e-4 , dt ) ;

// ------> ADAPTIVE SCHEME
// It is phi_L: > 1e-3 for 16x16 and 8x8  ; > 5e-4 for 32x32
// It is phi_H: > 1e-4 for 16x16 and 8x8  ; > 5e-4 for 32x32
        if (sub_dt > 1e-3) {
            sub_dt = 1e-3;
            N_sub = floor(dt / sub_dt); // N_sub varies between 10 and 100 depending on sub_dt

        }


        auto level_set_old = Level_set_berstein<Mesh, Fonction, FiniteSpace, T>(level_set_function);
        auto ls_old = LS_cell_high_order_grad_cont_div_disc<T, Mesh, Level_Set, Fonction, FiniteSpace>(level_set_old,
                                                                                                       msh_i);


        std::cout << "-----> Implemented sub time dt_m = " << sub_dt << std::endl;

// ------------- NEW IMPLEMENTATION WITH FAST LEVEL SET ---------------------
        std::cout << yellow << bold << "----------- STARTING TRANSPORT PROBLEM SUB-ROUTINE: tilde{N} = " << N_sub
                  << "-----------" << reset << std::endl;
        timecounter tc_case_tr;
        tc_case_tr.tic();
        while (sub_time < sub_dt * N_sub) {
// std::pair<T,T> CFL_numb;
            if (high_order) {
//           CFL_numb = run_transport_high_order_FTC_M_lumped( level_set_function.msh , fe_data , level_set_function , u_projected , method_transport_pb , sub_dt , false );

                run_transport_high_order_FTC_M_consistent(level_set_function.msh, fe_data, level_set_function,
                                                          u_projected, method_transport_pb, sub_dt, false);

//            CFL_numb = run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D_NEW_DIRICHLET_COND_NEW_LS_M_CONS_LIMITED( level_set_function.msh , fe_data , level_set_function , u_projected , method_transport_pb , sub_dt , false ); // CORRECT 25/01/21

            } else if (entropic) {
                run_transport_entropic_M_lumped_Ri_correct(level_set_function.msh, fe_data, level_set_function,
                                                           u_projected, method_transport_pb, sub_dt, false);
//             CFL_numb = run_transport_entropic_M_lumped ( level_set_function.msh , fe_data , level_set_function , u_projected , method_transport_pb , sub_dt , false );
//            CFL_numb = run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D_NEW_DIRICHLET_COND_NEW_LS_M_LUMPED_NO_LIMITING ( level_set_function.msh , fe_data , level_set_function , u_projected , method_transport_pb , sub_dt , false ); // CORRECT 25/01/21
            } else {
//                if(degree_velocity == degree_FEM) // IT IS FASTER
                run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST_NEW_DIRICHLET_COND_NEW_LS(level_set_function.msh, fe_data,
                                                                                   level_set_function, u_projected,
                                                                                   method_transport_pb,
                                                                                   sub_dt); // CORRECT 25/01/21
//                else // IT WORKS BUT IT'S MUCH SLOWER
//                    run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST_NEW_DIRICHLET_COND_NEW_LS( level_set_function.msh , fe_data , level_set_function , u_projected , method_transport_pb , sub_dt  , fe_data_Lagrange);

            }


            sub_time += sub_dt;

        }
        tot_time += sub_time;

        tc_case_tr.toc();
        std::cout << yellow << bold << "----------- THE END OF TRANSPORT PROBLEM SUB-ROUTINE: machine time t = "
                  << tc_case_tr << " s -----------" << reset << std::endl;
        tot_amount_transport_routine += N_sub;

/**************************************************   POST-PROCESSING **************************************************/


// Updating continuous normal function
//            level_set_function.gradient_continuous_setting() ;
// IF GRADIENT CONTINUOUS
        level_set_function.gradient_continuous_setting(method_transport_pb_grad);
//         // IF DIVERGENCE CONTINUOUS
//         level_set_function.divergence_continuous_setting(method_transport_pb_div) ;

// Updating mesh data to check out differences in mass and areas
        crr_mesh.current_mesh = msh;
        msh_i = crr_mesh.current_mesh;
        offset_definition(msh_i);

        tc.tic();
        detect_node_position3(msh_i, level_set_function); // In cuthho_geom
        detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom


        if (agglomeration) {
            detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom

            refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
            set_integration_mesh(msh_i, degree_curve);

            detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
            make_neighbors_info_cartesian(msh_i); // Non serve modificarla
//refine_interface_angle(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
//refine_interface_pro3(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
            make_agglomeration_no_double_points(msh_i, level_set_function, degree_det_jac_curve);
// make_agglomeration(msh_i, level_set_function); // Non serve modificarla
            set_integration_mesh(msh_i, degree_curve);
        } else {
            move_nodes(msh_i, level_set_function);
            detect_cut_cells3(msh_i, level_set_function);
            refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
//refine_interface_pro3(msh_i, level_set_function, int_refsteps);
        }


        tc.toc();
        std::cout << '\n' << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << std::endl;

        if (dump_debug) {
            dump_mesh(msh_i);
            output_mesh_info(msh_i, level_set_function);
        }

// TOLTO TO BE FASTER
        if (time_step % time_gap == 0) {
            output_mesh_info2_time(msh_i, level_set_function, tot_time, time_step, folder);
            testing_level_set_time(msh, level_set_function, tot_time, time_step, folder);
            plot_u_n_interface(msh_i, ls_cell, u_projected, time_step, folder);
            testing_velocity_field(msh, u_projected, time_step, folder);
        }
//        output_mesh_info2_time_fixed_mesh(msh_i, level_set_function,tot_time,time_step);
        testing_level_set(msh, level_set_function, folder);
// Updating level set
        ls_cell.level_set = level_set_function;
        ls_cell.agglo_msh = msh_i;
        u_projected.set_agglo_mesh(msh_i);


        T max_u_n_val = 0.0;
        T max_u_n_val_abs = 0.0;

        T diff_in_time_interface = 0.0;
        T l1_normal_interface_status = 0.;

        T L1_normal_interface_status = 0.;

        size_t counter_interface_pts = 0;
        for (auto &cl: msh_i.cells) {
            if (cl.user_data.location == element_location::ON_INTERFACE) {
                ls_cell.cell_assignment(cl);
                ls_old.cell_assignment(cl);
                u_projected.cell_assignment(cl);

                auto qps = integrate_interface(msh_i, cl, degree_FEM + degree_velocity, element_location::ON_INTERFACE);
                for (auto &qp: qps) {
                    auto u_pt = u_projected(qp.first);
                    auto ls_n_pt = ls_cell.normal(qp.first);
                    T u_n_val = u_pt.first * ls_n_pt(0) + u_pt.second * ls_n_pt(1);
                    L1_normal_interface_status += qp.second * std::abs(u_n_val);
                    max_u_n_val_abs = std::max(max_u_n_val_abs, std::abs(u_n_val));

                    diff_in_time_interface = std::max(diff_in_time_interface,
                                                      (std::abs(ls_cell(qp.first) - ls_old(qp.first))) / sub_time);
                    if (std::abs(u_n_val) == max_u_n_val_abs)
                        max_u_n_val = u_n_val;

                    l1_normal_interface_status += std::abs(u_n_val);
                    counter_interface_pts++;

                }


            }
        }

        if (time_step == 0)
            max_u_n_val_new = max_u_n_val;

        if (time_step > 0) {
            max_u_n_val_old = max_u_n_val_new;
            max_u_n_val_new = max_u_n_val;
            std::cout << bold << yellow << "l^{inf} u*n(t^n) = " << max_u_n_val_old << " , l^{inf} u*n(t^{n+1}) = "
                      << max_u_n_val_new << reset << std::endl;
        }

        std::cout << "Number of interface points is " << counter_interface_pts << std::endl;

        l1_normal_interface_status /= counter_interface_pts;

        std::cout << "-----------------------> The l1 error of u*n over the INTERFACE, at time t = " << tot_time
                  << " is " << l1_normal_interface_status << reset << std::endl;

        std::cout << bold << green << "-----------------------> The linf error of u*n over the INTERFACE, at time t = "
                  << tot_time << " is " << max_u_n_val_abs << reset << std::endl;

        std::cout << "-----------------------> The L1 error of u*n over the INTERFACE, at time t = " << tot_time
                  << " is " << L1_normal_interface_status << reset << std::endl;

        std::cout << bold << green << "---> The linf error of (phi^{n+1}-phi{n})/dt over the INTERFACE, at time t = "
                  << tot_time << " is " << diff_in_time_interface << reset << std::endl;








// ----------------- CHECKING GOAL QUANTITIES FOR t = t^{n+1} ------------------



        check = l1_normal_interface_status;
/// DA AGGIUNGERE UNA VOLTA SISTEMATO IL CODICE

//if(check < 1e-8 )
//{
//    std::cout<<" check = "<<check<<" , STOP!"<<std::endl;
//    return 0;
//}





        T mass_fin = 0., area_fin = 0.;
        T centre_mass_x = 0., centre_mass_y = 0.;
        T l1_divergence_error_fin = 0., l2_divergence_error_fin = 0.;
        T linf_divergence_error_fin = 0.;
        T perimeter = 0.;
        T L1_divergence_error_fin = 0.;


        size_t counter_interface_pts_fin = 0.0;
//
//------------ Updating Parametric interface
        para_curve_cont.updating_parametric_interface(msh_i, ls_cell, l2proj, avg, l2proj_para, disc);

        if (filter) {
            plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(msh_i, int_refsteps, para_curve_cont, time_step);
            para_curve_cont.make_smooth_filter_curvature();
        }

//        para_curve_cont.plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(msh, n_int);

        T L1_normal_interface_para = 0.0;
        T linf_u_n_para = 0.0;
        T max_u_n_val_para = 0.0;
        T l1_normal_interface_para = 0.0;
        size_t counter_interface_pts_para = 0;
        T max_curvature = 0;
        T min_curvature = 1e+6;
        plot_curvature_normal_vs_curv_abscisse(msh_i, ls_cell, degree_curve, int_refsteps, 1, folder);
        plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(msh_i, degree_curve, int_refsteps, 1, degree_curvature,
                                                          para_curve_cont, folder);

//        check_goal_quantities_final_para( msh_i ,ls_cell, para_curve_tmp , u_projected, perimeter, d_a,  area_fin, centre_mass_x ,   centre_mass_y , degree_FEM , mass_fin , degree_velocity , l1_divergence_error_fin , l2_divergence_error_fin , linf_divergence_error_fin , radius , L1_divergence_error_fin ,  time_step ,rise_vel0 , rise_vel1 ,flux_interface, counter_interface_pts_fin,degree_curve,int_refsteps, L1_normal_interface_para,linf_u_n_para,max_u_n_val_para,l1_normal_interface_para,counter_interface_pts_para);

        check_goal_quantities_final_para_no_plot(msh_i, ls_cell, para_curve_cont, u_projected, perimeter, d_a, area_fin,
                                                 centre_mass_x, centre_mass_y, degree_FEM, mass_fin, degree_velocity,
                                                 l1_divergence_error_fin, l2_divergence_error_fin,
                                                 linf_divergence_error_fin, radius, L1_divergence_error_fin, 1,
                                                 rise_vel0, rise_vel1, flux_interface, counter_interface_pts_fin,
                                                 degree_curve, int_refsteps, L1_normal_interface_para, linf_u_n_para,
                                                 max_u_n_val_para, l1_normal_interface_para, counter_interface_pts_para,
                                                 max_curvature, min_curvature);

        std::cout << "PARA-FLUX at the INTERFACE, at time " << tot_time << " is " << flux_interface << std::endl;
//        std::cout<<"PARA-L1(un) at the INTERFACE, at time "<< tot_time <<" is " << L1_normal_interface_para <<std::endl;

        l1_normal_interface_para /= counter_interface_pts_para;
//        std::cout<<"PARA-l1(un) at the INTERFACE, at time "<< tot_time <<" is " << l1_normal_interface_para <<std::endl;
//        std::cout<<"PARA-linf(un) at the INTERFACE, at time "<< tot_time <<" is " << linf_u_n_para <<std::endl;
//        std::cout<<"PARA-linf(un) SIGNED at the INTERFACE, at time "<< tot_time <<" is " << max_u_n_val_para <<std::endl;

        l1_err_u_n_time_para.push_back(l1_normal_interface_para);
        linf_err_u_n_time_para.push_back(linf_u_n_para);
        L1_err_u_n_time_para.push_back(L1_normal_interface_para);
        max_val_u_n_time_para.push_back(max_u_n_val_para);

//        check_goal_quantities_final( msh_i , ls_cell , u_projected, perimeter, d_a,  area_fin, centre_mass_x ,   centre_mass_y , degree_FEM , mass_fin , degree_velocity , l1_divergence_error_fin , l2_divergence_error_fin , linf_divergence_error_fin , radius , L1_divergence_error_fin ,  time_step ,rise_vel0 , rise_vel1 ,flux_interface,counter_interface_pts_fin,degree_curve,int_refsteps);
//

        testing_level_set_max_min(msh, level_set_function, time_step, min_max_vec);


        l1_divergence_error_fin /= counter_interface_pts_fin;


        std::cout << "The l1 error of the PARA_CONT CURVATURE at the INTERFACE, at time " << tot_time << " is "
                  << l1_divergence_error_fin << std::endl;

        std::cout << bold << green << "The linf error of the PARA_CONT CURVATURE at the INTERFACE, at time " << tot_time
                  << " is " << linf_divergence_error_fin << reset << std::endl;

        std::cout << "The L1 error of the PARA_CONT CURVATURE at the INTERFACE, at time " << tot_time << " is "
                  << L1_divergence_error_fin << std::endl;








//        std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter << ", Initial PERIMETER =  "<<perimeter_initial<<std::endl;

//        std::cout<<"NORMALISED DIFFERENCE PERIMETER, at time "<< tot_time <<" is " << (perimeter - perimeter_initial)/perimeter_initial <<std::endl;

        d_a = sqrt(4.0 * area_fin / M_PI);

        std::cout << "The CIRCULARITY (OLD), at time " << tot_time << " is " << M_PI * d_a / perimeter << std::endl;
        std::cout << "The CIRCULARITY, at time " << tot_time << " is "
                  << 4.0 * M_PI * area_fin / (perimeter * perimeter) << std::endl;
        std::cout << "Area at time step: " << tot_time << " is " << area_fin << std::endl;
//        std::cout << "Internal mass at time step: "<<tot_time<<" is "<< mass_fin << std::endl;

        std::cout << bold << yellow << "NORMALISED Difference in AREA AT TIME " << tot_time << " IS "
                  << (area_fin - initial_area) / initial_area << reset << std::endl;
//        std::cout << "NORMALISED Difference in INTERNAL MASS AT TIME "<<tot_time<<" IS "<< (std::abs(mass_fin - initial_mass))/(std::abs( initial_mass )) << std::endl;
        std::cout << "CENTRE OF MASS at time step: " << tot_time << " is " << " ( " << centre_mass_x / area_fin << " , "
                  << centre_mass_y / area_fin << " ). " << std::endl;
        std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: " << tot_time << " is " << " ( "
                  << centre_mass_x / area_fin - centre_mass_x_inital / initial_area << " , "
                  << centre_mass_y / area_fin - centre_mass_y_inital / initial_area << " ). " << std::endl;
        R_phi = sqrt(area_fin / M_PI);
        std::cout << "Abs error over expected radius = " << std::abs(R_phi - radius) << '\n' << std::endl;




// Eccentricity (good only for elliptic solutions)
        T radius_max = 1.0 / min_curvature;
        T radius_min = 1.0 / max_curvature;

        T eccentricity = sqrt(1.0 - radius_min * radius_min / (radius_max * radius_max));

        std::cout << bold << green << "Eccentricity = " << eccentricity << reset << std::endl;
        eccentricity_vec.push_back(eccentricity);

        time_vec.push_back(tot_time);
        area_time.push_back(area_fin);
        l1_err_u_n_time.push_back(l1_normal_interface_status);
        linf_err_u_n_time.push_back(max_u_n_val_abs);
        L1_err_u_n_time.push_back(L1_normal_interface_status);

        linf_der_time_interface.push_back(diff_in_time_interface);

        max_val_u_n_time.push_back(max_u_n_val);
        l1_err_curvature_time.push_back(l1_divergence_error_fin);
        linf_err_curvature_time.push_back(linf_divergence_error_fin);


        circularity_time2.push_back(4.0 * M_PI * area_fin / (perimeter * perimeter));
        circularity_time.push_back(M_PI * d_a / perimeter);
        perimeter_time.push_back(perimeter);
        centre_mass_err_time.push_back(std::make_pair(centre_mass_x / area_fin, centre_mass_y / area_fin));

        flux_interface_time.push_back(flux_interface);
        rise_velocity_time.push_back(std::make_pair(rise_vel0 / area_fin, rise_vel1 / area_fin));


        if (tot_time >= final_time) {
            std::cout << "Final time T = " << tot_time << " achieved. Simulation ended at iteration = " << time_step
                      << std::endl;
            std::cout << "Tot amount HHO routines (fixed-point scheme), N = " << time_step << std::endl;
            break;
        }
        tc_iteration.toc();
        std::cout << "Time for this iteration = " << tc_iteration << std::endl;
        plotting_in_time_new(time_vec, area_time, l1_err_u_n_time, linf_err_u_n_time, max_val_u_n_time,
                             l1_err_curvature_time, linf_err_curvature_time, dt_M, min_max_vec, flux_interface_time,
                             rise_velocity_time, centre_mass_err_time, perimeter_time, circularity_time2,
                             circularity_ref, perim_ref, area_ref, radius,
                             L1_err_u_n_time, l1_err_u_n_time_para, linf_err_u_n_time_para, L1_err_u_n_time_para,
                             max_val_u_n_time_para, linf_der_time_interface, eccentricity_vec, folder);
    } // End of the temporal loop








    std::cout << "Tot amount transport problem sub-routines tilde{N} = " << tot_amount_transport_routine << std::endl;


    std::cout << "FINAL TIME IS t = " << tot_time << std::endl;

    tc_tot.toc();
    std::cout << "Simulation machine time t = " << tc_tot << std::endl;

    return 0;
}

#endif




// Interface Stokes Problem: Two-fluid problem with FIXED POINT
// -------- Code paper: interface evolution under shear flow - perturbed flow - null flow
// Starting from numerical velocity field and level set - to be uploaded before the simulation

#if 0

int main(int argc, char **argv) {
    using RealType = double;
    RealType sizeBox = 0.5;
    size_t degree = 1;
    size_t int_refsteps = 0; // 4
    size_t degree_FEM = 2;
    size_t degree_curve = 2;
    size_t degree_curvature = 1; // degree_curve -1 ;
    bool dump_debug = false;
    bool solve_interface = true;
    bool solve_fictdom = false;
    bool agglomeration = true;

    bool high_order = false; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool entropic = false; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool entropic_mass_consistent = false;
    bool compressed = false; // IF FALSE IS PHI_L, IF TRUE  PHI_HP
    bool cut_off_active = false; // IF FALSE IS SMOOTH, IF TRUE  CUT_OFF

    mesh_init_params <RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;

    mip.min_x = -sizeBox;
    mip.min_y = -sizeBox;
    mip.max_x = sizeBox;
    mip.max_y = sizeBox;


    size_t T_N = 10000;
    int ch;
    while ((ch = getopt(argc, argv, "k:q:M:N:r:T:l:p:ifDAdhesgc")) != -1) {
        switch (ch) {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'q':
                degree_FEM = atoi(optarg);
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;

            case 'T':
                T_N = atoi(optarg);
                break;

            case 'l':
                degree_curve = atoi(optarg);
                break;

            case 'p':
                degree_curvature = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;

            case 'h':
                high_order = true;
                break;

            case 'e':
                entropic = true;
                break;

            case 's':
                entropic_mass_consistent = true;
                break;

            case 'g':
                compressed = true;
                break;

            case 'c':
                cut_off_active = true;
                break;


            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;


    timecounter tc;

    timecounter tc_tot;
    tc_tot.tic();


/************** BUILD MESH **************/

    cuthho_poly_mesh <RealType> msh(mip);
    typedef cuthho_poly_mesh <RealType> Mesh;
    typedef RealType T;
//    typedef typename Mesh::point_type point_type;
    offset_definition(msh);

    std::cout << "Mesh size = " << mip.Nx << "x" << mip.Ny << std::endl;
    std::cout << "Number of refine interface points: r = " << int_refsteps << std::endl;



/************** FINITE ELEMENT INITIALIZATION **************/
    auto fe_data = Finite_Element<RealType, Mesh>(msh, degree_FEM, mip);
    typedef Finite_Element<RealType, Mesh> FiniteSpace;
    std::cout << "Level Set (finite element approximation): Bernstein basis in space Q^{k_phi},  k_phi = " << degree_FEM
              << std::endl;

/**************************************TRANSPORT PROBLEM METHOD *****************************************/
    auto method_transport_pb = Transport_problem_method<Mesh, FiniteSpace>(fe_data, msh);
//typedef  Transport_problem_method<Mesh, FiniteSpace> Method_Transport;

    size_t degree_gradient = degree_FEM;
    size_t degree_div = degree_FEM;
    std::cout << "Finite element space for gradient and divergence of the LS: grad deg = " << degree_gradient
              << " , div deg = " << degree_div << std::endl;

    auto fe_data_gradient = Finite_Element<RealType, Mesh>(msh, degree_gradient, mip);
    auto method_transport_pb_grad = Transport_problem_method<Mesh, FiniteSpace>(fe_data_gradient, msh);
    auto fe_data_div = Finite_Element<RealType, Mesh>(msh, degree_div, mip);
    auto method_transport_pb_div = Transport_problem_method<Mesh, FiniteSpace>(fe_data_div, msh);




/**************  VELOCITY FIELD  INITIALISATION  **************/


    size_t degree_velocity = degree_FEM; // std::max(degree + 1 , degree_FEM) ;


// **************** --------> STORING OF LAGRANGIAN NODES
    nodes_Lagrangian_cell_definition(msh, degree_velocity);


    auto fe_data_Lagrange = Finite_Element<RealType, Mesh>(msh, degree_velocity, mip);


    std::cout << "Velocity field: high order Lagrange basis: degree = " << degree_velocity << std::endl;
    auto u_projected = velocity_high_order<Mesh, FiniteSpace, T>(fe_data_Lagrange, msh);

    std::cout << "Uploading velocity field.. " << std::endl;
    u_projected.upload_velocity("FEM_velocityX.txt", "FEM_velocityY.txt");

/************** LEVEL SET FUNCTION DISCRETISATION **************/
    bool circle = true, ellipse = false;
    bool flower = true;
    RealType radius_a, radius_b, radius;
    RealType x_centre = 0.0;
    RealType y_centre = 0.0;
    if (circle) {
        radius = 1.0 / 3.0;
    }

    if (ellipse) {
        radius_a = 1.0 / 12.0;
        radius_b = 1.0 / 24.0;
    }
    if (flower) {

// T radiusOLD = 0.31 ;
// radius_a = 1.0/12.0;
// radius_b = 1.0/24.0;
// radius_a = 1.0/6.0;
// radius_b = 1.0/12.0;
// radius =  sqrt( radius_a * radius_b ) ;
// T ratioR = (radiusOLD*radiusOLD)/(radius*radius);
// T oscillation = 0.04/ratioR ;
        T oscillation = 0.04;
        radius = 1.0 / 3.0;
        std::cout << "Initial interface: FLOWER" << std::endl;
        auto level_set_function_anal = flower_level_set<T>(radius, x_centre, y_centre, 4, oscillation); //0.11
        typedef flower_level_set <T> Fonction;
    }

    auto level_set_function_anal = circle_level_set<RealType>(radius, x_centre,
                                                              y_centre); // random data, useful only to initialize the level set
    typedef circle_level_set <T> Fonction;
    auto level_set_function = Level_set_berstein<Mesh, Fonction, FiniteSpace, T>(fe_data, level_set_function_anal, msh,
                                                                                 fe_data_gradient, fe_data_div);
    typedef Level_set_berstein<Mesh, Fonction, FiniteSpace, T> Level_Set;

    std::cout << "Updating level set.. " << std::endl;
    level_set_function.upload_level_set("FEM_level_set.txt");
    level_set_function.converting_into_FE_formulation(level_set_function.sol_HHO);

// ------------------  IF GRADIENT CONTINUOUS --------------
    level_set_function.gradient_continuous_setting(method_transport_pb_grad);
//  ------------------ IF DIVERGENCE CONTINUOUS  ------------------
//    level_set_function.divergence_continuous_setting(method_transport_pb_div ) ;

//-------->  OLD FORMULATION GRAD CONT
//    auto level_set_function = Level_set_berstein_high_order_interpolation_grad_cont< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
//    level_set_function.gradient_continuous_setting() ;
//    typedef Level_set_berstein_high_order_interpolation_grad_cont< Mesh , Fonction , FiniteSpace , T > Level_Set;




    std::cout << "Parametric interface: degree_curve = " << degree_curve << std::endl;
    auto curve = Interface_parametrisation_mesh1d(degree_curve);
    size_t degree_det_jac_curve = curve.degree_det; // 2*degree_curve INUTILE PER ORA
// integration CUT CELL degree += 2*degree_curve
// integration INTERFACE degree += degree_curve-1

//auto level_set_function = Level_set_berstein_high_order_interpolation_grad_cont_fast< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);
//auto level_set_function = Level_set_berstein_high_order_interpolation< Mesh , Fonction , FiniteSpace , T > (fe_data , level_set_function_anal , msh);





/************** MESH INITIALISATION FOR ROUTINE  **************/


    auto crr_mesh = Current_Mesh<Mesh>(msh);
    crr_mesh.current_mesh = msh;
    Mesh msh_i = crr_mesh.current_mesh;      // MESH at t=t^n (FOR THE PROCESSING)
    offset_definition(msh_i);


/************** INITIAL DATA INITIALISATION (t = 0) **************/
    T dt = 0.;
    T initial_area = 0., initial_mass = 0.;
    T d_a = 0.;
    T perimeter_initial = 0.;
    T centre_mass_x_inital = 0., centre_mass_y_inital = 0.;
    T max_u_n_val_old = 1e+6, max_u_n_val_new = 1e+5;
    T check = 10.0;
    T tot_time = 0.;

/************** BOUNDARY CONDITIONS **************/
    std::cout << yellow << bold << "INLET BDRY: UP AND DOWN FOR DIRCIRCHLET_eps FP" << reset << std::endl;
    bool bdry_bottom = false, bdry_up = false;
    bool bdry_left = false, bdry_right = false;
    check_inlet(msh, fe_data, bdry_bottom, bdry_right, bdry_up, bdry_left, 1e-14);


//************ DO cutHHO MESH PROCESSING **************
    tc.tic();
    detect_node_position3(msh_i, level_set_function); // In cuthho_geom
//detect_node_position3_parallel(msh_i, level_set_function); // In cuthho_geom
    detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom

    if (agglomeration) {
        detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom
//detect_cut_cells3_parallelized(msh_i, level_set_function); // In cuthho_geom
        refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
        set_integration_mesh(msh_i, degree_curve);
        detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
        make_neighbors_info_cartesian(msh_i); // Non serve modificarla
//refine_interface_pro3(msh_i, level_set_function, int_refsteps);

        make_agglomeration_no_double_points(msh_i, level_set_function, degree_det_jac_curve);
        set_integration_mesh(msh_i, degree_curve); // TOLTO PER IL MOMENTO SENNO RADDOPPIO
//        make_agglomeration(msh_i, level_set_function); // Non serve modificarla

    } else {
        detect_cut_cells3(msh_i, level_set_function);
//refine_interface_pro3(msh_i, level_set_function, int_refsteps);
        refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
    }

    tc.toc();
    std::cout << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << '\n' << std::endl;

    if (dump_debug) {
        dump_mesh(msh_i);
        output_mesh_info(msh_i, level_set_function);
    }



// IN cuthho_export..Points/Nodes don't change-> it's fast
    output_mesh_info2_pre_FEM(msh_i, level_set_function); // IN cuthho_export

/************** UPDATING  LEVEL SET  AND VELOCITY  **************/
//    level_set_function.gradient_continuous_setting() ;
//    // IF GRADIENT CONTINUOUS
    level_set_function.gradient_continuous_setting(method_transport_pb_grad);
//    // IF DIVERGENCE CONTINUOUS
//    level_set_function.divergence_continuous_setting(method_transport_pb_div) ;


// --------------------- LS_CELL: CHOICE OF DISC/CONT ------------------------------- \\

// IF grad cont -> normal cont -> (divergence disc) -> divergence cont
//auto ls_cell = LS_cell_high_order_curvature_cont< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);

// IF grad cont -> normal cont -> divergence disc
    auto ls_cell = LS_cell_high_order_grad_cont_div_disc<T, Mesh, Level_Set, Fonction, FiniteSpace>(level_set_function,
                                                                                                    msh_i);

// IF grad disc -> normal disc -> divergence disc
//    auto ls_cell = LS_cell_high_order_grad_disc_div_disc< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);
// IF grad disc -> normal disc -> divergence disc -> normal and grad cont
//    auto ls_cell = LS_cell_high_order_div_disc_grad_n_cont< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i);

//-------------------------- OLD CASE LS_CELL --------------------------
//    auto ls_cell = LS_cell_high_order_grad_cont< T , Mesh , Level_Set, Fonction , FiniteSpace >(level_set_function,msh_i );



    ls_cell.radius = radius;
    u_projected.set_agglo_mesh(msh_i);

    timecounter tc_initial;
    tc_initial.tic();



/************** PLOTTINGS + GOAL QUANTITIES  **************/
    std::vector <T> max_val_u_n_time_para, L1_err_u_n_time_para, l1_err_u_n_time_para, linf_err_u_n_time_para, L1_err_u_n_time;
    std::vector <T> area_time, l1_err_u_n_time, linf_err_u_n_time, time_vec;
    std::vector <T> linf_der_time_interface, eccentricity_vec;
    std::vector <T> max_val_u_n_time, l1_err_curvature_time, linf_err_curvature_time;
    std::vector <T> circularity_time, circularity_time2, flux_interface_time, perimeter_time;
    std::vector <std::pair<T, T>> centre_mass_err_time, rise_velocity_time, min_max_vec;
    T circularity_ref = 0.0, perim_ref = 0.0, area_ref = 0.0;
    T l1_divergence_error = 0., l2_divergence_error = 0.;
    T linf_divergence_error = -10.;
    T L1_divergence_error = 0.;

    check_goal_quantities(msh_i, ls_cell, perimeter_initial, d_a, initial_area, centre_mass_x_inital,
                          centre_mass_y_inital, degree_FEM, initial_mass, flower, l1_divergence_error,
                          l2_divergence_error, linf_divergence_error, radius, L1_divergence_error, ellipse,
                          degree_curve, int_refsteps);



//------------------------ CHECK REFERENCE QUANTITIES ---------------------------//
    reference_quantities_computation(perim_ref, area_ref, circularity_ref, radius, x_centre, y_centre, fe_data, msh,
                                     degree_curve, perimeter_initial, initial_area, int_refsteps, degree_det_jac_curve);

//    plot_curvature_normal_vs_curv_abscisse(msh_i, ls_cell, degree_curve,int_refsteps , 0 );

    plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(msh_i, ls_cell, degree_curve, int_refsteps, 0, degree_curvature);


    tc_initial.toc();
//    std::cout << "Time Machine for checking INITAL GOAL QUANTITIES: " << tc_initial << " seconds" << std::endl;

    circularity_time.push_back(M_PI * d_a / perimeter_initial);
    circularity_time2.push_back(4.0 * M_PI * initial_area / (perimeter_initial * perimeter_initial));
    perimeter_time.push_back(perimeter_initial);
    centre_mass_err_time.push_back(
            std::make_pair(centre_mass_x_inital / initial_area, centre_mass_y_inital / initial_area));
    time_vec.push_back(0);
    area_time.push_back(initial_area);

    min_max_vec.push_back(std::make_pair(level_set_function.phi_min, level_set_function.phi_max));

    l1_err_curvature_time.push_back(l1_divergence_error);
    linf_err_curvature_time.push_back(linf_divergence_error);

    T dt_M;
    T R_phi = radius;

    size_t tot_amount_transport_routine = 0;


    bool l2proj_para = false;

    bool l2proj = true;
    bool avg = false;
    bool disc = false;
    bool filter = false;

    Interface_parametrisation_mesh1d_global <Mesh> para_curve_cont(msh_i, degree_curve, degree_curvature);

// *********************** DERIVATIVE / NORMAL PARA *************************//
//------------- L2 cont curvature from parametric interface  r ---------- //
    para_curve_cont.make_L2_proj_para_derivative(msh_i);

//---------------------------- L2 global Normal from LS  ----------------------- //
    if (l2proj) {
        if (!disc)
            para_curve_cont.make_L2_proj_para_normal(msh_i, ls_cell);
        else
            para_curve_cont.make_L2_proj_para_normal_disc(msh_i, ls_cell);
    }
//---------------------------- Avg Normal from LS  ---------------------------- //
    if (avg) {
        if (!disc)
            para_curve_cont.make_avg_L2_local_proj_para_normal(msh_i, ls_cell);
        else
            para_curve_cont.make_avg_L2_local_proj_para_normal_disc(msh_i, ls_cell);
    }


// *********************** CURVATURE PARA *************************//

//------------- L2 cont curvature from parametric interface  r ---------- //
    if (l2proj_para)
        para_curve_cont.make_L2_proj_para_curvature(msh_i);



//---------------------------- L2 global Curvature from LS  ----------------------- //
    if (l2proj) {
        if (!disc)
            para_curve_cont.make_L2_proj_para_curvature(msh_i, ls_cell);
        else
            para_curve_cont.make_L2_proj_para_curvature_disc(msh_i, ls_cell);
    }
//---------------------------- Avg Curvature from LS  ---------------------------- //
    if (avg) {
        if (!disc)
            para_curve_cont.make_avg_L2_local_proj_para_curvature(msh_i, ls_cell);
        else
            para_curve_cont.make_avg_L2_local_proj_para_curvature_disc(msh_i, ls_cell);

    }
    if (filter)
        para_curve_cont.make_smooth_filter_curvature();

// ******** TO FASTER THE SIMULATION, ERASED THE PLOTTINGS
    plotting_para_curvature_cont_time_fast(msh_i, para_curve_cont, degree_curve, degree_FEM, radius, 0, int_refsteps);

    T final_time = 3.0;

    T eps_dirichlet_cond = 0.26; //0.26; 0.59 ; // 0.01 -->  0.1

    for (size_t time_step = 0; time_step <= T_N; time_step++) {
        timecounter tc_iteration;
        tc_iteration.tic();
        std::cout << '\n' << bold << yellow << "Starting iteration numero  = " << time_step << " --> time t = "
                  << tot_time << reset << std::endl;
        std::cout << "Dirichlet eps Cond = " << eps_dirichlet_cond << std::endl;

// -----------------------------------------------------------------------------------------
// ----------------- RESOLUTION OF THE STOKES PROBLEM (HHO) ------------------
// -----------------------------------------------------------------------------------------

        bool sym_grad = TRUE;
        auto prm = params<T>();
        prm.kappa_1 = 1.0;
        prm.kappa_2 = 1.0;
        T gamma = 1.0; // 0.05;


        savingVelocityLevelSet(level_set_function, u_projected);


        std::cout << '\n' << bold << yellow << "HHO flow resolution." << reset << '\n' << std::endl;


// ------------------ OLD VERSIONS ------------------
//auto test_case = make_test_case_eshelby(msh_i, ls_cell,  prm , sym_grad);
// Non serve modificare Gamma = 1/2
//auto test_case = make_test_case_eshelby_2(msh_i, ls_cell,  prm , sym_grad );
//auto test_case = make_test_case_eshelby_analytic(msh_i, ls_cell,  prm , sym_grad , radius);
// ------------- OLD GUILLAUME VERSIONS --------------
// auto test_case = make_test_case_stokes_1(msh, level_set_function);
// auto test_case = make_test_case_stokes_2(msh, ls_cell); //level_set_function);

// ----------------- ESHELBY VERSION - CORRECT BUT PRESSURE ------------------
//auto test_case_prova = make_test_case_eshelby_2_prova(msh_i, ls_cell,  prm , sym_grad );
// PRESSURE SIGN NOT CORRECT
// ---------------------- ESHELBY VERSION LEVEL SET - CORRECT ------------------------
//        auto test_case = make_test_case_eshelby_correct(msh_i, ls_cell,  prm , sym_grad,gamma);
// PRESSURE SIGN NOT CORRECT
// -------------------- ESHELBY VERSION PARAMETRIC (DISC) - CORRECT -------------------
//        auto test_case = make_test_case_eshelby_correct_parametric(msh_i, ls_cell,  prm , sym_grad,gamma);
// PRESSURE SIGN NOT CORRECT




// -------------------- ESHELBY VERSION PARAMETRIC (CONT) - CORRECT -------------------
// domain  (0,1)^2 - null flow
//        auto test_case_prova = make_test_case_eshelby_correct_parametric_cont( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma );

// domain  (-a,a)^2 - shear flow
//  auto test_case_prova = make_test_case_eshelby_parametric_cont_eps_DIR_domSym( msh_i, ls_cell ,para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond); // sizeBox

// domain  (0,1)^2 - shear flow

//        auto test_case_prova = make_test_case_eshelby_correct_parametric_cont_DIRICHLET_eps( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond);

// domain  (-a,a)^2 - new test case perturbated
        T perturbation = 0.5;
        auto test_case_prova = make_test_case_eshelby_parametric_cont_eps_perturbated_DIR_domSym(msh_i, ls_cell,
                                                                                                 para_curve_cont, prm,
                                                                                                 sym_grad, gamma,
                                                                                                 eps_dirichlet_cond,
                                                                                                 perturbation);
// domain  (0,1)^2 - new test case perturbated
//        T perturbation = 0.5;
//     auto test_case_prova = make_test_case_shear_flow_perturbated( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond,perturbation );
//        auto test_case_prova = make_test_case_shear_y( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond,perturbation );
// New test case TGV - fixed point
//        auto test_case_prova = make_test_case_TGV_FPscheme( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma , eps_dirichlet_cond );



//        auto test_case_prova = make_test_case_eshelby_correct_parametric_cont_TGV_source( msh_i, ls_cell , para_curve_cont, prm , sym_grad , gamma );
// ------------------------ HHO METHOD FOR LEVEL SET  ---------------------------
//        auto method = make_sym_gradrec_stokes_interface_method(msh_i, 1.0, 0.0, test_case, sym_grad);
// -------------------- HHO METHOD FOR DISC PARAMETRIC INTERFACE  -----------------------
//         auto method = make_sym_gradrec_stokes_interface_method_perturbationref_pts(msh_i, 1.0, 0.0, test_case, sym_grad);


// -------------------- HHO METHOD FOR CONT PARAMETRIC INTERFACE  -----------------------
        auto method_prova = make_sym_gradrec_stokes_interface_method_ref_pts_cont(msh_i, 1.0, 0.0, test_case_prova,
                                                                                  sym_grad);




//  ******************** - HHO RESOLUTION - ********************
        if (solve_interface) {
// ----------------- HHO RESOLUTION OLD CASE  --------------------------
//            TI = run_cuthho_interface_numerical_ls(msh_i, degree, method, test_case_prova , ls_cell ,  normal_analysis );
//            run_cuthho_interface_velocity_parallel(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad );

// ----------------- HHO RESOLUTION LS / PARAMETRIC DISC  ---------------------
//            run_cuthho_interface_velocity_prova(msh_i, degree, method,test_case, ls_cell , u_projected ,sym_grad , time_step); // THE ONE CORRECT THAT I'M USING NOW

// ----------------- HHO RESOLUTION PARAMETRIC CONT  --------------------------
//             run_cuthho_interface_velocity_new(msh_i, degree, method_prova,test_case_prova, ls_cell , u_projected ,sym_grad , time_step); // THE ONE CORRECT THAT I'M USING NOW
            run_cuthho_interface_velocity_fast(msh_i, degree, method_prova, test_case_prova, ls_cell, u_projected,
                                               sym_grad, time_step); // CORRECT BUT DOEST NOT COMPUTE ERRORS
        }
//        testing_velocity_field(msh , u_projected) ;
/************************************ FEM -  PRE-PROCESSING ******************************************/
// ----------------- PROJECTION OF THE VELOCITY FIELD ------------------
        if (0)
            std::cout << bold << green
                      << "CASE WITH VELOCITY DISCONTINUOUS: ho solo sol_HHO, sol_FEM non salvato, va cambiato il transport pb!!!"
                      << reset << std::endl;

        if (0) //results of the paper with this
        {
            std::cout << '\n' << "Smoothing operator from velocity HHO to FE (continuity imposed): geometrical average."
                      << std::endl;
            u_projected.smooth_converting_into_FE_formulation(u_projected.sol_HHO);
        }
        if (1) //weighted implementation - reviewer results
        {
            std::cout << '\n' << "Smoothing operator from velocity HHO to FE (continuity imposed): geometrical average."
                      << std::endl;
            u_projected.smooth_converting_into_FE_formulation(u_projected.sol_HHO);
        }
        if (0) {
            std::cout << '\n' << "------------------>>>> NOTICE: NON SMOOTH OPERATOR FROM HHO TO FEM." << std::endl;
            u_projected.converting_into_FE_formulation(u_projected.sol_HHO);
        }
        if (0) {
            std::cout << '\n' << "------------------>>>>NOTICE: L^2 PROJECTION FROM HHO TO FEM." << std::endl;
            u_projected.L2_proj_into_FE_formulation(level_set_function, msh, method_transport_pb);
        }


//testing_velocity_field(msh , u_projected) ;
//auto u_prova = velocity_high_order <Mesh,FiniteSpace,T> (fe_data , msh);
//u_prova.sol_HHO = u_projected.sol_HHO ;
//u_prova.L2_proj_into_FE_formulation( level_set_function , msh );
//testing_velocity_field_L2projected(msh , u_prova) ;

        T rise_vel0 = 0.0, rise_vel1 = 0.0;
        T flux_interface = 0.0;

        if (time_step == 0) {
            T max_u_n_val_initial = 0.0;
            T max_u_n_val_abs_initial = 0.0;
            T l1_normal_interface_status_initial = 0.;

            T L1_normal_interface_status_initial = 0.;

            size_t counter_interface_pts_initial = 0;
            for (auto &cl: msh_i.cells) {
                if (cl.user_data.location == element_location::ON_INTERFACE) {
                    ls_cell.cell_assignment(cl);
                    u_projected.cell_assignment(cl);

                    auto qps = integrate_interface(msh_i, cl, degree_FEM + degree_velocity,
                                                   element_location::ON_INTERFACE);
                    for (auto &qp: qps) {
                        auto u_pt = u_projected(qp.first);

                        auto ls_n_pt = ls_cell.normal(qp.first);
                        T u_n_val = u_pt.first * ls_n_pt(0) + u_pt.second * ls_n_pt(1);
                        L1_normal_interface_status_initial += qp.second * std::abs(u_n_val);
                        max_u_n_val_abs_initial = std::max(max_u_n_val_abs_initial, std::abs(u_n_val));
                        if (std::abs(u_n_val) == max_u_n_val_abs_initial)
                            max_u_n_val_initial = u_n_val;

                        l1_normal_interface_status_initial += std::abs(u_n_val);
                        counter_interface_pts_initial++;


                    }

                }

            }
            l1_normal_interface_status_initial /= counter_interface_pts_initial;
            l1_err_u_n_time.push_back(l1_normal_interface_status_initial);
            linf_err_u_n_time.push_back(max_u_n_val_abs_initial);
//            linf_der_time_interface.push_back(0) ;
            max_val_u_n_time.push_back(max_u_n_val_initial);
            L1_err_u_n_time.push_back(L1_normal_interface_status_initial);

            std::cout << "------> The l1 error of u*n along the INTERFACE at INITIAL TIME is "
                      << l1_normal_interface_status_initial << std::endl;

            std::cout << bold << green << "------> The linf error of u*n along the INTERFACE at INITIAL TIME is "
                      << max_u_n_val_abs_initial << reset << std::endl;

            std::cout << "------> The L1 error of u*n along the INTERFACE at INITIAL TIME is "
                      << L1_normal_interface_status_initial << std::endl;


            size_t degree_curvature = para_curve_cont.dd_degree;
            size_t degree_jacobian = para_curve_cont.degree_det;

            T L1_normal_interface_para = 0.0;
            T linf_u_n_para = 0.0;
            T max_u_n_val_para = 0.0;
            T l1_normal_interface_para = 0.0;
            size_t counter_interface_pts_para = 0;


            T area_para = 0.0;

            for (auto &cl: msh_i.cells) {

                if ((location(msh_i, cl) == element_location::IN_NEGATIVE_SIDE) ||
                    (location(msh_i, cl) == element_location::ON_INTERFACE)) {
                    u_projected.cell_assignment(cl);
                    T partial_area = measure(msh_i, cl, element_location::IN_NEGATIVE_SIDE);
                    area_para += partial_area;

                    size_t max_deg = std::max(degree_velocity, degree_FEM);
                    auto qps_fin = integrate(msh_i, cl, max_deg, element_location::IN_NEGATIVE_SIDE);


                    for (auto &qp: qps_fin) {
                        auto u_pt = u_projected(qp.first);
                        rise_vel0 += qp.second * u_pt.first;
                        rise_vel1 += qp.second * u_pt.second;
                    }

                }


                if (cl.user_data.location == element_location::ON_INTERFACE) {
                    u_projected.cell_assignment(cl);
                    auto global_cells_i = para_curve_cont.get_global_cells_interface(msh_i, cl);
                    auto integration_msh = cl.user_data.integration_msh;
//                    auto degree_int = degree_curvature + degree_jacobian ;


                    auto qps_un = edge_quadrature<T>(degree_jacobian + degree_curvature + degree_velocity);

                    for (size_t i_cell = 0; i_cell < integration_msh.cells.size(); i_cell++) {
                        auto pts = points(integration_msh, integration_msh.cells[i_cell]);
                        size_t global_cl_i = global_cells_i[i_cell];

                        for (auto &qp: qps_un) {
                            auto t = 0.5 * qp.first.x() + 0.5;

                            T jacobian = para_curve_cont.jacobian_cont(t, global_cl_i);
                            auto w = 0.5 * qp.second * jacobian;
                            auto p = para_curve_cont(t, global_cl_i);
                            auto pt = typename Mesh::point_type(p(0), p(1));
                            auto u_pt = u_projected(pt);
                            auto curve_n_pt = para_curve_cont.normal_cont(t, global_cl_i);
                            T flux = u_pt.first * curve_n_pt(0) + u_pt.second * curve_n_pt(1);
                            flux_interface += w * flux;

                            L1_normal_interface_para += w * std::abs(flux);
                            linf_u_n_para = std::max(linf_u_n_para, std::abs(flux));
                            if (std::abs(flux) == linf_u_n_para)
                                max_u_n_val_para = flux;

                            l1_normal_interface_para += std::abs(flux);
                            counter_interface_pts_para++;

                        }
                    }
                }
            }
            l1_normal_interface_para /= counter_interface_pts_para;
            l1_err_u_n_time_para.push_back(l1_normal_interface_para);
            linf_err_u_n_time_para.push_back(linf_u_n_para);
            L1_err_u_n_time_para.push_back(L1_normal_interface_para);
            max_val_u_n_time_para.push_back(max_u_n_val_para);
            flux_interface_time.push_back(flux_interface);
            rise_velocity_time.push_back(std::make_pair(rise_vel0 / area_para, rise_vel1 / area_para));

            std::cout << "------> The l1 error of PARA_CONT u*n along the INTERFACE at INITIAL TIME is "
                      << l1_normal_interface_para << std::endl;

            std::cout << bold << yellow
                      << "------> The linf error of PARA_CONT u*n along the INTERFACE at INITIAL TIME is "
                      << linf_u_n_para << reset << std::endl;

            std::cout << "------> The L1 error of PARA_CONT u*n along the INTERFACE at INITIAL TIME is "
                      << L1_normal_interface_para << std::endl;
        }
// -----------------------------------------------------------------------------------------
// ----------------- TIME EVOLUTION (u^n,phi^n) (FEM) ------------------
// -----------------------------------------------------------------------------------------


//------------- MACRO TIME STEP dt -------------
//(before checking if it is too big, I need to find the new interface)
        T dt_M_max = 0.1; // 0.1 ; //5e-2 ; //5e-2; // phi_h 1e-3 e dt_m^MAX = 1e-4 not enought (oscillation )

//if(tot_time < 0.75)
        T eps = 0.05; //1.0 ; // 0.05 ; // factor to be inside CFL stability zone IT WAS 0.4
        T dt_one_cell = time_step_CFL_new(u_projected, mip, eps);

// Stokes time step maximum
        T c2 = 2.0; // 2.0; // it was 1
        T eta = std::min(prm.kappa_1, prm.kappa_2);
        T dt_STK = time_step_STK(eta, gamma, mip, c2);
        dt_M = std::min(dt_one_cell, dt_STK);

// dt_M =  dt_M * 0.1 ; // add 23/09/2021

// ------> ADAPTIVE SCHEME
//        dt_M = dt_one_cell * 0.1 ;
//        dt_M = floorf(dt_M * 1000) / 1000;

//        T dt_M_max ;
//        if( std::max(mip.hx() , mip.hy() ) <= 16 )
//            dt_M_max = 0.1;
//        else if( std::max(mip.hx() , mip.hy() ) == 32 )
//            dt_M_max = 5e-2;
//        else
//            dt_M_max = 5e-2;



        dt = std::min(dt_M_max, dt_M); //5e-2 for mesh 32x32, 0.1 for  mesh 16x16, 8x8

        if (final_time < (dt + tot_time)) {
            std::cout << "Last step, dt_M changed to arrive at T." << std::endl;
            dt = final_time - tot_time;

        }
        int N_sub = 10; //N_sub = 20 for BOH, N_sub = 10 for  mesh 16x16, 8x8, 32x32
// ------> FIXED  SCHEME
//        dt_M =  8*1e-3; // FOR LS low phi_L order-> 8*1e-3; // FOR LS high order phi_H ->  2*1e-3;
//        dt = std::min(dt_one_cell , dt_M);
//        T N_sub = 20 ;  // N_sub = 20 for mesh 32x32, N_sub = 10 for  mesh 16x16,  N_sub = 40 for phi_H
//std::cout<<"dt1 is "<<dt1<<std::endl;


        std::cout << '\n' << "Macro time step dt_M = " << dt << ", dt_h = " << dt_one_cell << ", dt_STK = " << dt_STK
                  << std::endl;

// I can create a sub-time. I solve several time the FEM problem, given a Stokes field. The amount of time is s.t. at maximum there is a displacement of a cell of the interface and no more than a maximum T
        T sub_time = 0.;
        T sub_dt = dt / N_sub; //std::min(4*1e-4 , dt ) ;

// ------> ADAPTIVE SCHEME
// It is phi_L: > 1e-3 for 16x16 and 8x8  ; > 5e-4 for 32x32
// It is phi_H: > 1e-4 for 16x16 and 8x8  ; > 5e-4 for 32x32
        if (sub_dt > 1e-3) {
            sub_dt = 1e-3;
            N_sub = floor(dt / sub_dt); // N_sub varies between 10 and 100 depending on sub_dt

        }


        auto level_set_old = Level_set_berstein<Mesh, Fonction, FiniteSpace, T>(level_set_function);
        auto ls_old = LS_cell_high_order_grad_cont_div_disc<T, Mesh, Level_Set, Fonction, FiniteSpace>(level_set_old,
                                                                                                       msh_i);


        std::cout << "-----> Implemented sub time dt_m = " << sub_dt << std::endl;

// ------------- NEW IMPLEMENTATION WITH FAST LEVEL SET ---------------------
        std::cout << yellow << bold << "----------- STARTING TRANSPORT PROBLEM SUB-ROUTINE: tilde{N} = " << N_sub
                  << "-----------" << reset << std::endl;
        timecounter tc_case_tr;
        tc_case_tr.tic();
        while (sub_time < sub_dt * N_sub) {
// std::pair<T,T> CFL_numb;
            if (high_order) {
//           CFL_numb = run_transport_high_order_FTC_M_lumped( level_set_function.msh , fe_data , level_set_function , u_projected , method_transport_pb , sub_dt , false );

                run_transport_high_order_FTC_M_consistent(level_set_function.msh, fe_data, level_set_function,
                                                          u_projected, method_transport_pb, sub_dt, false);

//            CFL_numb = run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D_NEW_DIRICHLET_COND_NEW_LS_M_CONS_LIMITED( level_set_function.msh , fe_data , level_set_function , u_projected , method_transport_pb , sub_dt , false ); // CORRECT 25/01/21

            } else if (entropic) {
                run_transport_entropic_M_lumped_Ri_correct(level_set_function.msh, fe_data, level_set_function,
                                                           u_projected, method_transport_pb, sub_dt, false);
//             CFL_numb = run_transport_entropic_M_lumped ( level_set_function.msh , fe_data , level_set_function , u_projected , method_transport_pb , sub_dt , false );
//            CFL_numb = run_FEM_BERNSTEIN_CORRECT_FAST_NEW_D_NEW_DIRICHLET_COND_NEW_LS_M_LUMPED_NO_LIMITING ( level_set_function.msh , fe_data , level_set_function , u_projected , method_transport_pb , sub_dt , false ); // CORRECT 25/01/21
            } else {
//                if(degree_velocity == degree_FEM) // IT IS FASTER
                run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST_NEW_DIRICHLET_COND_NEW_LS(level_set_function.msh, fe_data,
                                                                                   level_set_function, u_projected,
                                                                                   method_transport_pb,
                                                                                   sub_dt); // CORRECT 25/01/21
//                else // IT WORKS BUT IT'S MUCH SLOWER
//                    run_FEM_BERNSTEIN_LOW_ORDER_CORRECT_FAST_NEW_DIRICHLET_COND_NEW_LS( level_set_function.msh , fe_data , level_set_function , u_projected , method_transport_pb , sub_dt  , fe_data_Lagrange);

            }


            sub_time += sub_dt;

        }
        tot_time += sub_time;

        tc_case_tr.toc();
        std::cout << yellow << bold << "----------- THE END OF TRANSPORT PROBLEM SUB-ROUTINE: machine time t = "
                  << tc_case_tr << " s -----------" << reset << std::endl;
        tot_amount_transport_routine += N_sub;

/**************************************************   POST-PROCESSING **************************************************/


// Updating continuous normal function
//            level_set_function.gradient_continuous_setting() ;
// IF GRADIENT CONTINUOUS
        level_set_function.gradient_continuous_setting(method_transport_pb_grad);
//         // IF DIVERGENCE CONTINUOUS
//         level_set_function.divergence_continuous_setting(method_transport_pb_div) ;

// Updating mesh data to check out differences in mass and areas
        crr_mesh.current_mesh = msh;
        msh_i = crr_mesh.current_mesh;
        offset_definition(msh_i);

        tc.tic();
        detect_node_position3(msh_i, level_set_function); // In cuthho_geom
        detect_cut_faces3(msh_i, level_set_function); // In cuthho_geom


        if (agglomeration) {
            detect_cut_cells3(msh_i, level_set_function); // In cuthho_geom

            refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
            set_integration_mesh(msh_i, degree_curve);

            detect_cell_agglo_set(msh_i, level_set_function); // Non serve modificarla
            make_neighbors_info_cartesian(msh_i); // Non serve modificarla
//refine_interface_angle(msh_i2, level_set_function, int_refsteps); // IN cuthho_geom
//refine_interface_pro3(msh_i, level_set_function, int_refsteps); // IN cuthho_geom
            make_agglomeration_no_double_points(msh_i, level_set_function, degree_det_jac_curve);
// make_agglomeration(msh_i, level_set_function); // Non serve modificarla
            set_integration_mesh(msh_i, degree_curve);
        } else {
            move_nodes(msh_i, level_set_function);
            detect_cut_cells3(msh_i, level_set_function);
            refine_interface_pro3_curve_para(msh_i, level_set_function, int_refsteps, degree_curve);
//refine_interface_pro3(msh_i, level_set_function, int_refsteps);
        }


        tc.toc();
        std::cout << '\n' << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << std::endl;

        if (dump_debug) {
            dump_mesh(msh_i);
            output_mesh_info(msh_i, level_set_function);
        }

// TOLTO TO BE FASTER
        if (time_step % 20 == 0) {
            output_mesh_info2_time(msh_i, level_set_function, tot_time, time_step);
            testing_level_set_time(msh, level_set_function, tot_time, time_step);
            plot_u_n_interface(msh_i, ls_cell, u_projected, time_step);
            testing_velocity_field(msh, u_projected, time_step);
        }
//        output_mesh_info2_time_fixed_mesh(msh_i, level_set_function,tot_time,time_step);
        testing_level_set(msh, level_set_function);
// Updating level set
        ls_cell.level_set = level_set_function;
        ls_cell.agglo_msh = msh_i;
        u_projected.set_agglo_mesh(msh_i);


        T max_u_n_val = 0.0;
        T max_u_n_val_abs = 0.0;

        T diff_in_time_interface = 0.0;
        T l1_normal_interface_status = 0.;

        T L1_normal_interface_status = 0.;

        size_t counter_interface_pts = 0;
        for (auto &cl: msh_i.cells) {
            if (cl.user_data.location == element_location::ON_INTERFACE) {
                ls_cell.cell_assignment(cl);
                ls_old.cell_assignment(cl);
                u_projected.cell_assignment(cl);

                auto qps = integrate_interface(msh_i, cl, degree_FEM + degree_velocity, element_location::ON_INTERFACE);
                for (auto &qp: qps) {
                    auto u_pt = u_projected(qp.first);
                    auto ls_n_pt = ls_cell.normal(qp.first);
                    T u_n_val = u_pt.first * ls_n_pt(0) + u_pt.second * ls_n_pt(1);
                    L1_normal_interface_status += qp.second * std::abs(u_n_val);
                    max_u_n_val_abs = std::max(max_u_n_val_abs, std::abs(u_n_val));

                    diff_in_time_interface = std::max(diff_in_time_interface,
                                                      (std::abs(ls_cell(qp.first) - ls_old(qp.first))) / sub_time);
                    if (std::abs(u_n_val) == max_u_n_val_abs)
                        max_u_n_val = u_n_val;

                    l1_normal_interface_status += std::abs(u_n_val);
                    counter_interface_pts++;

                }


            }
        }

        if (time_step == 0)
            max_u_n_val_new = max_u_n_val;

        if (time_step > 0) {
            max_u_n_val_old = max_u_n_val_new;
            max_u_n_val_new = max_u_n_val;
            std::cout << bold << yellow << "l^{inf} u*n(t^n) = " << max_u_n_val_old << " , l^{inf} u*n(t^{n+1}) = "
                      << max_u_n_val_new << reset << std::endl;
        }

        std::cout << "Number of interface points is " << counter_interface_pts << std::endl;

        l1_normal_interface_status /= counter_interface_pts;

        std::cout << "-----------------------> The l1 error of u*n over the INTERFACE, at time t = " << tot_time
                  << " is " << l1_normal_interface_status << reset << std::endl;

        std::cout << bold << green << "-----------------------> The linf error of u*n over the INTERFACE, at time t = "
                  << tot_time << " is " << max_u_n_val_abs << reset << std::endl;

        std::cout << "-----------------------> The L1 error of u*n over the INTERFACE, at time t = " << tot_time
                  << " is " << L1_normal_interface_status << reset << std::endl;

        std::cout << bold << green << "---> The linf error of (phi^{n+1}-phi{n})/dt over the INTERFACE, at time t = "
                  << tot_time << " is " << diff_in_time_interface << reset << std::endl;








// ----------------- CHECKING GOAL QUANTITIES FOR t = t^{n+1} ------------------



        check = l1_normal_interface_status;
/// DA AGGIUNGERE UNA VOLTA SISTEMATO IL CODICE

//if(check < 1e-8 )
//{
//    std::cout<<" check = "<<check<<" , STOP!"<<std::endl;
//    return 0;
//}





        T mass_fin = 0., area_fin = 0.;
        T centre_mass_x = 0., centre_mass_y = 0.;
        T l1_divergence_error_fin = 0., l2_divergence_error_fin = 0.;
        T linf_divergence_error_fin = 0.;
        T perimeter = 0.;
        T L1_divergence_error_fin = 0.;


        size_t counter_interface_pts_fin = 0.0;
//
//------------ Updating Parametric interface
        para_curve_cont.updating_parametric_interface(msh_i, ls_cell, l2proj, avg, l2proj_para, disc);

        if (filter) {
            plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(msh_i, int_refsteps, para_curve_cont, time_step);
            para_curve_cont.make_smooth_filter_curvature();
        }

//        para_curve_cont.plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(msh, n_int);

        T L1_normal_interface_para = 0.0;
        T linf_u_n_para = 0.0;
        T max_u_n_val_para = 0.0;
        T l1_normal_interface_para = 0.0;
        size_t counter_interface_pts_para = 0;
        T max_curvature = 0;
        T min_curvature = 1e+6;
        plot_curvature_normal_vs_curv_abscisse(msh_i, ls_cell, degree_curve, int_refsteps, 1);
        plot_curvature_normal_vs_curv_abscisse_PARAMETRIC(msh_i, degree_curve, int_refsteps, 1, degree_curvature,
                                                          para_curve_cont);

//        check_goal_quantities_final_para( msh_i ,ls_cell, para_curve_tmp , u_projected, perimeter, d_a,  area_fin, centre_mass_x ,   centre_mass_y , degree_FEM , mass_fin , degree_velocity , l1_divergence_error_fin , l2_divergence_error_fin , linf_divergence_error_fin , radius , L1_divergence_error_fin ,  time_step ,rise_vel0 , rise_vel1 ,flux_interface, counter_interface_pts_fin,degree_curve,int_refsteps, L1_normal_interface_para,linf_u_n_para,max_u_n_val_para,l1_normal_interface_para,counter_interface_pts_para);

        check_goal_quantities_final_para_no_plot(msh_i, ls_cell, para_curve_cont, u_projected, perimeter, d_a, area_fin,
                                                 centre_mass_x, centre_mass_y, degree_FEM, mass_fin, degree_velocity,
                                                 l1_divergence_error_fin, l2_divergence_error_fin,
                                                 linf_divergence_error_fin, radius, L1_divergence_error_fin, 1,
                                                 rise_vel0, rise_vel1, flux_interface, counter_interface_pts_fin,
                                                 degree_curve, int_refsteps, L1_normal_interface_para, linf_u_n_para,
                                                 max_u_n_val_para, l1_normal_interface_para, counter_interface_pts_para,
                                                 max_curvature, min_curvature);

        std::cout << "PARA-FLUX at the INTERFACE, at time " << tot_time << " is " << flux_interface << std::endl;
//        std::cout<<"PARA-L1(un) at the INTERFACE, at time "<< tot_time <<" is " << L1_normal_interface_para <<std::endl;

        l1_normal_interface_para /= counter_interface_pts_para;
//        std::cout<<"PARA-l1(un) at the INTERFACE, at time "<< tot_time <<" is " << l1_normal_interface_para <<std::endl;
//        std::cout<<"PARA-linf(un) at the INTERFACE, at time "<< tot_time <<" is " << linf_u_n_para <<std::endl;
//        std::cout<<"PARA-linf(un) SIGNED at the INTERFACE, at time "<< tot_time <<" is " << max_u_n_val_para <<std::endl;

        l1_err_u_n_time_para.push_back(l1_normal_interface_para);
        linf_err_u_n_time_para.push_back(linf_u_n_para);
        L1_err_u_n_time_para.push_back(L1_normal_interface_para);
        max_val_u_n_time_para.push_back(max_u_n_val_para);

//        check_goal_quantities_final( msh_i , ls_cell , u_projected, perimeter, d_a,  area_fin, centre_mass_x ,   centre_mass_y , degree_FEM , mass_fin , degree_velocity , l1_divergence_error_fin , l2_divergence_error_fin , linf_divergence_error_fin , radius , L1_divergence_error_fin ,  time_step ,rise_vel0 , rise_vel1 ,flux_interface,counter_interface_pts_fin,degree_curve,int_refsteps);
//

        testing_level_set_max_min(msh, level_set_function, time_step, min_max_vec);


        l1_divergence_error_fin /= counter_interface_pts_fin;


        std::cout << "The l1 error of the PARA_CONT CURVATURE at the INTERFACE, at time " << tot_time << " is "
                  << l1_divergence_error_fin << std::endl;

        std::cout << bold << green << "The linf error of the PARA_CONT CURVATURE at the INTERFACE, at time " << tot_time
                  << " is " << linf_divergence_error_fin << reset << std::endl;

        std::cout << "The L1 error of the PARA_CONT CURVATURE at the INTERFACE, at time " << tot_time << " is "
                  << L1_divergence_error_fin << std::endl;








//        std::cout<<"The PERIMETER, at time "<< tot_time <<" is " << perimeter << ", Initial PERIMETER =  "<<perimeter_initial<<std::endl;

//        std::cout<<"NORMALISED DIFFERENCE PERIMETER, at time "<< tot_time <<" is " << (perimeter - perimeter_initial)/perimeter_initial <<std::endl;

        d_a = sqrt(4.0 * area_fin / M_PI);

        std::cout << "The CIRCULARITY (OLD), at time " << tot_time << " is " << M_PI * d_a / perimeter << std::endl;
        std::cout << "The CIRCULARITY, at time " << tot_time << " is "
                  << 4.0 * M_PI * area_fin / (perimeter * perimeter) << std::endl;
        std::cout << "Area at time step: " << tot_time << " is " << area_fin << std::endl;
//        std::cout << "Internal mass at time step: "<<tot_time<<" is "<< mass_fin << std::endl;

        std::cout << bold << yellow << "NORMALISED Difference in AREA AT TIME " << tot_time << " IS "
                  << (area_fin - initial_area) / initial_area << reset << std::endl;
//        std::cout << "NORMALISED Difference in INTERNAL MASS AT TIME "<<tot_time<<" IS "<< (std::abs(mass_fin - initial_mass))/(std::abs( initial_mass )) << std::endl;
        std::cout << "CENTRE OF MASS at time step: " << tot_time << " is " << " ( " << centre_mass_x / area_fin << " , "
                  << centre_mass_y / area_fin << " ). " << std::endl;
        std::cout << "TRANSLATION OF THE CENTRE OF MASS at time step: " << tot_time << " is " << " ( "
                  << centre_mass_x / area_fin - centre_mass_x_inital / initial_area << " , "
                  << centre_mass_y / area_fin - centre_mass_y_inital / initial_area << " ). " << std::endl;
        R_phi = sqrt(area_fin / M_PI);
        std::cout << "Abs error over expected radius = " << std::abs(R_phi - radius) << '\n' << std::endl;




// Eccentricity (good only for elliptic solutions)
        T radius_max = 1.0 / min_curvature;
        T radius_min = 1.0 / max_curvature;

        T eccentricity = sqrt(1.0 - radius_min * radius_min / (radius_max * radius_max));

        std::cout << bold << green << "Eccentricity = " << eccentricity << reset << std::endl;
        eccentricity_vec.push_back(eccentricity);

        time_vec.push_back(tot_time);
        area_time.push_back(area_fin);
        l1_err_u_n_time.push_back(l1_normal_interface_status);
        linf_err_u_n_time.push_back(max_u_n_val_abs);
        L1_err_u_n_time.push_back(L1_normal_interface_status);

        linf_der_time_interface.push_back(diff_in_time_interface);

        max_val_u_n_time.push_back(max_u_n_val);
        l1_err_curvature_time.push_back(l1_divergence_error_fin);
        linf_err_curvature_time.push_back(linf_divergence_error_fin);


        circularity_time2.push_back(4.0 * M_PI * area_fin / (perimeter * perimeter));
        circularity_time.push_back(M_PI * d_a / perimeter);
        perimeter_time.push_back(perimeter);
        centre_mass_err_time.push_back(std::make_pair(centre_mass_x / area_fin, centre_mass_y / area_fin));

        flux_interface_time.push_back(flux_interface);
        rise_velocity_time.push_back(std::make_pair(rise_vel0 / area_fin, rise_vel1 / area_fin));


        if (tot_time >= final_time) {
            std::cout << "Final time T = " << tot_time << " achieved. Simulation ended at iteration = " << time_step
                      << std::endl;
            std::cout << "Tot amount HHO routines (fixed-point scheme), N = " << time_step << std::endl;
            break;
        }
        tc_iteration.toc();
        std::cout << "Time for this iteration = " << tc_iteration << std::endl;
        plotting_in_time_new(time_vec, area_time, l1_err_u_n_time, linf_err_u_n_time, max_val_u_n_time,
                             l1_err_curvature_time, linf_err_curvature_time, dt_M, min_max_vec, flux_interface_time,
                             rise_velocity_time, centre_mass_err_time, perimeter_time, circularity_time2,
                             circularity_ref, perim_ref, area_ref, radius,
                             L1_err_u_n_time, l1_err_u_n_time_para, linf_err_u_n_time_para, L1_err_u_n_time_para,
                             max_val_u_n_time_para, linf_der_time_interface, eccentricity_vec);
    } // End of the temporal loop








    std::cout << "Tot amount transport problem sub-routines tilde{N} = " << tot_amount_transport_routine << std::endl;


    std::cout << "FINAL TIME IS t = " << tot_time << std::endl;

    tc_tot.toc();
    std::cout << "Simulation machine time t = " << tc_tot << std::endl;

    return 0;
}

#endif





// Interface Stokes Problem: Fictitious domain
// Couette problem & M-shaped domain - comparison with HDG
// Ongoing work --> maybe use code of cuthho_stokes
#if 0

int main(int argc, char **argv) {
    using RealType = double;

    size_t degree = 0;
    size_t int_refsteps = 4;

    bool dump_debug = false;
    bool solve_interface = false;
    bool solve_fictdom = false;
    bool agglomeration = false;

    mesh_init_params <RealType> mip;
    mip.Nx = 5;
    mip.Ny = 5;

/* k <deg>:     method degree
     * M <num>:     number of cells in x direction
     * N <num>:     number of cells in y direction
     * r <num>:     number of interface refinement steps
     *
     * i:           solve interface problem
     * f:           solve fictitious domain problem
     *
     * D:           use node displacement to solve bad cuts (default)
     * A:           use agglomeration to solve bad cuts
     *
     * d:           dump debug data
     */

    int ch;
    while ((ch = getopt(argc, argv, "k:M:N:r:ifDAd")) != -1) {
        switch (ch) {
            case 'k':
                degree = atoi(optarg);
                break;

            case 'M':
                mip.Nx = atoi(optarg);
                break;

            case 'N':
                mip.Ny = atoi(optarg);
                break;

            case 'r':
                int_refsteps = atoi(optarg);
                break;

            case 'i':
                solve_interface = true;
                break;

            case 'f':
                solve_fictdom = true;
                break;

            case 'D':
                agglomeration = false;
                break;

            case 'A':
                agglomeration = true;
                break;

            case 'd':
                dump_debug = true;
                break;

            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }

    argc -= optind;
    argv += optind;


    timecounter tc;

/************** BUILD MESH **************/
    tc.tic();
    cuthho_poly_mesh <RealType> msh(mip);
    tc.toc();
    std::cout << bold << yellow << "Mesh generation: " << tc << " seconds" << reset << std::endl;
/************** LEVEL SET FUNCTION **************/
//    RealType radius = 1.0/3.0;
//    auto level_set_function = circle_level_set<RealType>(radius, 0.5, 0.5);
//     auto level_set_function = line_level_set<RealType>(0.5);
//     auto level_set_function = flower_level_set<RealType>(0.31, 0.5, 0.5, 4, 0.04);
    RealType eps = 0.005; // 0.25-1e-10; // 0.1;
    RealType epsBndry = 1e-8; //  0.15;
    RealType pos_sides = 0.25; // 0.0;
    auto level_set_function = m_shaped_level_set<RealType>(eps, epsBndry, pos_sides);


//    RealType R = 0.35;
//    auto level_set_function = rotated_square<RealType>(0.5 , 0.5 , R );
// RealType epsBndry  = -2.0*1e-2 ;
// auto level_set_function = square_level_set<RealType>(0.75-epsBndry,0.25+epsBndry,0.25+epsBndry,0.75-epsBndry);

//    RealType radius_a , radius_b,radius ;
//    RealType x_centre = 0.5;
//    RealType y_centre = 0.5;
//
//    radius_a = 1.0/6.0;
//    radius_b = 1.0/3.0;
//    radius = radius_a;
//
//    std::cout<<"Couette immersed domain: R1 = "<<radius_a<<", R2 = "<<radius_b<<std::endl;
//    auto level_set_function = couette_level_set<RealType>( radius_a, radius_b, x_centre, y_centre);




/************** DO cutHHO MESH PROCESSING **************/

    tc.tic();
    detect_node_position(msh, level_set_function);
    detect_cut_faces(msh, level_set_function);

    if (agglomeration) {
        detect_cut_cells(msh, level_set_function);
        detect_cell_agglo_set(msh, level_set_function);
        make_neighbors_info_cartesian(msh);
// make_neighbors_info(msh);
        refine_interface(msh, level_set_function, int_refsteps);
        make_agglomeration(msh, level_set_function);
    } else {
        move_nodes(msh, level_set_function);
        detect_cut_faces(msh, level_set_function); //do it again to update intersection points
        detect_cut_cells(msh, level_set_function);
        refine_interface(msh, level_set_function, int_refsteps);
    }


    tc.toc();
    std::cout << bold << yellow << "cutHHO-specific mesh preprocessing: " << tc << " seconds" << reset << std::endl;

    if (dump_debug) {
        dump_mesh(msh);
        output_mesh_info(msh, level_set_function);
    }

    output_mesh_info(msh, level_set_function);

// auto test_case = make_test_case_stokes_1(msh, level_set_function);
//    auto test_case = make_test_case_stokes_2(msh, level_set_function);
// auto test_case = make_test_case_kink_velocity(msh, .... );
    auto test_case = make_test_case_M_shaped(msh, 1.0, level_set_function);

//auto test_case = make_test_case_kink_velocity2(msh, level_set_function);

//    checkNormal(msh,level_set_function,2.0);
    auto cauchyFormulation = false;
    auto method = make_sym_gradrec_stokes_interface_method(msh, 1.0, 0.0, test_case, cauchyFormulation);


    if (solve_interface)
        run_cuthho_interface(msh, degree, method, test_case);

    if (solve_fictdom)
        run_cuthho_fictdom(msh, degree, test_case);


    return 0;
}

#endif
