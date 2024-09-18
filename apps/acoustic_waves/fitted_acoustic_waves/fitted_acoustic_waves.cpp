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


// All these defines are going to be delete. However for the case of unfitted implementation it is not required.
#define fancy_stabilization_Q
#define compute_energy_Q
#define spatial_errors_Q
#define InhomogeneousQ
#define contrast 10.0
#define n_terms 100


class TAnalyticalFunction
{
    public:
    
    /// Enumerate defining the function type
    enum EFunctionType { EFunctionNonPolynomial = 0, EFunctionQuadraticInTime = 1, EFunctionQuadraticInSpace = 2, EFunctionQuadraticInSpaceTime = 3, EFunctionInhomogeneousInSpace = 4};
    
    
    TAnalyticalFunction(){
        m_function_type = EFunctionNonPolynomial;
    }
    
    ~TAnalyticalFunction(){
        
    }
    
    void SetFunctionType(EFunctionType function_type){
        m_function_type = function_type;
    }

    std::function<double(const typename poly_mesh<double>::point_type& )> Evaluate_u(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                            return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                        return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::cos(std::sqrt(2.0)*M_PI*t);
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                        return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
                        };
                }
                break;
            case EFunctionQuadraticInSpaceTime:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                        return t*t*(1 - pt.x())*pt.x()*(1 - pt.y())*pt.y();
                        };
                }
                break;
            case EFunctionInhomogeneousInSpace:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                        
                        size_t n = n_terms;
                        double x,y,c1,c2;
                        c1 = contrast;
                        c2 = 1.0;
                        x = pt.x();
                        y = pt.y();

                        double p = 0.0;
                        double c = 1.0;
                        for (int k = 0; k <= n; k++) {

                            if (x < 0.5) {
                                double plvp, plvm;
                                plvp = (1.0/100.0)*exp(-(20.0*((k+x-c1*t)-0.2))*(20.0*((k+x-c1*t)-0.2)));
                                plvm = (1.0/100.0)*exp(-(20.0*((k-x-c1*t)-0.2))*(20.0*((k-x-c1*t)-0.2)));
                                p+=c*(plvp-plvm);
                            }else{
                                double pl;
                                pl = (1.0/100.0)*exp(-(20.0*(((c1/c2)*(x-0.5)+0.5+k-c1*t)-0.2))*(20.0*(((c1/c2)*(x-0.5)+0.5+k-c1*t)-0.2)));
                                p+=((2.0*c1)/(c2+c1))*c*pl;
                            }
                            c*=(c2-c1)/(c2+c1);
                        }
                        return p;
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename poly_mesh<double>::point_type& pt) -> double {
                        return 0;
                    };
            }
                break;
        }
        
    }
    
    std::function<double(const typename poly_mesh<double>::point_type& )> Evaluate_v(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                            return std::cos(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                            return -(std::sqrt(2.0)*M_PI*(1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::sin(std::sqrt(2.0)*M_PI*t));
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                        return 2.0*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
                        };
                }
                break;
            case EFunctionQuadraticInSpaceTime:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                        return 2.0*t*(1 - pt.x())*pt.x()*(1 - pt.y())*pt.y();
                        };
                }
                break;
            case EFunctionInhomogeneousInSpace:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                        
                            size_t n = n_terms;
                            double x,y,c1,c2;
                            c1 = contrast;
                            c2 = 1.0;
                            x = pt.x();
                            y = pt.y();

                            double v = 0.0;
                            double c = 1.0;
                            for (int k = 0; k <= n; k++) {

                                if (x < 0.5) {
                                    double vlvp, vlvm;
                                    vlvp = (8.0*c1)* exp(-(20.0*((k+x-c1*t)-0.2))*(20.0*((k+x-c1*t)-0.2)))*((k+x-c1*t)-0.2);
                                    vlvm = (8.0*c1)* exp(-(20.0*((k-x-c1*t)-0.2))*(20.0*((k-x-c1*t)-0.2)))*((k-x-c1*t)-0.2);
                                    v+=c*(vlvp-vlvm);
                                }else{
                                    double vl;
                                    vl = (8.0*c1)*exp(-(20.0*(((c1/c2)*(x-0.5)+0.5+k-c1*t)-0.2))*(20.0*(((c1/c2)*(x-0.5)+0.5+k-c1*t)-0.2)))*(((c1/c2)*(x-0.5)+0.5+k-c1*t)-0.2);
                                    v+=((2.0*c1)/(c2+c1))*c*vl;
                                }
                                c*=(c2-c1)/(c2+c1);
                            }
                            return v;
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename poly_mesh<double>::point_type& pt) -> double {
                        return 0;
                    };
            }
                break;
        }
        
    }
    
    std::function<double(const typename poly_mesh<double>::point_type& )> Evaluate_a(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                            return -std::sqrt(2.0) * M_PI * std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                            return -2*M_PI*M_PI*(1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::cos(std::sqrt(2)*M_PI*t);
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [](const typename poly_mesh<double>::point_type& pt) -> double {
                        return 2.0*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
                        };
                }
                break;
            case EFunctionQuadraticInSpaceTime:
                {
                    return [](const typename poly_mesh<double>::point_type& pt) -> double {
                        return 2.0*(1 - pt.x())*pt.x()*(1 - pt.y())*pt.y();
                        };
                }
                break;
            case EFunctionInhomogeneousInSpace:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                        
                            size_t n = n_terms;
                            double x,y,c1,c2;
                            c1 = contrast;
                            c2 = 1.0;
                            x = pt.x();
                            y = pt.y();

                            double a = 0.0;
                            double c = 1.0;
                            for (int k = 0; k <= n; k++) {

                                if (x < 0.5) {
                                    double alvp, alvm, xip, xim;
                                    xip=((k+x-c1*t)-0.2);
                                    alvp = (8.0*c1*c1)* exp(-(20.0*((k+x-c1*t)-0.2))*(20.0*((k+x-c1*t)-0.2)))*(800.0*xip*xip-1.0);
                                    
                                    xim=((k-x-c1*t)-0.2);
                                    alvm = (8.0*c1*c1)* exp(-(20.0*((k-x-c1*t)-0.2))*(20.0*((k-x-c1*t)-0.2)))*(800.0*xim*xim-1.0);
                                    a+=c*(alvp-alvm);
                                }else{
                                    double al, xi;
                                    xi= (((c1/c2)*(x-0.5)+0.5+k-c1*t)-0.2);
                                    al = (8.0*c1*c1)*exp(-(20.0*(((c1/c2)*(x-0.5)+0.5+k-c1*t)-0.2))*(20.0*(((c1/c2)*(x-0.5)+0.5+k-c1*t)-0.2)))*(800.0*xi*xi-1.0);
                                    a+=((2.0*c1)/(c2+c1))*c*al;
                                }
                                c*=(c2-c1)/(c2+c1);
                            }
                        return a;
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename poly_mesh<double>::point_type& pt) -> double {
                        return 0;
                    };
            }
                break;
        }
        
    }
    
    std::function<double(const typename poly_mesh<double>::point_type& )> Evaluate_f(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [](const typename poly_mesh<double>::point_type& pt) -> double {
                            return 0;
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                            double x,y,f;
                            x = pt.x();
                            y = pt.y();
                            f = 2*(x - x*x + y - M_PI*M_PI*(-1 + x)*x*(-1 + y)*y - y*y)*std::cos(std::sqrt(2.0)*M_PI*t);
                            return f;
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                        return 2.0*(1.0 + M_PI*M_PI*t*t)*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
                        };
                }
                break;
            case EFunctionQuadraticInSpaceTime:
                 {
                     return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                            double x,y,f;
                            x = pt.x();
                            y = pt.y();
                            f = 2.0*((-1.0 + x)*x*(-1.0 + y)*y + t*t*(x - x*x + y - y*y));
                            return f;
                         };
                 }
                 break;
            case EFunctionInhomogeneousInSpace:
                {
                    return [](const typename poly_mesh<double>::point_type& pt) -> double {
                            return 0;
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename poly_mesh<double>::point_type& pt) -> double {
                        return 0;
                    };
            }
                break;
        }
        
    }
    
    
    std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> Evaluate_q(double & t){
        
        switch (m_function_type) {
            case EFunctionNonPolynomial:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> std::vector<double> {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            std::vector<double> flux(2);
                            flux[0] = (std::sin(std::sqrt(2)*M_PI*t)*std::cos(M_PI*x)*std::sin(M_PI*y))/std::sqrt(2.0);
                            flux[1] = (std::sin(std::sqrt(2)*M_PI*t)*std::sin(M_PI*x)*std::cos(M_PI*y))/std::sqrt(2.0);
                            flux[0] *=-1.0;
                            flux[1] *=-1.0;
                            return flux;
                        };
                }
                break;
            case EFunctionQuadraticInSpace:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> std::vector<double> {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            std::vector<double> flux(2);
                            flux[0] = (1 - x)*(1 - y)*y*std::cos(std::sqrt(2.0)*M_PI*t) - x*(1 - y)*y*std::cos(std::sqrt(2.0)*M_PI*t);
                            flux[1] = (1 - x)*x*(1 - y)*std::cos(std::sqrt(2.0)*M_PI*t) - (1 - x)*x*y*std::cos(std::sqrt(2.0)*M_PI*t);
                            flux[0] *=-1.0;
                            flux[1] *=-1.0;
                            return flux;
                        };
                }
                break;
            case EFunctionQuadraticInTime:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> std::vector<double> {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            std::vector<double> flux(2);
                            flux[0] = M_PI*t*t*std::cos(M_PI*x)*std::sin(M_PI*y);
                            flux[1] = M_PI*t*t*std::sin(M_PI*x)*std::cos(M_PI*y);
                            flux[0] *=-1.0;
                            flux[1] *=-1.0;
                            return flux;
                        };
                }
                break;
            case EFunctionQuadraticInSpaceTime:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> std::vector<double> {
                            double x,y;
                            x = pt.x();
                            y = pt.y();
                            std::vector<double> flux(2);
                            flux[0] = t*t*(1 - x)*(1 - y)*y - t*t*x*(1 - y)*y;
                            flux[1] = t*t*(1 - x)*x*(1 - y) - t*t*(1 - x)*x*y;
                            flux[0] *=-1.0;
                            flux[1] *=-1.0;
                            return flux;
                        };
                }
                break;
            case EFunctionInhomogeneousInSpace:
                {
                    return [&t](const typename poly_mesh<double>::point_type& pt) -> std::vector<double> {
                        
                            size_t n = n_terms;
                            double x,y,c1,c2;
                            c1 = contrast;
                            c2 = 1.0;
                            x = pt.x();
                            y = pt.y();

                            double qx = 0.0;
                            double c = 1.0;
                            for (int k = 0; k <= n; k++) {

                                if (x < 0.5) {
                                    double qxlvp, qxlvm;
                                    qxlvp = -(8.0)* exp(-(20.0*((k+x-c1*t)-0.2))*(20.0*((k+x-c1*t)-0.2)))*((k+x-c1*t)-0.2);
                                    qxlvm = (8.0)* exp(-(20.0*((k-x-c1*t)-0.2))*(20.0*((k-x-c1*t)-0.2)))*((k-x-c1*t)-0.2);
                                    qx+=c*(qxlvp-qxlvm);
                                }else{
                                    double qxl;
                                    qxl = -(8.0*(c1/c2))*exp(-(20.0*(((c1/c2)*(x-0.5)+0.5+k-c1*t)-0.2))*(20.0*(((c1/c2)*(x-0.5)+0.5+k-c1*t)-0.2)))*(((c1/c2)*(x-0.5)+0.5+k-c1*t)-0.2);
                                    qx+=((2.0*c1)/(c2+c1))*c*qxl;
                                }
                                c*=(c2-c1)/(c2+c1);
                            }
                        
                            std::vector<double> flux(2);
                            flux[0] = qx;
                            flux[1] = 0.0;
                            flux[0] *=-1.0;
                            flux[1] *=-1.0;
                            return flux;
                        };
                }
                break;
            default:
            {
                std::cout << " Function not implemented " << std::endl;
                return [](const typename poly_mesh<double>::point_type& pt) -> std::vector<double> {
                        std::vector<double> f;
                        return f;
                    };
            }
                break;
        }
        
    }
    
    private:
    
    EFunctionType m_function_type;
  
};

class TDIRKSchemes
{
    public:
    
    static void DIRKSchemesSS(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c){
        
        a = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
        b = Matrix<double, Dynamic, 1>::Zero(s, 1);
        c = Matrix<double, Dynamic, 1>::Zero(s, 1);
        
        // stiffly-accurate, L-stable, DIRK methods
        switch (s) {
            case 1:
                {
                    a(0,0) = 0.5;
                    b(0,0) = 1.0;
                    c(0,0) = 0.5;
                }
                break;
            case 2:
                {
                    a(0,0) = 1.0/3.0;
                    a(1,0) = 3.0/4.0;
                    a(1,1) = 1.0/4.0;
                    
                    b(0,0) = 3.0/4.0;
                    b(1,0) = 1.0/4.0;
                    
                    c(0,0) = 1.0/3.0;
                    c(1,0) = 1.0;
                    
                }
                break;
            case 3:
                {
                    
                    a(0,0) = 1.0;
                    a(1,0) = -1.0/12.0;
                    a(1,1) = 5.0/12.0;
                    a(2,0) = 0.0;
                    a(2,1) = 3.0/4.0;
                    a(2,2) = 1.0/4.0;
                    
                    b(0,0) = 0.0;
                    b(1,0) = 3.0/4.0;
                    b(2,0) = 3.0/4.0;
                    
                    c(0,0) = 1.0;
                    c(1,0) = 1.0/3.0;
                    c(2,0) = 1.0;
                    
                }
                break;
            default:
            {
                std::cout << "Error:: Method not implemented." << std::endl;
            }
                break;
        }
        
    }
    
    static void DIRKSchemesS(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c){
        
        a = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
        b = Matrix<double, Dynamic, 1>::Zero(s, 1);
        c = Matrix<double, Dynamic, 1>::Zero(s, 1);
        
        // Optimized diagonally implicit Runge-Kutta schemes for time-dependent wave propagation problems
        switch (s) {
            case 1:
                {
                    a(0,0) = 0.5;
                    b(0,0) = 1.0;
                    c(0,0) = 0.5;
                }
                break;
            case 2:
                {
                    a(0,0) = 0.780078125000;
                    a(1,0) = -0.595072059507;
                    a(1,1) = 0.797536029754;
                    
                    b(0,0) = 0.515112081837;
                    b(1,0) = 0.484887918163;
                    
                    c(0,0) = 0.780078125;
                    c(1,0) = 0.202463970246;
                    
                }
                break;
            case 3:
                {
                    
                    a(0,0) = 0.956262348020;
                    a(1,0) = -0.676995728936;
                    a(1,1) = 1.092920059741;
                    a(2,0) = 4.171447220367;
                    a(2,1) = -5.550750999686;
                    a(2,2) = 1.189651889660;
                    
                    b(0,0) = 0.228230955547;
                    b(1,0) = 0.706961029433;
                    b(2,0) = 0.064808015020;
                    
                    c(0,0) = 0.956262348020;
                    c(1,0) = 0.415924330804;
                    c(2,0) = -0.189651889660;
                    
                }
                break;
            default:
            {
                std::cout << "Error:: Method not implemented." << std::endl;
            }
                break;
        }
        
    }
    
    static void SDIRKSchemes(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c){
        
        // Optimized diagonally implicit Runge-Kutta schemes for time-dependent wave propagation problems
        a = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
        b = Matrix<double, Dynamic, 1>::Zero(s, 1);
        c = Matrix<double, Dynamic, 1>::Zero(s, 1);
        
        switch (s) {
            case 1:
                {
                    a(0,0) = 0.5;
                    b(0,0) = 1.0;
                    c(0,0) = 0.5;
                }
                break;
            case 2:
                {
                    a(0,0) = 0.25;
                    a(1,0) = 0.5;
                    a(1,1) = 0.25;
                    
                    b(0,0) = 0.5;
                    b(1,0) = 0.5;
                    
                    c(0,0) = 0.25;
                    c(1,0) = 0.75;
                    
                }
                break;
            case 3:
                {
                    
                    a(0,0) = 0.333361958530;
                    a(1,0) = 0.203587820425;
                    a(1,1) = 0.333361958530;
                    a(2,0) = 1.169622123547;
                    a(2,1) = -0.632757519576;
                    a(2,2) = 0.333361958530;
                    
                    b(0,0) = 0.887593107230;
                    b(1,0) = -0.318926384159;
                    b(2,0) = 0.431333276929;
                    
                    c(0,0) = 0.333361958530;
                    c(1,0) = 0.536949778955;
                    c(2,0) = 0.87022656250;
                    
//                    a(0,0) = 1.068579021302;
//                    a(1,0) = -0.568579021302;
//                    a(1,1) = 1.068579021302;
//                    a(2,0) = 2.137158042603;
//                    a(2,1) = -3.274316085207;
//                    a(2,2) = 1.068579021302;
//
//                    b(0,0) = 0.128886400516;
//                    b(1,0) = 0.742227198969;
//                    b(2,0) = 0.128886400516;
//
//                    c(0,0) = 1.068579021302;
//                    c(1,0) = 0.5;
//                    c(2,0) = -0.068579021302;
                    
                }
                break;
                
            case 4:
                {
                    
                    a(0,0) = 0.333361958530;
                    a(1,0) = 0.203587820425;
                    a(1,1) = 0.333361958530;
                    a(2,0) = 1.169622123547;
                    a(2,1) = -0.632757519576;
                    a(2,2) = 0.333361958530;
                    
                    b(0,0) = 0.887593107230;
                    b(1,0) = -0.318926384159;
                    b(2,0) = 0.431333276929;
                    
                    c(0,0) = 0.333361958530;
                    c(1,0) = 0.536949778955;
                    c(2,0) = 0.87022656250;
                    
                    // The three-stage, fourth-order, SDIRK family of methods
                    // A-stable
//                    a(0,0) = 1.068579021302;
//                    a(1,0) = -0.568579021302;
//                    a(1,1) = 1.068579021302;
//                    a(2,0) = 2.137158042603;
//                    a(2,1) = -3.274316085207;
//                    a(2,2) = 1.068579021302;
//
//                    b(0,0) = 0.128886400516;
//                    b(1,0) = 0.742227198969;
//                    b(2,0) = 0.128886400516;
//
//                    c(0,0) = 1.068579021302;
//                    c(1,0) = 0.5;
//                    c(2,0) = -0.068579021302;
                    
                }
                break;
            default:
            {
                std::cout << "Error:: Method not implemented." << std::endl;
            }
                break;
        }
        
    }
    
    private:
  
};


class TSSPRKSchemes
{
    public:
    
    static void OSSPRKSS(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double,Dynamic, Dynamic> &b){
        
        // Optimal Strong-Stability-Preserving Runge-Kutta Time Discretizations for Discontinuous Galerkin Methods
        switch (s) {
            case 1:
                {
                    a = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
                    b = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
                    a(0,0) = 1.0;
                    b(0,0) = 1.0;
                }
                break;
            case 2:
                {
                    // DG book (Alexandre)
                    a = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
                    b = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
                    a(0,0) = 1.0;
                    a(1,0) = 1.0/2.0;
                    a(1,1) = 1.0/2.0;

                    b(0,0) = 1.0;
                    b(1,1) = 1.0/2.0;
                                        
                }
                break;
            case 3:
                {
                    a = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
                    b = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
                    a(0,0) = 1.0;
                    a(1,0) = 0.087353119859156;
                    a(1,1) = 0.912646880140844;
                    a(2,0) = 0.344956917166841;
                    a(2,1) = 0.0;
                    a(2,2) = 0.655043082833159;

                    b(0,0) = 0.528005024856522;
                    b(1,0) = 0.0;
                    b(1,1) = 0.481882138633993;
                    b(2,0) = 0.022826837460491;
                    b(2,1) = 0.0;
                    b(2,2) = 0.345866039233415;
        
                    
                }
                break;
            case 4:
                {
                    a = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
                    b = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
                    a(0,0) = 1.0;
                    a(1,0) = 0.522361915162541;
                    a(1,1) = 0.477638084837459;
                    a(2,0) = 0.368530939472566;
                    a(2,1) = 0.0;
                    a(2,2) = 0.631469060527434;
                    a(3,0) = 0.334082932462285;
                    a(3,1) = 0.006966183666289;
                    a(3,2) = 0.0;
                    a(3,3) = 0.658950883871426;
                    
                    b(0,0) = 0.594057152884440;
                    b(1,0) = 0.0;
                    b(1,1) = 0.283744320787718;
                    b(2,0) = 0.000000038023030;
                    b(2,1) = 0.0;
                    b(2,2) = 0.375128712231540;
                    b(3,0) = 0.116941419604231;
                    b(3,1) = 0.004138311235266;
                    b(3,2) = 0.0;
                    b(3,3) = 0.391454485963345;
                }
                break;
            case 5:
            {
                a = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
                b = Matrix<double, Dynamic, Dynamic>::Zero(s, s);
                a(0,0) = 1.0;
                a(1,0) = 0.261216512493821;
                a(1,1) = 0.738783487506179;
                a(2,0) = 0.623613752757655;
                a(2,1) = 0.0;
                a(2,2) = 0.376386247242345;
                a(3,0) = 0.444745181201454;
                a(3,1) = 0.120932584902288;
                a(3,2) = 0.0;
                a(3,3) = 0.434322233896258;
                a(4,0) = 0.213357715199957;
                a(4,1) = 0.209928473023448;
                a(4,2) = 0.063353148180384;
                a(4,3) = 0.0;
                a(4,4) = 0.513360663596212;
                
                b(0,0) = 0.605491839566400;
                b(1,0) = 0.0;
                b(1,1) = 0.447327372891397;
                b(2,0) = 0.000000844149769;
                b(2,1) = 0.0;
                b(2,2) = 0.227898801230261;
                b(3,0) = 0.002856233144485;
                b(3,1) = 0.073223693296006;
                b(3,2) = 0.0;
                b(3,3) = 0.262978568366434;
                b(4,0) = 0.002362549760441;
                b(4,1) = 0.127109977308333;
                b(4,2) = 0.038359814234063;
                b(4,3) = 0.0;
                b(4,4) = 0.310835692561898;
            }
                break;
            default:
            {
                std::cout << "Error:: Method not implemented." << std::endl;
            }
                break;
        }
        
    }
  
};

class TSSPRKHHOAnalyses
{
    
    public:
    
    TSSPRKHHOAnalyses(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg, size_t n_f_dof){
        
        
        size_t n_c_dof = Kg.rows() - n_f_dof;
        // Building blocks
        m_Mc = Mg.block(0, 0, n_c_dof, n_c_dof);
        m_Kc = Kg.block(0, 0, n_c_dof, n_c_dof);
        m_Kcf = Kg.block(0, n_c_dof, n_c_dof, n_f_dof);
        m_Kfc = Kg.block(n_c_dof,0, n_f_dof, n_c_dof);
        m_Sff = Kg.block(n_c_dof,n_c_dof, n_f_dof, n_f_dof);
        m_Fc = Fg.block(0, 0, n_c_dof, 1);
        
        DecomposeMassTerm();
        DecomposeFaceTerm();
    }
    
    void DecomposeMassTerm(){
        m_analysis_c.analyzePattern(m_Mc);
        m_analysis_c.factorize(m_Mc);
    }
    
    void DecomposeFaceTerm(){
        m_analysis_f.analyzePattern(m_Sff);
        m_analysis_f.factorize(m_Sff);
    }
    
    SparseLU<SparseMatrix<double>> & CellsAnalysis(){
        return m_analysis_c;
    }
    
    SparseLU<SparseMatrix<double>> & FacesAnalysis(){
        return m_analysis_f;
    }
    
    SparseMatrix<double> & Mc(){
        return m_Mc;
    }

    SparseMatrix<double> & Kc(){
        return m_Kc;
    }
    
    SparseMatrix<double> & Kcf(){
        return m_Kcf;
    }
    
    SparseMatrix<double> & Kfc(){
        return m_Kfc;
    }
    
    SparseMatrix<double> & Sff(){
        return m_Sff;
    }
    
    Matrix<double, Dynamic, 1> & Fc(){
        return m_Fc;
    }
    
    private:

    SparseMatrix<double> m_Mc;
    SparseMatrix<double> m_Kc;
    SparseMatrix<double> m_Kcf;
    SparseMatrix<double> m_Kfc;
    SparseMatrix<double> m_Sff;
    Matrix<double, Dynamic, 1> m_Fc;
    SparseLU<SparseMatrix<double>> m_analysis_c;
    SparseLU<SparseMatrix<double>> m_analysis_f;
    
};

class TDIRKHHOAnalyses
{
    
    public:
    
    TDIRKHHOAnalyses(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg){
        
        m_Mg = Mg;
        m_Kg = Kg;
        m_Fg = Fg;
        m_scale = 0.0;
    }
    
    TDIRKHHOAnalyses(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg, double scale){
        
        m_Mg = Mg;
        m_Kg = Kg;
        m_Fg = Fg;
        m_scale = scale;
        DecomposeMatrix();
    }
    
    void DecomposeMatrix(){
        SparseMatrix<double> K = m_Mg + m_scale * m_Kg;
        m_analysis.analyzePattern(K);
        m_analysis.factorize(K);
    }
    
    SparseLU<SparseMatrix<double>> & DirkAnalysis(){
        return m_analysis;
    }
    
    SparseMatrix<double> & Mg(){
        return m_Mg;
    }
    
    SparseMatrix<double> & Kg(){
        return m_Kg;
    }
    
    Matrix<double, Dynamic, 1> & Fg(){
        return m_Fg;
    }
    
    void SetScale(double & scale){
        m_scale = scale;
    }
    
    void SetFg(Matrix<double, Dynamic, 1> & Fg){
        m_Fg = Fg;
    }
    
    private:

    double m_scale;
    SparseMatrix<double> m_Mg;
    SparseMatrix<double> m_Kg;
    Matrix<double, Dynamic, 1> m_Fg;
    SparseLU<SparseMatrix<double>> m_analysis;
    
};

double ComputeEnergySecondOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, Matrix<double, Dynamic, 1> & p, Matrix<double, Dynamic, 1> & v);

double ComputeEnergyFirstOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
                               std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun);

void RenderSiloFileTwoFields(std::string silo_file_name, size_t it, poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun, bool cell_centered_Q = true);

void RenderSiloFileScalarField(std::string silo_file_name, size_t it, poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
                               std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun, bool cell_centered_Q = true);

void ComputeL2ErrorSingleField(poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, Matrix<double, Dynamic, 1> & x_dof,
std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
                             std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun);

void ComputeL2ErrorTwoFields(poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun);

void ComputeKGFG(SparseMatrix<double> & Kg, Matrix<double, Dynamic, 1> & Fg, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions);

void ComputeFG(Matrix<double, Dynamic, 1> & Fg, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions);

void ComputeInhomogeneousKGFG(SparseMatrix<double> & Kg, Matrix<double, Dynamic, 1> & Fg, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions);

void ComputeKGFGSecondOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions);

void DIRKStep(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, SparseMatrix<double> & Mg, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n);

void IRKWeight(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k, double dt, double a);

void DIRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, TDIRKHHOAnalyses & dirk_an, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, bool is_sdirk_Q = false);

void IRKWeightOpt(TDIRKHHOAnalyses & dirk_an, Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k, double dt, double a, bool is_sdirk_Q = false);

void SSPRKStep(int s, Matrix<double, Dynamic, Dynamic> &alpha, Matrix<double, Dynamic, Dynamic> &beta, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, SparseMatrix<double> & Mg, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof);

void ERKWeight(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_n_dof, Matrix<double, Dynamic, 1> & x_dof, size_t n_f_dof, double dt, double a, double b);

void SSPRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &alpha, Matrix<double, Dynamic, Dynamic> &beta, TSSPRKHHOAnalyses & ssprk_an, double & dt, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof);

void ERKWeightOpt(TSSPRKHHOAnalyses & ssprk_an,  Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof, double dt, double a, double b);

void FaceDoFUpdate(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_dof, size_t n_f_dof);

void IHHOSecondOrder(int argc, char **argv);

void HHOFirstOrderExample(int argc, char **argv);

void IHHOFirstOrder(int argc, char **argv);

void EHHOFirstOrder(int argc, char **argv);

void HeterogeneousIHHOFirstOrder(int argc, char **argv);

void HeterogeneousEHHOFirstOrder(int argc, char **argv);

void HeterogeneousIHHOSecondOrder(int argc, char **argv);

int main(int argc, char **argv)
{
        
    // fitted methods
//    HHOFirstOrderExample(argc, argv);
//    HeterogeneousEHHOFirstOrder(argc, argv);
//    HeterogeneousIHHOFirstOrder(argc, argv);
//    HeterogeneousIHHOSecondOrder(argc, argv);
//    EHHOFirstOrder(argc, argv);
//    IHHOFirstOrder(argc, argv);
    IHHOSecondOrder(argc, argv);

    return 0;
}

void HeterogeneousEHHOFirstOrder(int argc, char **argv){
    
    bool render_silo_files_Q = true;
    bool render_zonal_vars_Q = false;
    using RealType = double;
    size_t k_degree = 0;
    size_t n_divs   = 0;
    
    // Final time value 0.5
    std::vector<size_t> nt_v = {5,10,20,40,80,160,320,640,1280,2560,5120,10240};
    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625,0.00078125,0.000390625,0.0001953125,0.00009765625,0.000048828125};
    
    int tref = 9;
    int s = 5;
    
    size_t nt       = nt_v[tref];
    RealType dt     = dt_v[tref];
    RealType ti = 0.0;
    
    int opt;
    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
    {
        switch(opt)
        {
            case 'k':
            {
                k_degree = atoi(optarg);
            }
                break;
            case 'l':
            {
                n_divs = atoi(optarg);
            }
                break;
            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }
    
    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
    std::cout << bold << red << "l : " << n_divs << reset << std::endl;
    std::cout << bold << red << "nt : " << nt << reset << std::endl;
    std::cout << bold << red << "dt : " << dt << reset << std::endl;

    // The mesh in ProtoN seems like is always 2D
     mesh_init_params<RealType> mip;
     mip.Nx = 10;
     mip.Ny = 1;
     mip.max_y = 0.1;
    
    for (size_t i = 0; i < n_divs; i++) {
        mip.Nx *= 2;
    }
    
    timecounter tc;
    
    // Building the cartesian mesh
    tc.tic();
    poly_mesh<RealType> msh(mip);
    tc.toc();

    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;

    // Projection of initial data
    
    
    // Creating HHO approximation spaces and corresponding linear operator
    hho_degree_info hho_di(k_degree,k_degree);
    
    // Solving a HDG/HHO mixed problem
    auto assembler = make_assembler(msh, hho_di, true); // another assemble version
    auto mass_assembler = make_assembler(msh, hho_di, true); // another assemble version

    
    auto is_dirichlet = [&](const typename poly_mesh<RealType>::face_type& fc) -> bool {
        return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
    };
    auto num_all_faces = msh.faces.size();
    auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
    auto num_other_faces = num_all_faces - num_dirichlet_faces;
    auto fbs = face_basis<poly_mesh<RealType>,RealType>::size(hho_di.face_degree());
    size_t n_f_dof = num_other_faces * fbs;
    
    TAnalyticalFunction functions;
    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionInhomogeneousInSpace);
    RealType t = ti;
    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
    
    tc.tic();
    // Projecting initial state(flux and velocity)
    Matrix<RealType, Dynamic, 1> x_dof_n_m = assembler.RHS; // probably not needed
    {
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            { /// global mass
                
                auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
                auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
                auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
        
                auto n_rows = reconstruction_operator.second.rows();
                auto n_cols = reconstruction_operator.second.cols();
                
                auto n_s_rows = stabilization_operator.rows();
                auto n_s_cols = stabilization_operator.cols();
                
                Matrix<RealType, Dynamic, Dynamic> M_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
                Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
                Matrix<RealType, Dynamic, Dynamic> M_q = R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols);
                
                M_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) = M_q;
                
                Matrix<RealType, Dynamic, Dynamic> v_mass_operator = make_cell_mass_matrix(msh, cell, hho_di);
                size_t cell_dof_c;
                {
                    cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                    cell_dof_c = cell_basis.size();
                }
                Matrix<RealType, Dynamic, Dynamic> M_v = v_mass_operator.block(0, 0, cell_dof_c, cell_dof_c);
                M_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, cell_dof_c, cell_dof_c) = M_v;
                
                // Compossing objects
                Matrix<RealType, Dynamic, 1> f_loc = Matrix<RealType, Dynamic, 1>::Zero(n_rows, 1);
                mass_assembler.assemble_mixed(msh, cell, M_operator, f_loc, exact_vel_sol_fun);

            }
        
            auto mass_flux_operator = make_flux_cell_mass_matrix(msh, cell, hho_di);
            Matrix<RealType, Dynamic, 1> f_q = make_vector_variable_rhs(msh, cell, hho_di.cell_degree()+1, exact_flux_sol_fun);
            Matrix<RealType, Dynamic, 1> dof_q = mass_flux_operator.llt().solve(f_q);
            Matrix<RealType, Dynamic, 1> dof_v = project_function(msh, cell, hho_di, exact_vel_sol_fun);
            
            size_t cell_dof;
            size_t cell_rec_dof;
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                cell_dof = cell_basis.size();

            }
            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
            cell_rec_dof = cell_basis.size()-1;
            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof), 0, cell_rec_dof, 1) = dof_q;
            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof)+cell_rec_dof, 0, cell_dof, 1) = dof_v.block(0, 0, cell_dof, 1);
            cell_i++;
            // Initial projection of face unknows is not implemented yet
        }
    }
    mass_assembler.finalize();
    tc.toc();
    
    size_t it = 0;
    if (render_silo_files_Q) {
        std::string silo_file_name = "e_scalar_inhomogeneous_wave_";
        RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
    }
    
#ifdef compute_energy_Q
    Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
    tc.tic();
    energy_h_values(0,0) = 0.0;
    energy_h_values(0,1) = 1.0;
    RealType energy_h0 = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun);
    tc.toc();
    std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
#endif

    Matrix<double, Dynamic, Dynamic> alpha;
    Matrix<double, Dynamic, Dynamic> beta;
    TSSPRKSchemes::OSSPRKSS(s, alpha, beta);

    Matrix<double, Dynamic, 1> x_dof_n;
    // Transient problem
    bool optimized_Q = true;
    
    if (optimized_Q) {
        
        double tv = 0.0;
        SparseMatrix<double> Kg;
        Matrix<double, Dynamic, 1> Fg;
        tc.tic();
        ComputeKGFG(Kg, Fg, msh, hho_di, assembler, tv, functions);
        TSSPRKHHOAnalyses ssprk_an(Kg,Fg,mass_assembler.LHS,n_f_dof);
        tc.toc();
        std::cout << bold << cyan << "Linear transformations completed: " << tc << " seconds" << reset << std::endl;
        
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            
            RealType tn = dt*(it-1)+ti;
            tc.tic();
            SSPRKStepOpt(s, alpha, beta, ssprk_an, dt, x_dof_n_m, x_dof_n, n_f_dof);
            tc.toc();
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            
            RealType t = tn + dt;
            auto exact_vel_sol_fun = functions.Evaluate_v(t);
            auto exact_flux_sol_fun = functions.Evaluate_q(t);
            if (render_silo_files_Q) {
                std::string silo_file_name = "e_scalar_inhomogeneous_wave_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
            }
            
#ifdef compute_energy_Q
            {
                RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
                energy_h_values(it,0) = t;
                energy_h_values(it,1) = energy_h/energy_h0;
            }
#endif
            
            if(it == nt){
                
                std::string silo_file_name = "e_scalar_inhomogeneous_wave_at_tf_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
                
                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
                RealType h =  sqrt((1.0/mip.Nx)*(1.0/mip.Nx)+(1.0/mip.Ny)*(1.0/mip.Ny));
                std::cout << green << "dt size = " << std::endl << dt << std::endl;
                std::cout << green << "dt/h ratio = " << std::endl << dt/h << std::endl;
                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
            }
            x_dof_n_m = x_dof_n;
        }
    }else{
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            
            RealType tn = dt*(it-1)+ti;
            tc.tic();
            SSPRKStep(s, alpha, beta, msh, hho_di, assembler, mass_assembler.LHS, tn, dt, functions, x_dof_n_m, x_dof_n, n_f_dof);
            tc.toc();
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            
            RealType t = tn + dt;
            auto exact_vel_sol_fun = functions.Evaluate_v(t);
            auto exact_flux_sol_fun = functions.Evaluate_q(t);
            if (render_silo_files_Q) {
                std::string silo_file_name = "e_scalar_inhomogeneous_wave_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun,render_zonal_vars_Q);
            }
            
#ifdef compute_energy_Q
            {
                RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
                energy_h_values(it,0) = t;
                energy_h_values(it,1) = energy_h/energy_h0;
            }
#endif
            
            if(it == nt){
                
                std::string silo_file_name = "e_scalar_inhomogeneous_wave_at_tf_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun,render_zonal_vars_Q);
                
                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
                RealType h =  sqrt((1.0/mip.Nx)*(1.0/mip.Nx)+(1.0/mip.Ny)*(1.0/mip.Ny));
                std::cout << green << "dt size = " << std::endl << dt << std::endl;
                std::cout << green << "dt/h ratio = " << std::endl << dt/h << std::endl;
                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
            }
            x_dof_n_m = x_dof_n;
        }
    }
    
#ifdef compute_energy_Q
    std::ofstream energy_file ("EHHO_energy.txt");
    if (energy_file.is_open())
    {
      energy_file << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
      energy_file << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
      energy_file.close();
    }
#endif
    
}

void EHHOFirstOrder(int argc, char **argv){
    
    bool render_silo_files_Q = true;
    bool render_zonal_vars_Q = false;
    using RealType = double;
    size_t k_degree = 0;
    size_t n_divs   = 0;
    
    // Final time value 1.0
    std::vector<size_t> nt_v = {10,20,40,80,160,320,640,1280,2560,5120,10240,20480,40960,81920}; //13
    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625,0.00078125,0.000390625,0.0001953125,0.00009765625,0.000048828125,0.0000244140625,0.00001220703125};
    
    int tref = 5 + 0;
    int s = 5; // order s - 1
    
//    size_t nt       = nt_v[tref];
//    RealType dt     = dt_v[tref];
//    RealType ti = 0.0;
    
    int opt;
    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
    {
        switch(opt)
        {
            case 'k':
            {
                k_degree = atoi(optarg);
            }
                break;
            case 'l':
            {
                n_divs = atoi(optarg);
            }
                break;
            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }
    
    tref = n_divs + 5;
    size_t nt       = nt_v[tref];
    RealType dt     = dt_v[tref];
    RealType ti = 0.0;
    
    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
    std::cout << bold << red << "l : " << n_divs << reset << std::endl;
    std::cout << bold << red << "nt : " << nt << reset << std::endl;
    std::cout << bold << red << "dt : " << dt << reset << std::endl;

    // The mesh in ProtoN seems like is always 2D
     mesh_init_params<RealType> mip;
     mip.Nx = 1;
     mip.Ny = 1;
    
    for (size_t i = 0; i < n_divs; i++) {
        mip.Nx *= 2;
        mip.Ny *= 2;
    }
    
    timecounter tc;
    
    // Building the cartesian mesh
    tc.tic();
    poly_mesh<RealType> msh(mip);
    tc.toc();

    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;

    // Projection of initial data
    
    
    // Creating HHO approximation spaces and corresponding linear operator
    hho_degree_info hho_di(k_degree,k_degree);
    
    // Solving a HDG/HHO mixed problem
    auto assembler = make_assembler(msh, hho_di, true); // another assemble version
    auto mass_assembler = make_assembler(msh, hho_di, true); // another assemble version

    
    auto is_dirichlet = [&](const typename poly_mesh<RealType>::face_type& fc) -> bool {
        return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
    };
    auto num_all_faces = msh.faces.size();
    auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
    auto num_other_faces = num_all_faces - num_dirichlet_faces;
    auto fbs = face_basis<poly_mesh<RealType>,RealType>::size(hho_di.face_degree());
    size_t n_f_dof = num_other_faces * fbs;
    
    TAnalyticalFunction functions;
    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionNonPolynomial);
    RealType t = 0.0;
    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
    
    tc.tic();
    // Projecting initial state(flux and velocity)
    Matrix<RealType, Dynamic, 1> x_dof_n_m = assembler.RHS; // probably not needed
    {
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            { /// global mass
                
                auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
                auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
                auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
        
                auto n_rows = reconstruction_operator.second.rows();
                auto n_cols = reconstruction_operator.second.cols();
                
                auto n_s_rows = stabilization_operator.rows();
                auto n_s_cols = stabilization_operator.cols();
                
                Matrix<RealType, Dynamic, Dynamic> M_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
                Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
                Matrix<RealType, Dynamic, Dynamic> M_q = R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols);
                
                M_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) = M_q;
                
                Matrix<RealType, Dynamic, Dynamic> v_mass_operator = make_cell_mass_matrix(msh, cell, hho_di);
                size_t cell_dof_c;
                {
                    cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                    cell_dof_c = cell_basis.size();
                }
                Matrix<RealType, Dynamic, Dynamic> M_v = v_mass_operator.block(0, 0, cell_dof_c, cell_dof_c);
                M_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, cell_dof_c, cell_dof_c) = M_v;
                
                // Compossing objects
                Matrix<RealType, Dynamic, 1> f_loc = Matrix<RealType, Dynamic, 1>::Zero(n_rows, 1);
                mass_assembler.assemble_mixed(msh, cell, M_operator, f_loc, exact_vel_sol_fun);

            }
        
            auto mass_flux_operator = make_flux_cell_mass_matrix(msh, cell, hho_di);
            Matrix<RealType, Dynamic, 1> f_q = make_vector_variable_rhs(msh, cell, hho_di.cell_degree()+1, exact_flux_sol_fun);
            Matrix<RealType, Dynamic, 1> dof_q = mass_flux_operator.llt().solve(f_q);
            Matrix<RealType, Dynamic, 1> dof_v = project_function(msh, cell, hho_di, exact_vel_sol_fun);
            
            size_t cell_dof;
            size_t cell_rec_dof;
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                cell_dof = cell_basis.size();

            }
            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
            cell_rec_dof = cell_basis.size()-1;
            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof), 0, cell_rec_dof, 1) = dof_q;
            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof)+cell_rec_dof, 0, cell_dof, 1) = dof_v.block(0, 0, cell_dof, 1);
            cell_i++;
            // Initial projection of face unknows is not implemented yet
        }
    }
    mass_assembler.finalize();
    tc.toc();
    
    size_t it = 0;
    if (render_silo_files_Q) {
        std::string silo_file_name = "scalar_wave_";
        RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
    }
    
#ifdef compute_energy_Q
    Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
    tc.tic();
    energy_h_values(0,0) = 0.0;
    energy_h_values(0,1) = 1.0;
    RealType energy_h0 = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun);
    tc.toc();
    std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
#endif

    Matrix<double, Dynamic, Dynamic> alpha;
    Matrix<double, Dynamic, Dynamic> beta;
    TSSPRKSchemes::OSSPRKSS(s, alpha, beta);

    Matrix<double, Dynamic, 1> x_dof_n;
    // Transient problem
    bool optimized_Q = true;
    
    if (optimized_Q) {
        
        double tv = 0.0;
        SparseMatrix<double> Kg;
        Matrix<double, Dynamic, 1> Fg;
        tc.tic();
        ComputeKGFG(Kg, Fg, msh, hho_di, assembler, tv, functions);
        TSSPRKHHOAnalyses ssprk_an(Kg,Fg,mass_assembler.LHS,n_f_dof);
        tc.toc();
        std::cout << bold << cyan << "Linear transformations completed: " << tc << " seconds" << reset << std::endl;
        
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            
            RealType tn = dt*(it-1)+ti;
            tc.tic();
            SSPRKStepOpt(s, alpha, beta, ssprk_an, dt, x_dof_n_m, x_dof_n, n_f_dof);
            tc.toc();
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            
            RealType t = tn + dt;
            auto exact_vel_sol_fun = functions.Evaluate_v(t);
            auto exact_flux_sol_fun = functions.Evaluate_q(t);
            if (render_silo_files_Q) {
                std::string silo_file_name = "scalar_wave_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
            }
            
#ifdef compute_energy_Q
            {
                RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
                energy_h_values(it,0) = t;
                energy_h_values(it,1) = energy_h/energy_h0;
            }
#endif
            
            if(it == nt){
                
                std::string silo_file_name = "scalar_wave_at_tf_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
                
                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
                RealType h =  sqrt((1.0/mip.Nx)*(1.0/mip.Nx)+(1.0/mip.Ny)*(1.0/mip.Ny));
                std::cout << green << "dt size = " << std::endl << dt << std::endl;
                std::cout << green << "dt/h ratio = " << std::endl << dt/h << std::endl;
                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
            }
            x_dof_n_m = x_dof_n;
        }
    }else{
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            
            RealType tn = dt*(it-1)+ti;
            tc.tic();
            SSPRKStep(s, alpha, beta, msh, hho_di, assembler, mass_assembler.LHS, tn, dt, functions, x_dof_n_m, x_dof_n, n_f_dof);
            tc.toc();
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            
            RealType t = tn + dt;
            auto exact_vel_sol_fun = functions.Evaluate_v(t);
            auto exact_flux_sol_fun = functions.Evaluate_q(t);
            if (render_silo_files_Q) {
                std::string silo_file_name = "scalar_wave_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
            }
            
#ifdef compute_energy_Q
            {
                RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
                energy_h_values(it,0) = t;
                energy_h_values(it,1) = energy_h/energy_h0;
            }
#endif
            
            if(it == nt){
                
                std::string silo_file_name = "scalar_wave_at_tf_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
                
                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
                RealType h =  sqrt((1.0/mip.Nx)*(1.0/mip.Nx)+(1.0/mip.Ny)*(1.0/mip.Ny));
                std::cout << green << "dt size = " << std::endl << dt << std::endl;
                std::cout << green << "dt/h ratio = " << std::endl << dt/h << std::endl;
                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
            }
            x_dof_n_m = x_dof_n;
        }
    }
    
#ifdef compute_energy_Q
    std::ofstream energy_file ("EHHO_energy.txt");
    if (energy_file.is_open())
    {
      energy_file << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
      energy_file << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
      energy_file.close();
    }
#endif

}

void SSPRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &alpha, Matrix<double, Dynamic, Dynamic> &beta, TSSPRKHHOAnalyses & ssprk_an, double & dt, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof){
    
    size_t n_dof = x_dof_n_m.rows();
    Matrix<double, Dynamic, Dynamic> ys = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s+1);

    Matrix<double, Dynamic, 1> yn, ysi, yj;
    ys.block(0, 0, n_dof, 1) = x_dof_n_m;
    for (int i = 0; i < s; i++) {

        ysi = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
        for (int j = 0; j <= i; j++) {
            yn = ys.block(0, j, n_dof, 1);
            ERKWeightOpt(ssprk_an, yn, yj, n_f_dof, dt, alpha(i,j), beta(i,j));
            ysi += yj;
        }
        ys.block(0, i+1, n_dof, 1) = ysi;
    }
    
    x_dof_n = ys.block(0, s, n_dof, 1);

}

void SSPRKStep(int s, Matrix<double, Dynamic, Dynamic> &alpha, Matrix<double, Dynamic, Dynamic> &beta, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, SparseMatrix<double> & Mg, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof){
    
    size_t n_dof = x_dof_n_m.rows();
    Matrix<double, Dynamic, Dynamic> ys = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s+1);
    SparseMatrix<double> Kg;
    Matrix<double, Dynamic, 1> Fg;

    double t = tn + 0.0*dt;
    ComputeKGFG(Kg, Fg, msh, hho_di, assembler, t, functions);
    
    Matrix<double, Dynamic, 1> yn, ysi, yj;
    ys.block(0, 0, n_dof, 1) = x_dof_n_m;
    for (int i = 0; i < s; i++) {

        ysi = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
        for (int j = 0; j <= i; j++) {
            yn = ys.block(0, j, n_dof, 1);
            ERKWeight(Kg, Fg, Mg, yn, yj, n_f_dof, dt, alpha(i,j), beta(i,j));
            ysi += yj;
        }
        ys.block(0, i+1, n_dof, 1) = ysi;
    }
    
    x_dof_n = ys.block(0, s, n_dof, 1);

}

void HeterogeneousIHHOFirstOrder(int argc, char **argv){
    
    bool render_silo_files_Q = true;
    bool render_zonal_vars_Q = false;
    using RealType = double;
    size_t k_degree = 0;
    size_t n_divs   = 0;
        
    // Final time value 0.5
    std::vector<size_t> nt_v = {5,10,20,40,80,160,320,640,1280,2560,5120};
    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625,0.00078125,0.000390625,0.0001953125,0.00009765625};
    
    int tref = 10;
    int s = 3;
    
    size_t nt       = nt_v[tref];
    RealType dt     = dt_v[tref];
    RealType ti = 0.0;
    
    int opt;
    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
    {
        switch(opt)
        {
            case 'k':
            {
                k_degree = atoi(optarg);
            }
                break;
            case 'l':
            {
                n_divs = atoi(optarg);
            }
                break;
            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }
    
    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
    std::cout << bold << red << "l : " << n_divs << reset << std::endl;
    std::cout << bold << red << "nt : " << nt << reset << std::endl;
    std::cout << bold << red << "dt : " << dt << reset << std::endl;

    // The mesh in ProtoN seems like is always 2D
     mesh_init_params<RealType> mip;
     mip.Nx = 10;
     mip.Ny = 2;
     mip.max_y = 0.1;
    for (size_t i = 0; i < n_divs; i++) {
        mip.Nx *= 2;
    }
    
    timecounter tc;
    
    // Building the cartesian mesh
    tc.tic();
    poly_mesh<RealType> msh(mip);
    tc.toc();

    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;

    // Projection of initial data
    
    
    // Creating HHO approximation spaces and corresponding linear operator
    hho_degree_info hho_di(k_degree,k_degree);
    
    // Solving a HDG/HHO mixed problem
    auto assembler = make_assembler(msh, hho_di, true); // another assemble version
    auto mass_assembler = make_assembler(msh, hho_di, true); // another assemble version
    
    TAnalyticalFunction functions;
    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionInhomogeneousInSpace);
    RealType t = ti;
    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
    
    tc.tic();
    // Projecting initial state(flux and velocity)
    Matrix<RealType, Dynamic, 1> x_dof_n_m = assembler.RHS;
    {
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            { /// global mass
                
                double c = 1.0;
                auto bar = barycenter(msh, cell);
                double x = bar.x();
                if (x < 0.5) {
                    c *= contrast;
                }
                
                auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
                auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
                auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
        
                auto n_rows = reconstruction_operator.second.rows();
                auto n_cols = reconstruction_operator.second.cols();
                
                auto n_s_rows = stabilization_operator.rows();
                auto n_s_cols = stabilization_operator.cols();
                
                Matrix<RealType, Dynamic, Dynamic> M_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
                Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
                Matrix<RealType, Dynamic, Dynamic> M_q = R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols);
                
                M_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) = M_q;
                
                Matrix<RealType, Dynamic, Dynamic> v_mass_operator = make_cell_mass_matrix(msh, cell, hho_di);
                size_t cell_dof_c;
                {
                    cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                    cell_dof_c = cell_basis.size();
                }
                Matrix<RealType, Dynamic, Dynamic> M_v = v_mass_operator.block(0, 0, cell_dof_c, cell_dof_c);
                M_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, cell_dof_c, cell_dof_c) = (1.0/(c*c))*M_v;
                
                // Compossing objects
                Matrix<RealType, Dynamic, 1> f_loc = Matrix<RealType, Dynamic, 1>::Zero(n_rows, 1);
                mass_assembler.assemble_mixed(msh, cell, M_operator, f_loc, exact_vel_sol_fun);

            }
            
            
            auto mass_flux_operator = make_flux_cell_mass_matrix(msh, cell, hho_di);
            Matrix<RealType, Dynamic, 1> f_q = make_vector_variable_rhs(msh, cell, hho_di.cell_degree()+1, exact_flux_sol_fun);
            Matrix<RealType, Dynamic, 1> dof_q = mass_flux_operator.llt().solve(f_q);
            Matrix<RealType, Dynamic, 1> dof_v = project_function(msh, cell, hho_di, exact_vel_sol_fun);
            
            size_t cell_dof;
            size_t cell_rec_dof;
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                cell_dof = cell_basis.size();

            }
            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
            cell_rec_dof = cell_basis.size()-1;
            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof), 0, cell_rec_dof, 1) = dof_q;
            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof)+cell_rec_dof, 0, cell_dof, 1) = dof_v.block(0, 0, cell_dof, 1);
            cell_i++;
        }
    }
    mass_assembler.finalize();
    
//    // face update
//    if(0){
//        auto is_dirichlet = [&](const typename poly_mesh<RealType>::face_type& fc) -> bool {
//            return fc.is_boundary && fc.bndtype == boundary::DIRICHLET;
//        };
//        auto num_all_faces = msh.faces.size();
//        auto num_dirichlet_faces = std::count_if(msh.faces.begin(), msh.faces.end(), is_dirichlet);
//        auto num_other_faces = num_all_faces - num_dirichlet_faces;
//        auto fbs = face_basis<poly_mesh<RealType>,RealType>::size(hho_di.face_degree());
//        size_t n_f_dof = num_other_faces * fbs;
//
//        SparseMatrix<double> Kg;
//        Matrix<double, Dynamic, 1> Fg;
//        ComputeKGFG(Kg, Fg, msh, hho_di, assembler, t, functions);
//        TSSPRKHHOAnalyses ssprk_an(Kg, Fg, mass_assembler.LHS, n_f_dof);
//
//        size_t n_c_dof = x_dof_n_m.rows() - n_f_dof;
//        Matrix<double, Dynamic, 1> x_c_dof = x_dof_n_m.block(0, 0, n_c_dof, 1);
//        Matrix<double, Dynamic, 1> x_f_dof = x_dof_n_m.block(n_c_dof, 0, n_f_dof, 1);
//
//        // Faces update (last state)
//        {
//            Matrix<double, Dynamic, 1> RHSf = ssprk_an.Kfc()*x_c_dof;
//            x_f_dof = -ssprk_an.FacesAnalysis().solve(RHSf);
//        }
//        x_dof_n_m.block(n_c_dof, 0, n_f_dof, 1) = x_f_dof;
//
//    }
    
    tc.toc();
    std::cout << bold << cyan << "Initial state computed: " << tc << " seconds" << reset << std::endl;
    size_t it = 0;
    if (render_silo_files_Q) {
        std::string silo_file_name = "scalar_inhomogeneous_wave_";
        RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
    }

#ifdef compute_energy_Q
    Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
    tc.tic();
    energy_h_values(0,0) = 0.0;
    energy_h_values(0,1) = 1.0;
    RealType energy_h0 = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun);
    tc.toc();
    std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
#endif

    // Solving a HDG/HHO mixed problem
    Matrix<RealType, Dynamic, Dynamic> a;
    Matrix<RealType, Dynamic, 1> b;
    Matrix<RealType, Dynamic, 1> c;
    
    // DIRK(s) schemes
    bool is_sdirk_Q = true;
    
    if (is_sdirk_Q) {
        TDIRKSchemes::SDIRKSchemes(s, a, b, c);
    }else{
        TDIRKSchemes::DIRKSchemesSS(s, a, b, c);
    }
    
    Matrix<double, Dynamic, 1> x_dof_n;
    bool optimized_Q = true;
    
    if (optimized_Q) {
        // Transient problem
        
#ifdef InhomogeneousQ
        ComputeInhomogeneousKGFG(assembler.LHS, assembler.RHS, msh, hho_di, assembler, t, functions);
#else
        ComputeKGFG(assembler.LHS, assembler.RHS, msh, hho_di, assembler, t, functions); // Fixed boundary data
#endif
        TDIRKHHOAnalyses dirk_an(assembler.LHS,assembler.RHS,mass_assembler.LHS);
        
        if (is_sdirk_Q) {
            double scale = a(0,0) * dt;
            dirk_an.SetScale(scale);
            dirk_an.DecomposeMatrix();
        }
        
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            RealType tn = dt*(it-1)+ti;
            tc.tic();
            DIRKStepOpt(s, a, b, c, msh, hho_di, assembler, dirk_an, tn, dt, functions, x_dof_n_m, x_dof_n, is_sdirk_Q);
            tc.toc();
            
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            RealType t = tn + dt;
            auto exact_vel_sol_fun = functions.Evaluate_v(t);
            auto exact_flux_sol_fun = functions.Evaluate_q(t);
            
            if (render_silo_files_Q) {
                std::string silo_file_name = "scalar_inhomogeneous_wave_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
            }
            
           
    #ifdef compute_energy_Q
                {
                    RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
                    energy_h_values(it,0) = t;
                    energy_h_values(it,1) = energy_h/energy_h0;
                }
    #endif
            
    #ifdef spatial_errors_Q
            if(it == nt){
                std::string silo_file_name = "scalar_inhomogeneous_wave_at_tf_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
                std::cout << green << "dt size = " << std::endl << dt << std::endl;
                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
            }
    #endif
            x_dof_n_m = x_dof_n;
        }
    }else{
        // Transient problem
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            
            RealType tn = dt*(it-1)+ti;
            tc.tic();
            DIRKStep(s, a, b, c, msh, hho_di, assembler, mass_assembler.LHS, tn, dt, functions, x_dof_n_m, x_dof_n);
            tc.toc();
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            RealType t = tn + dt;
            auto exact_vel_sol_fun = functions.Evaluate_v(t);
            auto exact_flux_sol_fun = functions.Evaluate_q(t);
            
            if (render_silo_files_Q) {
                std::string silo_file_name = "scalar_inhomogeneous_wave_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
            }
            
           
    #ifdef compute_energy_Q
                {
                    RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
                    energy_h_values(it,0) = t;
                    energy_h_values(it,1) = energy_h/energy_h0;
                }
    #endif
            
    #ifdef spatial_errors_Q
            if(it == nt){
                std::string silo_file_name = "scalar_inhomogeneous_wave_at_tf_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
                std::cout << green << "dt size = " << std::endl << dt << std::endl;
                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
            }
    #endif
            x_dof_n_m = x_dof_n;
        }
    }
    
#ifdef compute_energy_Q
    std::ofstream energy_file ("IHHO_energy.txt");
    if (energy_file.is_open())
    {
      energy_file << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
      energy_file << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
      energy_file.close();
    }
#endif
    
}

void IHHOFirstOrder(int argc, char **argv){
    
    bool render_silo_files_Q = true;
    bool render_zonal_vars_Q = false;
    using RealType = double;
    size_t k_degree = 0;
    size_t n_divs   = 0;
    
    // Final time value 1.0
    std::vector<size_t> nt_v = {10,20,40,80,160,320,640,1280,2560,5120,10240,20480};
    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625,0.00078125,0.000390625,0.0001953125,0.00009765625,0.00009765625/2};
    
    int tref = 5;
    int s = 2;
    
    size_t nt       = nt_v[tref];
    RealType dt     = dt_v[tref];
    RealType ti = 0.0;
    
    int opt;
    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
    {
        switch(opt)
        {
            case 'k':
            {
                k_degree = atoi(optarg);
            }
                break;
            case 'l':
            {
                n_divs = atoi(optarg); //3
            }
                break;
            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }
    
    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
    std::cout << bold << red << "l : " << n_divs << reset << std::endl;
    std::cout << bold << red << "nt : " << nt << reset << std::endl;
    std::cout << bold << red << "dt : " << dt << reset << std::endl;

     mesh_init_params<RealType> mip;
     mip.Nx = 1;
     mip.Ny = 1;
    
    for (size_t i = 0; i < n_divs; i++) {
        mip.Nx *= 2;
        mip.Ny *= 2;
    }
    
    timecounter tc;
    
    // Building the cartesian mesh
    tc.tic();
    poly_mesh<RealType> msh(mip);
    tc.toc();

    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;

    // Projection of initial data
    
    
    // Creating HHO approximation spaces and corresponding linear operator
    hho_degree_info hho_di(k_degree,k_degree);
    
    // Solving a HDG/HHO mixed problem
    auto assembler = make_assembler(msh, hho_di, true); // another assemble version
    auto mass_assembler = make_assembler(msh, hho_di, true); // another assemble version
    
    TAnalyticalFunction functions;
    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionNonPolynomial);
    RealType t = 0.0;
    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
    
    tc.tic();
    // Projecting initial state(flux and velocity)
    Matrix<RealType, Dynamic, 1> x_dof_n_m = assembler.RHS;
    {
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            { /// global mass
                
                auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
                auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
                auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
        
                auto n_rows = reconstruction_operator.second.rows();
                auto n_cols = reconstruction_operator.second.cols();
                
                auto n_s_rows = stabilization_operator.rows();
                auto n_s_cols = stabilization_operator.cols();
                
                Matrix<RealType, Dynamic, Dynamic> M_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
                Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
                Matrix<RealType, Dynamic, Dynamic> M_q = R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols);
                
                M_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) = M_q;
                
                Matrix<RealType, Dynamic, Dynamic> v_mass_operator = make_cell_mass_matrix(msh, cell, hho_di);
                size_t cell_dof_c;
                {
                    cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                    cell_dof_c = cell_basis.size();
                }
                Matrix<RealType, Dynamic, Dynamic> M_v = v_mass_operator.block(0, 0, cell_dof_c, cell_dof_c);
                M_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, cell_dof_c, cell_dof_c) = M_v;
                
                // Compossing objects
                Matrix<RealType, Dynamic, 1> f_loc = Matrix<RealType, Dynamic, 1>::Zero(n_rows, 1);
                mass_assembler.assemble_mixed(msh, cell, M_operator, f_loc, exact_vel_sol_fun);

            }
            
            
            auto mass_flux_operator = make_flux_cell_mass_matrix(msh, cell, hho_di);
            Matrix<RealType, Dynamic, 1> f_q = make_vector_variable_rhs(msh, cell, hho_di.cell_degree()+1, exact_flux_sol_fun);
            Matrix<RealType, Dynamic, 1> dof_q = mass_flux_operator.llt().solve(f_q);
            Matrix<RealType, Dynamic, 1> dof_v = project_function(msh, cell, hho_di, exact_vel_sol_fun);
            
            size_t cell_dof;
            size_t cell_rec_dof;
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                cell_dof = cell_basis.size();

            }
            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
            cell_rec_dof = cell_basis.size()-1;
            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof), 0, cell_rec_dof, 1) = dof_q;
            x_dof_n_m.block(cell_i*(cell_dof+cell_rec_dof)+cell_rec_dof, 0, cell_dof, 1) = dof_v.block(0, 0, cell_dof, 1);
            cell_i++;
        }
    }
    mass_assembler.finalize();
    tc.toc();
    std::cout << bold << cyan << "Initial state computed: " << tc << " seconds" << reset << std::endl;
    size_t it = 0;
    if (render_silo_files_Q) {
        std::string silo_file_name = "scalar_wave_";
        RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
    }
    
#ifdef compute_energy_Q
    Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
    tc.tic();
    energy_h_values(0,0) = 0.0;
    energy_h_values(0,1) = 1.0;
    RealType energy_h0 = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n_m, exact_vel_sol_fun, exact_flux_sol_fun);
    tc.toc();
    std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
#endif

    // Solving a HDG/HHO mixed problem
    Matrix<RealType, Dynamic, Dynamic> a;
    Matrix<RealType, Dynamic, 1> b;
    Matrix<RealType, Dynamic, 1> c;
    
    // DIRK(s) schemes
    bool is_sdirk_Q = true;
    
    if (is_sdirk_Q) {
        TDIRKSchemes::SDIRKSchemes(s, a, b, c);
    }else{
        TDIRKSchemes::DIRKSchemesSS(s, a, b, c);
    }
    
    Matrix<double, Dynamic, 1> x_dof_n;
    bool optimized_Q = false;
    
    if (optimized_Q) {
        // Transient problem
        
#ifdef InhomogeneousQ
        ComputeInhomogeneousKGFG(assembler.LHS, assembler.RHS, msh, hho_di, assembler, t, functions);
#else
        ComputeKGFG(assembler.LHS, assembler.RHS, msh, hho_di, assembler, t, functions);
#endif
        TDIRKHHOAnalyses dirk_an(assembler.LHS,assembler.RHS,mass_assembler.LHS);
        
        if (is_sdirk_Q) {
            double scale = a(0,0) * dt;
            dirk_an.SetScale(scale);
            dirk_an.DecomposeMatrix();
        }
        
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            RealType tn = dt*(it-1)+ti;
            tc.tic();
            DIRKStepOpt(s, a, b, c, msh, hho_di, assembler, dirk_an, tn, dt, functions, x_dof_n_m, x_dof_n, is_sdirk_Q);
            tc.toc();
            
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            RealType t = tn + dt;
            auto exact_vel_sol_fun = functions.Evaluate_v(t);
            auto exact_flux_sol_fun = functions.Evaluate_q(t);
            
            if (render_silo_files_Q) {
                std::string silo_file_name = "scalar_wave_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
            }
            
           
    #ifdef compute_energy_Q
                {
                    RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
                    energy_h_values(it,0) = t;
                    energy_h_values(it,1) = energy_h/energy_h0;
                }
    #endif
            
    #ifdef spatial_errors_Q
            if(it == nt){
                std::string silo_file_name = "scalar_wave_at_tf_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
                std::cout << green << "dt size = " << std::endl << dt << std::endl;
                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
            }
    #endif
            x_dof_n_m = x_dof_n;
        }
    }else{
        // Transient problem
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            
            RealType tn = dt*(it-1)+ti;
            tc.tic();
            DIRKStep(s, a, b, c, msh, hho_di, assembler, mass_assembler.LHS, tn, dt, functions, x_dof_n_m, x_dof_n);
            tc.toc();
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            RealType t = tn + dt;
            auto exact_vel_sol_fun = functions.Evaluate_v(t);
            auto exact_flux_sol_fun = functions.Evaluate_q(t);
            
            if (render_silo_files_Q) {
                std::string silo_file_name = "scalar_wave_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
            }
            
           
    #ifdef compute_energy_Q
                {
                    RealType energy_h = ComputeEnergyFirstOrder(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
                    energy_h_values(it,0) = t;
                    energy_h_values(it,1) = energy_h/energy_h0;
                }
    #endif
            
    #ifdef spatial_errors_Q
            if(it == nt){
                std::string silo_file_name = "scalar_wave_at_tf_";
                RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun, render_zonal_vars_Q);
                std::cout << green << "dt size = " << std::endl << dt << std::endl;
                std::cout << bold << cyan << "Reporting errors values : " << reset << std::endl;
                ComputeL2ErrorTwoFields(msh, hho_di, x_dof_n, exact_vel_sol_fun, exact_flux_sol_fun);
            }
    #endif
            x_dof_n_m = x_dof_n;
        }
    }
    
#ifdef compute_energy_Q
    std::ofstream energy_file ("IHHO_energy.txt");
    if (energy_file.is_open())
    {
      energy_file << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
      energy_file << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
      energy_file.close();
    }
#endif
    
}

void DIRKStep(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, SparseMatrix<double> & Mg, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n){
    
    size_t n_dof = x_dof_n_m.rows();
    Matrix<double, Dynamic, Dynamic> k = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s);
    SparseMatrix<double> Kg;
    Matrix<double, Dynamic, 1> Fg;
    
    double t;
    Matrix<double, Dynamic, 1> yn, ki;

    x_dof_n = x_dof_n_m;
    for (int i = 0; i < s; i++) {
        
        yn = x_dof_n_m;
        for (int j = 0; j < s - 1; j++) {
            yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
        }
        
        t = tn + c(i,0) * dt;
#ifdef InhomogeneousQ
        ComputeInhomogeneousKGFG(Kg, Fg, msh, hho_di, assembler, t, functions);
#else
        ComputeKGFG(Kg, Fg, msh, hho_di, assembler, t, functions);
#endif
    
        IRKWeight(Kg, Fg, Mg, yn, ki, dt, a(i,i));
        
        // Accumulated solution
        x_dof_n += dt*b(i,0)*ki;
        k.block(0, i, n_dof, 1) = ki;
    }
    
}


void ComputeInhomogeneousKGFG(SparseMatrix<double> & Kg, Matrix<double, Dynamic, 1> & Fg, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions){
    using RealType = double;
    

    auto exact_vel_sol_fun = functions.Evaluate_v(t);
    auto rhs_fun = functions.Evaluate_f(t);
    
    assembler.LHS *= 0.0;
    assembler.RHS *= 0.0;
    for (auto& cell : msh.cells)
    {
        
        double c = 1.0;
        auto bar = barycenter(msh, cell);
        double x = bar.x();
        if (x < 0.5) {
            c *= contrast;
        }
        
        auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
        auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
        auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif

        auto n_rows = reconstruction_operator.second.rows();
        auto n_cols = reconstruction_operator.second.cols();
        
        auto n_s_rows = stabilization_operator.rows();
        auto n_s_cols = stabilization_operator.cols();
        
        Matrix<RealType, Dynamic, Dynamic> S_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
        
        Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
        R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) *= 0.0;
        // Compossing objects
        Matrix<RealType, Dynamic, Dynamic> laplacian_loc = R_operator + (1.0/(c))*S_operator;
        Matrix<RealType, Dynamic, 1> f_loc = make_mixed_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
        assembler.assemble_mixed(msh, cell, laplacian_loc, f_loc, exact_vel_sol_fun);
    }
    assembler.finalize();
    Kg = assembler.LHS;
    Fg = assembler.RHS;
    
    
}

void ComputeKGFG(SparseMatrix<double> & Kg, Matrix<double, Dynamic, 1> & Fg, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions){
    using RealType = double;
    

    auto exact_vel_sol_fun = functions.Evaluate_v(t);
    auto rhs_fun = functions.Evaluate_f(t);
    
    assembler.LHS *= 0.0;
    assembler.RHS *= 0.0;
    for (auto& cell : msh.cells)
    {
        auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
        auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
        auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif

        auto n_rows = reconstruction_operator.second.rows();
        auto n_cols = reconstruction_operator.second.cols();
        
        auto n_s_rows = stabilization_operator.rows();
        auto n_s_cols = stabilization_operator.cols();
        
        Matrix<RealType, Dynamic, Dynamic> S_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
        
        Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
        R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) *= 0.0;
        // Compossing objects
        Matrix<RealType, Dynamic, Dynamic> laplacian_loc = R_operator + S_operator;
        Matrix<RealType, Dynamic, 1> f_loc = make_mixed_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
        assembler.assemble_mixed(msh, cell, laplacian_loc, f_loc, exact_vel_sol_fun);
    }
    assembler.finalize();
    Kg = assembler.LHS;
    Fg = assembler.RHS;
    
    
}

void ComputeFG(Matrix<double, Dynamic, 1> & Fg, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions){
    
    using RealType = double;
    auto exact_vel_sol_fun = functions.Evaluate_v(t);
    auto rhs_fun = functions.Evaluate_f(t);
    
    assembler.LHS *= 0.0;
    assembler.RHS *= 0.0;
    
    for (auto& cell : msh.cells)
    {
        
        double c = 1.0;
        auto bar = barycenter(msh, cell);
        double x = bar.x();
        if (x < 0.5) {
            c *= contrast;
        }
    
        auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
        auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
        auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif

        auto n_rows = reconstruction_operator.second.rows();
        auto n_cols = reconstruction_operator.second.cols();
        
        auto n_s_rows = stabilization_operator.rows();
        auto n_s_cols = stabilization_operator.cols();
        
        Matrix<RealType, Dynamic, Dynamic> S_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
        
        Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
        R_operator.block(0, 0, n_rows - n_s_rows, n_cols - n_s_cols) *= 0.0;
        // Compossing objects
        Matrix<RealType, Dynamic, Dynamic> laplacian_loc = R_operator + (1.0/c)*S_operator;
        Matrix<RealType, Dynamic, 1> f_loc = make_mixed_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
        assembler.assemble_mixed_RHS(msh, cell, laplacian_loc, f_loc, exact_vel_sol_fun);
    }
    assembler.finalize();
    Fg = assembler.RHS;
    
}

void ERKWeightOpt(TSSPRKHHOAnalyses & ssprk_an,  Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof, double dt, double a, double b){
    
    timecounter tc;
    tc.tic();
    
    size_t n_c_dof = x_dof_n_m.rows() - n_f_dof;
    Matrix<double, Dynamic, 1> x_c_dof = x_dof_n_m.block(0, 0, n_c_dof, 1);
    Matrix<double, Dynamic, 1> x_f_dof = x_dof_n_m.block(n_c_dof, 0, n_f_dof, 1);
    
    // Faces update (last state)
    {
        Matrix<double, Dynamic, 1> RHSf = ssprk_an.Kfc()*x_c_dof;
        x_f_dof = -ssprk_an.FacesAnalysis().solve(RHSf);
    }
    
    // Cells update
    Matrix<double, Dynamic, 1> RHSc = ssprk_an.Fc() - ssprk_an.Kc()*x_c_dof - ssprk_an.Kcf()*x_f_dof;
    Matrix<double, Dynamic, 1> delta_x_c_dof = ssprk_an.CellsAnalysis().solve(RHSc);
    Matrix<double, Dynamic, 1> x_n_c_dof = a * x_c_dof + b * dt * delta_x_c_dof; // new state
    
    // Faces update
    Matrix<double, Dynamic, 1> RHSf = ssprk_an.Kfc()*x_n_c_dof;
    Matrix<double, Dynamic, 1> x_n_f_dof = -ssprk_an.FacesAnalysis().solve(RHSf); // new state
    
    // Composing global solution
    x_dof_n = x_dof_n_m;
    x_dof_n.block(0, 0, n_c_dof, 1) = x_n_c_dof;
    x_dof_n.block(n_c_dof, 0, n_f_dof, 1) = x_n_f_dof;
    tc.toc();
    
}

void ERKWeight(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, size_t n_f_dof, double dt, double a, double b){
    
    using RealType = double;
    timecounter tc;
    tc.tic();
      
    size_t n_c_dof = Kg.rows() - n_f_dof;
    
    // Composing objects
    SparseMatrix<RealType> Mc = Mg.block(0, 0, n_c_dof, n_c_dof);
    SparseMatrix<RealType> Kc = Kg.block(0, 0, n_c_dof, n_c_dof);
    SparseMatrix<RealType> Kcf = Kg.block(0, n_c_dof, n_c_dof, n_f_dof);
    SparseMatrix<RealType> Kfc = Kg.block(n_c_dof,0, n_f_dof, n_c_dof);
    SparseMatrix<RealType> Sff = Kg.block(n_c_dof,n_c_dof, n_f_dof, n_f_dof);
    Matrix<double, Dynamic, 1> Fc = Fg.block(0, 0, n_c_dof, 1);
    Matrix<double, Dynamic, 1> x_c_dof = x_dof_n_m.block(0, 0, n_c_dof, 1);
    Matrix<double, Dynamic, 1> x_f_dof = x_dof_n_m.block(n_c_dof, 0, n_f_dof, 1);
    
    
    SparseLU<SparseMatrix<RealType>> analysis_f;
    analysis_f.analyzePattern(Sff);
    analysis_f.factorize(Sff);
    {
        // Faces update (last state)
        Matrix<double, Dynamic, 1> RHSf = Kfc*x_c_dof;
        Matrix<double, Dynamic, 1> x_f_dof_c = -analysis_f.solve(RHSf);
        x_f_dof = x_f_dof_c;
    }
    
    // Cells update
    SparseLU<SparseMatrix<RealType>> analysis_c;
    analysis_c.analyzePattern(Mc);
    analysis_c.factorize(Mc);
    Matrix<double, Dynamic, 1> RHSc = Fc - Kc*x_c_dof - Kcf*x_f_dof;
    Matrix<double, Dynamic, 1> delta_x_c_dof = analysis_c.solve(RHSc);
    Matrix<double, Dynamic, 1> x_n_c_dof = a * x_c_dof + b * dt * delta_x_c_dof; // new state
    
    // Faces update
    Matrix<double, Dynamic, 1> RHSf = Kfc*x_n_c_dof;
    Matrix<double, Dynamic, 1> x_n_f_dof = -analysis_f.solve(RHSf); // new state
    
    // Composing global solution
    x_dof_n = x_dof_n_m;
    x_dof_n.block(0, 0, n_c_dof, 1) = x_n_c_dof;
    x_dof_n.block(n_c_dof, 0, n_f_dof, 1) = x_n_f_dof;
    tc.toc();
    
}

void FaceDoFUpdate(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & x_dof, size_t n_f_dof){
    
    using RealType = double;
    timecounter tc;
    tc.tic();
    
    size_t n_c_dof = x_dof.rows() - n_f_dof;
    
    // Composing objects
    SparseMatrix<RealType> Mc = Mg.block(0, 0, n_c_dof, n_c_dof);
    SparseMatrix<RealType> Kc = Kg.block(0, 0, n_c_dof, n_c_dof);
    SparseMatrix<RealType> Kcf = Kg.block(0, n_c_dof, n_c_dof, n_f_dof);
    SparseMatrix<RealType> Kfc = Kg.block(n_c_dof,0, n_f_dof, n_c_dof);
    SparseMatrix<RealType> Sff = Kg.block(n_c_dof,n_c_dof, n_f_dof, n_f_dof);
    Matrix<double, Dynamic, 1> x_c_dof = x_dof.block(0, 0, n_c_dof, 1);
    Matrix<double, Dynamic, 1> x_f_dof = x_dof.block(n_c_dof, 0, n_f_dof, 1);
    
    std::cout << "x_f_dof = " << x_f_dof << std::endl;
    
    // Faces update (last state)
    Matrix<double, Dynamic, 1> RHSf = Kfc*x_c_dof;
    SparseLU<SparseMatrix<RealType>> analysis_f;
    analysis_f.analyzePattern(Sff);
    analysis_f.factorize(Sff);
    Matrix<double, Dynamic, 1> x_f_dof_c = -analysis_f.solve(RHSf); // new state
    std::cout << "x_f_dof_c = " << x_f_dof_c << std::endl;
    x_f_dof = x_f_dof_c;
    x_dof.block(n_c_dof, 0, n_f_dof, 1) = x_f_dof;

    tc.toc();
    
}

void IRKWeight(SparseMatrix<double> Kg, Matrix<double, Dynamic, 1> Fg, SparseMatrix<double> & Mg,  Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k, double dt, double a){
    
    using RealType = double;
    timecounter tc;
    tc.tic();
    
    Fg -= Kg*y;
    Kg *= (a*dt);
    Kg += Mg;
    
    SparseLU<SparseMatrix<RealType>> analysis_t;
    analysis_t.analyzePattern(Kg);
    analysis_t.factorize(Kg);
    k = analysis_t.solve(Fg);
    tc.toc();

}

void DIRKStepOpt(int s, Matrix<double, Dynamic, Dynamic> &a, Matrix<double, Dynamic, 1> &b, Matrix<double, Dynamic, 1> &c, poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, TDIRKHHOAnalyses & dirk_an, double & tn, double & dt, TAnalyticalFunction & functions, Matrix<double, Dynamic, 1> & x_dof_n_m, Matrix<double, Dynamic, 1> & x_dof_n, bool is_sdirk_Q){
    
    size_t n_dof = x_dof_n_m.rows();
    Matrix<double, Dynamic, Dynamic> k = Matrix<double, Dynamic, Dynamic>::Zero(n_dof, s);
    Matrix<double, Dynamic, 1> Fg, Fg_c,xd;
    xd = Matrix<double, Dynamic, 1>::Zero(n_dof, 1);
    
    double t;
    Matrix<double, Dynamic, 1> yn, ki;

    x_dof_n = x_dof_n_m;
    for (int i = 0; i < s; i++) {
        
        yn = x_dof_n_m;
        for (int j = 0; j < s - 1; j++) {
            yn += a(i,j) * dt * k.block(0, j, n_dof, 1);
        }
        
        t = tn + c(i,0) * dt;
        ComputeFG(Fg, msh, hho_di, assembler, t, functions);
        dirk_an.SetFg(Fg);
        
        IRKWeightOpt(dirk_an, yn, ki, dt, a(i,i),is_sdirk_Q);
        
        // Accumulated solution
        x_dof_n += dt*b(i,0)*ki;
        k.block(0, i, n_dof, 1) = ki;
    }
    
}

void IRKWeightOpt(TDIRKHHOAnalyses & dirk_an, Matrix<double, Dynamic, 1> & y, Matrix<double, Dynamic, 1> & k, double dt, double a, bool is_sdirk_Q){
    
    timecounter tc;
    tc.tic();
    
    Matrix<double, Dynamic, 1> Fg = dirk_an.Fg();
    Fg -= dirk_an.Kg()*y;
    
    if (is_sdirk_Q) {
        k = dirk_an.DirkAnalysis().solve(Fg);
    }else{
        double scale = a * dt;
        dirk_an.SetScale(scale);
        dirk_an.DecomposeMatrix();
        k = dirk_an.DirkAnalysis().solve(Fg);
    }
    tc.toc();
}

void HHOFirstOrderExample(int argc, char **argv){
    
    using RealType = double;
    size_t k_degree = 0;
    size_t n_divs   = 0;
    
    int opt;
    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
    {
        switch(opt)
        {
            case 'k':
            {
                k_degree = atoi(optarg);
            }
                break;
            case 'l':
            {
                n_divs = atoi(optarg);
            }
                break;
            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }
    
    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
    std::cout << bold << red << "l : " << n_divs << reset << std::endl;

    // The mesh in ProtoN seems like is always 2D
     mesh_init_params<RealType> mip;
     mip.Nx = 1;
     mip.Ny = 1;
    
    for (size_t i = 0; i < n_divs; i++) {
        mip.Nx *= 2;
        mip.Ny *= 2;
    }
    
    timecounter tc;
    
    // Building the cartesian mesh
    tc.tic();
    poly_mesh<RealType> msh(mip);
    tc.toc();

    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;

    
    // Creating HHO approximation spaces and corresponding linear operator
    hho_degree_info hho_di(k_degree,k_degree);
    
    // Solving a HDG/HHO mixed problem
    auto assembler = make_assembler(msh, hho_di, true); // another assemble version
    tc.tic();
    
    // Manufactured solution
#ifdef quadratic_space_solution_Q
    
    auto exact_scal_sol_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
        return (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
    };
    
    auto exact_flux_sol_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        flux[0] = (1 - x)*(1 - y)*y - x*(1 - y)*y;
        flux[1] = (1 - x)*x*(1 - y) - (1 - x)*x*y;
        flux[0] *=-1.0;
        flux[1] *=-1.0;
        return flux;
    };
    
    auto rhs_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        return -2.0*((x - 1)*x + (y - 1)*y);
    };
    
#else
    
    auto exact_scal_sol_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
        return std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
    };
    
    auto exact_flux_sol_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> std::vector<RealType> {
        double x,y;
        x = pt.x();
        y = pt.y();
        std::vector<RealType> flux(2);
        flux[0] =  M_PI*std::cos(M_PI*pt.x())*std::sin(M_PI*pt.y());
        flux[1] =  M_PI*std::sin(M_PI*pt.x())*std::cos(M_PI*pt.y());
        flux[0] *=-1.0;
        flux[1] *=-1.0;
        return flux;
    };
    
    auto rhs_fun = [](const typename poly_mesh<RealType>::point_type& pt) -> RealType {
        double x,y;
        x = pt.x();
        y = pt.y();
        return 2.0*M_PI*M_PI*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
    };
    
#endif
    
    for (auto& cell : msh.cells)
    {
            auto reconstruction_operator = make_hho_mixed_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
            auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
            auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
        
        auto n_rows = reconstruction_operator.second.rows();
        auto n_cols = reconstruction_operator.second.cols();
        
        auto n_s_rows = stabilization_operator.rows();
        auto n_s_cols = stabilization_operator.cols();
        
        Matrix<RealType, Dynamic, Dynamic> R_operator = reconstruction_operator.second;
        Matrix<RealType, Dynamic, Dynamic> S_operator = Matrix<RealType, Dynamic, Dynamic>::Zero(n_rows, n_cols);
        S_operator.block(n_rows-n_s_rows, n_cols-n_s_cols, n_s_rows, n_s_cols) = stabilization_operator;
        
        // Compossing objects
        Matrix<RealType, Dynamic, Dynamic> mixed_operator_loc = R_operator - S_operator;
        Matrix<RealType, Dynamic, 1> f_loc = make_mixed_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
        assembler.assemble_mixed(msh, cell, mixed_operator_loc, f_loc, exact_scal_sol_fun);
    }
    assembler.finalize();
    
    tc.tic();
    SparseLU<SparseMatrix<RealType>> analysis_t;
    analysis_t.analyzePattern(assembler.LHS);
    analysis_t.factorize(assembler.LHS);
    Matrix<RealType, Dynamic, 1> x_dof = analysis_t.solve(assembler.RHS); // new state
    tc.toc();
    
    // Computing errors
    ComputeL2ErrorTwoFields(msh, hho_di, x_dof, exact_scal_sol_fun, exact_flux_sol_fun);
    
    size_t it = 0;
    std::string silo_file_name = "scalar_mixed_";
    RenderSiloFileTwoFields(silo_file_name, it, msh, hho_di, x_dof, exact_scal_sol_fun, exact_flux_sol_fun);
    return;
    
}

void HeterogeneousIHHOSecondOrder(int argc, char **argv){
    
    bool render_silo_files_Q = true;
    bool render_zonal_vars_Q = false;
    using RealType = double;
    size_t k_degree = 0;
    size_t n_divs   = 0;
    
    // Final time value 0.5
    std::vector<size_t> nt_v = {5,10,20,40,80,160,320,640,1280,2560,5120,10240};
    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625,0.00078125,0.000390625,0.0001953125,0.00009765625,0.000048828125};
    
    int tref = 11;
    size_t nt       = nt_v[tref];
    RealType dt     = dt_v[tref];
    RealType ti = 0.0;
    
    int opt;
    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
    {
        switch(opt)
        {
            case 'k':
            {
                k_degree = atoi(optarg);
            }
                break;
            case 'l':
            {
                n_divs = atoi(optarg);
            }
                break;
            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }
    
    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
    std::cout << bold << red << "l : " << n_divs << reset << std::endl;
    std::cout << bold << red << "nt : " << nt << reset << std::endl;
    std::cout << bold << red << "dt : " << dt << reset << std::endl;

    // The mesh in ProtoN seems like is always 2D
     mesh_init_params<RealType> mip;
     mip.Nx = 10;
     mip.Ny = 1;
     mip.max_y = 0.1;
    
    for (size_t i = 0; i < n_divs; i++) {
        mip.Nx *= 2;
    }
    
    timecounter tc;
    
    // Building the cartesian mesh
    tc.tic();
    poly_mesh<RealType> msh(mip);
    tc.toc();

    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    
    // Creating HHO approximation spaces and corresponding linear operator
    hho_degree_info hho_di(k_degree,k_degree);
    // Construct Mass matrix
    auto mass_assembler_p = make_assembler(msh, hho_di);
    auto mass_assembler_v = make_assembler(msh, hho_di);
    auto mass_assembler_a = make_assembler(msh, hho_di);
    tc.tic();
    
    // Projection for acceleration
    TAnalyticalFunction functions;
    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionInhomogeneousInSpace);
    RealType t = ti;
    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
    
    for (auto& cell : msh.cells)
    {
        
        double c = 1.0;
        auto bar = barycenter(msh, cell);
        double x = bar.x();
        if (x < 0.5) {
            c *= contrast;
        }
        
        auto mass_operator = make_mass_matrix(msh, cell, hho_di);
        auto mass_operator_a = make_cell_mass_matrix(msh, cell, hho_di);
        
        Matrix<RealType, Dynamic, 1> f_p = make_rhs(msh, cell, hho_di.cell_degree(), exact_scal_sol_fun);
        Matrix<RealType, Dynamic, 1> f_v = make_rhs(msh, cell, hho_di.cell_degree(), exact_vel_sol_fun);
        Matrix<RealType, Dynamic, 1> f_a = make_rhs(msh, cell, hho_di.cell_degree(), exact_accel_sol_fun);
        
        mass_assembler_p.assemble(msh, cell, mass_operator, f_p, exact_scal_sol_fun);
        mass_assembler_v.assemble(msh, cell, mass_operator, f_v, exact_vel_sol_fun);
        
        mass_operator_a *= (1.0/(c*c));
        mass_assembler_a.assemble(msh, cell, mass_operator_a, f_a, exact_accel_sol_fun);
    }
    mass_assembler_p.finalize();
    mass_assembler_v.finalize();
    mass_assembler_a.finalize();
    
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> p_dof_n, v_dof_n, a_dof_n;
    
    tc.tic();
    SparseLU<SparseMatrix<RealType>> analysis;
    analysis.analyzePattern(mass_assembler_p.LHS);
    analysis.factorize(mass_assembler_p.LHS);
    p_dof_n = analysis.solve(mass_assembler_p.RHS); // Initial scalar
    v_dof_n = analysis.solve(mass_assembler_v.RHS); // Initial velocity
    a_dof_n = analysis.solve(mass_assembler_a.RHS); // Initial acceleration
    
    // updating face dofs
//    for (auto & face : msh.faces) {
//
//            auto cbs = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
//            auto fbs = face_basis<poly_mesh<RealType>,RealType>::size(hho_di.face_degree());
//            Matrix<RealType, Dynamic, Dynamic> face_mm = make_mass_matrix(msh, face, hho_di.face_degree());
//
//            Matrix<RealType, Dynamic, 1> p_face_rhs = make_rhs(msh, face, hho_di.face_degree(), exact_scal_sol_fun);
//            Matrix<RealType, Dynamic, 1> v_face_rhs = make_rhs(msh, face, hho_di.face_degree(), exact_vel_sol_fun);
//            Matrix<RealType, Dynamic, 1> a_face_rhs = make_rhs(msh, face, hho_di.face_degree(), exact_accel_sol_fun);
//            Matrix<RealType, Dynamic, 1> p_face_dof = face_mm.llt().solve(p_face_rhs);
//            Matrix<RealType, Dynamic, 1> v_face_dof = face_mm.llt().solve(v_face_rhs);
//            Matrix<RealType, Dynamic, 1> a_face_dof = face_mm.llt().solve(a_face_rhs);
//
//            bool dirichlet = face.is_boundary && face.bndtype == boundary::DIRICHLET;
//            if (dirichlet)
//            {
//    //            Matrix<T, Dynamic, Dynamic> mass = make_mass_matrix(msh, fc, facdeg);
//    //            Matrix<T, Dynamic, 1> rhs = make_rhs(msh, fc, facdeg, dirichlet_bf);
//    //            ret.block(cbs+face_i*fbs, 0, fbs, 1) = mass.llt().solve(rhs);
//            }
//            else
//            {
//                auto face_offset = offset(msh, face);
//                auto face_SOL_offset = cbs * msh.cells.size() + mass_assembler_p.get_compress_table().at(face_offset)*fbs;
//                p_dof_n.block(face_SOL_offset, 0, fbs, 1) = p_face_dof;
//                v_dof_n.block(face_SOL_offset, 0, fbs, 1) = v_face_dof;
//                a_dof_n.block(face_SOL_offset, 0, fbs, 1) = a_face_dof;
//            }
//
//        }
    
#ifdef compute_energy_Q
        Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
        energy_h_values(0,0) = 0.0;
        energy_h_values(0,1) = 1.0;
        RealType energy_h0 = ComputeEnergySecondOrder(msh, hho_di, mass_assembler_a, p_dof_n, v_dof_n);
        std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
#endif
    
    tc.toc();
    
    if(render_silo_files_Q){
        size_t it = 0;
        std::string silo_file_name = "scalar_inhomogeneous_wave_";
        RenderSiloFileScalarField(silo_file_name, it, msh, hho_di, p_dof_n, exact_scal_sol_fun, render_zonal_vars_Q);
    }
    
    // Transient problem
    bool is_implicit_Q = true;
    if (is_implicit_Q) {
        
        
        
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;
        auto assembler = make_assembler(msh, hho_di);
        RealType beta = 0.25;
        RealType gamma = 0.5;

        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            
            // Manufactured solution
            RealType t = dt*it+ti;
            tc.tic();
            ComputeKGFGSecondOrder(msh, hho_di, assembler, t, functions);
            
            // Compute intermediate state for scalar and rate
            p_dof_n = p_dof_n + dt*v_dof_n + 0.5*dt*dt*(1.0-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1.0-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = assembler.LHS*p_dof_n;
            
            assembler.LHS *= beta*(dt*dt);
            assembler.LHS += mass_assembler_a.LHS;
            assembler.RHS -= res;
            tc.toc();
            std::cout << bold << cyan << "Assembly completed: " << tc << " seconds" << reset << std::endl;
                    
            tc.tic();
            SparseLU<SparseMatrix<RealType>> analysis;
            analysis.analyzePattern(assembler.LHS);
            analysis.factorize(assembler.LHS);
            a_dof_np = analysis.solve(assembler.RHS); // new acceleration
            tc.toc();

            // update scalar and rate
            p_dof_n += beta*dt*dt*a_dof_np;
            v_dof_n += gamma*dt*a_dof_np;
            a_dof_n  = a_dof_np;
        
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            if(render_silo_files_Q){
                auto exact_scalar_sol_fun = functions.Evaluate_u(t);
                std::string silo_file_name = "scalar_inhomogeneous_wave_";
                RenderSiloFileScalarField(silo_file_name, it, msh, hho_di, p_dof_n, exact_scalar_sol_fun, render_zonal_vars_Q);
            }
            
#ifdef compute_energy_Q
            {
                tc.tic();
                RealType energy_h = ComputeEnergySecondOrder(msh, hho_di, assembler, p_dof_n, v_dof_n);
                tc.toc();
                energy_h_values(it,0) = t;
                energy_h_values(it,1) = energy_h/energy_h0;
                std::cout << bold << cyan << "Energy computed: " << tc << " seconds" << reset << std::endl;
            }
#endif
            
#ifdef spatial_errors_Q
            if(it == nt){
                auto exact_scal_sol_fun = functions.Evaluate_u(t);
                ComputeL2ErrorSingleField(msh, hho_di, assembler, p_dof_n, exact_scal_sol_fun, exact_flux_sol_fun);
            }
#endif

        }
        
#ifdef compute_energy_Q
    std::cout << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
    std::cout << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
#endif
        
    }
    
}

void IHHOSecondOrder(int argc, char **argv){
    
    bool render_silo_files_Q = true;
    bool render_zonal_vars_Q = false;
    using RealType = double;
    size_t k_degree = 0;
    size_t n_divs   = 0;
    
    // Final time value 1.0
    std::vector<size_t> nt_v = {10,20,40,80,160,320,640};
    std::vector<double> dt_v = {0.1,0.05,0.025,0.0125,0.00625,0.003125,0.0015625};
    int tref = 4;
    size_t nt       = nt_v[tref];
    RealType dt     = dt_v[tref];
    RealType ti = 0.0;
    
    int opt;
    while ( (opt = getopt(argc, argv, "k:l:n")) != -1 )
    {
        switch(opt)
        {
            case 'k':
            {
                k_degree = atoi(optarg);
            }
                break;
            case 'l':
            {
                n_divs = atoi(optarg);
            }
                break;
            case '?':
            default:
                std::cout << "wrong arguments" << std::endl;
                exit(1);
        }
    }
    
    std::cout << bold << red << "k : " << k_degree << reset << std::endl;
    std::cout << bold << red << "l : " << n_divs << reset << std::endl;
    std::cout << bold << red << "nt : " << nt << reset << std::endl;
    std::cout << bold << red << "dt : " << dt << reset << std::endl;

     mesh_init_params<RealType> mip;
     mip.Nx = 1;
     mip.Ny = 1;
    
    for (size_t i = 0; i < n_divs; i++) {
        mip.Nx *= 2;
        mip.Ny *= 2;
    }
    
    timecounter tc;
    
    // Building the cartesian mesh
    tc.tic();
    poly_mesh<RealType> msh(mip);
    tc.toc();

    std::cout << bold << cyan << "Mesh generation: " << tc << " seconds" << reset << std::endl;
    
    // Creating HHO approximation spaces and corresponding linear operator
    hho_degree_info hho_di(k_degree,k_degree);
    // Construct Mass matrix
    auto mass_assembler_p = make_assembler(msh, hho_di);
    auto mass_assembler_v = make_assembler(msh, hho_di);
    auto mass_assembler_a = make_assembler(msh, hho_di);
    tc.tic();
    // Projection for acceleration
    
    TAnalyticalFunction functions;
    functions.SetFunctionType(TAnalyticalFunction::EFunctionType::EFunctionNonPolynomial);
    RealType t = ti;
    auto exact_scal_sol_fun     = functions.Evaluate_u(t);
    auto exact_vel_sol_fun      = functions.Evaluate_v(t);
    auto exact_accel_sol_fun    = functions.Evaluate_a(t);
    auto exact_flux_sol_fun     = functions.Evaluate_q(t);
    
    for (auto& cell : msh.cells)
    {
        auto mass_operator = make_mass_matrix(msh, cell, hho_di);
        auto mass_operator_a = make_cell_mass_matrix(msh, cell, hho_di);
        
        Matrix<RealType, Dynamic, 1> f_p = make_rhs(msh, cell, hho_di.cell_degree(), exact_scal_sol_fun);
        Matrix<RealType, Dynamic, 1> f_v = make_rhs(msh, cell, hho_di.cell_degree(), exact_vel_sol_fun);
        Matrix<RealType, Dynamic, 1> f_a = make_rhs(msh, cell, hho_di.cell_degree(), exact_accel_sol_fun);
        
        mass_assembler_p.assemble(msh, cell, mass_operator, f_p, exact_scal_sol_fun);
        mass_assembler_v.assemble(msh, cell, mass_operator, f_v, exact_vel_sol_fun);
        mass_assembler_a.assemble(msh, cell, mass_operator_a, f_a, exact_accel_sol_fun);
    }
    mass_assembler_p.finalize();
    mass_assembler_v.finalize();
    mass_assembler_a.finalize();
    
    tc.toc();
    std::cout << bold << cyan << "Mass Assembly completed: " << tc << " seconds" << reset << std::endl;
    
    // Projecting initial scalar, velocity and acceleration
    Matrix<RealType, Dynamic, 1> p_dof_n, v_dof_n, a_dof_n;
    
    tc.tic();
    SparseLU<SparseMatrix<RealType>> analysis;
    analysis.analyzePattern(mass_assembler_p.LHS);
    analysis.factorize(mass_assembler_p.LHS);
    p_dof_n = analysis.solve(mass_assembler_p.RHS); // Initial scalar
    v_dof_n = analysis.solve(mass_assembler_v.RHS); // Initial velocity
    a_dof_n = analysis.solve(mass_assembler_a.RHS); // Initial acceleration
    
#ifdef compute_energy_Q
        Matrix<RealType, Dynamic, 2> energy_h_values(nt+1,2);
        energy_h_values(0,0) = 0.0;
        energy_h_values(0,1) = 1.0;
        RealType energy_h0 = ComputeEnergySecondOrder(msh, hho_di, mass_assembler_a, p_dof_n, v_dof_n);
        std::cout << bold << cyan << "Initial energy computed: " << tc << " seconds" << reset << std::endl;
#endif
    
    tc.toc();
    
    if(render_silo_files_Q){
        size_t it = 0;
        std::string silo_file_name = "scalar_wave_";
        RenderSiloFileScalarField(silo_file_name, it, msh, hho_di, v_dof_n, exact_vel_sol_fun, render_zonal_vars_Q);
    }
    
    // Transient problem
    bool is_implicit_Q = true;
    
    if (is_implicit_Q) {
        
        Matrix<RealType, Dynamic, 1> a_dof_np = a_dof_n;
        
        RealType beta = 0.25;
        RealType gamma = 0.5;
        for(size_t it = 1; it <= nt; it++){
                
            std::cout << bold << yellow << "Time step number : " << it << " being executed." << reset << std::endl;
            
            // Manufactured solution
            RealType t = dt*it+ti;
            auto exact_scal_sol_fun     = functions.Evaluate_u(t);
            auto exact_vel_sol_fun      = functions.Evaluate_v(t);
            auto exact_flux_sol_fun     = functions.Evaluate_q(t);
            auto rhs_fun      = functions.Evaluate_f(t);
            
            auto assembler = make_assembler(msh, hho_di);
            tc.tic();
            for (auto& cell : msh.cells)
            {
                auto reconstruction_operator = make_hho_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
                auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
                auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
                Matrix<RealType, Dynamic, Dynamic> laplacian_loc = reconstruction_operator.second + stabilization_operator;
                Matrix<RealType, Dynamic, 1> f_loc = make_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
                assembler.assemble(msh, cell, laplacian_loc, f_loc, exact_scal_sol_fun);
            }
            assembler.finalize();
            
            
            // Compute intermediate state for scalar and rate
            p_dof_n = p_dof_n + dt*v_dof_n + 0.5*dt*dt*(1-2.0*beta)*a_dof_n;
            v_dof_n = v_dof_n + dt*(1-gamma)*a_dof_n;
            Matrix<RealType, Dynamic, 1> res = assembler.LHS*p_dof_n;
            
            assembler.LHS *= beta*(dt*dt);
            assembler.LHS += mass_assembler_a.LHS;
            assembler.RHS -= res;
            tc.toc();
            std::cout << bold << cyan << "Assembly completed: " << tc << " seconds" << reset << std::endl;
                    
            tc.tic();
            SparseLU<SparseMatrix<RealType>> analysis;
            analysis.analyzePattern(assembler.LHS);
            analysis.factorize(assembler.LHS);
            a_dof_np = analysis.solve(assembler.RHS); // new acceleration
            tc.toc();

            // update scalar and rate
            p_dof_n += beta*dt*dt*a_dof_np;
            v_dof_n += gamma*dt*a_dof_np;
            a_dof_n  = a_dof_np;
        
            std::cout << bold << cyan << "Solution completed: " << tc << " seconds" << reset << std::endl;
            
            if(render_silo_files_Q){
                std::string silo_file_name = "scalar_wave_";
                RenderSiloFileScalarField(silo_file_name, it, msh, hho_di, v_dof_n, exact_vel_sol_fun, render_zonal_vars_Q);
            }
            
#ifdef compute_energy_Q
            {
                tc.tic();
                RealType energy_h = ComputeEnergySecondOrder(msh, hho_di, assembler, p_dof_n, v_dof_n);
                tc.toc();
                energy_h_values(it,0) = t;
                energy_h_values(it,1) = energy_h/energy_h0;
                std::cout << bold << cyan << "Energy computed: " << tc << " seconds" << reset << std::endl;
            }
#endif
            
#ifdef spatial_errors_Q
            if(it == nt){
                ComputeL2ErrorSingleField(msh, hho_di, assembler, p_dof_n, exact_scal_sol_fun, exact_flux_sol_fun);
            }
#endif
            
        }
        

        
#ifdef compute_energy_Q
    std::cout << std::setprecision(20) << bold << cyan << "Reporting initial energy value : " << energy_h0 << reset << std::endl;
    std::cout << std::setprecision(20) << bold << cyan << "Reporting energy values : " << energy_h_values << reset << std::endl;
#endif
        
    }
    
}

void ComputeKGFGSecondOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, double & t, TAnalyticalFunction & functions){
    using RealType = double;
    

    auto exact_scal_sol_fun = functions.Evaluate_u(t);
    auto rhs_fun = functions.Evaluate_f(t);
    
    assembler.LHS *= 0.0;
    assembler.RHS *= 0.0;
    for (auto& cell : msh.cells)
    {
        double c = 1.0;
        auto bar = barycenter(msh, cell);
        double x = bar.x();
        if (x < 0.5) {
            c *= contrast;
        }
        
        auto reconstruction_operator = make_hho_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
        auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
        auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
//        Matrix<RealType, Dynamic, Dynamic> laplacian_loc = reconstruction_operator.second + (c*c)* stabilization_operator;
        Matrix<RealType, Dynamic, Dynamic> laplacian_loc = reconstruction_operator.second + stabilization_operator;
        Matrix<RealType, Dynamic, 1> f_loc = make_rhs(msh, cell, hho_di.cell_degree(), rhs_fun);
        assembler.assemble(msh, cell, laplacian_loc, f_loc, exact_scal_sol_fun);
    }
    assembler.finalize();
}

void ComputeL2ErrorSingleField(poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, Matrix<double, Dynamic, 1> & x_dof,
std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
                             std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun){
    
    timecounter tc;
    tc.tic();
    
    using RealType = double;
    RealType scalar_l2_error = 0.0;
    RealType flux_l2_error = 0.0;
    size_t cell_i = 0;
    for (auto& cell : msh.cells)
    {
        if(cell_i == 0){
            RealType h = diameter(msh, cell);
            std::cout << green << "h size = " << std::endl << h << std::endl;
        }
        
        size_t cell_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
        Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
        
        // scalar evaluation
        {
            Matrix<RealType, Dynamic, Dynamic> mass = make_mass_matrix(msh, cell, hho_di.cell_degree());
            Matrix<RealType, Dynamic, 1> rhs = make_rhs(msh, cell, hho_di.cell_degree(), scal_fun);
            Matrix<RealType, Dynamic, 1> real_dofs = mass.llt().solve(rhs);
            Matrix<RealType, Dynamic, 1> diff = real_dofs - scalar_cell_dof;
            scalar_l2_error += diff.dot(mass*diff);
            
        }
        
        // flux evaluation
        {
            auto int_rule = integrate(msh, cell, 2*(hho_di.cell_degree()+1));
            cell_basis<poly_mesh<RealType>, RealType> rec_basis(msh, cell, hho_di.reconstruction_degree());
            auto gr = make_hho_laplacian(msh, cell, hho_di);
            Matrix<RealType, Dynamic, 1> all_dofs = assembler.take_local_data(msh, cell, x_dof, scal_fun);
            Matrix<RealType, Dynamic, 1> recdofs = -1.0 * gr.first * all_dofs;

            // Error integrals
            for (auto & point_pair : int_rule) {

                RealType omega = point_pair.second;
                auto t_dphi = rec_basis.eval_gradients( point_pair.first );
                Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();

                for (size_t i = 1; i < t_dphi.rows(); i++){
                    grad_uh = grad_uh + recdofs(i-1)*t_dphi.block(i, 0, 1, 2);
                }

                Matrix<RealType, 1, 2> grad_u_exact = Matrix<RealType, 1, 2>::Zero();
                grad_u_exact(0,0) =  flux_fun(point_pair.first)[0];
                grad_u_exact(0,1) =  flux_fun(point_pair.first)[1];
                flux_l2_error += omega * (grad_u_exact - grad_uh).dot(grad_u_exact - grad_uh);

            }
        }
        
        cell_i++;
    }
    
    std::cout << green << "scalar L2-norm error = " << std::endl << std::sqrt(scalar_l2_error) << std::endl;
    std::cout << green << "flux L2-norm error = " << std::endl << std::sqrt(flux_l2_error) << std::endl;
    std::cout << std::endl;
    tc.toc();
    std::cout << bold << cyan << "Error completed: " << tc << " seconds" << reset << std::endl;
    
    
}

void ComputeL2ErrorTwoFields(poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
                             std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun){
    
    timecounter tc;
    tc.tic();
    
    using RealType = double;
    RealType scalar_l2_error = 0.0;
    RealType flux_l2_error = 0.0;
    size_t cell_i = 0;
    for (auto& cell : msh.cells)
    {
        if(cell_i == 0){
            RealType h = diameter(msh, cell);
            std::cout << green << "h size = " << std::endl << h << std::endl;
        }
        
        size_t cell_scal_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
        size_t cell_flux_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree()+1)-1;
        size_t cell_dof = cell_scal_dof+cell_flux_dof;
        
        // scalar evaluation
        {
            Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof+cell_flux_dof, 0, cell_scal_dof, 1);
            Matrix<RealType, Dynamic, Dynamic> mass = make_mass_matrix(msh, cell, hho_di.cell_degree());
            Matrix<RealType, Dynamic, 1> rhs = make_rhs(msh, cell, hho_di.cell_degree(), scal_fun, 2);
            Matrix<RealType, Dynamic, 1> real_dofs = mass.llt().solve(rhs);
            Matrix<RealType, Dynamic, 1> diff = real_dofs - scalar_cell_dof;
            scalar_l2_error += diff.dot(mass*diff);
        }
        
        // flux evaluation
        auto int_rule = integrate(msh, cell, 2*(hho_di.cell_degree()+1));
        cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
        Matrix<RealType, Dynamic, 1> flux_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_flux_dof, 1);
        for (auto & point_pair : int_rule) {
            
            RealType omega = point_pair.second;
            
            auto t_dphi = cell_basis.eval_gradients(point_pair.first);
            Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
            for (size_t i = 1; i < t_dphi.rows(); i++){
              grad_uh = grad_uh + flux_cell_dof(i-1)*t_dphi.block(i, 0, 1, 2);
            }

            Matrix<RealType, 1, 2> grad_u_exact = Matrix<RealType, 1, 2>::Zero();
            grad_u_exact(0,0) =  flux_fun(point_pair.first)[0];
            grad_u_exact(0,1) =  flux_fun(point_pair.first)[1];
            flux_l2_error += omega * (grad_u_exact - grad_uh).dot(grad_u_exact - grad_uh);
        }
        
        cell_i++;
    }
    
    std::cout << green << "scalar L2-norm error = " << std::endl << std::sqrt(scalar_l2_error) << std::endl;
    std::cout << green << "flux L2-norm error = " << std::endl << std::sqrt(flux_l2_error) << std::endl;
    std::cout << std::endl;
    tc.toc();
    std::cout << bold << cyan << "Error completed: " << tc << " seconds" << reset << std::endl;
    
    
}

void RenderSiloFileTwoFields(std::string silo_file_name, size_t it, poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
                             std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
                             std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun, bool cell_centered_Q){
    
    auto num_cells = msh.cells.size();
    auto num_points = msh.points.size();
    using RealType = double;
    std::vector<RealType> exact_u, approx_u;
    std::vector<RealType> exact_dux, exact_duy, approx_dux, approx_duy;

    if (cell_centered_Q) {
        exact_u.reserve( num_cells );
        approx_u.reserve( num_cells );
        exact_dux.reserve( num_cells );
        exact_duy.reserve( num_cells );
        approx_dux.reserve( num_cells );
        approx_duy.reserve( num_cells );
        
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            auto bar = barycenter(msh, cell);
            exact_u.push_back( scal_fun(bar) );
            exact_dux.push_back( flux_fun(bar)[0] );
            exact_duy.push_back( flux_fun(bar)[1] );

            size_t cell_scal_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
            size_t cell_flux_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree()+1)-1;
            size_t cell_dof = cell_scal_dof+cell_flux_dof;

            // scalar evaluation
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof+cell_flux_dof, 0, cell_scal_dof, 1);
                auto t_phi = cell_basis.eval_basis( bar );
                RealType uh = scalar_cell_dof.dot( t_phi );
                approx_u.push_back(uh);
            }

            // flux evaluation
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
                Matrix<RealType, Dynamic, 1> flux_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_flux_dof, 1);
                auto t_dphi = cell_basis.eval_gradients(bar);

                Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
                for (size_t i = 1; i < t_dphi.rows(); i++){
                  grad_uh = grad_uh + flux_cell_dof(i-1)*t_dphi.block(i, 0, 1, 2);
                }

                approx_dux.push_back(grad_uh(0,0));
                approx_duy.push_back(grad_uh(0,1));
            }
            cell_i++;
        }
        
    }else{
        
        exact_u.reserve( num_points );
        approx_u.reserve( num_points );
        exact_dux.reserve( num_points );
        exact_duy.reserve( num_points );
        approx_dux.reserve( num_points );
        approx_duy.reserve( num_points );
        
        // scan for selected cells, common cells are discardable
        std::map<size_t, size_t> point_to_cell;
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            size_t n_p = cell.ptids.size();
            for (size_t l = 0; l < n_p; l++)
            {
                auto pt_id = cell.ptids[l];
                point_to_cell[pt_id] = cell_i;
            }
            cell_i++;
        }
        
        size_t cell_scal_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
        size_t cell_flux_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree()+1)-1;
        size_t cell_dof = cell_scal_dof+cell_flux_dof;
        for (auto& pt_id : point_to_cell)
        {
            auto bar = msh.points.at( pt_id.first );
            exact_u.push_back( scal_fun(bar) );
            exact_dux.push_back( flux_fun(bar)[0] );
            exact_duy.push_back( flux_fun(bar)[1] );
            
            cell_i = pt_id.second;
            auto& cell = msh.cells.at(cell_i);


            // scalar evaluation
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof+cell_flux_dof, 0, cell_scal_dof, 1);
                auto t_phi = cell_basis.eval_basis( bar );
                RealType uh = scalar_cell_dof.dot( t_phi );
                approx_u.push_back(uh);
            }

            // flux evaluation
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
                Matrix<RealType, Dynamic, 1> flux_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_flux_dof, 1);
                auto t_dphi = cell_basis.eval_gradients(bar);

                Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
                for (size_t i = 1; i < t_dphi.rows(); i++){
                  grad_uh = grad_uh + flux_cell_dof(i-1)*t_dphi.block(i, 0, 1, 2);
                }

                approx_dux.push_back(grad_uh(0,0));
                approx_duy.push_back(grad_uh(0,1));
            }
        }
        
    }
    
    silo_database silo;
    silo_file_name += std::to_string(it) + ".silo";
    silo.create(silo_file_name.c_str());
    silo.add_mesh(msh, "mesh");
    if (cell_centered_Q) {
        silo.add_variable("mesh", "v", exact_u.data(), exact_u.size(), zonal_variable_t);
        silo.add_variable("mesh", "vh", approx_u.data(), approx_u.size(), zonal_variable_t);
        silo.add_variable("mesh", "qx", exact_dux.data(), exact_dux.size(), zonal_variable_t);
        silo.add_variable("mesh", "qy", exact_duy.data(), exact_duy.size(), zonal_variable_t);
        silo.add_variable("mesh", "qhx", approx_dux.data(), approx_dux.size(), zonal_variable_t);
        silo.add_variable("mesh", "qhy", approx_duy.data(), approx_duy.size(), zonal_variable_t);
    }else{
        silo.add_variable("mesh", "v", exact_u.data(), exact_u.size(), nodal_variable_t);
        silo.add_variable("mesh", "vh", approx_u.data(), approx_u.size(), nodal_variable_t);
        silo.add_variable("mesh", "qx", exact_dux.data(), exact_dux.size(), nodal_variable_t);
        silo.add_variable("mesh", "qy", exact_duy.data(), exact_duy.size(), nodal_variable_t);
        silo.add_variable("mesh", "qhx", approx_dux.data(), approx_dux.size(), nodal_variable_t);
        silo.add_variable("mesh", "qhy", approx_duy.data(), approx_duy.size(), nodal_variable_t);
    }
    
    silo.close();
    
}

void RenderSiloFileScalarField(std::string silo_file_name, size_t it, poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
                             std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun, bool cell_centered_Q){
    
    auto num_cells = msh.cells.size();
    auto num_points = msh.points.size();
    using RealType = double;
    std::vector<RealType> exact_u, approx_u;
    std::vector<RealType> exact_dux, exact_duy, approx_dux, approx_duy;

    if (cell_centered_Q) {
        exact_u.reserve( num_cells );
        approx_u.reserve( num_cells );
        
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            auto bar = barycenter(msh, cell);
            exact_u.push_back( scal_fun(bar) );

            size_t cell_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
            // scalar evaluation
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
                auto t_phi = cell_basis.eval_basis( bar );
                RealType uh = scalar_cell_dof.dot( t_phi );
                approx_u.push_back(uh);
            }
            cell_i++;
        }
        
    }else{
        
        exact_u.reserve( num_points );
        approx_u.reserve( num_points );
        
        // scan for selected cells, common cells are discardable
        std::map<size_t, size_t> point_to_cell;
        size_t cell_i = 0;
        for (auto& cell : msh.cells)
        {
            size_t n_p = cell.ptids.size();
            for (size_t l = 0; l < n_p; l++)
            {
                auto pt_id = cell.ptids[l];
                point_to_cell[pt_id] = cell_i;
            }
            cell_i++;
        }
        
        size_t cell_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
        for (auto& pt_id : point_to_cell)
        {
            auto bar = msh.points.at( pt_id.first );
            exact_u.push_back( scal_fun(bar) );
            
            cell_i = pt_id.second;
            auto& cell = msh.cells.at(cell_i);


            // scalar evaluation
            {
                cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
                Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_dof, 1);
                auto t_phi = cell_basis.eval_basis( bar );
                RealType uh = scalar_cell_dof.dot( t_phi );
                approx_u.push_back(uh);
            }
        }
        
    }
    
    silo_database silo;
    silo_file_name += std::to_string(it) + ".silo";
    silo.create(silo_file_name.c_str());
    silo.add_mesh(msh, "mesh");
    if (cell_centered_Q) {
        silo.add_variable("mesh", "v", exact_u.data(), exact_u.size(), zonal_variable_t);
        silo.add_variable("mesh", "vh", approx_u.data(), approx_u.size(), zonal_variable_t);
    }else{
        silo.add_variable("mesh", "v", exact_u.data(), exact_u.size(), nodal_variable_t);
        silo.add_variable("mesh", "vh", approx_u.data(), approx_u.size(), nodal_variable_t);
    }
    
    silo.close();
    
}

double ComputeEnergySecondOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, assembler<poly_mesh<double>> & assembler, Matrix<double, Dynamic, 1> & p_dof_n, Matrix<double, Dynamic, 1> & v_dof_n){
    using RealType = double;
    RealType energy_h = 0.0;
    size_t cell_i = 0;
    
    double t = 0.0
    ;    auto exact_sol_fun = [&t](const typename poly_mesh<double>::point_type& pt) -> double {

#ifdef quadratic_time_solution_Q
                return t * t * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
#else
#ifdef quadratic_space_solution_Q
                return std::cos(std::sqrt(2.0)*M_PI*t) * (1.0-pt.x())*pt.x() * (1.0-pt.y())*pt.y();
#else
                return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y());
#endif
#endif
    };
    
    for (auto &cell : msh.cells) {
        
            cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree());
            auto cell_dof = cell_basis.size();

            Matrix<RealType, Dynamic, Dynamic> mass = make_cell_mass_matrix(msh, cell, hho_di);

            Matrix<RealType, Dynamic, Dynamic> cell_mass = mass.block(0, 0, cell_dof, cell_dof);
            Matrix<RealType, Dynamic, 1> cell_alpha_dof_n_v = v_dof_n.block(cell_i*cell_dof, 0, cell_dof, 1);
            
            Matrix<RealType, Dynamic, 1> cell_mass_tested = cell_mass * cell_alpha_dof_n_v;
            Matrix<RealType, 1, 1> term_1 = cell_alpha_dof_n_v.transpose() * cell_mass_tested;
            energy_h += term_1(0,0);
            
            auto reconstruction_operator = make_hho_laplacian(msh, cell, hho_di);
#ifdef fancy_stabilization_Q
            auto stabilization_operator = make_hho_fancy_stabilization(msh, cell, reconstruction_operator.first, hho_di);
#else
            auto stabilization_operator = make_hho_naive_stabilization(msh, cell, hho_di);
#endif
            Matrix<RealType, Dynamic, Dynamic> laplacian_loc = reconstruction_operator.second + stabilization_operator;
            Matrix<RealType, Dynamic, 1> cell_p_dofs = assembler.take_local_data(msh, cell, p_dof_n, exact_sol_fun);
            Matrix<RealType, Dynamic, 1> cell_stiff_tested = laplacian_loc * cell_p_dofs;
            Matrix<RealType, 1, 1> term_2 = cell_p_dofs.transpose() * cell_stiff_tested;
            energy_h += term_2(0,0);
        cell_i++;
    }

    energy_h *= 0.5;
    std::cout << green << "Energy_h = " << std::endl << energy_h << std::endl;
    std::cout << std::endl;
    
    return energy_h;
}


double ComputeEnergyFirstOrder(poly_mesh<double> & msh, hho_degree_info & hho_di, Matrix<double, Dynamic, 1> & x_dof,
std::function<double(const typename poly_mesh<double>::point_type& )> scal_fun,
                             std::function<std::vector<double>(const typename poly_mesh<double>::point_type& )> flux_fun){
    
    timecounter tc;
    tc.tic();
    
    using RealType = double;
    RealType energy_h = 0.0;
    RealType scalar_l2_energy = 0.0;
    RealType flux_l2_energy = 0.0;
    size_t cell_i = 0;
    for (auto& cell : msh.cells)
    {
        if(cell_i == 0){
            RealType h = diameter(msh, cell);
            std::cout << green << "h size = " << std::endl << h << std::endl;
        }
        
        double c = 1.0;
        auto bar = barycenter(msh, cell);
        double x = bar.x();
        if (x < 0.5) {
            c *= contrast;
        }
        
        size_t cell_scal_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree());
        size_t cell_flux_dof = cell_basis<poly_mesh<RealType>,RealType>::size(hho_di.cell_degree()+1)-1;
        size_t cell_dof = cell_scal_dof+cell_flux_dof;
        
        // scalar evaluation
        {
            Matrix<RealType, Dynamic, 1> scalar_cell_dof = x_dof.block(cell_i*cell_dof+cell_flux_dof, 0, cell_scal_dof, 1);
            Matrix<RealType, Dynamic, Dynamic> mass = make_mass_matrix(msh, cell, hho_di.cell_degree());
            Matrix<RealType, Dynamic, 1> rhs = make_rhs(msh, cell, hho_di.cell_degree(), scal_fun);
            Matrix<RealType, Dynamic, 1> real_dofs = mass.llt().solve(rhs);
            Matrix<RealType, Dynamic, 1> diff = 0.0*real_dofs - scalar_cell_dof;

            if (x > 0.5) {
                scalar_l2_energy += ((1.0/(contrast*contrast))) * (1.0/(c*c))*diff.dot(mass*diff);
             }else{
                 scalar_l2_energy += (1.0/(c*c))*diff.dot(mass*diff);
             }
        }
        
        // flux evaluation
        auto int_rule = integrate(msh, cell, 2*(hho_di.cell_degree()+1));
        cell_basis<poly_mesh<RealType>, RealType> cell_basis(msh, cell, hho_di.cell_degree()+1);
        Matrix<RealType, Dynamic, 1> flux_cell_dof = x_dof.block(cell_i*cell_dof, 0, cell_flux_dof, 1);
        for (auto & point_pair : int_rule) {
            
            RealType omega = point_pair.second;
            
            auto t_dphi = cell_basis.eval_gradients(point_pair.first);
            Matrix<RealType, 1, 2> grad_uh = Matrix<RealType, 1, 2>::Zero();
            for (size_t i = 1; i < t_dphi.rows(); i++){
              grad_uh = grad_uh + flux_cell_dof(i-1)*t_dphi.block(i, 0, 1, 2);
            }

            Matrix<RealType, 1, 2> grad_u_exact = Matrix<RealType, 1, 2>::Zero();
            grad_u_exact(0,0) =  flux_fun(point_pair.first)[0];
            grad_u_exact(0,1) =  flux_fun(point_pair.first)[1];
            if (x > 0.5) {
                flux_l2_energy += omega * ((1.0/(contrast*contrast))) * (0.0*grad_u_exact - grad_uh).dot(0.0*grad_u_exact - grad_uh);
            }else{
                flux_l2_energy += omega * (0.0*grad_u_exact - grad_uh).dot(0.0*grad_u_exact - grad_uh);
            }
            
        }

        
        cell_i++;
    }
    
    energy_h = (scalar_l2_energy + flux_l2_energy)/2.0;
    std::cout << green << "Energy_h = " << std::endl << energy_h << std::endl;
    std::cout << bold << cyan << "Energy completed: " << tc << " seconds" << reset << std::endl;
    tc.toc();
    
    return energy_h;
    
}
