#ifndef TAnalyticalFunction_hpp
#define TAnalyticalFunction_hpp

class TAnalyticalFunction
{
    public:
    
    /// Enumerate defining the function type
    enum EFunctionType { EFunctionNonPolynomial = 0, 
                         EFunctionQuadraticInTime = 1, 
                         EFunctionQuadraticInSpace = 2, 
                         EFunctionQuadraticInSpaceTime = 3, 
                         EFunctionInhomogeneousInSpace = 4 };
    
    
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
          
          case EFunctionNonPolynomial : {
             return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                return (1.0/(std::sqrt(2.0)*M_PI))*std::sin(std::sqrt(2.0)*M_PI*t) * std::sin(M_PI*pt.x()) * std::sin(M_PI*pt.y()); 
             };
          }   
          break;
            
          case EFunctionQuadraticInSpace : {
             return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                return (1 - pt.x())*pt.x()*(1 - pt.y())*pt.y()*std::cos(std::sqrt(2.0)*M_PI*t);
             };
          }
          break;
          
          case EFunctionQuadraticInTime : {
             return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                return t*t*std::sin(M_PI*pt.x())*std::sin(M_PI*pt.y());
             };
          }
          break;
          
          case EFunctionQuadraticInSpaceTime : {
             return [&t](const typename poly_mesh<double>::point_type& pt) -> double {
                return t*t*(1 - pt.x())*pt.x()*(1 - pt.y())*pt.y();
             };
          }
          break;
          
          case EFunctionInhomogeneousInSpace : {
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
                   }
                   
                   else {
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
            
          default : {
             
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

#endif
