#ifndef TDIRKSchemes_hpp
#define TDIRKSchemes_hpp

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

#endif
