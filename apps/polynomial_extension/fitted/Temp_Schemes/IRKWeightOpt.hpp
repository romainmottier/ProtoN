
#ifndef IRKWeightOpt_hpp
#define IRKWeightOpt_hpp

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

#endif
