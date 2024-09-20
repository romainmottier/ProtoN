

#ifndef IRKWeight_hpp
#define IRKWeight_hpp





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

#endif
