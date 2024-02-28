#ifndef TDIRKHHOAnalyses_hpp
#define TDIRKHHOAnalyses_hpp

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


#endif
