#ifndef TSSPRKHHOAnalyses_hpp
#define TSSPRKHHOAnalyses_hpp

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


#endif
