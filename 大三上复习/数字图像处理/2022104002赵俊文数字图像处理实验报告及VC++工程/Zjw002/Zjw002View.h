// Zjw002View.h : interface of the CZjw002View class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_ZJW002VIEW_H__BD253466_6A45_45BA_AFD5_95A1668C47D5__INCLUDED_)
#define AFX_ZJW002VIEW_H__BD253466_6A45_45BA_AFD5_95A1668C47D5__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000


class CZjw002View : public CScrollView
{
protected: // create from serialization only
	CZjw002View();
	DECLARE_DYNCREATE(CZjw002View)

// Attributes
public:
	CZjw002Doc* GetDocument();

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CZjw002View)
	public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	protected:
	virtual void OnInitialUpdate(); // called first time after construct
	virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
	virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
	virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CZjw002View();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CZjw002View)
	afx_msg void OnGray();
	afx_msg void OnUpdateGray(CCmdUI* pCmdUI);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
	afx_msg void OnHistogram();
	afx_msg void OnUpdateHistogram(CCmdUI* pCmdUI);
	afx_msg void OnEqualize();
	afx_msg void OnUpdateEqualize(CCmdUI* pCmdUI);
	afx_msg void OnFt();
	afx_msg void OnUpdateFt(CCmdUI* pCmdUI);
	afx_msg void OnIft();
	afx_msg void OnUpdateIft(CCmdUI* pCmdUI);
	afx_msg void OnFft();
	afx_msg void OnUpdateFft(CCmdUI* pCmdUI);
	afx_msg void OnIfft();
	afx_msg void OnUpdateIfft(CCmdUI* pCmdUI);
	afx_msg void OnAvgFilter();
	afx_msg void OnUpdateAvgFilter(CCmdUI* pCmdUI);
	afx_msg void OnGradSharp();
	afx_msg void OnUpdateGradSharp(CCmdUI* pCmdUI);
	afx_msg void OnMedFileter();
	afx_msg void OnUpdateMedFileter(CCmdUI* pCmdUI);
	afx_msg void OnRaplasSharp();
	afx_msg void OnUpdateRaplasSharp(CCmdUI* pCmdUI);
	afx_msg void OnLinetrans();
	afx_msg void OnUpdateLinetrans(CCmdUI* pCmdUI);
	afx_msg void OnFftFilter();
	afx_msg void OnUpdateFftFilter(CCmdUI* pCmdUI);
	afx_msg void OnGlpFilter();
	afx_msg void OnUpdateGlpFilter(CCmdUI* pCmdUI);
	afx_msg void OnIlpFilter();
	afx_msg void OnUpdateIlpFilter(CCmdUI* pCmdUI);
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

#ifndef _DEBUG  // debug version in Zjw002View.cpp
inline CZjw002Doc* CZjw002View::GetDocument()
   { return (CZjw002Doc*)m_pDocument; }
#endif

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_ZJW002VIEW_H__BD253466_6A45_45BA_AFD5_95A1668C47D5__INCLUDED_)
