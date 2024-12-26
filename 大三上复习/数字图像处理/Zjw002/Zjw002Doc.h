// Zjw002Doc.h : interface of the CZjw002Doc class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_ZJW002DOC_H__E4E32EA4_79DE_4F23_AFD8_7D3D8C886AE5__INCLUDED_)
#define AFX_ZJW002DOC_H__E4E32EA4_79DE_4F23_AFD8_7D3D8C886AE5__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000


class CZjw002Doc : public CDocument
{
protected: // create from serialization only
	CZjw002Doc();
	DECLARE_DYNCREATE(CZjw002Doc)

// Attributes
public:

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CZjw002Doc)
	public:
	virtual BOOL OnNewDocument();
	virtual void Serialize(CArchive& ar);
	virtual BOOL OnOpenDocument(LPCTSTR lpszPathName);
	//}}AFX_VIRTUAL

// Implementation
public:
	virtual ~CZjw002Doc();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:

// Generated message map functions
protected:
	//{{AFX_MSG(CZjw002Doc)
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_ZJW002DOC_H__E4E32EA4_79DE_4F23_AFD8_7D3D8C886AE5__INCLUDED_)
