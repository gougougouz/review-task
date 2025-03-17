#if !defined(AFX_TESTDLG_H__46E10F8F_91B7_4F3A_AD27_F526C9CED5F4__INCLUDED_)
#define AFX_TESTDLG_H__46E10F8F_91B7_4F3A_AD27_F526C9CED5F4__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// testDlg.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CtestDlg dialog

class CtestDlg : public CDialog
{
// Construction
public:
	CtestDlg(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
	//{{AFX_DATA(CtestDlg)
	enum { IDD = IDD_DIALOG1 };
		// NOTE: the ClassWizard will add data members here
	//}}AFX_DATA


// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CtestDlg)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:

	// Generated message map functions
	//{{AFX_MSG(CtestDlg)
		// NOTE: the ClassWizard will add member functions here
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_TESTDLG_H__46E10F8F_91B7_4F3A_AD27_F526C9CED5F4__INCLUDED_)
