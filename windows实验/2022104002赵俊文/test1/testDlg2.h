#if !defined(AFX_TESTDLG2_H__EE8B7005_DDEB_4CAB_8DAC_48AC5EA0B439__INCLUDED_)
#define AFX_TESTDLG2_H__EE8B7005_DDEB_4CAB_8DAC_48AC5EA0B439__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000
// testDlg2.h : header file
//

/////////////////////////////////////////////////////////////////////////////
// CtestDlg2 dialog

class CtestDlg2 : public CDialog
{
// Construction
public:
	CtestDlg2(CWnd* pParent = NULL);   // standard constructor

// Dialog Data
	//{{AFX_DATA(CtestDlg2)
	enum { IDD = IDD_DIALOG2 };
	CEdit	m_edit3;
	CEdit	m_edit2;
	CEdit	m_edit1;
	int		m_num1;
	int		m_num2;
	int		m_num3;
	//}}AFX_DATA


// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CtestDlg2)
	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
	//}}AFX_VIRTUAL

// Implementation
protected:

	// Generated message map functions
	//{{AFX_MSG(CtestDlg2)
	afx_msg void OnBtnAdd();
	afx_msg void OnNumber1();
	afx_msg void OnNumber2();
	afx_msg void OnButton1();
	afx_msg void OnButton2();
	afx_msg void OnBtnDlladd();
	afx_msg void OnBtnDllsubtrac();
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()
private:
	BOOL m_bIsCreate;
	CButton m_btn;
};

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_TESTDLG2_H__EE8B7005_DDEB_4CAB_8DAC_48AC5EA0B439__INCLUDED_)
