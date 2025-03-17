// testDlg2.cpp : implementation file
//

#include "stdafx.h"
#include "test1.h"
#include "testDlg2.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CtestDlg2 dialog


CtestDlg2::CtestDlg2(CWnd* pParent /*=NULL*/)
	: CDialog(CtestDlg2::IDD, pParent)
{
	//{{AFX_DATA_INIT(CtestDlg2)
	m_num1 = 0;
	m_num2 = 0;
	m_num3 = 0;
	//}}AFX_DATA_INIT
	m_bIsCreate=FALSE;
}


void CtestDlg2::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CtestDlg2)
	DDX_Control(pDX, IDC_EDIT3, m_edit3);
	DDX_Control(pDX, IDC_EDIT2, m_edit2);
	DDX_Control(pDX, IDC_EDIT1, m_edit1);
	DDX_Text(pDX, IDC_EDIT1, m_num1);
	DDX_Text(pDX, IDC_EDIT2, m_num2);
	DDX_Text(pDX, IDC_EDIT3, m_num3);
	//}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CtestDlg2, CDialog)
	//{{AFX_MSG_MAP(CtestDlg2)
	ON_BN_CLICKED(IDC_BTN_ADD, OnBtnAdd)
	ON_BN_CLICKED(IDC_NUMBER1, OnNumber1)
	ON_BN_CLICKED(IDC_NUMBER2, OnNumber2)
	ON_BN_CLICKED(IDC_BUTTON1, OnButton1)
	ON_BN_CLICKED(IDC_BUTTON2, OnButton2)
	ON_BN_CLICKED(IDC_BTN_DLLADD, OnBtnDlladd)
	ON_BN_CLICKED(IDC_BTN_DLLSUBTRAC, OnBtnDllsubtrac)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CtestDlg2 message handlers

void CtestDlg2::OnBtnAdd() 
{
	// TODO: Add your control notification handler code here
	/*if(m_bIsCreate==FALSE){

		m_btn.Create("Add被点击了",BS_DEFPUSHBUTTON|WS_VISIBLE|WS_CHILD,CRect(0,0,100,100),this,123);
		m_bIsCreate=TRUE;
	}else{
		m_btn.DestroyWindow();
		m_bIsCreate=FALSE;
	}*/
	
	
	/*//第一种访问控件方法//常用//掌握
	int num1,num2,num3;
	char c1[10], c2[10], c3[10];
	GetDlgItem(IDC_EDIT1)->GetWindowText(c1, 10);//第一个编辑框的内容获取
	GetDlgItem(IDC_EDIT2)->GetWindowText(c2, 10);//第二个
    num1 = atoi(c1);
	num2 = atoi(c2);
	num3 = num1 + num2;
	itoa(num3,c3,10);
	GetDlgItem(IDC_EDIT3)->SetWindowText(c3);*/

	
	/*//第二种访问控件方法
	int num1, num2, num3;
	char c1[10], c2[10], c3[10];
	GetDlgItemText(IDC_EDIT1, c1, 10);
	GetDlgItemText(IDC_EDIT2, c2, 10);
	num1 = atoi(c1);
	num2 = atoi(c2);
	num3 = num1 + num2;
	itoa(num3,c3,10);
	SetDlgItemText(IDC_EDIT3, c3);*/


	/*//第三种访问控件方法,会将str转为int//常用
	int num1, num2, num3;
	num1 = GetDlgItemInt(IDC_EDIT1);
	num2 = GetDlgItemInt(IDC_EDIT2);
    num3 = num1 + num2;
	SetDlgItemInt(IDC_EDIT3, num3);
	*/


	//第四种访问控件方法，定义了m_num1,2,3,三个成员变量和各自的编辑框对应
	/*UpdateData();
	m_num3=m_num1+m_num2;
	UpdateData(FALSE);*/

	
	
	//第五种访问控件，控件和控件对象关联
	/*int num1, num2, num3;
	char c1[10], c2[10], c3[10];
	m_edit1.GetWindowText(c1, 10);
	m_edit2.GetWindowText(c2, 10);
	num1 = atoi(c1);
	num2 = atoi(c2);
	num3 = num1 + num2;
	itoa(num3,c3,10);
	m_edit3.SetWindowText(c3);*/

	
	//第六种访问控件方法
	/*int num1, num2, num3;	
	char c1[10], c2[10], c3[10];
	::SendMessage(GetDlgItem(IDC_EDIT1)->m_hWnd, WM_GETTEXT, 10, (LPARAM)c1);//全局函数，GetDlgItem(IDC_EDIT1)->m_hWnd，利用控件ID获取他的句柄
	::SendMessage(m_edit2.m_hWnd, WM_GETTEXT, 10, (LPARAM)c2);//m_edit2.m_hWnd，利用控件对象获取当前他的句柄
	num1 = atoi(c1);
	num2 = atoi(c2);
	num3 = num1 + num2;
	itoa(num3, c3, 10);
	m_edit3.SendMessage(WM_SETTEXT, 0, (LPARAM)c3);//直接利用控件对象也可以Send消息，这时就不用获取句柄了*/

	//第七种访问控件方法，直接给对话框的子控件发送消息

	/*
直接给对话框的子控件发送消息：
LRESULT SendDlgItemMessage(int nID, UINT message, WPARA wParam = 0, LPARAM lParam = 0)
该函数功能相当于把上面GetDlgItem和SendMessage这两个函数的组合
  */
	
	int num1, num2, num3;	
	char c1[10], c2[10], c3[10];
	SendDlgItemMessage(IDC_EDIT1, WM_GETTEXT, 10, (LPARAM)c1);
	SendDlgItemMessage(IDC_EDIT2, WM_GETTEXT, 10, (LPARAM)c2);
	num1 = atoi(c1);
	num2 = atoi(c2);
	num3 = num1 + num2;
	itoa(num3, c3, 10);
	SendDlgItemMessage(IDC_EDIT3, WM_SETTEXT, 0, (LPARAM)c3);




}

void CtestDlg2::OnNumber1() 
{
	// TODO: Add your control notification handler code here
	CString str;
	GetDlgItem(IDC_NUMBER1)->GetWindowText(str);
	if (str == "Number1:")
		GetDlgItem(IDC_NUMBER1)->SetWindowText("数值1：");
	else
		GetDlgItem(IDC_NUMBER1)->SetWindowText("Number1:");

	
}

void CtestDlg2::OnNumber2() 
{
	// TODO: Add your control notification handler code here
	CWnd* pWnd = GetDlgItem(IDC_NUMBER2);

	CRect rc;
	pWnd->GetClientRect(&rc);
	CBrush brush(RGB(255,0,0));

	CDC* pDC = pWnd->GetDC();
	pDC->FillRect(&rc,&brush);
	pDC->SetBkMode(TRANSPARENT);
	pDC->TextOut(13,5,"在控件中绘图");	
	ReleaseDC(pDC);

}

void CtestDlg2::OnButton1() 
{
	// TODO: Add your control notification handler code here
	//改变对话框大小
	CString str;

	if(GetDlgItemText(IDC_BUTTON1,str),str=="收缩<<"){
		SetDlgItemText(IDC_BUTTON1,"扩展>>");
	}else{
		SetDlgItemText(IDC_BUTTON1,"收缩<<");
	}
	static CRect rectLarge;
	static CRect rectSmall;//切割之后的尺寸

	if(rectLarge.IsRectNull()){
		CRect rectSeparator;
		GetWindowRect(&rectLarge);
		GetDlgItem(IDC_SEPARATOR)->GetWindowRect(&rectSeparator);
		rectSmall.left=rectLarge.left;
		rectSmall.top=rectLarge.top;
		rectSmall.right=rectLarge.right;
		rectSmall.bottom=rectSeparator.bottom;
	}
	if(str=="收缩<<"){
		SetWindowPos(NULL,0,0,rectSmall.Width(),rectSmall.Height(),SWP_NOMOVE | SWP_NOZORDER);
	}else{
		SetWindowPos(NULL,0,0,rectLarge.Width(),rectLarge.Height(),SWP_NOMOVE | SWP_NOZORDER);
	}

}

void CtestDlg2::OnButton2() 
{
	// TODO: Add your control notification handler code here
	CRect rc;
	GetClientRect(rc);
	CRgn rgn;
	rgn.CreateEllipticRgn(0, 0, rc.Width(), rc.Height());
	SetWindowRgn((HRGN) rgn.m_hObject, TRUE);
	
}
_declspec (dllimport) int add(int a,int b);
_declspec (dllimport) int subtract(int a,int b);

void CtestDlg2::OnBtnDlladd() 
{
	// TODO: Add your control notification handler code here
	CString str;
	str.Format("7 + 5 = %d", add(7,5));
	MessageBox(str);

}

void CtestDlg2::OnBtnDllsubtrac() 
{
	// TODO: Add your control notification handler code here
	CString str;
	str.Format("7 - 5 = %d", subtract(7,5));
	MessageBox(str);

}
