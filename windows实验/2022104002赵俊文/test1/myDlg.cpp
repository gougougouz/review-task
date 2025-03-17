// myDlg.cpp : implementation file
//
/*
对话框是一个窗口，与对话框资源相关的类为CDialog，由CWnd类派生而来。
可以将对话框看成是一个大容器，在它上面能够放置各种标准和扩展控件，是用户与程序进行交互的重要手段。
在MFC中，所有的控件都是由CWnd派生而来，因此，控件实际上也是窗口。

*/
/*
模式对话框：
      当其显示时，程序会暂停执行，直到关闭这个对话框后，才能继续执行程序中其他任务。例如“文件/打开” 对话框。
无模式对话框：
      当其显示时，允许转而执行程序中其他任务，而不用关闭这个对话框。该类型对话框不会垄断用户的操作，用户仍可以与其他界面对象进行交互。例如“查找”对话框。

  */
#include "stdafx.h"
#include "test1.h"
#include "myDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CmyDlg dialog


CmyDlg::CmyDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CmyDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CmyDlg)
		// NOTE: the ClassWizard will add member initialization here
	//}}AFX_DATA_INIT
}


void CmyDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CmyDlg)
		// NOTE: the ClassWizard will add DDX and DDV calls here
	//}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CmyDlg, CDialog)
	//{{AFX_MSG_MAP(CmyDlg)
		// NOTE: the ClassWizard will add message map macros here
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CmyDlg message handlers
