// myDlg.cpp : implementation file
//
/*
�Ի�����һ�����ڣ���Ի�����Դ��ص���ΪCDialog����CWnd������������
���Խ��Ի��򿴳���һ�������������������ܹ����ø��ֱ�׼����չ�ؼ������û��������н�������Ҫ�ֶΡ�
��MFC�У����еĿؼ�������CWnd������������ˣ��ؼ�ʵ����Ҳ�Ǵ��ڡ�

*/
/*
ģʽ�Ի���
      ������ʾʱ���������ִͣ�У�ֱ���ر�����Ի���󣬲��ܼ���ִ�г����������������硰�ļ�/�򿪡� �Ի���
��ģʽ�Ի���
      ������ʾʱ������ת��ִ�г������������񣬶����ùر�����Ի��򡣸����ͶԻ��򲻻�¢���û��Ĳ������û��Կ������������������н��������硰���ҡ��Ի���

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
