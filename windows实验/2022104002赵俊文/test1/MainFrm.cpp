// MainFrm.cpp : implementation of the CMainFrame class
//

#include "stdafx.h"
#include "test1.h"

#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CMainFrame

IMPLEMENT_DYNCREATE(CMainFrame, CFrameWnd)

BEGIN_MESSAGE_MAP(CMainFrame, CFrameWnd)
	//{{AFX_MSG_MAP(CMainFrame)
	ON_WM_CREATE()
	ON_COMMAND(IDM_MYMENU, OnMymenu)
	ON_UPDATE_COMMAND_UI(ID_EDIT_COPY, OnUpdateEditCopy)
	ON_COMMAND(IDM_SHOW, OnShow)
	ON_COMMAND(IDM_VIEW_NEWTOOL, OnViewNewtool)
	ON_UPDATE_COMMAND_UI(IDM_VIEW_NEWTOOL, OnUpdateViewNewtool)
	ON_WM_TIMER()
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

static UINT indicators[] =
{
	ID_SEPARATOR,           // status line indicator
	IDS_TIMER,
	IDS_PROGRESS,
	ID_INDICATOR_CAPS,
	ID_INDICATOR_NUM,
	ID_INDICATOR_SCRL,
};

/////////////////////////////////////////////////////////////////////////////
// CMainFrame construction/destruction

CMainFrame::CMainFrame()
{
	// TODO: add member initialization code here
	m_bAutoMenuEnable=FALSE;
}

CMainFrame::~CMainFrame()
{
}

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CFrameWnd::OnCreate(lpCreateStruct) == -1)
		return -1;
	
	if (!m_wndToolBar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | WS_VISIBLE | CBRS_TOP
		| CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC) ||
		!m_wndToolBar.LoadToolBar(IDR_MAINFRAME))
	{
		TRACE0("Failed to create toolbar\n");
		return -1;      // fail to create
	}

	if (!m_wndStatusBar.Create(this) ||
		!m_wndStatusBar.SetIndicators(indicators,
		  sizeof(indicators)/sizeof(UINT)))
	{
		TRACE0("Failed to create status bar\n");
		return -1;      // fail to create
	}

	// TODO: Delete these three lines if you don't want the toolbar to
	//  be dockable
	m_wndToolBar.EnableDocking(CBRS_ALIGN_ANY);
	EnableDocking(CBRS_ALIGN_ANY);
	DockControlBar(&m_wndToolBar);



	//1.�˵�����
	//ͨ����������
	//GetMenu()->GetSubMenu(0)->CheckMenuItem(0,MF_BYPOSITION|MF_CHECKED);//MF_CHECKED֮�󣬱����ʵĲ˵�ǰ�����һ���̣�MF_UNCHECKED��֮
	//ͨ��ID�ŷ���
	GetMenu()->GetSubMenu(0)->CheckMenuItem(ID_FILE_NEW,MF_BYCOMMAND|MF_CHECKED);
	
	//2.���ļ��µĴ���Ϊȱʡ�˵�����Ӵ֣�
	//ȱʡ�˵���\
	//������������
	//GetMenu()->GetSubMenu(0)->SetDefaultItem(1,TRUE);
	//ͨ��ID�ŷ���
	GetMenu()->GetSubMenu(0)->SetDefaultItem(ID_FILE_OPEN);

	//3.ͼ�α�ǲ˵�bitmap3
	
	m_bitmap.LoadBitmap(IDB_BITMAP3);
	GetMenu()->GetSubMenu(0)->SetMenuItemBitmaps(0,MF_BYPOSITION,&m_bitmap,&m_bitmap); //��ѡ��û�и�ѡ����ͬһ��λͼ
	//ע�⣺���ڱ�ǲ˵���λͼ��С����Ϊ13 x 13��bmp1��bmp2Ϊ������ĳ�Ա������Ϊ�ֲ��������˵���Ǻ�Ҫ����bmp.Detach()
	//m_bitmap.Detach();



	//4.���ò˵������ļ��µġ����桱����ʹ��
	//�ú���Ҫ��Ч��������CMainFrame��Ĺ��캯���аѳ�Ա����m_bAutoMenuEnable����ΪFALSE��Ҫʹ�ò˵�������»��ƣ������н�������ñ���Ӧ����ΪTRUE��ȱʡֵ����
	GetMenu()->GetSubMenu(0)->EnableMenuItem(2,MF_BYPOSITION | MF_DISABLED| MF_GRAYED );//��Ȼ�����԰���id�ŷ���
	

	//5.�Ƴ��ͼ��ز˵�
	//SetMenu(NULL);//5.1�Ƴ��˵�

	//5.2���ز˵�
	//���CMenu������һ����ʱ�������ڼ������֮��������menu.Detach()��
	//Detach��Ѳ˵����������˵�������룬������������ֲ�������������ڽ���ʱ��������ȥ����һ�������پ���ӵ��Ȩ�Ĳ˵���Դ
	//CMenu menu;
	//menu.LoadMenu(IDR_MAINFRAME);
	//SetMenu(&menu);
	//menu.Detach();//****
	
	



	//�˵�8:������̬�˵�
	//8.1���
	CMenu my_menu;
	my_menu.CreatePopupMenu();//����һ���յĵ����˵���my_menu����
	GetMenu()->AppendMenu( MF_POPUP,  (UINT) my_menu.m_hMenu, "��̬�˵�");
	GetMenu()->GetSubMenu(0)->AppendMenu(MF_STRING, 777, "��Ӷ���˵��µĲ˵���");
	GetMenu()->GetSubMenu(0)->AppendMenu(MF_STRING, 666, "��Ӷ���˵��µĲ˵���");
	//GetMenu()->GetSubMenu(6)->AppendMenu(MF_STRING, 999, "��Ӷ���˵��µĲ˵���");
	my_menu.Detach();
	

	//�˵�8.2����
	CMenu my_menu1;
	my_menu1.CreateMenu();
	GetMenu()->InsertMenu(2, MF_BYPOSITION | MF_POPUP, (UINT)my_menu1.m_hMenu,"���붥���˵�");
	GetMenu()->GetSubMenu(0)->InsertMenu (ID_FILE_OPEN, MF_STRING , 888, " ���붥���˵��µĲ˵���");
	my_menu1.Detach();


	//�˵�8.3ɾ��
	//GetMenu()->DeleteMenu(1, MF_BYPOSITION );//ɾ�������˵�
	//GetMenu()->GetSubMenu(0)->DeleteMenu (0, MF_BYPOSITION);//ɾ�������˵��µĲ˵���

	
	SetTimer(1,1000, NULL);

	//�Զ��幤����
	if (!m_newToolBar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | WS_VISIBLE | CBRS_RIGHT
		| CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC) ||
		!m_newToolBar.LoadToolBar(IDR_TOOLBAR1))
	{
		TRACE0("Failed to create toolbar\n");
		return -1;      // fail to create
	}
	m_newToolBar.EnableDocking(CBRS_ALIGN_ANY);
	DockControlBar(&m_newToolBar);


	//״̬��ʱ��
	
	CTime t = CTime::GetCurrentTime();
	CString str = t.Format("%H:%M:%:%S");
	CClientDC dc(this);
	CSize sz = dc.GetTextExtent(str);
	m_wndStatusBar.SetPaneInfo(1, IDS_TIMER,SBPS_NORMAL, sz.cx);
	m_wndStatusBar.SetPaneText(1, str);

	
	return 0;
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	if( !CFrameWnd::PreCreateWindow(cs) )
		return FALSE;
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return TRUE;
}

/////////////////////////////////////////////////////////////////////////////
// CMainFrame diagnostics

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CFrameWnd::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CFrameWnd::Dump(dc);
}

#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CMainFrame message handlers


void CMainFrame::OnMymenu() //�˵���Ӧ����
{
	// TODO: Add your command handler code here
	//����һ����Ӧ��
	MessageBox("�˵�ʾ���������!");
	
}

void CMainFrame::OnUpdateEditCopy(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	//�˵�6.MFC�˵��������
	//��������UI��������Ӧ���ڵ���ʽ�˵����ϵ���Ŀ����ID�ţ�������Ӧ���ڶ���˵���Ŀ����ID�ţ�
	pCmdUI->Enable(FALSE);//�˵��Ƿ����
	pCmdUI->SetCheck();
	pCmdUI->SetText("123");

}

void CMainFrame::OnShow() 
{
	// TODO: Add your command handler code here
	//�˵�7��������ݲ˵�


	//4.�����Ӧ����
	MessageBox("MainFrame show!");
}

//DEL void CMainFrame::OnDlg2() 
//DEL {
//DEL 	// TODO: Add your command handler code here
//DEL 	
//DEL }

//DEL void CMainFrame::OnDlg2() 
//DEL {
//DEL 	// TODO: Add your command handler code here
//DEL 	// TODO: Add your command handler code here
//DEL 	//��ģ̬�Ի���
//DEL 	CtestDlg2 *pDlg = new CtestDlg2();
//DEL 	pDlg->Create(IDD_DIALOG2,this);
//DEL 	//������Create����������ģʽ�Ի���ʱ������Ҫ����ShowWindow����������Ի�����ʾ������ģʽ�Ի����ã���ΪDoModal()�Ѿ����ˡ�
//DEL 	pDlg->ShowWindow(SW_SHOW);
//DEL 	//dlg.Detach();
//DEL }

void CMainFrame::OnViewNewtool() 
{
	// TODO: Add your command handler code here
	//��ʾ�������Զ��幤����
	ShowControlBar(&m_newToolBar, 
		!m_newToolBar.IsWindowVisible(), FALSE);

	

}

void CMainFrame::OnUpdateViewNewtool(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	//Ϊ�Զ���״̬���Ĳ˵�����ϸ�ѡ���
	pCmdUI->SetCheck(m_newToolBar.IsWindowVisible());
}

void CMainFrame::OnTimer(UINT nIDEvent) 
{
	// TODO: Add your message handler code here and/or call default
	//״̬����ʾʱ��	
	CTime t = CTime::GetCurrentTime();
	CString str = t.Format("%H:%M:%:%S");
	CClientDC dc(this);
	CSize sz = dc.GetTextExtent(str);
	m_wndStatusBar.SetPaneInfo(1, IDS_TIMER,SBPS_NORMAL, sz.cx);
	m_wndStatusBar.SetPaneText(1, str);
	

	CFrameWnd::OnTimer(nIDEvent);
}
