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



	//1.菜单访问
	//通过索引访问
	//GetMenu()->GetSubMenu(0)->CheckMenuItem(0,MF_BYPOSITION|MF_CHECKED);//MF_CHECKED之后，被访问的菜单前面会有一个√，MF_UNCHECKED反之
	//通过ID号访问
	GetMenu()->GetSubMenu(0)->CheckMenuItem(ID_FILE_NEW,MF_BYCOMMAND|MF_CHECKED);
	
	//2.把文件下的打开作为缺省菜单项（即加粗）
	//缺省菜单项\
	//根据索引访问
	//GetMenu()->GetSubMenu(0)->SetDefaultItem(1,TRUE);
	//通过ID号访问
	GetMenu()->GetSubMenu(0)->SetDefaultItem(ID_FILE_OPEN);

	//3.图形标记菜单bitmap3
	
	m_bitmap.LoadBitmap(IDB_BITMAP3);
	GetMenu()->GetSubMenu(0)->SetMenuItemBitmaps(0,MF_BYPOSITION,&m_bitmap,&m_bitmap); //复选和没有复选的用同一个位图
	//注意：用于标记菜单的位图大小必须为13 x 13。bmp1、bmp2为主框类的成员对象；若为局部变量，菜单标记后要加上bmp.Detach()
	//m_bitmap.Detach();



	//4.禁用菜单项让文件下的“保存”不能使用
	//该函数要生效，必须在CMainFrame类的构造函数中把成员变量m_bAutoMenuEnable设置为FALSE。要使用菜单命令更新机制（后面有讲），则该变量应设置为TRUE（缺省值）。
	GetMenu()->GetSubMenu(0)->EnableMenuItem(2,MF_BYPOSITION | MF_DISABLED| MF_GRAYED );//当然还可以按照id号访问
	

	//5.移除和加载菜单
	//SetMenu(NULL);//5.1移除菜单

	//5.2加载菜单
	//如果CMenu对象是一个临时对象，则在加载完成之后必须加上menu.Detach()。
	//Detach会把菜单句柄与这个菜单对象分离，这样，当这个局部对象的生命周期结束时，它不会去销毁一个它不再具有拥有权的菜单资源
	//CMenu menu;
	//menu.LoadMenu(IDR_MAINFRAME);
	//SetMenu(&menu);
	//menu.Detach();//****
	
	



	//菜单8:操作动态菜单
	//8.1添加
	CMenu my_menu;
	my_menu.CreatePopupMenu();//创建一个空的弹出菜单和my_menu关联
	GetMenu()->AppendMenu( MF_POPUP,  (UINT) my_menu.m_hMenu, "动态菜单");
	GetMenu()->GetSubMenu(0)->AppendMenu(MF_STRING, 777, "添加顶层菜单下的菜单项");
	GetMenu()->GetSubMenu(0)->AppendMenu(MF_STRING, 666, "添加顶层菜单下的菜单项");
	//GetMenu()->GetSubMenu(6)->AppendMenu(MF_STRING, 999, "添加顶层菜单下的菜单项");
	my_menu.Detach();
	

	//菜单8.2插入
	CMenu my_menu1;
	my_menu1.CreateMenu();
	GetMenu()->InsertMenu(2, MF_BYPOSITION | MF_POPUP, (UINT)my_menu1.m_hMenu,"插入顶级菜单");
	GetMenu()->GetSubMenu(0)->InsertMenu (ID_FILE_OPEN, MF_STRING , 888, " 插入顶级菜单下的菜单项");
	my_menu1.Detach();


	//菜单8.3删除
	//GetMenu()->DeleteMenu(1, MF_BYPOSITION );//删除顶级菜单
	//GetMenu()->GetSubMenu(0)->DeleteMenu (0, MF_BYPOSITION);//删除顶级菜单下的菜单项

	
	SetTimer(1,1000, NULL);

	//自定义工具栏
	if (!m_newToolBar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | WS_VISIBLE | CBRS_RIGHT
		| CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC) ||
		!m_newToolBar.LoadToolBar(IDR_TOOLBAR1))
	{
		TRACE0("Failed to create toolbar\n");
		return -1;      // fail to create
	}
	m_newToolBar.EnableDocking(CBRS_ALIGN_ANY);
	DockControlBar(&m_newToolBar);


	//状态栏时钟
	
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


void CMainFrame::OnMymenu() //菜单响应函数
{
	// TODO: Add your command handler code here
	//弹出一个响应框
	MessageBox("菜单示例被点击了!");
	
}

void CMainFrame::OnUpdateEditCopy(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	//菜单6.MFC菜单命令更新
	//更新命令UI处理程序仅应用于弹出式菜单项上的项目（有ID号），不能应用于顶层菜单项目（无ID号）
	pCmdUI->Enable(FALSE);//菜单是否可用
	pCmdUI->SetCheck();
	pCmdUI->SetText("123");

}

void CMainFrame::OnShow() 
{
	// TODO: Add your command handler code here
	//菜单7：制作快捷菜单


	//4.添加响应函数
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
//DEL 	//非模态对话框
//DEL 	CtestDlg2 *pDlg = new CtestDlg2();
//DEL 	pDlg->Create(IDD_DIALOG2,this);
//DEL 	//当利用Create函数创建无模式对话框时，还需要调用ShowWindow函数将这个对话框显示出来。模式对话框不用，因为DoModal()已经做了。
//DEL 	pDlg->ShowWindow(SW_SHOW);
//DEL 	//dlg.Detach();
//DEL }

void CMainFrame::OnViewNewtool() 
{
	// TODO: Add your command handler code here
	//显示和隐藏自定义工具栏
	ShowControlBar(&m_newToolBar, 
		!m_newToolBar.IsWindowVisible(), FALSE);

	

}

void CMainFrame::OnUpdateViewNewtool(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	//为自定义状态栏的菜单项加上复选标记
	pCmdUI->SetCheck(m_newToolBar.IsWindowVisible());
}

void CMainFrame::OnTimer(UINT nIDEvent) 
{
	// TODO: Add your message handler code here and/or call default
	//状态栏显示时钟	
	CTime t = CTime::GetCurrentTime();
	CString str = t.Format("%H:%M:%:%S");
	CClientDC dc(this);
	CSize sz = dc.GetTextExtent(str);
	m_wndStatusBar.SetPaneInfo(1, IDS_TIMER,SBPS_NORMAL, sz.cx);
	m_wndStatusBar.SetPaneText(1, str);
	

	CFrameWnd::OnTimer(nIDEvent);
}
