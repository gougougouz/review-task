// test1View.cpp : implementation of the CTest1View class
//

#include "stdafx.h"
#include "test1.h"

#include "test1Doc.h"
#include "test1View.h"

#include"testDlg.h"
#include"testDlg2.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CTest1View

IMPLEMENT_DYNCREATE(CTest1View, CView)

BEGIN_MESSAGE_MAP(CTest1View, CView)
	//{{AFX_MSG_MAP(CTest1View)
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_MOUSEMOVE()
	ON_WM_CREATE()
	ON_WM_CHAR()
	ON_WM_TIMER()
	ON_WM_RBUTTONDOWN()
	ON_COMMAND(IDM_SHOW, OnShow)
	ON_COMMAND(IDM_DILOG, OnDilog)
	ON_COMMAND(IDM_DLG2, OnDlg2)
	//}}AFX_MSG_MAP
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, CView::OnFilePrintPreview)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CTest1View construction/destruction

CTest1View::CTest1View()
{
	// TODO: add construction code here
	strline="";
	lb=FALSE;
	caretPos=0;
	width=0;//与定时器以及平滑变色有关
}

CTest1View::~CTest1View()
{
}

BOOL CTest1View::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

/////////////////////////////////////////////////////////////////////////////
// CTest1View drawing

void CTest1View::OnDraw(CDC* pDC)
{
	CTest1Doc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	CString str;
	str="赵俊文";
	pDC->TextOut(50,50,str);
	CSize sz=pDC->GetTextExtent(str);
	
	
	str.LoadString(IDS_ZHAOJUNWEN);
	pDC->TextOut(0,200,str);
	
	pDC->BeginPath();
	pDC->Rectangle(50,50,50+sz.cx,50+sz.cy);
	pDC->EndPath();
	// TODO: add draw code for native data here
}

/////////////////////////////////////////////////////////////////////////////
// CTest1View printing

BOOL CTest1View::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CTest1View::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CTest1View::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}

/////////////////////////////////////////////////////////////////////////////
// CTest1View diagnostics

#ifdef _DEBUG
void CTest1View::AssertValid() const
{
	CView::AssertValid();
}

void CTest1View::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CTest1Doc* CTest1View::GetDocument() // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CTest1Doc)));
	return (CTest1Doc*)m_pDocument;
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CTest1View message handlers

void CTest1View::OnLButtonDown(UINT nFlags, CPoint point) 
{
	// TODO: Add your message handler code here and/or call default
	myPoint1=point;
	lb=TRUE;
	
	//文本输出用
	SetCaretPos(point);
	caretPos=point;
	strline.Empty();
	CView::OnLButtonDown(nFlags, point);
	
}

void CTest1View::OnLButtonUp(UINT nFlags, CPoint point) 
{
	// TODO: Add your message handler code here and/or call default
	/*CWindowDC dc(NULL);
	dc.MoveTo(myPoint1);
	dc.LineTo(point);*/

	/*CPen pen(PS_SOLID,1,RGB(128,0,128));
	CClientDC dc(this);
	CPen *prePen = dc.SelectObject(&pen);
	dc.MoveTo(myPoint1);
	dc.LineTo(point);
	dc.SelectObject(prePen);*/
	//画刷
	/*CBrush brush(RGB(255,0,0));
	CClientDC dc(this);
	dc.FillRect(CRect(myPoint1,point),&brush);
	*/
	//透明画刷
	/*CBitmap bmp;
	bmp.LoadBitmap(IDB_BITMAP1);
	CBrush brush(&bmp);*/
	//绘制矩形框(透明化刷)
	/*CClientDC dc(this);
	CBrush *brush=CBrush::FromHandle((HBRUSH)GetStockObject(NULL_BRUSH));
	CBrush *preBrush =dc.SelectObject(brush);
	dc.Rectangle(CRect(myPoint1,point));*/
	lb=FALSE;
	CView::OnLButtonUp(nFlags, point);

}

void CTest1View::OnMouseMove(UINT nFlags, CPoint point) 
{
	// TODO: Add your message handler code here and/or call default
	//绘制连续曲线
	CClientDC dc(this);
	CPen pen(PS_SOLID,1,RGB(255,0,0));//选择颜色为红色
	CPen *prePen = dc.SelectObject(&pen);
	if(lb==TRUE){
		dc.MoveTo(myPoint1);
		dc.LineTo(point);
		myPoint1=point;
	}
	CView::OnMouseMove(nFlags, point);
}

int CTest1View::OnCreate(LPCREATESTRUCT lpCreateStruct) 
{
	if (CView::OnCreate(lpCreateStruct) == -1)
		return -1;
	CClientDC dc(this);
	//创建条形插入符号
	TEXTMETRIC tm;
	dc.GetTextMetrics(&tm);
	//CreateSolidCaret(tm.tmAveCharWidth/8,tm.tmHeight);
	//图形插入符
	
	bitmap.LoadBitmap(IDB_BITMAP2);
	CreateCaret(&bitmap);
	ShowCaret();




	//设置定时器
	SetTimer(1,100,NULL);
	return 0;
}



//DEL void CTest1View::OnCancelMode() 
//DEL {
//DEL 	CView::OnCancelMode();
//DEL 	
//DEL 	// TODO: Add your message handler code here
//DEL 	
//DEL }

void CTest1View::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags) 
{
	// TODO: Add your message handler code here and/or call default
	//处理回车
	CClientDC dc(this);
	//字体
	CFont font;
	font.CreatePointFont(300,"华文行楷",NULL);//将字体改为华文行楷
	CFont *oldFont= dc.SelectObject(&font);
	TEXTMETRIC tm;
	dc.GetTextMetrics(&tm);
	if(0x0d==nChar){
		strline.Empty();
		caretPos.y+=tm.tmHeight;
	}
	//处理退格键
	else if(0x08==nChar){
		//获取背景色
		COLORREF clr =dc.SetTextColor(dc.GetBkColor());
		dc.TextOut(caretPos.x,caretPos.y,strline);
		strline=strline.Left(strline.GetLength()-1);
		dc.SetTextColor(clr);
	}/*其他普通按键*/else{
		strline+=nChar;

	}
	CSize sz =dc.GetTextExtent(strline);
	CPoint pt;
	pt.x=caretPos.x+sz.cx;
	pt.y=caretPos.y;
	SetCaretPos(pt);
	dc.TextOut(caretPos.x,caretPos.y,strline);
	
	//将字体改回
	dc.SelectObject(oldFont);

	CView::OnChar(nChar, nRepCnt, nFlags);
}

//DEL void CTest1View::OnCancelMode() 
//DEL {
//DEL 	CView::OnCancelMode();
//DEL 	
//DEL 	// TODO: Add your message handler code here
//DEL 	
//DEL }

void CTest1View::OnTimer(UINT nIDEvent) 
{
	// TODO: Add your message handler code here and/or call default
	
	width+=5;
	CClientDC dc(this);
	TEXTMETRIC tm;
	dc.GetTextMetrics(&tm);
	CRect rect ;
	rect.left=0;
	rect.top=200;
	rect.right=width;
	rect.bottom=rect.top+tm.tmHeight;
	dc.SetTextColor(RGB(255,0,0));
	
	CString str;
	str.LoadString(IDS_ZHAOJUNWEN);
	dc.DrawText(str,rect,DT_LEFT);//从左往右
	
	//改变矩形位置
	rect.top=140;
	
	dc.DrawText(str,rect,DT_RIGHT);
	CSize sz = dc.GetTextExtent(str);
	if(width>sz.cx){
		width=0;
		dc.SetTextColor(RGB(0,255,0));
		dc.TextOut(0,200,str);
	}
	CView::OnTimer(nIDEvent);
}

void CTest1View::OnRButtonDown(UINT nFlags, CPoint point) 
{	
	// TODO: Add your message handler code here and/or call default

	//菜单7：制作快捷菜单
	//1.为Menu程序增加一个新的菜单资源
	//2.给视类添加WM_RBUTTONDOWN消息响应函数
	CMenu menu;
	menu.LoadMenu(IDR_MENU1);
	ClientToScreen(&point);//将鼠标点的客户去坐标转换为屏幕坐标
	//3.调用TrackPopupMenu函数
	CMenu* pPopup = menu.GetSubMenu(0);

	////对于快捷菜单，如果将其拥有者窗口设置为框架类窗口，则框架类窗口才能有机会获得对该快捷菜单中的菜单项的命令响应，否则，就只能有视类窗口作出响应。
	//这了将this改为GetParent()，则框架类窗口才能有机会（仅仅是有机会，如果视类有对右键的响应函数，则还是由视类响应）获得对该快捷菜单中的菜单项的命令响应，否则，就只能有视类窗口作出响应。
	//子窗口优先响应
	pPopup->TrackPopupMenu(TPM_LEFTALIGN, point.x ,point.y, this);
	CView::OnRButtonDown(nFlags, point);

	CView::OnRButtonDown(nFlags, point);
}

void CTest1View::OnShow() 
{
	// TODO: Add your command handler code here
	//菜单7：制作快捷菜单
	//4.添加响应函数
	MessageBox("View show!");

}


void CTest1View::OnDilog() 
{
	// TODO: Add your command handler code here
	//对话框菜单响应函数
	//模态对话框
	CtestDlg dlg;
	dlg.DoModal();
	

}


void CTest1View::OnDlg2() 
{
	// TODO: Add your command handler code here
	//非模态对话框
	CtestDlg2 *pDlg = new CtestDlg2();
	pDlg->Create(IDD_DIALOG2,this);
	//当利用Create函数创建无模式对话框时，还需要调用ShowWindow函数将这个对话框显示出来。模式对话框不用，因为DoModal()已经做了。
	pDlg->ShowWindow(SW_SHOW);
	//dlg.Detach();

}
