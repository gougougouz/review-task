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
	width=0;//�붨ʱ���Լ�ƽ����ɫ�й�
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
	str="�Կ���";
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
	
	//�ı������
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
	//��ˢ
	/*CBrush brush(RGB(255,0,0));
	CClientDC dc(this);
	dc.FillRect(CRect(myPoint1,point),&brush);
	*/
	//͸����ˢ
	/*CBitmap bmp;
	bmp.LoadBitmap(IDB_BITMAP1);
	CBrush brush(&bmp);*/
	//���ƾ��ο�(͸����ˢ)
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
	//������������
	CClientDC dc(this);
	CPen pen(PS_SOLID,1,RGB(255,0,0));//ѡ����ɫΪ��ɫ
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
	//�������β������
	TEXTMETRIC tm;
	dc.GetTextMetrics(&tm);
	//CreateSolidCaret(tm.tmAveCharWidth/8,tm.tmHeight);
	//ͼ�β����
	
	bitmap.LoadBitmap(IDB_BITMAP2);
	CreateCaret(&bitmap);
	ShowCaret();




	//���ö�ʱ��
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
	//����س�
	CClientDC dc(this);
	//����
	CFont font;
	font.CreatePointFont(300,"�����п�",NULL);//�������Ϊ�����п�
	CFont *oldFont= dc.SelectObject(&font);
	TEXTMETRIC tm;
	dc.GetTextMetrics(&tm);
	if(0x0d==nChar){
		strline.Empty();
		caretPos.y+=tm.tmHeight;
	}
	//�����˸��
	else if(0x08==nChar){
		//��ȡ����ɫ
		COLORREF clr =dc.SetTextColor(dc.GetBkColor());
		dc.TextOut(caretPos.x,caretPos.y,strline);
		strline=strline.Left(strline.GetLength()-1);
		dc.SetTextColor(clr);
	}/*������ͨ����*/else{
		strline+=nChar;

	}
	CSize sz =dc.GetTextExtent(strline);
	CPoint pt;
	pt.x=caretPos.x+sz.cx;
	pt.y=caretPos.y;
	SetCaretPos(pt);
	dc.TextOut(caretPos.x,caretPos.y,strline);
	
	//������Ļ�
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
	dc.DrawText(str,rect,DT_LEFT);//��������
	
	//�ı����λ��
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

	//�˵�7��������ݲ˵�
	//1.ΪMenu��������һ���µĲ˵���Դ
	//2.���������WM_RBUTTONDOWN��Ϣ��Ӧ����
	CMenu menu;
	menu.LoadMenu(IDR_MENU1);
	ClientToScreen(&point);//������Ŀͻ�ȥ����ת��Ϊ��Ļ����
	//3.����TrackPopupMenu����
	CMenu* pPopup = menu.GetSubMenu(0);

	////���ڿ�ݲ˵����������ӵ���ߴ�������Ϊ����ര�ڣ������ര�ڲ����л����öԸÿ�ݲ˵��еĲ˵����������Ӧ�����򣬾�ֻ�������ര��������Ӧ��
	//���˽�this��ΪGetParent()�������ര�ڲ����л��ᣨ�������л��ᣬ��������ж��Ҽ�����Ӧ������������������Ӧ����öԸÿ�ݲ˵��еĲ˵����������Ӧ�����򣬾�ֻ�������ര��������Ӧ��
	//�Ӵ���������Ӧ
	pPopup->TrackPopupMenu(TPM_LEFTALIGN, point.x ,point.y, this);
	CView::OnRButtonDown(nFlags, point);

	CView::OnRButtonDown(nFlags, point);
}

void CTest1View::OnShow() 
{
	// TODO: Add your command handler code here
	//�˵�7��������ݲ˵�
	//4.�����Ӧ����
	MessageBox("View show!");

}


void CTest1View::OnDilog() 
{
	// TODO: Add your command handler code here
	//�Ի���˵���Ӧ����
	//ģ̬�Ի���
	CtestDlg dlg;
	dlg.DoModal();
	

}


void CTest1View::OnDlg2() 
{
	// TODO: Add your command handler code here
	//��ģ̬�Ի���
	CtestDlg2 *pDlg = new CtestDlg2();
	pDlg->Create(IDD_DIALOG2,this);
	//������Create����������ģʽ�Ի���ʱ������Ҫ����ShowWindow����������Ի�����ʾ������ģʽ�Ի����ã���ΪDoModal()�Ѿ����ˡ�
	pDlg->ShowWindow(SW_SHOW);
	//dlg.Detach();

}
