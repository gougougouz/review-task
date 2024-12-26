// Zjw002View.cpp : implementation of the CZjw002View class
//

#include "stdafx.h"
#include "Zjw002.h"

#include "Zjw002Doc.h"
#include "Zjw002View.h"
#include "HistogramDlg.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CZjw002View

IMPLEMENT_DYNCREATE(CZjw002View, CScrollView)

BEGIN_MESSAGE_MAP(CZjw002View, CScrollView)
	//{{AFX_MSG_MAP(CZjw002View)
	ON_COMMAND(ID_GRAY, OnGray)
	ON_UPDATE_COMMAND_UI(ID_GRAY, OnUpdateGray)
	ON_WM_MOUSEMOVE()
	ON_COMMAND(ID_Histogram, OnHistogram)
	ON_UPDATE_COMMAND_UI(ID_Histogram, OnUpdateHistogram)
	ON_COMMAND(ID_EQUALIZE, OnEqualize)
	ON_UPDATE_COMMAND_UI(ID_EQUALIZE, OnUpdateEqualize)
	ON_COMMAND(ID_FT, OnFt)
	ON_UPDATE_COMMAND_UI(ID_FT, OnUpdateFt)
	ON_COMMAND(ID_IFT, OnIft)
	ON_UPDATE_COMMAND_UI(ID_IFT, OnUpdateIft)
	ON_COMMAND(ID_FFT, OnFft)
	ON_UPDATE_COMMAND_UI(ID_FFT, OnUpdateFft)
	ON_COMMAND(ID_IFFT, OnIfft)
	ON_UPDATE_COMMAND_UI(ID_IFFT, OnUpdateIfft)
	ON_COMMAND(ID_AVG_FILTER, OnAvgFilter)
	ON_UPDATE_COMMAND_UI(ID_AVG_FILTER, OnUpdateAvgFilter)
	ON_COMMAND(ID_GRAD_SHARP, OnGradSharp)
	ON_UPDATE_COMMAND_UI(ID_GRAD_SHARP, OnUpdateGradSharp)
	ON_COMMAND(ID_MED_FILETER, OnMedFileter)
	ON_UPDATE_COMMAND_UI(ID_MED_FILETER, OnUpdateMedFileter)
	ON_COMMAND(ID_RAPLAS_SHARP, OnRaplasSharp)
	ON_UPDATE_COMMAND_UI(ID_RAPLAS_SHARP, OnUpdateRaplasSharp)
	ON_COMMAND(ID_LINETRANS, OnLinetrans)
	ON_UPDATE_COMMAND_UI(ID_LINETRANS, OnUpdateLinetrans)
	ON_COMMAND(ID_FFT_FILTER, OnFftFilter)
	ON_UPDATE_COMMAND_UI(ID_FFT_FILTER, OnUpdateFftFilter)
	ON_COMMAND(ID_GLP_FILTER, OnGlpFilter)
	ON_UPDATE_COMMAND_UI(ID_GLP_FILTER, OnUpdateGlpFilter)
	ON_COMMAND(ID_ILP_FILTER, OnIlpFilter)
	ON_UPDATE_COMMAND_UI(ID_ILP_FILTER, OnUpdateIlpFilter)
	//}}AFX_MSG_MAP
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, CScrollView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, CScrollView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, CScrollView::OnFilePrintPreview)
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CZjw002View construction/destruction

CZjw002View::CZjw002View()
{	
	// TODO: add construction code here

}

CZjw002View::~CZjw002View()
{
}

BOOL CZjw002View::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CScrollView::PreCreateWindow(cs);
}

/////////////////////////////////////////////////////////////////////////////
// CZjw002View drawing
extern BITMAPINFO* lpBitsInfo; 
extern BITMAPINFO* lpDIB_FT ;
extern BITMAPINFO* lpDIB_IFT;
void CZjw002View::OnDraw(CDC* pDC)
{
	CZjw002Doc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	// TODO: add draw code for native data here
	if (NULL == lpBitsInfo)
		return;	
	LPVOID lpBits =
		(LPVOID)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];
	StretchDIBits(
		pDC->GetSafeHdc(),
		0,0,lpBitsInfo->bmiHeader.biWidth, lpBitsInfo->bmiHeader.biHeight,
		0,0,lpBitsInfo->bmiHeader.biWidth, lpBitsInfo->bmiHeader.biHeight,
		lpBits,
		lpBitsInfo,
		DIB_RGB_COLORS,
		SRCCOPY);

	if(lpDIB_FT){
		LPVOID lpBits =
		(LPVOID)&lpDIB_FT->bmiColors[lpDIB_FT->bmiHeader.biClrUsed];
	StretchDIBits(
		pDC->GetSafeHdc(),
		600,0,lpDIB_FT->bmiHeader.biWidth, lpDIB_FT->bmiHeader.biHeight,
		0,0,lpDIB_FT->bmiHeader.biWidth, lpDIB_FT->bmiHeader.biHeight,
		lpBits,
		lpDIB_FT,
		DIB_RGB_COLORS,
		SRCCOPY);	
	}
	if(lpDIB_IFT){
		LPVOID lpBits =
		(LPVOID)&lpDIB_IFT->bmiColors[lpDIB_IFT->bmiHeader.biClrUsed];
	StretchDIBits(
		pDC->GetSafeHdc(),
		0,600,lpDIB_IFT->bmiHeader.biWidth, lpDIB_IFT->bmiHeader.biHeight,
		0,0,lpDIB_IFT->bmiHeader.biWidth, lpDIB_IFT->bmiHeader.biHeight,
		lpBits,
		lpDIB_IFT,//
		DIB_RGB_COLORS,
		SRCCOPY);
	}
}

void CZjw002View::OnInitialUpdate()
{
	CScrollView::OnInitialUpdate();

	CSize sizeTotal;
	// TODO: calculate the total size of this view
	sizeTotal.cx = sizeTotal.cy = 1200;
	SetScrollSizes(MM_TEXT, sizeTotal);
}

/////////////////////////////////////////////////////////////////////////////
// CZjw002View printing

BOOL CZjw002View::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CZjw002View::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CZjw002View::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}

/////////////////////////////////////////////////////////////////////////////
// CZjw002View diagnostics

#ifdef _DEBUG
void CZjw002View::AssertValid() const
{
	CScrollView::AssertValid();
}

void CZjw002View::Dump(CDumpContext& dc) const
{
	CScrollView::Dump(dc);
}

CZjw002Doc* CZjw002View::GetDocument() // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CZjw002Doc)));
	return (CZjw002Doc*)m_pDocument;
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CZjw002View message handlers
void Gray();
BOOL IsGray();
void CZjw002View::OnGray() 
{
	// TODO: Add your command handler code here
	Gray();
	Invalidate();
	
}

void CZjw002View::OnUpdateGray(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpBitsInfo != NULL);
	
}
void pixel(int i,int j,char* str);

void CZjw002View::OnMouseMove(UINT nFlags, CPoint point) 
{
	// TODO: Add your message handler code here and/or call default
	char xy[100];
	memset(xy,0,100);
	sprintf(xy,"x:%d y:%d",point.x,point.y);

	char rgb[100];
	memset(rgb,0,100);
	pixel(point.x,point.y,rgb);
	
	strcat(xy, rgb);

	((CFrameWnd*)GetParent())->SetMessageText(xy);
	
	CScrollView::OnMouseMove(nFlags, point);
}

void CZjw002View::OnHistogram() 
{
	// TODO: Add your command handler code here
	CHistogramDlg dlg;
	
	dlg.DoModal();
}

void CZjw002View::OnUpdateHistogram(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpBitsInfo != NULL );
}

void LineTrans(float a,float b);
void CZjw002View::OnLinetrans() 
{
	// TODO: Add your command handler code here
	LineTrans(1.5,-50);
	Invalidate();
}

void CZjw002View::OnUpdateLinetrans(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpBitsInfo != NULL&& IsGray() );
}
//DEL void CZjw002View::OnLinetrans() 
//DEL {
//DEL 	// TODO: Add your command handler code here
//DEL 	LineTrans(1.5,-50);
//DEL 	Invalidate();
//DEL }


//DEL void CZjw002View::OnUpdateLinetrans(CCmdUI* pCmdUI) 
//DEL {
//DEL 	// TODO: Add your command update UI handler code here
//DEL 	pCmdUI->Enable(lpBitsInfo != NULL&& IsGray() );
//DEL }
void Eql();
void CZjw002View::OnEqualize() 
{
	// TODO: Add your command handler code here
	Eql();
	Invalidate();
}

void CZjw002View::OnUpdateEqualize(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpBitsInfo != NULL);
}

void Fourier();
void CZjw002View::OnFt() 
{
	// TODO: Add your command handler code here
	Fourier();
	Invalidate();
}

void CZjw002View::OnUpdateFt(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpBitsInfo != NULL&& IsGray());
}
void IFourier();
void CZjw002View::OnIft() 
{
	// TODO: Add your command handler code here
	IFourier();
	Invalidate();
}

BOOL is_gFD_OK();
void CZjw002View::OnUpdateIft(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpBitsInfo != NULL && IsGray() && is_gFD_OK());
}

void FFourier();
void CZjw002View::OnFft() 
{
	// TODO: Add your command handler code here
	if (lpDIB_FT)
	{
		free(lpDIB_FT);
		lpDIB_FT = NULL;
	}

	if (lpDIB_IFT)
	{
		free(lpDIB_IFT);
		lpDIB_IFT = NULL;
	} 

	FFourier();
	Invalidate();

}

void CZjw002View::OnUpdateFft(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpBitsInfo != NULL && IsGray()); 
}

void IFFourier();
void CZjw002View::OnIfft() 
{
	// TODO: Add your command handler code here
	
	if (lpDIB_IFT)
	{
		free(lpDIB_IFT);
		lpDIB_IFT = NULL;
	}

	IFFourier();
	Invalidate();

}

void CZjw002View::OnUpdateIfft(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(is_gFD_OK());
}


void AvgFilter();
void MedFilter();
void GradSharp();
void RaplasSharp();
void CZjw002View::OnAvgFilter() 
{
	// TODO: Add your command handler code here
	AvgFilter();
	Invalidate();
}

void CZjw002View::OnUpdateAvgFilter(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpBitsInfo != NULL && IsGray());
}

void CZjw002View::OnGradSharp() 
{

	// TODO: Add your command handler code here
	GradSharp();
	Invalidate();
}

void CZjw002View::OnUpdateGradSharp(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpBitsInfo != NULL && IsGray());
}

void CZjw002View::OnMedFileter() 
{
	// TODO: Add your command handler code here
	MedFilter();
	Invalidate();
}

void CZjw002View::OnUpdateMedFileter(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpBitsInfo != NULL && IsGray());
}

void CZjw002View::OnRaplasSharp() 
{
	// TODO: Add your command handler code here
	RaplasSharp();
	Invalidate();
}

void CZjw002View::OnUpdateRaplasSharp(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpBitsInfo != NULL && IsGray());
}


void FFT_Filter(int);                       //截至半径
void CZjw002View::OnFftFilter() 
{
	// TODO: Add your command handler code here
	FFT_Filter(-20);
	Invalidate();
}

void CZjw002View::OnUpdateFftFilter(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpDIB_FT != NULL && is_gFD_OK());          //经过傅里叶变换
}



void GLP_Filter(int);
void CZjw002View::OnGlpFilter() 
{
	// TODO: Add your command handler code here
	GLP_Filter(-20);
	Invalidate();
}

void CZjw002View::OnUpdateGlpFilter(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpDIB_FT != NULL && is_gFD_OK());
	
}





void ILP_Filter(int);
void CZjw002View::OnIlpFilter() 
{
	// TODO: Add your command handler code here
	ILP_Filter(-20);
	Invalidate();
}

void CZjw002View::OnUpdateIlpFilter(CCmdUI* pCmdUI) 
{
	// TODO: Add your command update UI handler code here
	pCmdUI->Enable(lpDIB_FT != NULL && is_gFD_OK());
}
