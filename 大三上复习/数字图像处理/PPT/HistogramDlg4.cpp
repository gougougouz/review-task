// HistogramDlg.cpp : implementation file
//

#include "stdafx.h"
#include "zby274.h"
#include "HistogramDlg.h"
#include "windows.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CHistogramDlg dialog


CHistogramDlg::CHistogramDlg(CWnd* pParent /*=NULL*/)
	: CDialog(CHistogramDlg::IDD, pParent)
{
	//{{AFX_DATA_INIT(CHistogramDlg)
		// NOTE: the ClassWizard will add member initialization here
	//}}AFX_DATA_INIT
}


void CHistogramDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(CHistogramDlg)
		// NOTE: the ClassWizard will add DDX and DDV calls here
	//}}AFX_DATA_MAP
}


BEGIN_MESSAGE_MAP(CHistogramDlg, CDialog)
	//{{AFX_MSG_MAP(CHistogramDlg)
	ON_WM_PAINT()
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CHistogramDlg message handlers
extern DWORD H[256];
extern DWORD HR[256],HB[256],HG[256];
BOOL IsGray();
void His();

BOOL CHistogramDlg::OnInitDialog() 
{
	CDialog::OnInitDialog();
	// TODO: Add extra initialization here
	His();
	return TRUE;  // return TRUE unless you set the focus to a control
	              // EXCEPTION: OCX Property Pages should return FALSE
}

void CHistogramDlg::OnPaint() 
{
	CPaintDC dc(this); // device context for painting
	if(IsGray()){
		// TODO: Add your message handler code here
		dc.Rectangle(20,20,287,221);
		int i;
		DWORD graymax;
		graymax = 0;

		for (i = 0; i < 256; i ++)
		{
			if (H[i] > graymax)
				graymax = H[i];
		}

		for (i = 0; i < 256; i++)
		{
			dc.MoveTo(i + 20, 220);
			dc.LineTo(i + 20, 220 - (int)H[i] * 200 / graymax);
		}
	}
	else{
		int i;
		DWORD maxRed = 0, maxGreen = 0, maxBlue = 0;
		int offsetX = 0; // 每个直方图的横向偏移量
		dc.Rectangle(offsetX + 20, 20, offsetX + 277, 221);
		for (i = 0; i < 256; i++) {
			if (HR[i] > maxRed) maxRed = HR[i];
		}

		for (i = 0; i < 256; i++) {
			dc.MoveTo(i + offsetX + 20, 220);
			dc.LineTo(i + offsetX + 20, 220 - (int)(HR[i] * 200 / maxRed));
		}

		// 绘制绿色通道直方图
		offsetX += 300; // 更新偏移量
		dc.Rectangle(offsetX + 20, 20, offsetX + 277, 221);
		for (i = 0; i < 256; i++) {
			if (HG[i] > maxGreen) maxGreen = HG[i];
		}


		for (i = 0; i < 256; i++) {
			dc.MoveTo(i + offsetX + 20, 220);
			dc.LineTo(i + offsetX + 20, 220 - (int)(HG[i] * 200 / maxGreen));
		}

		// 绘制蓝色通道直方图
		offsetX += 300; // 更新偏移量
		dc.Rectangle(offsetX + 20, 20, offsetX + 277, 221);
		for (i = 0; i < 256; i++) {
			if (HB[i] > maxBlue) maxBlue = HB[i];
		}
		for (i = 0; i < 256; i++) {
			dc.MoveTo(i + offsetX + 20, 220);
			dc.LineTo(i + offsetX + 20, 220 - (int)(HB[i] * 200 / maxBlue));
		}
	}
	// Do not call CDialog::OnPaint() for painting messages
}
