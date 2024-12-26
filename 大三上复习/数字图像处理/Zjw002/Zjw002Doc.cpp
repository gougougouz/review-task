// Zjw002Doc.cpp : implementation of the CZjw002Doc class
//

#include "stdafx.h"
#include "Zjw002.h"

#include "Zjw002Doc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CZjw002Doc

IMPLEMENT_DYNCREATE(CZjw002Doc, CDocument)

BEGIN_MESSAGE_MAP(CZjw002Doc, CDocument)
	//{{AFX_MSG_MAP(CZjw002Doc)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/////////////////////////////////////////////////////////////////////////////
// CZjw002Doc construction/destruction

CZjw002Doc::CZjw002Doc()
{
	// TODO: add one-time construction code here

}

CZjw002Doc::~CZjw002Doc()
{
}

BOOL CZjw002Doc::OnNewDocument()
{
	if (!CDocument::OnNewDocument())
		return FALSE;

	// TODO: add reinitialization code here
	// (SDI documents will reuse this document)

	return TRUE;
}



/////////////////////////////////////////////////////////////////////////////
// CZjw002Doc serialization

void CZjw002Doc::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		// TODO: add storing code here
	}
	else
	{
		// TODO: add loading code here
	}
}

/////////////////////////////////////////////////////////////////////////////
// CZjw002Doc diagnostics

#ifdef _DEBUG
void CZjw002Doc::AssertValid() const
{
	CDocument::AssertValid();
}

void CZjw002Doc::Dump(CDumpContext& dc) const
{
	CDocument::Dump(dc);
}
#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CZjw002Doc commands

//DEL void CZjw002Doc::OnFileOpen() 
//DEL {
//DEL 	// TODO: Add your command handler code here
//DEL 	
//DEL }
BOOL LoadBmpFile(char *);
BOOL CZjw002Doc::OnOpenDocument(LPCTSTR lpszPathName) 
{
	if (!CDocument::OnOpenDocument(lpszPathName))
		return FALSE;
	
	// TODO: Add your specialized creation code here
	LoadBmpFile((char*)lpszPathName);
	return TRUE;
}

//DEL void CZjw002Doc::OnUpdateFileOpen(CCmdUI* pCmdUI) 
//DEL {
//DEL 	// TODO: Add your command update UI handler code here
//DEL 	
//DEL }
