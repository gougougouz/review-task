#include "stdafx.h"
#include "windows.h"

#include <complex>
using namespace std;
#define PI 3.1415926535
BITMAPINFO* lpBitsInfo = NULL; 

//����Ҷ�任���ͼ��ָ��
BITMAPINFO* lpDIB_FT = NULL;
BITMAPINFO* lpDIB_IFT = NULL;

complex <double> *gFD =NULL;

BITMAPINFOHEADER bi;
BOOL LoadBmpFile(char* BmpFileName){
	FILE* fp;
	if (NULL==(fp=fopen(BmpFileName,"rb")))
		return FALSE;

	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;
	
	fread(&bf,14,1,fp);
	fread(&bi,40,1,fp);

	DWORD NumColors;
	if (bi.biClrUsed!=0)
		NumColors = bi.biClrUsed;
	else
	{
		switch(bi.biBitCount)
		{
		case 1:
			NumColors = 2;
			break;
		case 4:
			NumColors = 16;
			break;
		case 8:
			NumColors = 256;
			break;
		case 24:
			NumColors = 0;
			break;
		}
	}

	DWORD PalSize = NumColors * 4;
	DWORD ImgSize = (bi.biWidth * bi.biBitCount + 31) / 32 * 4 * bi.biHeight;
	DWORD Size = 40 + PalSize + ImgSize;

	if (NULL == (lpBitsInfo = (BITMAPINFO*)malloc(Size)))
		return FALSE;

	fseek(fp,14,SEEK_SET);
	fread((char*)lpBitsInfo,Size,1,fp);

	lpBitsInfo->bmiHeader.biClrUsed = NumColors;

	return TRUE;
}
void Gray(){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader. biClrUsed];

	int LineBytes_gray =(w *8+31)/32 *4;
	BITMAPINFO* lpBitsInfo_gray=(BITMAPINFO*) malloc(40 + 1024 + LineBytes_gray * h);
	memcpy(lpBitsInfo_gray,lpBitsInfo, 40);

	lpBitsInfo_gray->bmiHeader.biBitCount = 8;
	lpBitsInfo_gray->bmiHeader.biClrUsed = 256;

	int i,j;
	for(i=0;i<256;i++)
    {
		lpBitsInfo_gray->bmiColors[i].rgbRed =i;
		lpBitsInfo_gray->bmiColors[i].rgbGreen=i;
		lpBitsInfo_gray->bmiColors[i].rgbBlue=i;
		lpBitsInfo_gray->bmiColors[i].rgbReserved=0;
	}
	

	BYTE* lpBits_gray=(BYTE*) &lpBitsInfo_gray->bmiColors[256];
	BYTE *R ,*G,*B,avg,*pixel;
	for(i=0;i<h;i++){
		for(j=0;j<w;j++){
			B=lpBits+LineBytes*(h-i-1)+j*3;
			G=B+1;
			R=G+1;
			avg=(*R+*G+*B)/3;
			pixel=lpBits_gray+LineBytes_gray*(h-i-1)+j;
			*pixel=avg;
		}
	}

	free(lpBitsInfo);
	lpBitsInfo=lpBitsInfo_gray;
}


BOOL IsGray(){
	int r,g,b;
	if(lpBitsInfo->bmiHeader.biBitCount){
		r=lpBitsInfo->bmiColors[150].rgbRed;
		g=lpBitsInfo->bmiColors[150].rgbGreen;
		b=lpBitsInfo->bmiColors[150].rgbBlue;
		if(r==b&&r==g)
			return TRUE;

	}
	return FALSE;

	
}


void pixel(int i,int j,char* str)
{
	if(NULL == lpBitsInfo)
		return;

    int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	if (i >= h || j >=w)
		return;

	BYTE* pixel, bv;
	int r,g,b;
	int colorIdx = 0;

	switch(lpBitsInfo->bmiHeader.biBitCount)
	{
	case 8:

		pixel = lpBits + LineBytes * (h - 1 - i) + j;
		if (IsGray())
            sprintf(str,"�Ҷ�:%d", *pixel);
		else
		{

			r = lpBitsInfo->bmiColors[*pixel].rgbRed;
			g = lpBitsInfo->bmiColors[*pixel].rgbGreen;
			b = lpBitsInfo->bmiColors[*pixel].rgbBlue;
			sprintf(str,"RGB(%d, %d, %d)", r,g,b);
		}
		break;
	case 24:
		pixel = lpBits + LineBytes * (h - 1 - i) + j * 3;
		r = pixel[0];
		g = pixel[1];
		b = pixel[2];
		sprintf(str,"RGB(%d, %d, %d)", r,g,b);
		break;
	case 4:
		bv = *(lpBits + LineBytes * (h - 1 - i) + j / 2);
		colorIdx = (j % 2 == 0) ? (bv >> 4) : (bv & 0x0f);
		r = lpBitsInfo->bmiColors[colorIdx].rgbRed;
		g = lpBitsInfo->bmiColors[colorIdx].rgbGreen;
		b = lpBitsInfo->bmiColors[colorIdx].rgbBlue;
		sprintf(str,"RGB(%d, %d, %d)", r,g,b);
		break;
	case 1:
		bv = *(lpBits + LineBytes * (h - 1 - i) + j/8) & (1 << (7 - j % 8));
		if(0 == bv)
			strcpy(str,"������");
		else
			strcpy(str,"ǰ����");
		break;
	}

}

DWORD H[256];
void HGray()
{
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i,j;
	BYTE* pixel;
	for(i = 0; i < 256; i++)
		H[i] = 0;

	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			pixel = lpBits + LineBytes * (h - 1 - i) + j;
			H[*pixel] ++;
		}
	}
}
DWORD HR[256], HG[256], HB[256];
void H256(){
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

    int i,j;
    for (i = 0; i < 256; i++) HR[i] = HG[i] = HB[i] = 0;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            BYTE* pixel = lpBits + LineBytes * (h - 1 - i) + j;
			RGBQUAD color = lpBitsInfo->bmiColors[* pixel];
            HR[color.rgbRed]++;
            HG[color.rgbBlue]++;
            HB[color.rgbGreen]++;
        }
    }
}

// 24λλͼ��ֱ��ͼ���㺯��
void H24() {
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];
	BYTE *R,*G,*B;

	int i,j;
    for (i = 0; i < 256; i++) HR[i] = HG[i] = HB[i] = 0;
    for (i = 0; i < h; i++) {
        BYTE* line = lpBits + i * LineBytes;
        for (j = 0; j < w; j++) {
			B = lpBits + LineBytes * (h - 1 - i) + j * 3;
			G = B + 1;
			R = G + 1;
            HR[*R]++;
            HG[*G]++;
            HB[*B]++;
        }
    }
}

// 4λλͼ��ֱ��ͼ���㺯��
void H16() {
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

    int i, j, byteIndex, bitIndex;

    for (i = 0; i < 16; i++) HR[i] = HG[i] = HB[i] = 0;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            byteIndex = LineBytes * (h - 1 - i) + j / 2;
            bitIndex = (j % 2) ? 0 : 4;

            BYTE pixelValue = (lpBits[byteIndex] >> bitIndex) & 0x0F;  // ��ȡ��ǰ���ص� 4 λֵ

            // ��ȡ��ɫ���е���ɫ
            RGBQUAD color = lpBitsInfo->bmiColors[pixelValue];
            HR[color.rgbRed]++;
            HG[color.rgbGreen]++;
            HB[color.rgbBlue]++;
        }
    }
}


void Histogram(){
	if(IsGray()){
		HGray();
	}
	else{
		switch(bi.biBitCount){
			case 8:
				H256();
				break;
			case 24:
				H24();
				break;
			case 4:
				H16();
				break;

		}
	}
}
void LineTrans(float a,float b){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i,j;
	BYTE* pixel;
	float temp;
    for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			pixel = lpBits + LineBytes * (h - 1 - i) + j;
			temp = a * (*pixel) + b;
			if(temp > 255)
				*pixel = 255;
			else if(temp < 0)
				*pixel = 0;
			else
				*pixel = (BYTE)(temp + 0.5);
		}
	}
}


void Equalize(){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i,j;
	BYTE* pixel;
	DWORD temp;

	BYTE Map[256];
	Histogram();

	for(i = 0;i < 256; i++){
		temp = 0;
		for(j = 0;j <= i; j++){
			temp += H[j];
		}
		Map[i] = 255 * temp / (h * w);
	}

	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			pixel = lpBits + LineBytes * (h - 1 - i) + j;
			*pixel = Map[*pixel];
		}
	}
}

void E256(){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i,j;
	DWORD tempr,tempg,tempb;

	RGBQUAD* palette = lpBitsInfo->bmiColors;
	BYTE MapR[256],MapG[256],MapB[256];
	Histogram();
    for(i = 0;i < 256; i++){
		tempr = 0;
		tempg = 0;
		tempb = 0;
		for(j = 0;j <= i; j++){
			tempr += HR[j];
			tempg += HG[j];
			tempb += HB[j];
		}
		MapR[i] = 255.0 * tempr / (h * w);
		MapG[i] = 255.0 * tempg / (h * w);
		MapB[i] = 255.0 * tempb / (h * w);
	}
    
	RGBQUAD newPalette[256];  
    for(i = 0; i < 256; i++){  

        newPalette[i].rgbRed = MapR[palette[i].rgbRed];  
        newPalette[i].rgbGreen = MapG[palette[i].rgbGreen];  
        newPalette[i].rgbBlue = MapB[palette[i].rgbBlue];  
    } 
    
    // ���µ�ɫ��  
    for(i = 0; i < 256; i++){  
        palette[i] = newPalette[i];  
    }  
}


void E24(){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i,j;

	DWORD tempr,tempg,tempb;
	BYTE *R,*G,*B;

	BYTE MapR[256],MapG[256],MapB[256];
	Histogram();
    for(i = 0;i < 256; i++){
		tempr = 0;
		tempg = 0;
		tempb = 0;
		for(j = 0;j <= i; j++){
			tempr += HR[j];
			tempg += HG[j];
			tempb += HB[j];
		}
		MapR[i] = 255.0 * tempr / (h * w);
		MapG[i] = 255.0 * tempg / (h * w);
		MapB[i] = 255.0 * tempb / (h * w);
	}
     
    for (i = 0; i < h; i++) {
        BYTE* line = lpBits + i * LineBytes;
        for (j = 0; j < w; j++) {
			B = lpBits + LineBytes * (h - 1 - i) + j * 3;
			G = B + 1;
			R = G + 1;
		    *B = MapB[*B];
			*G = MapG[*G];
            *R = MapR[*R];
        }
    }

}

void E16(){
	int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[16];


	int i,j;
	DWORD tempr,tempg,tempb;

	RGBQUAD* palette = lpBitsInfo->bmiColors;
	BYTE MapR[256],MapG[256],MapB[256];
	Histogram();
    for(i = 0;i < 256; i++){
		tempr = 0;
		tempg = 0;
		tempb = 0;
		for(j = 0;j <= i; j++){
			tempr += HR[j];
			tempg += HG[j];
			tempb += HB[j];
		}
		MapR[i] = 255.0 * tempr / (h * w);
		MapG[i] = 255.0 * tempg / (h * w);
		MapB[i] = 255.0 * tempb / (h * w);
	}
    
	RGBQUAD newPalette[16];  
    for(i = 0; i < 16; i++){  
        newPalette[i].rgbRed = MapR[palette[i].rgbRed];  
        newPalette[i].rgbGreen = MapG[palette[i].rgbGreen];  
        newPalette[i].rgbBlue = MapB[palette[i].rgbBlue];  
    }  
    
    // ���µ�ɫ��  
    for(i = 0; i < 16; i++){  
        palette[i] = newPalette[i];  
    }  

}


void Eql(){
	if(IsGray()){
		Equalize();
	}
	else{
		switch(bi.biBitCount){
		case 8:
			E256();
			break;
		case 24:
			E24();
			break;
		case 4:
			E16();
			break;
		}
	}
}

//һά����Ҷ�任
void FT(complex<double>* TD,complex<double>* FD,int m)
{
	int x,u;
	double angle;
	for (u=0;u<m;u++){
		FD[u]=0;
		for(x=0;x<m;x++){
			angle=-2*PI*u*x/m;
			FD[u]+=TD[x]*complex<double>(cos(angle),sin(angle));
		}
		FD[u]/=m;
	}

}

//���任
void IFT(complex<double>* FD,complex<double>* TD,int m)
{
	int x,u;
	double angle;
	for (x=0;x<m;x++){
		TD[x]=0;
		for(u=0;u<m;u++){
			angle=2*PI*u*x/m;
			TD[x]+=FD[u]*complex<double>(cos(angle),sin(angle));
		}
		
	}

}

void Fourier(){
	int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	complex<double >* TD = new complex<double>[w*h];
	complex<double >* FD = new complex<double>[w*h];

	int i,j;
	BYTE* pixel;
	for(i=0;i<h;i++){
		for(j=0;j<w;j++){
			pixel = lpBits+LineBytes*(h-1-i)+j;
			TD[i*w+j] = complex<double>(*pixel * pow(-1,i+j) , 0);
		}
	}
	for(i = 0;i<h;i++){
		FT(&TD[w*i],&FD[w*i],w);
	}
	for(i=0;i<h;i++){
		for(j=0;j<w;j++){
			TD[j*h+i] = FD[i*w+j];
		}
	}
	for(i=0;i<w;i++){
		FT(&TD[h*i],&FD[h*i],h);
	}
	
	DWORD Size = 40 + 1024+LineBytes*h;

	lpDIB_FT=(BITMAPINFO*)malloc(Size);
	memcpy(lpDIB_FT,lpBitsInfo,Size);

	lpBits = (BYTE*)&lpDIB_FT->bmiColors[256];
	double temp;
	for(i=0;i<h;i++){
		for(j=0;j<w;j++){
			pixel =lpBits+LineBytes*(h-1-i)+j;
			temp=  sqrt(FD[j*h+i].real()*FD[j*h+i].real()+FD[j*h+i].imag()*FD[j*h+i].imag())*1000;
			if(temp>255)
				temp = 255;
			*pixel  = (BYTE)(temp);
		}
	}
	delete TD;
	//delete FD;
	gFD = FD;
}

void IFourier(){
	int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];
	complex<double >* TD = new complex<double>[w*h];
	int i,j;
	
	
	
	for(i = 0;i<w;i++){
		IFT(&gFD[h*i],&TD[h*i],h);
	}
	for(i=0;i<h;i++){
		for(j=0;j<w;j++){
			gFD[j*h+i] = TD[i*w+j];
		}
	}
	for(i=0;i<h;i++){
		IFT(&gFD[w*i],&TD[w*i],w);
	}




	DWORD Size = 40 + 1024+LineBytes*h;

	lpDIB_IFT=(BITMAPINFO*)malloc(Size);
	memcpy(lpDIB_IFT,lpBitsInfo,Size);

	lpBits = (BYTE*)&lpDIB_IFT->bmiColors[256];

	BYTE *pixel;
	
	for(i=0;i<h;i++){
		for(j=0;j<w;j++){
			pixel =lpBits+LineBytes*(h-1-i)+j;
			
			*pixel  = (BYTE)(TD[w*i+j].real()/pow(-1,i+j));
		}
	}
	delete TD;
	delete gFD;
	gFD=NULL;
}
BOOL is_gFD_OK(){
	return (gFD != NULL);
}


//���ٸ���Ҷ�任
void FFT(complex<double> * TD, complex<double> * FD, int r)
{
	// ���㸶��Ҷ�任����
	LONG count = 1 << r;
	// �����Ȩϵ��
	int i;
	double angle;
	complex<double>* W = new complex<double>[count / 2];
	for(i = 0; i < count / 2; i++)
	{
		angle = -i * PI * 2 / count;
		W[i] = complex<double> (cos(angle), sin(angle));
	}
	// ��ʱ���д��X1
	complex<double>* X1 = new complex<double>[count];
	memcpy(X1, TD, sizeof(complex<double>) * count);
	
	// ���õ����㷨���п��ٸ���Ҷ�任�����ΪƵ��ֵX2
	complex<double>* X2 = new complex<double>[count]; 

	int k,j,p,size;
	complex<double>* temp;
	for (k = 0; k < r; k++)
	{
		for (j = 0; j < 1 << k; j++)
		{
			size = 1 << (r-k);
			for (i = 0; i < size/2; i++)
			{
				p = j * size;
				X2[i + p] = X1[i + p] + X1[i + p + size/2];
				X2[i + p + size/2] = (X1[i + p] - X1[i + p + size/2]) * W[i * (1<<k)];
			}
		}
		temp  = X1;
		X1 = X2;
		X2 = temp;
	}
	
	// ����������λ�������У�
	for (j = 0; j < count; j++)
	{
		p = 0;
		for (i = 0; i < r; i++)
		{
			if (j & (1<<i))
			{
				p += 1<<(r-i-1);
			}
		}
		FD[j]=X1[p];
		FD[j] /= count;
	}
	
	// �ͷ��ڴ�
	delete W;
	delete X1;
	delete X2;
}

void FFourier()
{
	//ͼ��Ŀ�Ⱥ͸߶�
	int width = lpBitsInfo->bmiHeader.biWidth;
	int height = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (width * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	//ָ��ͼ������ָ��
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	// FFT��ȣ�����Ϊ2�������η���
	int FFT_w = 1;
	// FFT��ȵ�����������������
	int wp = 0;
	while(FFT_w * 2 <= width)
	{
		FFT_w *= 2;
		wp ++;
	}

	// FFT�߶ȣ�����Ϊ2�������η���
	int FFT_h = 1;
	// FFT�߶ȵ�����������������
	int hp = 0;
	while(FFT_h * 2 <= height)
	{
		FFT_h *= 2;
		hp ++;
	}

	// �����ڴ�
	complex<double>* TD = new complex<double>[FFT_w * FFT_h];
	complex<double>* FD = new complex<double>[FFT_w * FFT_h];
	
	int i, j;
	BYTE* pixel;
	
	for(i = 0; i < FFT_h; i++)  // ��
	{
		for(j = 0; j < FFT_w; j++)  // ��
		{
			// ָ��DIB��i�У���j�����ص�ָ��
			pixel = lpBits + LineBytes * (height - 1 - i) + j;

			// ��ʱ��ֵ
			TD[j + FFT_w * i] = complex<double>(*pixel* pow(-1,i+j), 0);
		}
	}
	
	for(i = 0; i < FFT_h; i++)
	{
		// ��y������п��ٸ���Ҷ�任
		FFT(&TD[FFT_w * i], &FD[FFT_w * i], wp);
	}
	
	// �����м�任���
	for(i = 0; i < FFT_h; i++)
	{
		for(j = 0; j < FFT_w; j++)
		{
			TD[i + FFT_h * j] = FD[j + FFT_w * i];
		}
	}
	
	for(i = 0; i < FFT_w; i++)
	{
		// ��x������п��ٸ���Ҷ�任
		FFT(&TD[i * FFT_h], &FD[i * FFT_h], hp);
	}

	//����Ƶ��ͼ��
	//ΪƵ��ͼ������ڴ�
	LONG size = 40 + 1024 + LineBytes * height;
	lpDIB_FT = (LPBITMAPINFO) malloc(size);
	if (NULL == lpDIB_FT)
		return;
	memcpy(lpDIB_FT, lpBitsInfo, size);

	//ָ��Ƶ��ͼ������ָ��
	lpBits = (BYTE*)&lpDIB_FT->bmiColors[lpDIB_FT->bmiHeader.biClrUsed];

	double temp;
	for(i = 0; i < FFT_h; i++) // ��
	{
		for(j = 0; j < FFT_w; j++) // ��
		{
			// ����Ƶ�׷���
			temp = sqrt(FD[j * FFT_h + i].real() * FD[j * FFT_h + i].real() + 
				        FD[j * FFT_h + i].imag() * FD[j * FFT_h + i].imag()) *2000;
			
			// �ж��Ƿ񳬹�255
			if (temp > 255)
			{
				// ���ڳ����ģ�ֱ������Ϊ255
				temp = 255;
			}
			
			pixel = lpBits + LineBytes * (height - 1 - i) + j;

			// ����Դͼ��
			*pixel = (BYTE)(temp);
		}
	}

	delete TD;
	gFD = FD;
}


//���ٸ���Ҷ���任
//IFFT���任
void IFFT(complex<double> * FD, complex<double> * TD, int r)
{
	// ����Ҷ�任����
	LONG	count;
	// ���㸶��Ҷ�任����
	count = 1 << r;

	// ������������洢��
	complex<double> * X = new complex<double>[count];
	// ��Ƶ���д��X
	memcpy(X, FD, sizeof(complex<double>) * count);
	
	// ����
	for(int i = 0; i < count; i++)
		X[i] = complex<double> (X[i].real(), -X[i].imag());
	
	// ���ÿ��ٸ���Ҷ�任
	FFT(X, TD, r);
	
	// ��ʱ���Ĺ���
	for(i = 0; i < count; i++)
		TD[i] = complex<double> (TD[i].real() * count, -TD[i].imag() * count);
	
	// �ͷ��ڴ�
	delete X;
}

void IFFourier()
{
	if (lpDIB_IFT)
	{
		free(lpDIB_IFT);
		lpDIB_IFT = NULL;
	}

	//ͼ��Ŀ�Ⱥ͸߶�
	int width = lpBitsInfo->bmiHeader.biWidth;
	int height = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (width * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;

	// FFT��ȣ�����Ϊ2�������η���
	int FFT_w = 1;
	// FFT��ȵ�����������������
	int wp = 0;
	while(FFT_w * 2 <= width)
	{
		FFT_w *= 2;
		wp ++;
	}

	// FFT�߶ȣ�����Ϊ2�������η���
	int FFT_h = 1;
	// FFT�߶ȵ�����������������
	int hp = 0;
	while(FFT_h * 2 <= height)
	{
		FFT_h *= 2;
		hp ++;
	}

	// �����ڴ�
	complex<double>* TD = new complex<double>[FFT_w * FFT_h];
	
	int i,j;
	//ע��FD�Ǿ���ת�õ�
	//���з�����w�Σ�h�����һά��ɢ����Ҷ�任
	//���������ת��ǰ����
	for (i = 0; i < FFT_w; i ++)
		IFFT(&gFD[i * FFT_h], &TD[i * FFT_h], hp); //���з���

	//ת��
	for (i = 0; i < FFT_h; i ++)
		for (j = 0; j < FFT_w; j ++)
			gFD[FFT_h * j + i] = TD[FFT_w * i + j];
	//�������任��ת��,���������ת�ã�ͼ�����ݻ�ԭ��ת��ǰ��˳����

	//���з�����h�Σ�w�����һά��ɢ����Ҷ�任
	for (i = 0; i < FFT_h; i++)
		IFFT(&gFD[FFT_w * i], &TD[FFT_w * i], wp);

	//Ϊ���任ͼ������ڴ�
	LONG size = 40 + 1024 + LineBytes * height;

	lpDIB_IFT = (LPBITMAPINFO) malloc(size);
	if (NULL == lpDIB_IFT)
		return;
	memcpy(lpDIB_IFT, lpBitsInfo, size);

	//ָ�򷴱任ͼ������ָ��
	BYTE* lpBits = (BYTE*)&lpDIB_IFT->bmiColors[lpDIB_IFT->bmiHeader.biClrUsed];
	BYTE* pixel;
	double temp;
	for(i = 0; i < FFT_h; i++) // ��
	{
		for(j = 0; j < FFT_w; j++) // ��
		{
			pixel = lpBits + LineBytes * (height - 1 - i) + j;
			temp= (TD[FFT_h * i + j].real() / pow(-1, i+j));
			if (temp < 0)
				temp = 0;
			else if (temp >255)
				temp = 255;
			*pixel = (BYTE)temp;
		}
	}

	// ɾ����ʱ����
	delete TD;
	delete gFD;
	gFD = NULL;
}


//�˲�

void Template(int *Array, float coef){
	int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];
	
	LONG Size = 40 + 1024+LineBytes*h;

	BITMAPINFO *new_lpBitsInfo=(BITMAPINFO*)malloc(Size);
	if(new_lpBitsInfo == NULL)
		return;
	memcpy(new_lpBitsInfo,lpBitsInfo,Size);

	BYTE* new_lpBits = (BYTE*)&new_lpBitsInfo->bmiColors[new_lpBitsInfo->bmiHeader.biClrUsed];



	int i,j,m,n;
	BYTE * pixel , *new_pixel;
	float result;

	for( i=1;i< h-1;i++){
		for(j=1;j<w-1;j++){
			new_pixel = new_lpBits+LineBytes*(h-1-i)+j;
			result =0;
			for(m=0;m<3;m++){
				for(n=0;n<3;n++){
					pixel =lpBits +LineBytes*(h-i-m)+j-1+n;
					result += (*pixel) *Array[m*3+n]; 
				}
			}
			result *=coef;
			if(result<0)
				*new_pixel = 0;
			else if(result>255)
				*new_pixel = 255;
			else
				*new_pixel = (BYTE)(result+0.5);

		}
			
	}
	free(lpBitsInfo);
	lpBitsInfo = new_lpBitsInfo;


}
void AvgFilter(){
	int Array[9];
	Array[0] =1;  Array[1]=2;  Array[2]=1;
	Array[3] =2;  Array[4]=4;  Array[5]=2;
	Array[6] =1;  Array[7]=2;  Array[8]=1;
	Template(Array,(float) 1/16);

}

BYTE GetMedianNum(BYTE* Array){
	int i,j;
	BYTE temp;

	for(j = 0;j < 8;j++){
		for(i = 0;i < 8 - j;i++){
			if(Array[i] > Array[i + 1]){
				temp = Array[i];
				Array[i] = Array[i + 1];
				Array[i + 1] = temp;
			}
		}
	}
	return Array[4];

}

void MedFilter(){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	LONG size = 40 + 1024 + LineBytes * h;
	BITMAPINFO* new_lpBitsInfo = (BITMAPINFO*) malloc(size);
	if(NULL == new_lpBitsInfo)
		return;

	memcpy(new_lpBitsInfo, lpBitsInfo, size);


	BYTE* new_lpBits = (BYTE*)&new_lpBitsInfo->bmiColors[new_lpBitsInfo->bmiHeader.biClrUsed];
	int i,j,m,n;
	BYTE *pixel,*new_pixel;
	BYTE Array[9];

	for(i = 1;i < h-1; i++){
		for(j = 1;j < w - 1;j++){
			for(m = 0;m < 3; m++){
				for(n = 0;n < 3; n++){
					pixel = lpBits + LineBytes * (h - m - i) + j - 1 + n;
			        Array[m * 3 + n] = *pixel;
				}
			}
            new_pixel = new_lpBits + LineBytes * (h - 1 - i) + j;
			*new_pixel = GetMedianNum(Array);
		}
	}
	free(lpBitsInfo);
	lpBitsInfo = new_lpBitsInfo;
}
void GradSharp(){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	LONG size = 40 + 1024 + LineBytes * h;
	BITMAPINFO* new_lpBitsInfo = (BITMAPINFO*) malloc(size);
	if(NULL == new_lpBitsInfo)
		return;

	memcpy(new_lpBitsInfo, lpBitsInfo, size);


	BYTE* new_lpBits = (BYTE*)&new_lpBitsInfo->bmiColors[new_lpBitsInfo->bmiHeader.biClrUsed];

	int i,j;
	BYTE *pixel,*pixel_right,*pixel_down,*new_pixel;


	for(i = 1;i < h-1; i++){
		for(j = 1;j < w - 1;j++){
		    pixel = lpBits + LineBytes * (h - 1 - i) + j;
			pixel_right = lpBits + LineBytes * (h - 1 - i) + j + 1;
			pixel_down = lpBits + LineBytes * (h - 2 - i) + j;

			new_pixel = new_lpBits +  LineBytes * (h - 1 - i) + j;
			*new_pixel = abs(*pixel-*pixel_right) + abs(*pixel-*pixel_down);
			
		}
	}
	free(lpBitsInfo);
	lpBitsInfo = new_lpBitsInfo;
}
void RaplasSharp(){
	int Array[9];
	Array[0] =0;  Array[1]=-1;  Array[2]=0;
	Array[3] =-1;  Array[4]=5;  Array[5]=-1;
	Array[6] =0;  Array[7]=-1;  Array[8]=0;
	Template(Array,(float) 1);

}

//������˹
void FFT_Filter(int D){
	int w = lpDIB_FT->bmiHeader.biWidth;
	int h = lpDIB_FT->bmiHeader.biHeight;
	int LineBytes = (w * lpDIB_FT->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpDIB_FT->bmiColors[lpDIB_FT->bmiHeader.biClrUsed];
	
	complex<double>* Backup_FD = new complex<double>[w * h];
	int i,j;
	for(i = 0;i < w * h; i++){
		Backup_FD[i] = gFD[i];
	}

	
	double dis;
	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			dis = sqrt((i - h / 2) * (i - h / 2) + (j - w / 2) * (j - w / 2));
			if(D > 0)
				gFD[i * w + j] *= 1 / (1 + pow(dis / D,4));


			else
				gFD[i * w + j] *= 1 / (1 + pow(-D/dis ,4));

		}
	}

	double temp;
	BYTE* pixel;
	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			pixel = lpBits + LineBytes * (h - 1 - i) + j;
		    temp = sqrt(gFD[j * h + i].real() * gFD[j * h + i].real() +
				        gFD[j * h + i].imag() * gFD[j * h + i].imag()) * 2000;

			if (temp > 255)
				temp = 255;
			*pixel = (BYTE)(temp);

		}
	}

	IFFourier();

	delete gFD;
	gFD = Backup_FD;

}

//�����ͨ�˲���
void ILP_Filter(int D){
	int w = lpDIB_FT->bmiHeader.biWidth;
	int h = lpDIB_FT->bmiHeader.biHeight;
	int LineBytes = (w * lpDIB_FT->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpDIB_FT->bmiColors[lpDIB_FT->bmiHeader.biClrUsed];
	
	complex<double>* Backup_FD = new complex<double>[w * h];
	int i,j;
	for(i = 0;i < w * h; i++){
		Backup_FD[i] = gFD[i];
	}

	
	double dis;
	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			dis = sqrt((i - h / 2) * (i - h / 2) + (j - w / 2) * (j - w / 2));

			// ������Ƶ
			if (D > 0) { 
				if(dis <= D) 
					gFD[i * w + j] *= 1; 
	
				else 
					gFD[i * w + j] = 0;
			}
			// ������Ƶ
			else {
				if(dis <= -1 * D) 
					gFD[i * w + j] = 0;
	
				else 
					gFD[i * w + j] *= 1;
			}
		}
	}

	double temp;
	BYTE* pixel;
	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			pixel = lpBits + LineBytes * (h - 1 - i) + j;
		    temp = sqrt(gFD[j * h + i].real() * gFD[j * h + i].real() +
				        gFD[j * h + i].imag() * gFD[j * h + i].imag()) * 2000;

			if (temp > 255)
				temp = 255;
			*pixel = (BYTE)(temp);

		}
	}

	IFFourier();

	delete gFD;
	gFD = Backup_FD;

}


// ��˹��ͨ�˲���
void GLP_Filter(int D){
	int w = lpDIB_FT->bmiHeader.biWidth;
	int h = lpDIB_FT->bmiHeader.biHeight;
	int LineBytes = (w * lpDIB_FT->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpDIB_FT->bmiColors[lpDIB_FT->bmiHeader.biClrUsed];
	
	complex<double>* Backup_FD = new complex<double>[w * h];
	int i,j;
	for(i = 0;i < w * h; i++){
		Backup_FD[i] = gFD[i];
	}

	
	double dis;
	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			dis = sqrt((i - h / 2) * (i - h / 2) + (j - w / 2) * (j - w / 2));
			if(D > 0) {
				gFD[i * w + j] *= exp(-1 * pow(dis, 2) / (2 * pow(D, 2)));
			}	
			else {
				gFD[i * w + j] *= 1 - exp(-1 * pow(dis, 2) / (2 * pow(D, 2)));
			}
		}
	}

	double temp;
	BYTE* pixel;
	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			pixel = lpBits + LineBytes * (h - 1 - i) + j;
		    temp = sqrt(gFD[j * h + i].real() * gFD[j * h + i].real() +
				        gFD[j * h + i].imag() * gFD[j * h + i].imag()) * 2000;

			if (temp > 255)
				temp = 255;
			*pixel = (BYTE)(temp);

		}
	}

	IFFourier();

	delete gFD;
	gFD = Backup_FD;

}
