#include "stdafx.h"
#include "windows.h"
BITMAPINFO* lpBitsInfo = NULL;
BITMAPINFOHEADER bi;
BOOL LoadBmpFile(char* BmpFileName){
	FILE* fp;
	if(NULL == (fp = fopen(BmpFileName,"rb")))
		return FALSE;

	BITMAPFILEHEADER bf;


	fread(&bf, 14, 1, fp);
	fread(&bi, 40, 1, fp);

	DWORD NumColors;
	if (bi.biClrUsed != 0)
		NumColors = bi.biClrUsed;
	else{
		switch(bi.biBitCount){
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

	if(NULL == (lpBitsInfo = (BITMAPINFO*)malloc(Size)))
		return FALSE;

	fseek(fp, 14, SEEK_SET);
	fread((char*)lpBitsInfo, Size,1,fp);

	lpBitsInfo->bmiHeader.biClrUsed = NumColors;
    return TRUE;
}


void Gray2()
{
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w + 7) / 8;  // ÿ���ֽ������� 8 ������ 1 �ֽڼ���

    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed]; // ָ��ͼ��ʵ�����ݵ�ָ��

    // Ϊ�Ҷ�ͼ������ڴ�
    int LineBytes_gray = (w + 3) / 4 * 4;  // ÿ������ռ 1 �ֽڣ�ȷ��ÿ���ֽ����� 4 �ı���
    BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD) + LineBytes_gray * h);

    // ����ͷ����Ϣ���޸�Ϊ�Ҷ�ͼ��
    memcpy(lpBitsInfo_gray, lpBitsInfo, sizeof(BITMAPINFOHEADER));
    lpBitsInfo_gray->bmiHeader.biBitCount = 8;   // ����Ϊ 8 λ�Ҷ�ͼ
    lpBitsInfo_gray->bmiHeader.biClrUsed = 256;  // ʹ�� 256 ɫ��ɫ��

    // ��ʼ���Ҷȵ�ɫ�壨�ڰף��Ҷ�ֵ 1 �� 255��
    lpBitsInfo_gray->bmiColors[0].rgbRed = 1;
    lpBitsInfo_gray->bmiColors[0].rgbGreen = 1;
    lpBitsInfo_gray->bmiColors[0].rgbBlue = 1;
    lpBitsInfo_gray->bmiColors[0].rgbReserved = 0;

    lpBitsInfo_gray->bmiColors[255].rgbRed = 255;
    lpBitsInfo_gray->bmiColors[255].rgbGreen = 255;
    lpBitsInfo_gray->bmiColors[255].rgbBlue = 255;
    lpBitsInfo_gray->bmiColors[255].rgbReserved = 0;

    BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];

	int i, j;

    // ����ÿ�����أ���ȡλ������Ҷ�ֵ
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            // ���㵱ǰ�ֽ���ͼ�������е�λ��
            int byteIndex = LineBytes * (h - 1 - i) + j / 8;
            int bitIndex = 7 - (j % 8);  // �������ҵ�˳��ȡÿ�����ص�λ

            // ��ȡ��ǰ����ֵ��0 �� 1��
            BYTE pixelValue = (lpBits[byteIndex] >> bitIndex) & 1;

            // �� 0 ӳ��Ϊ�Ҷ�ֵ 1��1 ӳ��Ϊ�Ҷ�ֵ 255
            BYTE grayValue = (pixelValue == 0) ? 0 : 255;

            // ���Ҷ�ֵд��Ҷ�ͼ������
            lpBits_gray[LineBytes_gray * (h - 1 - i) + j] = grayValue;
        }
    }

    free(lpBitsInfo);  // �ͷ�ԭʼͼ���ڴ�
    lpBitsInfo = lpBitsInfo_gray;  // ����Ϊ�Ҷ�ͼ��
}



void Gray16()
{
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;  // ÿ���ֽ���
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[16]; // �̶�ָ��16ɫ��ɫ��֮�������

    // Ϊ�Ҷ�ͼ������ڴ�
    int LineBytes_gray = (w + 3) & ~3;  // ÿ������ռ1�ֽڣ�ÿ�ж��뵽4�ı���
    BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD) + LineBytes_gray * h);

    // ����ͷ����Ϣ���޸�Ϊ�Ҷ�ͼ��
    memcpy(lpBitsInfo_gray, lpBitsInfo, sizeof(BITMAPINFOHEADER));
    lpBitsInfo_gray->bmiHeader.biBitCount = 8;   // ����Ϊ8λ�Ҷ�ͼ
    lpBitsInfo_gray->bmiHeader.biClrUsed = 256;  // ʹ��256ɫ��ɫ��

    int i,j;

    // ��ʼ���Ҷȵ�ɫ�壨ÿ���Ҷ�ֵ�� 0 �� 255��
    for (i = 0; i < 256; i++) {
        lpBitsInfo_gray->bmiColors[i].rgbRed = i;
        lpBitsInfo_gray->bmiColors[i].rgbGreen = i;
        lpBitsInfo_gray->bmiColors[i].rgbBlue = i;
        lpBitsInfo_gray->bmiColors[i].rgbReserved = 0;
    }

    BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];

    // ����ÿ������
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            int byteIndex = LineBytes * (h - 1 - i) + j / 2;
            int bitIndex = (j % 2) ? 0 : 4;

            BYTE pixelValue = (lpBits[byteIndex] >> bitIndex) & 0x0F;  // ��ȡ��ǰ���ص� 4 λֵ

            // ��ȡ��ɫ���е���ɫ
            RGBQUAD color = lpBitsInfo->bmiColors[pixelValue];

            // ��Ȩƽ������Ҷ�ֵ
            BYTE grayValue = (BYTE)((color.rgbRed + color.rgbGreen + color.rgbBlue)/3);

            // ���Ҷ�ֵд��Ҷ�ͼ������
            lpBits_gray[LineBytes_gray * (h - 1 - i) + j] = grayValue;
        }
    }

    free(lpBitsInfo);  // �ͷ�ԭʼͼ���ڴ�
    lpBitsInfo = lpBitsInfo_gray;  // ����Ϊ�Ҷ�ͼ��
}


void Gray256()
{
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;  // ÿ���ֽ���
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed]; // ָ��ͼ��ʵ�����ݵ�ָ��

    // Ϊ�Ҷ�ͼ������ڴ�
    int LineBytes_gray = (w + 3) / 4 * 4;  // 8λ�Ҷ�ͼ��ÿ������ռ1�ֽ�
    BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD) + LineBytes_gray * h);
    
    // ����ͷ����Ϣ���޸�Ϊ�Ҷ�ͼ��
    memcpy(lpBitsInfo_gray, lpBitsInfo, sizeof(BITMAPINFOHEADER));
    lpBitsInfo_gray->bmiHeader.biBitCount = 8;   // ����Ϊ8λ�Ҷ�ͼ
    lpBitsInfo_gray->bmiHeader.biClrUsed = 256;  // ʹ��256ɫ��ɫ��

    // ��ʼ���Ҷȵ�ɫ�壨ÿ���Ҷ�ֵ�� 0 �� 255��
	int i, j;
    for (i = 0; i < 256; i++) {
        lpBitsInfo_gray->bmiColors[i].rgbRed = i;
        lpBitsInfo_gray->bmiColors[i].rgbGreen = i;
        lpBitsInfo_gray->bmiColors[i].rgbBlue = i;
        lpBitsInfo_gray->bmiColors[i].rgbReserved = 0;
    }


    BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            // ȡ��ͼ���е�ÿ�����ص���ɫ����
            BYTE pixelIndex = lpBits[LineBytes * (h - 1 - i) + j];
            
            // ��ȡ��ɫ���е���ɫ��256ɫͼ���Ǵӵ�ɫ���ȡ��ɫ��
            RGBQUAD color = lpBitsInfo->bmiColors[pixelIndex];
            
            // ����Ҷ�ֵ������ʹ�ü򵥵�ƽ����
            BYTE avg = (color.rgbRed + color.rgbGreen + color.rgbBlue) / 3;
            
            // ���Ҷ�ֵ����Ҷ�ͼ������
            lpBits_gray[LineBytes_gray * (h - 1 - i) + j] = avg;
        }
    }

    free(lpBitsInfo);  // �ͷ�ԭʼͼ����ڴ�
    lpBitsInfo = lpBitsInfo_gray;  // ����Ϊ�Ҷ�ͼ��
}


void Gray24(){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int LineBytes_gray = (w * 8 + 31)/32 * 4;
	BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(40 + 1024 + LineBytes_gray * h);

	memcpy(lpBitsInfo_gray, lpBitsInfo, 40);
	lpBitsInfo_gray->bmiHeader.biBitCount = 8;
	lpBitsInfo_gray->bmiHeader.biClrUsed = 256;

	int i, j;
	for(i = 0;i < 256;i ++){
		lpBitsInfo_gray->bmiColors[i].rgbRed = i;
		lpBitsInfo_gray->bmiColors[i].rgbGreen = i;
		lpBitsInfo_gray->bmiColors[i].rgbBlue = i;
		lpBitsInfo_gray->bmiColors[i].rgbReserved = 0;
	}

	BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];
 
	BYTE *R,*G,*B,avg,*pixel;
    lpBitsInfo->bmiHeader.biBitCount;

	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			B = lpBits + LineBytes * (h - 1 - i) + j * 3;
			G = B + 1;
			R = G + 1;
			avg = (*R + *G + *B)/3;
			pixel = lpBits_gray + LineBytes_gray * (h - 1 - i) + j;
			*pixel = avg;
		}
	}
	free(lpBitsInfo);
	lpBitsInfo = lpBitsInfo_gray;
}

void Gray(){
    switch(bi.biBitCount)
	{
	case 8:
		Gray256();
		break;
	case 24:
		Gray24();
		break;
	case 4:
		Gray16();
		break;
	case 1:
		Gray2();
		break;
	}
}

BOOL IsGray()
{
	int r,g,b;
	if (8 == lpBitsInfo->bmiHeader.biBitCount)
	{
		r = lpBitsInfo->bmiColors[150].rgbRed;
		g = lpBitsInfo->bmiColors[150].rgbGreen;
		b = lpBitsInfo->bmiColors[150].rgbBlue;
           
		if(r == b && r == g)
			return TRUE;
	}
	return FALSE;

}
void pixel(int i, int j, char* str)
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
void Histogram()
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
            HG[color.rgbGreen]++;
            HB[color.rgbBlue]++;
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

void H16()
{
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j;

	for(i = 0; i < 256; i++) {
		HR[i] = 0;
		HG[i] = 0;
		HB[i] = 0;
	}

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			int byteIndex = LineBytes * (h - 1 - i) + j / 2;
			// 4λ��ʾһ�����أ�2 ������һ���ֽڣ�jΪż��ǰ��λ��jΪ��������λ������λ��������
            int bitIndex = (j % 2) ? 0 : 4;  

			// ��ȡ��ǰ���ص� 4 λֵ,һ��BYTE����ռһ���ֽ�
            BYTE pixelValue = (lpBits[byteIndex] >> bitIndex) & 0x0F;  

			// ��ȡ��ɫ���е���ɫ
            RGBQUAD color = lpBitsInfo->bmiColors[pixelValue];

            HR[color.rgbRed] += 1;
			HG[color.rgbGreen] += 1;
			HB[color.rgbBlue] +=1 ;
		}
	}
}


void His(){
	if(IsGray()){
		Histogram();
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
	His();

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
	His();
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
	His();
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
	His();
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