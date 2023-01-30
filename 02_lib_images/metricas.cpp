
/* metricas.cpp */

/* .........................................................
*/
#include <metricas.h>
#include <math.h>
#include <stdio.h>
#include <iostream>

/* .........................................................
*/
SI_EN_GPU float distanciaEuclidea (PixelRGB* p1, PixelRGB* p2) {

  int dr = p1->r - p2->r;
  int dg = p1->g - p2->g;
  int db = p1->b - p2->b;

  // float dist = sqrt( (dr*dr) + (dg*dg) + (db*db) );
   float dist =  (dr*dr) + (dg*dg) + (db*db) ;
  
   return  dist;
} 

/* .........................................................
*/
#define mayor(a,b) ((a)>=(b) ? (a) : (b))
#define menor(a,b) ((a)<=(b) ? (a) : (b))

SI_EN_GPU float distanciaFuzzy1 (PixelRGB* p1, PixelRGB* p2) {

  int a = menor(p1->r,p2->r) / mayor(p1->r,p2->r);
  int b = menor(p1->g,p2->g) / mayor(p1->g,p2->g);
  int c = menor(p1->b,p2->b) / mayor(p1->b,p2->b);

  float dist = a * b * c;
  
  return  dist;
} 

/* .........................................................
*/
// SI_EN_GPU float distancia(PixelRGB* p1, PixelRGB* p2) {
SI_EN_GPU float distanciaDiferencias (PixelRGB* p1, PixelRGB* p2) {

  int dr = p1->r - p2->r;
  int dg = p1->g - p2->g;
  int db = p1->b - p2->b;

  dr = (dr<0 ? -dr : dr);
  dg = (dg<0 ? -dg : dg);
  db = (db<0 ? -db : db);

  float dist = (dr+dg+db) / 765.0;
  
  return  dist;
} 

SI_EN_GPU float distanciaDiferenciasPrueba (PixelRGB* p1, PixelRGB* p2) {

  int dr = p1->r & (~(p2->r));
  int dg = p1->g & (~(p2->g));
  int db = p1->b & (~(p2->b));

  float dist = (dr+dg+db);
  
  return  dist;
} 

/* .........................................................
*/
SI_EN_GPU void media(PixelRGB* pix[], unsigned int n, PixelRGB * res) {

  if (n==0) {
	return; 
  }

  unsigned int sumR = 0;
  unsigned int sumG = 0;
  unsigned int sumB = 0;
  unsigned int i;
  
  /* pruebas
  printf ("--->%x\n",  pix );
  printf ("--->%d\n",  pix[0].r );
  printf ("---> ja esta \n");
  */

  for (i=0; i<=n-1; i++) {
	sumR += pix[i]->r;
	sumG += pix[i]->g;
	sumB += pix[i]->b;
  }

  res->r = sumR / n;
  res->g = sumG / n;
  res->b = sumB / n;

} /* () */

/* .........................................................
*/
SI_EN_GPU void mediaC(PixelRGB pix[], unsigned int n, PixelRGB * res) {

  if (n==0) {
	return; 
  }

  unsigned int sumR = 0;
  unsigned int sumG = 0;
  unsigned int sumB = 0;
  unsigned int i;
  
  for (i=0; i<=n-1; i++) {
	sumR += pix[i].r;
	sumG += pix[i].g;
	sumB += pix[i].b;
  }

  res->r = sumR / n;
  res->g = sumG / n;
  res->b = sumB / n;

} /* () */

