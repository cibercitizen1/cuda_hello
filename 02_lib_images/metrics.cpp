
// ---------------------------------------------------------
/* metrics.cpp */
// ---------------------------------------------------------

/* .........................................................
*/
#include <metrics.h>
#include <math.h>
#include <stdio.h>
#include <iostream>

/* .........................................................
*/
IF_GPU float euclidean_distance (PixelRGBA* p1, PixelRGBA* p2) {

  int dr = p1->r - p2->r;
  int dg = p1->g - p2->g;
  int db = p1->b - p2->b;

  // float dist = sqrt( (dr*dr) + (dg*dg) + (db*db) );
   float dist =  (dr*dr) + (dg*dg) + (db*db) ;
  
   return  dist;
} 

/* .........................................................
*/
#define greater_equal_than(a,b) ((a)>=(b) ? (a) : (b))
#define lesser_equal_than(a,b) ((a)<=(b) ? (a) : (b))

IF_GPU float fuzzy_distance_1 (PixelRGBA* p1, PixelRGBA* p2) {

  int a = lesser_equal_than(p1->r,p2->r) / greater_equal_than(p1->r,p2->r);
  int b = lesser_equal_than(p1->g,p2->g) / greater_equal_than(p1->g,p2->g);
  int c = lesser_equal_than(p1->b,p2->b) / greater_equal_than(p1->b,p2->b);

  float dist = a * b * c;
  
  return  dist;
} 

/* .........................................................
 */
IF_GPU float differences_distance (PixelRGBA* p1, PixelRGBA* p2) {

  int dr = p1->r - p2->r;
  int dg = p1->g - p2->g;
  int db = p1->b - p2->b;

  dr = (dr<0 ? -dr : dr);
  dg = (dg<0 ? -dg : dg);
  db = (db<0 ? -db : db);

  float dist = (dr+dg+db) / 765.0;
  
  return  dist;
} 

/* .........................................................
 */
IF_GPU void mean_of_pixels_wtf(PixelRGBA* pix[], unsigned int n, PixelRGBA * res) {

  if (n==0) {
	return; 
  }

  unsigned int sumR = 0;
  unsigned int sumG = 0;
  unsigned int sumB = 0;
  unsigned int i;
  
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
IF_GPU void mean_of_pixels(PixelRGBA pix[], unsigned int n, PixelRGBA * res) {

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

// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
// ---------------------------------------------------------
