#include <stdio.h>
#include <complex.h>

int main() {
  float a = 1;
  float b = 4;
  complex compNum = a + b * I;
  printf("%f%+fi\n", compNum);
  return 0;
}
