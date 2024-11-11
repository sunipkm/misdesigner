#ifndef _INTEGRATE_H_
#define _INTEGRATE_H_

#include <stdio.h>
#include <stdlib.h>
#ifndef NO_OMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    double *x;
    double *y;
    size_t n;
} data_t;

typedef struct {
    double *lower;
    double *upper;
    double *result;
    size_t n;
} args_t;

typedef struct {
    double *in;
    double *out;
    size_t *index;
    size_t n;
} unsort_t;

typedef struct {
    double *wavelength;
    double *r;
    double *g;
    double *b;
    double gamma;
    size_t n;
} wavelength_to_rgb_t;

data_t *create_data(double *x, double *y, size_t n);
args_t *create_args(double *lower, double *upper, double *out, size_t n);
unsort_t *create_unsort(double *in, double *out, size_t *index, size_t n);
wavelength_to_rgb_t *create_wavelength_to_rgb(double *wavelength, double *r, double *g, double *b, double gamma, size_t n);
void free_data(data_t *data);
void free_args(args_t *args);
void free_unsort(unsort_t *unsort);
void free_wavelength_to_rgb(wavelength_to_rgb_t *wavelength_to_rgb);
void integrate(data_t *data, args_t *args);
void unsort(unsort_t *unsort);
void wavelength_to_rgb(wavelength_to_rgb_t *wavelength_to_rgb);

#ifdef __cplusplus
}
#endif

#endif // _INTEGRATE_H_