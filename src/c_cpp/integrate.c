#include <stdlib.h>
#include <math.h>
#include "integrate.h"

void integrate(data_t *data, args_t *args)
{
// integrate data->y over data->x with limits->lower and limits->upper
// store the result in result->result
// for each element in the bounds
#ifndef NO_OMP
#pragma omp parallel for shared(data, args)
#endif
    for (size_t i = 0; i < args->n; i++)
    {
        double low = args->lower[i];
        double up = args->upper[i];
        double sum = 0.0;
        size_t lo_ind = 0;
        size_t hi_ind = 0;
        // search lower bound
        size_t j = 0;
        for (; j < data->n - 1; j++)
        {
            if (data->x[j] >= low)
            {
                lo_ind = j;
                break;
            }
        }
        hi_ind = lo_ind;
        // search upper bound
        for (; j < data->n - 1; j++)
        {
            if (data->x[j] >= up)
            {
                hi_ind = j;
                break;
            }
        }
        // if the bounds are the same, the result is the value at the bound
        if (lo_ind == hi_ind)
        {
            args->result[i] = data->y[lo_ind] * (up - low);
            continue;
        }
        // sum the values between the bounds
        for (size_t j = lo_ind; j < hi_ind; j++)
        {
            sum += data->y[j];
        }
        sum *= (up - low);
        args->result[i] = sum;
    }
    return;
}

void unsort(unsort_t *unsort)
{
    // unsort the array in->out using the index array index
    for (size_t i = 0; i < unsort->n; i++)
    {
        unsort->out[unsort->index[i]] = unsort->in[i];
    }
    return;
}

void wavelength_to_rgb(wavelength_to_rgb_t *wavelength_to_rgb)
{
    double gamma = wavelength_to_rgb->gamma;
#ifndef NO_OMP
#pragma omp parallel for shared(wavelength_to_rgb)
#endif
    for (size_t i = 0; i < wavelength_to_rgb->n; i++)
    {
        double wave = wavelength_to_rgb->wavelength[i];
        double r = 0.0;
        double g = 0.0;
        double b = 0.0;
        if (wave >= 3800 && wave < 4400)
        {
            double factor = 0.0;
            factor = 0.3 + 0.7 * (wave - 3800.0) / (4400 - 3800);
            r = pow((-(wave - 4400.0) / (4400 - 3800)) * factor, gamma);
            g = 0.0;
            b = pow((1.0 * factor), gamma);
        }
        else if (wave >= 4400 && wave < 4900)
        {
            r = 0.0;
            g = pow(((wave - 4400.0) / (4900 - 4400)), gamma);
            b = 1.0;
        }
        else if (wave >= 4900 && wave < 5100)
        {
            r = 0.0;
            g = 1.0;
            b = pow((-(wave - 5100.0) / (5100 - 4900)), gamma);
        }
        else if (wave >= 5100 && wave < 5800)
        {
            r = pow(((wave - 5100.0) / (5800 - 5100)), gamma);
            g = 1.0;
            b = 0.0;
        }
        else if (wave >= 5800 && wave < 6450)
        {
            r = 1.0;
            g = pow((-(wave - 6450.) / (6450 - 5800)), gamma);
            b = 0.0;
        }
        else if (wave >= 6450. && wave <= 7500.)
        {
            double factor = 0.0;
            factor = 0.3 + 0.7 * (7500.0 - wave) / (7500 - 6450);
            r = pow((1.0 * factor), gamma);
            g = 0;
            b = 0;
        } 
        else if (wave > 7500 & wave < 10000)
        {
            r = 0.5;
            g = 0.5;
            b = 0.5;
        }
        else
        {
            r = 0.0;
            g = 0.0;
            b = 0.0;
        }
        if (r > 1.0)
        {
            r = 1.0;
        }
        if (g > 1.0)
        {
            g = 1.0;
        }
        if (b > 1.0)
        {
            b = 1.0;
        }
        wavelength_to_rgb->r[i] = r;
        wavelength_to_rgb->g[i] = g;
        wavelength_to_rgb->b[i] = b;
    }
}

data_t *create_data(double *x, double *y, size_t n)
{
    // create a data_t struct with x, y, and n
    data_t *data = (data_t *)malloc(sizeof(data_t));
    data->x = x;
    data->y = y;
    data->n = n;
    return data;
}

args_t *create_args(double *lower, double *upper, double *result, size_t n)
{
    // create an args_t struct with lower, upper, and n
    args_t *args = (args_t *)malloc(sizeof(args_t));
    args->lower = lower;
    args->upper = upper;
    args->result = result;
    args->n = n;
    return args;
}

unsort_t *create_unsort(double *in, double *out, size_t *index, size_t n)
{
    // create an unsort_t struct with in, out, and n
    unsort_t *unsort = (unsort_t *)malloc(sizeof(unsort_t));
    unsort->in = in;
    unsort->out = out;
    unsort->index = index;
    unsort->n = n;
    return unsort;
}

wavelength_to_rgb_t *create_wavelength_to_rgb(double *wavelength, double *r, double *g, double *b, double gamma, size_t n)
{
    // create a wavelength_to_rgb_t struct with wavelength, r, g, b, and n
    wavelength_to_rgb_t *wavelength_to_rgb = (wavelength_to_rgb_t *)malloc(sizeof(wavelength_to_rgb_t));
    wavelength_to_rgb->wavelength = wavelength;
    wavelength_to_rgb->r = r;
    wavelength_to_rgb->g = g;
    wavelength_to_rgb->b = b;
    wavelength_to_rgb->gamma = gamma;
    wavelength_to_rgb->n = n;
    return wavelength_to_rgb;
}

void free_data(data_t *data)
{
    // free the data_t struct
    free(data);
    return;
}

void free_args(args_t *args)
{
    // free the args_t struct
    free(args);
    return;
}

void free_unsort(unsort_t *unsort)
{
    // free the unsort_t struct
    free(unsort);
    return;
}

void free_wavelength_to_rgb(wavelength_to_rgb_t *wavelength_to_rgb)
{
    // free the wavelength_to_rgb_t struct
    free(wavelength_to_rgb);
    return;
}
