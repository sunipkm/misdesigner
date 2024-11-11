#include<stdlib.h>
#include "integrate.h"


void integrate(data_t *data, args_t *args) {
    // integrate data->y over data->x with limits->lower and limits->upper
    // store the result in result->result
    // for each element in the bounds
    #ifndef NO_OMP
    #pragma omp parallel for shared(data, args)
    #endif
    for (size_t i = 0; i < args->n; i++) {
        double low = args->lower[i];
        double up = args->upper[i];
        double sum = 0.0;
        size_t lo_ind = 0;
        size_t hi_ind = 0;
        // search lower bound
        size_t j = 0;
        for (; j < data->n - 1; j++) {
            if (data->x[j] >= low) {
                lo_ind = j;
                break;
            }
        }
        hi_ind = lo_ind;
        // search upper bound
        for (; j < data->n - 1; j++) {
            if (data->x[j] >= up) {
                hi_ind = j;
                break;
            }
        }
        // if the bounds are the same, the result is the value at the bound
        if (lo_ind == hi_ind) {
            args->result[i] = data->y[lo_ind] * (up - low);
            continue;
        }
        // sum the values between the bounds
        for (size_t j = lo_ind; j < hi_ind; j++) {
            sum += data->y[j];
        }
        sum *= (up - low);
        args->result[i] = sum;
    }
    return;
}

void unsort(unsort_t *unsort) {
    // unsort the array in->out using the index array index
    for (size_t i = 0; i < unsort->n; i++) {
        unsort->out[unsort->index[i]] = unsort->in[i];
    }
    return;
}

data_t *create_data(double *x, double *y, size_t n) {
    // create a data_t struct with x, y, and n
    data_t *data = (data_t *)malloc(sizeof(data_t));
    data->x = x;
    data->y = y;
    data->n = n;
    return data;
}

args_t *create_args(double *lower, double *upper, double *result, size_t n) {
    // create an args_t struct with lower, upper, and n
    args_t *args = (args_t *)malloc(sizeof(args_t));
    args->lower = lower;
    args->upper = upper;
    args->result = result;
    args->n = n;
    return args;
}

unsort_t *create_unsort(double *in, double *out, size_t *index, size_t n) {
    // create an unsort_t struct with in, out, and n
    unsort_t *unsort = (unsort_t *)malloc(sizeof(unsort_t));
    unsort->in = in;
    unsort->out = out;
    unsort->index = index;
    unsort->n = n;
    return unsort;
}

void free_data(data_t *data) {
    // free the data_t struct
    free(data);
    return;
}

void free_args(args_t *args) {
    // free the args_t struct
    free(args);
    return;
}

void free_unsort(unsort_t *unsort) {
    // free the unsort_t struct
    free(unsort);
    return;
}

