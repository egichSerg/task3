#define max(X, Y) ((X) < (Y) ? (Y) : (X))

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <openacc.h>

int main(int argc, char* argv[])
{
    acc_set_device_num(3, acc_device_default);
    int netSize = 128;
    double minError = 0.000001;
    int maxIterations = 1000000;
    char* end;
    if (argc != 4){
        printf("You must enter excatly 3 arguments:\n1. Grid size (one number)\n2. Minimal error\n3. Iterations amount\n");
        return -1;
    }
    else{
        netSize = strtol(argv[1], &end, 10);
        minError = strtod(argv[2], &end);
        maxIterations = strtol(argv[3], &end, 10);
    }

    printf("%d, %0.20g, %d\n", netSize, minError, maxIterations);


    //values of net edges
    double tl = 10, //top left
        tr = 20, //top right
        bl = 30, //bottom left
        br = 20; //bottom right

    double horizontalStepTop = (tr - tl) / (netSize - 1), 
        horizontalStepBottom = (br - bl) / (netSize - 1), 
        verticalStepLeft = (bl - tl) / (netSize - 1), 
        verticalStepRight = (br - tr) / (netSize - 1);

    int arrSize = netSize * netSize; //pre-calculate to use every iteration for Daxpy and for data allocation

    double* thermalConductivityMatrix = (double*)malloc(sizeof(double*) * netSize * netSize);
    double* thermalConductivityMatrixMod = (double*)malloc(sizeof(double*) * netSize * netSize);
    double* thermalConductivityMatrixModCopy = (double*)malloc(sizeof(double*) * netSize * netSize);
    double* temp;
    //i - x (cols), j - y (rows), element(i, j) = i * netSize + j


#pragma acc data copyin(thermalConductivityMatrixMod[0:arrSize], thermalConductivityMatrixModCopy[0:arrSize]) copy(thermalConductivityMatrix[0:arrSize])
{
    //init matrix
    #pragma acc parallel loop
    for (int i = 0; i < netSize; i++)
    {
        thermalConductivityMatrix[i * netSize] = verticalStepLeft * i + tl;
        thermalConductivityMatrix[i] = horizontalStepTop * i + tl;
        thermalConductivityMatrix[((netSize - 1) - i) * netSize + (netSize - 1)] = verticalStepRight * i + br;
        thermalConductivityMatrix[(netSize - 1) * netSize + ((netSize - 1) - i)] = horizontalStepBottom * i + br;

        thermalConductivityMatrixMod[i * netSize] = verticalStepLeft * i + tl;
        thermalConductivityMatrixMod[i] = horizontalStepTop * i + tl;
        thermalConductivityMatrixMod[((netSize - 1) - i) * netSize + (netSize - 1)] = verticalStepRight * i + br;
        thermalConductivityMatrixMod[(netSize - 1) * netSize + ((netSize - 1) - i)] = horizontalStepBottom * i + br;
    }

    double error = 10.;
    int iteration = 0;
    int errorIdx;
    const double alpha = -1.0;

    cublasHandle_t handle;
    cublasStatus_t stat;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) { printf("cublas handle creation failed!\n"); return -1; }

    while (error > minError && iteration < maxIterations) {

        #pragma acc data present(thermalConductivityMatrix, thermalConductivityMatrixMod)
        #pragma acc parallel loop reduction(max:error) collapse(2)
        for (int i = 1; i < netSize - 1; i++) {
            for (int j = 1; j < netSize - 1; j++) {
                thermalConductivityMatrixMod[i * netSize + j] = 0.25 * (
                    thermalConductivityMatrix[i * netSize + j + 1] + 
                    thermalConductivityMatrix[i * netSize + j - 1] + 
                    thermalConductivityMatrix[(i + 1) * netSize + j] + 
                    thermalConductivityMatrix[(i - 1) * netSize + j]);
            }
        }

        if (iteration % 100 == 0) {
        error = 0.;
        errorIdx = 0;

        #pragma acc host_data use_device(thermalConductivityMatrix, thermalConductivityMatrixMod, thermalConductivityMatrixModCopy)
        {
                stat = cublasDcopy(handle, arrSize, thermalConductivityMatrixMod, 1, thermalConductivityMatrixModCopy, 1);
                if (stat != CUBLAS_STATUS_SUCCESS) { printf("cublas Dcopy failed!\n"); break; }

                stat = cublasDaxpy(handle, arrSize, &alpha, thermalConductivityMatrix, 1, thermalConductivityMatrixModCopy, 1);
                if (stat != CUBLAS_STATUS_SUCCESS) { printf("cublas daxpy failed!\n"); break; }      

                stat = cublasIdamax(handle, arrSize, thermalConductivityMatrixModCopy, 1, &errorIdx);
                if (stat != CUBLAS_STATUS_SUCCESS) { printf("cublas idamax failed!\n"); break; }
            }
            error = thermalConductivityMatrixModCopy[errorIdx - 1];
        }

        if (iteration % 1000 == 0)
            printf("iteration: %d error = %0.20g\n", iteration, error);

        temp = thermalConductivityMatrix;
        thermalConductivityMatrix = thermalConductivityMatrixMod;
        thermalConductivityMatrixMod = temp;

        iteration++;

    }
    printf("Final error: %0.20g\n", error);
}
    return 0;
}