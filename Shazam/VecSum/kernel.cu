#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>
#include<iostream>
//#include "device_functions.h"

int main(int argc, char *argv[])
{
	FILE *fp;
	int temp;
	int *grades = NULL;
	long long count = 1;
	long index;

	printf("The grade is %d arguments:\r\n", argc);

	for (int i = 0; i < argc; ++i)
		printf("%s \r\n", argv[i]);


	fp = fopen(argv[1], "rb+");

	while (fscanf(fp, "%d", &temp) != EOF)

	{


		if (grades == NULL)

		{

			grades = (int *) malloc(sizeof(temp));
			*grades = temp;

			printf("The grade is %d\r\n", temp);
		}

		else
		{
			//printf("The grade is realloc %d\r\n", temp);
			count++;
			grades = (int *)realloc(grades, sizeof(grades)*count);
			index = count - 1;
			*(grades + index) = temp;
			//printf("the index is %d\r\n",index);

		}

	}
	printf("Done");

	/** lets print the data now **/

	temp = 0;
	/*
	while (index >= 0)
	{

		printf("the read value is %d\r\n", *(grades + temp));
		index--;
		temp++;

	} */

	fclose(fp);

	free(grades);
	grades = NULL;


}
