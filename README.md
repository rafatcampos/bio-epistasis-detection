Please check the Hiperbio organization repository https://github.com/hiperbio/cross-dpc-episdet for the updated versions of this application.

# bio-epistasis-detection

## DPC++ GPU implementation

The DPC++ implementations for GPU epistasis detection are designed for Gen9 GT2.
They implement 2nd order epistasis detection, with single-objective optimization, using the K2 score objective function.
The user can specify a given number of samples and SNPs. The application will randomly generate a data set and process it, presenting the score, solution and processing time.
Two implementations are available, using either buffers/accessors or Unified Shared Memory (USM) for memory management.

To build:
```sh
make
```
To run for a given number of samples and SNPs:
```sh
./epi_k2 N_SAMPLES N_SNPS
```
