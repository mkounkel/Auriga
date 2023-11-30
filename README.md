# Auriga

Auriga neural net predicts age, extinction, and distance to stellar populations

## Installation:

```pip install auriga```
(requires Python3)

## Keywords:
```
positional arguments:
  tableIn               Input table with Gaia DR2 source ids and cluster ids

optional arguments:
  -h, --help            show this help message and exit
  --tutorial            Use included test.fits or test.csv files as inputs
  --tableOut TABLEOUT   Prefix of the csv file into which the cluster properties should be written, default tableIn-out
  --iters ITERS         Number of iterations of each cluster is passed through Auriga to generate the errors, default 10
  --localFlux           Download necessary flux from Gaia archive for all source ids, default True
  --saveFlux SAVEFLUX   If downloading flux, prefix of file where to save it, default empty
  --silent              Suppress print statements, default False
  --cluster CLUSTER     Column with cluster membership
  --source_id SOURCE_ID
                        Column with Gaia DR2 source id,
  --gaiaFluxErrors      If loading flux, whether uncertainties in Gaia bands have been converted from flux to magnitude, default True
  --g G                 If loading flux, column for G magnitude
  --bp BP               If loading flux, column for BP magnitude
  --rp RP               If loading flux, column for RP magnitude
  --j J                 If loading flux, column for J magnitude
  --h H                 If loading flux, column for H magnitude
  --k K                 If loading flux, column for K magnitude
  --parallax PARALLAX   If loading flux, column for parallax
  --eg EG               If loading flux, column for uncertainty in G magnitude
  --ebp EBP             If loading flux, column for uncertainty in BP magnitude
  --erp ERP             If loading flux, column for uncertainty in RP magnitude
  --ej EJ               If loading flux, column for uncertainty in J magnitude
  --eh EH               If loading flux, column for uncertainty in H magnitude
  --ek EK               If loading flux, column for uncertainty in K magnitude
  --eparallax EPARALLAX
                        If loading flux, column for uncertainty in parallax
  --gf GF               If uncertainties have not been converted to magnitudes, column for G flux
  --bpf BPF             If uncertainties have not been converted to magnitudes, column for BP flux
  --rpf RPF             If uncertainties have not been converted to magnitudes, column for RP flux
  --egf EGF             If uncertainties have not been converted to magnitudes, column for uncertainty in G flux
  --ebpf EBPF           If uncertainties have not been converted to magnitudes, column for uncertainty in BP flux
  --erpf ERPF           If uncertainties have not been converted to magnitudes, column for uncertainty in RP flux
  --memoryOnly          Store table only in memory without saving to disk
  --ver VER             Version of Gaia data to download, default DR3

```

## Examples:
Downloading photometry from the Gaia Archive for the sources defined in the fits table, saving the fluxes, and generating the outputs
```
auriga test.fits --tableOut test-out --saveFlux test --tutorial 
```

Using previously downloaded fluxes to generate predictions. 20 implementations of each cluster are generated instead of 10, to estimate the uncertainties in the cluster parameters
```
auriga test.csv --localFlux --iters=20 --tutorial
```

Using previously downloaded fluxes, defining all the necessary columns
```
auriga test.fits --localFlux --gaiaFluxErrors --g phot_g_mean_mag --bp phot_bp_mean_mag \
           --rp phot_rp_mean_mag --j j_m --h h_m --k ks_m --ej j_msigcom --eh h_msigcom \
           --ek ks_msigcom --eparallax parallax_error --tutorial --silent

```
Using from within a code, outside of a command line
```
from auriga.auriga import getClusterAge
t=Table.read('test.csv')
out=main(t,localFluxes=True)
out=main('test.csv',tutorial=True,memoryOnly=True)
```

## Required packages:
* Astropy
* Astroquery
* Pytorch
* Pandas
