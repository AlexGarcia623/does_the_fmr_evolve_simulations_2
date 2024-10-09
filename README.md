# Does the FMR evolve with redshift? II: The Evolution in Normalisation of the Mass-Metallicity Relation

This repo contains the analysis scripts, figures, and data products from the paper Garcia et al. (2024c)

## Contents

### Data:

Contains minimal working example data for generation of the figures associated with this work for Illustris, IllustrisTNG, EAGLE, and SIMBA simulations from redshift 0-8.

All credit for simulation data goes to the respective collaborations: see [Illustris](https://www.illustris-project.org/), [IllustrisTNG](https://www.tng-project.org/), [EAGLE](https://icc.dur.ac.uk/Eagle/), and [SIMBA](http://simba.roe.ac.uk/).

Also includes data from particles on different metallicity types in the simulations. This data is used to support the findings of Appendix B.

### Data Reduction

!!! Not required to run the scripts to generate plots !!!
!!! Is required if you would like to significantly alter selection criteria !!!

Contains scripts used to generate data products from raw simulation data 

### Figures (pdfs)

Contains pdf versions of all figures 

### Rest of files

Make Figures

- make_all_figs.py -- Generate all figures (excluding .key files)

Individual Figures:

- appendix_A1.py -- Generate Figure A1
- appendix_B1.py -- Generate Figure B1
- appendix_B2.py -- Generate Figure B2
- appendix_C1.py -- Generate Figure C1
- appendix_C2.py -- Generate Figure C2
- appendix_D1.py -- Generate Figure D1
- Figure1.key -- Keynote file used to make Figure 7
- Figure2.py  -- Generate Figure 2
- Figure3.py  -- Generate Figure 3
- Figure4.py  -- Generate Figure 4
- Figure5.py  -- Generate Figure 5
- Figure6.py  -- Generate Figure 6
- Figure7.key -- Keynote file used to make Figure 7
- Figure8.py  -- Generate Figure 8

Helpers

- alpha_types       -- Contains useful functions for epsilon portion of analysis (alpha name carried over from previous study)
- Curti.py          -- Contains functions related to Appendix C
- helpers.py        -- Contains helper functions for all files
- helpers_appendixB -- Contains helper functions for Appendix B
- plotting.py       -- Contains useful functions for plotting

## Questions Regarding scripts/paper

Please contact [alexgarcia@virginia.edu](mailto:alexgarcia@virginia.edu), Alex Garcia PhD Student University of Virginia
