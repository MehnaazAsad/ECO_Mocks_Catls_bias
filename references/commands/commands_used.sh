#!/usr/bin/env bash

##############################################################################
# ---- Author ----:
# - Victor Calderon (victor.calderon@vanderbilt.edu)
#
# ---- Description ----:
# List of commands used for data preprocesssing and analysis for this
# ML project.
#
# ---- Date Modified ----:
# 2019-07-26
#
##############################################################################

##### --------------------------- 2019-07-26 --------------------------- #####

##############################################################################
##################### -------- MAIN ANALYSIS -------- ########################
##############################################################################

# Deleting old catalogues
make clean
make delete_mock_catls

# Creating new mock catalogues
make CPU_FRAC=0.75 COSMO_CHOICE="Planck" HALOTYPE="m200b" REMOVE_FILES="True" SURVEY="ECO" catl_mr_make
make CPU_FRAC=0.75 COSMO_CHOICE="Planck" HALOTYPE="m200b" REMOVE_FILES="True" SURVEY="A" catl_mr_make
make CPU_FRAC=0.75 COSMO_CHOICE="Planck" HALOTYPE="m200b" REMOVE_FILES="True" SURVEY="B" catl_mr_make

make CPU_FRAC=0.75 COSMO_CHOICE="Planck" HALOTYPE="mvir" REMOVE_FILES="True" SURVEY="ECO" catl_mr_make
make CPU_FRAC=0.75 COSMO_CHOICE="Planck" HALOTYPE="mvir" REMOVE_FILES="True" SURVEY="A" catl_mr_make
make CPU_FRAC=0.75 COSMO_CHOICE="Planck" HALOTYPE="mvir" REMOVE_FILES="True" SURVEY="B" catl_mr_make