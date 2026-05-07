# depLCA

Dependent Latent Class Analysis (depLCA) - An R package for performing latent class analysis with dependent structures.

## Overview

depLCA is an R package that implements dependent latent class analysis methods. It combines R with C++ (via Rcpp) to provide efficient computation for latent class analysis with complex dependency structures.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Functions](#functions)
- [Requirements](#requirements)
- [License](#license)
- [Author](#author)

## Features

- **Latent Class Analysis**: Perform LCA on categorical data
- **Dependent Structures**: Handle dependent observations and complex relationships
- **SLCM**: Standard latent class models
- **C++ Implementation**: Fast computation using Rcpp and RcppEigen
- **Flexible Model Specification**: Support for various model configurations

## Installation

### Prerequisites

Make sure you have the following R packages installed:

```r
install.packages("Rcpp")
install.packages("RcppEigen")
```

### Setup

1. Clone or download the repository
2. Navigate to the project directory
3. Load all source files in R:

```r
library(Rcpp)  
library(RcppEigen)  
sourceCpp("rlca.cpp")  
sourceCpp("depLCA.cpp")  
sourceCpp("depLCA_5.cpp")  
source("depLCA.R")  
sourceCpp("slcm.cpp")
```

## Quick Start

Here's a basic example of using depLCA:

```r
# Load the package
library(Rcpp)
library(RcppEigen)

# Source all required files
sourceCpp("rlca.cpp")
sourceCpp("depLCA.cpp")
sourceCpp("depLCA_5.cpp")
source("depLCA.R")
sourceCpp("slcm.cpp")

# Prepare your data
# data should be a matrix or data frame with categorical observations

# Perform latent class analysis
# result <- depLCA(data, nclasses = 3)
```

## Usage

### Basic Analysis

```r
# Define number of classes
nclasses <- 3

# Run analysis
# result <- depLCA(data, nclasses = nclasses)

# View results
# summary(result)
```

### Advanced Analysis

For more complex models with spatial or dependent structures:

```r
# Standard Latent Class Model
# result_spatial <- slcm(data, nclasses = 3, spatial_weights = weights_matrix)
```

## Functions

### Main Functions

- **`depLCA()`** - Perform dependent latent class analysis
- **`slcm()`** - Standard latent class models
- **`rlca()`** - Basic latent class analysis

### Supporting Functions

Refer to the source code documentation for detailed function signatures and parameters.

