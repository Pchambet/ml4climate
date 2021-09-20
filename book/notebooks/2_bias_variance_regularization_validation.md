---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

+++ {"slideshow": {"slide_type": "slide"}}

# Bias-Variance, Regularization, Validation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab.in2p3.fr%2Fenergy4climate%2Fpublic%2Feducation%2Fmachine_learning_for_climate_and_energy/master?filepath=book%2Fnotebooks%2F2_bias_variance_regularization_validation.ipynb)

+++ {"slideshow": {"slide_type": "subslide"}}

<div class="alert alert-block alert-warning">
    <b>Prerequisites</b>
    
- [Elements of Probability Theory](appendix_elements_of_probability_theory.ipynb)  
- Define *supervized* and *unsupervized* learning
- Give the difference between *qualitative* and *quantitative* variables and define of *regression* and *classification*
</div>

+++ {"slideshow": {"slide_type": "subslide"}}

<div class="alert alert-block alert-info">
    <b>Learning Outcomes</b>
    
- Define a supervised learning problem
- Apply the methodology to a multiple linear regression
- Understand when and why a model does or does not generalize well on unseen data
- Understand overfitting/underfitting trade-off
- Use regularization to prevent overfitting
</div>

+++ {"slideshow": {"slide_type": "slide"}}

## Supervised Learning Problem

+++ {"slideshow": {"slide_type": "subslide"}}

### Example: Home Heating

<img alt="Thermostat" src="images/erik-mclean-fSLI8RdCdyk-unsplash.jpg" width=500  style="float:right">

When its cold in my accommodation, I heat it.

$\rightarrow$ I suspect a relationship between the energy I consume to heat my accommodation and the outdoor temperature, although other factors may also play a role.

+++ {"slideshow": {"slide_type": "fragment"}}

How to *predict* how much energy I consume on average depending on the outdoor temperature ?

+++ {"slideshow": {"slide_type": "subslide"}}

### Three different approaches

| Process-based | Expert-based | *Statistical* |
| --- | --- | --- |
| Use some approximation of the heat equation in my accommodation given heat sources (radiators) and sinks (outdoor). | A thermal engineer diagnoses my accommodation based on his/her knowledge and/or on conventions. | Use energy-consumption and outdoor-temperature data to estimate parameters of a model. |
| <img alt="Building Energy Model" src="images/Heat_losses_of_the_building-fr.svg" width="180"> | <img alt="DPE" src="images/Diagnostic_de_performance_énergétique.svg" width="150"> | <img alt="Statistical Model" src="images/linear_ols.svg" width="280"> |

+++ {"slideshow": {"slide_type": "slide"}}

### Supervised Learning Objective

To define a supervised-learning problem we need:

- an *input* vector $X$ of $X_1, \ldots, X_p$ input variables, or *features* and
- an *output* or *target* variable $Y$ used for supervision.

+++ {"slideshow": {"slide_type": "fragment"}}

<hr>

**Supervised Learning Objective**
<br>
Construct the "best" prediction rule to predict $Y$ based on some *training data*: $(\boldsymbol{x}_i, y_i), i = 1, \ldots N$. 

<hr>

+++ {"slideshow": {"slide_type": "subslide"}}

#### Supervised-Learning Flow: Fit and Predict

<img alt="Pipeline Fit" src="images/api_diagram-predictor.fit.svg" width="500">

+++ {"slideshow": {"slide_type": "fragment"}}

<img alt="Pipeline Fit" src="images/api_diagram-predictor.predict.svg" width="600">

+++ {"slideshow": {"slide_type": "subslide"}}

#### Supervised-Learning Flow: Transform-Fit and Transform-Predict

<img alt="Pipeline Fit" src="images/api_diagram-pipeline.fit.svg" width="750">

+++ {"slideshow": {"slide_type": "fragment"}}

<img alt="Pipeline Fit" src="images/api_diagram-pipeline.predict.svg" width="860">

+++ {"slideshow": {"slide_type": "subslide"}}

The $i$th *observation* of $X_j$ in the *sample* is given by the element $x_{ij}$ of the $N\times p$ *input-data matrix* $\mathbf{X}$.

The $i$th observation of $y$ is given by the element $y_i$ of the $N \times 1$ *output-data vector* $\mathbf{y}$.

+++ {"slideshow": {"slide_type": "subslide"}}

#### Regression / Classification

| Regression | Classification |
| ------------------------------------- | --------------------------------------- |
| $Y$ is quantitative | $Y$ is qualitative |
| <img src="images/artur-solarz-hihmzc-TToc-unsplash.jpg" width="350"> | <img src="images/supervised.png" width="300"> |

+++ {"slideshow": {"slide_type": "subslide"}}

#### Example: Electricity consumption dependence on temperature

- *Raw input*: temperature averaged over an administrative region of metropolitan France
- *Target*: regional electricity consumption

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# Import modules
from pathlib import Path
import numpy as np
import pandas as pd
import hvplot.pandas
import panel as pn
pn.extension()

# Set data directory
data_dir = Path('data')

# Set keyword arguments for pd.read_csv
kwargs_read_csv = dict(header=0, index_col=0, parse_dates=True)

# Set first and last years
FIRST_YEAR = 2014
LAST_YEAR = 2019

# Define file path
filename = 'surface_temperature_merra2_{}-{}.csv'.format(
    FIRST_YEAR, LAST_YEAR)
filepath = Path(data_dir, filename)

# Read hourly temperature data averaged over each region
df_temp = pd.read_csv(filepath, **kwargs_read_csv).resample('D').mean()
label_temp = 'Temperature (°C)'

# Read hourly demand data summed over each region
filename = 'reseaux_energies_demand_demand.csv'
filepath = Path(data_dir, filename)
df_dem = pd.read_csv(filepath, **kwargs_read_csv).resample('D').sum()
label_dem = 'Demand (MWh)'
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# Scatter plot of demand versus temperature
def scatter_temp_dem(region_name, year):
    df = pd.concat([df_temp[region_name], df_dem[region_name]],
                   axis='columns', ignore_index=True).loc[str(year)]
    df.columns = [label_temp, label_dem]
    return df.hvplot.scatter(x=label_temp, y=label_dem, width=500,
                             xlim=[-5, 30])


text = pn.pane.Markdown("""
## Generalizing vs. Memorizing
### New data will differ from training data.         
### Plus there is *noise* from unresolved factors.      
### -> we want to be able to *generalize*, not just *memorize*""")
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# Show
pn.Row(pn.interact(scatter_temp_dem, region_name=df_dem.columns,
                   year=range(FIRST_YEAR, LAST_YEAR)),
       pn.Spacer(width=100), text)
```

+++ {"slideshow": {"slide_type": "slide"}}

### Supervised Learning Problem Definition

Given the output $Y$,

- define features $X = (X_1, \ldots, X_p)$ based on (transformed) raw inputs
- define model by a function $\mathcal{M}: X \mapsto \mathcal{M}(X)$
- define *loss function* $L(Y, \mathcal{M}(X))$
- choose a training set $(\boldsymbol{x}_i, y_i), i = 1, \ldots, N$

+++ {"slideshow": {"slide_type": "fragment"}}

Linear model: $\mathcal{M}_{\boldsymbol{\beta}}(X) = \beta_0 + \sum_{j = 1}^p X_j \beta_j$

Squared error loss: $L(Y, \mathcal{M}(X)) = \left(Y - \mathcal{M}(X)\right)^2$

+++ {"slideshow": {"slide_type": "subslide"}}

<div class="alert alert-block alert-warning">
    <b>Assumption</b>
    
All random variables and random vectors have finite variance and have densities (they are absolutely continuous with respect to the Lebesgue measure).
</div>

+++ {"slideshow": {"slide_type": "slide"}}

<hr>

**Expected Prediction Error**
<br>
\begin{equation}
\mathrm{EPE}(\mathcal{M}) = \mathbb{E}(L(Y, \mathcal{M}(X)))
= \int L(y, \mathcal{M}(x)) f_{X, Y}(x, y) dx dy.
\end{equation}

<hr>

<hr>

**Supervised Learning Objective (Concrete)**
<br>
Find $\mathcal{M}$ such that the EPE is minimized.

<hr>

+++ {"slideshow": {"slide_type": "subslide"}}

From the law of total expectation, we have that

\begin{align}
\mathrm{EPE}(\mathcal{M})
&= \mathbb{E}(\mathbb{E}[L(Y, \mathcal{M}(X)) | X])\\
&= \int L(y, \mathcal{M}(x)) f_{Y | X = x}(y) f_X(x)dy dx.
\end{align}

+++ {"slideshow": {"slide_type": "fragment"}}

The EPE can thus be interpreted as averaging over the inputs the prediction error for any input and can be minimized pointwise:

\begin{equation}
\mathcal{M}(x) = \mathrm{argmin}_c \mathbb{E}\left[L(Y, c) | X = x\right].
\end{equation}

+++ {"slideshow": {"slide_type": "subslide"}}

### The Case of Squared Error Loss

The EPE using the squared error loss is

\begin{equation}
\mathrm{EPE}(f) = \mathbb{E}((Y - f(X))^2).
\end{equation}

Then
\begin{equation}
\mathcal{M}(x) = \underset{c}{\mathrm{argmin}} \ \mathbb{E}\left[(Y - c)^2 | X = x\right].
\end{equation}

+++ {"slideshow": {"slide_type": "fragment"}}

Since the expectation is the value that minimizes the expectation of the squared deviations (see [Appendix: Elements of Probability Theory](appendix_elements_of_probability_theory.ipynb)), the optimal solution is 

\begin{equation}
\mathcal{M}(x) = \mathbb{E}(Y | X = x)
\end{equation}

+++ {"slideshow": {"slide_type": "subslide"}}

In other words:

<div class="alert alert-block alert-info">
    <b>Theorem</b>
    
The best prediction of the output for any input is the conditional expectation, when best is measured by average squared error.
</div>

+++ {"slideshow": {"slide_type": "fragment"}}

> ***Question (optional)***
> - What is the statistic giving the solution minimizing the EPE if we use the absolute error loss $|Y - f(X)|$ instead of the squared error loss?

+++ {"slideshow": {"slide_type": "slide"}}

## Ordinary Least Squares (OLS)

+++ {"slideshow": {"slide_type": "subslide"}}

### Strengths

- Simple to use;
- Easily interpretable in terms of variances and covariances;
- Can outperform fancier nonlinear models for prediction, especially in situations with:
  - small training samples,
  - low signal-to-noise ratio,
  - sparse data.
- Expandable to nonlinear transformations of the inputs;
- Can be used as a simple reference to learn about machine learning methodologies (supervised learning, in particular).

+++ {"slideshow": {"slide_type": "subslide"}}

### Linear Model

\begin{equation}
\mathcal{M}_{\boldsymbol{\beta}}(X) = \underbrace{\beta_0}_{\mathrm{intercept}} + \sum_{j = 1}^p X_j \beta_j
\end{equation}

+++ {"slideshow": {"slide_type": "subslide"}}

$X_j$ can come from :
- quantitative inputs;
- transformations of quantitative inputs, such as log or square;
- basis expansions, such as $X_2 = X_1^2$, $X_3 = X_1^3$;
- interactions between variables, for example, $X_3 = X_1 \cdot X_2$;
- numeric or "dummy" coding of the levels of qualitative inputs. For example, $X_j, j = 1, \ldots, 5$, such that $X_j = I(G = j)$.

+++ {"slideshow": {"slide_type": "subslide"}}

### Residual Sum of Squares

The sample-mean estimate of the Expected Training Error with Squared Error Loss gives the *Residual Sum of Squares* (RSS) depending on the parameters:

\begin{equation}
\mathrm{RSS}(\beta)
= \sum_{i = 1}^N \left(y_i - \mathcal{M}(x_i)\right)^2
 = \sum_{i = 1}^N \left(y_i - \beta_0 - \sum_{j = 1}^p x_{ij} \beta_j\right)^2.
\end{equation}

<img alt="Linear fit" src="images/linear_fit_red.svg" width="360" style="float:left">
<img alt="Linear fit" src="images/lin_reg_3D.svg" width="400" style="float:right">

+++ {"slideshow": {"slide_type": "subslide"}}

The *coefficient of determination* $R^2$ relates to the RSS as such,

\begin{equation}
R^2(\boldsymbol{\beta}) = 1 - \frac{\mathrm{RSS}(\boldsymbol{\beta})}{\mathrm{TSS}},
\end{equation}

where $\mathrm{TSS} = \sum_{i = 1}^N (y_i - \bar{y})^2$ is the *Total Sum of Squares* and $\bar{y} = \sum_{i = 1}^N y_i$ is the sample mean.

> ***Question (optional)***
> - Express $R^2$ in terms of explained variance.
> - Show that $R^2$ is invariant under linear transformations the target.

+++ {"slideshow": {"slide_type": "slide"}}

### How to Minimize the RSS ?

Denote by $\mathbf{X}$ the $N \times (p + 1)$ input-data matrix.

The 1st column of $\mathbf{X}$ is associated with the intercept and is given by the $N$-dimensional vector $\mathbf{1}$ with all elements equal to 1.

Then,
\begin{equation}
\mathrm{RSS}(\beta) = \left(\mathbf{y} - \mathbf{X} \boldsymbol{\beta}\right)^\top \left(\mathbf{y} - \mathbf{X} \boldsymbol{\beta}\right).
\end{equation}

+++ {"slideshow": {"slide_type": "subslide"}}

> ***Question***
> - Show that the following parameter estimate minimizes the RSS.
> - Show that this solution is unique if and only if $\mathbf{X}^\top\mathbf{X}$ is positive definite (optional).
> - When could this condition not be fulfilled (optional)?

\begin{equation}
    \hat{\boldsymbol{\beta}} = \left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \left(\mathbf{X}^\top \mathbf{y}\right)
\end{equation}

+++ {"slideshow": {"slide_type": "subslide"}}

The predictions with parameters $\hat{\boldsymbol{\beta}}$ from the input data are given by

\begin{equation}
\hat{\mathbf{y}} = \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X} \left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \left(\mathbf{X}^\top \mathbf{y}\right).
\end{equation}

The residual vector is given by $\hat{\mathbf{z}} = \mathbf{y} - \hat{\mathbf{y}}$.

> ***Question (optional)***
> - Show that $\hat{\mathbf{y}}$ is the orthogonal projection of $\mathbf{y}$ on the subspace of $\mathbb{R}^N$ spanned by the columns of $\mathbf{X}$ (i.e the column space of $\mathbf{X}$) and that $\hat{\mathbf{z}}$ is orthogonal to this space.

+++ {"slideshow": {"slide_type": "subslide"}}

Suppose that the inputs $\mathbf{x}_1, \ldots, \mathbf{x}_p$ (the columns of the data matrix $\mathbf{X}$) are orthogonal; that is $\mathbf{x}_j^\top \mathbf{x}_k = 0$ for all $j \ne k$.

> ***Question***
> - Show that $\hat{\beta} = \mathbf{x}_j^\top \mathbf{y} / (\mathbf{x}_j^\top \mathbf{x}_j)$ for all $j$.
> - How do the inputs influence each other's parameter estimates in the model?

+++ {"slideshow": {"slide_type": "slide"}}

### Graphical Interpretation and Gram-Schmidt Algorithm

Let $\mathbf{x}$, $\mathbf{y}$ be the data vectors of the 1-dimensional random variables $X$, $Y$.

> ***Question***
> - Without intercept, what is the OLS fit of $\hat{\beta}_1$?
> - With intercept, but assuming that $\bar{x} = 0$, what is the OLS fit of $\hat{\beta}_0$ and $\hat{\beta}_1$?
> - What if $\mathbf{x}$ and $\mathbf{1}$ are not orthogonal?
> - How does the colinearity of $\mathbf{x}$ and $\mathbf{1}$ affect the sensitivity of $\hat{\beta}_1$ to sampling ?

+++ {"slideshow": {"slide_type": "subslide"}}

By *regressing* $\mathbf{b}$ on $\mathbf{a}$ we mean regressing with input $\mathbf{a}$ and target $\mathbf{b}$.

> ***Question***
> - Regress $\mathbf{x}$ on $\mathbf{1}$ and compute the resulting residual $\hat{\mathbf{z}}_1$.
> - Regress $\mathbf{y}$ on $\hat{\mathbf{z}}_1$. The result should be familiar.
> - Interpret the above procedure graphically.
> - Generalize this procedure to the case of $p$ inputs and express the $j$th estimate in terms of some $\hat{\mathbf{z}}_j$ as $\hat{\beta}_j = \hat{\mathbf{z}_j}^\top \mathbf{y} / (\hat{\mathbf{z}_j}^\top \hat{\mathbf{z}_j})$ (optional).

+++ {"slideshow": {"slide_type": "slide"}}

### Gauss-Markov Theorem

We now assume that $Y = \boldsymbol{X}^\top \boldsymbol{\beta} + \epsilon$, where the observations of $\epsilon$ are *uncorrelated* and with *mean zero* and *constant variance* $\sigma^2$.

> ***Question (optional)***
> - Knowing that $\boldsymbol{X} = \boldsymbol{x}$, show that the observations of $y$ are uncorrelated, with mean $\boldsymbol{x}^\top \boldsymbol{\beta}$ and variance $\sigma^2$.
> - Show that $\mathbb{E}(\hat{\boldsymbol{\beta}} | \mathbf{X}) = \boldsymbol{\beta}$ and $\mathrm{Var}(\hat{\boldsymbol{\beta}} | \mathbf{X}) = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}$.
> - Show that $\hat{\sigma}^2 = \sum_{i = 1}^N (y_i - \hat{y}_i)^2 / (N - p - 1)$ is an unbiased estimate of $\sigma^2$, i.e $\mathbb{E}(\hat{\sigma}^2) = \sigma^2$.

+++ {"slideshow": {"slide_type": "subslide"}}

> ***Question (optional)***
> - Express the variances of the parameter estimates in terms of the orthogonal basis of the column space of $\mathbf{X}$ constructed above.
> - How does the precision of $\hat{\beta}_j$ depend on the input data?

+++ {"slideshow": {"slide_type": "subslide"}}

<div class="alert alert-block alert-info">
    <strong>Gauss-Markov Theorem</strong>
    
Least-squares estimates of the parameters have the smallest variance among all linear unbiased estimates. The OLS is BLUE (Best Linear Unbiased Estimator).
</div>

Let $\tilde{\boldsymbol{\beta}}$ be any estimate of the parameters.
We mean that for any linear combination defined by the vector $\boldsymbol{a}$,

\begin{equation}
    \mathrm{Var}(\boldsymbol{a}^\top \hat{\boldsymbol{\beta}}) \le \mathrm{Var}(\boldsymbol{a}^\top \tilde{\boldsymbol{\beta}}).
\end{equation}

> ***Question (optional)***
> - Prove this theorem.

+++ {"slideshow": {"slide_type": "slide"}}

### Confidence Intervals

We now assume that the error $\epsilon$ is a Gaussian random variable, i.e $\epsilon \sim N(0, \sigma^2)$ and would like to test the null hypothesis that $\beta_j = 0$.

> ***Question (optional)***
> - Show that $\hat{\boldsymbol{\beta}} \sim N(\boldsymbol{\beta}, (\mathbf{X}^\top \mathbf{X}) \sigma^2)$.
> - Show that $(N - p - 1) \hat{\sigma}^2 \sim \sigma^2 \ \chi^2_{N - p - 1}$, a chi-squared distribution with $N - p - 1$ degrees of freedom. 
> - Show that $\hat{\boldsymbol{\beta}}$ and $\hat{\sigma}^2$ are statistically independent.

+++ {"slideshow": {"slide_type": "subslide"}}

With $v_j = [(\mathbf{X}^\top \mathbf{X})^{-1}]_{jj}$, we define the *standardized coefficient* or *Z-score*
\begin{equation}
z_j = \frac{\hat{\beta}_j}{\hat{\sigma} \sqrt{v_j}}.
\end{equation}

> ***Question (optional)***
> - Show that $z_j$ is distributed as $t_{N - p - 1}$ (a Student's-$t$ distribution with $N - p - 1$ degrees of freedom).
> - Show that the $1 - 2 \alpha$ confidence interval for $\beta_j$ is $(\hat{\beta}_j - z^{(1 - \alpha)}_{N - p - 1} \hat{\sigma} \sqrt{v_j}, \hat{\beta}_j + z^{(1 - \alpha)}_{N - p - 1} \hat{\sigma} \sqrt{v_j})$, where $z^{(1 - \alpha)}_{N - p - 1}$ is the $(1 - \alpha)$ percentile of $t_{N - p - 1}$.

+++ {"slideshow": {"slide_type": "slide"}}

## Overfitting and Underfitting

Which fit do you prefer?

<img alt="Linear fit" src="images/linear_ols.svg" width="450" style="float:left">
<img alt="Linear fit" src="images/linear_splines.svg" width="450" style="float:right">

+++ {"slideshow": {"slide_type": "subslide"}}

Which model performs better on new data?

<img alt="Linear fit" src="images/linear_ols_test.svg" width="450" style="float:left">
<img alt="Linear fit" src="images/linear_splines_test.svg" width="450" style="float:right">

+++ {"slideshow": {"slide_type": "subslide"}}

A harder example:

<img alt="Linear fit" src="images/ols_simple_test.svg" width="450" style="float:left">
<img alt="Linear fit" src="images/splines_cubic_test.svg" width="450" style="float:right">

+++ {"slideshow": {"slide_type": "subslide"}}

### Varying model complexity

<img alt="Linear fit" src="images/polynomial_overfit_truth.svg" width="450" style="float:left">

- Data generated by a random process
  - Sample a value of $X$
  - Transform with 9th-degree polynomial
  - Add noise to get samples of $Y$

+++ {"slideshow": {"slide_type": "subslide"}}

### Varying model complexity

<img alt="Linear fit" src="images/polynomial_overfit_0.svg" width="450" style="float:left">

- Data generated by a random process
- In fact, this process is unknown
- We can only access observations

+++ {"slideshow": {"slide_type": "subslide"}}

### Varying model complexity

<img alt="Linear fit" src="images/polynomial_overfit_1.svg" width="450" style="float:left">

- Data generated by a random process
- In fact, this process is unknown
- We can only access observations
- Fit polynomials of various degrees

+++ {"slideshow": {"slide_type": "subslide"}}

### Varying model complexity

<img alt="Linear fit" src="images/polynomial_overfit_2.svg" width="450" style="float:left">

- Data generated by a random process
- In fact, this process is unknown
- We can only access observations
- Fit polynomials of various degrees

+++ {"slideshow": {"slide_type": "subslide"}}

### Varying model complexity

<img alt="Linear fit" src="images/polynomial_overfit_5.svg" width="450" style="float:left">

- Data generated by a random process
- In fact, this process is unknown
- We can only access observations
- Fit polynomials of various degrees

+++ {"slideshow": {"slide_type": "subslide"}}

### Varying model complexity

<img alt="Linear fit" src="images/polynomial_overfit_9.svg" width="450" style="float:left">

- Data generated by a random process
- In fact, this process is unknown
- We can only access observations
- Fit polynomials of various degrees

+++ {"slideshow": {"slide_type": "subslide"}}

### Varying model complexity

<img alt="Linear fit" src="images/polynomial_overfit.svg" width="450" style="float:left">

- Data generated by a random process
- In fact, this process is unknown
- We can only access observations
- Fit polynomials of various degrees

+++ {"slideshow": {"slide_type": "subslide"}}

### Overfit: model too complex

<img alt="Linear fit" src="images/polynomial_overfit_simple_legend.svg" width="450" style="float:left;margin-right:40px">

Model too complex for the data:
- Its best fit would approximate well the process
- However, its flexibility captures noise

+++ {"slideshow": {"slide_type": "fragment"}}

**Not enough data - Too much noise**

+++ {"slideshow": {"slide_type": "subslide"}}

### Underfit: model too simple

<img alt="Linear fit" src="images/polynomial_underfit_simple.svg" width="450" style="float:left;margin-right:40px">

Model too simple for the data:
- Best fit would not approximate well the process
- Yet it captures little noise

+++ {"slideshow": {"slide_type": "fragment"}}

**Plenty of data - Low noise**

+++ {"slideshow": {"slide_type": "subslide"}}

### Partial Summary

- Models too complex for the data **overfit**:
  - they explain too well the data that they have seen
  - they do not generalize
- Models too simple for the data **underfit**:
  - they capture no noise
  - they are limited by their expressivity

+++ {"slideshow": {"slide_type": "slide"}}

## Comparing train and test errors

+++ {"slideshow": {"slide_type": "subslide"}}

### Train vs Test (Prediction) Error

<img src="images/linear_splines_test.svg" style="float:left;margin-right:20px" width="600">

- Errors on the *train data*:

\begin{equation}
\overline{\mathrm{err}} = \frac{1}{N} \sum_{i = 1}^N L(y_i, \hat{\mathcal{M}}(\mathbf{x}_i))
\end{equation}

- Errors on the *test data* (generalization):

\begin{equation}
\mathrm{Err}_\mathcal{T} = \mathbb{E}\left[L(Y, \hat{\mathcal{M}}(\boldsymbol{X})) | \mathcal{T}\right]
\end{equation}

where $\hat{\mathcal{M}}$ is estimated based on a fixed training set $\mathcal{T}$.

+++ {"slideshow": {"slide_type": "subslide"}}

### Train vs test error: increasing complexity

<img src="images/polynomial_overfit_test_1.svg" width="450" style="float:left">
<img src="images/polynomial_validation_curve_1.svg" width="450" style="float:right">

+++ {"slideshow": {"slide_type": "subslide"}}

### Train vs test error: increasing complexity

<img src="images/polynomial_overfit_test_2.svg" width="450" style="float:left">
<img src="images/polynomial_validation_curve_2.svg" width="450" style="float:right">

+++ {"slideshow": {"slide_type": "subslide"}}

### Train vs test error: increasing complexity

<img src="images/polynomial_overfit_test_5.svg" width="450" style="float:left">
<img src="images/polynomial_validation_curve_5.svg" width="450" style="float:right">

+++ {"slideshow": {"slide_type": "subslide"}}

### Train vs test error: increasing complexity

<img src="images/polynomial_overfit_test_9.svg" width="450" style="float:left">
<img src="images/polynomial_validation_curve_15.svg" width="450" style="float:right">

+++ {"slideshow": {"slide_type": "subslide"}}

### Train vs Test Error: Validation Curve

<img src="images/polynomial_validation_curve_15_annotated.png" width="800">

+++ {"slideshow": {"slide_type": "slide"}}

### Train vs Test Error: Varying Sample Size

<img src="images/polynomial_overfit_ntrain_42.svg" width="400" style="float:left">
<img src="images/polynomial_learning_curve_42.svg" width="500" style="float:right">
<div style="clear:both;"></div>

<center><b>Overfit</b></center>

+++ {"slideshow": {"slide_type": "subslide"}}

### Train vs Test Error: Varying Sample Size

<img src="images/polynomial_overfit_ntrain_145.svg" width="400" style="float:left">
<img src="images/polynomial_learning_curve_145.svg" width="500" style="float:right">
<div style="clear:both;"></div>

<center><b>Overfit less</b></center>

+++ {"slideshow": {"slide_type": "subslide"}}

### Train vs Test Error: Varying Sample Size

<img src="images/polynomial_overfit_ntrain_1179.svg" width="400" style="float:left">
<img src="images/polynomial_learning_curve_1179.svg" width="500" style="float:right">
<div style="clear:both;"></div>

<center><b>Sweet spot?</b></center>

+++ {"slideshow": {"slide_type": "subslide"}}

### Train vs Test Error: Learning Curve

<img src="images/polynomial_overfit_ntrain_6766.svg" width="400" style="float:left">
<img src="images/polynomial_learning_curve_6766.svg" width="500" style="float:right">
<div style="clear:both;"></div>

<center><b>Diminishing returns &#8594; Try more complex models?</b></center>

+++ {"slideshow": {"slide_type": "subslide"}}

### Irreducible Error

<img src="images/polynomial_overfit_ntrain_6766.svg" width="400" style="float:left;margin-right:20px">

Error of best model trained on unlimited data

Here, the data-generating process is a degree-9 polynomial

A higher-degree polynomial will not do better

**Predictions limited by noise**

+++ {"slideshow": {"slide_type": "subslide"}}

### Model Families

Crucial to match:
- statistical model
- data-generating process

So far: polynomial for both

Some family names: *linear models, decision trees, random forests, kernel machines, multi-layer perceptrons*

+++ {"slideshow": {"slide_type": "subslide"}}

### Different Model Families

<img src="images/different_models_complex_4.svg" width="450" style="float:left;margin-right:50px">

- Different inductive (learning) bias
- Different notion of complexity

+++ {"slideshow": {"slide_type": "subslide"}}

### Different Model Families

<img src="images/different_models_complex_4.svg" width="450" style="float:left">
<img src="images/different_models_complex_16.svg" width="450" style="float:right">
<div style="clear:both"></div>
<div style="float:left"><b>Simple variant</b></div>
<div style="float:right"><b>Complex variant</b></div>

+++ {"slideshow": {"slide_type": "subslide"}}

### Partial Summary

Models **overfit**:
- number of samples in the training set is too small for model's complexity
- testing error is much bigger than training error

Models **underfit**:
- models fail to capture the shape of the training set
- even the training error is large

Different model families = different complexity

+++ {"slideshow": {"slide_type": "subslide"}}

## Bias-Variance Decomposition of the EPE

+++ {"slideshow": {"slide_type": "subslide"}}

## Cross-Validation

+++ {"slideshow": {"slide_type": "slide"}}

## Law of Large Numbers?

Estimates attempt to minimize a function of the training error $\overline{\mathrm{err}}$.

For estimates to converge with the sample size, so should $\overline{\mathrm{err}}$.

+++ {"slideshow": {"slide_type": "fragment"}}

$\rightarrow$ We need some **Law of Large Numbers** to be applicable.

Basic assumptions: **independent** and **identically distributed**.

+++ {"slideshow": {"slide_type": "subslide"}}

### What could go wrong?

In the natural and engineering sciences many problems depend on **time**.

So far, we have assumed that the joint distribution $f_{\boldsymbol{X}, Y}$ is **independent of time**.

In particular, we have assumed that the joint process is **statistically stationary**.

+++ {"slideshow": {"slide_type": "subslide"}}

Variations in time can rarely be considered purely random:

$\rightarrow$ some **dependence** persist between realizations

+++ {"slideshow": {"slide_type": "fragment"}}

Yet, we are fine if we can show that:
- there is a **stationary distribution**
- realizations sufficiently distant in time **no longer correlate**

+++ {"slideshow": {"slide_type": "fragment"}}

However, distributions may change with **cycles** and **trends**.

+++ {"slideshow": {"slide_type": "subslide"}}

### Violation of Statistical Stationarity

<div style="float:left;margin-right:20px">
<img src="images/640px-20200324_Global_average_temperature_-_NASA-GISS_HadCrut_NOAA_Japan_BerkeleyE.svg.png" width="600">

[By RCraig09 - Own work, CC BY-SA 4.0](https://commons.wikimedia.org/w/index.php?curid=88535596)
</div>

Surface air temperature variability can be decomposed into:

$-$ (pseudo-)periodic **cycles** (diurnal, annual, Milankovitch)

$-$ a **continuous spectrum** of frequencies due to chaotic dynamics

$-$ an increasing **trend** due to global warming

$-$ other non-equilibrium variations (effect of volcanoes, solar activity, ...)

+++ {"slideshow": {"slide_type": "subslide"}}

## Take Home Messages


-

+++ {"slideshow": {"slide_type": "slide"}}

## References

- [James, G., Witten, D., Hastie, T., Tibshirani, R., n.d. *An Introduction to Statistical Learning*, 2st ed. Springer, New York, NY.](https://www.statlearning.com/)
- Chap. 2, 3 and 7 in [Hastie, T., Tibshirani, R., Friedman, J., 2009. *The Elements of Statistical Learning*, 2nd ed. Springer, New York.](https://doi.org/10.1007/978-0-387-84858-7)
- Chap. 5 and 7 in [Wilks, D.S., 2019. *Statistical Methods in the Atmospheric Sciences*, 4th ed. Elsevier, Amsterdam.](https://doi.org/10.1016/C2017-0-03921-6)

+++ {"slideshow": {"slide_type": "slide"}}

***
## Credit

[//]: # "This notebook is part of [E4C Interdisciplinary Center - Education](https://gitlab.in2p3.fr/energy4climate/public/education)."
Contributors include Bruno Deremble and Alexis Tantet.
Several slides and images are taken from the very good [Scikit-learn course](https://inria.github.io/scikit-learn-mooc/).

<br>

<div style="display: flex; height: 70px">
    
<img alt="Logo LMD" src="images/logos/logo_lmd.jpg" style="display: inline-block"/>

<img alt="Logo IPSL" src="images/logos/logo_ipsl.png" style="display: inline-block"/>

<img alt="Logo E4C" src="images/logos/logo_e4c_final.png" style="display: inline-block"/>

<img alt="Logo EP" src="images/logos/logo_ep.png" style="display: inline-block"/>

<img alt="Logo SU" src="images/logos/logo_su.png" style="display: inline-block"/>

<img alt="Logo ENS" src="images/logos/logo_ens.jpg" style="display: inline-block"/>

<img alt="Logo CNRS" src="images/logos/logo_cnrs.png" style="display: inline-block"/>
    
</div>

<hr>

<div style="display: flex">
    <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0; margin-right: 10px" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a>
    <br>This work is licensed under a &nbsp; <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
</div>
