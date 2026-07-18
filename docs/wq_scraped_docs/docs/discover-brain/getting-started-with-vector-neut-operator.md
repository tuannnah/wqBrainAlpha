# Discover BRAIN / Getting Started with vector_neut Operator

<https://api.worldquantbrain.com/tutorial-pages/getting-started-vector_neut-operator>

**Getting Started with vector\_neut Operator**

The **vector\_neut** operator orthogonalizes one vector with respect to another. When vector x is "orthogonal" to vector y, they have zero dot product—they are perpendicular in vector space. The **vector\_neut** operator transforms a vector x into a new vector x\* such that x\* is orthogonal to a specified vector y while preserving as much of the original information in x as possible.

**Important Mathematical Properties**

1. **Orthogonality**: x\* · y = 0 (zero dot product)
2. **Variance decomposition**: Var(x) = Var(x\*) + Var(projection)
3. **Minimal norm**: x\* has the minimum Euclidean distance to x among all vectors orthogonal to y

With these orthogonalization properties, **vector\_neut** is useful for removing or controlling factor exposures in your Alpha, which can help reduce unwanted volatility and improve Alpha's Sharpe ratio

**Example Usage**

* Let's start with the most common risk factor – market beta.
* Step 1: Approximate market returns by taking the average of individual stock returns.
  + market\_returns = group\_mean(returns, 1, market);
* Step 2a: Calculate Beta using time series regression between stock and market returns. The parameter rettype=2 specifies that we want the slope coefficient (β).
  + beta = ts\_regression(returns, market\_returns, 252, rettype=2);
* Step 2b: An alternative Beta calculation using covariance and variance.
  + beta = ts\_covariance(returns, market\_returns, 252) / power(ts\_std\_dev(market\_returns, 252), 2);
* Step 3: Neutralize your Alpha against Beta.
  + alpha\* = vector\_neut(alpha, beta);
* The resulting Alpha\* is orthogonal to Beta, meaning dot(alpha\*, beta) = 0, which implies the neutralized alpha has no linear relationship with market beta.
