# Regions and Universes / GLB TOPDIV3000 Universe

<https://api.worldquantbrain.com/tutorial-pages/getting-started-glb-topdiv3000-universe>

# Characteristics of the GLB TOPDIV3000 Universe

The GLB TOPDIV3000 universe is designed to enhance Alpha generation with **better liquidity**, **broader coverage** (around 2,200 instruments), and **balanced global diversity**:

* **USA:** Slightly over 50%
* **Asia & Europe:** Roughly equal at ~25% each
* **Additional AMR coverage**

# Tips for Success

* Test your previous GLB Alphas on this new universe.
* Improved liquidity and broader coverage may allow Alphas that previously failed Sub-Universe tests to succeed.
* Transfer USA, EUR, and ASI Alphas to the GLB TOPDIV3000 universe for better performance.
* Use price-relative metrics (e.g., ratios) instead of raw prices for consistency across instruments.
* Apply **Double Neutralization** to improve reliability:
  + **Country Neutralization:** Remove country-specific risks by neutralizing against country mean values.
  + **Sector/Industry Neutralization:** Refine further by neutralizing sector or industry factors.
  + **Example:** new\_group = group\_cartesian\_product(country, industry)
* Select component Alphas from the **TOP3000** or **MINVOL1M** universes and combine them into **SuperAlphas** under the **GLB TOPDIV3000** universe for enhanced performance.

# More concepts that you can explore

[Global Alphas](https://platform.worldquantbrain.com/learn/documentation/regions-and-universes/global-region)

[Global SuperAlphas](https://platform.worldquantbrain.com/learn/documentation/superalpha/getting-started-global-superalphas)

[On Dealing With FX In Multi-Currency Regions](https://support.worldquantbrain.com/hc/en-us/community/posts/19381139332503--GLB-Theme-On-Dealing-With-FX-In-Multi-Currency-Regions)
