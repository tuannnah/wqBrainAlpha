# Advanced Topics / Fast D1 Documentation

<https://api.worldquantbrain.com/tutorial-pages/fast-d1-documentation>

# What is Fast D1?

Fast D1 is a framework for developing Alphas that use data available between day 0 (D0) and day 1 (D1). By leveraging the most current overnight information - such as news, earnings announcements, analyst updates, and pre-market activity - Fast D1 Alphas can capture predictive signals that deliver superior performance compared to regular D1 models.

**Key Advantage:** Access to real-time overnight catalysts that traditional D1 Alphas miss.

![fast-d1.png](https://api.worldquantbrain.com/content/images/Mc4b-dKXuyNtpGVUgr-NpRDNJC4=/450/original/fast-d1.png)

# How Fast D1 Works

**Data Timing Comparison**

Understanding the critical difference in data availability:

| Delay | Field Type | Data Timestamp | Information Captured |
| --- | --- | --- | --- |
| 1 | snt_buzz | Yesterday's market close |  Yesterday’s End-of-day data only |
| Fast D1 | snt_buzz_fast_d1 | Today's market open | Overnight news, events, pre-market moves |
| 0 | snt_buzz | 30 minutes before today's close | Close to end of today – harder to execute |

**Fast D1 captures the overnight information gap that drives market movements at the open**

# Implementation Guide

## How to Simulate Fast D1 Alphas

**Simple 2-Step Process:**

1. **Select Delay-1** in simulation settings
2. **Use fields with \_fast\_d1 suffix**

**Important Notes**

✅ **Every Fast D1 field has a regular D1 equivalent** (just without the **\_fast\_d1** suffix)  
 ✅ **No separate delay dropdown** for Fast D1-it's the same Delay-1 setting  
 ✅ **Currently available only in the USA region**  
 ✅ **Same submission criteria** as regular D1 alphas

**Example Field Naming Convention**

Regular D1 Field: snt\_buzz

Fast D1 Field: snt\_buzz\_fast\_d1

| Dataset Id  | Dataset Category | Field Count |
| --- | --- | --- |
| fundamental3 | fundamental | 393 |
| news21 | news | 83 |
| option24 | option | 82 |
| other696 | other | 66 |
| earningscall_sentiment | other | 59 |
| model240 | model | 52 |
| model216 | model | 45 |
| model136 | model | 45 |
| other384 | other | 33 |
| news46 | news | 30 |
| earnings7 | earnings | 20 |
| other296 | other | 19 |
| news7 | news | 19 |
| other492 | other | 15 |
| news36 | news | 14 |
| broker24 | broker | 13 |
| search_interest | other | 12 |
| news104 | news | 11 |
| news13 | news | 8 |
| fundamental110 | fundamental | 7 |
| earnings1 | earnings | 6 |
| socialmedia12 | socialmedia | 6 |
| pv90 | pv | 4 |
| news59 | news | 2 |
| news35 | news | 2 |
| shortinterest30 | shortinterest | 2 |
| other10 | other | 1 |
|  | Total | 1049 |

Additionally, you must use the datafield ‘*pv98\_vwap\_15m\_delay\_minus\_1\_returns\_fast\_d1*’ for using returns value and ‘*usa\_vwap\_15m\_delay\_minus\_1\_fast\_d1’* for using the price of the stock at the FastD1 cutoff time.

You can search for the above datasets on Data Explorer and find such fields.

Besides, since all fields in Fast D1 have a suffix of ‘\_fast\_d1’ it would be easier to filter such fields using the Brain API using Python

# Best Practices & Technical Considerations

* Almost 50% of the Fast D1 fields available for use are vector datafields. So make sure to use appropriate vector operators for such fields to prevent errors during simulation.
* Try using the winsorize operator or other outlier removal techniques incase you face weight concentration errors.

# Submission Guidelines

**✅ Mandatory Performance Check**

**Submit Fast D1 Alphas ONLY if they outperform their D1 equivalents**

Example:

Alpha A: Using snt\_buzz\_fast\_d1 → Sharpe 2.8, Turnover 33%

Alpha B: Using snt\_buzz (regular D1) → Sharpe 2.6, Turnover 30%

✓ Submit Alpha A only if performance is superior

**Production Correlation**

* **Same correlation pool** as regular D1 alphas
* **Approach:** Using delta (**field\_fast\_d1 - field**) may help reduce production correlation

# Research Directions & Alpha Development

**🚀 Priority 1: Overnight Information Capture**

Focus on datasets that provide unique overnight insights:

**High Priority:**

* **News datasets** (news21, news46, news7, news36, news104)
* **Options activity** (option24)
* **Social media sentiment** (socialmedia12)
* **Earnings data** (earnings7, earnings1, earningscall\_sentiment)
* **Short interest changes** (shortinterest30)

**💡 Priority 2: Delta/Change-Based Alphas**

Capture the **overnight information change**:

*# Example: Overnight social media sentiment shift* : Change in snt\_buzz\_fast\_d1 compared to snt\_buzz

**🎯 Priority 3: Event-Driven Models**

Target specific overnight catalysts:

* Earnings announcements
* Analyst upgrades/downgrades
* News events
* Pre-market price/returns discovery
* Short interest information

**Note: You can submit high turnover Alphas (~30%) on those Alphas that capture short lived overnight dislocations. However, ensure that you maintain a Minimum Margin > 4bps in these Alphas**

**Quick Start Checklist**

* Set simulation to **Delay-1**
* Select fields with **\_fast\_d1** suffix
* Use **vector operators** for vector fields
* Use special **returns and price fields**
* Apply **winsorize** if needed
* **Compare performance** with regular D1 version
* Verify **margin > 4 bps**
* Test for **production correlation**
* Submit only if **Fast D1 version outperforms D1 – an exception to this is if you create Alphas using delta between Fast D1 field and D1 field as then you do not have comparable D1 performance.**

**Key Takeaways**

✨ **Fast D1 captures the overnight edge** that traditional D1 misses  
 ✨ **Focus on news, options, social media, and earnings** datasets  
 ✨ **Use delta approaches** (fast\_d1 - regular) for unique signals  
 ✨ **Submit only superior performers** compared to D1 equivalents  
 ✨ **Target high-quality signals** with margin > 4 bps
