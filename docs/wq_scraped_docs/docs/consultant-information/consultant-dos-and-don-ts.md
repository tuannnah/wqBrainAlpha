# Consultant Information / Consultant Dos and Don'ts

<https://api.worldquantbrain.com/tutorial-pages/consultant-dos-and-donts>

# Standard Operating Guidelines

**Dos**

1. DO keep your personal information actual and updated always.
2. DO recertify your [BRAIN account](https://platform.worldquantbrain.com/profile/account/basic-details) every three months; keep your employment details and country of residence up-to-date.
3. DO consider flagging behavior you would deem inappropriate under the [Terms and Conditions](https://platform.worldquantbrain.com/profile/agreements) of BRAIN and this document. Please contact us at support@worldquantbrain.com . We protect the confidentiality of those who report to us in good faith. We verify all the reports at the backend and take appropriate action in the genuine cases.
4. DO read the [FAQs](https://support.worldquantbrain.com/hc/en-us/categories/4413011872791-General-FAQs) before raising a ticket or reaching out to support@worldquantbrain.com.
5. DO follow the requirements of the BRAIN API when utilizing scripts. Ensure the scripts deployed do not lay excessive load on the server.

**Don’ts**

1. DO NOT impersonate others or provide inaccurate information.
2. DO NOT share or disclose any User Codes [digital certificate(s), unique identifiers, user name(s) and/or password(s)] with any other person.
3. DO NOT share your Alpha code with any other person. You are encouraged to invite other consultants to form official teams, if you discuss ideas together. You are advised to discuss Alpha ideas on the [community page](https://support.worldquantbrain.com/hc/en-us/community/topics) and in Advisory sessions.
4. DO NOT disclose your Alpha details or payout information to any other person or post on any public forum or social media.
5. DO NOT engage in spidering, "screen scraping", "database scraping", or any other automatic or unauthorized means of accessing, logging-in or registering on BRAIN.
6. DO NOT use BRAIN in any manner that could interrupt, damage, disable, overburden or impair BRAIN or interfere with any other party's use and enjoyment of BRAIN
7. DO NOT distribute, publish, and exploit BRAIN or BRAIN Elements unless you have received our express written prior permission.
8. DO NOT act as representative of BRAIN or WorldQuant without approval from WorldQuant LLC.
9. DO NOT violate the above document or the terms included in the [Terms and Conditions](https://platform.worldquantbrain.com/profile/agreements).

Failure to follow the above rules would result in severe consequences including termination and suspension of accounts.

For research best practices, please refer to Recommended Practices for Alpha Research below.

# Recommended Practices for Alpha Research

**Dos**

1. DO make sure your Alpha has economic foundation. A good practice would be to check if you could explain your Alpha to someone within a minute.
2. DO simplify the implementation of an Alpha idea and remove unnecessary elements: parts that do not make sense or do not drive performance.
3. DO assign meaningful names to the variables in your Alpha code to ensure readability.
4. DO follow theme in your Alpha research to create Alphas for higher payout.
5. DO use different datasets & operators to build Alphas to ensure diversification
6. DO search new/different/unique idea, from different papers, blogs, and articles. For example, [papers on Quant research](https://platform.worldquantbrain.com/learn/documentation/learn-financial-concepts/papers).
7. DO run simulations with different sub-universe (TOP 500/1000/2000/3000) while tuning the Alpha parameters. This reduces the risk of universe fitting.
8. DO restrict parameter search to simple & reasonable ones. For example 5, 20, 60, 120, 252 in case of days, instead of 37, 14 etc.
9. DO check your Alpha’s performance stability across sub-universe, super-universe and across other regions for higher OS performance.
10. DO improve Alphas by refining ideas, not by adding or fitting parameters, factors or reversion elements.
11. DO ensure coverage by [backfilling](https://support.worldquantbrain.com/hc/en-us/articles/4902349883927-Click-here-for-a-list-of-terms-and-their-definitions#:~:text=B-,Backfill,-Replace%20missing%20values) data fields and the final Alpha using operators such as ts\_backfill, group\_backfill, group\_extra.
12. DO use exit triggers (e.g. stop-loss) to close position while using *trade\_when* operator.
13. DO normalize data fields to remove any firm related bias for cross-sectional comparison.
14. DO use automation for cutting down the redundant task in your Alpha making process
15. DO ensure that Alphas make financial sense after automation.

**Don’ts**

1. DO NOT spend too much time on a single simple idea.
2. DO NOT incorporate noise in the Alpha to reduce production correlation.
3. DO NOT add an existing Alpha from one data source to enhance the signal from another data source without any strong economic sense.
4. DO NOT over fit your Alpha to enhance performance or to reduce correlation by
   1. Trying many different parameters or using too many data fields
   2. Using too many if else
   3. Shortening IS period
   4. Filtering or presetting signals
5. DO NOT use automation as a source for generating Alpha ideas.
