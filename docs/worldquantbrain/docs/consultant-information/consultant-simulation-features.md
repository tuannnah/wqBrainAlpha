# Consultant Information / Consultant Simulation Features

<https://api.worldquantbrain.com/tutorial-pages/multi-alpha-simulation>

# What is Multi-Simulation?

Multi-Simulation is a feature that allows for faster simulation of multiple Alphas in a single run. It's particularly useful for experimenting with multiple variations of an Alpha idea.

Consultants can execute up to 8 simultaneous Multi-Simulations. Each Multi-Simulation can contain up to 10 Alphas that run sequentially, each of the 10 Alphas with distinct operators, data fields, and settings. However, all must share the same Region and Delay setting within a Multi-Simulation.

![multi_sim_alphas.png](https://api.worldquantbrain.com/content/images/8YTF7Iv-vFYDbAKDuSQK0NWCCYE=/302/original/multi_sim_alphas.png)

## Example

Consider you have an Alpha **rank(delta(close,1))**, and you want to see how the Alpha’s performance varies when replacing **close** with **open**, **high** and **low**:

| Multi-Simulation 1 | |  | | Multi-Simulation 2 | |  | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Simulation 1.1 | Simulation 1.2 | Simulation 2.1 | Simulation 2.2 |
| rank(delta(close,1)) USA, Delay 1 | rank(delta(open,1)) USA, Delay 1 | rank(delta(high,1)) USA, Delay 0 | rank(delta(low,1)) USA, Delay 0 |

## How can you use Multi-Simulation

1. Create a new simulation by clicking “New Multi-Simulation”

![new multi sim.png](https://api.worldquantbrain.com/content/images/iegtB0P_EFgY-1d1PvHPfC0xwT4=/303/original/new_multi_sim.png)

2. Click on "+" button to create new simulation within a Multi-Simulation (you can create up to 10 windows)

![sub multi sim.png](https://api.worldquantbrain.com/content/images/59Arn8vZSD5DzOIxEUq-az8JiNE=/304/original/sub_multi_sim.png)

3. Input your simulation settings, expressions for each Alpha.  
 *Note that REGION, DELAY, LANGUAGE and INSTRUMENT TYPE need to be the same for all Alphas within a single Multiple-Simulation*

4. Click the “Multi-Simulate” icon to start the simulation after inputting all Alphas.

![start_multi_sim.png](https://api.worldquantbrain.com/content/images/kSg3rpDNCtFP5IZQLjFjtHlvMnw=/305/original/start_multi_sim.png)

5. Review simulation results by switching between Alphas for detailed comparison.

6. You can submit one or more of the Alphas from the batch.

7. Refer to [this document](https://platform.worldquantbrain.com/learn/documentation/consultant-information/brain-api#simulations:~:text=the%20SUPER%20simulation%3E%22%0A%7D-,Multiple%20simulations,-can%20be%20run) for Multi Alpha Simulation API usage.

## Tips for Success

* **Naming Alphas**: To avoid confusion, name each Alpha.
* **Resource Error**: If the error “*The simulation requires more resources than are available*” occurs, reduce the number of Alphas within a Multi-Simulation and retry.

# Test Period

![Test_period.PNG](https://api.worldquantbrain.com/content/images/-FJkDHu6xsk2Bz60GuIysfBGp_8=/286/original/Test_period.PNG)

The Test Period setting offers you the flexibility to designate a separate testing period for your Alpha. This period corresponds to the final 0-6 years of the In-Sample (IS) period, providing a distinct timeframe for assessing your Alpha's performance before submission.
