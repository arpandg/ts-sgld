# ts-sgld

Code for paper "Bayesian Collaborative Bandits with Thompson Sampling for Improved Outreach in Maternal Health"

Deciding when to make automated voice calls in a maternal health program is an important problem to address for non-profits. The problem has been previously formulated as a collaborative multi-armed bandit problem. Previous solutions used a heuristic combination of matrix completion primitives with Boltzmann exploration. In this work, we propose a principled Thompson Sampling solution to the collaborative multi-armed bandit problem which is effective at utilizing prior information to converge faster to a solution. Our Bayesian inference routine uses alternating Gibbs sampling to track priors on matrix factors whose product gives the reward matrix. We test the effectiveness of this method in comparison to other methods on both simulated and real datasets. For the maternal health application, we show that this method effectively leads to a 16% reduction in number of calls made compared to matrix completion baselines, and a 58% reduction compared to the current random baseline on 49% for medium pick-up rate users. We also use synthetic simulations to show that our proposal excels in low data scenarios and also utilizes priors more effectively. We also provide theoretical analysis of the algorithm on clustered version of the problem through Eluder dimension based analysis.

### Running the code

The code runs and provides a single day of predictions on providing data in the form ```[(user_0, time_slot_0, pickup_0), ..., (user_i, time_slot_i, pickup_i)]``` where user and time slot ids are zero indexed, and pickup_i represents if the call was picked up or not.
The number of users and time slots must also be mentioned. Then just running `python3 ts_sgld.py` will suffice.
