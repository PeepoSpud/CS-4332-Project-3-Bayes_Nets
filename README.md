# CS-4332-Project-3-Bayes_Nets
CS 4332 Introduction to Artificial Intelligence

Project 3 – Bayes Nets

**Overview**

In this assignment, you will work with probabilistic models known as Bayesian networks to efficiently calculate the answer to probability questions concerning discrete random variables.

**Setup**

Run the following command in the command line to install and update the required packages

pip install torch===1.4.0 torchvision===0.5.0 -f <https://download.pytorch.org/whl/torch_stable.html>

pip install --upgrade -r requirements.txt

Note that if you use Anaconda, you should use “conda install” command to install packages.

**Submission**

Submit the submission.py file in Blackboard. The deliverable for the assignment is this 'submission.py' file with the necessary functions/methods completed. No other Python source files need to be submitted.

**The Files**

While you'll only have to edit and submit **submission.py**, there are a number of notable files:

1.  **submission.py**: Where you will implement your *Mindfulness_net.*
2.  **probability_tests.py**: Sample tests to validate your network.
3.  **Requirements.txt**: Modules needed to be installed to implement and test Bayesian networks.

**The Assignment**

Your task is to implement a *Mindfulness_net* Bayesian network that will calculate the probabilities of variables, which are observed in the mindfulness exercise. There is a probability_tests.py file to help you along the way. The Bayesian network to be implemented is given.

**Part 1 Bayesian network example (A tutorial)**

To start, we show how to design a basic probabilistic model for the following system:

There's a nuclear power plant in which an alarm is supposed to ring when the gauge reading exceeds a fixed threshold. The gauge reading is based on the actual temperature, and for simplicity, we assume that the temperature is represented as either high or normal. However, the alarm is sometimes faulty. The temperature gauge can also fail, with the chance of failing greater when the temperature is high.

You will test your implementation at the end of each section.

**1a: Casting the net**

Assume that the following statements about the system are true:

-   The temperature gauge reads the correct temperature with 95% probability when it is not faulty and 20% probability when it is faulty. For simplicity, say that the gauge's "true" value corresponds with its "hot" reading and "false" with its "normal" reading, so the gauge would have a 95% chance of returning "true" when the temperature is hot and it is not faulty.
-   The alarm is faulty 15% of the time.
-   The temperature is hot (call this "true") 20% of the time.
-   When the temperature is hot, the gauge is faulty 80% of the time. Otherwise, the gauge is faulty 5% of the time.
-   The alarm responds correctly to the gauge 55% of the time when the alarm is faulty, and it responds correctly to the gauge 90% of the time when the alarm is not faulty. For instance, when it is faulty, the alarm sounds 55% of the time that the gauge is "hot" and remains silent 55% of the time that the gauge is "normal."

Use the following name attributes:

-   "alarm" node: Represents the probability that an alarm system will be going off or not.
-   "faulty alarm" node: Represents the probability that the alarm system is broken or not.
-   "gauge": Represents the probability that the gauge will show either "above the threshold" or "below the threshold" (high = True, normal = False).
-   "faulty gauge": Represents the probability that the gauge is broken.
-   "temperature": Represents the probability that the temperature is HOT or NOT HOT (high = True, normal = False)

Use the description of the model above to design a Bayesian network for this model. The pgmpy package is used to represent nodes and conditional probability arcs connecting nodes. Use the function make_power_plant_net() to create the net. See the code in submission.py.

The following commands create a BayesNet instance add node with name "alarm":

BayesNet = BayesianModel()

BayesNet.add_node("alarm")

We use BayesNet.add_edge() to connect nodes. For example, to connect the alarm and temperature nodes that you've already made (i.e. assuming that temperature affects the alarm probability), use function:

BayesNet.add_edge(\<parent node name\>,\<child node name\>)

BayesNet.add_edge("temperature","alarm")

After you have implemented make_power_plant_net(), you can run the following test in the command line to make sure your network is set up correctly.

python probability_tests.py ProbabilityTests.test_network_setup

**1b: Setting the probabilities**

Now set the conditional probabilities for the necessary variables on the network you just built.

Implement the function set_probability()

Using pgmpy's factors.discrete.TabularCPD class: if you wanted to set the distribution for node 'A' with two possible values, where P(A) to 70% true, 30% false, you would invoke the following commands:

cpd_a = TabularCPD('A', 2, values=[[0.3], [0.7]])

**NOTE: Use index 0 to represent FALSE and index 1 to represent TRUE, or you may run into testing issues.**

If you wanted to set the distribution for P(A\|G) to be

| **G** | **P(A=true given G)** |
|-------|-----------------------|
| T     | 0.75                  |
| F     | 0.85                  |

you would invoke:

cpd_ag = TabularCPD('A', 2, values=[[0.15, 0.25], \\

[0.85, 0.75]], evidence=['G'], evidence_card=[2])

Reference for the function: https://pgmpy.org/_modules/pgmpy/factors/discrete/CPD.html

Modeling a three-variable relationship is a bit trickier. If you wanted to set the following distribution for P(A\|G,T) to be

| **G** | **T** | **P(A=true given G and T)** |
|-------|-------|-----------------------------|
| T     | T     | 0.15                        |
| T     | F     | 0.6                         |
| F     | T     | 0.2                         |
| F     | F     | 0.1                         |

you would invoke

cpd_agt = TabularCPD('A', 2, values=[[0.9, 0.8, 0.4, 0.85], \\

[0.1, 0.2, 0.6, 0.15]], evidence=['G', 'T'], evidence_card=[2, 2])

The key is to remember that first entry represents the probability for P(A==False), and second entry represents P(A==true).

Add Tabular conditional probability distributions to the bayesian model instance by using the following command.

bayes_net.add_cpds(cpd_a, cpd_ag, cpd_agt)

You can check your probability distributions in the command line with

python probability_tests.py ProbabilityTests.test_probability_setup

**1c: Probability calculations : Perform inference**

To finish up, we're going to perform inference on the network to calculate the following probabilities:

-   the marginal probability that the alarm sounds
-   the marginal probability that the gauge shows "hot"
-   the probability that the temperature is actually hot, given that the alarm sounds and the alarm and gauge are both working

You'll fill out the "get_prob" functions to calculate the probabilities:

-   get_alarm_prob()
-   get_gauge_prob()
-   get_temperature_prob()

Here's an example of how to do inference for the marginal probability of the "faulty alarm" node being True (assuming bayes_net is your network):

solver = VariableElimination(bayes_net)

marginal_prob = solver.query(variables=['faulty alarm'], joint=False)

prob = marginal_prob['faulty alarm'].values

To compute the conditional probability, set the evidence variables before computing the marginal as seen below (here we're computing P('A' = false \| 'B' = true, 'C' = False)):

solver = VariableElimination(bayes_net)

conditional_prob = solver.query(variables=['temperature'], \\

evidence={'alarm':1,'faulty alarm':0,'faulty gauge':0}, joint=False)

prob = conditional_prob['temperature'].values

**NOTE**: marginal_prob and conditional_prob return two probabilities corresponding to [False, True] case. You must index into the correct position in prob to obtain the particular probability value you are looking for.

If you need to sanity-check to make sure you're doing inference correctly, you can run inference on one of the probabilities that we gave you in 1a. For instance, running inference on P(T=true) should return 0.20 (i.e. 20%). However, due to imprecision in some machines it could appear as 0.199xx. You can also calculate the answers by hand to double-check.

**Part 2: Mindfulness Net (Your task)**

Read the attached report about an experiment performed in the service-learning project sponsored by the Center for Community Engagement and Service Learning, UHD, that aimed to study the effectiveness of mindfulness contemplative practices in enhancing learning performance.

Based on the findings in the report, we want to design a Bayesian network shown in the following figure, where the variables are:

-   Male (M): Whether the subject is a male
-   Adult (A): Whether the subject is an adult.
-   Faster (F): Whether the subject solves the problem faster after the mindfulness exercises or not.
-   Calm (C): Is the subject in a calm mind state the majority of the time?
-   Disconnect (D): Whether the device has ever been disconnected in reading EEG signals.
-   Birds (B): Whether the subject has entered a good meditating state.

Enter the above Bayesian network into submission.py, complete necessary functions to compute the following probabilities:

-   The marginal probability for a subject to enter good meditating state during mindfulness exercise.
-   The marginal probability for a subject to solve the problem faster after mindfulness exercise.
-   The conditional probability whether a male can solve the problem faster after mindfulness.
-   The conditional probability whether an adult can solve the problem faster after mindfulness.
-   The conditional probability for a female subject to enter a good meditating state, given that she doesn't solves the problem faster after mindfulness, but she is calm the majority of the time during mindfulness.

Test your design and compute the interested probabilities by running the probability_tests.py file.
