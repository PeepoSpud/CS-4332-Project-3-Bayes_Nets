import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32, random
import numpy as np
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """

    # TODO: finish this function
    BayesNet = BayesianModel()
    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")    
    BayesNet.add_edge("temperature","faulty gauge")
    BayesNet.add_edge("temperature","gauge")
    BayesNet.add_edge("faulty gauge","gauge")
    BayesNet.add_edge("gauge","alarm")
    BayesNet.add_edge("faulty alarm","alarm")
    #raise NotImplementedError
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    cpdTemphot = TabularCPD('temperature', 2, values=[[0.8], [0.2]])
    cpdAlarmfaul = TabularCPD('faulty alarm', 2, values=[[0.85], [0.15]])
    cpdFg_Th = TabularCPD('faulty gauge', 2, values=[[0.95, 0.2],[ 0.05, 0.8]], evidence=['temperature'], evidence_card=[2])
    cpdG_Fg_T = TabularCPD('gauge', 2, values=[[0.95, 0.2, 0.05, 0.8], \
                    [0.05, 0.8, 0.95, 0.2]], evidence=['temperature', 'faulty gauge'], evidence_card=[2, 2])
    cpdA_Fa_G = TabularCPD('alarm', 2, values=[[0.9, 0.1, 0.55, 0.45], \
                    [0.1, 0.9, 0.45, 0.55]], evidence=['faulty alarm', 'gauge'], evidence_card=[2, 2])
    bayes_net.add_cpds(cpdTemphot, cpdAlarmfaul, cpdFg_Th, cpdG_Fg_T, cpdA_Fa_G)
    #raise NotImplementedError    
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'], joint=False)
    prob = marginal_prob['alarm'].values
    alarm_prob = prob[-1]
    #print(prob)
    #raise NotImplementedError
    return alarm_prob



def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'], joint=False)
    prob = marginal_prob['gauge'].values
    # raise NotImplementedError
    gauge_prob = prob[-1]
    return gauge_prob


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'],evidence={'alarm':1,'faulty alarm':0,'faulty gauge':0}, joint=False)
    prob = conditional_prob['temperature'].values
    #raise NotImplementedError
    temp_prob = prob[-1]
    return temp_prob

# Functions to do for the mindfulness experiment
""" Variables in the experiment report:
    1. Gender: M or F
    2. Age: K (Kid) or A (Adult)
    3. Pre-Time: Total puzzle solving time in seconds in the pre-meditation learning session.
    4. Pre-Count: Total number of words found when reaching the 5 minutes limit.
    5. Active: Number of seconds in “Active” state.
    6. Neutral: Number of seconds in “Neutral” state.
    7. Calm: Number of seconds in “Calm” state.
    8. Recoveries: Number of times the Muse app loses connection to the Muse headband.
    9. Birds: Number of bird chirps heard. Every time the subject meditate well, there will be a bid chirp.
    10. Post-Time: Total puzzle solving time in seconds in the post-meditation learning session.
    11. Post-Counts: Total number of words found when reaching the 5 minutes limit.

Variables to be created in the Bayes net, all are Boolean variables:
    1. Male: Whether the subject is a male
    2. Adult: Whether the subject is an adult.
    3. Faster: Whether the subject solves the problem faster after the mindfulness exercises or not.
    4. Calm: Is the subject in a calm mind state the majority of the time?
    5. Disconnect: Whether the device has ever been disconnected in reading EEG signals.
    6. Birds: Whether the subject has entered a good meditating state.
"""
def make_mindfulness_net():
    """Create a Bayes Net representation of the mindfulness experiment. 
    Use the 6 variables listed above. (for the tests to work.)
    """
    # TODO: finish this function
    BayesNet = BayesianModel()
    BayesNet.add_node("male")
    BayesNet.add_node("adult")
    BayesNet.add_node("disconnect")
    BayesNet.add_node("faster")
    BayesNet.add_node("calm")
    BayesNet.add_node("birds")
    
    BayesNet.add_edge("adult","calm")
    BayesNet.add_edge("disconnect","calm")
    BayesNet.add_edge("male","calm")
    
    BayesNet.add_edge("calm","birds")
    BayesNet.add_edge("male","birds")
    
    BayesNet.add_edge("calm","faster")
    BayesNet.add_edge("birds","faster")
    

    
    #raise NotImplementedError
    return BayesNet


def set_mindfulness_probability(bayes_net):
    """Set probability distribution for each node in the mindfulness Bayes net.
    Use the 6 variables listed above. (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    cpdMale = TabularCPD('male', 2, values=[[0.5],[0.5]])
    
    cpdAdult = TabularCPD('adult', 2, values=[[0.75],[0.25]])
    
    cpdDisconnect = TabularCPD('disconnect', 2, values=[[0.8],[0.2]])
    
    cpdCalm = TabularCPD('calm', 2, values = [[0.1,0.7,0.4,0.75,0.3,0.8,0.5,0.9], \
                                              [0.9,0.3,0.6,0.25,0.7,0.2,0.5,0.1]],\
                         evidence = ['male','adult','disconnect'], evidence_card = [2,2,2])
        
    cpdBirds = TabularCPD('birds', 2, values=[[0.4, 0.6, 0.2, 0.3], \
                                              [0.6, 0.4 ,0.8, 0.7]],\
                          evidence = ['calm','male'], evidence_card = [2,2])
    
    cpdFaster = TabularCPD('faster', 2, values=[[0.8,0.1,0.2,0.05], \
                                               [0.2,0.9,0.8,0.95]], \
                               evidence = ['calm','birds'], evidence_card = [2,2])


    
    bayes_net.add_cpds(cpdMale, cpdAdult, cpdDisconnect, cpdCalm, cpdBirds ,cpdFaster)
        
    #raise NotImplementedError    
    return bayes_net


def get_faster_prob(bayes_net):
    """Calculate the marginal 
    probability for a subject to solve the problem faster after mindfulness exercise."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['faster'], joint=False)
    prob = marginal_prob['faster'].values
    faster_prob = prob[-1] # This is dummy value, needs to be changed.
    #print(prob)
    #raise NotImplementedError
    return faster_prob


def get_birds_prob(bayes_net):
    """Calculate the marginal
    probability for a subject to enter good meditating state durng mindfulness exercise."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['birds'], joint=False)
    prob = marginal_prob['birds'].values
    # raise NotImplementedError
    birds_prob = prob[-1] # This is dummy value, needs to be changed.
    return birds_prob


def get_male_faster_prob(bayes_net):
    """Calculate the conditional probability 
    whether a male can solve the problem faster after mindfulness."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['male'],evidence={'faster':1}, joint=False)
    prob = conditional_prob['male'].values
    #raise NotImplementedError
    male_faster_prob = prob[-1] # This is dummy value, needs to be changed.
    return male_faster_prob

def get_adult_faster_prob(bayes_net):
    """Calculate the conditional probability 
    whether an adult can solve the problem faster after mindfulness."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['adult'],evidence={'faster':1}, joint=False)
    prob = conditional_prob['adult'].values
    #raise NotImplementedError
    adult_faster_prob = prob[-1] # This is dummy value, needs to be changed.
    return adult_faster_prob

def get_female_birds_prob(bayes_net):
    """Calculate the conditional probability 
    for a female subject to enter a good meditating state, given that
    she doesn't solves the problem faster after mindfulness,
    but she is calm the majority of the time during mindfulness."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['male'],evidence={'faster':0, 'calm':1}, joint=False)
    prob = conditional_prob['male'].values
    #raise NotImplementedError
    female_birds_prob = prob[-1] # This is dummy value, needs to be changed.
    return female_birds_prob