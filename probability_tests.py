import unittest
from submission import *
"""
Contains various local tests for Assignment 3.
"""

class ProbabilityTests(unittest.TestCase):

    #Part 1a
    def test_network_setup(self):
        """Test that the power plant network has the proper number of nodes and edges."""
        power_plant = make_power_plant_net()
        nodes = power_plant.nodes()
        self.assertEqual(len(nodes), 5, msg="incorrect number of nodes")
        total_links = power_plant.number_of_edges()
        self.assertEqual(total_links, 5, msg="incorrect number of edges between nodes")

    #Part 1b
    def test_probability_setup(self):
        """Test that all nodes in the power plant network have proper probability distributions.
        Note that all nodes have to be named predictably for tests to run correctly."""
        # first test temperature distribution
        power_plant = set_probability(make_power_plant_net())
        T_node = power_plant.get_cpds('temperature')
        self.assertTrue(T_node is not None, msg='No temperature node initialized')

        T_dist = T_node.get_values()
        self.assertEqual(len(T_dist), 2, msg='Incorrect temperature distribution size')
        test_prob = T_dist[0]
        self.assertEqual(round(float(test_prob*100)), 80, msg='Incorrect temperature distribution')

        # then faulty gauge distribution
        F_G_node = power_plant.get_cpds('faulty gauge')
        self.assertTrue(F_G_node is not None, msg='No faulty gauge node initialized')

        F_G_dist = F_G_node.get_values()
        rows, cols = F_G_dist.shape
        self.assertEqual(rows, 2, msg='Incorrect faulty gauge distribution size')
        self.assertEqual(cols, 2, msg='Incorrect faulty gauge distribution size')
        test_prob1 = F_G_dist[0][1]
        test_prob2 = F_G_dist[1][0]
        self.assertEqual(round(float(test_prob2*100)), 5, msg='Incorrect faulty gauge distribution')
        self.assertEqual(round(float(test_prob1*100)), 20, msg='Incorrect faulty gauge distribution')

        # faulty alarm distribution
        F_A_node = power_plant.get_cpds('faulty alarm')
        self.assertTrue(F_A_node is not None, msg='No faulty alarm node initialized')
        F_A_dist = F_A_node.get_values()
        self.assertEqual(len(F_A_dist), 2, msg='Incorrect faulty alarm distribution size')

        test_prob = F_A_dist[0]

        self.assertEqual(round(float(test_prob*100)), 85, msg='Incorrect faulty alarm distribution')
        # gauge distribution
        # can't test exact probabilities because
        # order of probabilities is not guaranteed
        G_node = power_plant.get_cpds('gauge')
        self.assertTrue(G_node is not None, msg='No gauge node initialized')
        [cols, rows1, rows2] = G_node.cardinality
        self.assertEqual(rows1, 2, msg='Incorrect gauge distribution size')
        self.assertEqual(rows2, 2, msg='Incorrect gauge distribution size')
        self.assertEqual(cols,  2, msg='Incorrect gauge distribution size')

        # alarm distribution
        A_node = power_plant.get_cpds('alarm')
        self.assertTrue(A_node is not None, msg='No alarm node initialized')
        [cols, rows1, rows2] = A_node.cardinality
        self.assertEqual(rows1, 2, msg='Incorrect alarm distribution size')
        self.assertEqual(rows2, 2, msg='Incorrect alarm distribution size')
        self.assertEqual(cols,  2, msg='Incorrect alarm distribution size')

        try:
            power_plant.check_model()
        except:
            self.assertTrue(False, msg='Sum of the probabilities for each state is not equal to 1 or CPDs associated with nodes are not consistent with their parents')

    #Part 1c
    def test_probability_calculatios(self):
        """perform inference on the network to calculate the followingprobabilities:
            the marginal probability that the alarm sounds
            the marginal probability that the gauge shows "hot"
            the probability that the temperature is actually hot, given that the alarm soundsand the alarm and gauge are both working."""
        power_plant = set_probability(make_power_plant_net())
        solver = VariableElimination(power_plant)
        marginal_prob = solver.query(variables=['faulty alarm'], joint=False)
        prob = marginal_prob['faulty alarm'].values
        print("Faulty alarm probability is: ", prob)

        alarm_prob = get_alarm_prob(power_plant)
        gauge_prob = get_gauge_prob(power_plant)
        temperature_prob = get_temperature_prob(power_plant)
        print("Alarm, gauge, and temperature probabilities are: ", 
             alarm_prob, "   ", gauge_prob, "   ", temperature_prob)


class ProbabilityTests2(unittest.TestCase):

    #Part 2a
    def test_network_setup(self):
        """Test that the mindfulness network has the proper number of nodes and edges."""
        mindfulness_net = make_mindfulness_net()
        nodes = mindfulness_net.nodes()
        self.assertEqual(len(nodes), 6, msg="incorrect number of nodes")
        total_links = mindfulness_net.number_of_edges()
        self.assertEqual(total_links, 7, msg="incorrect number of edges between nodes")

    #Part 2b
    def test_probability_setup(self):
        """Test that all nodes in the mindfulness network have proper probability distributions.
        Note that all nodes have to be named predictably for tests to run correctly."""
        # first test temperature distribution
        mindfulness_net = set_mindfulness_probability(make_mindfulness_net())
        M_node = mindfulness_net.get_cpds('male')
        self.assertTrue(M_node is not None, msg='No Male node initialized')

        M_dist = M_node.get_values()
        self.assertEqual(len(M_dist), 2, msg='Incorrect Male distribution size')
        test_prob = M_dist[0]
        self.assertEqual(round(float(test_prob*100)), 50, msg='Incorrect temperature distribution')

        # test Adult distribution
        A_node = mindfulness_net.get_cpds('adult')
        self.assertTrue(A_node is not None, msg='No Adult node initialized')

        A_dist = A_node.get_values()
        self.assertEqual(len(A_dist), 2, msg='Incorrect Adult distribution size')
        test_prob = A_dist[0]
        self.assertEqual(round(float(test_prob*100)), 75, msg='Incorrect Adult distribution')

        # test Disconnect distribution
        D_node = mindfulness_net.get_cpds('disconnect')
        self.assertTrue(D_node is not None, msg='No Disconnect node initialized')

        D_dist = D_node.get_values()
        self.assertEqual(len(D_dist), 2, msg='Incorrect Disconnect distribution size')
        test_prob = D_dist[0]
        self.assertEqual(round(float(test_prob*100)), 80, msg='Incorrect Adult distribution')
        
        # then Calm distribution
        C_node = mindfulness_net.get_cpds('calm')
        self.assertTrue(C_node is not None, msg='No Calm node initialized')

        C_dist = C_node.get_values()
        #print('C_dist shape: ', C_dist.shape)
        #print('C_dist:', C_dist)
        cols, rows = C_dist.shape
        self.assertEqual(rows, 8, msg='Incorrect Calm distribution size')
        self.assertEqual(cols, 2, msg='Incorrect Calm distribution size')
        #self.assertEqual(lays, 2, msg='Incorrect Calm distribution size')
        test_probmad = C_dist[1][0]
        test_probmaD = C_dist[1][1]
        test_probmAd = C_dist[1][2]
        test_probmAD = C_dist[1][3]
        test_probMad = C_dist[1][4]
        test_probMaD = C_dist[1][5]
        test_probMAd = C_dist[1][6]
        test_probMAD = C_dist[1][7]
        #print('MAD C = ', test_prob010, test_prob100, test_prob011, test_prob101)
        self.assertEqual(round(float(test_probmad*100)), 90, msg='Incorrect Calm distribution')
        self.assertEqual(round(float(test_probmaD*100)), 30, msg='Incorrect Calm distribution')
        self.assertEqual(round(float(test_probmAd*100)), 60, msg='Incorrect Calm distribution')
        self.assertEqual(round(float(test_probmAD*100)), 25, msg='Incorrect Calm distribution')
        self.assertEqual(round(float(test_probMad*100)), 70, msg='Incorrect Calm distribution')
        self.assertEqual(round(float(test_probMaD*100)), 20, msg='Incorrect Calm distribution')
        self.assertEqual(round(float(test_probMAd*100)), 50, msg='Incorrect Calm distribution')
        self.assertEqual(round(float(test_probMAD*100)), 10, msg='Incorrect Calm distribution')
        
        # Birds distribution
        B_node = mindfulness_net.get_cpds('birds')
        self.assertTrue(B_node is not None, msg='No Birds node initialized')
        B_dist = B_node.get_values()
        cols, rows = B_dist.shape
        # ERROR BY WRITER: THERE SHOULD BE 4 ROWS!!
        self.assertEqual(rows, 4, msg='Incorrect Birds distribution size')
        self.assertEqual(cols, 2, msg='Incorrect Birds distribution size')
        test_probcm = B_dist[1][0]
        test_probcM = B_dist[1][1]
        test_probCm = B_dist[1][2]
        test_probCM = B_dist[1][3]
        # ERROR BY WRITER: PLEASE WRITE THE CORRECT VARIABLE YOU ARE COMPARING BELOW
        self.assertEqual(round(float(test_probcm*100)), 60, msg='Incorrect Birds distribution')
        self.assertEqual(round(float(test_probcM*100)), 40, msg='Incorrect Birds distribution')
        self.assertEqual(round(float(test_probCm*100)), 80, msg='Incorrect Birds distribution')
        self.assertEqual(round(float(test_probCM*100)), 70, msg='Incorrect Birds distribution')

        # Faster distribution
        F_node = mindfulness_net.get_cpds('faster')
        self.assertTrue(F_node is not None, msg='No Faster node initialized')
        F_dist = F_node.get_values()
        cols, rows = B_dist.shape
        # ERROR BY WRITER: THERE SHOULD BE 4 ROWS!!
        self.assertEqual(rows, 4, msg='Incorrect Faster distribution size')
        self.assertEqual(cols, 2, msg='Incorrect Faster distribution size')
        test_probcb = F_dist[1][0]
        test_probcB = F_dist[1][1]
        test_probCb = F_dist[1][2]
        test_probCB = F_dist[1][3]
        # ERROR BY WRITER: PLEASE WRITE THE CORRECT VARIABLE YOU ARE COMPARING BELOW
        self.assertEqual(round(float(test_probcb*100)), 20, msg='Incorrect Faster distribution')
        self.assertEqual(round(float(test_probcB*100)), 90, msg='Incorrect Faster distribution')
        self.assertEqual(round(float(test_probCb*100)), 80, msg='Incorrect Faster distribution')
        self.assertEqual(round(float(test_probCB*100)), 95, msg='Incorrect Faster distribution')

        
        try:
            mindfulness_net.check_model()
        except:
            self.assertTrue(False, msg='Sum of the probabilities for each state is not equal to 1 or CPDs associated with nodes are not consistent with their parents')

    
    #Part 2c
    def test_probability_calculatios(self):
        """perform inference on the network to calculate the following probabilities:
            •	The marginal probability for a subject to enter good meditating state during mindfulness exercise.
            •	The marginal probability for a subject to solve the problem faster after mindfulness exercise.
            •	The conditional probability whether a male can solve the problem faster after mindfulness.
            •	The conditional probability whether an adult can solve the problem faster after mindfulness.
            •	The conditional probability for a female subject to enter a good meditating state, 
                given that she doesn't solves the problem faster after mindfulness, 
                but she is calm the majority of the time during mindfulness. 
        """
        mindfulness_net = set_mindfulness_probability(make_mindfulness_net())

        birds_prob = get_birds_prob(mindfulness_net)
        male_faster_prob = get_male_faster_prob(mindfulness_net)
        adult_faster_prob = get_adult_faster_prob(mindfulness_net)
        female_birds_prob = get_female_birds_prob(mindfulness_net)
        print("Birds probability is: ", birds_prob)
        print("Male Faster probability is: ", male_faster_prob)
        print("Adult Faster probability is: ", adult_faster_prob)
        # ERROR BY WRITER: PLEASE WRITE THE CORRECT VARIABLE YOU ARE TRYING TO PRINT
        print("Female Birds probability is: ", female_birds_prob)

if __name__ == '__main__':
    unittest.main()
    
