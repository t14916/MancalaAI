from NeuralNetwork import *
from Game import *
import numpy as np
import random as rand


class GeneticAgent:

    def __init__(self, population_size, net_size):
        """
        population_size is the number of neural nets you start off with

        net_size is a list in which the length is the number of layers and the elements are the number
        of weights in that layer

        In this genetic algorithm, each singular weight in the neural network is considered a real valued allele.
        Concepts drawn from "Introduction to Evolutionary Computing," by A.E. Eiben and J.E. Smith.
        """
        self.population = [NeuralNetwork(net_size) for _ in range(population_size)]
        self.game = Game()
        self.net_size = net_size

    def fitness(self, nn):
        return self.game.test_against_random_agent(nn, 100)

    def compute_fitness(self, population):
        pop_performance = {}

        for i in range(len(population)):
            # print(population[i])
            win_pct = self.fitness(population[i])
            pop_performance[population[i]] = win_pct

        # return sorted(pop_performance.items(), key=lambda x: x[1], reverse=True)
        return pop_performance

    @staticmethod
    def basic_generation_selector(pop_performance, best_samples, lucky_few):
        next_generation = []

        for _ in range(best_samples):
            # Removes the element from pop_performance so it cannot be picked in the next step, and adds it to
            # next generation
            next_generation.append(pop_performance.pop(0)[0])

        for i in range(lucky_few):
            next_generation.append(random.choice(pop_performance)[0])

        random.shuffle(next_generation)
        return next_generation

    @staticmethod
    def ranked_generation_parent_selector(pop_performance, probability_mapping_function, mating_pool_size):
        return

    def tournament_generation_parent_selector(self, tournament_size, mating_pool_size, p_value):
        """
        This can be done with or without replacement. Here it is done without replacement to limit copies of the same
        nn being added to the mating_pool

        :param pop_performance:
        :param tournament_size:
        :param mating_pool_size:
        :param p_value: probability that the best memeber of a tourney is selected
        :return:
        """
        pop_performance = self.compute_fitness(self.population)
        assert(tournament_size < len(pop_performance)), "Invalid Tournament size, must be less than population size"

        mating_pool = []
        current_member = 0
        while current_member < mating_pool_size:
            # print(pop_performance.items())
            tourney_list = rand.choices(list(pop_performance.items()), k=tournament_size)
            tourney_list = sorted(tourney_list, key=lambda x: x[1])

            best_member = tourney_list.pop()[0]

            if np.random.uniform() < p_value:
                mating_pool.append(best_member)
                current_member += 1

        return mating_pool

    def mulambda_survival_selector(self, children_pool, population_size):
        assert len(children_pool) > population_size, "Not enough children to fit population specification"

        children_pool = self.compute_fitness(children_pool)

        children_pool = sorted(children_pool.items(), key=lambda x: x[1], reverse=True)

        children_pool = [x[0] for x in children_pool]
        return children_pool[:population_size]

    def recombination(self, mating_pool, population_size, mutation_probability):
        new_population = []

        for _ in range(population_size):
            # print(len(mating_pool))
            parents = random.choices(mating_pool, k=2)

            new_population.append(self.whole_arithmetic_recombination(parents[0], parents[1], 0.7))

        # print(new_population)
        for member in new_population:
            self.uniform_reset_mutator(member, mutation_probability)

        return new_population

    @staticmethod
    def uniform_reset_mutator(net, mutation_probability):
        # Note: currently uses np.random.uniform(). Remember to later change this scheme
        # to the same random used in NeuralNetwork.py
        weights = net.weights
        for layer in weights:
            for x in np.nditer(layer, op_flags=['readwrite']):
                if np.random.uniform() < mutation_probability:
                    x[...] = np.random.uniform()

    def nonuniform_creep_mutator(self):
        return

    def whole_arithmetic_recombination(self, parent1, parent2, a):
        weights1 = parent1.weights
        weights2 = parent2.weights
        child = NeuralNetwork(self.net_size)
        for layer_index in range(len(weights1)):
            layer1 = weights1[layer_index]
            layer2 = weights2[layer_index]
            child_layer = child.weights[layer_index]
            # print(layer1.shape)
            for row_index in range(layer1.shape[1]):
                for column_index in range(layer2.shape[0]):
                    x = layer1[column_index, row_index]
                    y = layer2[column_index, row_index]

                    child_layer[column_index, row_index] = x * a + (1 - a) * y

        return child
    def blend_recombination(self, parent1, parent2):
        return

    def test_fitness(self):
        pop_performance = self.compute_fitness(self.population)

        sorted_by_performance = sorted(pop_performance.items(), key = lambda x: x[1], reverse=True)

        return sorted_by_performance[:5]

    def run(self, num_generations):

        for _ in range(num_generations):
            print(_)
            mating_pool = self.tournament_generation_parent_selector(3, 30, 0.5)

            children_pool = self.recombination(mating_pool, len(self.population) * 2, 0.05)

            # print(len(children_pool))

            self.population = self.mulambda_survival_selector(children_pool, len(self.population))


# Note, the list first and last elements in the net_size list must remain the same in order for the algorithm to work
ga = GeneticAgent(100, [15, 100, 6])
print(ga.test_fitness())

ga.run(10)

print(ga.test_fitness())
