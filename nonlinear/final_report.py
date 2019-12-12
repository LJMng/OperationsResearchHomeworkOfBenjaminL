'''
Created on 2019年12月6日

This .py file includes three experiments for attempts to solve the given maximize problem:
    max f(x)=x1(625-ax1)+ x2(300-bx2)+ x3(100-cx3)
    s.t. :
    1) 70-x1<=0             =>  x1-70             >=0
    2) x1-90<=0             =>  90-x1             >=0
    3) 90-x2<=0             =>  x2-90             >=0
    4) x2-110<=0            =>  110-x2            >=0
    5) 120-x3<=0            =>  x3-120            >=0
    6) x3-149<=0            =>  149-x3            >=0
    7) 575-ax1-bx2-cx3<=0   =>  ax1+bx2+cx3-575   >=0
    where a = 4.41176, b = 2.04081633, c = 0.35971223
    
As shown in "main", executions include:
1. Exhaustion search;
2. Evolutionary Algorithm -- Genetic Algorithm;
3. One using scipy.optimize.minimize.

Results are wrapped in the class SearchOutput.

@author: Benjamin_L
'''

import copy
import math
import random
import numpy as np

from scipy.optimize import minimize

from utils.time_utils import Timer

A = 4.41176
B = 2.04081633 
C = 0.35971223

RANGE_STD = [70, 90]
RANGE_GLD = [90, 110]
RANGE_PLT = [120, 149]


class SearchOutput(object):
    """ A class to contain the output of the search. """
    def __init__(self, search_name, used_time, result):
        self.search_name = search_name
        self.used_time = used_time
        self.result = result
        
    def get_search_name(self):
        return self.search_name
    
    def get_used_time(self):
        return self.used_time
    
    def get_result(self):
        return self.result

    def to_string(self):
        return '%s| used time(sec): %-8.2f| result: %s' % (self.search_name, self.get_used_time(), self.get_result())
        

class ExhaustionSearch(object):
    
    @staticmethod
    def exec_full_1(max_func, constraints, x_step=1):
        # ---------------------------------------------------------------------------------------------
        # Time complexity: O(n^3)
        # ---------------------------------------------------------------------------------------------
        timer = Timer()
        timer.start()
        # ---------------------------------------------------------------------------------------------
        result, x = None, [None for _ in range(3)]
        for x1 in list(np.arange(RANGE_STD[0], RANGE_STD[1]+x_step, x_step)):
            x[0] = x1
            for x2 in list(np.arange(RANGE_GLD[0], RANGE_GLD[1]+x_step, x_step)):
                x[1] = x2
                for x3 in list(np.arange(RANGE_PLT[0], RANGE_PLT[1]+x_step, x_step)):
                    x[2] = x3
                    if constraints(x):
                        cal_value = max_func(x)
                        result = [cal_value, copy.deepcopy(x)] if result is None or result[0] < cal_value \
                                                                else result
        # ---------------------------------------------------------------------------------------------
        time = timer.stop()
        # ---------------------------------------------------------------------------------------------
        return SearchOutput('ExhaustionSearch.exec_full_1()', time, result)

    @staticmethod
    def step_exp(max_func, constraints, run_times=1):
        results = []
        step = 10
        for _ in range(3):
            print('ExhaustionSearch.step_exp | step=%.4f' % (step))
            avg_run_time = 0
            for t in range(run_times):
                print('    ', 'Run time %d' % t)
                result = ExhaustionSearch.exec_full_1(max_func, constraints, step)
                avg_run_time += result.get_used_time()
                print('        ', 'Used: %.4f sec(s)' % result.get_used_time())
            results.append(
                SearchOutput('ExhaustionSearch.step_exp(), step=%.2f'%step, 
                             avg_run_time/run_times,
                             result.get_result()
                )
            )
            step = step / 10.0
        return results


class GeneticAlgorithmParameters(object):
    def __init__(self, fitness_func, gene_ranges, population=100, chromosome_preserve_rate=0.5, 
                 cross_probability=0.5, mumate_rate=0.01, iteration=100, convergence=30):
        self.fitness_func = fitness_func
        self.gene_ranges = gene_ranges
        self.gene_length = len(gene_ranges)
        self.population = population
        self.chromosome_preserve_rate = chromosome_preserve_rate
        self.cross_probability = cross_probability
        self.mutate_rate = mumate_rate
        self.iteration = iteration
        self.convergence = convergence
    
    def calculate_fitness(self, param):
        return self.fitness_func(param)
    
    def get_gene_length(self):
        return self.gene_length
    
    def get_gene_range_of(self, index):
        return self.gene_ranges[index]
    
    def get_population(self):
        return self.population
    
    def get_chromosome_preserve_rate(self):
        return self.chromosome_preserve_rate
    
    def get_cross_probability(self):
        return self.cross_probability
    
    def get_mutate_rate(self):
        return self.mutate_rate
    
    def get_iteration(self):
        return self.iteration
    
    def get_convergence(self):
        return self.convergence
        
class GeneticAlgorithmSearch(object):
    
    def __init__(self, ga_params, deviation = 10e-10, print_iter_info=True):
        self.ga_params = ga_params
        self.chromosome_preserve_num = min([
            math.floor(ga_params.get_population() * ga_params.get_chromosome_preserve_rate()),
            ga_params.get_population()
        ])
        self.deviation = deviation
        self.print_iter_info = print_iter_info
        if print_iter_info: print('preserve number:', self.chromosome_preserve_num)
        
    def search(self):
        # ---------------------------------------------------------------------------------------------
        timer = Timer()
        timer.start()
        # ---------------------------------------------------------------------------------------------
        # Initiate chromsomes
        chromosome = [self.new_chromosome(self.ga_params) for _ in range(self.ga_params.get_population())]
        # Initiate
        conv, best_chromosomes, best_fitness = 0, {}, None
        for iter in range(self.ga_params.get_iteration()):
            if self.print_iter_info:
                print('Iteration %5d | Convergence = %3d | best fitness = %.12f' % 
                      (iter+1, conv, best_fitness if iter>0 else 0.0)
                )
            fitness = [self.ga_params.calculate_fitness(c) for c in chromosome]
            # calculate fitness
            chromosome_fitness = {str(chromosome[i]): fitness[i] for i in range(self.ga_params.get_population())}
            chromosome.sort(key=lambda c: chromosome_fitness[str(c)], reverse=True)
            # update global best
            updated = False
            for c in chromosome:
                fitness = chromosome_fitness[str(c)]
                if len(best_chromosomes)==0:
                    # update best fitness
                    best_chromosomes[str(c)]={'chromosome': copy.deepcopy(c), 'fitness': fitness}
                    best_fitness = fitness
                    updated = True
                else:
                    cmp = GeneticAlgorithmSearch.cmp_chr_fitness(
                                fitness, best_fitness, self.deviation
                            ) 
                    if cmp>0:
                        # update best fitness
                        best_chromosomes={str(c):{'chromosome': copy.deepcopy(c), 'fitness': fitness}}
                        best_fitness = fitness
                        updated = True
                    elif cmp==0:
                        # append best fitness
                        if str(c) not in best_chromosomes.keys():
                            best_chromosomes[str(c)]={'chromosome': copy.deepcopy(c), 'fitness': fitness}
                    else:
                        pass
            # update convergence status.
            if updated: conv = 0
            else:       conv = conv+1
            if conv>self.ga_params.get_convergence():  break
            # cross-over
            GeneticAlgorithmSearch.crossover_genes(chromosome, self.ga_params, self.chromosome_preserve_num)
            # generate new chromosomes
            for i in range(self.chromosome_preserve_num+1, self.ga_params.get_population()):
                chromosome[i] = GeneticAlgorithmSearch.new_chromosome(self.ga_params)
            for i in range(self.ga_params.get_population()):
                # mumate
                GeneticAlgorithmSearch.mumate(chromosome[i], self.ga_params)
        # ---------------------------------------------------------------------------------------------
        time = timer.stop()
        # ---------------------------------------------------------------------------------------------
        return SearchOutput('GeneticAlgorithm.search()', time, best_chromosomes)
    
    @staticmethod
    def cmp_chr_fitness(chr_fitness, global_best_fitness, deviation):
        cmp = chr_fitness - global_best_fitness
        if cmp > deviation:         return cmp
        elif abs(cmp)<=deviation:   return 0
        else:                       return cmp
        
    @staticmethod
    def random_double():
        return random.randint(0, 1000) / 1000.0
    
    @staticmethod
    def calculate_gene_value(gene_range, percent):
        min_v, max_v = gene_range
        return percent * (max_v-min_v) + min_v
        
    @staticmethod
    def new_chromosome(ga_params):
        genes = np.zeros(ga_params.get_gene_length())
        for i in range(ga_params.get_gene_length()):
            genes[i] = GeneticAlgorithmSearch.calculate_gene_value(
                        ga_params.get_gene_range_of(i), GeneticAlgorithmSearch.random_double()
                    )
        return genes
    
    @staticmethod
    def crossover_genes(genes, ga_params, preserve_num):
        for i in range(0, preserve_num, 2):
            for g_i in range(ga_params.get_gene_length()):
                if random.random() <= ga_params.get_cross_probability():
                    # perform gene cross-over
                    tmp = genes[i][g_i]
                    genes[i][g_i] = genes[i+1][g_i]
                    genes[i+1][g_i] = tmp

    @staticmethod
    def mumate(gene, ga_params):
        for i in range(len(gene)):
            if random.random()<=ga_params.get_mutate_rate():
                gene[i] = GeneticAlgorithmSearch.calculate_gene_value(
                            ga_params.get_gene_range_of(i), 
                            GeneticAlgorithmSearch.random_double()
                        )

def genetic_algorithm_exp(genetic_params, run_times=10):
    results = []
    avg_run_time = 0
    avg_fx_value = 0
    print('GeneticAlgorithmSearch.search')
    for t in range(run_times):
        print('    ', 'Run time %d' % t)
        genetic_alg = GeneticAlgorithmSearch(genetic_params, deviation=0.001, print_iter_info=False)
        result = genetic_alg.search()
        avg_run_time+=result.get_used_time()
        avg_chr_fx_values = [chr['fitness'] for chr in result.get_result().values()]
        avg_fx_value+=np.average(avg_chr_fx_values)
        print('        ', 'Used: %.4f sec(s)' % result.get_used_time())
        results.append(result)
    return results, avg_run_time/run_times, avg_fx_value/run_times


def scipy_minimize():
    # ---------------------------------------------------------------------------------------------
    timer = Timer()
    timer.start()
    # ---------------------------------------------------------------------------------------------
    # suppose a=4.41176, b=2.04081633, c=0.35971223:
    #     max f(x)=x1(625-ax1)+ x2(300-bx2)+ x3(100-cx3)
    #        ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
    #     min f(x)=-(x1(625-ax1)+ x2(300-bx2)+ x3(100-cx3))
    min_func = lambda x: -x[0]*(625-A*x[0]) - x[1]*(300-B*x[1]) - x[2]*(100-C*x[2])
    # s.t.
    # 1) x1^2-x2^2+x3^3<=10
    # 2) x1^3+x2^2+4x3^3>=20
    # s.t.
    scipy_cons = [
        # 1) 70-x1<=0 => x1-70  >=0
        # 2) x1-90<=0 => 90-x1 >=0
        {'type': 'ineq', 'fun': lambda x:  x[0]-70},
        {'type': 'ineq', 'fun': lambda x:  90-x[0]},
        # 3) 90-x2<=0 => x2-90  >=0
        # 4) x2-110<=0 => 110-x2  >=0
        {'type': 'ineq', 'fun': lambda x:  x[1]-90},
        {'type': 'ineq', 'fun': lambda x:  110-x[1]},
        # 5) 120-x3<=0 => x3-120  >=0
        # 6) x3-149<=0 => 149-x3  >=0
        {'type': 'ineq', 'fun': lambda x:  x[2]-120},
        {'type': 'ineq', 'fun': lambda x:  149-x[2]},
        # 7) 575-ax1-bx2-cx3<=0 => ax1+bx2+cx3-575  >=0
        {'type': 'ineq', 'fun': lambda x:  A*x[0]+B*x[1]+C*x[2]-575},
    ]
    scipy_init_x = [0, 0, 0]
    print('init x:', scipy_init_x)
    result = minimize(min_func, scipy_init_x, constraints=scipy_cons)
    # ---------------------------------------------------------------------------------------------
    time = timer.stop()
    # ---------------------------------------------------------------------------------------------
    return SearchOutput('scipy_minimize()', time, result)
    

if __name__ == '__main__':
    # Exhaustion search model settings: 
    # max f(x)=x1(625-ax1)+ x2(300-bx2)+ x3(100-cx3)
    max_func = lambda x: x[0]*(625-A*x[0]) + x[1]*(300-B*x[1]) + x[2]*(100-C*x[2])
    # s.t. :
    # 1) 70-x1<=0 => x1-70 >=0
    # 2) x1-90<=0 => 90-x1 >=0
    # 3) 90-x2<=0 => x2-90  >=0
    # 4) x2-110<=0 => 110-x2  >=0
    # 5) 120-x3<=0 => x3-120  >=0
    # 6) x3-149<=0 => 149-x3  >=0
    # 7) 575-ax1-bx2-cx3<=0 => ax1+bx2+cx3-575  >=0
    st = [
        lambda x: x[0]-70 >=0, lambda x: 90-x[0] >=0,
        lambda x: x[1]-90 >=0, lambda x: 110-x[1] >=0,
        lambda x: x[2]-120 >=0, lambda x: 149-x[2] >=0,
        lambda x: A*x[0]+B*x[1]+C*x[2]-575  >=0
    ]
    constraints = lambda x: np.array([st_i(x) for st_i in st]).all()
    
#     result1 = ExhaustionSearch.exec_full_1(max_func, constraints)
#     print(result1.to_string())
    result1s = ExhaustionSearch.step_exp(max_func, constraints, run_times=5)
    for r in result1s:  print(r.to_string())
     
    print('#'*100)
    
    fitness_func = lambda x: max_func(x) if st[-1](x) else A*x[0]+B*x[1]+C*x[2]-575
    genetic_params = GeneticAlgorithmParameters(
                        fitness_func, [RANGE_STD, RANGE_GLD, RANGE_PLT],
                        population=500, chromosome_preserve_rate=0.50, mumate_rate=0.05,
                        iteration=600, convergence=30
                    )
#     genetic_alg = GeneticAlgorithmSearch(genetic_params, deviation=0.001, print_iter_info=False)
#     result2 = genetic_alg.search()
#     print(result2.to_string())
    result2s, avg_run_time, avg_fx_value = genetic_algorithm_exp(genetic_params, run_times=10)
    for r in result2s:  print(r.to_string())
    print('Average run time(sec): %.4f | Average fitness: %.4f' % (avg_run_time, avg_fx_value))
    print('#'*100)
    
    result3 = scipy_minimize()
    print(result3.to_string())