'''
Created on 2019年11月5日

Implementation of solving transports from Producers to Sellers problem.

@author: Benjamin_L
'''

import copy
import numpy as np


class TransportSetting2D(object):
    
    def __init__(self, seller_numbers, producer_numbers):
        """
        Parameter
        ---------
            seller_number: int array
                The numbers of sellers who accept products from producer.
            producer_number: int array
                The numbers of producers who produce products.
        """
        self.cost = np.zeros((len(producer_number), len(seller_number)))
        self.producer_numbers = producer_numbers
        self.seller_numbers = seller_numbers
        
    def set_cost(self, producer_index, seller_index, value):
        self.cost[producer_index, seller_index] = value
    
    def get_cost_matrix(self):
        return self.cost
    
    def get_produce_number(self, index):
        return self.producer_numbers[index]
    
    def get_seller_number(self, index):
        return self.seller_numbers[index]
    
    def get_setting_table(self):
        table = np.zeros((len(self.producer_numbers)+1, len(self.seller_numbers)+1))
        for i in range(len(self.producer_numbers)):
            table[i, :-1] = self.cost[i, :]
        table[-1, :-1] = self.seller_numbers
        table[:-1, -1] = self.producer_numbers
        table[-1, -1] = np.sum(self.seller_numbers)
        return table


class InitiateAssignment(object):
    
    def __init__(self, settings):
        self.settings = settings
    
    @staticmethod
    def left_2_assign(assignment, settings):
        row, col = assignment.shape
        left_in_row, left_in_col= [], []
        for r in range(row):
            if np.sum(assignment[r, :])!=settings.get_produce_number(r):
                left_in_row.append(r)
        for c in range(col):
            if np.sum(assignment[:, c])!=settings.get_seller_number(c):
                left_in_col.append(c)
        return left_in_row, left_in_col
    
    def exec_by_min_element(self, print_on=True):
        cost_matrix = self.settings.get_cost_matrix()
        cost_matrix_shape = cost_matrix.shape
        
        # Initiate assignment
        assignment = np.zeros(cost_matrix_shape)
        # 1st small element
        min_cost_location = np.argmin(cost_matrix)
        min_cost_row = min_cost_location // cost_matrix_shape[1]
        min_cost_column = min_cost_location % cost_matrix_shape[1]
        assignment[min_cost_row][min_cost_column] = min(settings.get_produce_number(min_cost_row),
                                                        settings.get_seller_number(min_cost_column))
        # loop to assign
        loop_count = 0
        row, col = assignment.shape
        while True:
            loop_count = loop_count+1
            if print_on:
                print('Loop', loop_count)
                print(assignment)
            left_in_row, left_in_col = InitiateAssignment.left_2_assign(assignment, settings)
            if len(left_in_row)+len(left_in_col) == 0: 
                break
            else:
                assign_flag = False
                for r in left_in_row:
                    if np.sum(assignment[r, :])!=0:
                        InitiateAssignment.assign_row(assignment, r, settings, left_in_col, print_on)
                        assign_flag = True
                        break
                # if already assign once in this loop, continue next loop
                if assign_flag is True: continue
                for c in left_in_col:
                    if np.sum(assignment[:, c])!=0:
                        InitiateAssignment.assign_col(assignment, c, settings, left_in_row, print_on)
                        break
        return assignment
    
    @staticmethod
    def assign_row(assignment, row, settings, left_in_col, print_on=True):
        col_cost = [(settings.get_cost_matrix()[row][i], i) for i in left_in_col]
        cost, col = min(col_cost, key=lambda t: t[0])
        
        row_left = settings.get_produce_number(row) - np.sum(assignment[row, :])
        col_left = settings.get_seller_number(col) - np.sum(assignment[:, col])
        
        assignment[row][col] = min([row_left, col_left])
        if print_on:
            print('Assign row %d: [%d, %d] -> %d' % (row, row, col, assignment[row][col]))
        
    @staticmethod
    def assign_col(assignment, col, settings, left_in_row, print_on=True):
        row_cost = [(settings.get_cost_matrix()[i][col], i) for i in left_in_row]
        cost, row = min(row_cost, key=lambda t: t[0])
        
        row_left = settings.get_produce_number(row) - np.sum(assignment[row, :])
        col_left = settings.get_seller_number(col) - np.sum(assignment[:, col])
        
        assignment[row][col] = min([row_left, col_left])
        if print_on:
            print('Assign col %d: [%d, %d] -> %d' % (col, row, col, assignment[row][col]))


class AssignmentCheck(object):
    
    def __init__(self, settings):
        self.settings = settings
        
    def exec_by_check_num(self, assignment, print_on=True):
        """
        Parameter
        ---------
            assignment: 2-D array
                The assignment matrix originally get by InitiateAssignment
            
        return: tuple with 2 elements. 1st element is the result of check: if check numbers contain 
                negative numbers, return true, else return false. 2nd element is the check number 
                matrix.
        """
        row, col = assignment.shape
        # Initiate check table
        check_table = np.zeros((row+1, col+1))
        for r in range(row):    
            for c in range(col):
                check_table[r, c] = 0 if assignment[r][c]!=0 else np.inf
        check_table[:, -1] = np.inf
        check_table[-1, :] = np.inf
        # Initiate a check num
        check_table[0, -1] = 0
        
        # Initiate check numbers:
        if print_on:
            print('Initiate: ')
            print(check_table)
        loop_count = 0
        while np.isinf(check_table[:-1, -1]).any() or np.isinf(check_table[-1, :-1]).any():
            loop_count = loop_count +1
            if print_on:
                print('loop', loop_count)
            for r in range(row):
                for c in range(col):
                    # Skip if it is not assigned
                    if assignment[r][c]==0: continue
                    # One of the check num can be calculated
                    if (check_table[-1, c]!=np.inf and check_table[r, -1] == np.inf) or \
                       (check_table[-1, c]==np.inf and check_table[r, -1] != np.inf):
                        cost = self.settings.get_cost_matrix()[r][c]
                        if check_table[-1, c]==np.inf:
                            check_table[-1, c] = cost - check_table[r, -1]
                        else:
                            check_table[r, -1] = cost - check_table[-1, c]
            if print_on:
                print(check_table)
        # Calculate check numbers in table
        if print_on:
            print('Calculate check numbers in table:')
        for r in range(row):
            for c in range(col):
                # Skip check number == 0
                if assignment[r][c]!=0: continue
                # Calculate: check number = cost - u[i] - v[j]
                else:
                    cost = self.settings.get_cost_matrix()[r][c]
                    check_table[r][c] = cost - check_table[r, -1] - check_table[-1, c]
        if print_on:
            print(check_table)
        return (check_table[:-1, :-1]>=0).all(), check_table


class TreeNode4ShortCircuit(object):

    def __init__(self, check_num_matrix, row, col, father_node=None):
        self.check_num_matrix = check_num_matrix
        self.row = row
        self.col = col
        self.check_num_matrix = check_num_matrix
        self.value = check_num_matrix[row][col]
        self.childern = []
        self.father = father_node
        if father_node is not None:
            self.history = copy.deepcopy(father_node.get_history())
            self.add_history(father_node.get_row(), father_node.get_col())
        else:
            self.history = set()
            
        # filter and get check number is 0. which is not in history(not used)
        matrix_row, matrix_col = self.check_num_matrix.shape
        non_zeros = [TreeNode4ShortCircuit.hashcode_of(matrix_col, pair[0], pair[1]) \
                     for pair in np.argwhere(check_num_matrix)]
        self.possible_zeros = [(r, c) for r in range(matrix_row) for c in range(matrix_col) \
                                if TreeNode4ShortCircuit.hashcode_of(matrix_col, r, c) not in non_zeros and\
                                not self.in_history(r, c)]
        
    def add_child(self, node):
        self.childern.append(node)
        
    def get_row(self):
        return self.row
    
    def get_col(self):
        return self.col
        
    def get_value(self):
        return self.value
        
    def get_children(self):
        return self.childern
    
    def get_father(self):
        return self.father
    
    def get_history(self):
        return self.history
    
    def add_history(self, row, col):
        self.history.add(TreeNode4ShortCircuit.hashcode_of(self.check_num_matrix.shape[1], row, col))
    
    def in_history(self, row, col):
        return TreeNode4ShortCircuit.hashcode_of(
                    self.check_num_matrix.shape[1], 
                    row, col
                ) in self.history
    
    def get_branch_full_values(self):
        values = [(self.row, self.col)]
        node = self.father
        while node is not None:
            values.append((node.row, node.col))
            node = node.get_father()
        values.reverse()
        return values
    
    @staticmethod
    def hashcode_of(matrix_col, row, col):
        return matrix_col * row + col
    
    def circuit_children(self):
        possible = []
        center_row, center_col = self.get_row(), self.get_col()
        for point in self.possible_zeros:
            # only points at the same row or column can be selected
            if point[0]!= center_row and point[1] != center_col:
                continue
            # only points at different row and column comparing to father node can be selected
            if self.father is not None and \
                (point[0]==self.father.get_row() or point[1]==self.father.get_col()):
                continue
            # Skip if history contains point at the same column or the same row twice.
            row_count, col_count = 0, 0
            for h in self.history:
                h_row = h // self.check_num_matrix.shape[1]
                h_col = h % self.check_num_matrix.shape[1]
                if h_row==point[0]:  row_count = row_count+1
                if h_col==point[1]:  col_count = col_count+1
            if row_count<2 and col_count<2:
                possible.append([point[0], point[1]])
        
        child_nodes = []
        for loc in possible:
            i, j = loc
            if not self.in_history(i, j):
                child_node = TreeNode4ShortCircuit(self.check_num_matrix, i, j, self)
                child_nodes.append(child_node)
        return child_nodes
        

class Tree4ShortCircuit(object):
    
    def __init__(self, check_num_matrix):
        self.check_num_matrix = check_num_matrix
        self.matrix_row, self.matrix_col = self.check_num_matrix.shape

    @staticmethod
    def get_more_children_nodes(nodes):
        children_nodes = [each for node in nodes for each in node.circuit_children()]
        return children_nodes
    
    def search(self, head_i, head_j):
        head_node = TreeNode4ShortCircuit(self.check_num_matrix, head_i, head_j)
        children = Tree4ShortCircuit.get_more_children_nodes([head_node])
        loop_count = 0
        while len(children)!=0:
            loop_count = loop_count +1
#             print('Opt short circuit searching Loop', loop_count)
            # update children
            children = Tree4ShortCircuit.get_more_children_nodes(children)
            # check if solved a circuit.
            for child in children:
                check_row, check_col = Tree4ShortCircuit.is_a_circuit(head_node, child)
#                 print(child.get_history(), check_row, check_col)
                if check_row==0 and check_col==0:
                    return child
    
    @staticmethod
    def is_a_circuit(head, node):
        at_head_row_count, at_head_col_count = 0, 0
        while node is not None:
            row, col = node.get_row(), node.get_col()
            if row==head.get_row(): at_head_row_count = at_head_row_count+1
            if col==head.get_col(): at_head_col_count = at_head_col_count+1
            node = node.get_father()
        return 2-at_head_row_count, 2-at_head_col_count
            

class AssignmentOptimizatize(object):
    
    def __init__(self, settings):
        self.settings = settings
        
    @staticmethod
    def local_min_check_num(check_num_matrix):
        min_value_location = np.argmin(check_num_matrix)
        min_value_row = min_value_location // check_num_matrix.shape[1]
        min_value_col = min_value_location % check_num_matrix.shape[1]
        return min_value_row, min_value_col
        
    def exec_by_short_circuit(self, assignment, check_num_matrix):
        loc_row, loc_col = AssignmentOptimizatize.local_min_check_num(check_num_matrix)
        # build a tree for short circuit
        tree = Tree4ShortCircuit(check_num_matrix)
        a_circuit = tree.search(loc_row, loc_col)
        circuit_loc = a_circuit.get_branch_full_values()
        print('Circuit:')
        print(circuit_loc)
        # get the min. assigniment num in circuit points
        min_assignment = min([assignment[loc[0]][loc[1]] \
                            for loc in circuit_loc if check_num_matrix[loc[0]][loc[1]]==0
                        ])
        for i, loc in enumerate(circuit_loc):
            # add assignment number
            if i%2==0:  assignment[loc[0]][loc[1]] = assignment[loc[0]][loc[1]] + min_assignment
            # cut assignment number
            else:       assignment[loc[0]][loc[1]] = assignment[loc[0]][loc[1]] - min_assignment  
        
        

if __name__ == '__main__':
    producer_number, seller_number = [7, 4, 9], [3, 6, 5, 6]
    settings = TransportSetting2D(seller_number, producer_number)
    # set costs
    cost = [[3, 11,  3, 10], 
            [1,  9,  2,  8], 
            [7,  4, 10,  5]]
    for i in range(len(cost)):
        for j in range(len(cost[i])):
            settings.set_cost(i, j, cost[i][j])

    init_assignment = InitiateAssignment(settings)
    assignment = init_assignment.exec_by_min_element(print_on=False)
    
    check_assignment = AssignmentCheck(settings)
    assignment_opt = AssignmentOptimizatize(settings)
    exit_mark = False
    loop_count = 0
    while not exit_mark:
        loop_count = loop_count+1
        print()
        print('>> Control loop', loop_count)
        print('assignment: ')
        print(assignment)
        if loop_count>100:
            raise Exception("Loop exceeded 100 rounds: %d, maybe there is an error in the programme." % loop_count)
        check_result = check_assignment.exec_by_check_num(assignment, print_on=False)
        exit_mark, check_num_matrix = check_result
        print('Check numner matrix:')
        print(check_num_matrix)
        if exit_mark:       break
        # drop u[i], v[i] and leave check numbers only.
        check_num_matrix = check_num_matrix[:-1, :-1]
        assignment_opt.exec_by_short_circuit(assignment, check_num_matrix)