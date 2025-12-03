# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Tree class implementation for representing tree structures in genetic programming.
"""

from slim_gsgp_lib_np.algorithms.MULTI_SLIM.representations.tree_utils import _execute_tree, tree_depth_and_nodes
from slim_gsgp_lib_np.algorithms.MULTI_SLIM.utils.metrics import silhouette_score_ensemble

class Tree:
    """
    The Tree class representing the candidate solutions in MULTI-SLIM-GSGP.

    Attributes
    ----------
    repr_ : tuple or str
        Representation of the tree structure.
    FUNCTIONS : dict
        Dictionary of allowed functions in the tree representation.
    TERMINALS : dict
        Dictionary of terminal symbols allowed in the tree representation.
    CONSTANTS : dict
        Dictionary of constant values allowed in the tree representation.
    SPECIALISTS : dict
        Dictionary of specialist functions allowed in the tree representation.
    depth : int
        Depth of the tree.
    fitness : float
        Fitness value of the tree.
    test_fitness : float
        Test fitness value of the tree.
    nodes_count : int
        Number of nodes in the tree.
    total_nodes : int
        Total number of nodes in the tree, including the specialists.

    Methods
    -------
    apply_tree(inputs, testing, predict)
        Apply the tree to the input data.
    calculate_semantics(inputs, testing=False)
        Calculate the semantics for the tree, if they have not been calculated yet.
    evaluate(ffunction, X, y, testing=False)
        Evaluate the tree on the given data.
    predict(X)
        Predict the target values using the tree.
    get_tree_representation(indent="")
        Get the tree representation as a string.
    print_tree_representation(indent="")
        Print the tree representation.
    """

    TERMINALS = None
    FUNCTIONS = None
    CONSTANTS = None
    SPECIALISTS = None

    def __init__(self, collection):
        """
        Initializes a Tree object.

        Parameters
        ----------
        collection : tuple or str
            Representation of the tree structure.
        """
        self.FUNCTIONS = Tree.FUNCTIONS
        self.TERMINALS = Tree.TERMINALS
        self.CONSTANTS = Tree.CONSTANTS
        self.SPECIALISTS = Tree.SPECIALISTS

        self.collection = collection
        self.depth, self.nodes_count, self.total_nodes = tree_depth_and_nodes(collection, self.SPECIALISTS) 
        self.fitness, self.test_fitness = None, None
        self.train_semantics, self.test_semantics = None, None 
        self.silhouette_score = None


    def apply_tree(self, inputs, testing, predict):
        """
        Apply the tree to the input data.

        Parameters
        ----------
        inputs : np.ndarray
            Input data.
        testing : bool
            Whether the tree is being evaluated on a testing set.
        predict : bool
            Whether the tree is being used for prediction (i.e., not with the intended datasets (train, test)).

        Returns
        -------
        semantics : np.ndarray
            Predicted values.
        """

        return _execute_tree(
            collection=self.collection,
            X=inputs,
            FUNCTIONS=self.FUNCTIONS,
            TERMINALS=self.TERMINALS,
            CONSTANTS=self.CONSTANTS,
            SPECIALISTS=self.SPECIALISTS,
            testing=testing,
            predict=predict
        )
    
    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for the tree.

        Parameters
        ----------
        inputs : np.ndarray
            Input data for calculating semantics.
        testing : bool
            Boolean indicating if the calculation is for testing semantics.
        """
        if testing and self.test_semantics is None:
            self.test_semantics = self.apply_tree(inputs, testing=True, predict=False)
        elif self.train_semantics is None:
            self.train_semantics = self.apply_tree(inputs, testing=False, predict=False)


    def evaluate(self, ffunction, X, y, testing=False):
        """
        Evaluate the tree on the given data.

        Parameters
        ----------
        ffunction : callable
            Fitness function.
        X : np.ndarray
            Input data.
        y : np.ndarray
            Target data.
        testing : bool
            Whether the tree is being evaluated on a testing set.
        """

        preds = self.apply_tree(X, testing, False)
        if testing:
            self.test_fitness = ffunction(y, preds)
        else:
            self.fitness = ffunction(y, preds)      

    def predict(self, X):
        """
        Predict the target values using the tree.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Predicted values.
        """

        return self.apply_tree(X, testing=False, predict=True)
    

    def get_tree_representation(self, indent=""):
        """
        Get the tree representation as a string.

        Returns
        -------
        str
            A string representation of the tree.
        """
        representation = []

        # Check if the collection is a tuple.
        if isinstance(self.collection, tuple):
            # Check if the first element is a Condition object.
            if hasattr(self.collection[0], "repr_"):
                # This is an ensemble (conditional) node.
                condition_obj = self.collection[0]
                representation.append(indent + "if (\n")
                # Use the condition's own representation.
                if hasattr(condition_obj, "get_tree_representation"):
                    representation.append(condition_obj.get_tree_representation(indent + "  "))
                else:
                    representation.append(indent + "  " + str(condition_obj.repr_) + "\n")
                representation.append(indent + ") > 0 then\n")
                # Recursively display the true branch.
                representation.append(Tree(self.collection[1]).get_tree_representation(indent + "  "))
                representation.append(indent + "else\n")
                # Recursively display the false branch.
                representation.append(Tree(self.collection[2]).get_tree_representation(indent + "  "))
                representation.append(indent + "endif\n")
            # Otherwise, if the first element is a GP function node.
            elif isinstance(self.collection[0], str) and self.collection[0] in Tree.FUNCTIONS:
                function_name = self.collection[0]
                representation.append(indent + f"{function_name}(\n")
                arity = Tree.FUNCTIONS[function_name]["arity"]
                if arity == 2:
                    representation.append(Tree(self.collection[1]).get_tree_representation(indent + "  "))
                    representation.append(Tree(self.collection[2]).get_tree_representation(indent + "  "))
                elif arity == 1:
                    representation.append(Tree(self.collection[1]).get_tree_representation(indent + "  "))
                representation.append(indent + ")\n")
            else:
                # If the structure is unrecognized, just print it.
                representation.append(indent + str(self.collection) + "\n")
        else:
            # Terminal (specialist) node.
            representation.append(indent + str(self.collection) + "\n")

        return "".join(representation)


    def print_tree_representation(self, indent=""):
        """
        Print the tree representation with the given indentation.
        """
        print(self.get_tree_representation(indent=indent))


    def get_simple_representation(self):
        """
        Returns a one-line, simple string representation of the tree.
        For ensemble nodes, the format is:
            ( condition_simple, branch_true_simple, branch_false_simple )
        For GP function nodes, it is similar:
            ( function_name, child1_simple, child2_simple )
        For terminals, it simply returns the terminal string.
        """
        if isinstance(self.collection, tuple):
            if hasattr(self.collection[0], "repr_"):
                cond_str = self.collection[0].repr_
                true_str = Tree(self.collection[1]).get_simple_representation()
                false_str = Tree(self.collection[2]).get_simple_representation()
                return f"({cond_str}, {true_str}, {false_str})"
        # Specialist node.
        else:
            return str(self.collection)

    def __str__(self):
        return self.get_simple_representation()
    
    def __copy__(self):
        """
        Create a deep copy of the Tree object.
        """
        new_tree = Tree(self.collection)
        new_tree.FUNCTIONS = self.FUNCTIONS
        new_tree.TERMINALS = self.TERMINALS
        new_tree.CONSTANTS = self.CONSTANTS
        new_tree.SPECIALISTS = self.SPECIALISTS
        new_tree.depth = self.depth
        new_tree.nodes_count = self.nodes_count
        new_tree.total_nodes = self.total_nodes
        new_tree.train_semantics = self.train_semantics
        new_tree.test_semantics = self.test_semantics
        
        return new_tree
    
    def __getitem__(self, item):
        """
        Get an individual from the population by index.
        
        Parameters
        ----------
        item : int
            The index of the individual to retrieve.

        Returns
        -------
        Tree
            The individual at the specified index.
        """
        element = self.collection
        try: 
            for id in item: 
                element = element[id]
        except IndexError:
            print("Index not found")
        except TypeError:
            print("Invalid index type")
    
        return element