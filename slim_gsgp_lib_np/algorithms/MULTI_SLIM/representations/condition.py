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
Condition class implementation for representing condition structures in multi-slim programming.
"""

from slim_gsgp_lib_np.algorithms.GP.representations.tree_utils import _execute_tree, tree_depth_and_nodes

class Condition:
    """
    The Condition class representing the candidate solutions in genetic programming.

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
    depth : int
        Depth of the tree.
    node_count : int
        Number of nodes in the tree.
    """

    TERMINALS = None
    FUNCTIONS = None
    CONSTANTS = None

    def __init__(self, repr_):
        """
        Initializes a Tree object.

        Parameters
        ----------
        repr_ : tuple
            Representation of the tree structure.
        """
        self.FUNCTIONS = Condition.FUNCTIONS
        self.TERMINALS = Condition.TERMINALS
        self.CONSTANTS = Condition.CONSTANTS

        self.repr_ = repr_
        self.train_semantics, self.test_semantics = None, None  
        self.depth, self.nodes_count = tree_depth_and_nodes(repr_)

    def apply_tree(self, inputs):
        """
        Evaluates the tree on input vectors.

        Parameters
        ----------
        inputs : tuple
            Input vectors.

        Returns
        -------
        float
            Output of the evaluated tree.
        """

        return _execute_tree(
            repr_=self.repr_,
            X=inputs,
            FUNCTIONS=self.FUNCTIONS,
            TERMINALS=self.TERMINALS,
            CONSTANTS=self.CONSTANTS
        )
    
    def predict(self, inputs):
        """
        Predict the target values using the tree.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data for predicting target values.

        Returns
        -------
        torch.Tensor
            Predicted target values.
        """
        return self.apply_tree(inputs)
    
    def calculate_semantics(self, inputs, testing=False):
        """
        Calculate the semantics for the tree, if they have not been calculated yet.

        Parameters
        ----------
        inputs : torch.Tensor
            Input data for calculating semantics.

        Returns
        -------
        self.train_semantics or self.test_semantics
            Returns the calculated semantics.
        """
        if testing and self.test_semantics is None:
            self.test_semantics = self.apply_tree(inputs)
        elif self.train_semantics is None:
            self.train_semantics = self.apply_tree(inputs)

    def get_tree_representation(self, indent=""):
        """
        Returns the tree representation as a string with indentation.

        Parameters
        ----------
        indent : str, optional
            Indentation for tree structure representation. Default is an empty string.

        Returns
        -------
        str
            Returns the tree representation with the chosen indentation.
        """
        representation = []

        if isinstance(self.repr_, tuple):  # If it's a function node
            function_name = self.repr_[0]
            representation.append(indent + f"{function_name}(\n")

            # if the function has an arity of 2, process both left and right subtrees
            if Condition.FUNCTIONS[function_name]["arity"] == 2:
                left_subtree, right_subtree = self.repr_[1], self.repr_[2]
                representation.append(Condition(left_subtree).get_tree_representation(indent + "  "))
                representation.append(Condition(right_subtree).get_tree_representation(indent + "  "))
            # if the function has an arity of 1, process the left subtree
            else:
                left_subtree = self.repr_[1]
                representation.append(Condition(left_subtree).get_tree_representation(indent + "  "))

            representation.append(indent + ")\n")
        else:  # If it's a terminal node
            representation.append(indent + f"{self.repr_}\n")

        return "".join(representation)

    def print_tree_representation(self, indent=""):
        """
        Prints the tree representation with indentation.

        Parameters
        ----------
        indent : str, optional
            Indentation for tree structure representation. Default is an empty string.

        Returns
        -------
        None
            Prints the tree representation as a string with indentation.

        """

        print(self.get_tree_representation(indent=indent))

    def __str__(self):  
        return self.repr_