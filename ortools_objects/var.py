import logging
from numbers import Number
from typing import Dict, Union

from ortools.linear_solver import pywraplp

from ortools_objects.component import ORComponent
from ortools_objects.indexed_component import IndexedComponent


class IndexedORBoolVariable(IndexedComponent):
    """Indexed boolean variable object for an ORTools model.

    Args:
        A number of ORSet objects that will be crossed together in IndexedComponent init dunder to create the index set.

    Kwargs:
        doc (str): A doc string that can be used in the string representation of the constraint
        name (str): The name of the constraint that will be used to name the dictionary entries
        log_cardinality (bool, Optional): Boolean indicating if cardinality of variable should be sent to log. Defaults to True.
        log_solution (bool, Optional): Boolean indicating if solution of variable shoule be sent to log. Defaults to False.

    Example use: I want to have an indicator variable for whether or not a site is being used. First, I need a set of sites (set.py):

    model.s_sites = ORSet(name='foo', doc='foo', initialize=['site0', 'site1', 'site2', ...])

    I can now create a binary variable for each site:
    model.bv_site_active = IndexedORBoolVariable(model.s_sites, name='foo', doc='foo')
    """

    def __init__(self, *args, **kwds):
        # Set the component type attribute
        kwds.setdefault("ctype", "var")

        # Validate args to make sure that all args are indexed sets
        self._validate_args(args)
        self._validate_kwds(kwds)

        # Determines whether to log number of variables or number of constraints
        self._log_cardinality = kwds.pop("log_cardinality", True)
        self._log_solution = kwds.pop("log_solution", False)

        # Initialize indexed component attributes
        IndexedComponent.__init__(self, *args, **kwds)

    def construct(self, solver: pywraplp.Solver, logger: logging.Logger) -> None:
        """Adds the boolean variable to the solver for each index in the index set of this boolean variable object.

        Args:
            solver (pywraplp.Solver): Solver to which the boolean variables should be added

        Returns:
            dict: Dictionary containing indices/names as the keys and the variables as the values
        """
        if logger:
            logger.info(f"Began adding variable {self._name} to model")
        original_vars = solver.NumVariables()
        var_dict = {index: solver.BoolVar(f"{self._name}_{index}") for index in self}
        self._constructed = True
        self._solved = False
        new_vars = solver.NumVariables()
        if logger:
            logger.debug(f"Added variable {self._name} to model")
            if self._log_cardinality:
                logger.debug(
                    f"Variable {self._name} has {new_vars-original_vars} variables added."
                )
        self._data = var_dict

    def _validate_args(self, args):
        """Validates the args passed to the constructor"""
        IndexedComponent._validate_args(self, args)

    def _validate_kwds(self, kwds):
        if "log_cardinality" in kwds:
            assert isinstance(
                kwds["log_cardinality"], bool
            ), "Log cardinality argument must be a boolean"
        if "log_solution" in kwds:
            assert isinstance(
                kwds["log_solution"], bool
            ), "Log solution argument must be a boolean"

    def process_result(self, logger: logging.Logger = None) -> None:
        self._solution = {index: self[index].SolutionValue() for index in self}
        self._solved = True
        if logger and self._log_solution:
            for index in self:
                logger.debug(
                    f"Variable {self._name} at indices {self._index_name}: {index} has value {round(self[index],2)}"
                )

    def __getitem__(self, index: Union[str, float]) -> float:
        if self._solved:
            return self._solution[index]
        else:
            return super().__getitem__(index)


class ScalarORBoolVariable(ORComponent):
    """Creates a scalar boolean variable to be added to a solver.

    Kwargs:
        doc (str): A doc string that can be used in the string representation of the constraint
        name (str): The name of the constraint that will be used to name the dictionary entries

    Example use: I want a single boolean variable indicating whether a single site should be added or not. I can directly create a variable:

    model.bv_create_distribution_site = ScalarORBoolVar(name='foo', doc='foo')
    """

    def __init__(self, **kwds):
        kwds.setdefault("ctype", "var")
        self._solved = False
        self._validate_kwds(kwds)
        self._log_solution = kwds.pop("log_solution", False)

        IndexedComponent.__init__(self, (), **kwds)

    def __getitem__(self, index):
        if index is not None:
            raise KeyError(f"Index access for scalar var {self._name} must be None.")
        if self._solved:
            return self._solution
        else:
            return self._data

    def _validate_kwds(self, kwds):
        if "log_cardinality" in kwds:
            assert isinstance(
                kwds["log_cardinality"], bool
            ), "Log cardinality argument must be a boolean"
        if "log_solution" in kwds:
            assert isinstance(
                kwds["log_solution"], bool
            ), "Log solution argument must be a boolean"

    def construct(
        self, solver: pywraplp.Solver, logger: logging.Logger
    ) -> pywraplp.Variable:
        """Adds the boolean variable to the solver.

        Args:
            solver (pywraplp.Solver): The solver to which the boolean variable should be added

        Returns:
            pywraplp.Variable: The boolean variable added to the solver
        """
        self._constructed = True
        self._solved = False
        if logger:
            logger.debug(f"Added scalar variable {self._name} to model")
        self._data = solver.BoolVar(f"{self._name}")

    def process_result(self, logger: logging.Logger = None) -> None:
        self._solution = self._data.SolutionValue()
        self._solved = True
        if logger and self._log_solution:
            logger.debug(
                f"Scalar variable {self._name} has value {round(self._solution,2)}"
            )


class IndexedORContinuousVariable(IndexedComponent):
    """Indexed continuous variable object for an ORTools model.

    Args:
        A number of ORSet objects that will be crossed together in IndexedComponent init dunder to create the index set.

    Kwargs:
        doc (str): A doc string that can be used in the string representation of the constraint
        name (str): The name of the constraint that will be used to name the dictionary entries
        lb_default (Number, Optional): The default lower bound of the variable. Defaults to 0.
        ub_default (Number, Optional): The default upper bound of the variable. Defaults to infinity.
        lower_bounds (Dict(str|tuple: Number), Optional): Dictionary containing key: value with custom lower bounds. Defaults to lb_default.
        upper_bounds (Dict(str|tuple: Number), Optional): Dictionary containing key: value with custom upper bounds. Defaults to ub_default.
        log_cardinality (bool, Optional): Boolean indicating if cardinality of variable should be sent to log. Defaults to True.
        log_solution (bool, Optional): Boolean indicating if solution of variable shoule be sent to log. Defaults to True.

    Example use: I want to have a variable representing the products produced at a distribution site. First, I need a set of sites (set.py):

    model.s_sites = ORSet(name='foo', doc='foo', initialize=['site0', 'site1', 'site2', ...])

    I can now create a continuous variable for each site. At a basic level:
    model.v_products_produced = IndexedORBoolVariable(model.s_sites, name='foo', doc='foo')

    However, what if I want custom lower and upper bounds? One can add a lower bound/upper bound dictionary:
    sites_lb = {'site1': 10}

    model.v_products_produced = IndexedORContinuousVariable(model.s_sites, name='foo', doc='foo', lb_default=0, lower_bounds=sites_lb)

    This will create a variable with lower bounds of 0 for all sites except for site1, which will have a lower bound of 10.
    The same logic can be applied to upper bounds with ub_default and upper_bounds.
    """

    def __init__(self, *args, **kwds):
        # Set the component type to variable
        kwds.setdefault("ctype", "var")

        # Validate the args passed to the constructor
        self._validate_args(args)
        self._validate_kwds(kwds)

        # Populates lb and ub defaults as local attributes
        self._lb_default = kwds.pop("lb_default", 0)
        self._ub_default = kwds.pop("ub_default", pywraplp.Solver.Infinity())

        # Populates custom lower and upper bounds
        self._lbs = kwds.pop("lower_bounds", {})
        self._ubs = kwds.pop("upper_bounds", {})

        # Determines whether to log number of variables or number of constraints
        self._log_cardinality = kwds.pop("log_cardinality", True)
        self._log_solution = kwds.pop("log_solution", False)
        self._solved = False

        # Initialize the indexed component super class
        IndexedComponent.__init__(self, *args, **kwds)

        # Validate lower and upper bounds
        self._validate_data_indices(self._lbs)
        self._validate_data_indices(self._ubs)

        # Fill in default lower and upper bounds
        self._fill_bounds()

    def _fill_bounds(self):
        """Establish lower and upper bounds for values that do not have a custom lower and upper bound.
        This way of doing things is inefficient (list comprehension), but it does provide the most flexibility for
        setting upper and lower bounds.
        """
        self._lbs = {
            key: self._lbs[key] if key in self._lbs else self._lb_default
            for key in self._index_set
        }
        self._ubs = {
            key: self._ubs[key] if key in self._ubs else self._ub_default
            for key in self._index_set
        }

    def construct(self, solver: pywraplp.Solver, logger: logging.Logger) -> None:
        """Add the variable to the solver

        Args:
            solver (pywraplp.Solver): The solver to which the variable should be added.
            logger (logging.Logger, Optional): Logger object. Defaults to None

        Returns:
            dict: Dictionary with name/index as the key aand the numerical variables as the values
        """
        if logger:
            logger.info(f"Began adding variable {self._name} to model")
        original_vars = solver.NumVariables()
        var_dict = {
            index: solver.NumVar(
                self._lbs[index], self._ubs[index], f"{self._name}_{index}"
            )
            for index in self
        }
        new_vars = solver.NumVariables()
        if logger:
            logger.debug(f"Added variable {self._name} to model")
            if self._log_cardinality:
                logger.debug(
                    f"Variable {self._name} has {new_vars-original_vars} variables added."
                )
        self._solved = False
        self._constructed = True
        self._data = var_dict

    def _validate_args(self, args):
        """Validate the args passed to the constructor"""
        IndexedComponent._validate_args(self, args)

    def _validate_kwds(self, kwds):
        if "lb_default" in kwds:
            assert isinstance(
                kwds["lb_default"], Number
            ), "Lower bound default must be a number"
        if "ub_default" in kwds:
            assert isinstance(
                kwds["lb_default"], Number
            ), "Upper bound default must be a number"
        if "lower_bounds" in kwds:
            assert isinstance(
                kwds["lower_bounds"], Dict
            ), "Custom lower bounds must be a dictionary with str | Number | tuple as keys and Number as values"
        if "upper_bounds" in kwds:
            assert isinstance(
                kwds["upper_bounds"], Dict
            ), "Custom upper bounds must be a dictionary with str | Number | tuple as keys and Number as values"
        if "log_cardinality" in kwds:
            assert isinstance(
                kwds["log_cardinality"], bool
            ), "Log cardinality argument must be a boolean"
        if "log_solution" in kwds:
            assert isinstance(
                kwds["log_solution"], bool
            ), "Log solution argument must be a boolean"

    def process_result(self, logger: logging.Logger = None) -> None:
        self._solution = {index: self[index].SolutionValue() for index in self}
        self._solved = True
        if logger and self._log_solution:
            for index in self:
                logger.debug(
                    f"Variable {self._name} at indices {self._index_name}: {index} has value {round(self[index],2)}"
                )

    def __getitem__(self, index: Union[str, tuple]) -> float:
        if self._solved:
            return self._solution[index]
        else:
            return super().__getitem__(index)


class ScalarORContinuousVariable(ORComponent):
    """Scalar continuous variable to be added to the model.

    Kwargs:
        doc (str): A doc string that can be used in the string representation of the constraint
        name (str): The name of the constraint that will be used to name the dictionary entries
        lower_bound (Number, Optional): The default lower bound of the variable. Defaults to 0.
        upper_bound (Number, Optional): The default upper bound of the variable. Defaults to infinity.
        log_solution (bool, Optional): Boolean indicating if solution of variable shoule be sent to log. Defaults to True.

    Example use: I want to create a single variable representing final temperature constraint. I can do so using
    a scalar variable:

    model.v_final_temperature = ScalarORContinuousVariable(name='foo', doc='foo', lower_bound=-10, upper_bound=500)

    If no lower or upper bound is provided, model will assume 0 to infinity range.
    """

    def __init__(self, **kwds):
        # Set the component type to variable
        kwds.setdefault("ctype", "var")

        self._validate_kwds(kwds)

        # Set the lower and upper bound attributes
        self._lb = kwds.pop("lower_bound", 0)
        self._ub = kwds.pop("upper_bound", pywraplp.Solver.Infinity())

        # Determines whether to log number of variables or number of constraints
        self._log_solution = kwds.pop("log_solution", False)
        self._constructed = False
        self._solved = False

        # Initialize the component object
        ORComponent.__init__(self, **kwds)

    def __getitem__(self, index):
        if index is not None:
            raise KeyError(f"Index access for scalar var {self._name} must be None.")
        if self._solved:
            return self._solution
        else:
            return self._data

    def construct(self, solver: pywraplp.Solver, logger: logging.Logger):
        """Adds the variable to the solver

        Args:
            solver (pywraplp.Solver): The solver to which the variable is added
            logger (logging.Logger): The logger (optional). Defaults to None.
        """
        self._constructed = True
        self._solved = False
        if logger:
            logger.debug(f"Added scalar variable {self._name} to model")
        self._data = solver.NumVar(self._lb, self._ub, f"{self._name}")

    def process_result(self, logger: logging.Logger = None) -> None:
        self._solution = self._data.SolutionValue()
        self._solved = True
        if logger and self._log_solution:
            logger.debug(
                f"Scalar variable {self._name} has value {round(self._solution,2)}"
            )

    def _validate_kwds(self, kwds):
        if "lower_bound" in kwds:
            assert isinstance(
                kwds["lower_bound"], Number
            ), "Lower bound must be a number"
        if "upper_bound" in kwds:
            assert isinstance(
                kwds["upper_bound"], Number
            ), "Upper bound must be a number"
        if "log_cardinality" in kwds:
            assert isinstance(
                kwds["log_cardinality"], bool
            ), "Log cardinality argument must be a boolean"
        if "log_solution" in kwds:
            assert isinstance(
                kwds["log_solution"], bool
            ), "Log solution argument must be a boolean"
