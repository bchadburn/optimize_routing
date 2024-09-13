import logging
from numbers import Number
from typing import Any, Optional, Union

import numpy as np
from ortools.math_opt.python import mathopt

from ortools_objects.component import ORComponent
from ortools_objects.indexed_component import IndexedComponent
from ortools_objects.model import ORToolsCPModel


class IndexedORBoolVariable(IndexedComponent):
    """
    An indexed boolean variable object for an ORTools optimization model.
    This class represents a collection of binary (0 or 1) decision variables indexed over one or more sets. 
    Indexed boolean variables are commonly used in optimization models to represent binary decisions, such as whether a 
    facility is open or closed, whether a task is performed or not, or whether a constraint is satisfied or violated.

    Args:
        *sets: One or more ORSet objects that will be used to create the index set for the boolean variables.

    Kwargs:
        doc (str): A documentation string providing a description of the boolean variables.
        name (str): The name of the boolean variables, which will be used as the dictionary key for accessing their values.
        log_cardinality (bool, optional): A boolean indicating whether the cardinality (number of variables) should be logged. Defaults to True.
        log_solution (bool, optional): A boolean indicating whether the solution values of the variables should be logged. Defaults to False.

    Example:
        Suppose you have a set of distribution sites, and you want to create a binary variable for each site to indicate whether the site is active or not. First, create the set of sites:

        model.s_distribution_sites = ORSet(
            name='distribution_sites',
            doc='Set of distribution sites',
            initialize=['site0', 'site1', 'site2', 'site3']
        )

        Then, create an indexed boolean variable over the set of sites:

        model.bv_site_active = IndexedORBoolVariable(
            model.s_distribution_sites,
            name='site_active',
            doc='Binary variable indicating whether a distribution site is active'
        )

        You can now use this indexed boolean variable in constraints or other model components to represent the decision of whether each site should be active or not.
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


    def construct(self, model_wrapper: ORToolsCPModel, logger: Union[logging.Logger, None]) -> None:
        """
        Constructs and adds the indexed boolean variables to the given ORTools model.
        This method creates and adds the boolean variables represented by this object to the provided ORTools model. 
        The variables are indexed over the index set defined by the sets used to create this object.

        Args:
            model_wrapper (ORToolsCPModel): The ORTools model to which the boolean variables should be added.
            logger (logging.Logger): A logger object used for logging information about the model construction process.
        """
        
        if logger:
            logger.debug(f"Began adding variable {self._name} to model")
        original_vars = len(list(model_wrapper.mathopt_model.variables()))
        var_dict = {
            index: model_wrapper.mathopt_model.add_binary_variable(
                name=f"{self._name}_{index}"
            )
            for index in self
        }
        self._constructed = True
        self._solved = False
        new_vars = len(list(model_wrapper.mathopt_model.variables()))
        if logger:
            logger.info(f"Added variable {self._name} to model")
            if self._log_cardinality:
                logger.info(
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


    def process_result(
        self,
        result: mathopt.SolveResult,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self._solution = {
            index: round(result.variable_values(self[index])) for index in self
        }
        self._solved = True
        if logger and self._log_solution:
            for index in self:
                logger.info(
                    f"Variable {self._name} at indices {self._index_name}: {index} has value {round(self[index],2)}"
                )


    def __getitem__(self, index: Union[str, tuple[Any, ...]]) -> Any:
        if self._solved:
            return self._solution[index]
        else:
            return super().__getitem__(index)




class ScalarORBoolVariable(ORComponent):
    """
    A scalar boolean variable object for an ORTools optimization model.

    This class represents a single binary (0 or 1) decision variable. Scalar boolean variables are useful for representing 
    binary decisions that are not indexed over any set, such as whether to perform a specific action or not.

    Kwargs:
        doc (str): A documentation string providing a description of the boolean variable.
        name (str): The name of the boolean variable, which will be used to access its value.

    Example:
        Suppose you have a supply chain optimization problem, and you want to create a boolean variable to represent the 
        decision of whether to build a new distribution center or not. You can create a scalar boolean variable as follows:

        model.bv_build_new_center = ScalarORBoolVariable(
            name='build_new_center',
            doc='Binary variable indicating whether to build a new distribution center'
        )

        You can then use this scalar boolean variable in constraints or other model components to represent the decision of building the new distribution center or not.
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
        self, model_wrapper: ORToolsCPModel, logger: Union[logging.Logger, None]
    ) -> mathopt.Variable:
        """Adds the boolean variable to the model.


        Args:
            model_wrapper (ORToolsCPModel): The model to which the boolean variable should be added
            logger (logging.Logger): The logger object for the model.


        Returns:
            mathopt.Variable: The boolean variable added to the model
        """
        self._constructed = True
        self._solved = False
        if logger:
            logger.info(f"Added scalar variable {self._name} to model")
        self._data = model_wrapper.mathopt_model.add_binary_variable(
            name=f"{self._name}"
        )


    def process_result(
        self, result: mathopt.SolveResult, logger: Union[logging.Logger, None] = None
    ) -> None:
        self._solution = round(result.variable_values(self._data))
        self._solved = True
        if logger and self._log_solution:
            logger.info(
                f"Scalar variable {self._name} has value {round(self._solution,2)}"
            )



class IndexedORContinuousVariable(IndexedComponent):
    """    
    An indexed continuous variable object for an ORTools optimization model.
    This class represents a collection of continuous decision variables indexed over one or more sets. 
    Indexed continuous variables are commonly used in optimization models to represent quantities, levels, or 
    other numerical values that can take on any real value within specified bounds.

    Args:
        *sets: One or more ORSet objects that will be used to create the index set for the continuous variables.

    Kwargs:
        doc (str): A documentation string providing a description of the continuous variables.
        name (str): The name of the continuous variables, which will be used as the dictionary key for accessing their values.
        lb_default (Number, optional): The default lower bound for the continuous variables. Defaults to 0.
        ub_default (Number, optional): The default upper bound for the continuous variables. Defaults to positive infinity.
        lower_bounds (dict, optional): A dictionary containing custom lower bounds for specific index combinations, where the keys are the index tuples, and the values are the corresponding lower bounds.
        upper_bounds (dict, optional): A dictionary containing custom upper bounds for specific index combinations, where the keys are the index tuples, and the values are the corresponding upper bounds.
        log_cardinality (bool, optional): A boolean indicating whether the cardinality (number of variables) should be logged. Defaults to True.
        log_solution (bool, optional): A boolean indicating whether the solution values of the variables should be logged. Defaults to True.

    Example:
        Suppose you have a set of distribution sites, and you want to create a continuous variable for each site representing the suction pressure of a pump. First, create the set of sites:

        model.s_distribution_sites = ORSet(
            name='distribution_sites',
            doc='Set of distribution sites',
            initialize=['site0', 'site1', 'site2', 'site3']
        )

        Then, create an indexed continuous variable over the set of sites:

        model.v_suction_pressure = IndexedORContinuousVariable(
            model.s_distribution_sites,
            name='suction_pressure',
            doc='Suction pressure of the pump at each distribution site',
            lb_default=-20,
            lower_bounds={'site1': -10}
        )

        In this example, the lower bound for the suction pressure variable is -20 for all sites except 'site1', which has a lower bound of -10.
    """


    def __init__(self, *args, **kwds):
        # Set the component type to variable
        kwds.setdefault("ctype", "var")


        # Validate the args passed to the constructor
        self._validate_args(args)
        self._validate_kwds(kwds)


        # Populates lb and ub defaults as local attributes
        self._lb_default = kwds.pop("lb_default", 0)
        self._ub_default = kwds.pop("ub_default", np.inf)


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


    def construct(self, model_wrapper: ORToolsCPModel, logger: Union[logging.Logger, None]) -> None:
        """Add the variable to the model

        Args:
            model_wrapper (ORToolsCPModel): The model to which the variable should be added.
            logger (logging.Logger, Optional): Logger object. Defaults to None
        """
        if logger:
            logger.debug(f"Began adding variable {self._name} to model")
        original_vars = len(list(model_wrapper.mathopt_model.variables()))
        var_dict = {
            index: model_wrapper.mathopt_model.add_variable(
                lb=self._lbs[index], ub=self._ubs[index], name=f"{self._name}_{index}"
            )
            for index in self
        }
        new_vars = len(list(model_wrapper.mathopt_model.variables()))
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
                kwds["lower_bounds"], dict
            ), "Custom lower bounds must be a dictionary with str | Number | tuple as keys and Number as values"
        if "upper_bounds" in kwds:
            assert isinstance(
                kwds["upper_bounds"], dict
            ), "Custom upper bounds must be a dictionary with str | Number | tuple as keys and Number as values"
        if "log_cardinality" in kwds:
            assert isinstance(
                kwds["log_cardinality"], bool
            ), "Log cardinality argument must be a boolean"
        if "log_solution" in kwds:
            assert isinstance(
                kwds["log_solution"], bool
            ), "Log solution argument must be a boolean"


    def process_result(
        self, result: mathopt.SolveResult, logger: Union[logging.Logger, None] = None
    ) -> None:
        self._solution = {index: result.variable_values(self[index]) for index in self}
        self._solved = True
        if logger and self._log_solution:
            for index in self:
                logger.info(
                    f"Variable {self._name} at indices {self._index_name}: {index} has value {round(self[index],2)}"
                )


    def __getitem__(self, index: Union[str, tuple]) -> float:
        if self._solved:
            return self._solution[index]
        else:
            return super().__getitem__(index)




class ScalarORContinuousVariable(ORComponent):
    """
    A scalar continuous variable object for an ORTools optimization model.

    This class represents a single continuous decision variable that is not indexed over any set. Scalar continuous variables are useful for representing quantities, levels, or other numerical values that are not specific to any particular combination of set elements.

    Kwargs:
        doc (str): A documentation string providing a description of the continuous variable.
        name (str): The name of the continuous variable, which will be used to access its value.
        lower_bound (Number, optional): The lower bound for the continuous variable. Defaults to 0.
        upper_bound (Number, optional): The upper bound for the continuous variable. Defaults to positive infinity.
        log_solution (bool, optional): A boolean indicating whether the solution value of the variable should be logged. Defaults to True.

    Example:
        Suppose you have a supply chain optimization problem, and you want to create a continuous variable representing the final pressure of a pipeline. You can create a scalar continuous variable as follows:

        model.v_final_pressure = ScalarORContinuousVariable(
            name='final_pressure',
            doc='Final pressure of the pipeline',
            lower_bound=10,
            upper_bound=500
        )

        In this example, the final pressure variable is constrained to be between 10 and 500 units. If no lower or upper bound is provided, the variable will be bounded between 0 and positive infinity by default.
    """


    def __init__(self, **kwds):
        # Set the component type to variable
        kwds.setdefault("ctype", "var")


        self._validate_kwds(kwds)


        # Set the lower and upper bound attributes
        self._lb = kwds.pop("lower_bound", 0)
        self._ub = kwds.pop("upper_bound", np.inf)


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


    def construct(self, model_wrapper: ORToolsCPModel, logger: Union[logging.Logger, None]):
        """Adds the variable to the model


        Args:
            model_wrapper (ORToolsCPModel): The model to which the variable is added
            logger (logging.Logger): The logger for the model object
        """
        self._constructed = True
        self._solved = False
        if logger:
            logger.info(f"Added scalar variable {self._name} to model")
        self._data = model_wrapper.mathopt_model.add_variable(
            lb=self._lb, ub=self._ub, name=f"{self._name}"
        )


    def process_result(
        self, result: mathopt.SolveResult, logger: Union[logging.Logger, None] = None
    ) -> None:
        self._solution = result.variable_values(self._data)
        self._solved = True
        if logger and self._log_solution:
            logger.info(
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
