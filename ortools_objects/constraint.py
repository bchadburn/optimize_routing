import logging
from typing import Callable, Union

from ortools_objects.component import ORComponent
from ortools_objects.indexed_component import IndexedComponent
from ortools_objects.model import ORToolsCPModel


class IndexedORStandardConst(IndexedComponent):
    """
    An indexed standard constraint for an ORTools optimization model.
    This class represents a constraint that is defined over one or more sets. It takes a callable function (rule) that defines the 
    constraint expression, along with the sets over which the constraint should be indexed.

    Args:
        *sets: One or more ORSet objects that will be used to create the index set for the constraint.

    Kwargs:
        doc (str): A documentation string providing a description of the constraint.
        name (str): The name of the constraint, which will be used as the dictionary key for accessing its values.
        rule (callable): A callable function that takes the ORTools model and the index values as arguments, and returns the constraint expression.
        log_cardinality (bool, optional): A boolean indicating whether the cardinality (number of constraints) should be logged. Defaults to True.

    Example:
        Suppose you have a set of time periods and a set of distribution sites, and you want to create a constraint that ensures the actual 
        production at each site and time period is equal to the sum of piecewise productions for that site and time period. 
        You can define the constraint as follows:

        model.s_time_periods = ORSet(name='time_periods', initialize=[0, 1, 2, ...])
        model.s_distribution_sites = ORSet(name='distribution_sites', initialize=['site1', 'site2', 'site3', ...])

        def production_constraint(model, time_period, site):
            return model.v_actual_production[time_period, site] == sum(
                model.v_piecewise_production[time_period, site, piecewise_idx]
                for piecewise_idx in model.s_piecewise_segments[time_period, site]
            )

        model.c_production = IndexedORStandardConst(
            model.s_time_periods, model.s_distribution_sites,
            name='production_constraint',
            doc='Constraint ensuring actual production equals sum of piecewise productions',
            rule=production_constraint
        )

    In this example, the `production_constraint` function defines the constraint expression, and the `IndexedORStandardConst` object creates an indexed constraint over the sets of time periods and distribution sites, using the provided rule.
    """


    def __init__(
        self,
        *args,
        **kwds,
    ):
        # Declare component type
        kwds.setdefault("ctype", "constraint")
        # Validate the args to make sure they are a set
        self._validate_args(args)

        # Save the rule argument
        self._rule = kwds.pop("rule")
        assert isinstance(
            self._rule,
            Callable,
        ), "Rule argument supplied to constraint must be a callable function"
        self._log_cardinality = kwds.pop("log_cardinality", True)
        assert isinstance(
            self._log_cardinality, bool
        ), "Log cardinality supplied to constraint must be a boolean"

        # Initialize the rest of the component
        IndexedComponent.__init__(self, *args, **kwds)
        
        # Only check dimension if index set actually exists, otherwise will error unexpectedly
        if len(self._index_set) !=0:
            assert self._rule.__code__.co_argcount == (
                self.dim() + 1
            ), "Rule function must contain arguments for model and one for each index set type"

    def _validate_args(self, args):
        """Checks to make sure all args are ORSets.

        Raises:
            TypeError: Raised if one of the args is not an ORSet object.
        """
        IndexedComponent._validate_args(self, args)

    def construct(
        self,
        model_wrapper: ORToolsCPModel,
        logger: Union[logging.Logger, None] = None
    ) -> None:
        """Constructs the rule for each index in the index set of the indexed constraint.

        Args:
            model_wrapper (ORToolsCPModel): Full model object containing all variables, sets, and parameters.
            logger (logging.Logger): Logger (optional). Defaults to None.

        Returns:
            dict: Dictionary of name/index of the constraint with the pywraplp constraints as the values.
        """
        if logger:
            logger.debug(f"Began adding constraint {self._name} to model.")
            original_const = len(list(model_wrapper.mathopt_model.linear_constraints()))
        if not self._index_set or len(self._index_set()) == 0:
            return 
        if isinstance(self._index_set[0], tuple):
            rule_dict = {
                index: (
                    [
                        model_wrapper.mathopt_model.add_linear_constraint(expr) 
                        for expr in self._rule(model_wrapper, *index)
                ]
                if isinstance(self._rule(model_wrapper, *index), list)
                else model_wrapper.mathopt_model.add_linear_constraint(
                    self._rule(model_wrapper, *index)
                    )
                )
                for index in self._index_set
            }
        else:
            rule_dict = {
                index: (
                    [
                        model_wrapper.mathopt_model.add_linear_constraint(expr) 
                        for expr in self._rule(model_wrapper, index)
                ]
                if isinstance(self._rule(model_wrapper, index), list)
                else model_wrapper.mathopt_model.add_linear_constraint(
                    self._rule(model_wrapper, index)
                    )
                )
                for index in self._index_set
            }
            
        if logger:
            new_const = len(list(model_wrapper.mathopt_model.linear_constraints()))
            logger.debug(f"Added constraint {self._name} to model")
            if self._log_cardinality:
                logger.debug(
                    f"Constraint {self._name} has {new_const-original_const} constraints added."
                )
        self._data = rule_dict


class ScalarORStandardConst(ORComponent):
    """Scalar OR constraint, used when only one constraint should be declared.
    Should not have any args.

    Kwargs:
        doc (str): A doc string that can be used in the string representation of the constraint
        name (str): The name of the constraint that will be used to name the constraint
        rule (Callable): A callable function that takes in a "model" (ORToolsCPModel). Should not have any indices.

    Example Use: Let's say that I want to constraint a specific site's example2. I could accomplish this through a scalar
    constraint. First, I would need to create the example2 variable. Here, I will make it indexed for demonstration purposes.
    Then, I would need to make a callable rule with only the ORToolsCPModel as the argument:

    def constraint_specific_products_produced(model):
        return v_example_constraint[0, 'site0'] >= 50

    Then, I would create a standard scalar constraint and add it as an attribute to the model:

    model.example_constraint = ScalarORStandardConstraint(name='test', doc='test', rule=constraint_specific_constraint)
    """

    def __init__(
        self,
        **kwds,
    ):
        kwds.setdefault("ctype", "constraint")
        self._rule = kwds.pop("rule")
        assert isinstance(
            self._rule,
            Callable,
        ), "Rule argument supplied to constraint must be a callable function"
        ORComponent.__init__(self, (), **kwds)
        assert (
            self._rule.__code__.co_argcount == 1
        ), "Scalar constraint rule function must only contain a call to model object and nothing else"

    def construct(
        self,
        model_wrapper: ORToolsCPModel,
        logger: Union[logging.Logger, None] = None
    ):
        """Constructs a single constraint

        Args:
            model_wrapper (ORToolsCPModel): Full mode object with all constraints, variables, parameters, and objective
            logger (logging.Logger, optional): Optional logger. Defaults to None.
        """
        if logger:
            logger.info(f"Added scalar constraint {self._name} to model")
        self._data = model_wrapper.mathopt_model.add_linear_constraint(
            self._rule(model_wrapper)
            )
