import logging
from typing import Callable

from ortools_objects.component import ORComponent
from ortools_objects.indexed_component import IndexedComponent
from ortools_objects.model import ORToolsCPModel


class IndexedORStandardConst(IndexedComponent):
    """Indexed standard constraint. Takes in several args and kwargs and creates a constraint using a callable function.

    Args:
        A number of ORSet objects that will be crossed together in IndexedComponent init dunder to create the index set.

    Kwargs:
        doc (str): A doc string that can be used in the string representation of the constraint
        name (str): The name of the constraint that will be used to name the dictionary entries
        rule (Callable): A callable function that takes in a "model" (ORToolsCPModel), along with the indices (string args) for which the rule should be created as separate arguments.
        log_cardinality (bool, Optional): Boolean indicating if cardinality of variable should be sent to log. Defaults to True.

    Example Use: Let's say that we want to define the example example2 of a site during a given time period as the example2 example2 of the sit plus the deltap
    of the site if it is available. First, we want to have the variable objects defined. Next, we want to define a constraint Callable function that takes
    in the model (with its time period and site), as well as any indices that we want to create the constraint for (here, time period and site):

    def example_definition(model, time_period, site):
        if (time_period, site) in model.v_station_deltap:
            return (
                model.v_example[time_period, site]
                == model.v_example2[time_period, site]
                + model.v_station_deltap[time_period, site]
            )
        else:
            return (
                model.v_example[time_period, site]
                == model.v_example2_example2[time_period, site]
            )

    Once we havce the callable function, we can create a IndexedORConstraint object. Since the constraint is indexed over time period and site, we want to pass
    in these arguments as ORSet objects:

    model.time_set = ORSet(name='time_index', doc='time_index', initialize=[0, 1,...])
    model.site_set = ORSet(name='site_index', doc='site_index', initialize=['site0', 'site1', ...])

    Now, we can create our constraint object:
    model.example_constraint = IndexedORStandardConst(model.time_set, model.site_set, name='example', doc='example', rule=example_definition)

    When model.construct_model() is called, the construct method of model.example_constraint will be called, which will create a dictionary in the format:
    {(time_index, site_index): pywraplp.Constraint} for each combination of time period and site indices.


    In the above function, the indexed set allows the constraint to sum over unevent numbers of piecewise indices for different sets. It should be noted that
    there are other ways to accomplish this, but the the indexed set makes summations like this easier to write and comprehend.
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
        logger: logging.Logger = None,
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
        if len(self._index_set) == 0:
            return dict()
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
            new_const = len(list(model_wrapper.methopt_model.linear_constraints()))
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
        ), "Scalar constraint rule function must only contain a call to model object and nothinbg else"

    def construct(
        self,
        model_wrapper: ORToolsCPModel,
        logger: logging.Logger = None,
    ):
        """Constructs a single constraint

        Args:
            model_wrapper (ORToolsCPModel): Full mode object with all constraints, variables, parameters, and objective
            logger (logging.Logger, optional): Optional loggger. Defaults to None.
        """
        if logger:
            logger.info(f"Added scalar constraint {self._name} to model")
        self._data = model_wrapper.mathopt_model.add_linear_constraint(
            self._rule(model_wrapper)
            )
