from typing import Callable

from pipeline_rto_v2.core.optimizer.ortools.ortools_objects.constraint import (
    IndexedORStandardConst,
    ScalarORStandardConst,
)
from pipeline_rto_v2.core.optimizer.ortools.ortools_objects.indexed_set import (
    IndexedORSet,
)
from pipeline_rto_v2.core.optimizer.ortools.ortools_objects.model import ORToolsCPModel
from pipeline_rto_v2.core.optimizer.ortools.ortools_objects.objective import ORObjective
from pipeline_rto_v2.core.optimizer.ortools.ortools_objects.param import (
    IndexedORParam,
    ScalarORParam,
)
from pipeline_rto_v2.core.optimizer.ortools.ortools_objects.set import ORSet
from pipeline_rto_v2.core.optimizer.ortools.ortools_objects.var import (
    IndexedORBoolVariable,
    IndexedORContinuousVariable,
    ScalarORBoolVariable,
    ScalarORContinuousVariable,
)


class ComponentFactory:
    """Factory class to retrieve components for the ORToolsCPModel class. Will be used by wrapper architecture to create and add components to the model via decorators"""

    _register = {
        "ScalarORStandardConst": {
            "component_object": ScalarORStandardConst,
            "model_prefix": "c_",
        },
        "IndexedORStandardConst": {
            "component_object": IndexedORStandardConst,
            "model_prefix": "c_",
        },
        "IndexedORParam": {"component_object": IndexedORParam, "model_prefix": "p_"},
        "ScalarORParam": {"component_object": ScalarORParam, "model_prefix": "p_"},
        "ORSet": {"component_object": ORSet, "model_prefix": "s_"},
        "IndexedORSet": {"component_object": IndexedORSet, "model_prefix": "s_"},
        "IndexedORBoolVariable": {
            "component_object": IndexedORBoolVariable,
            "model_prefix": "v_",
        },
        "IndexedORContinuousVariable": {
            "component_object": IndexedORContinuousVariable,
            "model_prefix": "v_",
        },
        "ScalarORBoolVariable": {
            "component_object": ScalarORBoolVariable,
            "model_prefix": "v_",
        },
        "ScalarORContinuousVariable": {
            "component_object": ScalarORContinuousVariable,
            "model_prefix": "v_",
        },
        "ORObjective": {"component_object": ORObjective, "model_prefix": "o_"},
        "ORToolsCPModel": {"component_object": ORToolsCPModel, "model_prefix": "m_"},
    }

    def __contains__(cls, name: str) -> bool:
        return str(name) in cls._register.keys()

    @classmethod
    def retrieve_component(cls, name: str) -> Callable:
        if name in cls._register:
            return cls._register[name]


# create get attr of model
# Look for val in a component factory
# Call component decorator, which passes in the model object and the component object reference
# Pass *args and **kwargs to the call dunder of the specific compoennt decorator. This decorator will wrap the decorator that accesses the actual rule
# Call generic decorator, which passes in component, model object, and all *args and **kwargs
# Generic decorator call then takes in rule and sets the model attribute with the name of the rule, creates the args, and adds kwds
# If having trouble, reference: https://github.com/Pyomo/pyomo/blob/main/pyomo/core/base/block.py#L85
