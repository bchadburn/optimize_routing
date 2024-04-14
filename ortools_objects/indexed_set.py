from ortools_objects.indexed_component import (
    IndexedComponent,
)


class IndexedORSet(IndexedComponent):
    """Mainly used by constraints to iterate over certain portions of sets for certain indices. Should
    not be used as an indexed component initializer (which will throw an error).

    Args:
        Number of ORSets for which the set should be indexed.

    Kwargs:
        name (str): Name of the set
        doc (str): Docstring of the set
        initialize (dict): Dictionary with keys of index set and values of set.

    Example Use: In preprocessing, a series of piecewise segments are generated for each site and time period
    in a set of sites and time periods. However, the number of piecewise segments for each site/time period
    is different. To sum over all piecewise segments for a given site/time period, the developer will want to
    have an implicit summation. This can be accomplished through the following:

    model.example_set = IndexedORSet(name="foo", doc="foo", initialize={(0, 'site1'): [0, 1, 2], (0, 'site2'): [0, 1]})

    def example_rule(model, time_period, site):
        return v_ActualProdProduced[time_period, site] == sum(v_PiecewiseProducts[time_period, site, piecewise_idx] for piecewise_idx in example_set[time_period, site])

    In the above function, the indexed set allows the constraint to sum over uneven numbers of piecewise indices for different sets. It should be noted that
    there are other ways to accomplish this, but the the indexed set makes summations like this easier to write and comprehend.

    """

    def __init__(self, *args, **kwds):
        kwds.setdefault("ctype", "set")

        # self._validate_args(args)

        self._initialize = kwds.pop("initialize")

        IndexedComponent.__init__(self, *args, **kwds)
        if isinstance(self._initialize, dict):
            self._data = self._initialize
        else:
            raise TypeError(
                f"Indexed ORSet {self._name} initialize keyword must contain dictionary"
            )

    def _validate_args(self, args):
        IndexedComponent._validate_args(args)
