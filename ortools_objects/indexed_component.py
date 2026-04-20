
from ortools_objects.component import ORComponent
from ortools_objects.set import ORSet


def process_set(arg):
    """Takes in any arg and makes sure that it is an ORSet. If not, raise an error."""
    if isinstance(arg, ORSet):
        return arg
    else:
        raise TypeError(
            "Argument passed to indexed component cannot be anything other than ORSet"
        )


class IndexedComponent(ORComponent):
    """Extends the base ORComponent object to allow for indices. All indexed parameter, set, and variable objects extend this one, while
    scalar objects extend the base component class directly. The way this is meant to work is to provide any number of sets as the argument array,
    and to provide other arguments (name, doc, etc.) as kwds.

    All indexed components contain a dictionary as their data attribute with indices as the keys in the form of a single elemnt or tuple, and a value as the key.

    The increased complexity comes from the _index_set attribute. If there are no arguments, the index set should not exist. If there is 1 argument, the
    init dunder will make sure the arg is a set and assign it as the index set. Because index sets have a getitem dunder, it is then accessable via index.
    In addition, for sets, since the dunder for iter is defined, you can iterate through the items/data in an indexed component using the index set indices.

    This should never be constructed directly. Instead, it should be extended.
    """

    def __init__(self, *args, **kwds):
        # Initialize the name, etc.
        ORComponent.__init__(self, **kwds)

        # Create local data attribute
        self._data = {}

        # Test and store the indexed set
        if len(args) == 0:
            self._index_set = None
            raise ValueError(
                "No index sets detected in arguments. If creating a scalar component, use respective scalar class instead of indexed class."
            )
        elif len(args) == 1:
            self._index_set = process_set(args[0])
            self._index_name = args[0]._name
        else:
            tmp = [process_set(arg) for arg in args]
            self._index_set = tmp[0].cross(*tmp[1:])
            self._index_name = tuple(arg._name for arg in args)

    def __len__(self):
        """Finds the size of an indexed component. Suggested use is to find impact of different components of a math model.

        Returns:
            int: Number of items in the data attribute of the current indexed object.
        """
        return len(self._data)

    def __contains__(self, idx: str | tuple) -> bool:
        """Overrides original contains dunder to find out whether the index is in the data object.

        Args:
            idx (str | tuple): Takes in a string or tuple index value to lookup in the indexed component dict.

        Returns:
            bool: Returns a bool indicating whether or not the index is in the data set keys
        """
        return idx in self._data

    def __getitem__(self, index: str | tuple) -> float:
        """Get the value at a given index. If the index is not there, use the get item when not present (can either raise error or return an indexed component's default value)

        Args:
            index (str | tuple): The index value of the array for which the data of an indexed component should be accessed.

        Returns:
            str | list | tuple | Number: Value of the indexed component at the index.
        """
        value = self._data.get(index)

        if value is None:
            return self._getitem_when_not_present(index)
        else:
            return value

    def __iter__(self):
        return self.keys()

    def keys(self):
        if self._data.__class__ is not dict:
            pass
        else:
            return iter(self._index_set)

    def values(self):
        return map(self.__getitem__, self.keys())

    def items(self):
        return self._data

    def _getitem_when_not_present(self, index: str | tuple):
        """Method to retrieve the value of an item in the indexed component when the index does not exist.
        Here, it raises a value error, but in subclasses, this method is overriden to take the default value
        of a component.

        Args:
            index (str | tuple): The index value of the array for which the data of an indexed component should be accessed.

        Raises:
            ValueError: Notifies programmer that the index cannot be found
        """
        raise KeyError(
            f"Index {index} is not found for {self.ctype} object named {self._name}. Either add to initialize or skip construction explicitly."
        )

    def _validate_data_indices(self, dict: dict):
        """Takes in the data passed in the initialize kwds in all subclasses and makes sure the keys are in the
        index set. Otherwise, raise a KeyError to indicate that the key does not exist in the index set created

        Args:
            dict (dict): Dictionary of data

        Raises:
            KeyError: One of the keys is not present in the index set of the indexed component.
        """
        for key in dict.keys():
            if key not in self._index_set:
                raise KeyError(
                    f"Key {key} from initialize is not present in set values passed to parameter constructor."
                )

    def _validate_args(self, args: tuple):
        """Checks to make sure all args are ORSets.

        Raises:
            TypeError: Raised if one of the args is not an ORSet object.
        """
        for arg in args:
            if not isinstance(arg, ORSet):
                raise TypeError(
                    f"Argument {str(arg)} passed to indexed OR parameter is not an ORSet object"
                )

    def __call__(self, index):
        return self._data[index]

    def dim(self):
        return len(self._index_set[0]) if isinstance(self._index_set[0], tuple) else 1
