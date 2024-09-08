from sympy import Symbol


class CustomUse(Symbol):
    """Custom symbol implementation of Sympy symbol. Modified New such that construction can have a more
    complex symbolic __repr__. Also creates methods such that all subclasses must either implement or extend.
    Used to represent power use in cost expression


    Returns:
        Symbol: Creates new symbol
    """

    def __new__(cls, *args):
        obj = Symbol.__new__(cls, str(args))
        obj.__init__(*args)
        obj.construction_args = args
        return obj

    # When stringifying a complex expression, the expression will ask the symbol (via _sympystr) how to represent itself.
    def _sympystr(self, a):
        return self.__repr__()


    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        return f"{self.__class__.__name__}{self.construction_args}"


class PowerUse(CustomUse):
    def __init__(self, period: int, site_name: str):
        self.site_name = site_name
        self.period = period


class DraUse(CustomUse):
    def __init__(self, period: int, site_name: str):
        self.site_name = site_name
        self.period = period
        

class ConfigurationChange(CustomUse):
    def __init__(self, period: int, site_name: str):
        self.site_name = site_name
        self.period = period
