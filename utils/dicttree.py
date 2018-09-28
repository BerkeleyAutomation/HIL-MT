import collections
import json
import operator


class DictTree(collections.Mapping):
    """Nested dict.

    Values can be accessed as attributes as well as items:
    >>> d = DictTree({'a': 1, 'sub': {'x': 10}}, b=2, **{'c': 3})
    >>> d.a
    1
    >>> d['b']
    2
    >>> d.e = 5
    >>> d['f'] = 6
    >>> d.sub.x
    10
    >>> d.sub.y = 11
    >>> d[['sub', 'y']]
    11
    >>> d[['sub', 'z']] = 12
    >>> d |= {'g': 7}
    >>> d += {'a': 100}
    >>> d + DictTree(sub=DictTree(z=200))
    DictTree(a=101, sub=DictTree(x=10, y=11, z=212), b=2, c=3, e=5, f=6, g=7)

    @DynamicAttrs
    """

    def __init__(self, *maps, **attr):
        """

        Args:
            *maps (Mapping): Deep copy values from these mappings.
            **attr: Extra attributes.
        """
        super(DictTree, self).__init__()
        for m in maps:
            for k, v in m.items():
                if isinstance(v, collections.Mapping):
                    self[k] = DictTree(v)
                else:
                    self[k] = v
        self.update(attr)

    def __getitem__(self, k):
        """

        Args:
            k: Item key. May be list for deep access.

        Returns:
            Item value.
        """
        if isinstance(k, list):
            if len(k) == 0:
                return self
            elif len(k) == 1:
                return self[k[0]]
            else:
                return self[k[0]][k[1:]]
        else:
            return vars(self)[k]

    def setdefault(self, k, v):
        """

        Args:
            k: Item key. May be list for deep access, which creates nested `DictTree`s as needed.
            v: Item value. This value is set if item doesn't exist.

        Returns:
            Item value.

        Raises:
            KeyError: Item key is reserved.
        """
        if isinstance(k, list):
            if len(k) == 0:
                return self
            elif len(k) == 1:
                return self.setdefault(k[0], v)
            else:
                return self.setdefault(k[0], DictTree()).setdefault(k[1:], v)
        elif k in FORBIDDEN:
            raise KeyError('key "{}" is reserved'.format(k))
        else:
            return vars(self).setdefault(k, v)

    def __setitem__(self, k, v):
        """

        Args:
            k: Item key. May be list for deep access, which creates nested `DictTree`s as needed.
            v: Item value.

        Raises:
            KeyError: Item key is empty list or reserved.
        """
        if isinstance(k, list):
            if len(k) == 0:
                raise KeyError("cannot assign to root")
            elif len(k) == 1:
                self[k[0]] = v
            else:
                self.setdefault(k[0], DictTree())[k[1:]] = v
        elif k in FORBIDDEN:
            raise KeyError('key "{}" is reserved'.format(k))
        else:
            vars(self)[k] = v

    def __setattr__(self, k, v):
        self[k] = v

    def __delitem__(self, k):
        """

        Args:
            k: Item key. May be list for deep access.
        """
        if isinstance(k, list):
            if len(k) == 0:
                raise KeyError("cannot delete root")
            elif len(k) == 1:
                self.__delitem__(k[0])
            else:
                self[k[0]].__delitem__(k[1:])
        else:
            vars(self).__delitem__(k)

    def __len__(self):
        """

        Returns:
            int: Number of top-level elements.
        """
        return len(vars(self))

    def __str__(self):
        return ", ".join("{}={}".format(k, repr(v)) for k, v in self.items())

    def __repr__(self):
        return 'DictTree({})'.format(str(self))

    class JSONEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, DictTree):
                return vars(o)
            else:
                return super(DictTree.JSONEncoder, self).default(o)

    class JSONDecoder(json.JSONDecoder):
        def __init__(self, *args, **kwargs):
            super(DictTree.JSONDecoder, self).__init__(object_hook=self.object_hook, *args, **kwargs)

        @staticmethod
        def object_hook(d):
            return DictTree(d)

    def __contains__(self, k):
        """

        Args:
            k: Item key. May be list for deep access.

        Returns:
            bool: Key existence.
        """
        if isinstance(k, list):
            if len(k) == 0:
                return True
            elif len(k) == 1:
                return k[0] in self
            else:
                if k[0] in self:
                    return k[1:] in self[k[0]]
                else:
                    return False
        else:
            return k in vars(self)

    def __iter__(self):
        """

        Returns:
            Iterator: Shallow iterator over keys.
        """
        return iter(vars(self))

    def keys(self):
        """

        Returns:
            Iterator: Shallow iterator over items.
        """
        return vars(self).keys()

    def items(self):
        """

        Returns:
            Iterator: Shallow iterator over items.
        """
        return vars(self).items()

    def values(self):
        """

        Returns:
            Iterator: Shallow iterator over values.
        """
        return vars(self).values()

    def allkeys(self):
        """

        Returns:
            Iterator: Deep iterator over keys.
        """
        for k, v in self.items():
            if isinstance(v, DictTree):
                for k_ in v.allkeys():
                    yield [k] + k_
            else:
                yield [k]

    def allitems(self):
        """

        Returns:
            Iterator: Deep iterator over items.
        """
        for k, v in self.items():
            if isinstance(v, DictTree):
                for k_, v_ in v.allitems():
                    yield [k] + k_, v_
            else:
                yield [k], v

    def allvalues(self):
        """

        Returns:
            Iterator: Deep iterator over values.
        """
        for v in self.values():
            if isinstance(v, DictTree):
                for v_ in v.allvalues():
                    yield v_
            else:
                yield v

    def copy(self):
        """

        Returns:
            DictTree: Deep copy of `self`.
        """
        res = DictTree()
        for k, v in self.items():
            if isinstance(v, DictTree):
                res[k] = v.copy()
            else:
                res[k] = v
        return res

    def update(self, d, op=None):
        """

        Args:
            d (Mapping): Mapping from which to update.
            op ((Any, Any) -> Any): Update operator, such that `x = op(x, y)` updates value `x` with new value `y`.
                Default: replace current value, `x = y`.

        Returns:
            DictTree: `self` after update.
        """
        for k, v in d.items():
            if k in self and isinstance(self[k], DictTree) and isinstance(v, collections.Mapping):
                self[k].update(v, op)
            elif op is None:
                self[k] = v
            else:
                self[k] = op(self[k], v)
        return self

    def __ior__(self, d):
        return self.update(d)

    def __or__(self, d):
        return self.copy().update(d)

    def __iadd__(self, d):
        return self.update(d, operator.iadd)

    def __add__(self, d):
        return self.copy().update(d, operator.iadd)


FORBIDDEN = set(dir(DictTree()))
