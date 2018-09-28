def encode(x, l):
    return [y == x for y in l]


def one_hot(i, l):
    assert i < l, (i, l)
    res = [0.] * l
    res[i] = 1.
    return res


def pad(x, l):
    if x is None:
        return [0.] * l
    assert len(x) <= l, (len(x), l, x)
    return x + ([0.] * (l - len(x)))
