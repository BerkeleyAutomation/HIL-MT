import dishes
import pyramid


def catalog(name):
    return {
        'dishes': dishes.DishesEnv,
        'pyramid': pyramid.PyramidEnv,
    }[name]()
