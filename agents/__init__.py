import dishes
import pyramid


def catalog(config):
    return {
        'dishes': dishes.DishesAgent,
        'pyramid': pyramid.PyramidAgent,
    }[config.domain_name](config)
