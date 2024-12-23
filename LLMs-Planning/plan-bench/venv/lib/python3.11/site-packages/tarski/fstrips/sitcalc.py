import itertools
from collections import defaultdict

from tarski import Predicate, Function
from . import AddEffect, DelEffect, FunctionalEffect


class AtomBinding:
    def __init__(self, schema, point, condition):
        self.schema = schema
        self.point = point
        self.condition = condition

    def __repr__(self):
        return f"<{self.schema.ident()}, {self.point}, {self.condition}>"
    __str__ = __repr__


class TermBinding:
    def __init__(self, schema, point, value, condition):
        self.schema = schema
        self.point = point
        self.value = value
        self.condition = condition

    def __repr__(self):
        return f"<{self.schema.ident()}, {self.point}, {self.value}, {self.condition}>"
    __str__ = __repr__


class ExpressionIndex:
    def __init__(self, problem):
        self.problem = problem
        self.adds = defaultdict(list)
        self.dels = defaultdict(list)
        self.funcs = defaultdict(list)
        self.index(problem)

        # Some simple sanity checks
        assert all(isinstance(x, Predicate) for x in itertools.chain(self.adds, self.dels))
        assert all(isinstance(x, Function) for x in self.funcs)

    def index(self, problem):
        for aname, a in problem.actions.items():
            index_action(a, self.adds, self.dels, self.funcs)


def index_action(a, adds, dels, funcs):
    for effect in a.effects:
        if isinstance(effect, (AddEffect, DelEffect)):
            dest = adds if isinstance(effect, AddEffect) else dels
            atom = effect.atom
            dest[atom.symbol].append(AtomBinding(a, atom.subterms, effect.condition))

        elif isinstance(effect, FunctionalEffect):
            funcs[effect.lhs.symbol].append(TermBinding(a, effect.lhs.subterms, effect.rhs, effect.condition))

        else:
            raise RuntimeError(f'Effect "{effect}" of type "{type(effect)}" cannot be analysed')


class Sitcalc:
    def __init__(self, problem):
        self.problem = problem
        self.index = ExpressionIndex(problem)

    def generate_frame_axioms(self):
        # Let's start with predicates
        x = 0
        # TODO: Functions
