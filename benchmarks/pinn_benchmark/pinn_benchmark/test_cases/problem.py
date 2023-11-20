class Problem:
    def __init__(self, domain, boundary_conditions, source_term, exact_solution=None):
        self.domain = domain
        self.boundary_conditions = boundary_conditions
        self.source_term = source_term
        self.exact_solution = exact_solution
