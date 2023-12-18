'''Classes for Variable Elimination Routines
   A) class Variable

      This class allows one to define Bayes Net variables.

      On initialization the variable object can be given a name and a
      domain of values. This list of domain values can be added to or
      deleted from in support of an incremental specification of the
      variable domain.

      The variable also has a set and get value method. These set a
      value for the variable that can be used by the factor class.

    B) class Factor

      This class allows one to define a factor specified by a table
      of values.

      On initialization the variables in the factor is
      specified. These variable must be in a list. This list of
      variables cannot be changed once the constraint object is
      created.

      Once created the factor can be incrementally initialized with a
      list of values. To interact with the factor object one first
      sets the value of each variable in its scope (using the
      variable's set_value method), then one can set or get the value
      of the factor (a number) on those fixed values of the variables
      in its scope.

      Initially, one creates a factor object for every conditional
      probability table in a bayes-net. Then one initializes the
      factor by iteratively setting the values of the factor's
      variables and then adding the factor's numeric value using the
      add_value method.

    C) class BN
       This class allows one to put factors and variables together to form a Bayes net.
       It serves as a convenient place to store factors and variables associated
       with a Bayes Net in one place. It also has some utility routines to, e.g,., find
       factors a variable is involved in.

    D) Several methods to support VE have been included as well.  These include:
    multiply_factors, restrict_factor, sum_out_variable, normalize
    '''


class Variable:
    '''Class for defining Bayes Net variables. '''

    def __init__(self, name, domain=[]):
        '''Create a variable object, specifying its name (a
        string). Optionally specify the initial domain.
        '''
        self.name = name  # text name for variable
        self.dom = list(domain)  # Make a copy of passed domain
        self.evidence_index = -1  # evidence value (stored as index into self.dom)
        self.assignment_index = -1  # For use by factors. We can assign variables values
        # and these assigned values can be used by factors
        # to index into their tables.

    def reset_assignment(self):
        self.assignment_index = -1  # For use by factors. We can assign variables values

    def reset_evidence(self):
        self.reset_evidence = -1  # For use by factors. We can assign variables values

    def add_domain_values(self, values):
        '''Add domain values to the domain. values should be a list.'''
        for val in values: self.dom.append(val)

    def value_index(self, value):
        '''Domain values need not be numbers, so return the index
           in the domain list of a variable value'''
        return self.dom.index(value)

    def domain_size(self):
        '''Return the size of the domain'''
        return (len(self.dom))

    def domain(self):
        '''return the variable domain'''
        return (list(self.dom))

    def set_evidence(self, val):
        '''set this variable's value when it operates as evidence'''
        self.evidence_index = self.value_index(val)

    def get_evidence(self):
        #print(f"evidence index {self.evidence_index}")
        if self.evidence_index >= 0:
            return (self.dom[self.evidence_index])
        else:
            return None

    def set_assignment(self, val):
        '''Set this variable's assignment value for factor lookups'''
        self.assignment_index = self.value_index(val)

    def get_assignment(self):
        #print(f"assignment index {self.assignment_index}")
        if self.assignment_index >= 0:
            return (self.dom[self.assignment_index])
        else:
            return None

    ##These routines are special low-level routines used directly by the
    ##factor objects
    def set_assignment_index(self, index):
        '''This routine is used by the factor objects'''
        self.assignment_index = index

    def get_assignment_index(self):
        '''This routine is used by the factor objects'''
        return (self.assignment_index)

    def __repr__(self):
        '''string to return when evaluating the object'''
        return ("{}".format(self.name))

    def __str__(self):
        '''more elaborate string for printing'''
        return ("{}, Dom = {}".format(self.name, self.dom))


class Factor:
    '''Class for defining factors. A factor is a function that is over
    an ORDERED sequence of variables called its scope. It maps every
    assignment of values to these variables to a number. In a Bayes
    Net every CPT is represented as a factor. Pr(A|B,C) for example
    will be represented by a factor over the variables (A,B,C). If we
    assign A = a, B = b, and C = c, then the factor will map this
    assignment, A=a, B=b, C=c, to a number that is equal to Pr(A=a|
    B=b, C=c). During variable elimination new factors will be
    generated. However, the factors computed during variable
    elimination do not necessarily correspond to conditional
    probabilities. Nevertheless, they still map assignments of values
    to the variables in their scope to numbers.

    Note that if the factor's scope is empty it is a constraint factor
    that stores only one value. add_values would be passed something
    like [[0.25]] to set the factor's single value. The get_value
    functions will still work.  E.g., get_value([]) will return the
    factor's single value. Constraint factors might be created when a
    factor is restricted.'''

    def __init__(self, name, scope):
        '''create a Factor object, specify the Factor name (a string)
        and its scope (an ORDERED list of variable objects).'''
        self.scope = list(scope)
        self.name = name
        size = 1
        for v in scope:
            size = size * v.domain_size()
        self.values = [0] * size  # initialize values to be long list of zeros.

    def get_scope(self):
        '''returns copy of scope...you can modify this copy without affecting
           the factor object'''
        return list(self.scope)

    def add_values(self, values):
        '''This routine can be used to initialize the factor. We pass
        it a list of lists. Each sublist is a ORDERED sequence of
        values, one for each variable in self.scope followed by a
        number that is the factor's value when its variables are
        assigned these values. For example, if self.scope = [A, B, C],
        and A.domain() = [1,2,3], B.domain() = ['a', 'b'], and
        C.domain() = ['heavy', 'light'], then we could pass add_values the
        following list of lists
        [[1, 'a', 'heavy', 0.25], [1, 'a', 'light', 1.90],
         [1, 'b', 'heavy', 0.50], [1, 'b', 'light', 0.80],
         [2, 'a', 'heavy', 0.75], [2, 'a', 'light', 0.45],
         [2, 'b', 'heavy', 0.99], [2, 'b', 'light', 2.25],
         [3, 'a', 'heavy', 0.90], [3, 'a', 'light', 0.111],
         [3, 'b', 'heavy', 0.01], [3, 'b', 'light', 0.1]]

         This list initializes the factor so that, e.g., its value on
         (A=2,B=b,C='light) is 2.25'''

        for t in values:
            index = 0
            for v in self.scope:
                index = index * v.domain_size() + v.value_index(t[0])
                t = t[1:]
            self.values[index] = t[0]

    def add_value_at_current_assignment(self, number):

        '''This function allows adding values to the factor in a way
        that will often be more convenient. We pass it only a single
        number. It then looks at the assigned values of the variables
        in its scope and initializes the factor to have value equal to
        number on the current assignment of its variables. Hence, to
        use this function one first must set the current values of the
        variables in its scope.

        For example, if self.scope = [A, B, C],
        and A.domain() = [1,2,3], B.domain() = ['a', 'b'], and
        C.domain() = ['heavy', 'light'], and we first set an assignment for A, B
        and C:
        A.set_assignment(1)
        B.set_assignment('a')
        C.set_assignment('heavy')
        then we call
        add_value_at_current_assignment(0.33)
         with the value 0.33, we would have initialized this factor to have
        the value 0.33 on the assigments (A=1, B='1', C='heavy')
        This has the same effect as the call
        add_values([1, 'a', 'heavy', 0.33])

        One advantage of the current_assignment interface to factor values is that
        we don't have to worry about the order of the variables in the factor's
        scope. add_values on the other hand has to be given tuples of values where
        the values must be given in the same order as the variables in the factor's
        scope.

        See recursive_print_values called by print_table to see an example of
        where the current_assignment interface to the factor values comes in handy.
        '''

        index = 0
        for v in self.scope:
            index = index * v.domain_size() + v.get_assignment_index()
        self.values[index] = number

    def get_values(self):
        return self.values

    def get_value(self, variable_values):

        '''This function is used to retrieve a value from the
        factor. We pass it an ordered list of values, one for every
        variable in self.scope. It then returns the factor's value on
        that set of assignments.  For example, if self.scope = [A, B,
        C], and A.domain() = [1,2,3], B.domain() = ['a', 'b'], and
        C.domain() = ['heavy', 'light'], and we invoke this function
        on the list [1, 'b', 'heavy'] we would get a return value
        equal to the value of this factor on the assignment (A=1,
        B='b', C='light')'''

        index = 0
        for v in self.scope:
            index = index * v.domain_size() + v.value_index(variable_values[0])
            variable_values = variable_values[1:]
        return self.values[index]

    def get_value_at_current_assignments(self):

        '''This function is used to retrieve a value from the
        factor. The value retrieved is the value of the factor when
        evaluated at the current assignment to the variables in its
        scope.

        For example, if self.scope = [A, B, C], and A.domain() =
        [1,2,3], B.domain() = ['a', 'b'], and C.domain() = ['heavy',
        'light'], and we had previously invoked A.set_assignment(1),
        B.set_assignment('a') and C.set_assignment('heavy'), then this
        function would return the value of the factor on the
        assigments (A=1, B='1', C='heavy')'''
        index = 0
        for v in self.scope:
            index = index * v.domain_size() + v.get_assignment_index()
        return self.values[index]

    def print_table(self):
        '''print the factor's table'''
        saved_values = []  # save and then restore the variable assigned values.
        for v in self.scope:
            saved_values.append(v.get_assignment_index())

        self.recursive_print_values(self.scope)

        for v in self.scope:
            v.set_assignment_index(saved_values[0])
            saved_values = saved_values[1:]

    def recursive_print_values(self, vars):
        if len(vars) == 0:
            print("[", end=""),
            for v in self.scope:
                print("{} = {},".format(v.name, v.get_assignment()), end="")
            print("] = {}".format(self.get_value_at_current_assignments()))
        else:
            for val in vars[0].domain():
                vars[0].set_assignment(val)
                self.recursive_print_values(vars[1:])

    def __repr__(self):
        return ("{}".format(self.name))


class BN:
    '''Class for defining a Bayes Net.
       This class is simple, it just is a wrapper for a list of factors. And it also
       keeps track of all variables in the scopes of these factors'''

    def __init__(self, name, Vars, Factors):
        self.name = name
        self.Variables = list(Vars)
        self.Factors = list(Factors)
        for f in self.Factors:
            for v in f.get_scope():
                if not v in self.Variables:
                    print("Bayes net initialization error")
                    print("Factor scope {} has variable {} that", end='')
                    print(" does not appear in list of variables {}.".format(list(map(lambda x: x.name, f.get_scope())),
                                                                         v.name, list(map(lambda x: x.name, Vars))))

    def factors(self):
        return list(self.Factors)

    def variables(self):
        return list(self.Variables)

    def get_variable(self, name):
        for v in list(self.Variables):
            if v.name == name:
                return v
        return None

    def reset_variables(self):
        for v in list(self.Variables):
            v.evidence_index = -1
            v.assignment_index = -1


def adultDatasetBN():

    ms = Variable("MaritalStatus", ['Not-Married', 'Married', 'Separated', 'Widowed'])
    F0 = Factor("P(ms)", [ms])
    values = [['Not-Married', 0.33], ['Married', 0.47], ['Separated', 0.17], ['Widowed', 0.03]]
    F0.add_values(values)

    re = Variable("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    F1 = Factor("P(re|ms)", [re,ms])
    values = [['Own-child', 'Not-Married', 0.39], ['Not-in-family', 'Not-Married', 0.46], ['Other-relative', 'Not-Married', 0.06], ['Unmarried', 'Not-Married', 0.09], ['Wife', 'Not-Married', 0.0], ['Husband', 'Not-Married', 0.0], ['Wife', 'Married', 0.1], ['Own-child', 'Married', 0.01], ['Husband', 'Married', 0.88], ['Not-in-family', 'Married', 0.0], ['Other-relative', 'Married', 0.01], ['Unmarried', 'Married', 0.0], ['Own-child', 'Separated', 0.08], ['Not-in-family', 'Separated', 0.51], ['Other-relative', 'Separated', 0.03], ['Unmarried', 'Separated', 0.38], ['Wife', 'Separated', 0.0], ['Husband', 'Separated', 0.0], ['Own-child', 'Widowed', 0.01], ['Not-in-family', 'Widowed', 0.52], ['Other-relative', 'Widowed', 0.05], ['Unmarried', 'Widowed', 0.41], ['Wife', 'Widowed', 0.0], ['Husband', 'Widowed', 0.0]]
    F1.add_values(values)

    rc = Variable("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    F2 = Factor("P(rc|re)", [rc, re])
    values = [['White', 'Wife', 0.85], ['Black', 'Wife', 0.09], ['Asian-Pac-Islander', 'Wife', 0.04], ['Amer-Indian-Eskimo', 'Wife', 0.01], ['Other', 'Wife', 0.01], ['White', 'Own-child', 0.84], ['Black', 'Own-child', 0.11], ['Asian-Pac-Islander', 'Own-child', 0.03], ['Amer-Indian-Eskimo', 'Own-child', 0.01], ['Other', 'Own-child', 0.01], ['White', 'Husband', 0.91], ['Black', 'Husband', 0.05], ['Asian-Pac-Islander', 'Husband', 0.03], ['Amer-Indian-Eskimo', 'Husband', 0.01], ['Other', 'Husband', 0.01], ['White', 'Not-in-family', 0.86], ['Black', 'Not-in-family', 0.09], ['Asian-Pac-Islander', 'Not-in-family', 0.02], ['Amer-Indian-Eskimo', 'Not-in-family', 0.01], ['Other', 'Not-in-family', 0.01], ['White', 'Other-relative', 0.72], ['Black', 'Other-relative', 0.16], ['Asian-Pac-Islander', 'Other-relative', 0.08], ['Amer-Indian-Eskimo', 'Other-relative', 0.01], ['Other', 'Other-relative', 0.03], ['White', 'Unmarried', 0.73], ['Black', 'Unmarried', 0.22], ['Asian-Pac-Islander', 'Unmarried', 0.03], ['Amer-Indian-Eskimo', 'Unmarried', 0.02], ['Other', 'Unmarried', 0.01]]
    F2.add_values(values)

    ge = Variable("Gender", ['Male', 'Female'])
    F3 = Factor("P(ge|re,ms)", [ge, re, ms])
    values = [['Male', 'Wife', 'Married', 0.0], ['Female', 'Wife', 'Married', 1.0], ['Male', 'Own-child', 'Not-Married', 0.57], ['Female', 'Own-child', 'Not-Married', 0.43], ['Male', 'Own-child', 'Married', 0.58], ['Female', 'Own-child', 'Married', 0.42], ['Male', 'Own-child', 'Separated', 0.51], ['Female', 'Own-child', 'Separated', 0.49], ['Male', 'Own-child', 'Widowed', 0.08], ['Female', 'Own-child', 'Widowed', 0.92], ['Male', 'Husband', 'Married', 1.0], ['Female', 'Husband', 'Married', 0.0], ['Male', 'Not-in-family', 'Not-Married', 0.58], ['Female', 'Not-in-family', 'Not-Married', 0.42], ['Male', 'Not-in-family', 'Married', 0.86], ['Female', 'Not-in-family', 'Married', 0.14], ['Male', 'Not-in-family', 'Separated', 0.53], ['Female', 'Not-in-family', 'Separated', 0.47], ['Male', 'Not-in-family', 'Widowed', 0.18], ['Female', 'Not-in-family', 'Widowed', 0.82], ['Male', 'Other-relative', 'Not-Married', 0.63], ['Female', 'Other-relative', 'Not-Married', 0.37], ['Male', 'Other-relative', 'Married', 0.59], ['Female', 'Other-relative', 'Married', 0.41], ['Male', 'Other-relative', 'Separated', 0.4], ['Female', 'Other-relative', 'Separated', 0.6], ['Male', 'Other-relative', 'Widowed', 0.15], ['Female', 'Other-relative', 'Widowed', 0.85], ['Male', 'Unmarried', 'Not-Married', 0.34], ['Female', 'Unmarried', 'Not-Married', 0.66], ['Male', 'Unmarried', 'Separated', 0.19], ['Female', 'Unmarried', 'Separated', 0.81], ['Male', 'Unmarried', 'Widowed', 0.17], ['Female', 'Unmarried', 'Widowed', 0.83]]
    F3.add_values(values)

    oc = Variable("Occupation", ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'])
    F4 = Factor("P(oc|ge, re)", [oc, ge, re])
    values = [['Office Labour', 'Male', 'Wife', 1.0], ['Admin', 'Male', 'Wife', 0.0], ['Military', 'Male', 'Wife', 0.0], ['Manual Labour', 'Male', 'Wife', 0.0], ['Service', 'Male', 'Wife', 0.0], ['Professional', 'Male', 'Wife', 0.0], ['Admin', 'Male', 'Own-child', 0.09], ['Military', 'Male', 'Own-child', 0.02], ['Manual Labour', 'Male', 'Own-child', 0.47], ['Office Labour', 'Male', 'Own-child', 0.18], ['Service', 'Male', 'Own-child', 0.17], ['Professional', 'Male', 'Own-child', 0.05], ['Admin', 'Male', 'Husband', 0.05], ['Military', 'Male', 'Husband', 0.03], ['Manual Labour', 'Male', 'Husband', 0.42], ['Office Labour', 'Male', 'Husband', 0.32], ['Service', 'Male', 'Husband', 0.04], ['Professional', 'Male', 'Husband', 0.14], ['Admin', 'Male', 'Not-in-family', 0.07], ['Military', 'Male', 'Not-in-family', 0.03], ['Manual Labour', 'Male', 'Not-in-family', 0.41], ['Office Labour', 'Male', 'Not-in-family', 0.26], ['Service', 'Male', 'Not-in-family', 0.09], ['Professional', 'Male', 'Not-in-family', 0.14], ['Admin', 'Male', 'Other-relative', 0.07], ['Military', 'Male', 'Other-relative', 0.02], ['Manual Labour', 'Male', 'Other-relative', 0.52], ['Office Labour', 'Male', 'Other-relative', 0.16], ['Service', 'Male', 'Other-relative', 0.18], ['Professional', 'Male', 'Other-relative', 0.05], ['Admin', 'Male', 'Unmarried', 0.07], ['Military', 'Male', 'Unmarried', 0.02], ['Manual Labour', 'Male', 'Unmarried', 0.54], ['Office Labour', 'Male', 'Unmarried', 0.21], ['Service', 'Male', 'Unmarried', 0.07], ['Professional', 'Male', 'Unmarried', 0.09], ['Admin', 'Female', 'Wife', 0.25], ['Military', 'Female', 'Wife', 0.0], ['Manual Labour', 'Female', 'Wife', 0.11], ['Office Labour', 'Female', 'Wife', 0.29], ['Service', 'Female', 'Wife', 0.13], ['Professional', 'Female', 'Wife', 0.22], ['Admin', 'Female', 'Own-child', 0.27], ['Military', 'Female', 'Own-child', 0.01], ['Manual Labour', 'Female', 'Own-child', 0.1], ['Office Labour', 'Female', 'Own-child', 0.3], ['Service', 'Female', 'Own-child', 0.24], ['Professional', 'Female', 'Own-child', 0.09], ['Office Labour', 'Female', 'Husband', 1.0], ['Admin', 'Female', 'Husband', 0.0], ['Military', 'Female', 'Husband', 0.0], ['Manual Labour', 'Female', 'Husband', 0.0], ['Service', 'Female', 'Husband', 0.0], ['Professional', 'Female', 'Husband', 0.0], ['Admin', 'Female', 'Not-in-family', 0.24], ['Military', 'Female', 'Not-in-family', 0.01], ['Manual Labour', 'Female', 'Not-in-family', 0.1], ['Office Labour', 'Female', 'Not-in-family', 0.29], ['Service', 'Female', 'Not-in-family', 0.17], ['Professional', 'Female', 'Not-in-family', 0.19], ['Admin', 'Female', 'Other-relative', 0.24], ['Military', 'Female', 'Other-relative', 0.01], ['Manual Labour', 'Female', 'Other-relative', 0.16], ['Office Labour', 'Female', 'Other-relative', 0.23], ['Service', 'Female', 'Other-relative', 0.28], ['Professional', 'Female', 'Other-relative', 0.08], ['Admin', 'Female', 'Unmarried', 0.27], ['Military', 'Female', 'Unmarried', 0.01], ['Manual Labour', 'Female', 'Unmarried', 0.13], ['Office Labour', 'Female', 'Unmarried', 0.25], ['Service', 'Female', 'Unmarried', 0.21], ['Professional', 'Female', 'Unmarried', 0.13]]
    F4.add_values(values)

    co = Variable("Country", ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'])
    F5 = Factor("P(co|rc)", [co, rc])
    values = [['North-America', 'White', 0.94], ['South-America', 'White', 0.04], ['Europe', 'White', 0.02], ['Asia', 'White', 0.0], ['Middle-East', 'White', 0.0], ['Carribean', 'White', 0.0], ['North-America', 'Black', 0.93], ['South-America', 'Black', 0.01], ['Europe', 'Black', 0.0], ['Asia', 'Black', 0.0], ['Carribean', 'Black', 0.05], ['Middle-East', 'Black', 0.0], ['North-America', 'Asian-Pac-Islander', 0.32], ['South-America', 'Asian-Pac-Islander', 0.0], ['Europe', 'Asian-Pac-Islander', 0.09], ['Asia', 'Asian-Pac-Islander', 0.58], ['Middle-East', 'Asian-Pac-Islander', 0.01], ['Carribean', 'Asian-Pac-Islander', 0.0], ['North-America', 'Amer-Indian-Eskimo', 0.95], ['South-America', 'Amer-Indian-Eskimo', 0.03], ['Europe', 'Amer-Indian-Eskimo', 0.01], ['Asia', 'Amer-Indian-Eskimo', 0.01], ['Middle-East', 'Amer-Indian-Eskimo', 0.0], ['Carribean', 'Amer-Indian-Eskimo', 0.0], ['North-America', 'Other', 0.49], ['South-America', 'Other', 0.37], ['Europe', 'Other', 0.0], ['Asia', 'Other', 0.03], ['Middle-East', 'Other', 0.01], ['Carribean', 'Other', 0.1]]
    F5.add_values(values)

    ed = Variable("Education",  ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'])
    F6 = Factor("P(ed|oc)", [ed, oc])
    values = [['<Gr12', 'Admin', 0.07], ['HS-Graduate', 'Admin', 0.53], ['Associate', 'Admin', 0.15], ['Professional', 'Admin', 0.0], ['Bachelors', 'Admin', 0.21], ['Masters', 'Admin', 0.03], ['Doctorate', 'Admin', 0.0], ['<Gr12', 'Military', 0.08], ['HS-Graduate', 'Military', 0.46], ['Associate', 'Military', 0.19], ['Professional', 'Military', 0.0], ['Bachelors', 'Military', 0.23], ['Masters', 'Military', 0.04], ['Doctorate', 'Military', 0.0], ['<Gr12', 'Manual Labour', 0.41], ['HS-Graduate', 'Manual Labour', 0.36], ['Associate', 'Manual Labour', 0.12], ['Professional', 'Manual Labour', 0.0], ['Bachelors', 'Manual Labour', 0.09], ['Masters', 'Manual Labour', 0.01], ['Doctorate', 'Manual Labour', 0.0], ['<Gr12', 'Office Labour', 0.07], ['HS-Graduate', 'Office Labour', 0.33], ['Associate', 'Office Labour', 0.11], ['Professional', 'Office Labour', 0.01], ['Bachelors', 'Office Labour', 0.36], ['Masters', 'Office Labour', 0.1], ['Doctorate', 'Office Labour', 0.01], ['<Gr12', 'Service', 0.43], ['HS-Graduate', 'Service', 0.38], ['Associate', 'Service', 0.09], ['Professional', 'Service', 0.0], ['Bachelors', 'Service', 0.09], ['Masters', 'Service', 0.01], ['Doctorate', 'Service', 0.0], ['<Gr12', 'Professional', 0.01], ['HS-Graduate', 'Professional', 0.06], ['Associate', 'Professional', 0.04], ['Professional', 'Professional', 0.55], ['Bachelors', 'Professional', 0.19], ['Masters', 'Professional', 0.11], ['Doctorate', 'Professional', 0.04]]
    F6.add_values(values)

    wc = Variable("Work",  ['Not Working', 'Government', 'Private', 'Self-emp'])
    F7 = Factor("P(wc|oc)", [wc, oc])
    values = [['Not Working', 'Admin', 0.0], ['Government', 'Admin', 0.23], ['Private', 'Admin', 0.75], ['Self-emp', 'Admin', 0.02], ['Government', 'Military', 0.7], ['Private', 'Military', 0.28], ['Self-emp', 'Military', 0.02], ['Not Working', 'Military', 0.0], ['Not Working', 'Manual Labour', 0.0], ['Government', 'Manual Labour', 0.06], ['Private', 'Manual Labour', 0.81], ['Self-emp', 'Manual Labour', 0.13], ['Government', 'Office Labour', 0.09], ['Private', 'Office Labour', 0.74], ['Self-emp', 'Office Labour', 0.17], ['Not Working', 'Office Labour', 0.0], ['Not Working', 'Service', 0.0], ['Government', 'Service', 0.1], ['Private', 'Service', 0.84], ['Self-emp', 'Service', 0.06], ['Government', 'Professional', 0.31], ['Private', 'Professional', 0.56], ['Self-emp', 'Professional', 0.13], ['Not Working', 'Professional', 0.0]]
    F7.add_values(values)

    sa = Variable("Salary", ['0', '1'])
    F8 = Factor("P(sa|ed,re)", [sa, ed, re])
    values = [['0', '<Gr12', 'Wife', 0.89], ['1', '<Gr12', 'Wife', 0.11], ['0', '<Gr12', 'Own-child', 0.99], ['1', '<Gr12', 'Own-child', 0.01], ['0', '<Gr12', 'Husband', 0.87], ['1', '<Gr12', 'Husband', 0.13], ['0', '<Gr12', 'Not-in-family', 0.97], ['1', '<Gr12', 'Not-in-family', 0.03], ['0', '<Gr12', 'Other-relative', 0.99], ['1', '<Gr12', 'Other-relative', 0.01], ['0', '<Gr12', 'Unmarried', 0.97], ['1', '<Gr12', 'Unmarried', 0.03], ['0', 'HS-Graduate', 'Wife', 0.53], ['1', 'HS-Graduate', 'Wife', 0.47], ['0', 'HS-Graduate', 'Own-child', 0.99], ['1', 'HS-Graduate', 'Own-child', 0.01], ['0', 'HS-Graduate', 'Husband', 0.56], ['1', 'HS-Graduate', 'Husband', 0.44], ['0', 'HS-Graduate', 'Not-in-family', 0.93], ['1', 'HS-Graduate', 'Not-in-family', 0.07], ['0', 'HS-Graduate', 'Other-relative', 0.95], ['1', 'HS-Graduate', 'Other-relative', 0.05], ['0', 'HS-Graduate', 'Unmarried', 0.96], ['1', 'HS-Graduate', 'Unmarried', 0.04], ['0', 'Associate', 'Wife', 0.44], ['1', 'Associate', 'Wife', 0.56], ['0', 'Associate', 'Own-child', 0.99], ['1', 'Associate', 'Own-child', 0.01], ['0', 'Associate', 'Husband', 0.54], ['1', 'Associate', 'Husband', 0.46], ['0', 'Associate', 'Not-in-family', 0.91], ['1', 'Associate', 'Not-in-family', 0.09], ['0', 'Associate', 'Other-relative', 0.88], ['1', 'Associate', 'Other-relative', 0.12], ['0', 'Associate', 'Unmarried', 0.93], ['1', 'Associate', 'Unmarried', 0.07], ['0', 'Professional', 'Wife', 0.25], ['1', 'Professional', 'Wife', 0.75], ['0', 'Professional', 'Own-child', 0.95], ['1', 'Professional', 'Own-child', 0.05], ['0', 'Professional', 'Husband', 0.29], ['1', 'Professional', 'Husband', 0.71], ['0', 'Professional', 'Not-in-family', 0.79], ['1', 'Professional', 'Not-in-family', 0.21], ['0', 'Professional', 'Other-relative', 0.86], ['1', 'Professional', 'Other-relative', 0.14], ['0', 'Professional', 'Unmarried', 0.82], ['1', 'Professional', 'Unmarried', 0.18], ['0', 'Bachelors', 'Wife', 0.31], ['1', 'Bachelors', 'Wife', 0.69], ['0', 'Bachelors', 'Own-child', 0.95], ['1', 'Bachelors', 'Own-child', 0.05], ['0', 'Bachelors', 'Husband', 0.31], ['1', 'Bachelors', 'Husband', 0.69], ['0', 'Bachelors', 'Not-in-family', 0.83], ['1', 'Bachelors', 'Not-in-family', 0.17], ['0', 'Bachelors', 'Other-relative', 0.93], ['1', 'Bachelors', 'Other-relative', 0.07], ['0', 'Bachelors', 'Unmarried', 0.82], ['1', 'Bachelors', 'Unmarried', 0.18], ['0', 'Masters', 'Wife', 0.17], ['1', 'Masters', 'Wife', 0.83], ['0', 'Masters', 'Own-child', 0.93], ['1', 'Masters', 'Own-child', 0.07], ['0', 'Masters', 'Husband', 0.22], ['1', 'Masters', 'Husband', 0.78], ['0', 'Masters', 'Not-in-family', 0.72], ['1', 'Masters', 'Not-in-family', 0.28], ['0', 'Masters', 'Other-relative', 0.67], ['1', 'Masters', 'Other-relative', 0.33], ['0', 'Masters', 'Unmarried', 0.73], ['1', 'Masters', 'Unmarried', 0.27], ['0', 'Doctorate', 'Wife', 0.11], ['1', 'Doctorate', 'Wife', 0.89], ['0', 'Doctorate', 'Own-child', 0.86], ['1', 'Doctorate', 'Own-child', 0.14], ['0', 'Doctorate', 'Husband', 0.16], ['1', 'Doctorate', 'Husband', 0.84], ['0', 'Doctorate', 'Not-in-family', 0.45], ['1', 'Doctorate', 'Not-in-family', 0.55], ['1', 'Doctorate', 'Other-relative', 1.0], ['0', 'Doctorate', 'Other-relative', 0.0], ['0', 'Doctorate', 'Unmarried', 0.41], ['1', 'Doctorate', 'Unmarried', 0.59]]
    F8.add_values(values)

    adultDataset = BN('Adult Dataset',
             [ms, re, rc, ge, oc, co, ed, wc, sa],
             [F0, F1, F2, F3, F4, F5, F6, F7, F8])

    return adultDataset

def adultDatasetBN2():

    ms = Variable("MaritalStatus", ['Not-Married', 'Married', 'Separated', 'Widowed'])
    F0 = Factor("P(ms)", [ms])
    values = [['Not-Married', 0.33], ['Married', 0.47], ['Separated', 0.17], ['Widowed', 0.03]]
    F0.add_values(values)

    re = Variable("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
    F1 = Factor("P(re|ms)", [re,ms])
    values = [['Own-child', 'Not-Married', 0.39], ['Not-in-family', 'Not-Married', 0.46], ['Other-relative', 'Not-Married', 0.06], ['Unmarried', 'Not-Married', 0.09], ['Wife', 'Not-Married', 0.0], ['Husband', 'Not-Married', 0.0], ['Wife', 'Married', 0.1], ['Own-child', 'Married', 0.01], ['Husband', 'Married', 0.88], ['Not-in-family', 'Married', 0.0], ['Other-relative', 'Married', 0.01], ['Unmarried', 'Married', 0.0], ['Own-child', 'Separated', 0.08], ['Not-in-family', 'Separated', 0.51], ['Other-relative', 'Separated', 0.03], ['Unmarried', 'Separated', 0.38], ['Wife', 'Separated', 0.0], ['Husband', 'Separated', 0.0], ['Own-child', 'Widowed', 0.01], ['Not-in-family', 'Widowed', 0.52], ['Other-relative', 'Widowed', 0.05], ['Unmarried', 'Widowed', 0.41], ['Wife', 'Widowed', 0.0], ['Husband', 'Widowed', 0.0]]
    F1.add_values(values)

    rc = Variable("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    F2 = Factor("P(rc|re)", [rc, re])
    values = [['White', 'Wife', 0.85], ['Black', 'Wife', 0.09], ['Asian-Pac-Islander', 'Wife', 0.04], ['Amer-Indian-Eskimo', 'Wife', 0.01], ['Other', 'Wife', 0.01], ['White', 'Own-child', 0.84], ['Black', 'Own-child', 0.11], ['Asian-Pac-Islander', 'Own-child', 0.03], ['Amer-Indian-Eskimo', 'Own-child', 0.01], ['Other', 'Own-child', 0.01], ['White', 'Husband', 0.91], ['Black', 'Husband', 0.05], ['Asian-Pac-Islander', 'Husband', 0.03], ['Amer-Indian-Eskimo', 'Husband', 0.01], ['Other', 'Husband', 0.01], ['White', 'Not-in-family', 0.86], ['Black', 'Not-in-family', 0.09], ['Asian-Pac-Islander', 'Not-in-family', 0.02], ['Amer-Indian-Eskimo', 'Not-in-family', 0.01], ['Other', 'Not-in-family', 0.01], ['White', 'Other-relative', 0.72], ['Black', 'Other-relative', 0.16], ['Asian-Pac-Islander', 'Other-relative', 0.08], ['Amer-Indian-Eskimo', 'Other-relative', 0.01], ['Other', 'Other-relative', 0.03], ['White', 'Unmarried', 0.73], ['Black', 'Unmarried', 0.22], ['Asian-Pac-Islander', 'Unmarried', 0.03], ['Amer-Indian-Eskimo', 'Unmarried', 0.02], ['Other', 'Unmarried', 0.01]]
    F2.add_values(values)

    ge = Variable("Gender", ['Male', 'Female'])
    F3 = Factor("P(ge|re,ms)", [ge, re, ms])
    values = [['Male', 'Wife', 'Married', 0.0], ['Female', 'Wife', 'Married', 1.0], ['Male', 'Own-child', 'Not-Married', 0.57], ['Female', 'Own-child', 'Not-Married', 0.43], ['Male', 'Own-child', 'Married', 0.58], ['Female', 'Own-child', 'Married', 0.42], ['Male', 'Own-child', 'Separated', 0.51], ['Female', 'Own-child', 'Separated', 0.49], ['Male', 'Own-child', 'Widowed', 0.08], ['Female', 'Own-child', 'Widowed', 0.92], ['Male', 'Husband', 'Married', 1.0], ['Female', 'Husband', 'Married', 0.0], ['Male', 'Not-in-family', 'Not-Married', 0.58], ['Female', 'Not-in-family', 'Not-Married', 0.42], ['Male', 'Not-in-family', 'Married', 0.86], ['Female', 'Not-in-family', 'Married', 0.14], ['Male', 'Not-in-family', 'Separated', 0.53], ['Female', 'Not-in-family', 'Separated', 0.47], ['Male', 'Not-in-family', 'Widowed', 0.18], ['Female', 'Not-in-family', 'Widowed', 0.82], ['Male', 'Other-relative', 'Not-Married', 0.63], ['Female', 'Other-relative', 'Not-Married', 0.37], ['Male', 'Other-relative', 'Married', 0.59], ['Female', 'Other-relative', 'Married', 0.41], ['Male', 'Other-relative', 'Separated', 0.4], ['Female', 'Other-relative', 'Separated', 0.6], ['Male', 'Other-relative', 'Widowed', 0.15], ['Female', 'Other-relative', 'Widowed', 0.85], ['Male', 'Unmarried', 'Not-Married', 0.34], ['Female', 'Unmarried', 'Not-Married', 0.66], ['Male', 'Unmarried', 'Separated', 0.19], ['Female', 'Unmarried', 'Separated', 0.81], ['Male', 'Unmarried', 'Widowed', 0.17], ['Female', 'Unmarried', 'Widowed', 0.83]]
    F3.add_values(values)

    oc = Variable("Occupation", ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'])
    F4 = Factor("P(oc|ge, re)", [oc, ge, re])
    values = [['Office Labour', 'Male', 'Wife', 1.0], ['Admin', 'Male', 'Wife', 0.0], ['Military', 'Male', 'Wife', 0.0], ['Manual Labour', 'Male', 'Wife', 0.0], ['Service', 'Male', 'Wife', 0.0], ['Professional', 'Male', 'Wife', 0.0], ['Admin', 'Male', 'Own-child', 0.09], ['Military', 'Male', 'Own-child', 0.02], ['Manual Labour', 'Male', 'Own-child', 0.47], ['Office Labour', 'Male', 'Own-child', 0.18], ['Service', 'Male', 'Own-child', 0.17], ['Professional', 'Male', 'Own-child', 0.05], ['Admin', 'Male', 'Husband', 0.05], ['Military', 'Male', 'Husband', 0.03], ['Manual Labour', 'Male', 'Husband', 0.42], ['Office Labour', 'Male', 'Husband', 0.32], ['Service', 'Male', 'Husband', 0.04], ['Professional', 'Male', 'Husband', 0.14], ['Admin', 'Male', 'Not-in-family', 0.07], ['Military', 'Male', 'Not-in-family', 0.03], ['Manual Labour', 'Male', 'Not-in-family', 0.41], ['Office Labour', 'Male', 'Not-in-family', 0.26], ['Service', 'Male', 'Not-in-family', 0.09], ['Professional', 'Male', 'Not-in-family', 0.14], ['Admin', 'Male', 'Other-relative', 0.07], ['Military', 'Male', 'Other-relative', 0.02], ['Manual Labour', 'Male', 'Other-relative', 0.52], ['Office Labour', 'Male', 'Other-relative', 0.16], ['Service', 'Male', 'Other-relative', 0.18], ['Professional', 'Male', 'Other-relative', 0.05], ['Admin', 'Male', 'Unmarried', 0.07], ['Military', 'Male', 'Unmarried', 0.02], ['Manual Labour', 'Male', 'Unmarried', 0.54], ['Office Labour', 'Male', 'Unmarried', 0.21], ['Service', 'Male', 'Unmarried', 0.07], ['Professional', 'Male', 'Unmarried', 0.09], ['Admin', 'Female', 'Wife', 0.25], ['Military', 'Female', 'Wife', 0.0], ['Manual Labour', 'Female', 'Wife', 0.11], ['Office Labour', 'Female', 'Wife', 0.29], ['Service', 'Female', 'Wife', 0.13], ['Professional', 'Female', 'Wife', 0.22], ['Admin', 'Female', 'Own-child', 0.27], ['Military', 'Female', 'Own-child', 0.01], ['Manual Labour', 'Female', 'Own-child', 0.1], ['Office Labour', 'Female', 'Own-child', 0.3], ['Service', 'Female', 'Own-child', 0.24], ['Professional', 'Female', 'Own-child', 0.09], ['Office Labour', 'Female', 'Husband', 1.0], ['Admin', 'Female', 'Husband', 0.0], ['Military', 'Female', 'Husband', 0.0], ['Manual Labour', 'Female', 'Husband', 0.0], ['Service', 'Female', 'Husband', 0.0], ['Professional', 'Female', 'Husband', 0.0], ['Admin', 'Female', 'Not-in-family', 0.24], ['Military', 'Female', 'Not-in-family', 0.01], ['Manual Labour', 'Female', 'Not-in-family', 0.1], ['Office Labour', 'Female', 'Not-in-family', 0.29], ['Service', 'Female', 'Not-in-family', 0.17], ['Professional', 'Female', 'Not-in-family', 0.19], ['Admin', 'Female', 'Other-relative', 0.24], ['Military', 'Female', 'Other-relative', 0.01], ['Manual Labour', 'Female', 'Other-relative', 0.16], ['Office Labour', 'Female', 'Other-relative', 0.23], ['Service', 'Female', 'Other-relative', 0.28], ['Professional', 'Female', 'Other-relative', 0.08], ['Admin', 'Female', 'Unmarried', 0.27], ['Military', 'Female', 'Unmarried', 0.01], ['Manual Labour', 'Female', 'Unmarried', 0.13], ['Office Labour', 'Female', 'Unmarried', 0.25], ['Service', 'Female', 'Unmarried', 0.21], ['Professional', 'Female', 'Unmarried', 0.13]]
    F4.add_values(values)

    ed = Variable("Education",  ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'])
    F5 = Factor("P(ed|oc)", [ed, oc])
    values = [['<Gr12', 'Admin', 0.07], ['HS-Graduate', 'Admin', 0.53], ['Associate', 'Admin', 0.15], ['Professional', 'Admin', 0.0], ['Bachelors', 'Admin', 0.21], ['Masters', 'Admin', 0.03], ['Doctorate', 'Admin', 0.0], ['<Gr12', 'Military', 0.08], ['HS-Graduate', 'Military', 0.46], ['Associate', 'Military', 0.19], ['Professional', 'Military', 0.0], ['Bachelors', 'Military', 0.23], ['Masters', 'Military', 0.04], ['Doctorate', 'Military', 0.0], ['<Gr12', 'Manual Labour', 0.41], ['HS-Graduate', 'Manual Labour', 0.36], ['Associate', 'Manual Labour', 0.12], ['Professional', 'Manual Labour', 0.0], ['Bachelors', 'Manual Labour', 0.09], ['Masters', 'Manual Labour', 0.01], ['Doctorate', 'Manual Labour', 0.0], ['<Gr12', 'Office Labour', 0.07], ['HS-Graduate', 'Office Labour', 0.33], ['Associate', 'Office Labour', 0.11], ['Professional', 'Office Labour', 0.01], ['Bachelors', 'Office Labour', 0.36], ['Masters', 'Office Labour', 0.1], ['Doctorate', 'Office Labour', 0.01], ['<Gr12', 'Service', 0.43], ['HS-Graduate', 'Service', 0.38], ['Associate', 'Service', 0.09], ['Professional', 'Service', 0.0], ['Bachelors', 'Service', 0.09], ['Masters', 'Service', 0.01], ['Doctorate', 'Service', 0.0], ['<Gr12', 'Professional', 0.01], ['HS-Graduate', 'Professional', 0.06], ['Associate', 'Professional', 0.04], ['Professional', 'Professional', 0.55], ['Bachelors', 'Professional', 0.19], ['Masters', 'Professional', 0.11], ['Doctorate', 'Professional', 0.04]]
    F5.add_values(values)

    sa = Variable("Salary", ['0', '1'])
    F6 = Factor("P(sa|ed,re)", [sa, ed, re])
    values = [['0', '<Gr12', 'Wife', 0.89], ['1', '<Gr12', 'Wife', 0.11], ['0', '<Gr12', 'Own-child', 0.99], ['1', '<Gr12', 'Own-child', 0.01], ['0', '<Gr12', 'Husband', 0.87], ['1', '<Gr12', 'Husband', 0.13], ['0', '<Gr12', 'Not-in-family', 0.97], ['1', '<Gr12', 'Not-in-family', 0.03], ['0', '<Gr12', 'Other-relative', 0.99], ['1', '<Gr12', 'Other-relative', 0.01], ['0', '<Gr12', 'Unmarried', 0.97], ['1', '<Gr12', 'Unmarried', 0.03], ['0', 'HS-Graduate', 'Wife', 0.53], ['1', 'HS-Graduate', 'Wife', 0.47], ['0', 'HS-Graduate', 'Own-child', 0.99], ['1', 'HS-Graduate', 'Own-child', 0.01], ['0', 'HS-Graduate', 'Husband', 0.56], ['1', 'HS-Graduate', 'Husband', 0.44], ['0', 'HS-Graduate', 'Not-in-family', 0.93], ['1', 'HS-Graduate', 'Not-in-family', 0.07], ['0', 'HS-Graduate', 'Other-relative', 0.95], ['1', 'HS-Graduate', 'Other-relative', 0.05], ['0', 'HS-Graduate', 'Unmarried', 0.96], ['1', 'HS-Graduate', 'Unmarried', 0.04], ['0', 'Associate', 'Wife', 0.44], ['1', 'Associate', 'Wife', 0.56], ['0', 'Associate', 'Own-child', 0.99], ['1', 'Associate', 'Own-child', 0.01], ['0', 'Associate', 'Husband', 0.54], ['1', 'Associate', 'Husband', 0.46], ['0', 'Associate', 'Not-in-family', 0.91], ['1', 'Associate', 'Not-in-family', 0.09], ['0', 'Associate', 'Other-relative', 0.88], ['1', 'Associate', 'Other-relative', 0.12], ['0', 'Associate', 'Unmarried', 0.93], ['1', 'Associate', 'Unmarried', 0.07], ['0', 'Professional', 'Wife', 0.25], ['1', 'Professional', 'Wife', 0.75], ['0', 'Professional', 'Own-child', 0.95], ['1', 'Professional', 'Own-child', 0.05], ['0', 'Professional', 'Husband', 0.29], ['1', 'Professional', 'Husband', 0.71], ['0', 'Professional', 'Not-in-family', 0.79], ['1', 'Professional', 'Not-in-family', 0.21], ['0', 'Professional', 'Other-relative', 0.86], ['1', 'Professional', 'Other-relative', 0.14], ['0', 'Professional', 'Unmarried', 0.82], ['1', 'Professional', 'Unmarried', 0.18], ['0', 'Bachelors', 'Wife', 0.31], ['1', 'Bachelors', 'Wife', 0.69], ['0', 'Bachelors', 'Own-child', 0.95], ['1', 'Bachelors', 'Own-child', 0.05], ['0', 'Bachelors', 'Husband', 0.31], ['1', 'Bachelors', 'Husband', 0.69], ['0', 'Bachelors', 'Not-in-family', 0.83], ['1', 'Bachelors', 'Not-in-family', 0.17], ['0', 'Bachelors', 'Other-relative', 0.93], ['1', 'Bachelors', 'Other-relative', 0.07], ['0', 'Bachelors', 'Unmarried', 0.82], ['1', 'Bachelors', 'Unmarried', 0.18], ['0', 'Masters', 'Wife', 0.17], ['1', 'Masters', 'Wife', 0.83], ['0', 'Masters', 'Own-child', 0.93], ['1', 'Masters', 'Own-child', 0.07], ['0', 'Masters', 'Husband', 0.22], ['1', 'Masters', 'Husband', 0.78], ['0', 'Masters', 'Not-in-family', 0.72], ['1', 'Masters', 'Not-in-family', 0.28], ['0', 'Masters', 'Other-relative', 0.67], ['1', 'Masters', 'Other-relative', 0.33], ['0', 'Masters', 'Unmarried', 0.73], ['1', 'Masters', 'Unmarried', 0.27], ['0', 'Doctorate', 'Wife', 0.11], ['1', 'Doctorate', 'Wife', 0.89], ['0', 'Doctorate', 'Own-child', 0.86], ['1', 'Doctorate', 'Own-child', 0.14], ['0', 'Doctorate', 'Husband', 0.16], ['1', 'Doctorate', 'Husband', 0.84], ['0', 'Doctorate', 'Not-in-family', 0.45], ['1', 'Doctorate', 'Not-in-family', 0.55], ['1', 'Doctorate', 'Other-relative', 1.0], ['0', 'Doctorate', 'Other-relative', 0.0], ['0', 'Doctorate', 'Unmarried', 0.41], ['1', 'Doctorate', 'Unmarried', 0.59]]
    F6.add_values(values)

    adultDataset = BN('Adult Dataset',
             [ms, re, rc, ge, oc, ed, sa],
             [F0, F1, F2, F3, F4, F5, F6])

    return adultDataset

# Methods to support VE
def restrict_factor(f, var, value):
    '''f is a factor, var is a Variable, and value is a value from var.domain.
    Returns a new factor that is the restriction of f by this var = value.
    If f has only one variable its restriction yields a
    constant factor'''

    scope = f.get_scope()
    scope.remove(var)
    F = Factor("Restrict {}|{}={}".format(f.name, var.name, value), scope)
    var.set_assignment(value)

    def recursive_restrict_factor(Vars):
        if len(Vars) == 0:
            F.add_value_at_current_assignment(f.get_value_at_current_assignments())
        elif Vars[0] != var:
            for val in Vars[0].domain():
                Vars[0].set_assignment(val)
                recursive_restrict_factor(Vars[1:])
        else:
            recursive_restrict_factor(Vars[1:])

    recursive_restrict_factor(f.get_scope())
    return F

def sum_out_variable(f, var):
    '''Returns a new factor that is the product of the factors in Factors
       followed by the summing out of Var'''
    scope = f.get_scope()
    scope.remove(var)
    F = Factor("Eliminate-{}-{}".format(var.name, f), scope)

    def recursive_eliminate_variable(Vars):
        if len(Vars) == 0:
            summ = 0
            for val in var.domain():
                var.set_assignment(val)
                prod = f.get_value_at_current_assignments()
                summ = summ + prod
            F.add_value_at_current_assignment(summ)
        else:
            for val in Vars[0].domain():
                Vars[0].set_assignment(val)
                recursive_eliminate_variable(Vars[1:])

    recursive_eliminate_variable(scope)
    return F

def normalize(nums):
    '''Takes as input a list of number and return a new list of numbers where
    now the numbers sum to 1, i.e., normalize the input numbers'''
    s = sum(nums)
    if s == 0:
        newnums = [0] * len(nums)
    else:
        newnums = []
        for n in nums:
            newnums.append(n / s)
    return newnums

