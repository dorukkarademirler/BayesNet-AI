import random  # for sampling methods
import numpy as np
import pandas as pd  # to simplify reading and holding data
from bnetbase import Variable, Factor, BN, restrict_factor, sum_out_variable, normalize
from itertools import product
import itertools


def multiply_factors(Factors):
    '''return a new factor that is the product of the factors in Factors'''
    factors = Factors.copy()
    while True:
        if len(factors) == 1:
            break
        factor1 = factors.pop()
        factor2 = factors.pop()

        scope1 = factor1.get_scope()
        scope2 = factor2.get_scope()

        remaining_scope1 = [var for var in scope1 if var not in scope2]

        # filtering the similar ones from 1
        new_scp = scope2 + remaining_scope1

        factor = Factor("Factor", new_scp)

        combs = list(itertools.product(*[scp.domain() for scp in new_scp]))

        for comb in combs:
            comb1, comb2 = [], []
            for elem in scope1:
                comb1.append(comb[new_scp.index(elem)])

            for elem in scope2:
                comb2.append(comb[new_scp.index(elem)])

            mult = factor1.get_value(comb1) * factor2.get_value(comb2)
            # Multiply the values and add to the new factor
            factor.add_values([list(comb) + [mult]])
        factors.append(factor)
    return factors[0]


###Orderings
def min_fill_ordering(Factors, QueryVar):
    """
    Ordering
    Do this
    """
    count_dct = {}
    for factor in Factors:
        scopes = factor.get_scope()
        for scope in scopes:
            if scope != QueryVar:
                count_dct.setdefault(scope, 0)
                count_dct[scope] += 1
    ordered_list = []
    for var, _ in sorted(count_dct.items(), key=lambda x: x[1]):
        ordered_list.append(var)
    return ordered_list


def VE(Net, QueryVar, EvidenceVars):
    '''
    Input: Net---a BN object (a Bayes Net)
           QueryVar---a Variable object (the variable whose distribution
                      we want to compute)
           EvidenceVars---a LIST of Variable objects. Each of these
                          variables has had its evidence set to a particular
                          value from its domain using set_evidence.

   VE returns a distribution over the values of QueryVar, i.e., a list
   of numbers one for every value in QueryVar's domain. These numbers
   sum to one, and the i'th number is the probability that QueryVar is
   equal to its i'th value given the setting of the evidence
   variables. For example if QueryVar = A with Dom[A] = ['a', 'b',
   'c'], EvidenceVars = [B, C], and we have previously called
   B.set_evidence(1) and C.set_evidence('c'), then VE would return a
   list of three numbers. E.g. [0.5, 0.24, 0.26]. These numbers would
   mean that Pr(A='a'|B=1, C='c') = 0.5 Pr(A='a'|B=1, C='c') = 0.24
   Pr(A='a'|B=1, C='c') = 0.26
    '''

    probabilities = []
    factors = Net.factors()

    for ev in EvidenceVars:
        i = 0
        while i < len(factors):
            factor = factors[i]
            if ev in factor.get_scope():
                factors[i] = restrict_factor(factor, ev, ev.get_evidence())
            i += 1

    ordered = min_fill_ordering(factors, QueryVar)

    # Eliminate each variable
    for var in ordered:
        factors = VE_cagan31(factors, var)
    last_factor = multiply_factors(factors)

    for dom in QueryVar.domain():
        prob = last_factor.get_value([dom])
        probabilities.append(prob)
    return normalize(probabilities)


def VE_cagan31(factors, var):
    """
    Helper for VE that eliminates the vars
    """
    lst = [factor for factor in factors if var in factor.get_scope()]
    if not lst:
        return factors
    multiplied_factor = multiply_factors(lst)
    new_lst = [factor for factor in factors if factor not in lst]
    summed = sum_out_variable(multiplied_factor, var)
    new_lst.append(summed)
    return new_lst

def SampleBN(Net, QueryVar, EvidenceVars):
    '''
     Input: Net---a BN object (a Bayes Net)
            QueryVar---a Variable object (the variable whose distribution
                       we want to compute)
            EvidenceVars---a LIST of Variable objects. Each of these
                           variables has had its evidence set to a particular
                           value from its domain using set_evidence.

    SampleBN returns a distribution over the values of QueryVar, i.e., a list
    of numbers one for every value in QueryVar's domain. These numbers
    sum to one, and the i'th number is the probability that QueryVar is
    equal to its i'th value given the setting of the evidence
    variables.

    SampleBN should generate *1000* samples using the likelihood
    weighting method described in class.  It should then calculate
    a distribution over value assignments to QueryVar based on the
    values of these samples.
    '''

    # YOUR CODE HERE
    weights = [0] * len(QueryVar.domain())
    total = 0
    evidence = EvidenceVars.copy()

    for i in range(1000):
        curr_weight = 1
        dct = {}

        # For each singular factor we have to modify the curr weight
        for ev in evidence:
            dct[ev.name] = ev.get_evidence()
            for factor in Net.factors():
                scp = factor.get_scope()
                if ev in scp and len(scp) == 1:
                    curr_weight *= factor.get_value([ev.get_evidence()])

        for variable in Net.variables():
            if variable not in EvidenceVars:
                for factor in Net.factors():
                    if variable in factor.get_scope():
                        curr_factor = factor
                        break

                if curr_factor is not None:
                    probs = []
                    for val in variable.domain():
                        combination = []
                        for v in curr_factor.get_scope():
                            if v == variable:
                                combination.append(val)
                            else:
                                combination.append(dct[v.name])

                        # Formed combinations like [Age, Country] so we can get the prob
                        probability = curr_factor.get_value(combination)
                        probs.append(probability)


                    # trying to normalize probs
                    prob_sum = sum(probs)
                    normalized_probs = []
                    for p in probs:
                        normalized_probs.append(p / prob_sum)

                    # choose a value
                    accumulator = 0
                    chosen_val = 0
                    random_num = random.random()
                    for i in range(len(normalized_probs)):
                        accumulator += normalized_probs[i]
                        if random_num <= accumulator:
                            chosen_val = variable.domain()[i]
                            break

                    dct[variable.name] = chosen_val

        query = dct.get(QueryVar.name, None)
        if query:
            for _, value in enumerate(QueryVar.domain()):
                if value == query:
                    weights[i] += curr_weight
                    break
        total += curr_weight

    probabilities = []
    for weight in weights:
        normalized_weight = weight / total
        probabilities.append(normalized_weight)

    return probabilities


def CausalModelMediator():
    """CausalModelConfounder returns a Causal model (i.e. a BN) that
    represents the joint distribution of value assignments to
    variables in COVID-19 data.

    The structure of this model should reflect the assumption
    that age is a MEDIATING variable in the network, and
    is mediates the causal effect of Country on Fatality.

     Returns:
         BN: A BN that represents the causal model
     """

    ### READ IN THE DATA
    df = pd.read_csv('data/covid.csv')

    ### DOMAIN INFORMATION
    variable_domains = {
        "Age": ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80'],
        "Country": ['Italy', 'China'],
        "Fatality": ['YES', 'NO']
    }
    # variables
    Age = Variable('Age', ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80'])
    Country = Variable("Country", ['Italy', 'China'])
    Fatality = Variable("Fatality", ['YES', 'NO'])

    # factors
    factor_country = Factor("A", [Country])
    factor_age = Factor("A|C", [Age, Country])
    factor_fatality = Factor("F|A,C", [Fatality, Age, Country])

    # entering factors
    country_populations = df['Country'].value_counts()
    for country_name, population in country_populations.items():
        if country_name in Country.domain():
            probability = population / len(df)
            factor_country.add_values([[country_name, probability]])

    # entering factor age
    age_country_counts = df.groupby(['Country', 'Age']).size()
    for elem in age_country_counts.items():
        country_data, population = elem[0], elem[1]
        name, age = country_data[0], country_data[1]
        country_count = country_populations.get(name, 0)
        factor_age.add_values([[age, name, population / country_count if country_count > 0 else 0]])

    fatality_age_country_counts = df.groupby(['Fatality', 'Age', 'Country']).size()
    for elem in fatality_age_country_counts.items():
        country_data = elem[0]
        fatality, age, name = country_data[0], country_data[1], country_data[2]
        population = elem[1]
        count = fatality_age_country_counts.get((fatality, age, name), 0)
        probability = population / count if count > 0 else 0
        factor_fatality.add_values([[fatality, age, name, probability]])

    # creating BN
    bn = BN('CausalModelMediator', [Country, Age, Fatality], [factor_country, factor_age, factor_fatality])
    print("Mediator Variables:", [var.name for var in bn.variables()])
    return bn


def CausalModelConfounder():
    """CausalModelConfounder returns a Causal model (i.e. a BN) that
   represents the joint distribution of value assignments to
   variables in COVID-19 data.

   The structure of this model should reflect the assumption
   that age is a COUNFOUNDING variable in the network, and
   is therefore a cause of both Country and Fatality.

    Returns:
        BN: A BN that represents the causal model
    """

    ### READ IN THE DATA
    df = pd.read_csv('data/covid.csv')

    ### DOMAIN INFORMATION
    variable_domains = {
        "Age": ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80'],
        "Country": ['Italy', 'China'],
        "Fatality": ['YES', 'NO']
    }
    ### DOMAIN INFORMATION
    variable_domains = {
        "Age": ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80'],
        "Country": ['Italy', 'China'],
        "Fatality": ['YES', 'NO']
    }

    Age = Variable('Age', ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80'])
    Country = Variable("Country", ['Italy', 'China'])
    Fatality = Variable("Fatality", ['YES', 'NO'])

    # define the factors
    factor_age = Factor("P(A)", [Age])
    factor_country_age = Factor("P(C|A)", [Country, Age])
    factor_fatality_age_country = Factor("P(F|A,C)", [Fatality, Age, Country])

    for country in Country.domain():
        for age in Age.domain():
            country_df = df['Country'] == country
            age_df = df['Age'] == age
            comb = df[country_df & age_df]
            temp_prob = len(comb) / len(df[age_df])
            factor_country_age.add_values([[country, age, temp_prob]])

            for fatality in Fatality.domain():
                country_df = df['Country'] == country
                age_df = df['Age'] == age
                fatality_df = df['Fatality'] == fatality
                comb = df[country_df & age_df & fatality_df]
                temp_prob = len(comb) / len(df[country_df & age_df])
                factor_fatality_age_country.add_values([[fatality, age, country, temp_prob]])
    # create the bn
    bn = BN('CausalModelConfounder', [Age, Country, Fatality], [factor_age, factor_country_age, factor_fatality_age_country])

    return bn


if __name__ == "__main__":
    # You can Calculate Causal Effects of Country on Fatality here!

    # MEDIATOR
    mediating = CausalModelMediator()
    country = mediating.get_variable("Country")
    fatality = mediating.get_variable("Fatality")
    age_new = mediating.get_variable("Age")

    country.set_evidence("Italy")
    italy_data = VE(mediating, fatality, [country])

    country.set_evidence("China")
    china_data = VE(mediating, fatality, [country])

    ace_mediator = abs(italy_data[1] - china_data[1])
    print("age: Mediator: ", ace_mediator)

    # CONFOUNDER
    confounder_bn = CausalModelConfounder()
    ace_confounding = 0
    for age in age_new.domain():
        age_new.set_evidence(age)

        country.set_evidence("Italy")
        italy_data = VE(confounder_bn, fatality, [country, age_new])

        country.set_evidence("China")
        china_data = VE(confounder_bn, fatality, [country, age_new])

        ace_confounding = (italy_data[1] - china_data[1])



    print("age: Confounder: ", ace_confounding)

    # I would prefer using the mediator as I believe the final fatality outcome is primarily determined by age.
    # If I said confounder, I would be stating that age could also play a role in where people decide to live, which
    # in this case I think it's false.

    # A missing mediator could be access to free healthcare, as it would basically a path between the country and
    # fatality. If a country doesn't have free healthcare, this could directly impact the outcome

    # A missing confounder could be socioeconomical status as it has impact on a person's daily life and the area
    # that they live in. If they live in a poor area, this would be the main cause of experiencing fatal outcomes,
    # which might misinform about the relationship between the country and fatality.