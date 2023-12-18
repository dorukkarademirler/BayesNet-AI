from solution import *

test_multiply = True
test_ve = True
test_sampling = True
test_confound = True
test_mediate = True

#  E,B,S,w,G example
E, B, S, G, W = Variable('E', ['e', '-e']), Variable('B', ['b', '-b']), Variable('S', ['s', '-s']), Variable('G', ['g',
                                                                                                                   '-g']), Variable(
    'W', ['w', '-w'])
FE, FB, FS, FG, FW = Factor('P(E)', [E]), Factor('P(B)', [B]), Factor('P(S|E,B)', [S, E, B]), Factor('P(G|S)',
                                                                                                     [G, S]), Factor(
    'P(W|S)', [W, S])

FE.add_values([['e', 0.1], ['-e', 0.9]])
FB.add_values([['b', 0.1], ['-b', 0.9]])
FS.add_values([['s', 'e', 'b', .9], ['s', 'e', '-b', .2], ['s', '-e', 'b', .8], ['s', '-e', '-b', 0],
               ['-s', 'e', 'b', .1], ['-s', 'e', '-b', .8], ['-s', '-e', 'b', .2], ['-s', '-e', '-b', 1]])
FG.add_values([['g', 's', 0.5], ['g', '-s', 0], ['-g', 's', 0.5], ['-g', '-s', 1]])
FW.add_values([['w', 's', 0.8], ['w', '-s', .2], ['-w', 's', 0.2], ['-w', '-s', 0.8]])

ExampleBN = BN('ExampleBN', [E, B, S, G, W], [FE, FB, FS, FG, FW])


def test_multiply_fun():
    print("\nMultiply Factors Test ... ", end='')
    factor = multiply_factors([FB, FE])
    tests = []
    values = []

    if (factor == None):
        print("failed.")
        return

    for e_val in E.domain():
        for b_val in B.domain():
            try:
                value = factor.get_value([e_val, b_val])
                values.append(value)
            except ValueError:
                value = factor.get_value([b_val, e_val])
                values.append(value)
            tests.append(value == FE.get_value([e_val]) * FB.get_value([b_val]))
    if all(tests):
        print("passed.")
    else:
        print("failed.")
    print('P(e,b) = {} P(-e,b) = {} P(e,-b) = {} P(-e,-b) = {}'.format(values[0], values[1], values[2], values[3]))

def test_ve_fun():
    print("\nVE Tests .... ")
    print("Test 1 ....", end='')
    S.set_evidence('-s')
    W.set_evidence('w')
    probs3 = VE(ExampleBN, G, [S, W])
    S.set_evidence('-s')
    W.set_evidence('-w')
    probs4 = VE(ExampleBN, G, [S, W])

    if (probs4 == None or probs3 == None ):
        print("failed.")
        return

    if probs3[0] == 0.0 and probs3[1] == 1.0 and probs4[0] == 0.0 and probs4[1] == 1.0:
        print("passed.")
    else:
        print("failed.")
    print('P(g|-s,w) = {} P(-g|-s,w) = {} P(g|-s,-w) = {} P(-g|-s,-w) = {}'.format(probs3[0], probs3[1], probs4[0],
                                                                                   probs4[1]))
    print("Test 2 ....", end='')
    W.set_evidence('w')
    probs1 = VE(ExampleBN, G, [W])
    W.set_evidence('-w')
    probs2 = VE(ExampleBN, G, [W])
    if abs(probs1[0] - 0.15265998457979954) < 0.0001 and abs(probs1[1] - 0.8473400154202004) < 0.0001 and abs(
            probs2[0] - 0.01336753983256819) < 0.0001 and abs(probs2[1] - 0.9866324601674318) < 0.0001:
        print("passed.")
    else:
        print("failed.")
    print('P(g|w) = {} P(-g|w) = {} P(g|-w) = {} P(-g|-w) = {}'.format(probs1[0], probs1[1], probs2[0], probs2[1]))


def test_sample_fun():

    model = CausalModelMediator()

    if (model == None):
        print("failed.")
        return

    Variables = model.variables()

    if (Variables == None):
        print("failed.")
        return

    Variables[0].set_evidence("Italy")

    probsEstimated1 = SampleBN(model, Variables[2], [Variables[0]])
    probsExact = VE(model, Variables[2], [Variables[0]])
    probsEstimated2 = SampleBN(model, Variables[2], [Variables[0]])

    if abs(probsEstimated1[0] - probsExact[0]) < 0.1 and abs(probsEstimated1[1] - probsExact[1]) < 0.1:
        print("Passed first test of sampling.")
    else:
        print("Failed first test of sampling.")

    if abs(probsEstimated2[0] - probsExact[0]) < 0.1 and abs(probsEstimated2[1] - probsExact[1]) < 0.1:
        print("Passed second test of sampling.")
    else:
        print("Failed second test of sampling.")

    if probsEstimated2[0] != probsEstimated1[0] and probsEstimated2[1] != probsEstimated1[1]:
        print("Passed third test of sampling.")
    else:
        print("Failed third test of sampling.")

    return



def test_confound_fun():
    model = CausalModelConfounder()

    if (model == None):
        print("failed.")
        return

    Factors = model.factors()
    Variables = model.variables()

    if (Variables == None or Factors == None):
        print("failed.")
        return

    ages = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80']

    for f in Factors:

        if (len(f.get_scope()) == 2):
            if (abs(f.get_value(['Italy', '70-79']) - 0.89) >= 0.01):
                print("Failed test of confounding model, P(Italy|Age=70-79) is not 0.89.")
                return False
            if (abs(f.get_value(['China', '10-19']) - 0.15) >= 0.01):
                print("Failed test of confounding model, P(China|Age=10-19) is not 0.15.")
                return False
            if (abs(f.get_value(['Italy','70-79']) + f.get_value(['China','70-79']) - 1.0) >= 0.001):
                print("Failed test of confounding model, P(Italy|Age=70-79) + P(China|Age=70-79) is not 1.0.")
                return False

        if (len(f.get_scope()) == 1):
            if len(f.get_values()) != len(ages): #prior on ages
                print("Failed test of confounding model.")
                return False

    print("Passed test of confounding model.")
    return True


def test_mediate_fun():
    model = CausalModelMediator()

    if (model == None):
        print("failed.")
        return

    Factors = model.factors()
    Variables = model.variables()

    if (Variables == None or Factors == None):
        print("failed.")
        return

    for f in Factors:
        if (len(f.get_scope()) == 2):
            if (abs(f.get_value(['70-79', 'Italy']) - 0.14) >= 0.01):
                print("Failed test of mediating model, P(Age=70-79|Italy) is not 0.14.")
                return False
            if (abs(f.get_value(['10-19', 'China']) - 0.01) >= 0.01):
                print("Failed test of mediating model, P(Age=10-19|China) is not 0.01.")
                return False
            if (abs(f.get_value(['70-79', 'Italy']) + f.get_value(['70-79','China']) - 1.0) <= 0.001):
                print("Failed test of confounding model, P(Age=70-79|Italy) + P(Age=70-79|China) is 1.0 but should not be.")
                return False

    print("Passed test of mediating model.")
    return True


if __name__ == '__main__':
    if test_multiply:
        test_multiply_fun()
    if test_ve:
        test_ve_fun()
    if test_sampling:
        test_sample_fun()
    if test_confound:
        test_confound_fun()
    if test_mediate:
        test_mediate_fun()
