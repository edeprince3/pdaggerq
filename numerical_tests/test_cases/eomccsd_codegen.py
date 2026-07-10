from pdaggerq.numerical_utils.autogen import cc_residual
from pdaggerq.numerical_utils.autogen import eomcc_right_sigma
from pdaggerq.numerical_utils.autogen import eomcc_left_sigma

def main():
    """
    generate t1 and t2 residuals for ccsd, plus sigma0, simga1, sigma2 for eomccsd
    """
    cc_residual('cc_energy', ['t1', 't2'], [['1']], 'cc_energy')
    cc_residual('r1', ['t1', 't2'], [['e1(i,a)']], 't1_residual')
    cc_residual('r2', ['t1', 't2'], [['e2(i,j,b,a)']], 't2_residual')

    eomcc_right_sigma('sigma0', ['t1', 't2'], [['1']], 'right_sigma0')
    eomcc_right_sigma('sigma1', ['t1', 't2'], [['e1(i,a)']], 'right_sigma1')
    eomcc_right_sigma('sigma2', ['t1', 't2'], [['e2(i,j,b,a)']], 'right_sigma2')

    eomcc_left_sigma('sigma0', ['t1', 't2'], [['1']], 'left_sigma0')
    eomcc_left_sigma('sigma1', ['t1', 't2'], [['e1(a,i)']], 'left_sigma1')
    eomcc_left_sigma('sigma2', ['t1', 't2'], [['e2(a,b,j,i)']], 'left_sigma2')

if __name__ == "__main__":
    main()

