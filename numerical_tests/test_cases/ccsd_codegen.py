from pdaggerq.numerical_utils.autogen import cc_residual

def main():
    """
    generate t1 and t2 residuals for ccsd
    """
    cc_residual('cc_energy', ['t1', 't2'], [['1']], 'cc_energy')
    cc_residual('r1', ['t1', 't2'], [['e1(i,a)']], 't1_residual')
    cc_residual('r2', ['t1', 't2'], [['e2(i,j,b,a)']], 't2_residual')

if __name__ == "__main__":
    main()

