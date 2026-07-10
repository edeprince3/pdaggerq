from pdaggerq.numerical_utils.autogen import cc_residual

def main():
    """
    generate t1, t2, and t3 residuals for ccsd
    """
    cc_residual('cc_energy', ['t1', 't2', 't3'], [['1']], 'cc_energy')
    cc_residual('r1', ['t1', 't2', 't3'], [['e1(i,a)']], 't1_residual')
    cc_residual('r2', ['t1', 't2', 't3'], [['e2(i,j,b,a)']], 't2_residual')
    cc_residual('r3', ['t1', 't2', 't3'], [['e3(i,j,k,c,b,a)']], 't3_residual')

if __name__ == "__main__":
    main()

