import pdaggerq


ahat = pdaggerq.ahat_helper()

# [h,T1]

ahat.set_string(['p*','q','a*','i'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.add_new_string()

ahat.set_string(['a*','i','p*','q'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_factor(-1.0)
ahat.add_new_string()

# [h,T2]

ahat.set_string(['p*','q','a*','b*','j','i'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','b','i','j'])
ahat.add_new_string()

ahat.set_string(['a*','b*','j','i','p*','q'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','b','i','j'])
ahat.set_factor(-1.0)
ahat.add_new_string()

# [[h,T1],T1]

ahat.set_string(['p*','q','a*','i','c*','k'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','k'])
ahat.set_factor(0.5)
ahat.add_new_string()

ahat.set_string(['a*','i','p*','q','c*','k'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','k'])
ahat.set_factor(-0.5)
ahat.add_new_string()

ahat.set_string(['c*','k','p*','q','a*','i'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','k'])
ahat.set_factor(-0.5)
ahat.add_new_string()

ahat.set_string(['c*','k','a*','i','p*','q'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','k'])
ahat.set_factor(0.5)
ahat.add_new_string()

# [[h,T1],T2]

ahat.set_string(['p*','q','a*','i','c*','d*','l','k'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(0.125)
ahat.add_new_string()

ahat.set_string(['a*','i','p*','q','c*','d*','l','k'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(-0.125)
ahat.add_new_string()

ahat.set_string(['c*','d*','l','k','p*','q','a*','i'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(-0.125)
ahat.add_new_string()

ahat.set_string(['c*','d*','l','k','a*','i','p*','q'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(0.125)
ahat.add_new_string()

# [[h,T2],T1]

ahat.set_string(['p*','q','c*','d*','l','k','a*','i'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(0.125)
ahat.add_new_string()

ahat.set_string(['c*','d*','l','k','p*','q','a*','i'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(-0.125)
ahat.add_new_string()

ahat.set_string(['a*','i','p*','q','c*','d*','l','k'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(-0.125)
ahat.add_new_string()

ahat.set_string(['a*','i','c*','d*','l','k','p*','q'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','i'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(0.125)
ahat.add_new_string()

# [[h,T2],T2]

ahat.set_string(['p*','q','a*','b*','j','i','c*','d*','l','k'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','b','i','j'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(0.03125)
ahat.add_new_string()

ahat.set_string(['a*','b*','j','i','p*','q','c*','d*','l','k'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','b','i','j'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(-0.03125)
ahat.add_new_string()

ahat.set_string(['c*','d*','l','k','p*','q','a*','b*','j','i'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','b','i','j'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(-0.03125)
ahat.add_new_string()

ahat.set_string(['c*','d*','l','k','a*','b*','j','i','p*','q'])
ahat.set_tensor(['p','q'])
ahat.set_amplitudes(['a','b','i','j'])
ahat.set_amplitudes(['c','d','k','l'])
ahat.set_factor(0.03125)
ahat.add_new_string()

ahat.simplify()
ahat.print()
