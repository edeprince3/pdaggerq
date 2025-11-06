# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Copyright (C) 2020 A. Eugene DePrince III
#
# This file is part of the pdaggerq package.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Functions to build EOM-CCSD sigma vectors or full EOM-CCSD Hamiltonian
"""

import numpy as np
from numpy import einsum

def build_eom_ccsd_H_by_block(kd, f, g, o, v, t1, t2):

    #    H(0;0) = <0| e(-T) H e(T) |0>
    
    #	  1.0000 f(i,i)
    H00 =  1.000000000000000 * einsum('ii', f[o, o])
    
    #	  1.0000 f(i,a)*t1(a,i)
    H00 +=  1.000000000000000 * einsum('ia,ai', f[o, v], t1)
    
    #	 -0.5000 <j,i||j,i>
    H00 += -0.500000000000000 * einsum('jiji', g[o, o, o, o])
    
    #	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    H00 +=  0.250000000000000 * einsum('jiab,abji', g[o, o, v, v], t2)
    
    #	 -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    H00 += -0.500000000000000 * einsum('jiab,ai,bj', g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    
    #    H(m,e;0) = <0|e1(m,e) e(-T) H e(T) |0>
    
    #	  1.0000 f(e,m)
    Hs0 =  1.000000000000000 * einsum('em->em', f[v, o])
    
    #	 -1.0000 f(i,m)*t1(e,i)
    Hs0 += -1.000000000000000 * einsum('im,ei->em', f[o, o], t1)
    
    #	  1.0000 f(e,a)*t1(a,m)
    Hs0 +=  1.000000000000000 * einsum('ea,am->em', f[v, v], t1)
    
    #	 -1.0000 f(i,a)*t2(a,e,m,i)
    Hs0 += -1.000000000000000 * einsum('ia,aemi->em', f[o, v], t2)
    
    #	 -1.0000 f(i,a)*t1(a,m)*t1(e,i)
    Hs0 += -1.000000000000000 * einsum('ia,am,ei->em', f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <i,e||a,m>*t1(a,i)
    Hs0 +=  1.000000000000000 * einsum('ieam,ai->em', g[o, v, v, o], t1)
    
    #	 -0.5000 <j,i||a,m>*t2(a,e,j,i)
    Hs0 += -0.500000000000000 * einsum('jiam,aeji->em', g[o, o, v, o], t2)
    
    #	 -0.5000 <i,e||a,b>*t2(a,b,m,i)
    Hs0 += -0.500000000000000 * einsum('ieab,abmi->em', g[o, v, v, v], t2)
    
    #	  1.0000 <j,i||a,b>*t1(a,i)*t2(b,e,m,j)
    Hs0 +=  1.000000000000000 * einsum('jiab,ai,bemj->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <j,i||a,b>*t1(a,m)*t2(b,e,j,i)
    Hs0 +=  0.500000000000000 * einsum('jiab,am,beji->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <j,i||a,b>*t1(e,i)*t2(a,b,m,j)
    Hs0 +=  0.500000000000000 * einsum('jiab,ei,abmj->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <j,i||a,m>*t1(a,i)*t1(e,j)
    Hs0 +=  1.000000000000000 * einsum('jiam,ai,ej->em', g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <i,e||a,b>*t1(a,i)*t1(b,m)
    Hs0 +=  1.000000000000000 * einsum('ieab,ai,bm->em', g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,i||a,b>*t1(a,i)*t1(b,m)*t1(e,j)
    Hs0 +=  1.000000000000000 * einsum('jiab,ai,bm,ej->em', g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    
    #    H(0;i,a) = <0| e(-T) H e(T) e1(a,i)|0>
    
    #	  1.0000 f(i,a)
    H0s =  1.000000000000000 * einsum('ia->ai', f[o, v])
    
    #	 -1.0000 <i,j||b,a>*t1(b,j)
    H0s += -1.000000000000000 * einsum('ijba,bj->ai', g[o, o, v, v], t1)
    
    
    #    H(m,n,e,f;0) = <0|e2(m,n,f,e) e(-T) H e(T) |0>
    
    #	 -1.0000 P(m,n)f(i,n)*t2(e,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('in,efmi->efmn', f[o, o], t2)
    Hd0 =  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)f(e,a)*t2(a,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('ea,afmn->efmn', f[v, v], t2)
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)f(i,a)*t1(a,n)*t2(e,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('ia,an,efmi->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)f(i,a)*t1(e,i)*t2(a,f,m,n)
    contracted_intermediate = -1.000000000000000 * einsum('ia,ei,afmn->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 <e,f||m,n>
    Hd0 +=  1.000000000000000 * einsum('efmn->efmn', g[v, v, o, o])
    
    #	  1.0000 P(e,f)<i,e||m,n>*t1(f,i)
    contracted_intermediate =  1.000000000000000 * einsum('iemn,fi->efmn', g[o, v, o, o], t1)
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 P(m,n)<e,f||a,n>*t1(a,m)
    contracted_intermediate =  1.000000000000000 * einsum('efan,am->efmn', g[v, v, v, o], t1)
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.5000 <j,i||m,n>*t2(e,f,j,i)
    Hd0 +=  0.500000000000000 * einsum('jimn,efji->efmn', g[o, o, o, o], t2)
    
    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t2(a,f,m,i)
    contracted_intermediate =  1.000000000000000 * einsum('iean,afmi->efmn', g[o, v, v, o], t2)
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.5000 <e,f||a,b>*t2(a,b,m,n)
    Hd0 +=  0.500000000000000 * einsum('efab,abmn->efmn', g[v, v, v, v], t2)
    
    #	  1.0000 P(m,n)<j,i||a,n>*t1(a,i)*t2(e,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jian,ai,efmj->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.5000 P(m,n)<j,i||a,n>*t1(a,m)*t2(e,f,j,i)
    contracted_intermediate =  0.500000000000000 * einsum('jian,am,efji->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)*P(e,f)<j,i||a,n>*t1(e,i)*t2(a,f,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('jian,ei,afmj->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)<i,e||a,b>*t1(a,i)*t2(b,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('ieab,ai,bfmn->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)*P(e,f)<i,e||a,b>*t1(a,n)*t2(b,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('ieab,an,bfmi->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.5000 P(e,f)<i,e||a,b>*t1(f,i)*t2(a,b,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('ieab,fi,abmn->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -1.0000 <j,i||m,n>*t1(e,i)*t1(f,j)
    Hd0 += -1.000000000000000 * einsum('jimn,ei,fj->efmn', g[o, o, o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t1(a,m)*t1(f,i)
    contracted_intermediate =  1.000000000000000 * einsum('iean,am,fi->efmn', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	 -1.0000 <e,f||a,b>*t1(a,n)*t1(b,m)
    Hd0 += -1.000000000000000 * einsum('efab,an,bm->efmn', g[v, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 P(m,n)<j,i||a,b>*t2(a,b,n,i)*t2(e,f,m,j)
    contracted_intermediate = -0.500000000000000 * einsum('jiab,abni,efmj->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.2500 <j,i||a,b>*t2(a,b,m,n)*t2(e,f,j,i)
    Hd0 +=  0.250000000000000 * einsum('jiab,abmn,efji->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||a,b>*t2(a,e,j,i)*t2(b,f,m,n)
    Hd0 += -0.500000000000000 * einsum('jiab,aeji,bfmn->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)<j,i||a,b>*t2(a,e,n,i)*t2(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,aeni,bfmj->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.5000 <j,i||a,b>*t2(a,e,m,n)*t2(b,f,j,i)
    Hd0 += -0.500000000000000 * einsum('jiab,aemn,bfji->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 P(m,n)<j,i||a,b>*t1(a,i)*t1(b,n)*t2(e,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,ai,bn,efmj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)<j,i||a,b>*t1(a,i)*t1(e,j)*t2(b,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,ai,ej,bfmn->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -0.5000 <j,i||a,b>*t1(a,n)*t1(b,m)*t2(e,f,j,i)
    Hd0 += -0.500000000000000 * einsum('jiab,an,bm,efji->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(m,n)*P(e,f)<j,i||a,b>*t1(a,n)*t1(e,i)*t2(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,an,ei,bfmj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	 -0.5000 <j,i||a,b>*t1(e,i)*t1(f,j)*t2(a,b,m,n)
    Hd0 += -0.500000000000000 * einsum('jiab,ei,fj,abmn->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(m,n)<j,i||a,n>*t1(a,m)*t1(e,i)*t1(f,j)
    contracted_intermediate = -1.000000000000000 * einsum('jian,am,ei,fj->efmn', g[o, o, v, o], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)<i,e||a,b>*t1(a,n)*t1(b,m)*t1(f,i)
    contracted_intermediate = -1.000000000000000 * einsum('ieab,an,bm,fi->efmn', g[o, v, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 <j,i||a,b>*t1(a,n)*t1(b,m)*t1(e,i)*t1(f,j)
    Hd0 +=  1.000000000000000 * einsum('jiab,an,bm,ei,fj->efmn', g[o, o, v, v], t1, t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    
    #    H(0;i,j,a,b) = <0| e(-T) H e(T) e2(a,b,j,i)|0>
    
    #	  1.0000 <i,j||a,b>
    H0d =  1.000000000000000 * einsum('ijab->abij', g[o, o, v, v])
    
    
    #    H(m,e;i,a) = <0|e1(m,e) e(-T) H e(T) e1(a,i)|0>
    
    #	  1.0000 d(e,a)*d(m,i)*f(j,j)
    Hss =  1.000000000000000 * einsum('ea,mi,jj->emai', kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*f(i,m)
    Hss += -1.000000000000000 * einsum('ea,im->emai', kd[v, v], f[o, o])
    
    #	  1.0000 d(m,i)*f(e,a)
    Hss +=  1.000000000000000 * einsum('mi,ea->emai', kd[o, o], f[v, v])
    
    #	  1.0000 d(e,a)*d(m,i)*f(j,b)*t1(b,j)
    Hss +=  1.000000000000000 * einsum('ea,mi,jb,bj->emai', kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*f(i,b)*t1(b,m)
    Hss += -1.000000000000000 * einsum('ea,ib,bm->emai', kd[v, v], f[o, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(m,i)*f(j,a)*t1(e,j)
    Hss += -1.000000000000000 * einsum('mi,ja,ej->emai', kd[o, o], f[o, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(m,i)*<k,j||k,j>
    Hss += -0.500000000000000 * einsum('ea,mi,kjkj->emai', kd[v, v], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <i,e||a,m>
    Hss +=  1.000000000000000 * einsum('ieam->emai', g[o, v, v, o])
    
    #	  1.0000 d(e,a)*<i,j||b,m>*t1(b,j)
    Hss +=  1.000000000000000 * einsum('ea,ijbm,bj->emai', kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 <i,j||a,m>*t1(e,j)
    Hss += -1.000000000000000 * einsum('ijam,ej->emai', g[o, o, v, o], t1)
    
    #	  1.0000 d(m,i)*<j,e||b,a>*t1(b,j)
    Hss +=  1.000000000000000 * einsum('mi,jeba,bj->emai', kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 <i,e||b,a>*t1(b,m)
    Hss += -1.000000000000000 * einsum('ieba,bm->emai', g[o, v, v, v], t1)
    
    #	  0.2500 d(e,a)*d(m,i)*<k,j||b,c>*t2(b,c,k,j)
    Hss +=  0.250000000000000 * einsum('ea,mi,kjbc,bckj->emai', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*<i,j||b,c>*t2(b,c,m,j)
    Hss += -0.500000000000000 * einsum('ea,ijbc,bcmj->emai', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(m,i)*<k,j||b,a>*t2(b,e,k,j)
    Hss += -0.500000000000000 * einsum('mi,kjba,bekj->emai', kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 <i,j||b,a>*t2(b,e,m,j)
    Hss +=  1.000000000000000 * einsum('ijba,bemj->emai', g[o, o, v, v], t2)
    
    #	 -0.5000 d(e,a)*d(m,i)*<k,j||b,c>*t1(b,j)*t1(c,k)
    Hss += -0.500000000000000 * einsum('ea,mi,kjbc,bj,ck->emai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (2, 3), (2, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*<i,j||b,c>*t1(b,j)*t1(c,m)
    Hss +=  1.000000000000000 * einsum('ea,ijbc,bj,cm->emai', kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(m,i)*<k,j||b,a>*t1(b,j)*t1(e,k)
    Hss +=  1.000000000000000 * einsum('mi,kjba,bj,ek->emai', kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <i,j||b,a>*t1(b,m)*t1(e,j)
    Hss +=  1.000000000000000 * einsum('ijba,bm,ej->emai', g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    
    #    H(m,e;i,j,a,b) = <0|e1(m,e) e(-T) H e(T) e2(a,b,j,i)|0>
    
    #	 -1.0000 d(e,b)*d(m,i)*f(j,a)
    Hsd = -1.000000000000000 * einsum('eb,mi,ja->emabij', kd[v, v], kd[o, o], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,b)*d(m,j)*f(i,a)
    Hsd +=  1.000000000000000 * einsum('eb,mj,ia->emabij', kd[v, v], kd[o, o], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,a)*d(m,i)*f(j,b)
    Hsd +=  1.000000000000000 * einsum('ea,mi,jb->emabij', kd[v, v], kd[o, o], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,a)*d(m,j)*f(i,b)
    Hsd += -1.000000000000000 * einsum('ea,mj,ib->emabij', kd[v, v], kd[o, o], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,b)*<i,j||a,m>
    Hsd += -1.000000000000000 * einsum('eb,ijam->emabij', kd[v, v], g[o, o, v, o])
    
    #	  1.0000 d(e,a)*<i,j||b,m>
    Hsd +=  1.000000000000000 * einsum('ea,ijbm->emabij', kd[v, v], g[o, o, v, o])
    
    #	 -1.0000 d(m,i)*<j,e||a,b>
    Hsd += -1.000000000000000 * einsum('mi,jeab->emabij', kd[o, o], g[o, v, v, v])
    
    #	  1.0000 d(m,j)*<i,e||a,b>
    Hsd +=  1.000000000000000 * einsum('mj,ieab->emabij', kd[o, o], g[o, v, v, v])
    
    #	  1.0000 d(e,b)*d(m,i)*<j,k||c,a>*t1(c,k)
    Hsd +=  1.000000000000000 * einsum('eb,mi,jkca,ck->emabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,j)*<i,k||c,a>*t1(c,k)
    Hsd += -1.000000000000000 * einsum('eb,mj,ikca,ck->emabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*<j,k||c,b>*t1(c,k)
    Hsd += -1.000000000000000 * einsum('ea,mi,jkcb,ck->emabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,j)*<i,k||c,b>*t1(c,k)
    Hsd +=  1.000000000000000 * einsum('ea,mj,ikcb,ck->emabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*<i,j||c,a>*t1(c,m)
    Hsd +=  1.000000000000000 * einsum('eb,ijca,cm->emabij', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*<i,j||c,b>*t1(c,m)
    Hsd += -1.000000000000000 * einsum('ea,ijcb,cm->emabij', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(m,i)*<j,k||a,b>*t1(e,k)
    Hsd +=  1.000000000000000 * einsum('mi,jkab,ek->emabij', kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(m,j)*<i,k||a,b>*t1(e,k)
    Hsd += -1.000000000000000 * einsum('mj,ikab,ek->emabij', kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    
    #    H(m,n,e,f;i,a) = <0|e2(m,n,f,e) e(-T) H e(T) e1(a,i)|0>
    
    #	 -1.0000 d(f,a)*d(m,i)*f(e,n)
    Hds = -1.000000000000000 * einsum('fa,mi,en->efmnai', kd[v, v], kd[o, o], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,a)*d(n,i)*f(e,m)
    Hds +=  1.000000000000000 * einsum('fa,ni,em->efmnai', kd[v, v], kd[o, o], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,a)*d(m,i)*f(f,n)
    Hds +=  1.000000000000000 * einsum('ea,mi,fn->efmnai', kd[v, v], kd[o, o], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,a)*d(n,i)*f(f,m)
    Hds += -1.000000000000000 * einsum('ea,ni,fm->efmnai', kd[v, v], kd[o, o], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,a)*d(m,i)*f(j,n)*t1(e,j)
    Hds +=  1.000000000000000 * einsum('fa,mi,jn,ej->efmnai', kd[v, v], kd[o, o], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*f(j,n)*t1(f,j)
    Hds += -1.000000000000000 * einsum('ea,mi,jn,fj->efmnai', kd[v, v], kd[o, o], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,i)*f(j,m)*t1(e,j)
    Hds += -1.000000000000000 * einsum('fa,ni,jm,ej->efmnai', kd[v, v], kd[o, o], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,i)*f(j,m)*t1(f,j)
    Hds +=  1.000000000000000 * einsum('ea,ni,jm,fj->efmnai', kd[v, v], kd[o, o], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*f(e,b)*t1(b,n)
    Hds += -1.000000000000000 * einsum('fa,mi,eb,bn->efmnai', kd[v, v], kd[o, o], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*f(e,b)*t1(b,m)
    Hds +=  1.000000000000000 * einsum('fa,ni,eb,bm->efmnai', kd[v, v], kd[o, o], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*f(f,b)*t1(b,n)
    Hds +=  1.000000000000000 * einsum('ea,mi,fb,bn->efmnai', kd[v, v], kd[o, o], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*f(f,b)*t1(b,m)
    Hds += -1.000000000000000 * einsum('ea,ni,fb,bm->efmnai', kd[v, v], kd[o, o], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,i)*f(j,b)*t2(b,e,n,j)
    Hds +=  1.000000000000000 * einsum('fa,mi,jb,benj->efmnai', kd[v, v], kd[o, o], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,i)*f(j,b)*t2(b,e,m,j)
    Hds += -1.000000000000000 * einsum('fa,ni,jb,bemj->efmnai', kd[v, v], kd[o, o], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*f(i,b)*t2(b,e,m,n)
    Hds +=  1.000000000000000 * einsum('fa,ib,bemn->efmnai', kd[v, v], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*f(j,b)*t2(b,f,n,j)
    Hds += -1.000000000000000 * einsum('ea,mi,jb,bfnj->efmnai', kd[v, v], kd[o, o], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,i)*f(j,b)*t2(b,f,m,j)
    Hds +=  1.000000000000000 * einsum('ea,ni,jb,bfmj->efmnai', kd[v, v], kd[o, o], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*f(i,b)*t2(b,f,m,n)
    Hds += -1.000000000000000 * einsum('ea,ib,bfmn->efmnai', kd[v, v], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(m,i)*f(j,a)*t2(e,f,n,j)
    Hds +=  1.000000000000000 * einsum('mi,ja,efnj->efmnai', kd[o, o], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(n,i)*f(j,a)*t2(e,f,m,j)
    Hds += -1.000000000000000 * einsum('ni,ja,efmj->efmnai', kd[o, o], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,i)*f(j,b)*t1(b,n)*t1(e,j)
    Hds +=  1.000000000000000 * einsum('fa,mi,jb,bn,ej->efmnai', kd[v, v], kd[o, o], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*f(j,b)*t1(b,n)*t1(f,j)
    Hds += -1.000000000000000 * einsum('ea,mi,jb,bn,fj->efmnai', kd[v, v], kd[o, o], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,i)*f(j,b)*t1(b,m)*t1(e,j)
    Hds += -1.000000000000000 * einsum('fa,ni,jb,bm,ej->efmnai', kd[v, v], kd[o, o], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,i)*f(j,b)*t1(b,m)*t1(f,j)
    Hds +=  1.000000000000000 * einsum('ea,ni,jb,bm,fj->efmnai', kd[v, v], kd[o, o], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*<i,e||m,n>
    Hds +=  1.000000000000000 * einsum('fa,iemn->efmnai', kd[v, v], g[o, v, o, o])
    
    #	 -1.0000 d(e,a)*<i,f||m,n>
    Hds += -1.000000000000000 * einsum('ea,ifmn->efmnai', kd[v, v], g[o, v, o, o])
    
    #	  1.0000 d(m,i)*<e,f||a,n>
    Hds +=  1.000000000000000 * einsum('mi,efan->efmnai', kd[o, o], g[v, v, v, o])
    
    #	 -1.0000 d(n,i)*<e,f||a,m>
    Hds += -1.000000000000000 * einsum('ni,efam->efmnai', kd[o, o], g[v, v, v, o])
    
    #	 -1.0000 d(f,a)*<i,j||m,n>*t1(e,j)
    Hds += -1.000000000000000 * einsum('fa,ijmn,ej->efmnai', kd[v, v], g[o, o, o, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*<i,j||m,n>*t1(f,j)
    Hds +=  1.000000000000000 * einsum('ea,ijmn,fj->efmnai', kd[v, v], g[o, o, o, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<j,e||b,n>*t1(b,j)
    Hds += -1.000000000000000 * einsum('fa,mi,jebn,bj->efmnai', kd[v, v], kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)d(f,a)*<i,e||b,n>*t1(b,m)
    contracted_intermediate =  1.000000000000000 * einsum('fa,iebn,bm->efmnai', kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	  1.0000 P(e,f)d(m,i)*<j,e||a,n>*t1(f,j)
    contracted_intermediate =  1.000000000000000 * einsum('mi,jean,fj->efmnai', kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 d(f,a)*d(n,i)*<j,e||b,m>*t1(b,j)
    Hds +=  1.000000000000000 * einsum('fa,ni,jebm,bj->efmnai', kd[v, v], kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 P(e,f)d(n,i)*<j,e||a,m>*t1(f,j)
    contracted_intermediate = -1.000000000000000 * einsum('ni,jeam,fj->efmnai', kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 d(e,a)*d(m,i)*<j,f||b,n>*t1(b,j)
    Hds +=  1.000000000000000 * einsum('ea,mi,jfbn,bj->efmnai', kd[v, v], kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 P(m,n)d(e,a)*<i,f||b,n>*t1(b,m)
    contracted_intermediate = -1.000000000000000 * einsum('ea,ifbn,bm->efmnai', kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	 -1.0000 d(e,a)*d(n,i)*<j,f||b,m>*t1(b,j)
    Hds += -1.000000000000000 * einsum('ea,ni,jfbm,bj->efmnai', kd[v, v], kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(m,i)*<e,f||b,a>*t1(b,n)
    Hds += -1.000000000000000 * einsum('mi,efba,bn->efmnai', kd[o, o], g[v, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(n,i)*<e,f||b,a>*t1(b,m)
    Hds +=  1.000000000000000 * einsum('ni,efba,bm->efmnai', kd[o, o], g[v, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(m,i)*<k,j||b,n>*t2(b,e,k,j)
    Hds +=  0.500000000000000 * einsum('fa,mi,kjbn,bekj->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 P(m,n)d(f,a)*<i,j||b,n>*t2(b,e,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('fa,ijbn,bemj->efmnai', kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	 -0.5000 d(e,a)*d(m,i)*<k,j||b,n>*t2(b,f,k,j)
    Hds += -0.500000000000000 * einsum('ea,mi,kjbn,bfkj->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)d(e,a)*<i,j||b,n>*t2(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('ea,ijbn,bfmj->efmnai', kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	  0.5000 d(m,i)*<k,j||a,n>*t2(e,f,k,j)
    Hds +=  0.500000000000000 * einsum('mi,kjan,efkj->efmnai', kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 P(m,n)<i,j||a,n>*t2(e,f,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('ijan,efmj->efmnai', g[o, o, v, o], t2)
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	 -0.5000 d(f,a)*d(n,i)*<k,j||b,m>*t2(b,e,k,j)
    Hds += -0.500000000000000 * einsum('fa,ni,kjbm,bekj->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(e,a)*d(n,i)*<k,j||b,m>*t2(b,f,k,j)
    Hds +=  0.500000000000000 * einsum('ea,ni,kjbm,bfkj->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(n,i)*<k,j||a,m>*t2(e,f,k,j)
    Hds += -0.500000000000000 * einsum('ni,kjam,efkj->efmnai', kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(m,i)*<j,e||b,c>*t2(b,c,n,j)
    Hds +=  0.500000000000000 * einsum('fa,mi,jebc,bcnj->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(n,i)*<j,e||b,c>*t2(b,c,m,j)
    Hds += -0.500000000000000 * einsum('fa,ni,jebc,bcmj->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(f,a)*<i,e||b,c>*t2(b,c,m,n)
    Hds +=  0.500000000000000 * einsum('fa,iebc,bcmn->efmnai', kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 P(e,f)d(m,i)*<j,e||b,a>*t2(b,f,n,j)
    contracted_intermediate = -1.000000000000000 * einsum('mi,jeba,bfnj->efmnai', kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 P(e,f)d(n,i)*<j,e||b,a>*t2(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('ni,jeba,bfmj->efmnai', kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)<i,e||b,a>*t2(b,f,m,n)
    contracted_intermediate = -1.000000000000000 * einsum('ieba,bfmn->efmnai', g[o, v, v, v], t2)
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	 -0.5000 d(e,a)*d(m,i)*<j,f||b,c>*t2(b,c,n,j)
    Hds += -0.500000000000000 * einsum('ea,mi,jfbc,bcnj->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(e,a)*d(n,i)*<j,f||b,c>*t2(b,c,m,j)
    Hds +=  0.500000000000000 * einsum('ea,ni,jfbc,bcmj->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(e,a)*<i,f||b,c>*t2(b,c,m,n)
    Hds += -0.500000000000000 * einsum('ea,ifbc,bcmn->efmnai', kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<k,j||b,c>*t1(b,j)*t2(c,e,n,k)
    Hds += -1.000000000000000 * einsum('fa,mi,kjbc,bj,cenk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<k,j||b,c>*t1(b,j)*t2(c,e,m,k)
    Hds +=  1.000000000000000 * einsum('fa,ni,kjbc,bj,cemk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*<i,j||b,c>*t1(b,j)*t2(c,e,m,n)
    Hds += -1.000000000000000 * einsum('fa,ijbc,bj,cemn->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*<k,j||b,c>*t1(b,j)*t2(c,f,n,k)
    Hds +=  1.000000000000000 * einsum('ea,mi,kjbc,bj,cfnk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<k,j||b,c>*t1(b,j)*t2(c,f,m,k)
    Hds += -1.000000000000000 * einsum('ea,ni,kjbc,bj,cfmk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*<i,j||b,c>*t1(b,j)*t2(c,f,m,n)
    Hds +=  1.000000000000000 * einsum('ea,ijbc,bj,cfmn->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(m,i)*<k,j||b,a>*t1(b,j)*t2(e,f,n,k)
    Hds += -1.000000000000000 * einsum('mi,kjba,bj,efnk->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(n,i)*<k,j||b,a>*t1(b,j)*t2(e,f,m,k)
    Hds +=  1.000000000000000 * einsum('ni,kjba,bj,efmk->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(m,i)*<k,j||b,c>*t1(b,n)*t2(c,e,k,j)
    Hds += -0.500000000000000 * einsum('fa,mi,kjbc,bn,cekj->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 P(m,n)d(f,a)*<i,j||b,c>*t1(b,n)*t2(c,e,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('fa,ijbc,bn,cemj->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	  0.5000 d(e,a)*d(m,i)*<k,j||b,c>*t1(b,n)*t2(c,f,k,j)
    Hds +=  0.500000000000000 * einsum('ea,mi,kjbc,bn,cfkj->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 P(m,n)d(e,a)*<i,j||b,c>*t1(b,n)*t2(c,f,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('ea,ijbc,bn,cfmj->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	 -0.5000 d(m,i)*<k,j||b,a>*t1(b,n)*t2(e,f,k,j)
    Hds += -0.500000000000000 * einsum('mi,kjba,bn,efkj->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 P(m,n)<i,j||b,a>*t1(b,n)*t2(e,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('ijba,bn,efmj->efmnai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	  0.5000 d(f,a)*d(n,i)*<k,j||b,c>*t1(b,m)*t2(c,e,k,j)
    Hds +=  0.500000000000000 * einsum('fa,ni,kjbc,bm,cekj->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(n,i)*<k,j||b,c>*t1(b,m)*t2(c,f,k,j)
    Hds += -0.500000000000000 * einsum('ea,ni,kjbc,bm,cfkj->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(n,i)*<k,j||b,a>*t1(b,m)*t2(e,f,k,j)
    Hds +=  0.500000000000000 * einsum('ni,kjba,bm,efkj->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(m,i)*<k,j||b,c>*t1(e,j)*t2(b,c,n,k)
    Hds += -0.500000000000000 * einsum('fa,mi,kjbc,ej,bcnk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(n,i)*<k,j||b,c>*t1(e,j)*t2(b,c,m,k)
    Hds +=  0.500000000000000 * einsum('fa,ni,kjbc,ej,bcmk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*<i,j||b,c>*t1(e,j)*t2(b,c,m,n)
    Hds += -0.500000000000000 * einsum('fa,ijbc,ej,bcmn->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 P(e,f)d(m,i)*<k,j||b,a>*t1(e,j)*t2(b,f,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('mi,kjba,ej,bfnk->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)d(n,i)*<k,j||b,a>*t1(e,j)*t2(b,f,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('ni,kjba,ej,bfmk->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 P(e,f)<i,j||b,a>*t1(e,j)*t2(b,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('ijba,ej,bfmn->efmnai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  0.5000 d(e,a)*d(m,i)*<k,j||b,c>*t1(f,j)*t2(b,c,n,k)
    Hds +=  0.500000000000000 * einsum('ea,mi,kjbc,fj,bcnk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(n,i)*<k,j||b,c>*t1(f,j)*t2(b,c,m,k)
    Hds += -0.500000000000000 * einsum('ea,ni,kjbc,fj,bcmk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*<i,j||b,c>*t1(f,j)*t2(b,c,m,n)
    Hds +=  0.500000000000000 * einsum('ea,ijbc,fj,bcmn->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<k,j||b,n>*t1(b,j)*t1(e,k)
    Hds += -1.000000000000000 * einsum('fa,mi,kjbn,bj,ek->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*<k,j||b,n>*t1(b,j)*t1(f,k)
    Hds +=  1.000000000000000 * einsum('ea,mi,kjbn,bj,fk->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(m,n)d(f,a)*<i,j||b,n>*t1(b,m)*t1(e,j)
    contracted_intermediate = -1.000000000000000 * einsum('fa,ijbn,bm,ej->efmnai', kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	  1.0000 P(m,n)d(e,a)*<i,j||b,n>*t1(b,m)*t1(f,j)
    contracted_intermediate =  1.000000000000000 * einsum('ea,ijbn,bm,fj->efmnai', kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	 -1.0000 d(m,i)*<k,j||a,n>*t1(e,j)*t1(f,k)
    Hds += -1.000000000000000 * einsum('mi,kjan,ej,fk->efmnai', kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<k,j||b,m>*t1(b,j)*t1(e,k)
    Hds +=  1.000000000000000 * einsum('fa,ni,kjbm,bj,ek->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<k,j||b,m>*t1(b,j)*t1(f,k)
    Hds += -1.000000000000000 * einsum('ea,ni,kjbm,bj,fk->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(n,i)*<k,j||a,m>*t1(e,j)*t1(f,k)
    Hds +=  1.000000000000000 * einsum('ni,kjam,ej,fk->efmnai', kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<j,e||b,c>*t1(b,j)*t1(c,n)
    Hds += -1.000000000000000 * einsum('fa,mi,jebc,bj,cn->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<j,e||b,c>*t1(b,j)*t1(c,m)
    Hds +=  1.000000000000000 * einsum('fa,ni,jebc,bj,cm->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*<i,e||b,c>*t1(b,n)*t1(c,m)
    Hds += -1.000000000000000 * einsum('fa,iebc,bn,cm->efmnai', kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 P(e,f)d(m,i)*<j,e||b,a>*t1(b,n)*t1(f,j)
    contracted_intermediate = -1.000000000000000 * einsum('mi,jeba,bn,fj->efmnai', kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 P(e,f)d(n,i)*<j,e||b,a>*t1(b,m)*t1(f,j)
    contracted_intermediate =  1.000000000000000 * einsum('ni,jeba,bm,fj->efmnai', kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 d(e,a)*d(m,i)*<j,f||b,c>*t1(b,j)*t1(c,n)
    Hds +=  1.000000000000000 * einsum('ea,mi,jfbc,bj,cn->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<j,f||b,c>*t1(b,j)*t1(c,m)
    Hds += -1.000000000000000 * einsum('ea,ni,jfbc,bj,cm->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*<i,f||b,c>*t1(b,n)*t1(c,m)
    Hds +=  1.000000000000000 * einsum('ea,ifbc,bn,cm->efmnai', kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<k,j||b,c>*t1(b,j)*t1(c,n)*t1(e,k)
    Hds += -1.000000000000000 * einsum('fa,mi,kjbc,bj,cn,ek->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*<k,j||b,c>*t1(b,j)*t1(c,n)*t1(f,k)
    Hds +=  1.000000000000000 * einsum('ea,mi,kjbc,bj,cn,fk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<k,j||b,c>*t1(b,j)*t1(c,m)*t1(e,k)
    Hds +=  1.000000000000000 * einsum('fa,ni,kjbc,bj,cm,ek->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<k,j||b,c>*t1(b,j)*t1(c,m)*t1(f,k)
    Hds += -1.000000000000000 * einsum('ea,ni,kjbc,bj,cm,fk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*<i,j||b,c>*t1(b,n)*t1(c,m)*t1(e,j)
    Hds +=  1.000000000000000 * einsum('fa,ijbc,bn,cm,ej->efmnai', kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*<i,j||b,c>*t1(b,n)*t1(c,m)*t1(f,j)
    Hds += -1.000000000000000 * einsum('ea,ijbc,bn,cm,fj->efmnai', kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    #	  1.0000 d(m,i)*<k,j||b,a>*t1(b,n)*t1(e,j)*t1(f,k)
    Hds +=  1.000000000000000 * einsum('mi,kjba,bn,ej,fk->efmnai', kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    #	 -1.0000 d(n,i)*<k,j||b,a>*t1(b,m)*t1(e,j)*t1(f,k)
    Hds += -1.000000000000000 * einsum('ni,kjba,bm,ej,fk->efmnai', kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    
    #    H(m,n,e,f;i,j,a,b) = <0|e2(m,n,f,e) e(-T) H e(T) e2(a,b,j,i)|0>
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,j)*d(m,i)*f(k,k)
    Hdd =  1.000000000000000 * einsum('ea,fb,nj,mi,kk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,j)*d(n,i)*f(k,k)
    Hdd += -1.000000000000000 * einsum('ea,fb,mj,ni,kk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,j)*d(m,i)*f(k,k)
    Hdd += -1.000000000000000 * einsum('fa,eb,nj,mi,kk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,j)*d(n,i)*f(k,k)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mj,ni,kk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,i)*f(j,n)
    Hdd += -1.000000000000000 * einsum('ea,fb,mi,jn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(m,j)*f(i,n)
    Hdd +=  1.000000000000000 * einsum('ea,fb,mj,in->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,i)*f(j,n)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mi,jn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(m,j)*f(i,n)
    Hdd += -1.000000000000000 * einsum('fa,eb,mj,in->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,i)*f(j,m)
    Hdd +=  1.000000000000000 * einsum('ea,fb,ni,jm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(n,j)*f(i,m)
    Hdd += -1.000000000000000 * einsum('ea,fb,nj,im->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,i)*f(j,m)
    Hdd += -1.000000000000000 * einsum('fa,eb,ni,jm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(n,j)*f(i,m)
    Hdd +=  1.000000000000000 * einsum('fa,eb,nj,im->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(f,b)*d(n,j)*d(m,i)*f(e,a)
    Hdd +=  1.000000000000000 * einsum('fb,nj,mi,ea->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(f,b)*d(m,j)*d(n,i)*f(e,a)
    Hdd += -1.000000000000000 * einsum('fb,mj,ni,ea->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(f,a)*d(n,j)*d(m,i)*f(e,b)
    Hdd += -1.000000000000000 * einsum('fa,nj,mi,eb->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(f,a)*d(m,j)*d(n,i)*f(e,b)
    Hdd +=  1.000000000000000 * einsum('fa,mj,ni,eb->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(e,b)*d(n,j)*d(m,i)*f(f,a)
    Hdd += -1.000000000000000 * einsum('eb,nj,mi,fa->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(e,b)*d(m,j)*d(n,i)*f(f,a)
    Hdd +=  1.000000000000000 * einsum('eb,mj,ni,fa->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(e,a)*d(n,j)*d(m,i)*f(f,b)
    Hdd +=  1.000000000000000 * einsum('ea,nj,mi,fb->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(e,a)*d(m,j)*d(n,i)*f(f,b)
    Hdd += -1.000000000000000 * einsum('ea,mj,ni,fb->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,j)*d(m,i)*f(k,c)*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('ea,fb,nj,mi,kc,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,j)*d(n,i)*f(k,c)*t1(c,k)
    Hdd += -1.000000000000000 * einsum('ea,fb,mj,ni,kc,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,j)*d(m,i)*f(k,c)*t1(c,k)
    Hdd += -1.000000000000000 * einsum('fa,eb,nj,mi,kc,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,j)*d(n,i)*f(k,c)*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mj,ni,kc,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,i)*f(j,c)*t1(c,n)
    Hdd += -1.000000000000000 * einsum('ea,fb,mi,jc,cn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(m,j)*f(i,c)*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('ea,fb,mj,ic,cn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,i)*f(j,c)*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mi,jc,cn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(m,j)*f(i,c)*t1(c,n)
    Hdd += -1.000000000000000 * einsum('fa,eb,mj,ic,cn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,i)*f(j,c)*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('ea,fb,ni,jc,cm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(n,j)*f(i,c)*t1(c,m)
    Hdd += -1.000000000000000 * einsum('ea,fb,nj,ic,cm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,i)*f(j,c)*t1(c,m)
    Hdd += -1.000000000000000 * einsum('fa,eb,ni,jc,cm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(n,j)*f(i,c)*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('fa,eb,nj,ic,cm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(n,j)*d(m,i)*f(k,a)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fb,nj,mi,ka,ek->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(m,j)*d(n,i)*f(k,a)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fb,mj,ni,ka,ek->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,j)*d(m,i)*f(k,b)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fa,nj,mi,kb,ek->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,j)*d(n,i)*f(k,b)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fa,mj,ni,kb,ek->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(n,j)*d(m,i)*f(k,a)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('eb,nj,mi,ka,fk->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,j)*d(n,i)*f(k,a)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('eb,mj,ni,ka,fk->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,j)*d(m,i)*f(k,b)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('ea,nj,mi,kb,fk->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,j)*d(n,i)*f(k,b)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('ea,mj,ni,kb,fk->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(f,b)*d(n,j)*d(m,i)*<l,k||l,k>
    Hdd += -0.500000000000000 * einsum('ea,fb,nj,mi,lklk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(f,b)*d(m,j)*d(n,i)*<l,k||l,k>
    Hdd +=  0.500000000000000 * einsum('ea,fb,mj,ni,lklk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(e,b)*d(n,j)*d(m,i)*<l,k||l,k>
    Hdd +=  0.500000000000000 * einsum('fa,eb,nj,mi,lklk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(e,b)*d(m,j)*d(n,i)*<l,k||l,k>
    Hdd += -0.500000000000000 * einsum('fa,eb,mj,ni,lklk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*<i,j||m,n>
    Hdd +=  1.000000000000000 * einsum('ea,fb,ijmn->efmnabij', kd[v, v], kd[v, v], g[o, o, o, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(f,a)*d(e,b)*<i,j||m,n>
    Hdd += -1.000000000000000 * einsum('fa,eb,ijmn->efmnabij', kd[v, v], kd[v, v], g[o, o, o, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,b)*d(m,i)*<j,e||a,n>
    Hdd +=  1.000000000000000 * einsum('fb,mi,jean->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(f,b)*d(m,j)*<i,e||a,n>
    Hdd += -1.000000000000000 * einsum('fb,mj,iean->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<j,e||b,n>
    Hdd += -1.000000000000000 * einsum('fa,mi,jebn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,a)*d(m,j)*<i,e||b,n>
    Hdd +=  1.000000000000000 * einsum('fa,mj,iebn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(f,b)*d(n,i)*<j,e||a,m>
    Hdd += -1.000000000000000 * einsum('fb,ni,jeam->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,b)*d(n,j)*<i,e||a,m>
    Hdd +=  1.000000000000000 * einsum('fb,nj,ieam->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,a)*d(n,i)*<j,e||b,m>
    Hdd +=  1.000000000000000 * einsum('fa,ni,jebm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(f,a)*d(n,j)*<i,e||b,m>
    Hdd += -1.000000000000000 * einsum('fa,nj,iebm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,b)*d(m,i)*<j,f||a,n>
    Hdd += -1.000000000000000 * einsum('eb,mi,jfan->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,b)*d(m,j)*<i,f||a,n>
    Hdd +=  1.000000000000000 * einsum('eb,mj,ifan->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,a)*d(m,i)*<j,f||b,n>
    Hdd +=  1.000000000000000 * einsum('ea,mi,jfbn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,a)*d(m,j)*<i,f||b,n>
    Hdd += -1.000000000000000 * einsum('ea,mj,ifbn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,b)*d(n,i)*<j,f||a,m>
    Hdd +=  1.000000000000000 * einsum('eb,ni,jfam->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,b)*d(n,j)*<i,f||a,m>
    Hdd += -1.000000000000000 * einsum('eb,nj,ifam->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<j,f||b,m>
    Hdd += -1.000000000000000 * einsum('ea,ni,jfbm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,a)*d(n,j)*<i,f||b,m>
    Hdd +=  1.000000000000000 * einsum('ea,nj,ifbm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(n,j)*d(m,i)*<e,f||a,b>
    Hdd +=  1.000000000000000 * einsum('nj,mi,efab->efmnabij', kd[o, o], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(m,j)*d(n,i)*<e,f||a,b>
    Hdd += -1.000000000000000 * einsum('mj,ni,efab->efmnabij', kd[o, o], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(m,i)*<j,k||c,n>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('ea,fb,mi,jkcn,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,j)*<i,k||c,n>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('ea,fb,mj,ikcn,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(m,i)*<j,k||c,n>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('fa,eb,mi,jkcn,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,j)*<i,k||c,n>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mj,ikcn,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 P(m,n)d(e,a)*d(f,b)*<i,j||c,n>*t1(c,m)
    contracted_intermediate =  1.000000000000000 * einsum('ea,fb,ijcn,cm->efmnabij', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    Hdd +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnabij->efnmabij', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)d(f,a)*d(e,b)*<i,j||c,n>*t1(c,m)
    contracted_intermediate = -1.000000000000000 * einsum('fa,eb,ijcn,cm->efmnabij', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    Hdd +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnabij->efnmabij', contracted_intermediate) 
    
    #	 -1.0000 d(f,b)*d(m,i)*<j,k||a,n>*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fb,mi,jkan,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,b)*d(m,j)*<i,k||a,n>*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fb,mj,ikan,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,i)*<j,k||b,n>*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fa,mi,jkbn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,j)*<i,k||b,n>*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fa,mj,ikbn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,i)*<j,k||a,n>*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('eb,mi,jkan,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,j)*<i,k||a,n>*t1(f,k)
    Hdd += -1.000000000000000 * einsum('eb,mj,ikan,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*<j,k||b,n>*t1(f,k)
    Hdd += -1.000000000000000 * einsum('ea,mi,jkbn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,j)*<i,k||b,n>*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('ea,mj,ikbn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(n,i)*<j,k||c,m>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('ea,fb,ni,jkcm,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,j)*<i,k||c,m>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('ea,fb,nj,ikcm,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(n,i)*<j,k||c,m>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('fa,eb,ni,jkcm,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,j)*<i,k||c,m>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('fa,eb,nj,ikcm,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,i)*<j,k||a,m>*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fb,ni,jkam,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(n,j)*<i,k||a,m>*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fb,nj,ikam,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,i)*<j,k||b,m>*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fa,ni,jkbm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,j)*<i,k||b,m>*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fa,nj,ikbm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(n,i)*<j,k||a,m>*t1(f,k)
    Hdd += -1.000000000000000 * einsum('eb,ni,jkam,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*d(n,j)*<i,k||a,m>*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('eb,nj,ikam,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,i)*<j,k||b,m>*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('ea,ni,jkbm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,j)*<i,k||b,m>*t1(f,k)
    Hdd += -1.000000000000000 * einsum('ea,nj,ikbm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,j)*d(m,i)*<k,e||c,a>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('fb,nj,mi,keca,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(m,j)*d(n,i)*<k,e||c,a>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('fb,mj,ni,keca,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,j)*d(m,i)*<k,e||c,b>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('fa,nj,mi,kecb,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,j)*d(n,i)*<k,e||c,b>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('fa,mj,ni,kecb,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(m,i)*<j,e||c,a>*t1(c,n)
    Hdd += -1.000000000000000 * einsum('fb,mi,jeca,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,b)*d(m,j)*<i,e||c,a>*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('fb,mj,ieca,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,i)*<j,e||c,b>*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('fa,mi,jecb,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,j)*<i,e||c,b>*t1(c,n)
    Hdd += -1.000000000000000 * einsum('fa,mj,iecb,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,i)*<j,e||c,a>*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('fb,ni,jeca,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(n,j)*<i,e||c,a>*t1(c,m)
    Hdd += -1.000000000000000 * einsum('fb,nj,ieca,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,i)*<j,e||c,b>*t1(c,m)
    Hdd += -1.000000000000000 * einsum('fa,ni,jecb,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,j)*<i,e||c,b>*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('fa,nj,iecb,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 P(e,f)d(n,j)*d(m,i)*<k,e||a,b>*t1(f,k)
    contracted_intermediate =  1.000000000000000 * einsum('nj,mi,keab,fk->efmnabij', kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    Hdd +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnabij->femnabij', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)d(m,j)*d(n,i)*<k,e||a,b>*t1(f,k)
    contracted_intermediate = -1.000000000000000 * einsum('mj,ni,keab,fk->efmnabij', kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    Hdd +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnabij->femnabij', contracted_intermediate) 
    
    #	 -1.0000 d(e,b)*d(n,j)*d(m,i)*<k,f||c,a>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('eb,nj,mi,kfca,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,j)*d(n,i)*<k,f||c,a>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('eb,mj,ni,kfca,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,j)*d(m,i)*<k,f||c,b>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('ea,nj,mi,kfcb,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,j)*d(n,i)*<k,f||c,b>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('ea,mj,ni,kfcb,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,i)*<j,f||c,a>*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('eb,mi,jfca,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,j)*<i,f||c,a>*t1(c,n)
    Hdd += -1.000000000000000 * einsum('eb,mj,ifca,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*<j,f||c,b>*t1(c,n)
    Hdd += -1.000000000000000 * einsum('ea,mi,jfcb,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,j)*<i,f||c,b>*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('ea,mj,ifcb,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(n,i)*<j,f||c,a>*t1(c,m)
    Hdd += -1.000000000000000 * einsum('eb,ni,jfca,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*d(n,j)*<i,f||c,a>*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('eb,nj,ifca,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,i)*<j,f||c,b>*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('ea,ni,jfcb,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,j)*<i,f||c,b>*t1(c,m)
    Hdd += -1.000000000000000 * einsum('ea,nj,ifcb,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 d(e,a)*d(f,b)*d(n,j)*d(m,i)*<l,k||c,d>*t2(c,d,l,k)
    Hdd +=  0.250000000000000 * einsum('ea,fb,nj,mi,lkcd,cdlk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 d(e,a)*d(f,b)*d(m,j)*d(n,i)*<l,k||c,d>*t2(c,d,l,k)
    Hdd += -0.250000000000000 * einsum('ea,fb,mj,ni,lkcd,cdlk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 d(f,a)*d(e,b)*d(n,j)*d(m,i)*<l,k||c,d>*t2(c,d,l,k)
    Hdd += -0.250000000000000 * einsum('fa,eb,nj,mi,lkcd,cdlk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 d(f,a)*d(e,b)*d(m,j)*d(n,i)*<l,k||c,d>*t2(c,d,l,k)
    Hdd +=  0.250000000000000 * einsum('fa,eb,mj,ni,lkcd,cdlk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(f,b)*d(m,i)*<j,k||c,d>*t2(c,d,n,k)
    Hdd += -0.500000000000000 * einsum('ea,fb,mi,jkcd,cdnk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(f,b)*d(m,j)*<i,k||c,d>*t2(c,d,n,k)
    Hdd +=  0.500000000000000 * einsum('ea,fb,mj,ikcd,cdnk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(e,b)*d(m,i)*<j,k||c,d>*t2(c,d,n,k)
    Hdd +=  0.500000000000000 * einsum('fa,eb,mi,jkcd,cdnk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(e,b)*d(m,j)*<i,k||c,d>*t2(c,d,n,k)
    Hdd += -0.500000000000000 * einsum('fa,eb,mj,ikcd,cdnk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(f,b)*d(n,i)*<j,k||c,d>*t2(c,d,m,k)
    Hdd +=  0.500000000000000 * einsum('ea,fb,ni,jkcd,cdmk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(f,b)*d(n,j)*<i,k||c,d>*t2(c,d,m,k)
    Hdd += -0.500000000000000 * einsum('ea,fb,nj,ikcd,cdmk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(e,b)*d(n,i)*<j,k||c,d>*t2(c,d,m,k)
    Hdd += -0.500000000000000 * einsum('fa,eb,ni,jkcd,cdmk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(e,b)*d(n,j)*<i,k||c,d>*t2(c,d,m,k)
    Hdd +=  0.500000000000000 * einsum('fa,eb,nj,ikcd,cdmk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(f,b)*<i,j||c,d>*t2(c,d,m,n)
    Hdd +=  0.500000000000000 * einsum('ea,fb,ijcd,cdmn->efmnabij', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(e,b)*<i,j||c,d>*t2(c,d,m,n)
    Hdd += -0.500000000000000 * einsum('fa,eb,ijcd,cdmn->efmnabij', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(f,b)*d(n,j)*d(m,i)*<l,k||c,a>*t2(c,e,l,k)
    Hdd += -0.500000000000000 * einsum('fb,nj,mi,lkca,celk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,b)*d(m,j)*d(n,i)*<l,k||c,a>*t2(c,e,l,k)
    Hdd +=  0.500000000000000 * einsum('fb,mj,ni,lkca,celk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(n,j)*d(m,i)*<l,k||c,b>*t2(c,e,l,k)
    Hdd +=  0.500000000000000 * einsum('fa,nj,mi,lkcb,celk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(m,j)*d(n,i)*<l,k||c,b>*t2(c,e,l,k)
    Hdd += -0.500000000000000 * einsum('fa,mj,ni,lkcb,celk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(m,i)*<j,k||c,a>*t2(c,e,n,k)
    Hdd +=  1.000000000000000 * einsum('fb,mi,jkca,cenk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(m,j)*<i,k||c,a>*t2(c,e,n,k)
    Hdd += -1.000000000000000 * einsum('fb,mj,ikca,cenk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<j,k||c,b>*t2(c,e,n,k)
    Hdd += -1.000000000000000 * einsum('fa,mi,jkcb,cenk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,j)*<i,k||c,b>*t2(c,e,n,k)
    Hdd +=  1.000000000000000 * einsum('fa,mj,ikcb,cenk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(n,i)*<j,k||c,a>*t2(c,e,m,k)
    Hdd += -1.000000000000000 * einsum('fb,ni,jkca,cemk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,j)*<i,k||c,a>*t2(c,e,m,k)
    Hdd +=  1.000000000000000 * einsum('fb,nj,ikca,cemk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<j,k||c,b>*t2(c,e,m,k)
    Hdd +=  1.000000000000000 * einsum('fa,ni,jkcb,cemk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,j)*<i,k||c,b>*t2(c,e,m,k)
    Hdd += -1.000000000000000 * einsum('fa,nj,ikcb,cemk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,b)*<i,j||c,a>*t2(c,e,m,n)
    Hdd += -1.000000000000000 * einsum('fb,ijca,cemn->efmnabij', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*<i,j||c,b>*t2(c,e,m,n)
    Hdd +=  1.000000000000000 * einsum('fa,ijcb,cemn->efmnabij', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(e,b)*d(n,j)*d(m,i)*<l,k||c,a>*t2(c,f,l,k)
    Hdd +=  0.500000000000000 * einsum('eb,nj,mi,lkca,cflk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,b)*d(m,j)*d(n,i)*<l,k||c,a>*t2(c,f,l,k)
    Hdd += -0.500000000000000 * einsum('eb,mj,ni,lkca,cflk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(n,j)*d(m,i)*<l,k||c,b>*t2(c,f,l,k)
    Hdd += -0.500000000000000 * einsum('ea,nj,mi,lkcb,cflk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(m,j)*d(n,i)*<l,k||c,b>*t2(c,f,l,k)
    Hdd +=  0.500000000000000 * einsum('ea,mj,ni,lkcb,cflk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,i)*<j,k||c,a>*t2(c,f,n,k)
    Hdd += -1.000000000000000 * einsum('eb,mi,jkca,cfnk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,j)*<i,k||c,a>*t2(c,f,n,k)
    Hdd +=  1.000000000000000 * einsum('eb,mj,ikca,cfnk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*<j,k||c,b>*t2(c,f,n,k)
    Hdd +=  1.000000000000000 * einsum('ea,mi,jkcb,cfnk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,j)*<i,k||c,b>*t2(c,f,n,k)
    Hdd += -1.000000000000000 * einsum('ea,mj,ikcb,cfnk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*d(n,i)*<j,k||c,a>*t2(c,f,m,k)
    Hdd +=  1.000000000000000 * einsum('eb,ni,jkca,cfmk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(n,j)*<i,k||c,a>*t2(c,f,m,k)
    Hdd += -1.000000000000000 * einsum('eb,nj,ikca,cfmk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<j,k||c,b>*t2(c,f,m,k)
    Hdd += -1.000000000000000 * einsum('ea,ni,jkcb,cfmk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,j)*<i,k||c,b>*t2(c,f,m,k)
    Hdd +=  1.000000000000000 * einsum('ea,nj,ikcb,cfmk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*<i,j||c,a>*t2(c,f,m,n)
    Hdd +=  1.000000000000000 * einsum('eb,ijca,cfmn->efmnabij', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*<i,j||c,b>*t2(c,f,m,n)
    Hdd += -1.000000000000000 * einsum('ea,ijcb,cfmn->efmnabij', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(n,j)*d(m,i)*<l,k||a,b>*t2(e,f,l,k)
    Hdd +=  0.500000000000000 * einsum('nj,mi,lkab,eflk->efmnabij', kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(m,j)*d(n,i)*<l,k||a,b>*t2(e,f,l,k)
    Hdd += -0.500000000000000 * einsum('mj,ni,lkab,eflk->efmnabij', kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(m,i)*<j,k||a,b>*t2(e,f,n,k)
    Hdd += -1.000000000000000 * einsum('mi,jkab,efnk->efmnabij', kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(m,j)*<i,k||a,b>*t2(e,f,n,k)
    Hdd +=  1.000000000000000 * einsum('mj,ikab,efnk->efmnabij', kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(n,i)*<j,k||a,b>*t2(e,f,m,k)
    Hdd +=  1.000000000000000 * einsum('ni,jkab,efmk->efmnabij', kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(n,j)*<i,k||a,b>*t2(e,f,m,k)
    Hdd += -1.000000000000000 * einsum('nj,ikab,efmk->efmnabij', kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(f,b)*d(n,j)*d(m,i)*<l,k||c,d>*t1(c,k)*t1(d,l)
    Hdd += -0.500000000000000 * einsum('ea,fb,nj,mi,lkcd,ck,dl->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(f,b)*d(m,j)*d(n,i)*<l,k||c,d>*t1(c,k)*t1(d,l)
    Hdd +=  0.500000000000000 * einsum('ea,fb,mj,ni,lkcd,ck,dl->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(e,b)*d(n,j)*d(m,i)*<l,k||c,d>*t1(c,k)*t1(d,l)
    Hdd +=  0.500000000000000 * einsum('fa,eb,nj,mi,lkcd,ck,dl->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(e,b)*d(m,j)*d(n,i)*<l,k||c,d>*t1(c,k)*t1(d,l)
    Hdd += -0.500000000000000 * einsum('fa,eb,mj,ni,lkcd,ck,dl->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(m,i)*<j,k||c,d>*t1(c,k)*t1(d,n)
    Hdd +=  1.000000000000000 * einsum('ea,fb,mi,jkcd,ck,dn->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,j)*<i,k||c,d>*t1(c,k)*t1(d,n)
    Hdd += -1.000000000000000 * einsum('ea,fb,mj,ikcd,ck,dn->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(m,i)*<j,k||c,d>*t1(c,k)*t1(d,n)
    Hdd += -1.000000000000000 * einsum('fa,eb,mi,jkcd,ck,dn->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,j)*<i,k||c,d>*t1(c,k)*t1(d,n)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mj,ikcd,ck,dn->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(n,i)*<j,k||c,d>*t1(c,k)*t1(d,m)
    Hdd += -1.000000000000000 * einsum('ea,fb,ni,jkcd,ck,dm->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,j)*<i,k||c,d>*t1(c,k)*t1(d,m)
    Hdd +=  1.000000000000000 * einsum('ea,fb,nj,ikcd,ck,dm->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(n,i)*<j,k||c,d>*t1(c,k)*t1(d,m)
    Hdd +=  1.000000000000000 * einsum('fa,eb,ni,jkcd,ck,dm->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,j)*<i,k||c,d>*t1(c,k)*t1(d,m)
    Hdd += -1.000000000000000 * einsum('fa,eb,nj,ikcd,ck,dm->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,j)*d(m,i)*<l,k||c,a>*t1(c,k)*t1(e,l)
    Hdd +=  1.000000000000000 * einsum('fb,nj,mi,lkca,ck,el->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(m,j)*d(n,i)*<l,k||c,a>*t1(c,k)*t1(e,l)
    Hdd += -1.000000000000000 * einsum('fb,mj,ni,lkca,ck,el->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,j)*d(m,i)*<l,k||c,b>*t1(c,k)*t1(e,l)
    Hdd += -1.000000000000000 * einsum('fa,nj,mi,lkcb,ck,el->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,j)*d(n,i)*<l,k||c,b>*t1(c,k)*t1(e,l)
    Hdd +=  1.000000000000000 * einsum('fa,mj,ni,lkcb,ck,el->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(n,j)*d(m,i)*<l,k||c,a>*t1(c,k)*t1(f,l)
    Hdd += -1.000000000000000 * einsum('eb,nj,mi,lkca,ck,fl->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,j)*d(n,i)*<l,k||c,a>*t1(c,k)*t1(f,l)
    Hdd +=  1.000000000000000 * einsum('eb,mj,ni,lkca,ck,fl->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,j)*d(m,i)*<l,k||c,b>*t1(c,k)*t1(f,l)
    Hdd +=  1.000000000000000 * einsum('ea,nj,mi,lkcb,ck,fl->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,j)*d(n,i)*<l,k||c,b>*t1(c,k)*t1(f,l)
    Hdd += -1.000000000000000 * einsum('ea,mj,ni,lkcb,ck,fl->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*<i,j||c,d>*t1(c,n)*t1(d,m)
    Hdd += -1.000000000000000 * einsum('ea,fb,ijcd,cn,dm->efmnabij', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*<i,j||c,d>*t1(c,n)*t1(d,m)
    Hdd +=  1.000000000000000 * einsum('fa,eb,ijcd,cn,dm->efmnabij', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(m,i)*<j,k||c,a>*t1(c,n)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fb,mi,jkca,cn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(m,j)*<i,k||c,a>*t1(c,n)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fb,mj,ikca,cn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<j,k||c,b>*t1(c,n)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fa,mi,jkcb,cn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,j)*<i,k||c,b>*t1(c,n)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fa,mj,ikcb,cn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,i)*<j,k||c,a>*t1(c,n)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('eb,mi,jkca,cn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,j)*<i,k||c,a>*t1(c,n)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('eb,mj,ikca,cn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*<j,k||c,b>*t1(c,n)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('ea,mi,jkcb,cn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,j)*<i,k||c,b>*t1(c,n)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('ea,mj,ikcb,cn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(n,i)*<j,k||c,a>*t1(c,m)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fb,ni,jkca,cm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,j)*<i,k||c,a>*t1(c,m)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fb,nj,ikca,cm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<j,k||c,b>*t1(c,m)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fa,ni,jkcb,cm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,j)*<i,k||c,b>*t1(c,m)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fa,nj,ikcb,cm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(n,i)*<j,k||c,a>*t1(c,m)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('eb,ni,jkca,cm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(n,j)*<i,k||c,a>*t1(c,m)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('eb,nj,ikca,cm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<j,k||c,b>*t1(c,m)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('ea,ni,jkcb,cm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,j)*<i,k||c,b>*t1(c,m)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('ea,nj,ikcb,cm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(n,j)*d(m,i)*<l,k||a,b>*t1(e,k)*t1(f,l)
    Hdd += -1.000000000000000 * einsum('nj,mi,lkab,ek,fl->efmnabij', kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(m,j)*d(n,i)*<l,k||a,b>*t1(e,k)*t1(f,l)
    Hdd +=  1.000000000000000 * einsum('mj,ni,lkab,ek,fl->efmnabij', kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    return H00, Hs0, H0s, Hd0, H0d, Hss, Hsd, Hds, Hdd
    
def pack_eom_ccsd_H(H00, Hs0, H0s, Hd0, H0d, Hss, Hsd, Hds, Hdd, nsocc, nsvirt, core_list):

    nsingles = 0
    for a in range (0,nsvirt):
        for i in range (0,nsocc):
            if i not in core_list:
                continue
            nsingles += 1

    ndoubles = 0
    for a in range (0,nsvirt):
        for b in range (a+1,nsvirt):
            for i in range (0,nsocc):
                for j in range (i+1,nsocc):
                    if i not in core_list and j not in core_list:
                        continue
                    ndoubles += 1

    #dim = int(1 + nsvirt*(nsvirt-1)/2*nsocc*(nsocc-1)/2 + nsvirt*nsocc)
    dim = int(1 + ndoubles + nsingles)
    H = np.zeros((dim,dim))

    # 00 block
    H[0,0] = H00

    # 0s, s0 blocks
    ai = 1
    for a in range (0,nsvirt):
        for i in range (0,nsocc):
            if i not in core_list:
                continue
            H[ai,0] = Hs0[a,i]
            H[0,ai] = H0s[a,i]
            ai += 1

    # ss block
    ai = 1
    for a in range (0,nsvirt):
        for i in range (0,nsocc):
            if i not in core_list:
                continue
            em = 1
            for e in range (0,nsvirt):
                for m in range (0,nsocc):
                    if m not in core_list:
                        continue
                    H[ai,em] = Hss[a,i,e,m]
                    em += 1
            ai += 1

    # sd, ds blocks
    ai = 1
    for a in range (0,nsvirt):
        for i in range (0,nsocc):
            if i not in core_list:
                continue
            efmn = 1 + nsingles
            for e in range (0,nsvirt):
                for f in range (e+1,nsvirt):
                    for m in range (0,nsocc):
                        for n in range (m+1,nsocc):
                            if m not in core_list and n not in core_list:
                                continue
                            H[ai,efmn] = Hsd[a,i,e,f,m,n]
                            H[efmn,ai] = Hds[e,f,m,n,a,i]
                            efmn += 1
            ai += 1

    # 0d, d0 blocks
    abij = 1 + nsingles
    for a in range (0,nsvirt):
        for b in range (a+1,nsvirt):
            for i in range (0,nsocc):
                for j in range (i+1,nsocc):
                    if i not in core_list and j not in core_list:
                        continue
                    H[abij,0] = Hd0[a,b,i,j]
                    H[0,abij] = H0d[a,b,i,j]
                    abij += 1

    # dd blocks
    abij = 1 + nsingles
    for a in range (0,nsvirt):
        for b in range (a+1,nsvirt):
            for i in range (0,nsocc):
                for j in range (i+1,nsocc):
                    if i not in core_list and j not in core_list:
                        continue
                    efmn = 1 + nsingles
                    for e in range (0,nsvirt):
                        for f in range (e+1,nsvirt):
                            for m in range (0,nsocc):
                                for n in range (m+1,nsocc):
                                    if n not in core_list and m not in core_list:
                                        continue
                                    H[abij,efmn] = Hdd[a,b,i,j,e,f,m,n]
                                    efmn += 1
                    abij += 1

    return H

def build_eom_ccsd_H(f, g, o, v, t1, t2, nsocc, nsvirt, core_list):

    kd = np.zeros((nsocc+nsvirt,nsocc+nsvirt))
    for i in range (0,nsocc+nsvirt):
        kd[i,i] = 1.0

    H00, Hs0, H0s, Hd0, H0d, Hss, Hsd, Hds, Hdd = build_eom_ccsd_H_by_block(kd,f, g, o, v, t1, t2)

    H = pack_eom_ccsd_H(H00, Hs0, H0s, Hd0, H0d, Hss, Hsd, Hds, Hdd, nsocc, nsvirt, core_list)

    return H

def sigma_ref(t1, t2, r0, r1, r2, f, g, o, v):
    """ 
    build <0| Hbar R |0>, spin-orbital basis

    :param t1: singles amplitudes, shaped as v, o
    :param t2: doubles amplitudes, shaped as v, v, o, o
    :param r0: reference eom-cc amplitude
    :param r1: singles eom-cc amplitudes, shaped as v, o
    :param r2: doubles eom-cc amplitudes, shaped as v, v, o, o
    :param f: fock matrix
    :param g: antisymmetrized eris
    :param o: occupied slice
    :param v: virtual slice
    
    :return sigma0: the reference part of the sigma vector
    """ 


    #         1.00 f(i,i)*r0
    sigma0 =  1.00 * einsum('ii,', f[o, o], r0)
    
    #         1.00 f(i,a)*r1(a,i)
    sigma0 +=  1.00 * einsum('ia,ai', f[o, v], r1)
    
    #         1.00 f(i,a)*r0*t1(a,i)
    sigma0 +=  1.00 * einsum('ia,,ai', f[o, v], r0, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #        -0.50 <j,i||j,i>*r0
    sigma0 += -0.50 * einsum('jiji,', g[o, o, o, o], r0)
    
    #         0.250 <j,i||a,b>*r2(a,b,j,i)
    sigma0 +=  0.250 * einsum('jiab,abji', g[o, o, v, v], r2)
    
    #        -1.00 <j,i||a,b>*r1(b,j)*t1(a,i)
    sigma0 += -1.00 * einsum('jiab,bj,ai', g[o, o, v, v], r1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #         0.250 <j,i||a,b>*r0*t2(a,b,j,i)
    sigma0 +=  0.250 * einsum('jiab,,abji', g[o, o, v, v], r0, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #        -0.50 <j,i||a,b>*r0*t1(a,i)*t1(b,j)
    sigma0 += -0.50 * einsum('jiab,,ai,bj', g[o, o, v, v], r0, t1, t1, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    
    return sigma0

def sigma_singles(t1, t2, r0, r1, r2, f, g, o, v):
    """ 
    build <0| i^a Hbar R |0>, spin-orbital basis

    :param t1: singles amplitudes, shaped as v, o
    :param t2: doubles amplitudes, shaped as v, v, o, o
    :param r0: reference eom-cc amplitude
    :param r1: singles eom-cc amplitudes, shaped as v, o
    :param r2: doubles eom-cc amplitudes, shaped as v, v, o, o
    :param f: fock matrix
    :param g: antisymmetrized eris
    :param o: occupied slice
    :param v: virtual slice
    
    :return sigma1: the singles part of the sigma vector, shaped as v, o
    """ 


    #	  1.00 f(a,i)*r0
    sigma1 =  1.00 * einsum('ai,->ai', f[v, o], r0)
    
    #	  1.00 f(j,j)*r1(a,i)
    sigma1 +=  1.00 * einsum('jj,ai->ai', f[o, o], r1)
    
    #	 -1.00 f(j,i)*r1(a,j)
    sigma1 += -1.00 * einsum('ji,aj->ai', f[o, o], r1)
    
    #	  1.00 f(a,b)*r1(b,i)
    sigma1 +=  1.00 * einsum('ab,bi->ai', f[v, v], r1)
    
    #	 -1.00 f(j,b)*r2(b,a,i,j)
    sigma1 += -1.00 * einsum('jb,baij->ai', f[o, v], r2)
    
    #	 -1.00 f(j,i)*r0*t1(a,j)
    sigma1 += -1.00 * einsum('ji,,aj->ai', f[o, o], r0, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.00 f(a,b)*r0*t1(b,i)
    sigma1 +=  1.00 * einsum('ab,,bi->ai', f[v, v], r0, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.00 f(j,b)*r1(a,i)*t1(b,j)
    sigma1 +=  1.00 * einsum('jb,ai,bj->ai', f[o, v], r1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    #	 -1.00 f(j,b)*r1(a,j)*t1(b,i)
    sigma1 += -1.00 * einsum('jb,aj,bi->ai', f[o, v], r1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.00 f(j,b)*r1(b,i)*t1(a,j)
    sigma1 += -1.00 * einsum('jb,bi,aj->ai', f[o, v], r1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.00 f(j,b)*r0*t2(b,a,i,j)
    sigma1 += -1.00 * einsum('jb,,baij->ai', f[o, v], r0, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.00 f(j,b)*r0*t1(a,j)*t1(b,i)
    sigma1 += -1.00 * einsum('jb,,aj,bi->ai', f[o, v], r0, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.50 <k,j||k,j>*r1(a,i)
    sigma1 += -0.50 * einsum('kjkj,ai->ai', g[o, o, o, o], r1)
    
    #	  1.00 <j,a||b,i>*r1(b,j)
    sigma1 +=  1.00 * einsum('jabi,bj->ai', g[o, v, v, o], r1)
    
    #	 -0.50 <k,j||b,i>*r2(b,a,k,j)
    sigma1 += -0.50 * einsum('kjbi,bakj->ai', g[o, o, v, o], r2)
    
    #	 -0.50 <j,a||b,c>*r2(b,c,i,j)
    sigma1 += -0.50 * einsum('jabc,bcij->ai', g[o, v, v, v], r2)
    
    #	  1.00 <j,a||b,i>*r0*t1(b,j)
    sigma1 +=  1.00 * einsum('jabi,,bj->ai', g[o, v, v, o], r0, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.00 <k,j||b,i>*r1(a,k)*t1(b,j)
    sigma1 +=  1.00 * einsum('kjbi,ak,bj->ai', g[o, o, v, o], r1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.00 <k,j||b,i>*r1(b,k)*t1(a,j)
    sigma1 += -1.00 * einsum('kjbi,bk,aj->ai', g[o, o, v, o], r1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.00 <j,a||b,c>*r1(c,i)*t1(b,j)
    sigma1 +=  1.00 * einsum('jabc,ci,bj->ai', g[o, v, v, v], r1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.00 <j,a||b,c>*r1(c,j)*t1(b,i)
    sigma1 += -1.00 * einsum('jabc,cj,bi->ai', g[o, v, v, v], r1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.00 <k,j||b,c>*r2(c,a,i,k)*t1(b,j)
    sigma1 +=  1.00 * einsum('kjbc,caik,bj->ai', g[o, o, v, v], r2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.50 <k,j||b,c>*r2(c,a,k,j)*t1(b,i)
    sigma1 +=  0.50 * einsum('kjbc,cakj,bi->ai', g[o, o, v, v], r2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.50 <k,j||b,c>*r2(b,c,i,k)*t1(a,j)
    sigma1 +=  0.50 * einsum('kjbc,bcik,aj->ai', g[o, o, v, v], r2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.50 <k,j||b,i>*r0*t2(b,a,k,j)
    sigma1 += -0.50 * einsum('kjbi,,bakj->ai', g[o, o, v, o], r0, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.50 <j,a||b,c>*r0*t2(b,c,i,j)
    sigma1 += -0.50 * einsum('jabc,,bcij->ai', g[o, v, v, v], r0, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.250 <k,j||b,c>*r1(a,i)*t2(b,c,k,j)
    sigma1 +=  0.250 * einsum('kjbc,ai,bckj->ai', g[o, o, v, v], r1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.50 <k,j||b,c>*r1(a,k)*t2(b,c,i,j)
    sigma1 += -0.50 * einsum('kjbc,ak,bcij->ai', g[o, o, v, v], r1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.50 <k,j||b,c>*r1(c,i)*t2(b,a,k,j)
    sigma1 += -0.50 * einsum('kjbc,ci,bakj->ai', g[o, o, v, v], r1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.00 <k,j||b,c>*r1(c,k)*t2(b,a,i,j)
    sigma1 +=  1.00 * einsum('kjbc,ck,baij->ai', g[o, o, v, v], r1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.00 <k,j||b,c>*r0*t2(c,a,i,k)*t1(b,j)
    sigma1 +=  1.00 * einsum('kjbc,,caik,bj->ai', g[o, o, v, v], r0, t2, t1, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  0.50 <k,j||b,c>*r0*t2(c,a,k,j)*t1(b,i)
    sigma1 +=  0.50 * einsum('kjbc,,cakj,bi->ai', g[o, o, v, v], r0, t2, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.50 <k,j||b,c>*r0*t1(a,j)*t2(b,c,i,k)
    sigma1 +=  0.50 * einsum('kjbc,,aj,bcik->ai', g[o, o, v, v], r0, t1, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	  1.00 <k,j||b,i>*r0*t1(a,k)*t1(b,j)
    sigma1 +=  1.00 * einsum('kjbi,,ak,bj->ai', g[o, o, v, o], r0, t1, t1, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	  1.00 <j,a||b,c>*r0*t1(b,j)*t1(c,i)
    sigma1 +=  1.00 * einsum('jabc,,bj,ci->ai', g[o, v, v, v], r0, t1, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.50 <k,j||b,c>*r1(a,i)*t1(b,j)*t1(c,k)
    sigma1 += -0.50 * einsum('kjbc,ai,bj,ck->ai', g[o, o, v, v], r1, t1, t1, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    
    #	  1.00 <k,j||b,c>*r1(a,k)*t1(b,j)*t1(c,i)
    sigma1 +=  1.00 * einsum('kjbc,ak,bj,ci->ai', g[o, o, v, v], r1, t1, t1, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.00 <k,j||b,c>*r1(c,i)*t1(a,k)*t1(b,j)
    sigma1 +=  1.00 * einsum('kjbc,ci,ak,bj->ai', g[o, o, v, v], r1, t1, t1, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  1.00 <k,j||b,c>*r1(c,k)*t1(a,j)*t1(b,i)
    sigma1 +=  1.00 * einsum('kjbc,ck,aj,bi->ai', g[o, o, v, v], r1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.00 <k,j||b,c>*r0*t1(a,k)*t1(b,j)*t1(c,i)
    sigma1 +=  1.00 * einsum('kjbc,,ak,bj,ci->ai', g[o, o, v, v], r0, t1, t1, t1, optimize=['einsum_path', (0, 3), (0, 1), (0, 1), (0, 1)])
    
    return sigma1

def sigma_doubles(t1, t2, r0, r1, r2, f, g, o, v):
    """ 
    build <0| i^j^ba Hbar R |0>, spin-orbital basis

    :param t1: singles amplitudes, shaped as v, o
    :param t2: doubles amplitudes, shaped as v, v, o, o
    :param r0: reference eom-cc amplitude
    :param r1: singles eom-cc amplitudes, shaped as v, o
    :param r2: doubles eom-cc amplitudes, shaped as v, v, o, o
    :param f: fock matrix
    :param g: antisymmetrized eris
    :param o: occupied slice
    :param v: virtual slice
    
    :return sigma2: the doubles part of the sigma vector, shaped as v, v, o, o
    """ 

    #	 -1.00 P(i,j)*P(a,b)f(a,j)*r1(b,i)
    contracted_intermediate = -1.00 * einsum('aj,bi->abij', f[v, o], r1)
    sigma2 =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.00 f(k,k)*r2(a,b,i,j)
    sigma2 +=  1.00 * einsum('kk,abij->abij', f[o, o], r2)
    
    #	 -1.00 P(i,j)f(k,j)*r2(a,b,i,k)
    contracted_intermediate = -1.00 * einsum('kj,abik->abij', f[o, o], r2)
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.00 P(a,b)f(a,c)*r2(c,b,i,j)
    contracted_intermediate =  1.00 * einsum('ac,cbij->abij', f[v, v], r2)
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.00 P(i,j)*P(a,b)f(k,j)*r1(b,i)*t1(a,k)
    contracted_intermediate =  1.00 * einsum('kj,bi,ak->abij', f[o, o], r1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)f(a,c)*r1(b,i)*t1(c,j)
    contracted_intermediate = -1.00 * einsum('ac,bi,cj->abij', f[v, v], r1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.00 f(k,c)*r2(a,b,i,j)*t1(c,k)
    sigma2 +=  1.00 * einsum('kc,abij,ck->abij', f[o, v], r2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.00 P(i,j)f(k,c)*r2(a,b,i,k)*t1(c,j)
    contracted_intermediate = -1.00 * einsum('kc,abik,cj->abij', f[o, v], r2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.00 P(a,b)f(k,c)*r2(c,b,i,j)*t1(a,k)
    contracted_intermediate = -1.00 * einsum('kc,cbij,ak->abij', f[o, v], r2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.00 P(i,j)f(k,j)*r0*t2(a,b,i,k)
    contracted_intermediate = -1.00 * einsum('kj,,abik->abij', f[o, o], r0, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.00 P(a,b)f(a,c)*r0*t2(c,b,i,j)
    contracted_intermediate =  1.00 * einsum('ac,,cbij->abij', f[v, v], r0, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.00 P(i,j)*P(a,b)f(k,c)*r1(b,i)*t2(c,a,j,k)
    contracted_intermediate =  1.00 * einsum('kc,bi,cajk->abij', f[o, v], r1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.00 P(a,b)f(k,c)*r1(b,k)*t2(c,a,i,j)
    contracted_intermediate =  1.00 * einsum('kc,bk,caij->abij', f[o, v], r1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.00 P(i,j)f(k,c)*r1(c,i)*t2(a,b,j,k)
    contracted_intermediate =  1.00 * einsum('kc,ci,abjk->abij', f[o, v], r1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.00 P(i,j)f(k,c)*r0*t2(a,b,i,k)*t1(c,j)
    contracted_intermediate = -1.00 * einsum('kc,,abik,cj->abij', f[o, v], r0, t2, t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.00 P(a,b)f(k,c)*r0*t1(a,k)*t2(c,b,i,j)
    contracted_intermediate = -1.00 * einsum('kc,,ak,cbij->abij', f[o, v], r0, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.00 P(i,j)*P(a,b)f(k,c)*r1(b,i)*t1(a,k)*t1(c,j)
    contracted_intermediate =  1.00 * einsum('kc,bi,ak,cj->abij', f[o, v], r1, t1, t1, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.50 <l,k||l,k>*r2(a,b,i,j)
    sigma2 += -0.50 * einsum('lklk,abij->abij', g[o, o, o, o], r2)
    
    #	  1.00 <a,b||i,j>*r0
    sigma2 +=  1.00 * einsum('abij,->abij', g[v, v, o, o], r0)
    
    #	  1.00 P(a,b)<k,a||i,j>*r1(b,k)
    contracted_intermediate =  1.00 * einsum('kaij,bk->abij', g[o, v, o, o], r1)
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.00 P(i,j)<a,b||c,j>*r1(c,i)
    contracted_intermediate =  1.00 * einsum('abcj,ci->abij', g[v, v, v, o], r1)
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.50 <l,k||i,j>*r2(a,b,l,k)
    sigma2 +=  0.50 * einsum('lkij,ablk->abij', g[o, o, o, o], r2)
    
    #	  1.00 P(i,j)*P(a,b)<k,a||c,j>*r2(c,b,i,k)
    contracted_intermediate =  1.00 * einsum('kacj,cbik->abij', g[o, v, v, o], r2)
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.50 <a,b||c,d>*r2(c,d,i,j)
    sigma2 +=  0.50 * einsum('abcd,cdij->abij', g[v, v, v, v], r2)
    
    #	  1.00 P(a,b)<k,a||i,j>*r0*t1(b,k)
    contracted_intermediate =  1.00 * einsum('kaij,,bk->abij', g[o, v, o, o], r0, t1, optimize=['einsum_path', (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.00 P(i,j)<a,b||c,j>*r0*t1(c,i)
    contracted_intermediate =  1.00 * einsum('abcj,,ci->abij', g[v, v, v, o], r0, t1, optimize=['einsum_path', (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.00 P(a,b)<l,k||i,j>*r1(b,l)*t1(a,k)
    contracted_intermediate = -1.00 * einsum('lkij,bl,ak->abij', g[o, o, o, o], r1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<k,a||c,j>*r1(b,i)*t1(c,k)
    contracted_intermediate = -1.00 * einsum('kacj,bi,ck->abij', g[o, v, v, o], r1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.00 P(i,j)*P(a,b)<k,a||c,j>*r1(b,k)*t1(c,i)
    contracted_intermediate =  1.00 * einsum('kacj,bk,ci->abij', g[o, v, v, o], r1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.00 P(i,j)*P(a,b)<k,a||c,j>*r1(c,i)*t1(b,k)
    contracted_intermediate =  1.00 * einsum('kacj,ci,bk->abij', g[o, v, v, o], r1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.00 P(i,j)<a,b||c,d>*r1(d,i)*t1(c,j)
    contracted_intermediate = -1.00 * einsum('abcd,di,cj->abij', g[v, v, v, v], r1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.00 P(i,j)<l,k||c,j>*r2(a,b,i,l)*t1(c,k)
    contracted_intermediate =  1.00 * einsum('lkcj,abil,ck->abij', g[o, o, v, o], r2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.50 P(i,j)<l,k||c,j>*r2(a,b,l,k)*t1(c,i)
    contracted_intermediate =  0.50 * einsum('lkcj,ablk,ci->abij', g[o, o, v, o], r2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>*r2(c,b,i,l)*t1(a,k)
    contracted_intermediate = -1.00 * einsum('lkcj,cbil,ak->abij', g[o, o, v, o], r2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.00 P(a,b)<k,a||c,d>*r2(d,b,i,j)*t1(c,k)
    contracted_intermediate =  1.00 * einsum('kacd,dbij,ck->abij', g[o, v, v, v], r2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>*r2(d,b,i,k)*t1(c,j)
    contracted_intermediate = -1.00 * einsum('kacd,dbik,cj->abij', g[o, v, v, v], r2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.50 P(a,b)<k,a||c,d>*r2(c,d,i,j)*t1(b,k)
    contracted_intermediate =  0.50 * einsum('kacd,cdij,bk->abij', g[o, v, v, v], r2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.50 <l,k||i,j>*r0*t2(a,b,l,k)
    sigma2 +=  0.50 * einsum('lkij,,ablk->abij', g[o, o, o, o], r0, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.00 P(i,j)*P(a,b)<k,a||c,j>*r0*t2(c,b,i,k)
    contracted_intermediate =  1.00 * einsum('kacj,,cbik->abij', g[o, v, v, o], r0, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.50 <a,b||c,d>*r0*t2(c,d,i,j)
    sigma2 +=  0.50 * einsum('abcd,,cdij->abij', g[v, v, v, v], r0, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.50 P(i,j)*P(a,b)<l,k||c,j>*r1(b,i)*t2(c,a,l,k)
    contracted_intermediate =  0.50 * einsum('lkcj,bi,calk->abij', g[o, o, v, o], r1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>*r1(b,l)*t2(c,a,i,k)
    contracted_intermediate = -1.00 * einsum('lkcj,bl,caik->abij', g[o, o, v, o], r1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.50 P(i,j)<l,k||c,j>*r1(c,i)*t2(a,b,l,k)
    contracted_intermediate =  0.50 * einsum('lkcj,ci,ablk->abij', g[o, o, v, o], r1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.00 P(i,j)<l,k||c,j>*r1(c,l)*t2(a,b,i,k)
    contracted_intermediate = -1.00 * einsum('lkcj,cl,abik->abij', g[o, o, v, o], r1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.50 P(i,j)*P(a,b)<k,a||c,d>*r1(b,i)*t2(c,d,j,k)
    contracted_intermediate =  0.50 * einsum('kacd,bi,cdjk->abij', g[o, v, v, v], r1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.50 P(a,b)<k,a||c,d>*r1(b,k)*t2(c,d,i,j)
    contracted_intermediate =  0.50 * einsum('kacd,bk,cdij->abij', g[o, v, v, v], r1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>*r1(d,i)*t2(c,b,j,k)
    contracted_intermediate = -1.00 * einsum('kacd,di,cbjk->abij', g[o, v, v, v], r1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.00 P(a,b)<k,a||c,d>*r1(d,k)*t2(c,b,i,j)
    contracted_intermediate = -1.00 * einsum('kacd,dk,cbij->abij', g[o, v, v, v], r1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.250 <l,k||c,d>*r2(a,b,i,j)*t2(c,d,l,k)
    sigma2 +=  0.250 * einsum('lkcd,abij,cdlk->abij', g[o, o, v, v], r2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.50 P(i,j)<l,k||c,d>*r2(a,b,i,l)*t2(c,d,j,k)
    contracted_intermediate = -0.50 * einsum('lkcd,abil,cdjk->abij', g[o, o, v, v], r2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.250 <l,k||c,d>*r2(a,b,l,k)*t2(c,d,i,j)
    sigma2 +=  0.250 * einsum('lkcd,ablk,cdij->abij', g[o, o, v, v], r2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.50 P(a,b)<l,k||c,d>*r2(d,b,i,j)*t2(c,a,l,k)
    contracted_intermediate = -0.50 * einsum('lkcd,dbij,calk->abij', g[o, o, v, v], r2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.00 P(i,j)*P(a,b)<l,k||c,d>*r2(d,b,i,l)*t2(c,a,j,k)
    contracted_intermediate =  1.00 * einsum('lkcd,dbil,cajk->abij', g[o, o, v, v], r2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.50 P(a,b)<l,k||c,d>*r2(d,b,l,k)*t2(c,a,i,j)
    contracted_intermediate = -0.50 * einsum('lkcd,dblk,caij->abij', g[o, o, v, v], r2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.250 <l,k||c,d>*r2(c,d,i,j)*t2(a,b,l,k)
    sigma2 +=  0.250 * einsum('lkcd,cdij,ablk->abij', g[o, o, v, v], r2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.50 P(i,j)<l,k||c,d>*r2(c,d,i,l)*t2(a,b,j,k)
    contracted_intermediate = -0.50 * einsum('lkcd,cdil,abjk->abij', g[o, o, v, v], r2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.00 P(i,j)<l,k||c,j>*r0*t2(a,b,i,l)*t1(c,k)
    contracted_intermediate =  1.00 * einsum('lkcj,,abil,ck->abij', g[o, o, v, o], r0, t2, t1, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.50 P(i,j)<l,k||c,j>*r0*t2(a,b,l,k)*t1(c,i)
    contracted_intermediate =  0.50 * einsum('lkcj,,ablk,ci->abij', g[o, o, v, o], r0, t2, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>*r0*t1(a,k)*t2(c,b,i,l)
    contracted_intermediate = -1.00 * einsum('lkcj,,ak,cbil->abij', g[o, o, v, o], r0, t1, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.00 P(a,b)<k,a||c,d>*r0*t2(d,b,i,j)*t1(c,k)
    contracted_intermediate =  1.00 * einsum('kacd,,dbij,ck->abij', g[o, v, v, v], r0, t2, t1, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>*r0*t2(d,b,i,k)*t1(c,j)
    contracted_intermediate = -1.00 * einsum('kacd,,dbik,cj->abij', g[o, v, v, v], r0, t2, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.50 P(a,b)<k,a||c,d>*r0*t1(b,k)*t2(c,d,i,j)
    contracted_intermediate =  0.50 * einsum('kacd,,bk,cdij->abij', g[o, v, v, v], r0, t1, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<l,k||c,d>*r1(b,i)*t2(d,a,j,l)*t1(c,k)
    contracted_intermediate = -1.00 * einsum('lkcd,bi,dajl,ck->abij', g[o, o, v, v], r1, t2, t1, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.00 P(a,b)<l,k||c,d>*r1(b,l)*t2(d,a,i,j)*t1(c,k)
    contracted_intermediate = -1.00 * einsum('lkcd,bl,daij,ck->abij', g[o, o, v, v], r1, t2, t1, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.00 P(i,j)<l,k||c,d>*r1(d,i)*t2(a,b,j,l)*t1(c,k)
    contracted_intermediate = -1.00 * einsum('lkcd,di,abjl,ck->abij', g[o, o, v, v], r1, t2, t1, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.50 P(i,j)*P(a,b)<l,k||c,d>*r1(b,i)*t2(d,a,l,k)*t1(c,j)
    contracted_intermediate = -0.50 * einsum('lkcd,bi,dalk,cj->abij', g[o, o, v, v], r1, t2, t1, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.00 P(i,j)*P(a,b)<l,k||c,d>*r1(b,l)*t2(d,a,i,k)*t1(c,j)
    contracted_intermediate =  1.00 * einsum('lkcd,bl,daik,cj->abij', g[o, o, v, v], r1, t2, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.50 P(i,j)<l,k||c,d>*r1(d,i)*t2(a,b,l,k)*t1(c,j)
    contracted_intermediate = -0.50 * einsum('lkcd,di,ablk,cj->abij', g[o, o, v, v], r1, t2, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.00 P(i,j)<l,k||c,d>*r1(d,l)*t2(a,b,i,k)*t1(c,j)
    contracted_intermediate =  1.00 * einsum('lkcd,dl,abik,cj->abij', g[o, o, v, v], r1, t2, t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.50 P(i,j)*P(a,b)<l,k||c,d>*r1(b,i)*t1(a,k)*t2(c,d,j,l)
    contracted_intermediate = -0.50 * einsum('lkcd,bi,ak,cdjl->abij', g[o, o, v, v], r1, t1, t2, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.50 P(a,b)<l,k||c,d>*r1(b,l)*t1(a,k)*t2(c,d,i,j)
    contracted_intermediate = -0.50 * einsum('lkcd,bl,ak,cdij->abij', g[o, o, v, v], r1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.00 P(i,j)*P(a,b)<l,k||c,d>*r1(d,i)*t1(a,k)*t2(c,b,j,l)
    contracted_intermediate =  1.00 * einsum('lkcd,di,ak,cbjl->abij', g[o, o, v, v], r1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.00 P(a,b)<l,k||c,d>*r1(d,l)*t1(a,k)*t2(c,b,i,j)
    contracted_intermediate =  1.00 * einsum('lkcd,dl,ak,cbij->abij', g[o, o, v, v], r1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.00 <l,k||i,j>*r0*t1(a,k)*t1(b,l)
    sigma2 += -1.00 * einsum('lkij,,ak,bl->abij', g[o, o, o, o], r0, t1, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  1.00 P(i,j)*P(a,b)<k,a||c,j>*r0*t1(b,k)*t1(c,i)
    contracted_intermediate =  1.00 * einsum('kacj,,bk,ci->abij', g[o, v, v, o], r0, t1, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.00 <a,b||c,d>*r0*t1(c,j)*t1(d,i)
    sigma2 += -1.00 * einsum('abcd,,cj,di->abij', g[v, v, v, v], r0, t1, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>*r1(b,i)*t1(a,l)*t1(c,k)
    contracted_intermediate = -1.00 * einsum('lkcj,bi,al,ck->abij', g[o, o, v, o], r1, t1, t1, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>*r1(b,l)*t1(a,k)*t1(c,i)
    contracted_intermediate = -1.00 * einsum('lkcj,bl,ak,ci->abij', g[o, o, v, o], r1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.00 P(i,j)<l,k||c,j>*r1(c,i)*t1(a,k)*t1(b,l)
    contracted_intermediate = -1.00 * einsum('lkcj,ci,ak,bl->abij', g[o, o, v, o], r1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>*r1(b,i)*t1(c,k)*t1(d,j)
    contracted_intermediate = -1.00 * einsum('kacd,bi,ck,dj->abij', g[o, v, v, v], r1, t1, t1, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.00 P(a,b)<k,a||c,d>*r1(b,k)*t1(c,j)*t1(d,i)
    contracted_intermediate = -1.00 * einsum('kacd,bk,cj,di->abij', g[o, v, v, v], r1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>*r1(d,i)*t1(b,k)*t1(c,j)
    contracted_intermediate = -1.00 * einsum('kacd,di,bk,cj->abij', g[o, v, v, v], r1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.50 <l,k||c,d>*r2(a,b,i,j)*t1(c,k)*t1(d,l)
    sigma2 += -0.50 * einsum('lkcd,abij,ck,dl->abij', g[o, o, v, v], r2, t1, t1, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    
    #	  1.00 P(i,j)<l,k||c,d>*r2(a,b,i,l)*t1(c,k)*t1(d,j)
    contracted_intermediate =  1.00 * einsum('lkcd,abil,ck,dj->abij', g[o, o, v, v], r2, t1, t1, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.00 P(a,b)<l,k||c,d>*r2(d,b,i,j)*t1(a,l)*t1(c,k)
    contracted_intermediate =  1.00 * einsum('lkcd,dbij,al,ck->abij', g[o, o, v, v], r2, t1, t1, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.50 <l,k||c,d>*r2(a,b,l,k)*t1(c,j)*t1(d,i)
    sigma2 += -0.50 * einsum('lkcd,ablk,cj,di->abij', g[o, o, v, v], r2, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.00 P(i,j)*P(a,b)<l,k||c,d>*r2(d,b,i,l)*t1(a,k)*t1(c,j)
    contracted_intermediate =  1.00 * einsum('lkcd,dbil,ak,cj->abij', g[o, o, v, v], r2, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.50 <l,k||c,d>*r2(c,d,i,j)*t1(a,k)*t1(b,l)
    sigma2 += -0.50 * einsum('lkcd,cdij,ak,bl->abij', g[o, o, v, v], r2, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.50 P(i,j)<l,k||c,d>*r0*t2(a,b,i,l)*t2(c,d,j,k)
    contracted_intermediate = -0.50 * einsum('lkcd,,abil,cdjk->abij', g[o, o, v, v], r0, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.250 <l,k||c,d>*r0*t2(a,b,l,k)*t2(c,d,i,j)
    sigma2 +=  0.250 * einsum('lkcd,,ablk,cdij->abij', g[o, o, v, v], r0, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.50 <l,k||c,d>*r0*t2(c,a,l,k)*t2(d,b,i,j)
    sigma2 += -0.50 * einsum('lkcd,,calk,dbij->abij', g[o, o, v, v], r0, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  1.00 P(i,j)<l,k||c,d>*r0*t2(c,a,j,k)*t2(d,b,i,l)
    contracted_intermediate =  1.00 * einsum('lkcd,,cajk,dbil->abij', g[o, o, v, v], r0, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.50 <l,k||c,d>*r0*t2(c,a,i,j)*t2(d,b,l,k)
    sigma2 += -0.50 * einsum('lkcd,,caij,dblk->abij', g[o, o, v, v], r0, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	  1.00 P(i,j)<l,k||c,d>*r0*t2(a,b,i,l)*t1(c,k)*t1(d,j)
    contracted_intermediate =  1.00 * einsum('lkcd,,abil,ck,dj->abij', g[o, o, v, v], r0, t2, t1, t1, optimize=['einsum_path', (0, 3), (0, 2), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.00 P(a,b)<l,k||c,d>*r0*t1(a,l)*t2(d,b,i,j)*t1(c,k)
    contracted_intermediate =  1.00 * einsum('lkcd,,al,dbij,ck->abij', g[o, o, v, v], r0, t1, t2, t1, optimize=['einsum_path', (0, 4), (0, 1), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.50 <l,k||c,d>*r0*t2(a,b,l,k)*t1(c,j)*t1(d,i)
    sigma2 += -0.50 * einsum('lkcd,,ablk,cj,di->abij', g[o, o, v, v], r0, t2, t1, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	  1.00 P(i,j)*P(a,b)<l,k||c,d>*r0*t1(a,k)*t2(d,b,i,l)*t1(c,j)
    contracted_intermediate =  1.00 * einsum('lkcd,,ak,dbil,cj->abij', g[o, o, v, v], r0, t1, t2, t1, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.50 <l,k||c,d>*r0*t1(a,k)*t1(b,l)*t2(c,d,i,j)
    sigma2 += -0.50 * einsum('lkcd,,ak,bl,cdij->abij', g[o, o, v, v], r0, t1, t1, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    
    #	 -1.00 P(i,j)<l,k||c,j>*r0*t1(a,k)*t1(b,l)*t1(c,i)
    contracted_intermediate = -1.00 * einsum('lkcj,,ak,bl,ci->abij', g[o, o, v, o], r0, t1, t1, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.00 P(a,b)<k,a||c,d>*r0*t1(b,k)*t1(c,j)*t1(d,i)
    contracted_intermediate = -1.00 * einsum('kacd,,bk,cj,di->abij', g[o, v, v, v], r0, t1, t1, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<l,k||c,d>*r1(b,i)*t1(a,l)*t1(c,k)*t1(d,j)
    contracted_intermediate = -1.00 * einsum('lkcd,bi,al,ck,dj->abij', g[o, o, v, v], r1, t1, t1, t1, optimize=['einsum_path', (0, 3), (1, 3), (1, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.00 P(a,b)<l,k||c,d>*r1(b,l)*t1(a,k)*t1(c,j)*t1(d,i)
    contracted_intermediate =  1.00 * einsum('lkcd,bl,ak,cj,di->abij', g[o, o, v, v], r1, t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.00 P(i,j)<l,k||c,d>*r1(d,i)*t1(a,k)*t1(b,l)*t1(c,j)
    contracted_intermediate =  1.00 * einsum('lkcd,di,ak,bl,cj->abij', g[o, o, v, v], r1, t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    sigma2 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.00 <l,k||c,d>*r0*t1(a,k)*t1(b,l)*t1(c,j)*t1(d,i)
    sigma2 +=  1.00 * einsum('lkcd,,ak,bl,cj,di->abij', g[o, o, v, v], r0, t1, t1, t1, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 2), (0, 2), (0, 1)])
    
    return sigma2

def unpack_antisym(tvec, a_idx, b_idx, i_idx, j_idx, nv, no):
    """
    unpack a flat array (a<b, i<j) into full antisymmetrized t2[a,b,i,j]

    :param tvec: the flat array
    :param a_idx: first of a<b pair (list)
    :param b_idx: second of a<b pair (list)
    :param i_idx: first of i<j pair (list)
    :param j_idx: second of i<j pair (list)
    :param nv: number of virtual orbitals
    :param no: number of occupied orbitals

    :return t2: the full antisymmetrized array
    """

    t2 = np.zeros((nv, nv, no, no))
    tmp = tvec.reshape(len(a_idx), len(i_idx))
    for sign, (aa, bb, ii, jj) in [
        (+1, (a_idx, b_idx, i_idx, j_idx)),
        (-1, (b_idx, a_idx, i_idx, j_idx)),
        (-1, (a_idx, b_idx, j_idx, i_idx)),
        (+1, (b_idx, a_idx, j_idx, i_idx)),
    ]:
        t2[aa[:, None], bb[:, None], ii[None, :], jj[None, :]] += sign * tmp
    return t2

def pack_antisym(t2, a_idx, b_idx, i_idx, j_idx):
    """
    pack antisymmetrized tensor t2[a,b,i,j] into a flat array with a<b, i<j.

    :param t2: the antisymmetrized array
    :param a_idx: first of a<b pair (list)
    :param b_idx: second of a<b pair (list)
    :param i_idx: first of i<j pair (list)
    :param j_idx: second of i<j pair (list)

    :return flat: the flat array
    """

    flat = t2[a_idx[:, None], b_idx[:, None], i_idx[None, :], j_idx[None, :]]
    return flat.reshape(-1)

    import scipy
    from scipy.sparse.linalg import LinearOperator

    nstates = 21

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    # unique oo/vv pairs
    i_idx, j_idx = np.triu_indices(no, k=1)
    a_idx, b_idx = np.triu_indices(nv, k=1)

class HbarOperator:
    """
    Hbar as a LinearOperator
    """

    def __init__(self, t1, t2, f, g, no, nv):
        """
        initialize HBarOperator

        :param t1: singles amplitudes, shaped as v, o
        :param t2: doubles amplitudes, shaped as v, v, o, o
        :param f: fock matrix
        :param g: antisymmetrized eris
        :param no: number of occupied orbitals
        :param nv: number of virtual orbitals
        """

        self.t1 = t1
        self.t2 = t2
        self.f = f
        self.g = g
        self.no = no
        self.nv = nv
        self.o = slice(None, no)
        self.v = slice(no, None)

        # unique oo/vv pairs
        self.i_idx, self.j_idx = np.triu_indices(no, k=1)
        self.a_idx, self.b_idx = np.triu_indices(nv, k=1)

    def matvec(self, R):
        """
        evaluate the action of Hbar on a vector, sigma = H.R

        :param R: the vector (flat, unique elements only)

        :return sigma: the sigma vector (flat, unique elements only)
        """

        start = 0

        r0 = R[0]
        start += 1

        r1 = R[start:start + self.no*self.nv].reshape(self.t1.shape)
        start += self.no*self.nv

        # antisymmetrized eom-cc doubles amplitudes
        r2 = unpack_antisym(R[start:], self.a_idx, self.b_idx, self.i_idx, self.j_idx, self.nv, self.no)

        sigma0 = sigma_ref(self.t1, self.t2, r0, r1, r2, self.f, self.g, self.o, self.v)
        sigma1 = sigma_singles(self.t1, self.t2, r0, r1, r2, self.f, self.g, self.o, self.v)
        sigma2 = sigma_doubles(self.t1, self.t2, r0, r1, r2, self.f, self.g, self.o, self.v)

        # packed doubles part of the eom-cc sigma vector
        sigma2 = pack_antisym(sigma2, self.a_idx, self.b_idx, self.i_idx, self.j_idx)

        return np.hstack((sigma0, sigma1.flatten(), sigma2.flatten()))
