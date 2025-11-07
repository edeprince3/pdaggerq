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

        sigma0 = self.sigma_ref(r0, r1, r2)
        sigma1 = self.sigma_singles(r0, r1, r2)
        sigma2 = self.sigma_doubles(r0, r1, r2)

        # packed doubles part of the eom-cc sigma vector
        sigma2 = pack_antisym(sigma2, self.a_idx, self.b_idx, self.i_idx, self.j_idx)

        return np.hstack((sigma0, sigma1.flatten(), sigma2.flatten()))

    def sigma_ref(self, r0, r1, r2):
        """ 
        build <0| Hbar R |0>, spin-orbital basis
    
        :param r0: reference eom-cc amplitude
        :param r1: singles eom-cc amplitudes, shaped as v, o
        :param r2: doubles eom-cc amplitudes, shaped as v, v, o, o
        :param f: fock matrix
        :param g: antisymmetrized eris
        :param o: occupied slice
        :param v: virtual slice
        
        :return sigma0: the reference part of the sigma vector
        """ 
   
        t1 = self.t1 
        t2 = self.t2
        f = self.f
        g = self.g
        o = self.o
        v = self.v
    
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
    
    def sigma_singles(self, r0, r1, r2):
        """ 
        build <0| i^a Hbar R |0>, spin-orbital basis
    
        :param r0: reference eom-cc amplitude
        :param r1: singles eom-cc amplitudes, shaped as v, o
        :param r2: doubles eom-cc amplitudes, shaped as v, v, o, o
        
        :return sigma1: the singles part of the sigma vector, shaped as v, o
        """ 
    
        t1 = self.t1 
        t2 = self.t2
        f = self.f
        g = self.g
        o = self.o
        v = self.v
    
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
    
    def sigma_doubles(self, r0, r1, r2):
        """ 
        build <0| i^j^ba Hbar R |0>, spin-orbital basis
    
        :param r0: reference eom-cc amplitude
        :param r1: singles eom-cc amplitudes, shaped as v, o
        :param r2: doubles eom-cc amplitudes, shaped as v, v, o, o
        
        :return sigma2: the doubles part of the sigma vector, shaped as v, v, o, o
        """ 
    
        t1 = self.t1 
        t2 = self.t2
        f = self.f
        g = self.g
        o = self.o
        v = self.v

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

class HbarOperatorWithSpin:
    """
    Hbar as a LinearOperator, with spin-traced expressions
    """

    def __init__(self, t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, noa, nob, nva, nvb):
        """
        initialize HBarOperator

        :param t1_aa: alpha-spin singles amplitudes, shaped as va, oa
        :param t1_bb: beta-spin singles amplitudes, shaped as vb, ob
        :param t2_aaaa: alpha-alpha doubles amplitudes, shaped as va, va, oa, oa
        :param t2_bbbb: beta-beta doubles amplitudes, shaped as vb, vb, ob, ob
        :param t2_abab: alpha-beta doubles amplitudes, shaped as va, vb, oa, ob
        :param f_aa: alpha-spin fock matrix
        :param f_bb: beta-spin fock matrix
        :param g_aaaa: alpha-alpha antisymmetrized eris
        :param g_bbbb: beta-beta antisymmetrized eris
        :param g_abab: alpha-beta antisymmetrized eris
        :param noa: number of alpha-spin occupied orbitals
        :param nob: number of beta-spin occupied orbitals
        :param nva: number of alpha-spin virtual orbitals
        :param nvb: number of beta-spin virtual orbitals
        """

        self.t1_aa = t1_aa
        self.t1_bb = t1_bb
        self.t2_aaaa = t2_aaaa
        self.t2_bbbb = t2_bbbb
        self.t2_abab = t2_abab
        self.f_aa = f_aa
        self.f_bb = f_bb
        self.g_aaaa = g_aaaa
        self.g_bbbb = g_bbbb
        self.g_abab = g_abab
        self.noa = noa
        self.nob = nob
        self.nva = nva
        self.nvb = nvb
        self.oa = slice(None, noa)
        self.va = slice(noa, None)
        self.ob = slice(None, nob)
        self.vb = slice(nob, None)

        # unique oo/vv pairs
        self.i_idx_a, self.j_idx_a = np.triu_indices(noa, k=1)
        self.i_idx_b, self.j_idx_b = np.triu_indices(nob, k=1)
        self.a_idx_a, self.b_idx_a = np.triu_indices(nva, k=1)
        self.a_idx_b, self.b_idx_b = np.triu_indices(nvb, k=1)

    def matvec(self, R):
        """
        evaluate the action of Hbar on a vector, sigma = H.R

        :param R: the vector (flat, unique elements only)

        :return sigma: the sigma vector (flat, unique elements only)
        """

        start = 0

        r0 = R[0]
        start += 1

        r1_aa = R[start:start + self.noa*self.nva].reshape(self.t1_aa.shape)
        start += self.noa*self.nva

        r1_bb = R[start:start + self.nob*self.nvb].reshape(self.t1_bb.shape)
        start += self.nob*self.nvb

        # antisymmetrized eom-cc alpha-alpha doubles amplitudes
        r2_aaaa = unpack_antisym(R[start:start+len(self.i_idx_a)*len(self.a_idx_a)], self.a_idx_a, self.b_idx_a, self.i_idx_a, self.j_idx_a, self.nva, self.noa)
        start += len(self.i_idx_a)*len(self.a_idx_a)

        # antisymmetrized eom-cc beta-beta doubles amplitudes
        r2_bbbb = unpack_antisym(R[start:start+len(self.i_idx_b)*len(self.a_idx_b)], self.a_idx_b, self.b_idx_b, self.i_idx_b, self.j_idx_b, self.nvb, self.nob)
        start += len(self.i_idx_b)*len(self.a_idx_b)

        r2_abab = R[start:].reshape(self.t2_abab.shape)

        sigma0 = self.sigma_ref(r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab)
        sigma1_aa = self.sigma_singles_aa(r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab)
        sigma1_bb = self.sigma_singles_bb(r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab)
        sigma2_aaaa = self.sigma_doubles_aaaa(r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab)
        sigma2_bbbb = self.sigma_doubles_bbbb(r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab)
        sigma2_abab = self.sigma_doubles_abab(r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab)

        # packed same-spin doubles parts of the eom-cc sigma vector
        sigma2_aaaa = pack_antisym(sigma2_aaaa, self.a_idx_a, self.b_idx_a, self.i_idx_a, self.j_idx_a)
        sigma2_bbbb = pack_antisym(sigma2_bbbb, self.a_idx_b, self.b_idx_b, self.i_idx_b, self.j_idx_b)

        return np.hstack((sigma0, sigma1_aa.flatten(), sigma1_bb.flatten(), sigma2_aaaa.flatten(), sigma2_bbbb.flatten(), sigma2_abab.flatten()))

    
    def sigma_ref(self, r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab):

        t1_aa = self.t1_aa
        t1_bb = self.t1_bb
        t2_aaaa = self.t2_aaaa
        t2_bbbb = self.t2_bbbb
        t2_abab = self.t2_abab
        f_aa = self.f_aa
        f_bb = self.f_bb
        g_aaaa = self.g_aaaa
        g_bbbb = self.g_bbbb
        g_abab = self.g_abab
        oa = self.oa
        va = self.va
        ob = self.ob
        vb = self.vb
    
        #	  1.00 f_aa(i,i)*r0
        sigma0 =  1.00 * einsum('ii,', f_aa[oa, oa], r0)
        
        #	  1.00 f_bb(i,i)*r0
        sigma0 +=  1.00 * einsum('ii,', f_bb[ob, ob], r0)
        
        #	  1.00 f_aa(i,a)*r1_aa(a,i)
        sigma0 +=  1.00 * einsum('ia,ai', f_aa[oa, va], r1_aa)
        
        #	  1.00 f_bb(i,a)*r1_bb(a,i)
        sigma0 +=  1.00 * einsum('ia,ai', f_bb[ob, vb], r1_bb)
        
        #	  1.00 f_aa(i,a)*t1_aa(a,i)*r0
        sigma0 +=  1.00 * einsum('ia,ai,', f_aa[oa, va], t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 f_bb(i,a)*t1_bb(a,i)*r0
        sigma0 +=  1.00 * einsum('ia,ai,', f_bb[ob, vb], t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,i||j,i>_aaaa*r0
        sigma0 += -0.50 * einsum('jiji,', g_aaaa[oa, oa, oa, oa], r0)
        
        #	 -0.50 <j,i||j,i>_abab*r0
        sigma0 += -0.50 * einsum('jiji,', g_abab[oa, ob, oa, ob], r0)
        
        #	 -0.50 <i,j||i,j>_abab*r0
        sigma0 += -0.50 * einsum('ijij,', g_abab[oa, ob, oa, ob], r0)
        
        #	 -0.50 <j,i||j,i>_bbbb*r0
        sigma0 += -0.50 * einsum('jiji,', g_bbbb[ob, ob, ob, ob], r0)
        
        #	  0.250 <j,i||a,b>_aaaa*r2_aaaa(a,b,j,i)
        sigma0 +=  0.250 * einsum('jiab,abji', g_aaaa[oa, oa, va, va], r2_aaaa)
        
        #	  0.250 <j,i||a,b>_abab*r2_abab(a,b,j,i)
        sigma0 +=  0.250 * einsum('jiab,abji', g_abab[oa, ob, va, vb], r2_abab)
        
        #	  0.250 <i,j||a,b>_abab*r2_abab(a,b,i,j)
        sigma0 +=  0.250 * einsum('ijab,abij', g_abab[oa, ob, va, vb], r2_abab)
        
        #	  0.250 <j,i||b,a>_abab*r2_abab(b,a,j,i)
        sigma0 +=  0.250 * einsum('jiba,baji', g_abab[oa, ob, va, vb], r2_abab)
        
        #	  0.250 <i,j||b,a>_abab*r2_abab(b,a,i,j)
        sigma0 +=  0.250 * einsum('ijba,baij', g_abab[oa, ob, va, vb], r2_abab)
        
        #	  0.250 <j,i||a,b>_bbbb*r2_bbbb(a,b,j,i)
        sigma0 +=  0.250 * einsum('jiab,abji', g_bbbb[ob, ob, vb, vb], r2_bbbb)
        
        #	 -1.00 <j,i||a,b>_aaaa*t1_aa(a,i)*r1_aa(b,j)
        sigma0 += -1.00 * einsum('jiab,ai,bj', g_aaaa[oa, oa, va, va], t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <i,j||a,b>_abab*t1_aa(a,i)*r1_bb(b,j)
        sigma0 +=  1.00 * einsum('ijab,ai,bj', g_abab[oa, ob, va, vb], t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <j,i||b,a>_abab*t1_bb(a,i)*r1_aa(b,j)
        sigma0 +=  1.00 * einsum('jiba,ai,bj', g_abab[oa, ob, va, vb], t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <j,i||a,b>_bbbb*t1_bb(a,i)*r1_bb(b,j)
        sigma0 += -1.00 * einsum('jiab,ai,bj', g_bbbb[ob, ob, vb, vb], t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <j,i||a,b>_aaaa*t2_aaaa(a,b,j,i)*r0
        sigma0 +=  0.250 * einsum('jiab,abji,', g_aaaa[oa, oa, va, va], t2_aaaa, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <j,i||a,b>_abab*t2_abab(a,b,j,i)*r0
        sigma0 +=  0.250 * einsum('jiab,abji,', g_abab[oa, ob, va, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <i,j||a,b>_abab*t2_abab(a,b,i,j)*r0
        sigma0 +=  0.250 * einsum('ijab,abij,', g_abab[oa, ob, va, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <j,i||b,a>_abab*t2_abab(b,a,j,i)*r0
        sigma0 +=  0.250 * einsum('jiba,baji,', g_abab[oa, ob, va, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <i,j||b,a>_abab*t2_abab(b,a,i,j)*r0
        sigma0 +=  0.250 * einsum('ijba,baij,', g_abab[oa, ob, va, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <j,i||a,b>_bbbb*t2_bbbb(a,b,j,i)*r0
        sigma0 +=  0.250 * einsum('jiab,abji,', g_bbbb[ob, ob, vb, vb], t2_bbbb, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,i||a,b>_aaaa*t1_aa(a,i)*t1_aa(b,j)*r0
        sigma0 += -0.50 * einsum('jiab,ai,bj,', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <i,j||a,b>_abab*t1_aa(a,i)*t1_bb(b,j)*r0
        sigma0 +=  0.50 * einsum('ijab,ai,bj,', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <j,i||b,a>_abab*t1_bb(a,i)*t1_aa(b,j)*r0
        sigma0 +=  0.50 * einsum('jiba,ai,bj,', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <j,i||a,b>_bbbb*t1_bb(a,i)*t1_bb(b,j)*r0
        sigma0 += -0.50 * einsum('jiab,ai,bj,', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        return sigma0
    
    
    def sigma_singles_aa(self, r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab):

        t1_aa = self.t1_aa
        t1_bb = self.t1_bb
        t2_aaaa = self.t2_aaaa
        t2_bbbb = self.t2_bbbb
        t2_abab = self.t2_abab
        f_aa = self.f_aa
        f_bb = self.f_bb
        g_aaaa = self.g_aaaa
        g_bbbb = self.g_bbbb
        g_abab = self.g_abab
        oa = self.oa
        va = self.va
        ob = self.ob
        vb = self.vb
    
        #	  1.00 f_aa(a,i)*r0
        sigma1_aa =  1.00 * einsum('ai,->ai', f_aa[va, oa], r0)
        
        #	  1.00 f_aa(j,j)*r1_aa(a,i)
        sigma1_aa +=  1.00 * einsum('jj,ai->ai', f_aa[oa, oa], r1_aa)
        
        #	  1.00 f_bb(j,j)*r1_aa(a,i)
        sigma1_aa +=  1.00 * einsum('jj,ai->ai', f_bb[ob, ob], r1_aa)
        
        #	 -1.00 f_aa(j,i)*r1_aa(a,j)
        sigma1_aa += -1.00 * einsum('ji,aj->ai', f_aa[oa, oa], r1_aa)
        
        #	  1.00 f_aa(a,b)*r1_aa(b,i)
        sigma1_aa +=  1.00 * einsum('ab,bi->ai', f_aa[va, va], r1_aa)
        
        #	 -1.00 f_aa(j,b)*r2_aaaa(b,a,i,j)
        sigma1_aa += -1.00 * einsum('jb,baij->ai', f_aa[oa, va], r2_aaaa)
        
        #	  1.00 f_bb(j,b)*r2_abab(a,b,i,j)
        sigma1_aa +=  1.00 * einsum('jb,abij->ai', f_bb[ob, vb], r2_abab)
        
        #	 -1.00 f_aa(j,i)*t1_aa(a,j)*r0
        sigma1_aa += -1.00 * einsum('ji,aj,->ai', f_aa[oa, oa], t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 f_aa(a,b)*t1_aa(b,i)*r0
        sigma1_aa +=  1.00 * einsum('ab,bi,->ai', f_aa[va, va], t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 f_aa(j,b)*r1_aa(a,i)*t1_aa(b,j)
        sigma1_aa +=  1.00 * einsum('jb,ai,bj->ai', f_aa[oa, va], r1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 f_bb(j,b)*r1_aa(a,i)*t1_bb(b,j)
        sigma1_aa +=  1.00 * einsum('jb,ai,bj->ai', f_bb[ob, vb], r1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_aa(j,b)*r1_aa(a,j)*t1_aa(b,i)
        sigma1_aa += -1.00 * einsum('jb,aj,bi->ai', f_aa[oa, va], r1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 f_aa(j,b)*r1_aa(b,i)*t1_aa(a,j)
        sigma1_aa += -1.00 * einsum('jb,bi,aj->ai', f_aa[oa, va], r1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_aa(j,b)*t2_aaaa(b,a,i,j)*r0
        sigma1_aa += -1.00 * einsum('jb,baij,->ai', f_aa[oa, va], t2_aaaa, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 f_bb(j,b)*t2_abab(a,b,i,j)*r0
        sigma1_aa +=  1.00 * einsum('jb,abij,->ai', f_bb[ob, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 f_aa(j,b)*t1_aa(a,j)*t1_aa(b,i)*r0
        sigma1_aa += -1.00 * einsum('jb,aj,bi,->ai', f_aa[oa, va], t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <k,j||k,j>_aaaa*r1_aa(a,i)
        sigma1_aa += -0.50 * einsum('kjkj,ai->ai', g_aaaa[oa, oa, oa, oa], r1_aa)
        
        #	 -0.50 <k,j||k,j>_abab*r1_aa(a,i)
        sigma1_aa += -0.50 * einsum('kjkj,ai->ai', g_abab[oa, ob, oa, ob], r1_aa)
        
        #	 -0.50 <j,k||j,k>_abab*r1_aa(a,i)
        sigma1_aa += -0.50 * einsum('jkjk,ai->ai', g_abab[oa, ob, oa, ob], r1_aa)
        
        #	 -0.50 <k,j||k,j>_bbbb*r1_aa(a,i)
        sigma1_aa += -0.50 * einsum('kjkj,ai->ai', g_bbbb[ob, ob, ob, ob], r1_aa)
        
        #	  1.00 <j,a||b,i>_aaaa*r1_aa(b,j)
        sigma1_aa +=  1.00 * einsum('jabi,bj->ai', g_aaaa[oa, va, va, oa], r1_aa)
        
        #	  1.00 <a,j||i,b>_abab*r1_bb(b,j)
        sigma1_aa +=  1.00 * einsum('ajib,bj->ai', g_abab[va, ob, oa, vb], r1_bb)
        
        #	 -0.50 <k,j||b,i>_aaaa*r2_aaaa(b,a,k,j)
        sigma1_aa += -0.50 * einsum('kjbi,bakj->ai', g_aaaa[oa, oa, va, oa], r2_aaaa)
        
        #	 -0.50 <k,j||i,b>_abab*r2_abab(a,b,k,j)
        sigma1_aa += -0.50 * einsum('kjib,abkj->ai', g_abab[oa, ob, oa, vb], r2_abab)
        
        #	 -0.50 <j,k||i,b>_abab*r2_abab(a,b,j,k)
        sigma1_aa += -0.50 * einsum('jkib,abjk->ai', g_abab[oa, ob, oa, vb], r2_abab)
        
        #	 -0.50 <j,a||b,c>_aaaa*r2_aaaa(b,c,i,j)
        sigma1_aa += -0.50 * einsum('jabc,bcij->ai', g_aaaa[oa, va, va, va], r2_aaaa)
        
        #	  0.50 <a,j||b,c>_abab*r2_abab(b,c,i,j)
        sigma1_aa +=  0.50 * einsum('ajbc,bcij->ai', g_abab[va, ob, va, vb], r2_abab)
        
        #	  0.50 <a,j||c,b>_abab*r2_abab(c,b,i,j)
        sigma1_aa +=  0.50 * einsum('ajcb,cbij->ai', g_abab[va, ob, va, vb], r2_abab)
        
        #	  1.00 <j,a||b,i>_aaaa*t1_aa(b,j)*r0
        sigma1_aa +=  1.00 * einsum('jabi,bj,->ai', g_aaaa[oa, va, va, oa], t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <a,j||i,b>_abab*t1_bb(b,j)*r0
        sigma1_aa +=  1.00 * einsum('ajib,bj,->ai', g_abab[va, ob, oa, vb], t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,j||b,i>_aaaa*t1_aa(b,j)*r1_aa(a,k)
        sigma1_aa +=  1.00 * einsum('kjbi,bj,ak->ai', g_aaaa[oa, oa, va, oa], t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,j||i,b>_abab*t1_bb(b,j)*r1_aa(a,k)
        sigma1_aa += -1.00 * einsum('kjib,bj,ak->ai', g_abab[oa, ob, oa, vb], t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,j||b,i>_aaaa*t1_aa(a,j)*r1_aa(b,k)
        sigma1_aa += -1.00 * einsum('kjbi,aj,bk->ai', g_aaaa[oa, oa, va, oa], t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <j,k||i,b>_abab*t1_aa(a,j)*r1_bb(b,k)
        sigma1_aa += -1.00 * einsum('jkib,aj,bk->ai', g_abab[oa, ob, oa, vb], t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <j,a||b,c>_aaaa*t1_aa(b,j)*r1_aa(c,i)
        sigma1_aa +=  1.00 * einsum('jabc,bj,ci->ai', g_aaaa[oa, va, va, va], t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <a,j||c,b>_abab*t1_bb(b,j)*r1_aa(c,i)
        sigma1_aa +=  1.00 * einsum('ajcb,bj,ci->ai', g_abab[va, ob, va, vb], t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <j,a||b,c>_aaaa*t1_aa(b,i)*r1_aa(c,j)
        sigma1_aa += -1.00 * einsum('jabc,bi,cj->ai', g_aaaa[oa, va, va, va], t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <a,j||b,c>_abab*t1_aa(b,i)*r1_bb(c,j)
        sigma1_aa +=  1.00 * einsum('ajbc,bi,cj->ai', g_abab[va, ob, va, vb], t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_aaaa*t1_aa(b,j)*r2_aaaa(c,a,i,k)
        sigma1_aa +=  1.00 * einsum('kjbc,bj,caik->ai', g_aaaa[oa, oa, va, va], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <j,k||b,c>_abab*t1_aa(b,j)*r2_abab(a,c,i,k)
        sigma1_aa +=  1.00 * einsum('jkbc,bj,acik->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,j||c,b>_abab*t1_bb(b,j)*r2_aaaa(c,a,i,k)
        sigma1_aa += -1.00 * einsum('kjcb,bj,caik->ai', g_abab[oa, ob, va, vb], t1_bb, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,j||b,c>_bbbb*t1_bb(b,j)*r2_abab(a,c,i,k)
        sigma1_aa += -1.00 * einsum('kjbc,bj,acik->ai', g_bbbb[ob, ob, vb, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <k,j||b,c>_aaaa*t1_aa(b,i)*r2_aaaa(c,a,k,j)
        sigma1_aa +=  0.50 * einsum('kjbc,bi,cakj->ai', g_aaaa[oa, oa, va, va], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_abab*t1_aa(b,i)*r2_abab(a,c,k,j)
        sigma1_aa += -0.50 * einsum('kjbc,bi,ackj->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <j,k||b,c>_abab*t1_aa(b,i)*r2_abab(a,c,j,k)
        sigma1_aa += -0.50 * einsum('jkbc,bi,acjk->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 <k,j||b,c>_aaaa*t1_aa(a,j)*r2_aaaa(b,c,i,k)
        sigma1_aa +=  0.50 * einsum('kjbc,aj,bcik->ai', g_aaaa[oa, oa, va, va], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,k||b,c>_abab*t1_aa(a,j)*r2_abab(b,c,i,k)
        sigma1_aa += -0.50 * einsum('jkbc,aj,bcik->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,k||c,b>_abab*t1_aa(a,j)*r2_abab(c,b,i,k)
        sigma1_aa += -0.50 * einsum('jkcb,aj,cbik->ai', g_abab[oa, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,j||b,i>_aaaa*t2_aaaa(b,a,k,j)*r0
        sigma1_aa += -0.50 * einsum('kjbi,bakj,->ai', g_aaaa[oa, oa, va, oa], t2_aaaa, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,j||i,b>_abab*t2_abab(a,b,k,j)*r0
        sigma1_aa += -0.50 * einsum('kjib,abkj,->ai', g_abab[oa, ob, oa, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,k||i,b>_abab*t2_abab(a,b,j,k)*r0
        sigma1_aa += -0.50 * einsum('jkib,abjk,->ai', g_abab[oa, ob, oa, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,a||b,c>_aaaa*t2_aaaa(b,c,i,j)*r0
        sigma1_aa += -0.50 * einsum('jabc,bcij,->ai', g_aaaa[oa, va, va, va], t2_aaaa, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <a,j||b,c>_abab*t2_abab(b,c,i,j)*r0
        sigma1_aa +=  0.50 * einsum('ajbc,bcij,->ai', g_abab[va, ob, va, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <a,j||c,b>_abab*t2_abab(c,b,i,j)*r0
        sigma1_aa +=  0.50 * einsum('ajcb,cbij,->ai', g_abab[va, ob, va, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,j||b,c>_aaaa*t2_aaaa(b,c,k,j)*r1_aa(a,i)
        sigma1_aa +=  0.250 * einsum('kjbc,bckj,ai->ai', g_aaaa[oa, oa, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,j||b,c>_abab*t2_abab(b,c,k,j)*r1_aa(a,i)
        sigma1_aa +=  0.250 * einsum('kjbc,bckj,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <j,k||b,c>_abab*t2_abab(b,c,j,k)*r1_aa(a,i)
        sigma1_aa +=  0.250 * einsum('jkbc,bcjk,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,j||c,b>_abab*t2_abab(c,b,k,j)*r1_aa(a,i)
        sigma1_aa +=  0.250 * einsum('kjcb,cbkj,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <j,k||c,b>_abab*t2_abab(c,b,j,k)*r1_aa(a,i)
        sigma1_aa +=  0.250 * einsum('jkcb,cbjk,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,j||b,c>_bbbb*t2_bbbb(b,c,k,j)*r1_aa(a,i)
        sigma1_aa +=  0.250 * einsum('kjbc,bckj,ai->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_aaaa*t2_aaaa(b,c,i,j)*r1_aa(a,k)
        sigma1_aa += -0.50 * einsum('kjbc,bcij,ak->ai', g_aaaa[oa, oa, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_abab*t2_abab(b,c,i,j)*r1_aa(a,k)
        sigma1_aa += -0.50 * einsum('kjbc,bcij,ak->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,j||c,b>_abab*t2_abab(c,b,i,j)*r1_aa(a,k)
        sigma1_aa += -0.50 * einsum('kjcb,cbij,ak->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_aaaa*t2_aaaa(b,a,k,j)*r1_aa(c,i)
        sigma1_aa += -0.50 * einsum('kjbc,bakj,ci->ai', g_aaaa[oa, oa, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,j||c,b>_abab*t2_abab(a,b,k,j)*r1_aa(c,i)
        sigma1_aa += -0.50 * einsum('kjcb,abkj,ci->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,k||c,b>_abab*t2_abab(a,b,j,k)*r1_aa(c,i)
        sigma1_aa += -0.50 * einsum('jkcb,abjk,ci->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,j||b,c>_aaaa*t2_aaaa(b,a,i,j)*r1_aa(c,k)
        sigma1_aa +=  1.00 * einsum('kjbc,baij,ck->ai', g_aaaa[oa, oa, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <j,k||b,c>_abab*t2_aaaa(b,a,i,j)*r1_bb(c,k)
        sigma1_aa += -1.00 * einsum('jkbc,baij,ck->ai', g_abab[oa, ob, va, vb], t2_aaaa, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,j||c,b>_abab*t2_abab(a,b,i,j)*r1_aa(c,k)
        sigma1_aa +=  1.00 * einsum('kjcb,abij,ck->ai', g_abab[oa, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,j||b,c>_bbbb*t2_abab(a,b,i,j)*r1_bb(c,k)
        sigma1_aa += -1.00 * einsum('kjbc,abij,ck->ai', g_bbbb[ob, ob, vb, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_aaaa*t2_aaaa(c,a,i,k)*t1_aa(b,j)*r0
        sigma1_aa +=  1.00 * einsum('kjbc,caik,bj,->ai', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <k,j||c,b>_abab*t2_aaaa(c,a,i,k)*t1_bb(b,j)*r0
        sigma1_aa += -1.00 * einsum('kjcb,caik,bj,->ai', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <j,k||b,c>_abab*t2_abab(a,c,i,k)*t1_aa(b,j)*r0
        sigma1_aa +=  1.00 * einsum('jkbc,acik,bj,->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <k,j||b,c>_bbbb*t2_abab(a,c,i,k)*t1_bb(b,j)*r0
        sigma1_aa += -1.00 * einsum('kjbc,acik,bj,->ai', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  0.50 <k,j||b,c>_aaaa*t2_aaaa(c,a,k,j)*t1_aa(b,i)*r0
        sigma1_aa +=  0.50 * einsum('kjbc,cakj,bi,->ai', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_abab*t2_abab(a,c,k,j)*t1_aa(b,i)*r0
        sigma1_aa += -0.50 * einsum('kjbc,ackj,bi,->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <j,k||b,c>_abab*t2_abab(a,c,j,k)*t1_aa(b,i)*r0
        sigma1_aa += -0.50 * einsum('jkbc,acjk,bi,->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  0.50 <k,j||b,c>_aaaa*t1_aa(a,j)*t2_aaaa(b,c,i,k)*r0
        sigma1_aa +=  0.50 * einsum('kjbc,aj,bcik,->ai', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <j,k||b,c>_abab*t1_aa(a,j)*t2_abab(b,c,i,k)*r0
        sigma1_aa += -0.50 * einsum('jkbc,aj,bcik,->ai', g_abab[oa, ob, va, vb], t1_aa, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <j,k||c,b>_abab*t1_aa(a,j)*t2_abab(c,b,i,k)*r0
        sigma1_aa += -0.50 * einsum('jkcb,aj,cbik,->ai', g_abab[oa, ob, va, vb], t1_aa, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <k,j||b,i>_aaaa*t1_aa(a,k)*t1_aa(b,j)*r0
        sigma1_aa +=  1.00 * einsum('kjbi,ak,bj,->ai', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
        
        #	 -1.00 <k,j||i,b>_abab*t1_aa(a,k)*t1_bb(b,j)*r0
        sigma1_aa += -1.00 * einsum('kjib,ak,bj,->ai', g_abab[oa, ob, oa, vb], t1_aa, t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
        
        #	  1.00 <j,a||b,c>_aaaa*t1_aa(b,j)*t1_aa(c,i)*r0
        sigma1_aa +=  1.00 * einsum('jabc,bj,ci,->ai', g_aaaa[oa, va, va, va], t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 <a,j||c,b>_abab*t1_bb(b,j)*t1_aa(c,i)*r0
        sigma1_aa +=  1.00 * einsum('ajcb,bj,ci,->ai', g_abab[va, ob, va, vb], t1_bb, t1_aa, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_aaaa*t1_aa(b,j)*t1_aa(c,k)*r1_aa(a,i)
        sigma1_aa += -0.50 * einsum('kjbc,bj,ck,ai->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <j,k||b,c>_abab*t1_aa(b,j)*t1_bb(c,k)*r1_aa(a,i)
        sigma1_aa +=  0.50 * einsum('jkbc,bj,ck,ai->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <k,j||c,b>_abab*t1_bb(b,j)*t1_aa(c,k)*r1_aa(a,i)
        sigma1_aa +=  0.50 * einsum('kjcb,bj,ck,ai->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_bbbb*t1_bb(b,j)*t1_bb(c,k)*r1_aa(a,i)
        sigma1_aa += -0.50 * einsum('kjbc,bj,ck,ai->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_aaaa*t1_aa(b,j)*t1_aa(c,i)*r1_aa(a,k)
        sigma1_aa +=  1.00 * einsum('kjbc,bj,ci,ak->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -1.00 <k,j||c,b>_abab*t1_bb(b,j)*t1_aa(c,i)*r1_aa(a,k)
        sigma1_aa += -1.00 * einsum('kjcb,bj,ci,ak->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_aaaa*t1_aa(a,k)*t1_aa(b,j)*r1_aa(c,i)
        sigma1_aa +=  1.00 * einsum('kjbc,ak,bj,ci->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <k,j||c,b>_abab*t1_aa(a,k)*t1_bb(b,j)*r1_aa(c,i)
        sigma1_aa += -1.00 * einsum('kjcb,ak,bj,ci->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_aaaa*t1_aa(a,j)*t1_aa(b,i)*r1_aa(c,k)
        sigma1_aa +=  1.00 * einsum('kjbc,aj,bi,ck->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <j,k||b,c>_abab*t1_aa(a,j)*t1_aa(b,i)*r1_bb(c,k)
        sigma1_aa += -1.00 * einsum('jkbc,aj,bi,ck->ai', g_abab[oa, ob, va, vb], t1_aa, t1_aa, r1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_aaaa*t1_aa(a,k)*t1_aa(b,j)*t1_aa(c,i)*r0
        sigma1_aa +=  1.00 * einsum('kjbc,ak,bj,ci,->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 3), (1, 2), (0, 1)])
        
        #	 -1.00 <k,j||c,b>_abab*t1_aa(a,k)*t1_bb(b,j)*t1_aa(c,i)*r0
        sigma1_aa += -1.00 * einsum('kjcb,ak,bj,ci,->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 3), (1, 2), (0, 1)])
        
        return sigma1_aa
    
    
    def sigma_singles_bb(self, r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab):

        t1_aa = self.t1_aa
        t1_bb = self.t1_bb
        t2_aaaa = self.t2_aaaa
        t2_bbbb = self.t2_bbbb
        t2_abab = self.t2_abab
        f_aa = self.f_aa
        f_bb = self.f_bb
        g_aaaa = self.g_aaaa
        g_bbbb = self.g_bbbb
        g_abab = self.g_abab
        oa = self.oa
        va = self.va
        ob = self.ob
        vb = self.vb
    
        #	  1.00 f_bb(a,i)*r0
        sigma1_bb =  1.00 * einsum('ai,->ai', f_bb[vb, ob], r0)
        
        #	  1.00 f_aa(j,j)*r1_bb(a,i)
        sigma1_bb +=  1.00 * einsum('jj,ai->ai', f_aa[oa, oa], r1_bb)
        
        #	  1.00 f_bb(j,j)*r1_bb(a,i)
        sigma1_bb +=  1.00 * einsum('jj,ai->ai', f_bb[ob, ob], r1_bb)
        
        #	 -1.00 f_bb(j,i)*r1_bb(a,j)
        sigma1_bb += -1.00 * einsum('ji,aj->ai', f_bb[ob, ob], r1_bb)
        
        #	  1.00 f_bb(a,b)*r1_bb(b,i)
        sigma1_bb +=  1.00 * einsum('ab,bi->ai', f_bb[vb, vb], r1_bb)
        
        #	  1.00 f_aa(j,b)*r2_abab(b,a,j,i)
        sigma1_bb +=  1.00 * einsum('jb,baji->ai', f_aa[oa, va], r2_abab)
        
        #	 -1.00 f_bb(j,b)*r2_bbbb(b,a,i,j)
        sigma1_bb += -1.00 * einsum('jb,baij->ai', f_bb[ob, vb], r2_bbbb)
        
        #	 -1.00 f_bb(j,i)*t1_bb(a,j)*r0
        sigma1_bb += -1.00 * einsum('ji,aj,->ai', f_bb[ob, ob], t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 f_bb(a,b)*t1_bb(b,i)*r0
        sigma1_bb +=  1.00 * einsum('ab,bi,->ai', f_bb[vb, vb], t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 f_aa(j,b)*r1_bb(a,i)*t1_aa(b,j)
        sigma1_bb +=  1.00 * einsum('jb,ai,bj->ai', f_aa[oa, va], r1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 f_bb(j,b)*r1_bb(a,i)*t1_bb(b,j)
        sigma1_bb +=  1.00 * einsum('jb,ai,bj->ai', f_bb[ob, vb], r1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_bb(j,b)*r1_bb(a,j)*t1_bb(b,i)
        sigma1_bb += -1.00 * einsum('jb,aj,bi->ai', f_bb[ob, vb], r1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_bb(j,b)*r1_bb(b,i)*t1_bb(a,j)
        sigma1_bb += -1.00 * einsum('jb,bi,aj->ai', f_bb[ob, vb], r1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 f_aa(j,b)*t2_abab(b,a,j,i)*r0
        sigma1_bb +=  1.00 * einsum('jb,baji,->ai', f_aa[oa, va], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 f_bb(j,b)*t2_bbbb(b,a,i,j)*r0
        sigma1_bb += -1.00 * einsum('jb,baij,->ai', f_bb[ob, vb], t2_bbbb, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 f_bb(j,b)*t1_bb(a,j)*t1_bb(b,i)*r0
        sigma1_bb += -1.00 * einsum('jb,aj,bi,->ai', f_bb[ob, vb], t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -0.50 <k,j||k,j>_aaaa*r1_bb(a,i)
        sigma1_bb += -0.50 * einsum('kjkj,ai->ai', g_aaaa[oa, oa, oa, oa], r1_bb)
        
        #	 -0.50 <k,j||k,j>_abab*r1_bb(a,i)
        sigma1_bb += -0.50 * einsum('kjkj,ai->ai', g_abab[oa, ob, oa, ob], r1_bb)
        
        #	 -0.50 <j,k||j,k>_abab*r1_bb(a,i)
        sigma1_bb += -0.50 * einsum('jkjk,ai->ai', g_abab[oa, ob, oa, ob], r1_bb)
        
        #	 -0.50 <k,j||k,j>_bbbb*r1_bb(a,i)
        sigma1_bb += -0.50 * einsum('kjkj,ai->ai', g_bbbb[ob, ob, ob, ob], r1_bb)
        
        #	  1.00 <j,a||b,i>_abab*r1_aa(b,j)
        sigma1_bb +=  1.00 * einsum('jabi,bj->ai', g_abab[oa, vb, va, ob], r1_aa)
        
        #	  1.00 <j,a||b,i>_bbbb*r1_bb(b,j)
        sigma1_bb +=  1.00 * einsum('jabi,bj->ai', g_bbbb[ob, vb, vb, ob], r1_bb)
        
        #	 -0.50 <k,j||b,i>_abab*r2_abab(b,a,k,j)
        sigma1_bb += -0.50 * einsum('kjbi,bakj->ai', g_abab[oa, ob, va, ob], r2_abab)
        
        #	 -0.50 <j,k||b,i>_abab*r2_abab(b,a,j,k)
        sigma1_bb += -0.50 * einsum('jkbi,bajk->ai', g_abab[oa, ob, va, ob], r2_abab)
        
        #	 -0.50 <k,j||b,i>_bbbb*r2_bbbb(b,a,k,j)
        sigma1_bb += -0.50 * einsum('kjbi,bakj->ai', g_bbbb[ob, ob, vb, ob], r2_bbbb)
        
        #	  0.50 <j,a||b,c>_abab*r2_abab(b,c,j,i)
        sigma1_bb +=  0.50 * einsum('jabc,bcji->ai', g_abab[oa, vb, va, vb], r2_abab)
        
        #	  0.50 <j,a||c,b>_abab*r2_abab(c,b,j,i)
        sigma1_bb +=  0.50 * einsum('jacb,cbji->ai', g_abab[oa, vb, va, vb], r2_abab)
        
        #	 -0.50 <j,a||b,c>_bbbb*r2_bbbb(b,c,i,j)
        sigma1_bb += -0.50 * einsum('jabc,bcij->ai', g_bbbb[ob, vb, vb, vb], r2_bbbb)
        
        #	  1.00 <j,a||b,i>_abab*t1_aa(b,j)*r0
        sigma1_bb +=  1.00 * einsum('jabi,bj,->ai', g_abab[oa, vb, va, ob], t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <j,a||b,i>_bbbb*t1_bb(b,j)*r0
        sigma1_bb +=  1.00 * einsum('jabi,bj,->ai', g_bbbb[ob, vb, vb, ob], t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <j,k||b,i>_abab*t1_aa(b,j)*r1_bb(a,k)
        sigma1_bb += -1.00 * einsum('jkbi,bj,ak->ai', g_abab[oa, ob, va, ob], t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,j||b,i>_bbbb*t1_bb(b,j)*r1_bb(a,k)
        sigma1_bb +=  1.00 * einsum('kjbi,bj,ak->ai', g_bbbb[ob, ob, vb, ob], t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,j||b,i>_abab*t1_bb(a,j)*r1_aa(b,k)
        sigma1_bb += -1.00 * einsum('kjbi,aj,bk->ai', g_abab[oa, ob, va, ob], t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,j||b,i>_bbbb*t1_bb(a,j)*r1_bb(b,k)
        sigma1_bb += -1.00 * einsum('kjbi,aj,bk->ai', g_bbbb[ob, ob, vb, ob], t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <j,a||b,c>_abab*t1_aa(b,j)*r1_bb(c,i)
        sigma1_bb +=  1.00 * einsum('jabc,bj,ci->ai', g_abab[oa, vb, va, vb], t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <j,a||b,c>_bbbb*t1_bb(b,j)*r1_bb(c,i)
        sigma1_bb +=  1.00 * einsum('jabc,bj,ci->ai', g_bbbb[ob, vb, vb, vb], t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <j,a||c,b>_abab*t1_bb(b,i)*r1_aa(c,j)
        sigma1_bb +=  1.00 * einsum('jacb,bi,cj->ai', g_abab[oa, vb, va, vb], t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <j,a||b,c>_bbbb*t1_bb(b,i)*r1_bb(c,j)
        sigma1_bb += -1.00 * einsum('jabc,bi,cj->ai', g_bbbb[ob, vb, vb, vb], t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,j||b,c>_aaaa*t1_aa(b,j)*r2_abab(c,a,k,i)
        sigma1_bb += -1.00 * einsum('kjbc,bj,caki->ai', g_aaaa[oa, oa, va, va], t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <j,k||b,c>_abab*t1_aa(b,j)*r2_bbbb(c,a,i,k)
        sigma1_bb += -1.00 * einsum('jkbc,bj,caik->ai', g_abab[oa, ob, va, vb], t1_aa, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,j||c,b>_abab*t1_bb(b,j)*r2_abab(c,a,k,i)
        sigma1_bb +=  1.00 * einsum('kjcb,bj,caki->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,j||b,c>_bbbb*t1_bb(b,j)*r2_bbbb(c,a,i,k)
        sigma1_bb +=  1.00 * einsum('kjbc,bj,caik->ai', g_bbbb[ob, ob, vb, vb], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,j||c,b>_abab*t1_bb(b,i)*r2_abab(c,a,k,j)
        sigma1_bb += -0.50 * einsum('kjcb,bi,cakj->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,k||c,b>_abab*t1_bb(b,i)*r2_abab(c,a,j,k)
        sigma1_bb += -0.50 * einsum('jkcb,bi,cajk->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <k,j||b,c>_bbbb*t1_bb(b,i)*r2_bbbb(c,a,k,j)
        sigma1_bb +=  0.50 * einsum('kjbc,bi,cakj->ai', g_bbbb[ob, ob, vb, vb], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_abab*t1_bb(a,j)*r2_abab(b,c,k,i)
        sigma1_bb += -0.50 * einsum('kjbc,aj,bcki->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,j||c,b>_abab*t1_bb(a,j)*r2_abab(c,b,k,i)
        sigma1_bb += -0.50 * einsum('kjcb,aj,cbki->ai', g_abab[oa, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 <k,j||b,c>_bbbb*t1_bb(a,j)*r2_bbbb(b,c,i,k)
        sigma1_bb +=  0.50 * einsum('kjbc,aj,bcik->ai', g_bbbb[ob, ob, vb, vb], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,j||b,i>_abab*t2_abab(b,a,k,j)*r0
        sigma1_bb += -0.50 * einsum('kjbi,bakj,->ai', g_abab[oa, ob, va, ob], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,k||b,i>_abab*t2_abab(b,a,j,k)*r0
        sigma1_bb += -0.50 * einsum('jkbi,bajk,->ai', g_abab[oa, ob, va, ob], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,j||b,i>_bbbb*t2_bbbb(b,a,k,j)*r0
        sigma1_bb += -0.50 * einsum('kjbi,bakj,->ai', g_bbbb[ob, ob, vb, ob], t2_bbbb, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <j,a||b,c>_abab*t2_abab(b,c,j,i)*r0
        sigma1_bb +=  0.50 * einsum('jabc,bcji,->ai', g_abab[oa, vb, va, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <j,a||c,b>_abab*t2_abab(c,b,j,i)*r0
        sigma1_bb +=  0.50 * einsum('jacb,cbji,->ai', g_abab[oa, vb, va, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,a||b,c>_bbbb*t2_bbbb(b,c,i,j)*r0
        sigma1_bb += -0.50 * einsum('jabc,bcij,->ai', g_bbbb[ob, vb, vb, vb], t2_bbbb, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,j||b,c>_aaaa*t2_aaaa(b,c,k,j)*r1_bb(a,i)
        sigma1_bb +=  0.250 * einsum('kjbc,bckj,ai->ai', g_aaaa[oa, oa, va, va], t2_aaaa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,j||b,c>_abab*t2_abab(b,c,k,j)*r1_bb(a,i)
        sigma1_bb +=  0.250 * einsum('kjbc,bckj,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <j,k||b,c>_abab*t2_abab(b,c,j,k)*r1_bb(a,i)
        sigma1_bb +=  0.250 * einsum('jkbc,bcjk,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,j||c,b>_abab*t2_abab(c,b,k,j)*r1_bb(a,i)
        sigma1_bb +=  0.250 * einsum('kjcb,cbkj,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <j,k||c,b>_abab*t2_abab(c,b,j,k)*r1_bb(a,i)
        sigma1_bb +=  0.250 * einsum('jkcb,cbjk,ai->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,j||b,c>_bbbb*t2_bbbb(b,c,k,j)*r1_bb(a,i)
        sigma1_bb +=  0.250 * einsum('kjbc,bckj,ai->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,k||b,c>_abab*t2_abab(b,c,j,i)*r1_bb(a,k)
        sigma1_bb += -0.50 * einsum('jkbc,bcji,ak->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <j,k||c,b>_abab*t2_abab(c,b,j,i)*r1_bb(a,k)
        sigma1_bb += -0.50 * einsum('jkcb,cbji,ak->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_bbbb*t2_bbbb(b,c,i,j)*r1_bb(a,k)
        sigma1_bb += -0.50 * einsum('kjbc,bcij,ak->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_abab*t2_abab(b,a,k,j)*r1_bb(c,i)
        sigma1_bb += -0.50 * einsum('kjbc,bakj,ci->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <j,k||b,c>_abab*t2_abab(b,a,j,k)*r1_bb(c,i)
        sigma1_bb += -0.50 * einsum('jkbc,bajk,ci->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_bbbb*t2_bbbb(b,a,k,j)*r1_bb(c,i)
        sigma1_bb += -0.50 * einsum('kjbc,bakj,ci->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,j||b,c>_aaaa*t2_abab(b,a,j,i)*r1_aa(c,k)
        sigma1_bb += -1.00 * einsum('kjbc,baji,ck->ai', g_aaaa[oa, oa, va, va], t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <j,k||b,c>_abab*t2_abab(b,a,j,i)*r1_bb(c,k)
        sigma1_bb +=  1.00 * einsum('jkbc,baji,ck->ai', g_abab[oa, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,j||c,b>_abab*t2_bbbb(b,a,i,j)*r1_aa(c,k)
        sigma1_bb += -1.00 * einsum('kjcb,baij,ck->ai', g_abab[oa, ob, va, vb], t2_bbbb, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_bbbb*t2_bbbb(b,a,i,j)*r1_bb(c,k)
        sigma1_bb +=  1.00 * einsum('kjbc,baij,ck->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,j||b,c>_aaaa*t2_abab(c,a,k,i)*t1_aa(b,j)*r0
        sigma1_bb += -1.00 * einsum('kjbc,caki,bj,->ai', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <k,j||c,b>_abab*t2_abab(c,a,k,i)*t1_bb(b,j)*r0
        sigma1_bb +=  1.00 * einsum('kjcb,caki,bj,->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <j,k||b,c>_abab*t2_bbbb(c,a,i,k)*t1_aa(b,j)*r0
        sigma1_bb += -1.00 * einsum('jkbc,caik,bj,->ai', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_bbbb*t2_bbbb(c,a,i,k)*t1_bb(b,j)*r0
        sigma1_bb +=  1.00 * einsum('kjbc,caik,bj,->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -0.50 <k,j||c,b>_abab*t2_abab(c,a,k,j)*t1_bb(b,i)*r0
        sigma1_bb += -0.50 * einsum('kjcb,cakj,bi,->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -0.50 <j,k||c,b>_abab*t2_abab(c,a,j,k)*t1_bb(b,i)*r0
        sigma1_bb += -0.50 * einsum('jkcb,cajk,bi,->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  0.50 <k,j||b,c>_bbbb*t2_bbbb(c,a,k,j)*t1_bb(b,i)*r0
        sigma1_bb +=  0.50 * einsum('kjbc,cakj,bi,->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_abab*t1_bb(a,j)*t2_abab(b,c,k,i)*r0
        sigma1_bb += -0.50 * einsum('kjbc,aj,bcki,->ai', g_abab[oa, ob, va, vb], t1_bb, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -0.50 <k,j||c,b>_abab*t1_bb(a,j)*t2_abab(c,b,k,i)*r0
        sigma1_bb += -0.50 * einsum('kjcb,aj,cbki,->ai', g_abab[oa, ob, va, vb], t1_bb, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  0.50 <k,j||b,c>_bbbb*t1_bb(a,j)*t2_bbbb(b,c,i,k)*r0
        sigma1_bb +=  0.50 * einsum('kjbc,aj,bcik,->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -1.00 <j,k||b,i>_abab*t1_bb(a,k)*t1_aa(b,j)*r0
        sigma1_bb += -1.00 * einsum('jkbi,ak,bj,->ai', g_abab[oa, ob, va, ob], t1_bb, t1_aa, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  1.00 <k,j||b,i>_bbbb*t1_bb(a,k)*t1_bb(b,j)*r0
        sigma1_bb +=  1.00 * einsum('kjbi,ak,bj,->ai', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  1.00 <j,a||b,c>_abab*t1_aa(b,j)*t1_bb(c,i)*r0
        sigma1_bb +=  1.00 * einsum('jabc,bj,ci,->ai', g_abab[oa, vb, va, vb], t1_aa, t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	  1.00 <j,a||b,c>_bbbb*t1_bb(b,j)*t1_bb(c,i)*r0
        sigma1_bb +=  1.00 * einsum('jabc,bj,ci,->ai', g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_aaaa*t1_aa(b,j)*t1_aa(c,k)*r1_bb(a,i)
        sigma1_bb += -0.50 * einsum('kjbc,bj,ck,ai->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <j,k||b,c>_abab*t1_aa(b,j)*t1_bb(c,k)*r1_bb(a,i)
        sigma1_bb +=  0.50 * einsum('jkbc,bj,ck,ai->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <k,j||c,b>_abab*t1_bb(b,j)*t1_aa(c,k)*r1_bb(a,i)
        sigma1_bb +=  0.50 * einsum('kjcb,bj,ck,ai->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <k,j||b,c>_bbbb*t1_bb(b,j)*t1_bb(c,k)*r1_bb(a,i)
        sigma1_bb += -0.50 * einsum('kjbc,bj,ck,ai->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 <j,k||b,c>_abab*t1_aa(b,j)*t1_bb(c,i)*r1_bb(a,k)
        sigma1_bb += -1.00 * einsum('jkbc,bj,ci,ak->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_bbbb*t1_bb(b,j)*t1_bb(c,i)*r1_bb(a,k)
        sigma1_bb +=  1.00 * einsum('kjbc,bj,ci,ak->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 <j,k||b,c>_abab*t1_bb(a,k)*t1_aa(b,j)*r1_bb(c,i)
        sigma1_bb += -1.00 * einsum('jkbc,ak,bj,ci->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_bbbb*t1_bb(a,k)*t1_bb(b,j)*r1_bb(c,i)
        sigma1_bb +=  1.00 * einsum('kjbc,ak,bj,ci->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -1.00 <k,j||c,b>_abab*t1_bb(a,j)*t1_bb(b,i)*r1_aa(c,k)
        sigma1_bb += -1.00 * einsum('kjcb,aj,bi,ck->ai', g_abab[oa, ob, va, vb], t1_bb, t1_bb, r1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_bbbb*t1_bb(a,j)*t1_bb(b,i)*r1_bb(c,k)
        sigma1_bb +=  1.00 * einsum('kjbc,aj,bi,ck->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        
        #	 -1.00 <j,k||b,c>_abab*t1_bb(a,k)*t1_aa(b,j)*t1_bb(c,i)*r0
        sigma1_bb += -1.00 * einsum('jkbc,ak,bj,ci,->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, r0, optimize=['einsum_path', (0, 2), (1, 3), (1, 2), (0, 1)])
        
        #	  1.00 <k,j||b,c>_bbbb*t1_bb(a,k)*t1_bb(b,j)*t1_bb(c,i)*r0
        sigma1_bb +=  1.00 * einsum('kjbc,ak,bj,ci,->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 2), (1, 3), (1, 2), (0, 1)])
        
        return sigma1_bb
    
    
    def sigma_doubles_aaaa(self, r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab):

        t1_aa = self.t1_aa
        t1_bb = self.t1_bb
        t2_aaaa = self.t2_aaaa
        t2_bbbb = self.t2_bbbb
        t2_abab = self.t2_abab
        f_aa = self.f_aa
        f_bb = self.f_bb
        g_aaaa = self.g_aaaa
        g_bbbb = self.g_bbbb
        g_abab = self.g_abab
        oa = self.oa
        va = self.va
        ob = self.ob
        vb = self.vb
    
        #	 -1.00 P(i,j)*P(a,b)f_aa(a,j)*r1_aa(b,i)
        contracted_intermediate = -1.00 * einsum('aj,bi->abij', f_aa[va, oa], r1_aa)
        sigma2_aaaa =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 f_aa(k,k)*r2_aaaa(a,b,i,j)
        sigma2_aaaa +=  1.00 * einsum('kk,abij->abij', f_aa[oa, oa], r2_aaaa)
        
        #	  1.00 f_bb(k,k)*r2_aaaa(a,b,i,j)
        sigma2_aaaa +=  1.00 * einsum('kk,abij->abij', f_bb[ob, ob], r2_aaaa)
        
        #	 -1.00 P(i,j)f_aa(k,j)*r2_aaaa(a,b,i,k)
        contracted_intermediate = -1.00 * einsum('kj,abik->abij', f_aa[oa, oa], r2_aaaa)
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(a,b)f_aa(a,c)*r2_aaaa(c,b,i,j)
        contracted_intermediate =  1.00 * einsum('ac,cbij->abij', f_aa[va, va], r2_aaaa)
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)f_aa(k,j)*t1_aa(a,k)*r1_aa(b,i)
        contracted_intermediate =  1.00 * einsum('kj,ak,bi->abij', f_aa[oa, oa], t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)f_aa(a,c)*t1_aa(c,j)*r1_aa(b,i)
        contracted_intermediate = -1.00 * einsum('ac,cj,bi->abij', f_aa[va, va], t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 f_aa(k,c)*r2_aaaa(a,b,i,j)*t1_aa(c,k)
        sigma2_aaaa +=  1.00 * einsum('kc,abij,ck->abij', f_aa[oa, va], r2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 f_bb(k,c)*r2_aaaa(a,b,i,j)*t1_bb(c,k)
        sigma2_aaaa +=  1.00 * einsum('kc,abij,ck->abij', f_bb[ob, vb], r2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 P(i,j)f_aa(k,c)*r2_aaaa(a,b,i,k)*t1_aa(c,j)
        contracted_intermediate = -1.00 * einsum('kc,abik,cj->abij', f_aa[oa, va], r2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)f_aa(k,c)*r2_aaaa(c,b,i,j)*t1_aa(a,k)
        contracted_intermediate = -1.00 * einsum('kc,cbij,ak->abij', f_aa[oa, va], r2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)f_aa(k,j)*t2_aaaa(a,b,i,k)*r0
        contracted_intermediate = -1.00 * einsum('kj,abik,->abij', f_aa[oa, oa], t2_aaaa, r0, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(a,b)f_aa(a,c)*t2_aaaa(c,b,i,j)*r0
        contracted_intermediate =  1.00 * einsum('ac,cbij,->abij', f_aa[va, va], t2_aaaa, r0, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)f_aa(k,c)*r1_aa(b,i)*t2_aaaa(c,a,j,k)
        contracted_intermediate =  1.00 * einsum('kc,bi,cajk->abij', f_aa[oa, va], r1_aa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)f_bb(k,c)*r1_aa(b,i)*t2_abab(a,c,j,k)
        contracted_intermediate = -1.00 * einsum('kc,bi,acjk->abij', f_bb[ob, vb], r1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(a,b)f_aa(k,c)*r1_aa(b,k)*t2_aaaa(c,a,i,j)
        contracted_intermediate =  1.00 * einsum('kc,bk,caij->abij', f_aa[oa, va], r1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)f_aa(k,c)*r1_aa(c,i)*t2_aaaa(a,b,j,k)
        contracted_intermediate =  1.00 * einsum('kc,ci,abjk->abij', f_aa[oa, va], r1_aa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)f_aa(k,c)*t2_aaaa(a,b,i,k)*t1_aa(c,j)*r0
        contracted_intermediate = -1.00 * einsum('kc,abik,cj,->abij', f_aa[oa, va], t2_aaaa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)f_aa(k,c)*t1_aa(a,k)*t2_aaaa(c,b,i,j)*r0
        contracted_intermediate = -1.00 * einsum('kc,ak,cbij,->abij', f_aa[oa, va], t1_aa, t2_aaaa, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)f_aa(k,c)*r1_aa(b,i)*t1_aa(a,k)*t1_aa(c,j)
        contracted_intermediate =  1.00 * einsum('kc,bi,ak,cj->abij', f_aa[oa, va], r1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 <l,k||l,k>_aaaa*r2_aaaa(a,b,i,j)
        sigma2_aaaa += -0.50 * einsum('lklk,abij->abij', g_aaaa[oa, oa, oa, oa], r2_aaaa)
        
        #	 -0.50 <l,k||l,k>_abab*r2_aaaa(a,b,i,j)
        sigma2_aaaa += -0.50 * einsum('lklk,abij->abij', g_abab[oa, ob, oa, ob], r2_aaaa)
        
        #	 -0.50 <k,l||k,l>_abab*r2_aaaa(a,b,i,j)
        sigma2_aaaa += -0.50 * einsum('klkl,abij->abij', g_abab[oa, ob, oa, ob], r2_aaaa)
        
        #	 -0.50 <l,k||l,k>_bbbb*r2_aaaa(a,b,i,j)
        sigma2_aaaa += -0.50 * einsum('lklk,abij->abij', g_bbbb[ob, ob, ob, ob], r2_aaaa)
        
        #	  1.00 <a,b||i,j>_aaaa*r0
        sigma2_aaaa +=  1.00 * einsum('abij,->abij', g_aaaa[va, va, oa, oa], r0)
        
        #	  1.00 P(a,b)<k,a||i,j>_aaaa*r1_aa(b,k)
        contracted_intermediate =  1.00 * einsum('kaij,bk->abij', g_aaaa[oa, va, oa, oa], r1_aa)
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)<a,b||c,j>_aaaa*r1_aa(c,i)
        contracted_intermediate =  1.00 * einsum('abcj,ci->abij', g_aaaa[va, va, va, oa], r1_aa)
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 <l,k||i,j>_aaaa*r2_aaaa(a,b,l,k)
        sigma2_aaaa +=  0.50 * einsum('lkij,ablk->abij', g_aaaa[oa, oa, oa, oa], r2_aaaa)
        
        #	  1.00 P(i,j)*P(a,b)<k,a||c,j>_aaaa*r2_aaaa(c,b,i,k)
        contracted_intermediate =  1.00 * einsum('kacj,cbik->abij', g_aaaa[oa, va, va, oa], r2_aaaa)
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<a,k||j,c>_abab*r2_abab(b,c,i,k)
        contracted_intermediate = -1.00 * einsum('akjc,bcik->abij', g_abab[va, ob, oa, vb], r2_abab)
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 <a,b||c,d>_aaaa*r2_aaaa(c,d,i,j)
        sigma2_aaaa +=  0.50 * einsum('abcd,cdij->abij', g_aaaa[va, va, va, va], r2_aaaa)
        
        #	  1.00 P(a,b)<k,a||i,j>_aaaa*t1_aa(b,k)*r0
        contracted_intermediate =  1.00 * einsum('kaij,bk,->abij', g_aaaa[oa, va, oa, oa], t1_aa, r0, optimize=['einsum_path', (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)<a,b||c,j>_aaaa*t1_aa(c,i)*r0
        contracted_intermediate =  1.00 * einsum('abcj,ci,->abij', g_aaaa[va, va, va, oa], t1_aa, r0, optimize=['einsum_path', (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<l,k||i,j>_aaaa*r1_aa(b,l)*t1_aa(a,k)
        contracted_intermediate = -1.00 * einsum('lkij,bl,ak->abij', g_aaaa[oa, oa, oa, oa], r1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,j>_aaaa*r1_aa(b,i)*t1_aa(c,k)
        contracted_intermediate = -1.00 * einsum('kacj,bi,ck->abij', g_aaaa[oa, va, va, oa], r1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<a,k||j,c>_abab*r1_aa(b,i)*t1_bb(c,k)
        contracted_intermediate = -1.00 * einsum('akjc,bi,ck->abij', g_abab[va, ob, oa, vb], r1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,a||c,j>_aaaa*r1_aa(b,k)*t1_aa(c,i)
        contracted_intermediate =  1.00 * einsum('kacj,bk,ci->abij', g_aaaa[oa, va, va, oa], r1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,a||c,j>_aaaa*r1_aa(c,i)*t1_aa(b,k)
        contracted_intermediate =  1.00 * einsum('kacj,ci,bk->abij', g_aaaa[oa, va, va, oa], r1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<a,b||c,d>_aaaa*r1_aa(d,i)*t1_aa(c,j)
        contracted_intermediate = -1.00 * einsum('abcd,di,cj->abij', g_aaaa[va, va, va, va], r1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,j>_aaaa*t1_aa(c,k)*r2_aaaa(a,b,i,l)
        contracted_intermediate =  1.00 * einsum('lkcj,ck,abil->abij', g_aaaa[oa, oa, va, oa], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||j,c>_abab*t1_bb(c,k)*r2_aaaa(a,b,i,l)
        contracted_intermediate = -1.00 * einsum('lkjc,ck,abil->abij', g_abab[oa, ob, oa, vb], t1_bb, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 P(i,j)<l,k||c,j>_aaaa*t1_aa(c,i)*r2_aaaa(a,b,l,k)
        contracted_intermediate =  0.50 * einsum('lkcj,ci,ablk->abij', g_aaaa[oa, oa, va, oa], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>_aaaa*t1_aa(a,k)*r2_aaaa(c,b,i,l)
        contracted_intermediate = -1.00 * einsum('lkcj,ak,cbil->abij', g_aaaa[oa, oa, va, oa], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,l||j,c>_abab*t1_aa(a,k)*r2_abab(b,c,i,l)
        contracted_intermediate =  1.00 * einsum('kljc,ak,bcil->abij', g_abab[oa, ob, oa, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(a,b)<k,a||c,d>_aaaa*t1_aa(c,k)*r2_aaaa(d,b,i,j)
        contracted_intermediate =  1.00 * einsum('kacd,ck,dbij->abij', g_aaaa[oa, va, va, va], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(a,b)<a,k||d,c>_abab*t1_bb(c,k)*r2_aaaa(d,b,i,j)
        contracted_intermediate =  1.00 * einsum('akdc,ck,dbij->abij', g_abab[va, ob, va, vb], t1_bb, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>_aaaa*t1_aa(c,j)*r2_aaaa(d,b,i,k)
        contracted_intermediate = -1.00 * einsum('kacd,cj,dbik->abij', g_aaaa[oa, va, va, va], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<a,k||c,d>_abab*t1_aa(c,j)*r2_abab(b,d,i,k)
        contracted_intermediate = -1.00 * einsum('akcd,cj,bdik->abij', g_abab[va, ob, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(a,b)<k,a||c,d>_aaaa*t1_aa(b,k)*r2_aaaa(c,d,i,j)
        contracted_intermediate =  0.50 * einsum('kacd,bk,cdij->abij', g_aaaa[oa, va, va, va], t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  0.50 <l,k||i,j>_aaaa*t2_aaaa(a,b,l,k)*r0
        sigma2_aaaa +=  0.50 * einsum('lkij,ablk,->abij', g_aaaa[oa, oa, oa, oa], t2_aaaa, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 P(i,j)*P(a,b)<k,a||c,j>_aaaa*t2_aaaa(c,b,i,k)*r0
        contracted_intermediate =  1.00 * einsum('kacj,cbik,->abij', g_aaaa[oa, va, va, oa], t2_aaaa, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<a,k||j,c>_abab*t2_abab(b,c,i,k)*r0
        contracted_intermediate = -1.00 * einsum('akjc,bcik,->abij', g_abab[va, ob, oa, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 <a,b||c,d>_aaaa*t2_aaaa(c,d,i,j)*r0
        sigma2_aaaa +=  0.50 * einsum('abcd,cdij,->abij', g_aaaa[va, va, va, va], t2_aaaa, r0, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 P(i,j)*P(a,b)<l,k||c,j>_aaaa*t2_aaaa(c,a,l,k)*r1_aa(b,i)
        contracted_intermediate =  0.50 * einsum('lkcj,calk,bi->abij', g_aaaa[oa, oa, va, oa], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<l,k||j,c>_abab*t2_abab(a,c,l,k)*r1_aa(b,i)
        contracted_intermediate =  0.50 * einsum('lkjc,aclk,bi->abij', g_abab[oa, ob, oa, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<k,l||j,c>_abab*t2_abab(a,c,k,l)*r1_aa(b,i)
        contracted_intermediate =  0.50 * einsum('kljc,ackl,bi->abij', g_abab[oa, ob, oa, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>_aaaa*t2_aaaa(c,a,i,k)*r1_aa(b,l)
        contracted_intermediate = -1.00 * einsum('lkcj,caik,bl->abij', g_aaaa[oa, oa, va, oa], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||j,c>_abab*t2_abab(a,c,i,k)*r1_aa(b,l)
        contracted_intermediate = -1.00 * einsum('lkjc,acik,bl->abij', g_abab[oa, ob, oa, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)<l,k||c,j>_aaaa*t2_aaaa(a,b,l,k)*r1_aa(c,i)
        contracted_intermediate =  0.50 * einsum('lkcj,ablk,ci->abij', g_aaaa[oa, oa, va, oa], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||c,j>_aaaa*t2_aaaa(a,b,i,k)*r1_aa(c,l)
        contracted_intermediate = -1.00 * einsum('lkcj,abik,cl->abij', g_aaaa[oa, oa, va, oa], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<k,l||j,c>_abab*t2_aaaa(a,b,i,k)*r1_bb(c,l)
        contracted_intermediate = -1.00 * einsum('kljc,abik,cl->abij', g_abab[oa, ob, oa, vb], t2_aaaa, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<k,a||c,d>_aaaa*t2_aaaa(c,d,j,k)*r1_aa(b,i)
        contracted_intermediate =  0.50 * einsum('kacd,cdjk,bi->abij', g_aaaa[oa, va, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)*P(a,b)<a,k||c,d>_abab*t2_abab(c,d,j,k)*r1_aa(b,i)
        contracted_intermediate = -0.50 * einsum('akcd,cdjk,bi->abij', g_abab[va, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)*P(a,b)<a,k||d,c>_abab*t2_abab(d,c,j,k)*r1_aa(b,i)
        contracted_intermediate = -0.50 * einsum('akdc,dcjk,bi->abij', g_abab[va, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(a,b)<k,a||c,d>_aaaa*t2_aaaa(c,d,i,j)*r1_aa(b,k)
        contracted_intermediate =  0.50 * einsum('kacd,cdij,bk->abij', g_aaaa[oa, va, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>_aaaa*t2_aaaa(c,b,j,k)*r1_aa(d,i)
        contracted_intermediate = -1.00 * einsum('kacd,cbjk,di->abij', g_aaaa[oa, va, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<a,k||d,c>_abab*t2_abab(b,c,j,k)*r1_aa(d,i)
        contracted_intermediate =  1.00 * einsum('akdc,bcjk,di->abij', g_abab[va, ob, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<k,a||c,d>_aaaa*t2_aaaa(c,b,i,j)*r1_aa(d,k)
        contracted_intermediate = -1.00 * einsum('kacd,cbij,dk->abij', g_aaaa[oa, va, va, va], t2_aaaa, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(a,b)<a,k||c,d>_abab*t2_aaaa(c,b,i,j)*r1_bb(d,k)
        contracted_intermediate =  1.00 * einsum('akcd,cbij,dk->abij', g_abab[va, ob, va, vb], t2_aaaa, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  0.250 <l,k||c,d>_aaaa*t2_aaaa(c,d,l,k)*r2_aaaa(a,b,i,j)
        sigma2_aaaa +=  0.250 * einsum('lkcd,cdlk,abij->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||c,d>_abab*t2_abab(c,d,l,k)*r2_aaaa(a,b,i,j)
        sigma2_aaaa +=  0.250 * einsum('lkcd,cdlk,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,l||c,d>_abab*t2_abab(c,d,k,l)*r2_aaaa(a,b,i,j)
        sigma2_aaaa +=  0.250 * einsum('klcd,cdkl,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||d,c>_abab*t2_abab(d,c,l,k)*r2_aaaa(a,b,i,j)
        sigma2_aaaa +=  0.250 * einsum('lkdc,dclk,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,l||d,c>_abab*t2_abab(d,c,k,l)*r2_aaaa(a,b,i,j)
        sigma2_aaaa +=  0.250 * einsum('kldc,dckl,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||c,d>_bbbb*t2_bbbb(c,d,l,k)*r2_aaaa(a,b,i,j)
        sigma2_aaaa +=  0.250 * einsum('lkcd,cdlk,abij->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(c,d,j,k)*r2_aaaa(a,b,i,l)
        contracted_intermediate = -0.50 * einsum('lkcd,cdjk,abil->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)<l,k||c,d>_abab*t2_abab(c,d,j,k)*r2_aaaa(a,b,i,l)
        contracted_intermediate = -0.50 * einsum('lkcd,cdjk,abil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)<l,k||d,c>_abab*t2_abab(d,c,j,k)*r2_aaaa(a,b,i,l)
        contracted_intermediate = -0.50 * einsum('lkdc,dcjk,abil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.250 <l,k||c,d>_aaaa*t2_aaaa(c,d,i,j)*r2_aaaa(a,b,l,k)
        sigma2_aaaa +=  0.250 * einsum('lkcd,cdij,ablk->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 P(a,b)<l,k||c,d>_aaaa*t2_aaaa(c,a,l,k)*r2_aaaa(d,b,i,j)
        contracted_intermediate = -0.50 * einsum('lkcd,calk,dbij->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -0.50 P(a,b)<l,k||d,c>_abab*t2_abab(a,c,l,k)*r2_aaaa(d,b,i,j)
        contracted_intermediate = -0.50 * einsum('lkdc,aclk,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -0.50 P(a,b)<k,l||d,c>_abab*t2_abab(a,c,k,l)*r2_aaaa(d,b,i,j)
        contracted_intermediate = -0.50 * einsum('kldc,ackl,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t2_aaaa(c,a,j,k)*r2_aaaa(d,b,i,l)
        contracted_intermediate =  1.00 * einsum('lkcd,cajk,dbil->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,l||c,d>_abab*t2_aaaa(c,a,j,k)*r2_abab(b,d,i,l)
        contracted_intermediate =  1.00 * einsum('klcd,cajk,bdil->abij', g_abab[oa, ob, va, vb], t2_aaaa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||d,c>_abab*t2_abab(a,c,j,k)*r2_aaaa(d,b,i,l)
        contracted_intermediate =  1.00 * einsum('lkdc,acjk,dbil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t2_abab(a,c,j,k)*r2_abab(b,d,i,l)
        contracted_intermediate =  1.00 * einsum('lkcd,acjk,bdil->abij', g_bbbb[ob, ob, vb, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 P(a,b)<l,k||c,d>_aaaa*t2_aaaa(c,a,i,j)*r2_aaaa(d,b,l,k)
        contracted_intermediate = -0.50 * einsum('lkcd,caij,dblk->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  0.50 P(a,b)<l,k||c,d>_abab*t2_aaaa(c,a,i,j)*r2_abab(b,d,l,k)
        contracted_intermediate =  0.50 * einsum('lkcd,caij,bdlk->abij', g_abab[oa, ob, va, vb], t2_aaaa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  0.50 P(a,b)<k,l||c,d>_abab*t2_aaaa(c,a,i,j)*r2_abab(b,d,k,l)
        contracted_intermediate =  0.50 * einsum('klcd,caij,bdkl->abij', g_abab[oa, ob, va, vb], t2_aaaa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  0.250 <l,k||c,d>_aaaa*t2_aaaa(a,b,l,k)*r2_aaaa(c,d,i,j)
        sigma2_aaaa +=  0.250 * einsum('lkcd,ablk,cdij->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(a,b,j,k)*r2_aaaa(c,d,i,l)
        contracted_intermediate = -0.50 * einsum('lkcd,abjk,cdil->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 P(i,j)<k,l||c,d>_abab*t2_aaaa(a,b,j,k)*r2_abab(c,d,i,l)
        contracted_intermediate =  0.50 * einsum('klcd,abjk,cdil->abij', g_abab[oa, ob, va, vb], t2_aaaa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 P(i,j)<k,l||d,c>_abab*t2_aaaa(a,b,j,k)*r2_abab(d,c,i,l)
        contracted_intermediate =  0.50 * einsum('kldc,abjk,dcil->abij', g_abab[oa, ob, va, vb], t2_aaaa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,j>_aaaa*t2_aaaa(a,b,i,l)*t1_aa(c,k)*r0
        contracted_intermediate =  1.00 * einsum('lkcj,abil,ck,->abij', g_aaaa[oa, oa, va, oa], t2_aaaa, t1_aa, r0, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||j,c>_abab*t2_aaaa(a,b,i,l)*t1_bb(c,k)*r0
        contracted_intermediate = -1.00 * einsum('lkjc,abil,ck,->abij', g_abab[oa, ob, oa, vb], t2_aaaa, t1_bb, r0, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 P(i,j)<l,k||c,j>_aaaa*t2_aaaa(a,b,l,k)*t1_aa(c,i)*r0
        contracted_intermediate =  0.50 * einsum('lkcj,ablk,ci,->abij', g_aaaa[oa, oa, va, oa], t2_aaaa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>_aaaa*t1_aa(a,k)*t2_aaaa(c,b,i,l)*r0
        contracted_intermediate = -1.00 * einsum('lkcj,ak,cbil,->abij', g_aaaa[oa, oa, va, oa], t1_aa, t2_aaaa, r0, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,l||j,c>_abab*t1_aa(a,k)*t2_abab(b,c,i,l)*r0
        contracted_intermediate =  1.00 * einsum('kljc,ak,bcil,->abij', g_abab[oa, ob, oa, vb], t1_aa, t2_abab, r0, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(a,b)<k,a||c,d>_aaaa*t2_aaaa(d,b,i,j)*t1_aa(c,k)*r0
        contracted_intermediate =  1.00 * einsum('kacd,dbij,ck,->abij', g_aaaa[oa, va, va, va], t2_aaaa, t1_aa, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(a,b)<a,k||d,c>_abab*t2_aaaa(d,b,i,j)*t1_bb(c,k)*r0
        contracted_intermediate =  1.00 * einsum('akdc,dbij,ck,->abij', g_abab[va, ob, va, vb], t2_aaaa, t1_bb, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>_aaaa*t2_aaaa(d,b,i,k)*t1_aa(c,j)*r0
        contracted_intermediate = -1.00 * einsum('kacd,dbik,cj,->abij', g_aaaa[oa, va, va, va], t2_aaaa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<a,k||c,d>_abab*t2_abab(b,d,i,k)*t1_aa(c,j)*r0
        contracted_intermediate = -1.00 * einsum('akcd,bdik,cj,->abij', g_abab[va, ob, va, vb], t2_abab, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(a,b)<k,a||c,d>_aaaa*t1_aa(b,k)*t2_aaaa(c,d,i,j)*r0
        contracted_intermediate =  0.50 * einsum('kacd,bk,cdij,->abij', g_aaaa[oa, va, va, va], t1_aa, t2_aaaa, r0, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t2_aaaa(d,a,j,l)*t1_aa(c,k)*r1_aa(b,i)
        contracted_intermediate = -1.00 * einsum('lkcd,dajl,ck,bi->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||d,c>_abab*t2_aaaa(d,a,j,l)*t1_bb(c,k)*r1_aa(b,i)
        contracted_intermediate =  1.00 * einsum('lkdc,dajl,ck,bi->abij', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,l||c,d>_abab*t2_abab(a,d,j,l)*t1_aa(c,k)*r1_aa(b,i)
        contracted_intermediate = -1.00 * einsum('klcd,adjl,ck,bi->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t2_abab(a,d,j,l)*t1_bb(c,k)*r1_aa(b,i)
        contracted_intermediate =  1.00 * einsum('lkcd,adjl,ck,bi->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<l,k||c,d>_aaaa*t2_aaaa(d,a,i,j)*t1_aa(c,k)*r1_aa(b,l)
        contracted_intermediate = -1.00 * einsum('lkcd,daij,ck,bl->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(a,b)<l,k||d,c>_abab*t2_aaaa(d,a,i,j)*t1_bb(c,k)*r1_aa(b,l)
        contracted_intermediate =  1.00 * einsum('lkdc,daij,ck,bl->abij', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(a,b,j,l)*t1_aa(c,k)*r1_aa(d,i)
        contracted_intermediate = -1.00 * einsum('lkcd,abjl,ck,di->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||d,c>_abab*t2_aaaa(a,b,j,l)*t1_bb(c,k)*r1_aa(d,i)
        contracted_intermediate =  1.00 * einsum('lkdc,abjl,ck,di->abij', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t2_aaaa(d,a,l,k)*t1_aa(c,j)*r1_aa(b,i)
        contracted_intermediate = -0.50 * einsum('lkcd,dalk,cj,bi->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<l,k||c,d>_abab*t2_abab(a,d,l,k)*t1_aa(c,j)*r1_aa(b,i)
        contracted_intermediate =  0.50 * einsum('lkcd,adlk,cj,bi->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<k,l||c,d>_abab*t2_abab(a,d,k,l)*t1_aa(c,j)*r1_aa(b,i)
        contracted_intermediate =  0.50 * einsum('klcd,adkl,cj,bi->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t2_aaaa(d,a,i,k)*t1_aa(c,j)*r1_aa(b,l)
        contracted_intermediate =  1.00 * einsum('lkcd,daik,cj,bl->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, r1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,d>_abab*t2_abab(a,d,i,k)*t1_aa(c,j)*r1_aa(b,l)
        contracted_intermediate = -1.00 * einsum('lkcd,adik,cj,bl->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(a,b,l,k)*t1_aa(c,j)*r1_aa(d,i)
        contracted_intermediate = -0.50 * einsum('lkcd,ablk,cj,di->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(a,b,i,k)*t1_aa(c,j)*r1_aa(d,l)
        contracted_intermediate =  1.00 * einsum('lkcd,abik,cj,dl->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, r1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<k,l||c,d>_abab*t2_aaaa(a,b,i,k)*t1_aa(c,j)*r1_bb(d,l)
        contracted_intermediate = -1.00 * einsum('klcd,abik,cj,dl->abij', g_abab[oa, ob, va, vb], t2_aaaa, t1_aa, r1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t1_aa(a,k)*t2_aaaa(c,d,j,l)*r1_aa(b,i)
        contracted_intermediate = -0.50 * einsum('lkcd,ak,cdjl,bi->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<k,l||c,d>_abab*t1_aa(a,k)*t2_abab(c,d,j,l)*r1_aa(b,i)
        contracted_intermediate =  0.50 * einsum('klcd,ak,cdjl,bi->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<k,l||d,c>_abab*t1_aa(a,k)*t2_abab(d,c,j,l)*r1_aa(b,i)
        contracted_intermediate =  0.50 * einsum('kldc,ak,dcjl,bi->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 P(a,b)<l,k||c,d>_aaaa*t1_aa(a,k)*t2_aaaa(c,d,i,j)*r1_aa(b,l)
        contracted_intermediate = -0.50 * einsum('lkcd,ak,cdij,bl->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t1_aa(a,k)*t2_aaaa(c,b,j,l)*r1_aa(d,i)
        contracted_intermediate =  1.00 * einsum('lkcd,ak,cbjl,di->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,l||d,c>_abab*t1_aa(a,k)*t2_abab(b,c,j,l)*r1_aa(d,i)
        contracted_intermediate = -1.00 * einsum('kldc,ak,bcjl,di->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(a,b)<l,k||c,d>_aaaa*t1_aa(a,k)*t2_aaaa(c,b,i,j)*r1_aa(d,l)
        contracted_intermediate =  1.00 * einsum('lkcd,ak,cbij,dl->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, r1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<k,l||c,d>_abab*t1_aa(a,k)*t2_aaaa(c,b,i,j)*r1_bb(d,l)
        contracted_intermediate = -1.00 * einsum('klcd,ak,cbij,dl->abij', g_abab[oa, ob, va, vb], t1_aa, t2_aaaa, r1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 <l,k||i,j>_aaaa*t1_aa(a,k)*t1_aa(b,l)*r0
        sigma2_aaaa += -1.00 * einsum('lkij,ak,bl,->abij', g_aaaa[oa, oa, oa, oa], t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	  1.00 P(i,j)*P(a,b)<k,a||c,j>_aaaa*t1_aa(b,k)*t1_aa(c,i)*r0
        contracted_intermediate =  1.00 * einsum('kacj,bk,ci,->abij', g_aaaa[oa, va, va, oa], t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 <a,b||c,d>_aaaa*t1_aa(c,j)*t1_aa(d,i)*r0
        sigma2_aaaa += -1.00 * einsum('abcd,cj,di,->abij', g_aaaa[va, va, va, va], t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>_aaaa*t1_aa(a,l)*t1_aa(c,k)*r1_aa(b,i)
        contracted_intermediate = -1.00 * einsum('lkcj,al,ck,bi->abij', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||j,c>_abab*t1_aa(a,l)*t1_bb(c,k)*r1_aa(b,i)
        contracted_intermediate =  1.00 * einsum('lkjc,al,ck,bi->abij', g_abab[oa, ob, oa, vb], t1_aa, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>_aaaa*t1_aa(a,k)*t1_aa(c,i)*r1_aa(b,l)
        contracted_intermediate = -1.00 * einsum('lkcj,ak,ci,bl->abij', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||c,j>_aaaa*t1_aa(a,k)*t1_aa(b,l)*r1_aa(c,i)
        contracted_intermediate = -1.00 * einsum('lkcj,ak,bl,ci->abij', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>_aaaa*t1_aa(c,k)*t1_aa(d,j)*r1_aa(b,i)
        contracted_intermediate = -1.00 * einsum('kacd,ck,dj,bi->abij', g_aaaa[oa, va, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<a,k||d,c>_abab*t1_bb(c,k)*t1_aa(d,j)*r1_aa(b,i)
        contracted_intermediate = -1.00 * einsum('akdc,ck,dj,bi->abij', g_abab[va, ob, va, vb], t1_bb, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<k,a||c,d>_aaaa*t1_aa(c,j)*t1_aa(d,i)*r1_aa(b,k)
        contracted_intermediate = -1.00 * einsum('kacd,cj,di,bk->abij', g_aaaa[oa, va, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>_aaaa*t1_aa(b,k)*t1_aa(c,j)*r1_aa(d,i)
        contracted_intermediate = -1.00 * einsum('kacd,bk,cj,di->abij', g_aaaa[oa, va, va, va], t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 <l,k||c,d>_aaaa*t1_aa(c,k)*t1_aa(d,l)*r2_aaaa(a,b,i,j)
        sigma2_aaaa += -0.50 * einsum('lkcd,ck,dl,abij->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <k,l||c,d>_abab*t1_aa(c,k)*t1_bb(d,l)*r2_aaaa(a,b,i,j)
        sigma2_aaaa +=  0.50 * einsum('klcd,ck,dl,abij->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <l,k||d,c>_abab*t1_bb(c,k)*t1_aa(d,l)*r2_aaaa(a,b,i,j)
        sigma2_aaaa +=  0.50 * einsum('lkdc,ck,dl,abij->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_bbbb*t1_bb(c,k)*t1_bb(d,l)*r2_aaaa(a,b,i,j)
        sigma2_aaaa += -0.50 * einsum('lkcd,ck,dl,abij->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 P(i,j)<l,k||c,d>_aaaa*t1_aa(c,k)*t1_aa(d,j)*r2_aaaa(a,b,i,l)
        contracted_intermediate =  1.00 * einsum('lkcd,ck,dj,abil->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||d,c>_abab*t1_bb(c,k)*t1_aa(d,j)*r2_aaaa(a,b,i,l)
        contracted_intermediate = -1.00 * einsum('lkdc,ck,dj,abil->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(a,b)<l,k||c,d>_aaaa*t1_aa(a,l)*t1_aa(c,k)*r2_aaaa(d,b,i,j)
        contracted_intermediate =  1.00 * einsum('lkcd,al,ck,dbij->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<l,k||d,c>_abab*t1_aa(a,l)*t1_bb(c,k)*r2_aaaa(d,b,i,j)
        contracted_intermediate = -1.00 * einsum('lkdc,al,ck,dbij->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -0.50 <l,k||c,d>_aaaa*t1_aa(c,j)*t1_aa(d,i)*r2_aaaa(a,b,l,k)
        sigma2_aaaa += -0.50 * einsum('lkcd,cj,di,ablk->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t1_aa(a,k)*t1_aa(c,j)*r2_aaaa(d,b,i,l)
        contracted_intermediate =  1.00 * einsum('lkcd,ak,cj,dbil->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,l||c,d>_abab*t1_aa(a,k)*t1_aa(c,j)*r2_abab(b,d,i,l)
        contracted_intermediate =  1.00 * einsum('klcd,ak,cj,bdil->abij', g_abab[oa, ob, va, vb], t1_aa, t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 <l,k||c,d>_aaaa*t1_aa(a,k)*t1_aa(b,l)*r2_aaaa(c,d,i,j)
        sigma2_aaaa += -0.50 * einsum('lkcd,ak,bl,cdij->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(a,b,i,l)*t2_aaaa(c,d,j,k)*r0
        contracted_intermediate = -0.50 * einsum('lkcd,abil,cdjk,->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)<l,k||c,d>_abab*t2_aaaa(a,b,i,l)*t2_abab(c,d,j,k)*r0
        contracted_intermediate = -0.50 * einsum('lkcd,abil,cdjk,->abij', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)<l,k||d,c>_abab*t2_aaaa(a,b,i,l)*t2_abab(d,c,j,k)*r0
        contracted_intermediate = -0.50 * einsum('lkdc,abil,dcjk,->abij', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.250 <l,k||c,d>_aaaa*t2_aaaa(a,b,l,k)*t2_aaaa(c,d,i,j)*r0
        sigma2_aaaa +=  0.250 * einsum('lkcd,ablk,cdij,->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_aaaa*t2_aaaa(c,a,l,k)*t2_aaaa(d,b,i,j)*r0
        sigma2_aaaa += -0.50 * einsum('lkcd,calk,dbij,->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <l,k||d,c>_abab*t2_abab(a,c,l,k)*t2_aaaa(d,b,i,j)*r0
        sigma2_aaaa += -0.50 * einsum('lkdc,aclk,dbij,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <k,l||d,c>_abab*t2_abab(a,c,k,l)*t2_aaaa(d,b,i,j)*r0
        sigma2_aaaa += -0.50 * einsum('kldc,ackl,dbij,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(c,a,j,k)*t2_aaaa(d,b,i,l)*r0
        contracted_intermediate =  1.00 * einsum('lkcd,cajk,dbil,->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<k,l||c,d>_abab*t2_aaaa(c,a,j,k)*t2_abab(b,d,i,l)*r0
        contracted_intermediate =  1.00 * einsum('klcd,cajk,bdil,->abij', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||d,c>_abab*t2_abab(a,c,j,k)*t2_aaaa(d,b,i,l)*r0
        contracted_intermediate =  1.00 * einsum('lkdc,acjk,dbil,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,d>_bbbb*t2_abab(a,c,j,k)*t2_abab(b,d,i,l)*r0
        contracted_intermediate =  1.00 * einsum('lkcd,acjk,bdil,->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 <l,k||c,d>_aaaa*t2_aaaa(c,a,i,j)*t2_aaaa(d,b,l,k)*r0
        sigma2_aaaa += -0.50 * einsum('lkcd,caij,dblk,->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  0.50 <l,k||c,d>_abab*t2_aaaa(c,a,i,j)*t2_abab(b,d,l,k)*r0
        sigma2_aaaa +=  0.50 * einsum('lkcd,caij,bdlk,->abij', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  0.50 <k,l||c,d>_abab*t2_aaaa(c,a,i,j)*t2_abab(b,d,k,l)*r0
        sigma2_aaaa +=  0.50 * einsum('klcd,caij,bdkl,->abij', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  1.00 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(a,b,i,l)*t1_aa(c,k)*t1_aa(d,j)*r0
        contracted_intermediate =  1.00 * einsum('lkcd,abil,ck,dj,->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 3), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||d,c>_abab*t2_aaaa(a,b,i,l)*t1_bb(c,k)*t1_aa(d,j)*r0
        contracted_intermediate = -1.00 * einsum('lkdc,abil,ck,dj,->abij', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 3), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(a,b)<l,k||c,d>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t1_aa(c,k)*r0
        contracted_intermediate =  1.00 * einsum('lkcd,al,dbij,ck,->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t1_aa, r0, optimize=['einsum_path', (0, 3), (0, 3), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<l,k||d,c>_abab*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t1_bb(c,k)*r0
        contracted_intermediate = -1.00 * einsum('lkdc,al,dbij,ck,->abij', g_abab[oa, ob, va, vb], t1_aa, t2_aaaa, t1_bb, r0, optimize=['einsum_path', (0, 3), (0, 3), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -0.50 <l,k||c,d>_aaaa*t2_aaaa(a,b,l,k)*t1_aa(c,j)*t1_aa(d,i)*r0
        sigma2_aaaa += -0.50 * einsum('lkcd,ablk,cj,di,->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t1_aa(a,k)*t2_aaaa(d,b,i,l)*t1_aa(c,j)*r0
        contracted_intermediate =  1.00 * einsum('lkcd,ak,dbil,cj,->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,l||c,d>_abab*t1_aa(a,k)*t2_abab(b,d,i,l)*t1_aa(c,j)*r0
        contracted_intermediate =  1.00 * einsum('klcd,ak,bdil,cj,->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 <l,k||c,d>_aaaa*t1_aa(a,k)*t1_aa(b,l)*t2_aaaa(c,d,i,j)*r0
        sigma2_aaaa += -0.50 * einsum('lkcd,ak,bl,cdij,->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t2_aaaa, r0, optimize=['einsum_path', (0, 1), (0, 2), (1, 2), (0, 1)])
        
        #	 -1.00 P(i,j)<l,k||c,j>_aaaa*t1_aa(a,k)*t1_aa(b,l)*t1_aa(c,i)*r0
        contracted_intermediate = -1.00 * einsum('lkcj,ak,bl,ci,->abij', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 2), (1, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<k,a||c,d>_aaaa*t1_aa(b,k)*t1_aa(c,j)*t1_aa(d,i)*r0
        contracted_intermediate = -1.00 * einsum('kacd,bk,cj,di,->abij', g_aaaa[oa, va, va, va], t1_aa, t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t1_aa(a,l)*t1_aa(c,k)*t1_aa(d,j)*r1_aa(b,i)
        contracted_intermediate = -1.00 * einsum('lkcd,al,ck,dj,bi->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||d,c>_abab*t1_aa(a,l)*t1_bb(c,k)*t1_aa(d,j)*r1_aa(b,i)
        contracted_intermediate =  1.00 * einsum('lkdc,al,ck,dj,bi->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(a,b)<l,k||c,d>_aaaa*t1_aa(a,k)*t1_aa(c,j)*t1_aa(d,i)*r1_aa(b,l)
        contracted_intermediate =  1.00 * einsum('lkcd,ak,cj,di,bl->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (2, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,d>_aaaa*t1_aa(a,k)*t1_aa(b,l)*t1_aa(c,j)*r1_aa(d,i)
        contracted_intermediate =  1.00 * einsum('lkcd,ak,bl,cj,di->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, r1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
        sigma2_aaaa +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 <l,k||c,d>_aaaa*t1_aa(a,k)*t1_aa(b,l)*t1_aa(c,j)*t1_aa(d,i)*r0
        sigma2_aaaa +=  1.00 * einsum('lkcd,ak,bl,cj,di,->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 3), (2, 3), (0, 2), (0, 1)])
        
        return sigma2_aaaa
    
    
    def sigma_doubles_abab(self, r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab):

        t1_aa = self.t1_aa
        t1_bb = self.t1_bb
        t2_aaaa = self.t2_aaaa
        t2_bbbb = self.t2_bbbb
        t2_abab = self.t2_abab
        f_aa = self.f_aa
        f_bb = self.f_bb
        g_aaaa = self.g_aaaa
        g_bbbb = self.g_bbbb
        g_abab = self.g_abab
        oa = self.oa
        va = self.va
        ob = self.ob
        vb = self.vb
    
        #	  1.00 f_bb(b,j)*r1_aa(a,i)
        sigma2_abab =  1.00 * einsum('bj,ai->abij', f_bb[vb, ob], r1_aa)
        
        #	  1.00 f_aa(a,i)*r1_bb(b,j)
        sigma2_abab +=  1.00 * einsum('ai,bj->abij', f_aa[va, oa], r1_bb)
        
        #	  1.00 f_aa(k,k)*r2_abab(a,b,i,j)
        sigma2_abab +=  1.00 * einsum('kk,abij->abij', f_aa[oa, oa], r2_abab)
        
        #	  1.00 f_bb(k,k)*r2_abab(a,b,i,j)
        sigma2_abab +=  1.00 * einsum('kk,abij->abij', f_bb[ob, ob], r2_abab)
        
        #	 -1.00 f_bb(k,j)*r2_abab(a,b,i,k)
        sigma2_abab += -1.00 * einsum('kj,abik->abij', f_bb[ob, ob], r2_abab)
        
        #	 -1.00 f_aa(k,i)*r2_abab(a,b,k,j)
        sigma2_abab += -1.00 * einsum('ki,abkj->abij', f_aa[oa, oa], r2_abab)
        
        #	  1.00 f_aa(a,c)*r2_abab(c,b,i,j)
        sigma2_abab +=  1.00 * einsum('ac,cbij->abij', f_aa[va, va], r2_abab)
        
        #	  1.00 f_bb(b,c)*r2_abab(a,c,i,j)
        sigma2_abab +=  1.00 * einsum('bc,acij->abij', f_bb[vb, vb], r2_abab)
        
        #	 -1.00 f_bb(k,j)*t1_bb(b,k)*r1_aa(a,i)
        sigma2_abab += -1.00 * einsum('kj,bk,ai->abij', f_bb[ob, ob], t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 f_aa(k,i)*t1_aa(a,k)*r1_bb(b,j)
        sigma2_abab += -1.00 * einsum('ki,ak,bj->abij', f_aa[oa, oa], t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 f_bb(b,c)*t1_bb(c,j)*r1_aa(a,i)
        sigma2_abab +=  1.00 * einsum('bc,cj,ai->abij', f_bb[vb, vb], t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 f_aa(a,c)*t1_aa(c,i)*r1_bb(b,j)
        sigma2_abab +=  1.00 * einsum('ac,ci,bj->abij', f_aa[va, va], t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 f_aa(k,c)*r2_abab(a,b,i,j)*t1_aa(c,k)
        sigma2_abab +=  1.00 * einsum('kc,abij,ck->abij', f_aa[oa, va], r2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 f_bb(k,c)*r2_abab(a,b,i,j)*t1_bb(c,k)
        sigma2_abab +=  1.00 * einsum('kc,abij,ck->abij', f_bb[ob, vb], r2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_bb(k,c)*r2_abab(a,b,i,k)*t1_bb(c,j)
        sigma2_abab += -1.00 * einsum('kc,abik,cj->abij', f_bb[ob, vb], r2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_aa(k,c)*r2_abab(a,b,k,j)*t1_aa(c,i)
        sigma2_abab += -1.00 * einsum('kc,abkj,ci->abij', f_aa[oa, va], r2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 f_aa(k,c)*r2_abab(c,b,i,j)*t1_aa(a,k)
        sigma2_abab += -1.00 * einsum('kc,cbij,ak->abij', f_aa[oa, va], r2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_bb(k,c)*r2_abab(a,c,i,j)*t1_bb(b,k)
        sigma2_abab += -1.00 * einsum('kc,acij,bk->abij', f_bb[ob, vb], r2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 f_bb(k,j)*r0*t2_abab(a,b,i,k)
        sigma2_abab += -1.00 * einsum('kj,,abik->abij', f_bb[ob, ob], r0, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 f_aa(k,i)*r0*t2_abab(a,b,k,j)
        sigma2_abab += -1.00 * einsum('ki,,abkj->abij', f_aa[oa, oa], r0, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 f_aa(a,c)*r0*t2_abab(c,b,i,j)
        sigma2_abab +=  1.00 * einsum('ac,,cbij->abij', f_aa[va, va], r0, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 f_bb(b,c)*r0*t2_abab(a,c,i,j)
        sigma2_abab +=  1.00 * einsum('bc,,acij->abij', f_bb[vb, vb], r0, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 f_aa(k,c)*r1_aa(a,i)*t2_abab(c,b,k,j)
        sigma2_abab +=  1.00 * einsum('kc,ai,cbkj->abij', f_aa[oa, va], r1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_bb(k,c)*r1_aa(a,i)*t2_bbbb(c,b,j,k)
        sigma2_abab += -1.00 * einsum('kc,ai,cbjk->abij', f_bb[ob, vb], r1_aa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_aa(k,c)*r1_bb(b,j)*t2_aaaa(c,a,i,k)
        sigma2_abab += -1.00 * einsum('kc,bj,caik->abij', f_aa[oa, va], r1_bb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 f_bb(k,c)*r1_bb(b,j)*t2_abab(a,c,i,k)
        sigma2_abab +=  1.00 * einsum('kc,bj,acik->abij', f_bb[ob, vb], r1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_bb(k,c)*r1_bb(b,k)*t2_abab(a,c,i,j)
        sigma2_abab += -1.00 * einsum('kc,bk,acij->abij', f_bb[ob, vb], r1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_aa(k,c)*r1_aa(a,k)*t2_abab(c,b,i,j)
        sigma2_abab += -1.00 * einsum('kc,ak,cbij->abij', f_aa[oa, va], r1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 f_aa(k,c)*r1_aa(c,i)*t2_abab(a,b,k,j)
        sigma2_abab += -1.00 * einsum('kc,ci,abkj->abij', f_aa[oa, va], r1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 f_bb(k,c)*r1_bb(c,j)*t2_abab(a,b,i,k)
        sigma2_abab += -1.00 * einsum('kc,cj,abik->abij', f_bb[ob, vb], r1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 f_bb(k,c)*r0*t2_abab(a,b,i,k)*t1_bb(c,j)
        sigma2_abab += -1.00 * einsum('kc,,abik,cj->abij', f_bb[ob, vb], r0, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	 -1.00 f_aa(k,c)*r0*t2_abab(a,b,k,j)*t1_aa(c,i)
        sigma2_abab += -1.00 * einsum('kc,,abkj,ci->abij', f_aa[oa, va], r0, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 f_aa(k,c)*r0*t1_aa(a,k)*t2_abab(c,b,i,j)
        sigma2_abab += -1.00 * einsum('kc,,ak,cbij->abij', f_aa[oa, va], r0, t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 f_bb(k,c)*r0*t2_abab(a,c,i,j)*t1_bb(b,k)
        sigma2_abab += -1.00 * einsum('kc,,acij,bk->abij', f_bb[ob, vb], r0, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 f_bb(k,c)*r1_aa(a,i)*t1_bb(b,k)*t1_bb(c,j)
        sigma2_abab += -1.00 * einsum('kc,ai,bk,cj->abij', f_bb[ob, vb], r1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        
        #	 -1.00 f_aa(k,c)*r1_bb(b,j)*t1_aa(a,k)*t1_aa(c,i)
        sigma2_abab += -1.00 * einsum('kc,bj,ak,ci->abij', f_aa[oa, va], r1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -0.50 <l,k||l,k>_aaaa*r2_abab(a,b,i,j)
        sigma2_abab += -0.50 * einsum('lklk,abij->abij', g_aaaa[oa, oa, oa, oa], r2_abab)
        
        #	 -0.50 <l,k||l,k>_abab*r2_abab(a,b,i,j)
        sigma2_abab += -0.50 * einsum('lklk,abij->abij', g_abab[oa, ob, oa, ob], r2_abab)
        
        #	 -0.50 <k,l||k,l>_abab*r2_abab(a,b,i,j)
        sigma2_abab += -0.50 * einsum('klkl,abij->abij', g_abab[oa, ob, oa, ob], r2_abab)
        
        #	 -0.50 <l,k||l,k>_bbbb*r2_abab(a,b,i,j)
        sigma2_abab += -0.50 * einsum('lklk,abij->abij', g_bbbb[ob, ob, ob, ob], r2_abab)
        
        #	  1.00 <a,b||i,j>_abab*r0
        sigma2_abab +=  1.00 * einsum('abij,->abij', g_abab[va, vb, oa, ob], r0)
        
        #	 -1.00 <a,k||i,j>_abab*r1_bb(b,k)
        sigma2_abab += -1.00 * einsum('akij,bk->abij', g_abab[va, ob, oa, ob], r1_bb)
        
        #	 -1.00 <k,b||i,j>_abab*r1_aa(a,k)
        sigma2_abab += -1.00 * einsum('kbij,ak->abij', g_abab[oa, vb, oa, ob], r1_aa)
        
        #	  1.00 <a,b||c,j>_abab*r1_aa(c,i)
        sigma2_abab +=  1.00 * einsum('abcj,ci->abij', g_abab[va, vb, va, ob], r1_aa)
        
        #	  1.00 <a,b||i,c>_abab*r1_bb(c,j)
        sigma2_abab +=  1.00 * einsum('abic,cj->abij', g_abab[va, vb, oa, vb], r1_bb)
        
        #	  0.50 <l,k||i,j>_abab*r2_abab(a,b,l,k)
        sigma2_abab +=  0.50 * einsum('lkij,ablk->abij', g_abab[oa, ob, oa, ob], r2_abab)
        
        #	  0.50 <k,l||i,j>_abab*r2_abab(a,b,k,l)
        sigma2_abab +=  0.50 * einsum('klij,abkl->abij', g_abab[oa, ob, oa, ob], r2_abab)
        
        #	 -1.00 <a,k||c,j>_abab*r2_abab(c,b,i,k)
        sigma2_abab += -1.00 * einsum('akcj,cbik->abij', g_abab[va, ob, va, ob], r2_abab)
        
        #	 -1.00 <k,b||c,j>_abab*r2_aaaa(c,a,i,k)
        sigma2_abab += -1.00 * einsum('kbcj,caik->abij', g_abab[oa, vb, va, ob], r2_aaaa)
        
        #	  1.00 <k,b||c,j>_bbbb*r2_abab(a,c,i,k)
        sigma2_abab +=  1.00 * einsum('kbcj,acik->abij', g_bbbb[ob, vb, vb, ob], r2_abab)
        
        #	  1.00 <k,a||c,i>_aaaa*r2_abab(c,b,k,j)
        sigma2_abab +=  1.00 * einsum('kaci,cbkj->abij', g_aaaa[oa, va, va, oa], r2_abab)
        
        #	 -1.00 <a,k||i,c>_abab*r2_bbbb(c,b,j,k)
        sigma2_abab += -1.00 * einsum('akic,cbjk->abij', g_abab[va, ob, oa, vb], r2_bbbb)
        
        #	 -1.00 <k,b||i,c>_abab*r2_abab(a,c,k,j)
        sigma2_abab += -1.00 * einsum('kbic,ackj->abij', g_abab[oa, vb, oa, vb], r2_abab)
        
        #	  0.50 <a,b||c,d>_abab*r2_abab(c,d,i,j)
        sigma2_abab +=  0.50 * einsum('abcd,cdij->abij', g_abab[va, vb, va, vb], r2_abab)
        
        #	  0.50 <a,b||d,c>_abab*r2_abab(d,c,i,j)
        sigma2_abab +=  0.50 * einsum('abdc,dcij->abij', g_abab[va, vb, va, vb], r2_abab)
        
        #	 -1.00 <a,k||i,j>_abab*r0*t1_bb(b,k)
        sigma2_abab += -1.00 * einsum('akij,,bk->abij', g_abab[va, ob, oa, ob], r0, t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
        
        #	 -1.00 <k,b||i,j>_abab*r0*t1_aa(a,k)
        sigma2_abab += -1.00 * einsum('kbij,,ak->abij', g_abab[oa, vb, oa, ob], r0, t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
        
        #	  1.00 <a,b||c,j>_abab*r0*t1_aa(c,i)
        sigma2_abab +=  1.00 * einsum('abcj,,ci->abij', g_abab[va, vb, va, ob], r0, t1_aa, optimize=['einsum_path', (1, 2), (0, 1)])
        
        #	  1.00 <a,b||i,c>_abab*r0*t1_bb(c,j)
        sigma2_abab +=  1.00 * einsum('abic,,cj->abij', g_abab[va, vb, oa, vb], r0, t1_bb, optimize=['einsum_path', (1, 2), (0, 1)])
        
        #	  1.00 <k,l||i,j>_abab*r1_bb(b,l)*t1_aa(a,k)
        sigma2_abab +=  1.00 * einsum('klij,bl,ak->abij', g_abab[oa, ob, oa, ob], r1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <l,k||i,j>_abab*r1_aa(a,l)*t1_bb(b,k)
        sigma2_abab +=  1.00 * einsum('lkij,al,bk->abij', g_abab[oa, ob, oa, ob], r1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,b||c,j>_abab*r1_aa(a,i)*t1_aa(c,k)
        sigma2_abab +=  1.00 * einsum('kbcj,ai,ck->abij', g_abab[oa, vb, va, ob], r1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,b||c,j>_bbbb*r1_aa(a,i)*t1_bb(c,k)
        sigma2_abab +=  1.00 * einsum('kbcj,ai,ck->abij', g_bbbb[ob, vb, vb, ob], r1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,a||c,i>_aaaa*r1_bb(b,j)*t1_aa(c,k)
        sigma2_abab +=  1.00 * einsum('kaci,bj,ck->abij', g_aaaa[oa, va, va, oa], r1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <a,k||i,c>_abab*r1_bb(b,j)*t1_bb(c,k)
        sigma2_abab +=  1.00 * einsum('akic,bj,ck->abij', g_abab[va, ob, oa, vb], r1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <a,k||c,j>_abab*r1_bb(b,k)*t1_aa(c,i)
        sigma2_abab += -1.00 * einsum('akcj,bk,ci->abij', g_abab[va, ob, va, ob], r1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||c,j>_abab*r1_aa(a,k)*t1_aa(c,i)
        sigma2_abab += -1.00 * einsum('kbcj,ak,ci->abij', g_abab[oa, vb, va, ob], r1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <a,k||i,c>_abab*r1_bb(b,k)*t1_bb(c,j)
        sigma2_abab += -1.00 * einsum('akic,bk,cj->abij', g_abab[va, ob, oa, vb], r1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,b||i,c>_abab*r1_aa(a,k)*t1_bb(c,j)
        sigma2_abab += -1.00 * einsum('kbic,ak,cj->abij', g_abab[oa, vb, oa, vb], r1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <a,k||c,j>_abab*r1_aa(c,i)*t1_bb(b,k)
        sigma2_abab += -1.00 * einsum('akcj,ci,bk->abij', g_abab[va, ob, va, ob], r1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||c,j>_abab*r1_aa(c,i)*t1_aa(a,k)
        sigma2_abab += -1.00 * einsum('kbcj,ci,ak->abij', g_abab[oa, vb, va, ob], r1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <a,k||i,c>_abab*r1_bb(c,j)*t1_bb(b,k)
        sigma2_abab += -1.00 * einsum('akic,cj,bk->abij', g_abab[va, ob, oa, vb], r1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||i,c>_abab*r1_bb(c,j)*t1_aa(a,k)
        sigma2_abab += -1.00 * einsum('kbic,cj,ak->abij', g_abab[oa, vb, oa, vb], r1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <a,b||d,c>_abab*r1_aa(d,i)*t1_bb(c,j)
        sigma2_abab +=  1.00 * einsum('abdc,di,cj->abij', g_abab[va, vb, va, vb], r1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <a,b||c,d>_abab*r1_bb(d,j)*t1_aa(c,i)
        sigma2_abab +=  1.00 * einsum('abcd,dj,ci->abij', g_abab[va, vb, va, vb], r1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,l||c,j>_abab*r2_abab(a,b,i,l)*t1_aa(c,k)
        sigma2_abab += -1.00 * einsum('klcj,abil,ck->abij', g_abab[oa, ob, va, ob], r2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,j>_bbbb*r2_abab(a,b,i,l)*t1_bb(c,k)
        sigma2_abab +=  1.00 * einsum('lkcj,abil,ck->abij', g_bbbb[ob, ob, vb, ob], r2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,i>_aaaa*r2_abab(a,b,l,j)*t1_aa(c,k)
        sigma2_abab +=  1.00 * einsum('lkci,ablj,ck->abij', g_aaaa[oa, oa, va, oa], r2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||i,c>_abab*r2_abab(a,b,l,j)*t1_bb(c,k)
        sigma2_abab += -1.00 * einsum('lkic,ablj,ck->abij', g_abab[oa, ob, oa, vb], r2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,j>_abab*r2_abab(a,b,l,k)*t1_aa(c,i)
        sigma2_abab +=  0.50 * einsum('lkcj,ablk,ci->abij', g_abab[oa, ob, va, ob], r2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <k,l||c,j>_abab*r2_abab(a,b,k,l)*t1_aa(c,i)
        sigma2_abab +=  0.50 * einsum('klcj,abkl,ci->abij', g_abab[oa, ob, va, ob], r2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <l,k||i,c>_abab*r2_abab(a,b,l,k)*t1_bb(c,j)
        sigma2_abab +=  0.50 * einsum('lkic,ablk,cj->abij', g_abab[oa, ob, oa, vb], r2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 <k,l||i,c>_abab*r2_abab(a,b,k,l)*t1_bb(c,j)
        sigma2_abab +=  0.50 * einsum('klic,abkl,cj->abij', g_abab[oa, ob, oa, vb], r2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,l||c,j>_abab*t1_aa(a,k)*r2_abab(c,b,i,l)
        sigma2_abab +=  1.00 * einsum('klcj,ak,cbil->abij', g_abab[oa, ob, va, ob], t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,j>_abab*t1_bb(b,k)*r2_aaaa(c,a,i,l)
        sigma2_abab +=  1.00 * einsum('lkcj,bk,cail->abij', g_abab[oa, ob, va, ob], t1_bb, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||c,j>_bbbb*t1_bb(b,k)*r2_abab(a,c,i,l)
        sigma2_abab += -1.00 * einsum('lkcj,bk,acil->abij', g_bbbb[ob, ob, vb, ob], t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||c,i>_aaaa*t1_aa(a,k)*r2_abab(c,b,l,j)
        sigma2_abab += -1.00 * einsum('lkci,ak,cblj->abij', g_aaaa[oa, oa, va, oa], t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,l||i,c>_abab*t1_aa(a,k)*r2_bbbb(c,b,j,l)
        sigma2_abab +=  1.00 * einsum('klic,ak,cbjl->abij', g_abab[oa, ob, oa, vb], t1_aa, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||i,c>_abab*t1_bb(b,k)*r2_abab(a,c,l,j)
        sigma2_abab +=  1.00 * einsum('lkic,bk,aclj->abij', g_abab[oa, ob, oa, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,a||c,d>_aaaa*r2_abab(d,b,i,j)*t1_aa(c,k)
        sigma2_abab +=  1.00 * einsum('kacd,dbij,ck->abij', g_aaaa[oa, va, va, va], r2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <a,k||d,c>_abab*r2_abab(d,b,i,j)*t1_bb(c,k)
        sigma2_abab +=  1.00 * einsum('akdc,dbij,ck->abij', g_abab[va, ob, va, vb], r2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,b||c,d>_abab*r2_abab(a,d,i,j)*t1_aa(c,k)
        sigma2_abab +=  1.00 * einsum('kbcd,adij,ck->abij', g_abab[oa, vb, va, vb], r2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,b||c,d>_bbbb*r2_abab(a,d,i,j)*t1_bb(c,k)
        sigma2_abab +=  1.00 * einsum('kbcd,adij,ck->abij', g_bbbb[ob, vb, vb, vb], r2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <a,k||d,c>_abab*t1_bb(c,j)*r2_abab(d,b,i,k)
        sigma2_abab += -1.00 * einsum('akdc,cj,dbik->abij', g_abab[va, ob, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||d,c>_abab*t1_bb(c,j)*r2_aaaa(d,a,i,k)
        sigma2_abab += -1.00 * einsum('kbdc,cj,daik->abij', g_abab[oa, vb, va, vb], t1_bb, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||c,d>_bbbb*t1_bb(c,j)*r2_abab(a,d,i,k)
        sigma2_abab += -1.00 * einsum('kbcd,cj,adik->abij', g_bbbb[ob, vb, vb, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,a||c,d>_aaaa*t1_aa(c,i)*r2_abab(d,b,k,j)
        sigma2_abab += -1.00 * einsum('kacd,ci,dbkj->abij', g_aaaa[oa, va, va, va], t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <a,k||c,d>_abab*t1_aa(c,i)*r2_bbbb(d,b,j,k)
        sigma2_abab += -1.00 * einsum('akcd,ci,dbjk->abij', g_abab[va, ob, va, vb], t1_aa, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,b||c,d>_abab*t1_aa(c,i)*r2_abab(a,d,k,j)
        sigma2_abab += -1.00 * einsum('kbcd,ci,adkj->abij', g_abab[oa, vb, va, vb], t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <a,k||c,d>_abab*r2_abab(c,d,i,j)*t1_bb(b,k)
        sigma2_abab += -0.50 * einsum('akcd,cdij,bk->abij', g_abab[va, ob, va, vb], r2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <a,k||d,c>_abab*r2_abab(d,c,i,j)*t1_bb(b,k)
        sigma2_abab += -0.50 * einsum('akdc,dcij,bk->abij', g_abab[va, ob, va, vb], r2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,b||c,d>_abab*r2_abab(c,d,i,j)*t1_aa(a,k)
        sigma2_abab += -0.50 * einsum('kbcd,cdij,ak->abij', g_abab[oa, vb, va, vb], r2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,b||d,c>_abab*r2_abab(d,c,i,j)*t1_aa(a,k)
        sigma2_abab += -0.50 * einsum('kbdc,dcij,ak->abij', g_abab[oa, vb, va, vb], r2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 <l,k||i,j>_abab*t2_abab(a,b,l,k)*r0
        sigma2_abab +=  0.50 * einsum('lkij,ablk,->abij', g_abab[oa, ob, oa, ob], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <k,l||i,j>_abab*t2_abab(a,b,k,l)*r0
        sigma2_abab +=  0.50 * einsum('klij,abkl,->abij', g_abab[oa, ob, oa, ob], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <a,k||c,j>_abab*t2_abab(c,b,i,k)*r0
        sigma2_abab += -1.00 * einsum('akcj,cbik,->abij', g_abab[va, ob, va, ob], t2_abab, r0, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,b||c,j>_abab*t2_aaaa(c,a,i,k)*r0
        sigma2_abab += -1.00 * einsum('kbcj,caik,->abij', g_abab[oa, vb, va, ob], t2_aaaa, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,b||c,j>_bbbb*t2_abab(a,c,i,k)*r0
        sigma2_abab +=  1.00 * einsum('kbcj,acik,->abij', g_bbbb[ob, vb, vb, ob], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,a||c,i>_aaaa*t2_abab(c,b,k,j)*r0
        sigma2_abab +=  1.00 * einsum('kaci,cbkj,->abij', g_aaaa[oa, va, va, oa], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <a,k||i,c>_abab*t2_bbbb(c,b,j,k)*r0
        sigma2_abab += -1.00 * einsum('akic,cbjk,->abij', g_abab[va, ob, oa, vb], t2_bbbb, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||i,c>_abab*t2_abab(a,c,k,j)*r0
        sigma2_abab += -1.00 * einsum('kbic,ackj,->abij', g_abab[oa, vb, oa, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <a,b||c,d>_abab*t2_abab(c,d,i,j)*r0
        sigma2_abab +=  0.50 * einsum('abcd,cdij,->abij', g_abab[va, vb, va, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <a,b||d,c>_abab*t2_abab(d,c,i,j)*r0
        sigma2_abab +=  0.50 * einsum('abdc,dcij,->abij', g_abab[va, vb, va, vb], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||c,j>_abab*t2_abab(c,b,l,k)*r1_aa(a,i)
        sigma2_abab += -0.50 * einsum('lkcj,cblk,ai->abij', g_abab[oa, ob, va, ob], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,l||c,j>_abab*t2_abab(c,b,k,l)*r1_aa(a,i)
        sigma2_abab += -0.50 * einsum('klcj,cbkl,ai->abij', g_abab[oa, ob, va, ob], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||c,j>_bbbb*t2_bbbb(c,b,l,k)*r1_aa(a,i)
        sigma2_abab += -0.50 * einsum('lkcj,cblk,ai->abij', g_bbbb[ob, ob, vb, ob], t2_bbbb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||c,i>_aaaa*t2_aaaa(c,a,l,k)*r1_bb(b,j)
        sigma2_abab += -0.50 * einsum('lkci,calk,bj->abij', g_aaaa[oa, oa, va, oa], t2_aaaa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||i,c>_abab*t2_abab(a,c,l,k)*r1_bb(b,j)
        sigma2_abab += -0.50 * einsum('lkic,aclk,bj->abij', g_abab[oa, ob, oa, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,l||i,c>_abab*t2_abab(a,c,k,l)*r1_bb(b,j)
        sigma2_abab += -0.50 * einsum('klic,ackl,bj->abij', g_abab[oa, ob, oa, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,l||c,j>_abab*t2_aaaa(c,a,i,k)*r1_bb(b,l)
        sigma2_abab +=  1.00 * einsum('klcj,caik,bl->abij', g_abab[oa, ob, va, ob], t2_aaaa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,j>_bbbb*t2_abab(a,c,i,k)*r1_bb(b,l)
        sigma2_abab +=  1.00 * einsum('lkcj,acik,bl->abij', g_bbbb[ob, ob, vb, ob], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,j>_abab*t2_abab(c,b,i,k)*r1_aa(a,l)
        sigma2_abab +=  1.00 * einsum('lkcj,cbik,al->abij', g_abab[oa, ob, va, ob], t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,l||i,c>_abab*t2_abab(a,c,k,j)*r1_bb(b,l)
        sigma2_abab +=  1.00 * einsum('klic,ackj,bl->abij', g_abab[oa, ob, oa, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,i>_aaaa*t2_abab(c,b,k,j)*r1_aa(a,l)
        sigma2_abab +=  1.00 * einsum('lkci,cbkj,al->abij', g_aaaa[oa, oa, va, oa], t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <l,k||i,c>_abab*t2_bbbb(c,b,j,k)*r1_aa(a,l)
        sigma2_abab +=  1.00 * einsum('lkic,cbjk,al->abij', g_abab[oa, ob, oa, vb], t2_bbbb, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,j>_abab*r1_aa(c,i)*t2_abab(a,b,l,k)
        sigma2_abab +=  0.50 * einsum('lkcj,ci,ablk->abij', g_abab[oa, ob, va, ob], r1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 <k,l||c,j>_abab*r1_aa(c,i)*t2_abab(a,b,k,l)
        sigma2_abab +=  0.50 * einsum('klcj,ci,abkl->abij', g_abab[oa, ob, va, ob], r1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 <l,k||i,c>_abab*r1_bb(c,j)*t2_abab(a,b,l,k)
        sigma2_abab +=  0.50 * einsum('lkic,cj,ablk->abij', g_abab[oa, ob, oa, vb], r1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <k,l||i,c>_abab*r1_bb(c,j)*t2_abab(a,b,k,l)
        sigma2_abab +=  0.50 * einsum('klic,cj,abkl->abij', g_abab[oa, ob, oa, vb], r1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <l,k||c,j>_abab*r1_aa(c,l)*t2_abab(a,b,i,k)
        sigma2_abab += -1.00 * einsum('lkcj,cl,abik->abij', g_abab[oa, ob, va, ob], r1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <l,k||c,j>_bbbb*r1_bb(c,l)*t2_abab(a,b,i,k)
        sigma2_abab += -1.00 * einsum('lkcj,cl,abik->abij', g_bbbb[ob, ob, vb, ob], r1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <l,k||c,i>_aaaa*r1_aa(c,l)*t2_abab(a,b,k,j)
        sigma2_abab += -1.00 * einsum('lkci,cl,abkj->abij', g_aaaa[oa, oa, va, oa], r1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,l||i,c>_abab*r1_bb(c,l)*t2_abab(a,b,k,j)
        sigma2_abab += -1.00 * einsum('klic,cl,abkj->abij', g_abab[oa, ob, oa, vb], r1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <k,b||c,d>_abab*t2_abab(c,d,k,j)*r1_aa(a,i)
        sigma2_abab +=  0.50 * einsum('kbcd,cdkj,ai->abij', g_abab[oa, vb, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <k,b||d,c>_abab*t2_abab(d,c,k,j)*r1_aa(a,i)
        sigma2_abab +=  0.50 * einsum('kbdc,dckj,ai->abij', g_abab[oa, vb, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,b||c,d>_bbbb*t2_bbbb(c,d,j,k)*r1_aa(a,i)
        sigma2_abab += -0.50 * einsum('kbcd,cdjk,ai->abij', g_bbbb[ob, vb, vb, vb], t2_bbbb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,a||c,d>_aaaa*t2_aaaa(c,d,i,k)*r1_bb(b,j)
        sigma2_abab += -0.50 * einsum('kacd,cdik,bj->abij', g_aaaa[oa, va, va, va], t2_aaaa, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <a,k||c,d>_abab*t2_abab(c,d,i,k)*r1_bb(b,j)
        sigma2_abab +=  0.50 * einsum('akcd,cdik,bj->abij', g_abab[va, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <a,k||d,c>_abab*t2_abab(d,c,i,k)*r1_bb(b,j)
        sigma2_abab +=  0.50 * einsum('akdc,dcik,bj->abij', g_abab[va, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <a,k||c,d>_abab*r1_bb(b,k)*t2_abab(c,d,i,j)
        sigma2_abab += -0.50 * einsum('akcd,bk,cdij->abij', g_abab[va, ob, va, vb], r1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <a,k||d,c>_abab*r1_bb(b,k)*t2_abab(d,c,i,j)
        sigma2_abab += -0.50 * einsum('akdc,bk,dcij->abij', g_abab[va, ob, va, vb], r1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,b||c,d>_abab*r1_aa(a,k)*t2_abab(c,d,i,j)
        sigma2_abab += -0.50 * einsum('kbcd,ak,cdij->abij', g_abab[oa, vb, va, vb], r1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,b||d,c>_abab*r1_aa(a,k)*t2_abab(d,c,i,j)
        sigma2_abab += -0.50 * einsum('kbdc,ak,dcij->abij', g_abab[oa, vb, va, vb], r1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,a||c,d>_aaaa*t2_abab(c,b,k,j)*r1_aa(d,i)
        sigma2_abab +=  1.00 * einsum('kacd,cbkj,di->abij', g_aaaa[oa, va, va, va], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <a,k||d,c>_abab*t2_bbbb(c,b,j,k)*r1_aa(d,i)
        sigma2_abab += -1.00 * einsum('akdc,cbjk,di->abij', g_abab[va, ob, va, vb], t2_bbbb, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||d,c>_abab*t2_abab(a,c,k,j)*r1_aa(d,i)
        sigma2_abab += -1.00 * einsum('kbdc,ackj,di->abij', g_abab[oa, vb, va, vb], t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <a,k||c,d>_abab*t2_abab(c,b,i,k)*r1_bb(d,j)
        sigma2_abab += -1.00 * einsum('akcd,cbik,dj->abij', g_abab[va, ob, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,b||c,d>_abab*t2_aaaa(c,a,i,k)*r1_bb(d,j)
        sigma2_abab += -1.00 * einsum('kbcd,caik,dj->abij', g_abab[oa, vb, va, vb], t2_aaaa, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 <k,b||c,d>_bbbb*t2_abab(a,c,i,k)*r1_bb(d,j)
        sigma2_abab +=  1.00 * einsum('kbcd,acik,dj->abij', g_bbbb[ob, vb, vb, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,a||c,d>_aaaa*r1_aa(d,k)*t2_abab(c,b,i,j)
        sigma2_abab += -1.00 * einsum('kacd,dk,cbij->abij', g_aaaa[oa, va, va, va], r1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <a,k||c,d>_abab*r1_bb(d,k)*t2_abab(c,b,i,j)
        sigma2_abab +=  1.00 * einsum('akcd,dk,cbij->abij', g_abab[va, ob, va, vb], r1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,b||d,c>_abab*r1_aa(d,k)*t2_abab(a,c,i,j)
        sigma2_abab +=  1.00 * einsum('kbdc,dk,acij->abij', g_abab[oa, vb, va, vb], r1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||c,d>_bbbb*r1_bb(d,k)*t2_abab(a,c,i,j)
        sigma2_abab += -1.00 * einsum('kbcd,dk,acij->abij', g_bbbb[ob, vb, vb, vb], r1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||c,d>_aaaa*t2_aaaa(c,d,l,k)*r2_abab(a,b,i,j)
        sigma2_abab +=  0.250 * einsum('lkcd,cdlk,abij->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||c,d>_abab*t2_abab(c,d,l,k)*r2_abab(a,b,i,j)
        sigma2_abab +=  0.250 * einsum('lkcd,cdlk,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,l||c,d>_abab*t2_abab(c,d,k,l)*r2_abab(a,b,i,j)
        sigma2_abab +=  0.250 * einsum('klcd,cdkl,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||d,c>_abab*t2_abab(d,c,l,k)*r2_abab(a,b,i,j)
        sigma2_abab +=  0.250 * einsum('lkdc,dclk,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,l||d,c>_abab*t2_abab(d,c,k,l)*r2_abab(a,b,i,j)
        sigma2_abab +=  0.250 * einsum('kldc,dckl,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||c,d>_bbbb*t2_bbbb(c,d,l,k)*r2_abab(a,b,i,j)
        sigma2_abab +=  0.250 * einsum('lkcd,cdlk,abij->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,l||c,d>_abab*t2_abab(c,d,k,j)*r2_abab(a,b,i,l)
        sigma2_abab += -0.50 * einsum('klcd,cdkj,abil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,l||d,c>_abab*t2_abab(d,c,k,j)*r2_abab(a,b,i,l)
        sigma2_abab += -0.50 * einsum('kldc,dckj,abil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_bbbb*t2_bbbb(c,d,j,k)*r2_abab(a,b,i,l)
        sigma2_abab += -0.50 * einsum('lkcd,cdjk,abil->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_aaaa*t2_aaaa(c,d,i,k)*r2_abab(a,b,l,j)
        sigma2_abab += -0.50 * einsum('lkcd,cdik,ablj->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_abab*t2_abab(c,d,i,k)*r2_abab(a,b,l,j)
        sigma2_abab += -0.50 * einsum('lkcd,cdik,ablj->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||d,c>_abab*t2_abab(d,c,i,k)*r2_abab(a,b,l,j)
        sigma2_abab += -0.50 * einsum('lkdc,dcik,ablj->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||c,d>_abab*t2_abab(c,d,i,j)*r2_abab(a,b,l,k)
        sigma2_abab +=  0.250 * einsum('lkcd,cdij,ablk->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,l||c,d>_abab*t2_abab(c,d,i,j)*r2_abab(a,b,k,l)
        sigma2_abab +=  0.250 * einsum('klcd,cdij,abkl->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||d,c>_abab*t2_abab(d,c,i,j)*r2_abab(a,b,l,k)
        sigma2_abab +=  0.250 * einsum('lkdc,dcij,ablk->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,l||d,c>_abab*t2_abab(d,c,i,j)*r2_abab(a,b,k,l)
        sigma2_abab +=  0.250 * einsum('kldc,dcij,abkl->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_aaaa*t2_aaaa(c,a,l,k)*r2_abab(d,b,i,j)
        sigma2_abab += -0.50 * einsum('lkcd,calk,dbij->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||d,c>_abab*t2_abab(a,c,l,k)*r2_abab(d,b,i,j)
        sigma2_abab += -0.50 * einsum('lkdc,aclk,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,l||d,c>_abab*t2_abab(a,c,k,l)*r2_abab(d,b,i,j)
        sigma2_abab += -0.50 * einsum('kldc,ackl,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_abab*t2_abab(c,b,l,k)*r2_abab(a,d,i,j)
        sigma2_abab += -0.50 * einsum('lkcd,cblk,adij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <k,l||c,d>_abab*t2_abab(c,b,k,l)*r2_abab(a,d,i,j)
        sigma2_abab += -0.50 * einsum('klcd,cbkl,adij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_bbbb*t2_bbbb(c,b,l,k)*r2_abab(a,d,i,j)
        sigma2_abab += -0.50 * einsum('lkcd,cblk,adij->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,l||d,c>_abab*t2_abab(a,c,k,j)*r2_abab(d,b,i,l)
        sigma2_abab +=  1.00 * einsum('kldc,ackj,dbil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*t2_abab(c,b,k,j)*r2_aaaa(d,a,i,l)
        sigma2_abab +=  1.00 * einsum('lkcd,cbkj,dail->abij', g_aaaa[oa, oa, va, va], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,l||c,d>_abab*t2_abab(c,b,k,j)*r2_abab(a,d,i,l)
        sigma2_abab +=  1.00 * einsum('klcd,cbkj,adil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||d,c>_abab*t2_bbbb(c,b,j,k)*r2_aaaa(d,a,i,l)
        sigma2_abab +=  1.00 * einsum('lkdc,cbjk,dail->abij', g_abab[oa, ob, va, vb], t2_bbbb, r2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*t2_bbbb(c,b,j,k)*r2_abab(a,d,i,l)
        sigma2_abab +=  1.00 * einsum('lkcd,cbjk,adil->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*t2_aaaa(c,a,i,k)*r2_abab(d,b,l,j)
        sigma2_abab +=  1.00 * einsum('lkcd,caik,dblj->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <k,l||c,d>_abab*t2_aaaa(c,a,i,k)*r2_bbbb(d,b,j,l)
        sigma2_abab +=  1.00 * einsum('klcd,caik,dbjl->abij', g_abab[oa, ob, va, vb], t2_aaaa, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||d,c>_abab*t2_abab(a,c,i,k)*r2_abab(d,b,l,j)
        sigma2_abab +=  1.00 * einsum('lkdc,acik,dblj->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*t2_abab(a,c,i,k)*r2_bbbb(d,b,j,l)
        sigma2_abab +=  1.00 * einsum('lkcd,acik,dbjl->abij', g_bbbb[ob, ob, vb, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,d>_abab*t2_abab(c,b,i,k)*r2_abab(a,d,l,j)
        sigma2_abab +=  1.00 * einsum('lkcd,cbik,adlj->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||d,c>_abab*t2_abab(a,c,i,j)*r2_abab(d,b,l,k)
        sigma2_abab += -0.50 * einsum('lkdc,acij,dblk->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,l||d,c>_abab*t2_abab(a,c,i,j)*r2_abab(d,b,k,l)
        sigma2_abab += -0.50 * einsum('kldc,acij,dbkl->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,d>_bbbb*t2_abab(a,c,i,j)*r2_bbbb(d,b,l,k)
        sigma2_abab +=  0.50 * einsum('lkcd,acij,dblk->abij', g_bbbb[ob, ob, vb, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,d>_aaaa*t2_abab(c,b,i,j)*r2_aaaa(d,a,l,k)
        sigma2_abab +=  0.50 * einsum('lkcd,cbij,dalk->abij', g_aaaa[oa, oa, va, va], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_abab*t2_abab(c,b,i,j)*r2_abab(a,d,l,k)
        sigma2_abab += -0.50 * einsum('lkcd,cbij,adlk->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,l||c,d>_abab*t2_abab(c,b,i,j)*r2_abab(a,d,k,l)
        sigma2_abab += -0.50 * einsum('klcd,cbij,adkl->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.250 <l,k||c,d>_abab*t2_abab(a,b,l,k)*r2_abab(c,d,i,j)
        sigma2_abab +=  0.250 * einsum('lkcd,ablk,cdij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||d,c>_abab*t2_abab(a,b,l,k)*r2_abab(d,c,i,j)
        sigma2_abab +=  0.250 * einsum('lkdc,ablk,dcij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,l||c,d>_abab*t2_abab(a,b,k,l)*r2_abab(c,d,i,j)
        sigma2_abab +=  0.250 * einsum('klcd,abkl,cdij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,l||d,c>_abab*t2_abab(a,b,k,l)*r2_abab(d,c,i,j)
        sigma2_abab +=  0.250 * einsum('kldc,abkl,dcij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 <l,k||c,d>_aaaa*t2_abab(a,b,k,j)*r2_aaaa(c,d,i,l)
        sigma2_abab +=  0.50 * einsum('lkcd,abkj,cdil->abij', g_aaaa[oa, oa, va, va], t2_abab, r2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,l||c,d>_abab*t2_abab(a,b,k,j)*r2_abab(c,d,i,l)
        sigma2_abab += -0.50 * einsum('klcd,abkj,cdil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <k,l||d,c>_abab*t2_abab(a,b,k,j)*r2_abab(d,c,i,l)
        sigma2_abab += -0.50 * einsum('kldc,abkj,dcil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_abab*t2_abab(a,b,i,k)*r2_abab(c,d,l,j)
        sigma2_abab += -0.50 * einsum('lkcd,abik,cdlj->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||d,c>_abab*t2_abab(a,b,i,k)*r2_abab(d,c,l,j)
        sigma2_abab += -0.50 * einsum('lkdc,abik,dclj->abij', g_abab[oa, ob, va, vb], t2_abab, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,d>_bbbb*t2_abab(a,b,i,k)*r2_bbbb(c,d,j,l)
        sigma2_abab +=  0.50 * einsum('lkcd,abik,cdjl->abij', g_bbbb[ob, ob, vb, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 <k,l||c,j>_abab*r0*t2_abab(a,b,i,l)*t1_aa(c,k)
        sigma2_abab += -1.00 * einsum('klcj,,abil,ck->abij', g_abab[oa, ob, va, ob], r0, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,j>_bbbb*r0*t2_abab(a,b,i,l)*t1_bb(c,k)
        sigma2_abab +=  1.00 * einsum('lkcj,,abil,ck->abij', g_bbbb[ob, ob, vb, ob], r0, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,i>_aaaa*r0*t2_abab(a,b,l,j)*t1_aa(c,k)
        sigma2_abab +=  1.00 * einsum('lkci,,ablj,ck->abij', g_aaaa[oa, oa, va, oa], r0, t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||i,c>_abab*r0*t2_abab(a,b,l,j)*t1_bb(c,k)
        sigma2_abab += -1.00 * einsum('lkic,,ablj,ck->abij', g_abab[oa, ob, oa, vb], r0, t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,j>_abab*r0*t2_abab(a,b,l,k)*t1_aa(c,i)
        sigma2_abab +=  0.50 * einsum('lkcj,,ablk,ci->abij', g_abab[oa, ob, va, ob], r0, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
        
        #	  0.50 <k,l||c,j>_abab*r0*t2_abab(a,b,k,l)*t1_aa(c,i)
        sigma2_abab +=  0.50 * einsum('klcj,,abkl,ci->abij', g_abab[oa, ob, va, ob], r0, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
        
        #	  0.50 <l,k||i,c>_abab*r0*t2_abab(a,b,l,k)*t1_bb(c,j)
        sigma2_abab +=  0.50 * einsum('lkic,,ablk,cj->abij', g_abab[oa, ob, oa, vb], r0, t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        
        #	  0.50 <k,l||i,c>_abab*r0*t2_abab(a,b,k,l)*t1_bb(c,j)
        sigma2_abab +=  0.50 * einsum('klic,,abkl,cj->abij', g_abab[oa, ob, oa, vb], r0, t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||c,j>_abab*t1_aa(a,k)*t2_abab(c,b,i,l)*r0
        sigma2_abab +=  1.00 * einsum('klcj,ak,cbil,->abij', g_abab[oa, ob, va, ob], t1_aa, t2_abab, r0, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,j>_abab*t2_aaaa(c,a,i,l)*t1_bb(b,k)*r0
        sigma2_abab +=  1.00 * einsum('lkcj,cail,bk,->abij', g_abab[oa, ob, va, ob], t2_aaaa, t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -1.00 <l,k||c,j>_bbbb*t2_abab(a,c,i,l)*t1_bb(b,k)*r0
        sigma2_abab += -1.00 * einsum('lkcj,acil,bk,->abij', g_bbbb[ob, ob, vb, ob], t2_abab, t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -1.00 <l,k||c,i>_aaaa*t1_aa(a,k)*t2_abab(c,b,l,j)*r0
        sigma2_abab += -1.00 * einsum('lkci,ak,cblj,->abij', g_aaaa[oa, oa, va, oa], t1_aa, t2_abab, r0, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||i,c>_abab*t1_aa(a,k)*t2_bbbb(c,b,j,l)*r0
        sigma2_abab +=  1.00 * einsum('klic,ak,cbjl,->abij', g_abab[oa, ob, oa, vb], t1_aa, t2_bbbb, r0, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||i,c>_abab*t2_abab(a,c,l,j)*t1_bb(b,k)*r0
        sigma2_abab +=  1.00 * einsum('lkic,aclj,bk,->abij', g_abab[oa, ob, oa, vb], t2_abab, t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	  1.00 <k,a||c,d>_aaaa*r0*t2_abab(d,b,i,j)*t1_aa(c,k)
        sigma2_abab +=  1.00 * einsum('kacd,,dbij,ck->abij', g_aaaa[oa, va, va, va], r0, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  1.00 <a,k||d,c>_abab*r0*t2_abab(d,b,i,j)*t1_bb(c,k)
        sigma2_abab +=  1.00 * einsum('akdc,,dbij,ck->abij', g_abab[va, ob, va, vb], r0, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  1.00 <k,b||c,d>_abab*r0*t2_abab(a,d,i,j)*t1_aa(c,k)
        sigma2_abab +=  1.00 * einsum('kbcd,,adij,ck->abij', g_abab[oa, vb, va, vb], r0, t2_abab, t1_aa, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        
        #	  1.00 <k,b||c,d>_bbbb*r0*t2_abab(a,d,i,j)*t1_bb(c,k)
        sigma2_abab +=  1.00 * einsum('kbcd,,adij,ck->abij', g_bbbb[ob, vb, vb, vb], r0, t2_abab, t1_bb, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <a,k||d,c>_abab*t2_abab(d,b,i,k)*t1_bb(c,j)*r0
        sigma2_abab += -1.00 * einsum('akdc,dbik,cj,->abij', g_abab[va, ob, va, vb], t2_abab, t1_bb, r0, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <k,b||d,c>_abab*t2_aaaa(d,a,i,k)*t1_bb(c,j)*r0
        sigma2_abab += -1.00 * einsum('kbdc,daik,cj,->abij', g_abab[oa, vb, va, vb], t2_aaaa, t1_bb, r0, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <k,b||c,d>_bbbb*t2_abab(a,d,i,k)*t1_bb(c,j)*r0
        sigma2_abab += -1.00 * einsum('kbcd,adik,cj,->abij', g_bbbb[ob, vb, vb, vb], t2_abab, t1_bb, r0, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <k,a||c,d>_aaaa*t2_abab(d,b,k,j)*t1_aa(c,i)*r0
        sigma2_abab += -1.00 * einsum('kacd,dbkj,ci,->abij', g_aaaa[oa, va, va, va], t2_abab, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -1.00 <a,k||c,d>_abab*t2_bbbb(d,b,j,k)*t1_aa(c,i)*r0
        sigma2_abab += -1.00 * einsum('akcd,dbjk,ci,->abij', g_abab[va, ob, va, vb], t2_bbbb, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||c,d>_abab*t2_abab(a,d,k,j)*t1_aa(c,i)*r0
        sigma2_abab += -1.00 * einsum('kbcd,adkj,ci,->abij', g_abab[oa, vb, va, vb], t2_abab, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -0.50 <a,k||c,d>_abab*r0*t1_bb(b,k)*t2_abab(c,d,i,j)
        sigma2_abab += -0.50 * einsum('akcd,,bk,cdij->abij', g_abab[va, ob, va, vb], r0, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
        
        #	 -0.50 <a,k||d,c>_abab*r0*t1_bb(b,k)*t2_abab(d,c,i,j)
        sigma2_abab += -0.50 * einsum('akdc,,bk,dcij->abij', g_abab[va, ob, va, vb], r0, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
        
        #	 -0.50 <k,b||c,d>_abab*r0*t1_aa(a,k)*t2_abab(c,d,i,j)
        sigma2_abab += -0.50 * einsum('kbcd,,ak,cdij->abij', g_abab[oa, vb, va, vb], r0, t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
        
        #	 -0.50 <k,b||d,c>_abab*r0*t1_aa(a,k)*t2_abab(d,c,i,j)
        sigma2_abab += -0.50 * einsum('kbdc,,ak,dcij->abij', g_abab[oa, vb, va, vb], r0, t1_aa, t2_abab, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||c,d>_aaaa*t2_abab(d,b,l,j)*t1_aa(c,k)*r1_aa(a,i)
        sigma2_abab += -1.00 * einsum('lkcd,dblj,ck,ai->abij', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||d,c>_abab*t2_abab(d,b,l,j)*t1_bb(c,k)*r1_aa(a,i)
        sigma2_abab +=  1.00 * einsum('lkdc,dblj,ck,ai->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <k,l||c,d>_abab*t2_bbbb(d,b,j,l)*t1_aa(c,k)*r1_aa(a,i)
        sigma2_abab += -1.00 * einsum('klcd,dbjl,ck,ai->abij', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*t2_bbbb(d,b,j,l)*t1_bb(c,k)*r1_aa(a,i)
        sigma2_abab +=  1.00 * einsum('lkcd,dbjl,ck,ai->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*t2_aaaa(d,a,i,l)*t1_aa(c,k)*r1_bb(b,j)
        sigma2_abab +=  1.00 * einsum('lkcd,dail,ck,bj->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||d,c>_abab*t2_aaaa(d,a,i,l)*t1_bb(c,k)*r1_bb(b,j)
        sigma2_abab += -1.00 * einsum('lkdc,dail,ck,bj->abij', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||c,d>_abab*t2_abab(a,d,i,l)*t1_aa(c,k)*r1_bb(b,j)
        sigma2_abab +=  1.00 * einsum('klcd,adil,ck,bj->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||c,d>_bbbb*t2_abab(a,d,i,l)*t1_bb(c,k)*r1_bb(b,j)
        sigma2_abab += -1.00 * einsum('lkcd,adil,ck,bj->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <k,l||c,d>_abab*t2_abab(a,d,i,j)*t1_aa(c,k)*r1_bb(b,l)
        sigma2_abab += -1.00 * einsum('klcd,adij,ck,bl->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*t2_abab(a,d,i,j)*t1_bb(c,k)*r1_bb(b,l)
        sigma2_abab +=  1.00 * einsum('lkcd,adij,ck,bl->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*t2_abab(d,b,i,j)*t1_aa(c,k)*r1_aa(a,l)
        sigma2_abab +=  1.00 * einsum('lkcd,dbij,ck,al->abij', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -1.00 <l,k||d,c>_abab*t2_abab(d,b,i,j)*t1_bb(c,k)*r1_aa(a,l)
        sigma2_abab += -1.00 * einsum('lkdc,dbij,ck,al->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*t2_abab(a,b,l,j)*t1_aa(c,k)*r1_aa(d,i)
        sigma2_abab +=  1.00 * einsum('lkcd,ablj,ck,di->abij', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||d,c>_abab*t2_abab(a,b,l,j)*t1_bb(c,k)*r1_aa(d,i)
        sigma2_abab += -1.00 * einsum('lkdc,ablj,ck,di->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <k,l||c,d>_abab*t2_abab(a,b,i,l)*t1_aa(c,k)*r1_bb(d,j)
        sigma2_abab += -1.00 * einsum('klcd,abil,ck,dj->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*t2_abab(a,b,i,l)*t1_bb(c,k)*r1_bb(d,j)
        sigma2_abab +=  1.00 * einsum('lkcd,abil,ck,dj->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -0.50 <l,k||d,c>_abab*t2_abab(d,b,l,k)*t1_bb(c,j)*r1_aa(a,i)
        sigma2_abab += -0.50 * einsum('lkdc,dblk,cj,ai->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -0.50 <k,l||d,c>_abab*t2_abab(d,b,k,l)*t1_bb(c,j)*r1_aa(a,i)
        sigma2_abab += -0.50 * einsum('kldc,dbkl,cj,ai->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,d>_bbbb*t2_bbbb(d,b,l,k)*t1_bb(c,j)*r1_aa(a,i)
        sigma2_abab +=  0.50 * einsum('lkcd,dblk,cj,ai->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,d>_aaaa*t2_aaaa(d,a,l,k)*t1_aa(c,i)*r1_bb(b,j)
        sigma2_abab +=  0.50 * einsum('lkcd,dalk,ci,bj->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_abab*t2_abab(a,d,l,k)*t1_aa(c,i)*r1_bb(b,j)
        sigma2_abab += -0.50 * einsum('lkcd,adlk,ci,bj->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <k,l||c,d>_abab*t2_abab(a,d,k,l)*t1_aa(c,i)*r1_bb(b,j)
        sigma2_abab += -0.50 * einsum('klcd,adkl,ci,bj->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||d,c>_abab*t2_aaaa(d,a,i,k)*t1_bb(c,j)*r1_bb(b,l)
        sigma2_abab +=  1.00 * einsum('kldc,daik,cj,bl->abij', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||c,d>_bbbb*t2_abab(a,d,i,k)*t1_bb(c,j)*r1_bb(b,l)
        sigma2_abab += -1.00 * einsum('lkcd,adik,cj,bl->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||d,c>_abab*t2_abab(d,b,i,k)*t1_bb(c,j)*r1_aa(a,l)
        sigma2_abab +=  1.00 * einsum('lkdc,dbik,cj,al->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  1.00 <k,l||c,d>_abab*t2_abab(a,d,k,j)*t1_aa(c,i)*r1_bb(b,l)
        sigma2_abab +=  1.00 * einsum('klcd,adkj,ci,bl->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||c,d>_aaaa*t2_abab(d,b,k,j)*t1_aa(c,i)*r1_aa(a,l)
        sigma2_abab += -1.00 * einsum('lkcd,dbkj,ci,al->abij', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, r1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_abab*t2_bbbb(d,b,j,k)*t1_aa(c,i)*r1_aa(a,l)
        sigma2_abab +=  1.00 * einsum('lkcd,dbjk,ci,al->abij', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, r1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  0.50 <l,k||d,c>_abab*t2_abab(a,b,l,k)*t1_bb(c,j)*r1_aa(d,i)
        sigma2_abab +=  0.50 * einsum('lkdc,ablk,cj,di->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  0.50 <k,l||d,c>_abab*t2_abab(a,b,k,l)*t1_bb(c,j)*r1_aa(d,i)
        sigma2_abab +=  0.50 * einsum('kldc,abkl,cj,di->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,d>_abab*t2_abab(a,b,l,k)*t1_aa(c,i)*r1_bb(d,j)
        sigma2_abab +=  0.50 * einsum('lkcd,ablk,ci,dj->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  0.50 <k,l||c,d>_abab*t2_abab(a,b,k,l)*t1_aa(c,i)*r1_bb(d,j)
        sigma2_abab +=  0.50 * einsum('klcd,abkl,ci,dj->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, r1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||d,c>_abab*r1_aa(d,l)*t2_abab(a,b,i,k)*t1_bb(c,j)
        sigma2_abab += -1.00 * einsum('lkdc,dl,abik,cj->abij', g_abab[oa, ob, va, vb], r1_aa, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*r1_bb(d,l)*t2_abab(a,b,i,k)*t1_bb(c,j)
        sigma2_abab +=  1.00 * einsum('lkcd,dl,abik,cj->abij', g_bbbb[ob, ob, vb, vb], r1_bb, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*r1_aa(d,l)*t2_abab(a,b,k,j)*t1_aa(c,i)
        sigma2_abab +=  1.00 * einsum('lkcd,dl,abkj,ci->abij', g_aaaa[oa, oa, va, va], r1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 <k,l||c,d>_abab*r1_bb(d,l)*t2_abab(a,b,k,j)*t1_aa(c,i)
        sigma2_abab += -1.00 * einsum('klcd,dl,abkj,ci->abij', g_abab[oa, ob, va, vb], r1_bb, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_abab*t1_bb(b,k)*t2_abab(c,d,l,j)*r1_aa(a,i)
        sigma2_abab += -0.50 * einsum('lkcd,bk,cdlj,ai->abij', g_abab[oa, ob, va, vb], t1_bb, t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||d,c>_abab*t1_bb(b,k)*t2_abab(d,c,l,j)*r1_aa(a,i)
        sigma2_abab += -0.50 * einsum('lkdc,bk,dclj,ai->abij', g_abab[oa, ob, va, vb], t1_bb, t2_abab, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,d>_bbbb*t1_bb(b,k)*t2_bbbb(c,d,j,l)*r1_aa(a,i)
        sigma2_abab +=  0.50 * einsum('lkcd,bk,cdjl,ai->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,d>_aaaa*t1_aa(a,k)*t2_aaaa(c,d,i,l)*r1_bb(b,j)
        sigma2_abab +=  0.50 * einsum('lkcd,ak,cdil,bj->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <k,l||c,d>_abab*t1_aa(a,k)*t2_abab(c,d,i,l)*r1_bb(b,j)
        sigma2_abab += -0.50 * einsum('klcd,ak,cdil,bj->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <k,l||d,c>_abab*t1_aa(a,k)*t2_abab(d,c,i,l)*r1_bb(b,j)
        sigma2_abab += -0.50 * einsum('kldc,ak,dcil,bj->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <k,l||c,d>_abab*t1_aa(a,k)*t2_abab(c,d,i,j)*r1_bb(b,l)
        sigma2_abab +=  0.50 * einsum('klcd,ak,cdij,bl->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <k,l||d,c>_abab*t1_aa(a,k)*t2_abab(d,c,i,j)*r1_bb(b,l)
        sigma2_abab +=  0.50 * einsum('kldc,ak,dcij,bl->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <l,k||c,d>_abab*t1_bb(b,k)*t2_abab(c,d,i,j)*r1_aa(a,l)
        sigma2_abab +=  0.50 * einsum('lkcd,bk,cdij,al->abij', g_abab[oa, ob, va, vb], t1_bb, t2_abab, r1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        
        #	  0.50 <l,k||d,c>_abab*t1_bb(b,k)*t2_abab(d,c,i,j)*r1_aa(a,l)
        sigma2_abab +=  0.50 * einsum('lkdc,bk,dcij,al->abij', g_abab[oa, ob, va, vb], t1_bb, t2_abab, r1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        
        #	 -1.00 <l,k||c,d>_aaaa*t1_aa(a,k)*t2_abab(c,b,l,j)*r1_aa(d,i)
        sigma2_abab += -1.00 * einsum('lkcd,ak,cblj,di->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_abab, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||d,c>_abab*t1_aa(a,k)*t2_bbbb(c,b,j,l)*r1_aa(d,i)
        sigma2_abab +=  1.00 * einsum('kldc,ak,cbjl,di->abij', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||d,c>_abab*t2_abab(a,c,l,j)*t1_bb(b,k)*r1_aa(d,i)
        sigma2_abab +=  1.00 * einsum('lkdc,aclj,bk,di->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||c,d>_abab*t1_aa(a,k)*t2_abab(c,b,i,l)*r1_bb(d,j)
        sigma2_abab +=  1.00 * einsum('klcd,ak,cbil,dj->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_abab*t2_aaaa(c,a,i,l)*t1_bb(b,k)*r1_bb(d,j)
        sigma2_abab +=  1.00 * einsum('lkcd,cail,bk,dj->abij', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, r1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||c,d>_bbbb*t2_abab(a,c,i,l)*t1_bb(b,k)*r1_bb(d,j)
        sigma2_abab += -1.00 * einsum('lkcd,acil,bk,dj->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, r1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*r1_aa(d,l)*t1_aa(a,k)*t2_abab(c,b,i,j)
        sigma2_abab +=  1.00 * einsum('lkcd,dl,ak,cbij->abij', g_aaaa[oa, oa, va, va], r1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 <k,l||c,d>_abab*r1_bb(d,l)*t1_aa(a,k)*t2_abab(c,b,i,j)
        sigma2_abab += -1.00 * einsum('klcd,dl,ak,cbij->abij', g_abab[oa, ob, va, vb], r1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||d,c>_abab*r1_aa(d,l)*t2_abab(a,c,i,j)*t1_bb(b,k)
        sigma2_abab += -1.00 * einsum('lkdc,dl,acij,bk->abij', g_abab[oa, ob, va, vb], r1_aa, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*r1_bb(d,l)*t2_abab(a,c,i,j)*t1_bb(b,k)
        sigma2_abab +=  1.00 * einsum('lkcd,dl,acij,bk->abij', g_bbbb[ob, ob, vb, vb], r1_bb, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||i,j>_abab*t1_aa(a,k)*t1_bb(b,l)*r0
        sigma2_abab +=  1.00 * einsum('klij,ak,bl,->abij', g_abab[oa, ob, oa, ob], t1_aa, t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -1.00 <a,k||c,j>_abab*t1_bb(b,k)*t1_aa(c,i)*r0
        sigma2_abab += -1.00 * einsum('akcj,bk,ci,->abij', g_abab[va, ob, va, ob], t1_bb, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||c,j>_abab*t1_aa(a,k)*t1_aa(c,i)*r0
        sigma2_abab += -1.00 * einsum('kbcj,ak,ci,->abij', g_abab[oa, vb, va, ob], t1_aa, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -1.00 <a,k||i,c>_abab*t1_bb(b,k)*t1_bb(c,j)*r0
        sigma2_abab += -1.00 * einsum('akic,bk,cj,->abij', g_abab[va, ob, oa, vb], t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||i,c>_abab*t1_aa(a,k)*t1_bb(c,j)*r0
        sigma2_abab += -1.00 * einsum('kbic,ak,cj,->abij', g_abab[oa, vb, oa, vb], t1_aa, t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	  1.00 <a,b||d,c>_abab*t1_bb(c,j)*t1_aa(d,i)*r0
        sigma2_abab +=  1.00 * einsum('abdc,cj,di,->abij', g_abab[va, vb, va, vb], t1_bb, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -1.00 <k,l||c,j>_abab*t1_bb(b,l)*t1_aa(c,k)*r1_aa(a,i)
        sigma2_abab += -1.00 * einsum('klcj,bl,ck,ai->abij', g_abab[oa, ob, va, ob], t1_bb, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,j>_bbbb*t1_bb(b,l)*t1_bb(c,k)*r1_aa(a,i)
        sigma2_abab +=  1.00 * einsum('lkcj,bl,ck,ai->abij', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,i>_aaaa*t1_aa(a,l)*t1_aa(c,k)*r1_bb(b,j)
        sigma2_abab +=  1.00 * einsum('lkci,al,ck,bj->abij', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||i,c>_abab*t1_aa(a,l)*t1_bb(c,k)*r1_bb(b,j)
        sigma2_abab += -1.00 * einsum('lkic,al,ck,bj->abij', g_abab[oa, ob, oa, vb], t1_aa, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||c,j>_abab*t1_aa(a,k)*t1_aa(c,i)*r1_bb(b,l)
        sigma2_abab +=  1.00 * einsum('klcj,ak,ci,bl->abij', g_abab[oa, ob, va, ob], t1_aa, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,j>_abab*t1_bb(b,k)*t1_aa(c,i)*r1_aa(a,l)
        sigma2_abab +=  1.00 * einsum('lkcj,bk,ci,al->abij', g_abab[oa, ob, va, ob], t1_bb, t1_aa, r1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||i,c>_abab*t1_aa(a,k)*t1_bb(c,j)*r1_bb(b,l)
        sigma2_abab +=  1.00 * einsum('klic,ak,cj,bl->abij', g_abab[oa, ob, oa, vb], t1_aa, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||i,c>_abab*t1_bb(b,k)*t1_bb(c,j)*r1_aa(a,l)
        sigma2_abab +=  1.00 * einsum('lkic,bk,cj,al->abij', g_abab[oa, ob, oa, vb], t1_bb, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  1.00 <k,l||c,j>_abab*r1_aa(c,i)*t1_aa(a,k)*t1_bb(b,l)
        sigma2_abab +=  1.00 * einsum('klcj,ci,ak,bl->abij', g_abab[oa, ob, va, ob], r1_aa, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||i,c>_abab*r1_bb(c,j)*t1_aa(a,k)*t1_bb(b,l)
        sigma2_abab +=  1.00 * einsum('klic,cj,ak,bl->abij', g_abab[oa, ob, oa, vb], r1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <k,b||c,d>_abab*t1_aa(c,k)*t1_bb(d,j)*r1_aa(a,i)
        sigma2_abab +=  1.00 * einsum('kbcd,ck,dj,ai->abij', g_abab[oa, vb, va, vb], t1_aa, t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <k,b||c,d>_bbbb*t1_bb(c,k)*t1_bb(d,j)*r1_aa(a,i)
        sigma2_abab +=  1.00 * einsum('kbcd,ck,dj,ai->abij', g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <k,a||c,d>_aaaa*t1_aa(c,k)*t1_aa(d,i)*r1_bb(b,j)
        sigma2_abab +=  1.00 * einsum('kacd,ck,di,bj->abij', g_aaaa[oa, va, va, va], t1_aa, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <a,k||d,c>_abab*t1_bb(c,k)*t1_aa(d,i)*r1_bb(b,j)
        sigma2_abab +=  1.00 * einsum('akdc,ck,di,bj->abij', g_abab[va, ob, va, vb], t1_bb, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 <a,k||d,c>_abab*r1_bb(b,k)*t1_bb(c,j)*t1_aa(d,i)
        sigma2_abab += -1.00 * einsum('akdc,bk,cj,di->abij', g_abab[va, ob, va, vb], r1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <k,b||d,c>_abab*r1_aa(a,k)*t1_bb(c,j)*t1_aa(d,i)
        sigma2_abab += -1.00 * einsum('kbdc,ak,cj,di->abij', g_abab[oa, vb, va, vb], r1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 <a,k||d,c>_abab*t1_bb(b,k)*t1_bb(c,j)*r1_aa(d,i)
        sigma2_abab += -1.00 * einsum('akdc,bk,cj,di->abij', g_abab[va, ob, va, vb], t1_bb, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <k,b||d,c>_abab*t1_aa(a,k)*t1_bb(c,j)*r1_aa(d,i)
        sigma2_abab += -1.00 * einsum('kbdc,ak,cj,di->abij', g_abab[oa, vb, va, vb], t1_aa, t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 <a,k||c,d>_abab*t1_bb(b,k)*t1_aa(c,i)*r1_bb(d,j)
        sigma2_abab += -1.00 * einsum('akcd,bk,ci,dj->abij', g_abab[va, ob, va, vb], t1_bb, t1_aa, r1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <k,b||c,d>_abab*t1_aa(a,k)*t1_aa(c,i)*r1_bb(d,j)
        sigma2_abab += -1.00 * einsum('kbcd,ak,ci,dj->abij', g_abab[oa, vb, va, vb], t1_aa, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_aaaa*t1_aa(c,k)*t1_aa(d,l)*r2_abab(a,b,i,j)
        sigma2_abab += -0.50 * einsum('lkcd,ck,dl,abij->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <k,l||c,d>_abab*t1_aa(c,k)*t1_bb(d,l)*r2_abab(a,b,i,j)
        sigma2_abab +=  0.50 * einsum('klcd,ck,dl,abij->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <l,k||d,c>_abab*t1_bb(c,k)*t1_aa(d,l)*r2_abab(a,b,i,j)
        sigma2_abab +=  0.50 * einsum('lkdc,ck,dl,abij->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_bbbb*t1_bb(c,k)*t1_bb(d,l)*r2_abab(a,b,i,j)
        sigma2_abab += -0.50 * einsum('lkcd,ck,dl,abij->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 <k,l||c,d>_abab*t1_aa(c,k)*t1_bb(d,j)*r2_abab(a,b,i,l)
        sigma2_abab += -1.00 * einsum('klcd,ck,dj,abil->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*t1_bb(c,k)*t1_bb(d,j)*r2_abab(a,b,i,l)
        sigma2_abab +=  1.00 * einsum('lkcd,ck,dj,abil->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*t1_aa(c,k)*t1_aa(d,i)*r2_abab(a,b,l,j)
        sigma2_abab +=  1.00 * einsum('lkcd,ck,di,ablj->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -1.00 <l,k||d,c>_abab*t1_bb(c,k)*t1_aa(d,i)*r2_abab(a,b,l,j)
        sigma2_abab += -1.00 * einsum('lkdc,ck,di,ablj->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*t1_aa(a,l)*t1_aa(c,k)*r2_abab(d,b,i,j)
        sigma2_abab +=  1.00 * einsum('lkcd,al,ck,dbij->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||d,c>_abab*t1_aa(a,l)*t1_bb(c,k)*r2_abab(d,b,i,j)
        sigma2_abab += -1.00 * einsum('lkdc,al,ck,dbij->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	 -1.00 <k,l||c,d>_abab*t1_bb(b,l)*t1_aa(c,k)*r2_abab(a,d,i,j)
        sigma2_abab += -1.00 * einsum('klcd,bl,ck,adij->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*t1_bb(b,l)*t1_bb(c,k)*r2_abab(a,d,i,j)
        sigma2_abab +=  1.00 * einsum('lkcd,bl,ck,adij->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  0.50 <l,k||d,c>_abab*t1_bb(c,j)*t1_aa(d,i)*r2_abab(a,b,l,k)
        sigma2_abab +=  0.50 * einsum('lkdc,cj,di,ablk->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  0.50 <k,l||d,c>_abab*t1_bb(c,j)*t1_aa(d,i)*r2_abab(a,b,k,l)
        sigma2_abab +=  0.50 * einsum('kldc,cj,di,abkl->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 <k,l||d,c>_abab*t1_aa(a,k)*t1_bb(c,j)*r2_abab(d,b,i,l)
        sigma2_abab +=  1.00 * einsum('kldc,ak,cj,dbil->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||d,c>_abab*t1_bb(b,k)*t1_bb(c,j)*r2_aaaa(d,a,i,l)
        sigma2_abab +=  1.00 * einsum('lkdc,bk,cj,dail->abij', g_abab[oa, ob, va, vb], t1_bb, t1_bb, r2_aaaa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*t1_bb(b,k)*t1_bb(c,j)*r2_abab(a,d,i,l)
        sigma2_abab +=  1.00 * einsum('lkcd,bk,cj,adil->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*t1_aa(a,k)*t1_aa(c,i)*r2_abab(d,b,l,j)
        sigma2_abab +=  1.00 * einsum('lkcd,ak,ci,dblj->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 <k,l||c,d>_abab*t1_aa(a,k)*t1_aa(c,i)*r2_bbbb(d,b,j,l)
        sigma2_abab +=  1.00 * einsum('klcd,ak,ci,dbjl->abij', g_abab[oa, ob, va, vb], t1_aa, t1_aa, r2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_abab*t1_bb(b,k)*t1_aa(c,i)*r2_abab(a,d,l,j)
        sigma2_abab +=  1.00 * einsum('lkcd,bk,ci,adlj->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  0.50 <k,l||c,d>_abab*t1_aa(a,k)*t1_bb(b,l)*r2_abab(c,d,i,j)
        sigma2_abab +=  0.50 * einsum('klcd,ak,bl,cdij->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  0.50 <k,l||d,c>_abab*t1_aa(a,k)*t1_bb(b,l)*r2_abab(d,c,i,j)
        sigma2_abab +=  0.50 * einsum('kldc,ak,bl,dcij->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <k,l||c,d>_abab*r0*t2_abab(a,b,i,l)*t2_abab(c,d,k,j)
        sigma2_abab += -0.50 * einsum('klcd,,abil,cdkj->abij', g_abab[oa, ob, va, vb], r0, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	 -0.50 <k,l||d,c>_abab*r0*t2_abab(a,b,i,l)*t2_abab(d,c,k,j)
        sigma2_abab += -0.50 * einsum('kldc,,abil,dckj->abij', g_abab[oa, ob, va, vb], r0, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_bbbb*r0*t2_abab(a,b,i,l)*t2_bbbb(c,d,j,k)
        sigma2_abab += -0.50 * einsum('lkcd,,abil,cdjk->abij', g_bbbb[ob, ob, vb, vb], r0, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_aaaa*r0*t2_abab(a,b,l,j)*t2_aaaa(c,d,i,k)
        sigma2_abab += -0.50 * einsum('lkcd,,ablj,cdik->abij', g_aaaa[oa, oa, va, va], r0, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_abab*r0*t2_abab(a,b,l,j)*t2_abab(c,d,i,k)
        sigma2_abab += -0.50 * einsum('lkcd,,ablj,cdik->abij', g_abab[oa, ob, va, vb], r0, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||d,c>_abab*r0*t2_abab(a,b,l,j)*t2_abab(d,c,i,k)
        sigma2_abab += -0.50 * einsum('lkdc,,ablj,dcik->abij', g_abab[oa, ob, va, vb], r0, t2_abab, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	  0.250 <l,k||c,d>_abab*t2_abab(a,b,l,k)*t2_abab(c,d,i,j)*r0
        sigma2_abab +=  0.250 * einsum('lkcd,ablk,cdij,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	  0.250 <l,k||d,c>_abab*t2_abab(a,b,l,k)*t2_abab(d,c,i,j)*r0
        sigma2_abab +=  0.250 * einsum('lkdc,ablk,dcij,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	  0.250 <k,l||c,d>_abab*t2_abab(a,b,k,l)*t2_abab(c,d,i,j)*r0
        sigma2_abab +=  0.250 * einsum('klcd,abkl,cdij,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	  0.250 <k,l||d,c>_abab*t2_abab(a,b,k,l)*t2_abab(d,c,i,j)*r0
        sigma2_abab +=  0.250 * einsum('kldc,abkl,dcij,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_aaaa*t2_aaaa(c,a,l,k)*t2_abab(d,b,i,j)*r0
        sigma2_abab += -0.50 * einsum('lkcd,calk,dbij,->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_abab, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <l,k||d,c>_abab*t2_abab(a,c,l,k)*t2_abab(d,b,i,j)*r0
        sigma2_abab += -0.50 * einsum('lkdc,aclk,dbij,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <k,l||d,c>_abab*t2_abab(a,c,k,l)*t2_abab(d,b,i,j)*r0
        sigma2_abab += -0.50 * einsum('kldc,ackl,dbij,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 <k,l||d,c>_abab*r0*t2_abab(a,c,k,j)*t2_abab(d,b,i,l)
        sigma2_abab +=  1.00 * einsum('kldc,,ackj,dbil->abij', g_abab[oa, ob, va, vb], r0, t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*r0*t2_aaaa(c,a,i,k)*t2_abab(d,b,l,j)
        sigma2_abab +=  1.00 * einsum('lkcd,,caik,dblj->abij', g_aaaa[oa, oa, va, va], r0, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||c,d>_abab*r0*t2_aaaa(c,a,i,k)*t2_bbbb(d,b,j,l)
        sigma2_abab +=  1.00 * einsum('klcd,,caik,dbjl->abij', g_abab[oa, ob, va, vb], r0, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||d,c>_abab*r0*t2_abab(a,c,i,k)*t2_abab(d,b,l,j)
        sigma2_abab +=  1.00 * einsum('lkdc,,acik,dblj->abij', g_abab[oa, ob, va, vb], r0, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*r0*t2_abab(a,c,i,k)*t2_bbbb(d,b,j,l)
        sigma2_abab +=  1.00 * einsum('lkcd,,acik,dbjl->abij', g_bbbb[ob, ob, vb, vb], r0, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||d,c>_abab*t2_abab(a,c,i,j)*t2_abab(d,b,l,k)*r0
        sigma2_abab += -0.50 * einsum('lkdc,acij,dblk,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -0.50 <k,l||d,c>_abab*t2_abab(a,c,i,j)*t2_abab(d,b,k,l)*r0
        sigma2_abab += -0.50 * einsum('kldc,acij,dbkl,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  0.50 <l,k||c,d>_bbbb*t2_abab(a,c,i,j)*t2_bbbb(d,b,l,k)*r0
        sigma2_abab +=  0.50 * einsum('lkcd,acij,dblk,->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -1.00 <k,l||c,d>_abab*r0*t2_abab(a,b,i,l)*t1_aa(c,k)*t1_bb(d,j)
        sigma2_abab += -1.00 * einsum('klcd,,abil,ck,dj->abij', g_abab[oa, ob, va, vb], r0, t2_abab, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (2, 3), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*r0*t2_abab(a,b,i,l)*t1_bb(c,k)*t1_bb(d,j)
        sigma2_abab +=  1.00 * einsum('lkcd,,abil,ck,dj->abij', g_bbbb[ob, ob, vb, vb], r0, t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (2, 3), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*r0*t2_abab(a,b,l,j)*t1_aa(c,k)*t1_aa(d,i)
        sigma2_abab +=  1.00 * einsum('lkcd,,ablj,ck,di->abij', g_aaaa[oa, oa, va, va], r0, t2_abab, t1_aa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])
        
        #	 -1.00 <l,k||d,c>_abab*r0*t2_abab(a,b,l,j)*t1_bb(c,k)*t1_aa(d,i)
        sigma2_abab += -1.00 * einsum('lkdc,,ablj,ck,di->abij', g_abab[oa, ob, va, vb], r0, t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*r0*t1_aa(a,l)*t2_abab(d,b,i,j)*t1_aa(c,k)
        sigma2_abab +=  1.00 * einsum('lkcd,,al,dbij,ck->abij', g_aaaa[oa, oa, va, va], r0, t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 4), (1, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||d,c>_abab*r0*t1_aa(a,l)*t2_abab(d,b,i,j)*t1_bb(c,k)
        sigma2_abab += -1.00 * einsum('lkdc,,al,dbij,ck->abij', g_abab[oa, ob, va, vb], r0, t1_aa, t2_abab, t1_bb, optimize=['einsum_path', (0, 4), (1, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <k,l||c,d>_abab*r0*t2_abab(a,d,i,j)*t1_bb(b,l)*t1_aa(c,k)
        sigma2_abab += -1.00 * einsum('klcd,,adij,bl,ck->abij', g_abab[oa, ob, va, vb], r0, t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 4), (0, 2), (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*r0*t2_abab(a,d,i,j)*t1_bb(b,l)*t1_bb(c,k)
        sigma2_abab +=  1.00 * einsum('lkcd,,adij,bl,ck->abij', g_bbbb[ob, ob, vb, vb], r0, t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 4), (0, 2), (0, 1), (0, 1)])
        
        #	  0.50 <l,k||d,c>_abab*t2_abab(a,b,l,k)*t1_bb(c,j)*t1_aa(d,i)*r0
        sigma2_abab +=  0.50 * einsum('lkdc,ablk,cj,di,->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 3), (0, 1), (0, 1)])
        
        #	  0.50 <k,l||d,c>_abab*t2_abab(a,b,k,l)*t1_bb(c,j)*t1_aa(d,i)*r0
        sigma2_abab +=  0.50 * einsum('kldc,abkl,cj,di,->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_aa, r0, optimize=['einsum_path', (0, 2), (0, 3), (0, 1), (0, 1)])
        
        #	  1.00 <k,l||d,c>_abab*t1_aa(a,k)*t2_abab(d,b,i,l)*t1_bb(c,j)*r0
        sigma2_abab +=  1.00 * einsum('kldc,ak,dbil,cj,->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t1_bb, r0, optimize=['einsum_path', (0, 1), (1, 2), (1, 2), (0, 1)])
        
        #	  1.00 <l,k||d,c>_abab*t2_aaaa(d,a,i,l)*t1_bb(b,k)*t1_bb(c,j)*r0
        sigma2_abab +=  1.00 * einsum('lkdc,dail,bk,cj,->abij', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 3), (0, 3), (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*t2_abab(a,d,i,l)*t1_bb(b,k)*t1_bb(c,j)*r0
        sigma2_abab +=  1.00 * einsum('lkcd,adil,bk,cj,->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 3), (0, 3), (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*t1_aa(a,k)*t2_abab(d,b,l,j)*t1_aa(c,i)*r0
        sigma2_abab +=  1.00 * einsum('lkcd,ak,dblj,ci,->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_abab, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
        
        #	  1.00 <k,l||c,d>_abab*t1_aa(a,k)*t2_bbbb(d,b,j,l)*t1_aa(c,i)*r0
        sigma2_abab +=  1.00 * einsum('klcd,ak,dbjl,ci,->abij', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 3), (0, 1), (0, 1)])
        
        #	  1.00 <l,k||c,d>_abab*t2_abab(a,d,l,j)*t1_bb(b,k)*t1_aa(c,i)*r0
        sigma2_abab +=  1.00 * einsum('lkcd,adlj,bk,ci,->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
        
        #	  0.50 <k,l||c,d>_abab*t1_aa(a,k)*t1_bb(b,l)*t2_abab(c,d,i,j)*r0
        sigma2_abab +=  0.50 * einsum('klcd,ak,bl,cdij,->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
        
        #	  0.50 <k,l||d,c>_abab*t1_aa(a,k)*t1_bb(b,l)*t2_abab(d,c,i,j)*r0
        sigma2_abab +=  0.50 * einsum('kldc,ak,bl,dcij,->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
        
        #	  1.00 <k,l||c,j>_abab*r0*t1_aa(a,k)*t1_bb(b,l)*t1_aa(c,i)
        sigma2_abab +=  1.00 * einsum('klcj,,ak,bl,ci->abij', g_abab[oa, ob, va, ob], r0, t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
        
        #	  1.00 <k,l||i,c>_abab*r0*t1_aa(a,k)*t1_bb(b,l)*t1_bb(c,j)
        sigma2_abab +=  1.00 * einsum('klic,,ak,bl,cj->abij', g_abab[oa, ob, oa, vb], r0, t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1), (0, 1), (0, 1)])
        
        #	 -1.00 <a,k||d,c>_abab*r0*t1_bb(b,k)*t1_bb(c,j)*t1_aa(d,i)
        sigma2_abab += -1.00 * einsum('akdc,,bk,cj,di->abij', g_abab[va, ob, va, vb], r0, t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 1), (0, 1), (0, 1)])
        
        #	 -1.00 <k,b||d,c>_abab*r0*t1_aa(a,k)*t1_bb(c,j)*t1_aa(d,i)
        sigma2_abab += -1.00 * einsum('kbdc,,ak,cj,di->abij', g_abab[oa, vb, va, vb], r0, t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1), (1, 2), (0, 1)])
        
        #	 -1.00 <k,l||c,d>_abab*t1_bb(b,l)*t1_aa(c,k)*t1_bb(d,j)*r1_aa(a,i)
        sigma2_abab += -1.00 * einsum('klcd,bl,ck,dj,ai->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (1, 3), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_bbbb*t1_bb(b,l)*t1_bb(c,k)*t1_bb(d,j)*r1_aa(a,i)
        sigma2_abab +=  1.00 * einsum('lkcd,bl,ck,dj,ai->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, r1_aa, optimize=['einsum_path', (0, 2), (1, 3), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||c,d>_aaaa*t1_aa(a,l)*t1_aa(c,k)*t1_aa(d,i)*r1_bb(b,j)
        sigma2_abab +=  1.00 * einsum('lkcd,al,ck,di,bj->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
        
        #	 -1.00 <l,k||d,c>_abab*t1_aa(a,l)*t1_bb(c,k)*t1_aa(d,i)*r1_bb(b,j)
        sigma2_abab += -1.00 * einsum('lkdc,al,ck,di,bj->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||d,c>_abab*t1_aa(a,k)*t1_bb(c,j)*t1_aa(d,i)*r1_bb(b,l)
        sigma2_abab +=  1.00 * einsum('kldc,ak,cj,di,bl->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
        
        #	  1.00 <l,k||d,c>_abab*t1_bb(b,k)*t1_bb(c,j)*t1_aa(d,i)*r1_aa(a,l)
        sigma2_abab +=  1.00 * einsum('lkdc,bk,cj,di,al->abij', g_abab[oa, ob, va, vb], t1_bb, t1_bb, t1_aa, r1_aa, optimize=['einsum_path', (0, 2), (2, 3), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||d,c>_abab*t1_aa(a,k)*t1_bb(b,l)*t1_bb(c,j)*r1_aa(d,i)
        sigma2_abab +=  1.00 * einsum('kldc,ak,bl,cj,di->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_bb, r1_aa, optimize=['einsum_path', (0, 1), (1, 3), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||c,d>_abab*t1_aa(a,k)*t1_bb(b,l)*t1_aa(c,i)*r1_bb(d,j)
        sigma2_abab +=  1.00 * einsum('klcd,ak,bl,ci,dj->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, r1_bb, optimize=['einsum_path', (0, 1), (2, 3), (0, 2), (0, 1)])
        
        #	  1.00 <k,l||d,c>_abab*t1_aa(a,k)*t1_bb(b,l)*t1_bb(c,j)*t1_aa(d,i)*r0
        sigma2_abab +=  1.00 * einsum('kldc,ak,bl,cj,di,->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_bb, t1_aa, r0, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 2), (0, 1)])
        
        return sigma2_abab
    
    def sigma_doubles_bbbb(self, r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab):

        t1_aa = self.t1_aa
        t1_bb = self.t1_bb
        t2_aaaa = self.t2_aaaa
        t2_bbbb = self.t2_bbbb
        t2_abab = self.t2_abab
        f_aa = self.f_aa
        f_bb = self.f_bb
        g_aaaa = self.g_aaaa
        g_bbbb = self.g_bbbb
        g_abab = self.g_abab
        oa = self.oa
        va = self.va
        ob = self.ob
        vb = self.vb
    
        #	 -1.00 P(i,j)*P(a,b)f_bb(a,j)*r1_bb(b,i)
        contracted_intermediate = -1.00 * einsum('aj,bi->abij', f_bb[vb, ob], r1_bb)
        sigma2_bbbb =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 f_aa(k,k)*r2_bbbb(a,b,i,j)
        sigma2_bbbb +=  1.00 * einsum('kk,abij->abij', f_aa[oa, oa], r2_bbbb)
        
        #	  1.00 f_bb(k,k)*r2_bbbb(a,b,i,j)
        sigma2_bbbb +=  1.00 * einsum('kk,abij->abij', f_bb[ob, ob], r2_bbbb)
        
        #	 -1.00 P(i,j)f_bb(k,j)*r2_bbbb(a,b,i,k)
        contracted_intermediate = -1.00 * einsum('kj,abik->abij', f_bb[ob, ob], r2_bbbb)
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(a,b)f_bb(a,c)*r2_bbbb(c,b,i,j)
        contracted_intermediate =  1.00 * einsum('ac,cbij->abij', f_bb[vb, vb], r2_bbbb)
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)f_bb(k,j)*t1_bb(a,k)*r1_bb(b,i)
        contracted_intermediate =  1.00 * einsum('kj,ak,bi->abij', f_bb[ob, ob], t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)f_bb(a,c)*t1_bb(c,j)*r1_bb(b,i)
        contracted_intermediate = -1.00 * einsum('ac,cj,bi->abij', f_bb[vb, vb], t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 f_aa(k,c)*r2_bbbb(a,b,i,j)*t1_aa(c,k)
        sigma2_bbbb +=  1.00 * einsum('kc,abij,ck->abij', f_aa[oa, va], r2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  1.00 f_bb(k,c)*r2_bbbb(a,b,i,j)*t1_bb(c,k)
        sigma2_bbbb +=  1.00 * einsum('kc,abij,ck->abij', f_bb[ob, vb], r2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 P(i,j)f_bb(k,c)*r2_bbbb(a,b,i,k)*t1_bb(c,j)
        contracted_intermediate = -1.00 * einsum('kc,abik,cj->abij', f_bb[ob, vb], r2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)f_bb(k,c)*r2_bbbb(c,b,i,j)*t1_bb(a,k)
        contracted_intermediate = -1.00 * einsum('kc,cbij,ak->abij', f_bb[ob, vb], r2_bbbb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)f_bb(k,j)*t2_bbbb(a,b,i,k)*r0
        contracted_intermediate = -1.00 * einsum('kj,abik,->abij', f_bb[ob, ob], t2_bbbb, r0, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(a,b)f_bb(a,c)*t2_bbbb(c,b,i,j)*r0
        contracted_intermediate =  1.00 * einsum('ac,cbij,->abij', f_bb[vb, vb], t2_bbbb, r0, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)f_aa(k,c)*r1_bb(b,i)*t2_abab(c,a,k,j)
        contracted_intermediate = -1.00 * einsum('kc,bi,cakj->abij', f_aa[oa, va], r1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)f_bb(k,c)*r1_bb(b,i)*t2_bbbb(c,a,j,k)
        contracted_intermediate =  1.00 * einsum('kc,bi,cajk->abij', f_bb[ob, vb], r1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(a,b)f_bb(k,c)*r1_bb(b,k)*t2_bbbb(c,a,i,j)
        contracted_intermediate =  1.00 * einsum('kc,bk,caij->abij', f_bb[ob, vb], r1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)f_bb(k,c)*r1_bb(c,i)*t2_bbbb(a,b,j,k)
        contracted_intermediate =  1.00 * einsum('kc,ci,abjk->abij', f_bb[ob, vb], r1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)f_bb(k,c)*t2_bbbb(a,b,i,k)*t1_bb(c,j)*r0
        contracted_intermediate = -1.00 * einsum('kc,abik,cj,->abij', f_bb[ob, vb], t2_bbbb, t1_bb, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)f_bb(k,c)*t1_bb(a,k)*t2_bbbb(c,b,i,j)*r0
        contracted_intermediate = -1.00 * einsum('kc,ak,cbij,->abij', f_bb[ob, vb], t1_bb, t2_bbbb, r0, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)f_bb(k,c)*r1_bb(b,i)*t1_bb(a,k)*t1_bb(c,j)
        contracted_intermediate =  1.00 * einsum('kc,bi,ak,cj->abij', f_bb[ob, vb], r1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 <l,k||l,k>_aaaa*r2_bbbb(a,b,i,j)
        sigma2_bbbb += -0.50 * einsum('lklk,abij->abij', g_aaaa[oa, oa, oa, oa], r2_bbbb)
        
        #	 -0.50 <l,k||l,k>_abab*r2_bbbb(a,b,i,j)
        sigma2_bbbb += -0.50 * einsum('lklk,abij->abij', g_abab[oa, ob, oa, ob], r2_bbbb)
        
        #	 -0.50 <k,l||k,l>_abab*r2_bbbb(a,b,i,j)
        sigma2_bbbb += -0.50 * einsum('klkl,abij->abij', g_abab[oa, ob, oa, ob], r2_bbbb)
        
        #	 -0.50 <l,k||l,k>_bbbb*r2_bbbb(a,b,i,j)
        sigma2_bbbb += -0.50 * einsum('lklk,abij->abij', g_bbbb[ob, ob, ob, ob], r2_bbbb)
        
        #	  1.00 <a,b||i,j>_bbbb*r0
        sigma2_bbbb +=  1.00 * einsum('abij,->abij', g_bbbb[vb, vb, ob, ob], r0)
        
        #	  1.00 P(a,b)<k,a||i,j>_bbbb*r1_bb(b,k)
        contracted_intermediate =  1.00 * einsum('kaij,bk->abij', g_bbbb[ob, vb, ob, ob], r1_bb)
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)<a,b||c,j>_bbbb*r1_bb(c,i)
        contracted_intermediate =  1.00 * einsum('abcj,ci->abij', g_bbbb[vb, vb, vb, ob], r1_bb)
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 <l,k||i,j>_bbbb*r2_bbbb(a,b,l,k)
        sigma2_bbbb +=  0.50 * einsum('lkij,ablk->abij', g_bbbb[ob, ob, ob, ob], r2_bbbb)
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,j>_abab*r2_abab(c,b,k,i)
        contracted_intermediate = -1.00 * einsum('kacj,cbki->abij', g_abab[oa, vb, va, ob], r2_abab)
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,a||c,j>_bbbb*r2_bbbb(c,b,i,k)
        contracted_intermediate =  1.00 * einsum('kacj,cbik->abij', g_bbbb[ob, vb, vb, ob], r2_bbbb)
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 <a,b||c,d>_bbbb*r2_bbbb(c,d,i,j)
        sigma2_bbbb +=  0.50 * einsum('abcd,cdij->abij', g_bbbb[vb, vb, vb, vb], r2_bbbb)
        
        #	  1.00 P(a,b)<k,a||i,j>_bbbb*t1_bb(b,k)*r0
        contracted_intermediate =  1.00 * einsum('kaij,bk,->abij', g_bbbb[ob, vb, ob, ob], t1_bb, r0, optimize=['einsum_path', (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)<a,b||c,j>_bbbb*t1_bb(c,i)*r0
        contracted_intermediate =  1.00 * einsum('abcj,ci,->abij', g_bbbb[vb, vb, vb, ob], t1_bb, r0, optimize=['einsum_path', (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<l,k||i,j>_bbbb*r1_bb(b,l)*t1_bb(a,k)
        contracted_intermediate = -1.00 * einsum('lkij,bl,ak->abij', g_bbbb[ob, ob, ob, ob], r1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,j>_abab*r1_bb(b,i)*t1_aa(c,k)
        contracted_intermediate = -1.00 * einsum('kacj,bi,ck->abij', g_abab[oa, vb, va, ob], r1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,j>_bbbb*r1_bb(b,i)*t1_bb(c,k)
        contracted_intermediate = -1.00 * einsum('kacj,bi,ck->abij', g_bbbb[ob, vb, vb, ob], r1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,a||c,j>_bbbb*r1_bb(b,k)*t1_bb(c,i)
        contracted_intermediate =  1.00 * einsum('kacj,bk,ci->abij', g_bbbb[ob, vb, vb, ob], r1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,a||c,j>_bbbb*r1_bb(c,i)*t1_bb(b,k)
        contracted_intermediate =  1.00 * einsum('kacj,ci,bk->abij', g_bbbb[ob, vb, vb, ob], r1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<a,b||c,d>_bbbb*r1_bb(d,i)*t1_bb(c,j)
        contracted_intermediate = -1.00 * einsum('abcd,di,cj->abij', g_bbbb[vb, vb, vb, vb], r1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<k,l||c,j>_abab*t1_aa(c,k)*r2_bbbb(a,b,i,l)
        contracted_intermediate = -1.00 * einsum('klcj,ck,abil->abij', g_abab[oa, ob, va, ob], t1_aa, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,j>_bbbb*t1_bb(c,k)*r2_bbbb(a,b,i,l)
        contracted_intermediate =  1.00 * einsum('lkcj,ck,abil->abij', g_bbbb[ob, ob, vb, ob], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 P(i,j)<l,k||c,j>_bbbb*t1_bb(c,i)*r2_bbbb(a,b,l,k)
        contracted_intermediate =  0.50 * einsum('lkcj,ci,ablk->abij', g_bbbb[ob, ob, vb, ob], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,j>_abab*t1_bb(a,k)*r2_abab(c,b,l,i)
        contracted_intermediate =  1.00 * einsum('lkcj,ak,cbli->abij', g_abab[oa, ob, va, ob], t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>_bbbb*t1_bb(a,k)*r2_bbbb(c,b,i,l)
        contracted_intermediate = -1.00 * einsum('lkcj,ak,cbil->abij', g_bbbb[ob, ob, vb, ob], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(a,b)<k,a||c,d>_abab*t1_aa(c,k)*r2_bbbb(d,b,i,j)
        contracted_intermediate =  1.00 * einsum('kacd,ck,dbij->abij', g_abab[oa, vb, va, vb], t1_aa, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(a,b)<k,a||c,d>_bbbb*t1_bb(c,k)*r2_bbbb(d,b,i,j)
        contracted_intermediate =  1.00 * einsum('kacd,ck,dbij->abij', g_bbbb[ob, vb, vb, vb], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||d,c>_abab*t1_bb(c,j)*r2_abab(d,b,k,i)
        contracted_intermediate = -1.00 * einsum('kadc,cj,dbki->abij', g_abab[oa, vb, va, vb], t1_bb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>_bbbb*t1_bb(c,j)*r2_bbbb(d,b,i,k)
        contracted_intermediate = -1.00 * einsum('kacd,cj,dbik->abij', g_bbbb[ob, vb, vb, vb], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(a,b)<k,a||c,d>_bbbb*t1_bb(b,k)*r2_bbbb(c,d,i,j)
        contracted_intermediate =  0.50 * einsum('kacd,bk,cdij->abij', g_bbbb[ob, vb, vb, vb], t1_bb, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  0.50 <l,k||i,j>_bbbb*t2_bbbb(a,b,l,k)*r0
        sigma2_bbbb +=  0.50 * einsum('lkij,ablk,->abij', g_bbbb[ob, ob, ob, ob], t2_bbbb, r0, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,j>_abab*t2_abab(c,b,k,i)*r0
        contracted_intermediate = -1.00 * einsum('kacj,cbki,->abij', g_abab[oa, vb, va, ob], t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,a||c,j>_bbbb*t2_bbbb(c,b,i,k)*r0
        contracted_intermediate =  1.00 * einsum('kacj,cbik,->abij', g_bbbb[ob, vb, vb, ob], t2_bbbb, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 <a,b||c,d>_bbbb*t2_bbbb(c,d,i,j)*r0
        sigma2_bbbb +=  0.50 * einsum('abcd,cdij,->abij', g_bbbb[vb, vb, vb, vb], t2_bbbb, r0, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.50 P(i,j)*P(a,b)<l,k||c,j>_abab*t2_abab(c,a,l,k)*r1_bb(b,i)
        contracted_intermediate =  0.50 * einsum('lkcj,calk,bi->abij', g_abab[oa, ob, va, ob], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<k,l||c,j>_abab*t2_abab(c,a,k,l)*r1_bb(b,i)
        contracted_intermediate =  0.50 * einsum('klcj,cakl,bi->abij', g_abab[oa, ob, va, ob], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<l,k||c,j>_bbbb*t2_bbbb(c,a,l,k)*r1_bb(b,i)
        contracted_intermediate =  0.50 * einsum('lkcj,calk,bi->abij', g_bbbb[ob, ob, vb, ob], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,l||c,j>_abab*t2_abab(c,a,k,i)*r1_bb(b,l)
        contracted_intermediate = -1.00 * einsum('klcj,caki,bl->abij', g_abab[oa, ob, va, ob], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>_bbbb*t2_bbbb(c,a,i,k)*r1_bb(b,l)
        contracted_intermediate = -1.00 * einsum('lkcj,caik,bl->abij', g_bbbb[ob, ob, vb, ob], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)<l,k||c,j>_bbbb*t2_bbbb(a,b,l,k)*r1_bb(c,i)
        contracted_intermediate =  0.50 * einsum('lkcj,ablk,ci->abij', g_bbbb[ob, ob, vb, ob], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||c,j>_abab*t2_bbbb(a,b,i,k)*r1_aa(c,l)
        contracted_intermediate = -1.00 * einsum('lkcj,abik,cl->abij', g_abab[oa, ob, va, ob], t2_bbbb, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||c,j>_bbbb*t2_bbbb(a,b,i,k)*r1_bb(c,l)
        contracted_intermediate = -1.00 * einsum('lkcj,abik,cl->abij', g_bbbb[ob, ob, vb, ob], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)*P(a,b)<k,a||c,d>_abab*t2_abab(c,d,k,j)*r1_bb(b,i)
        contracted_intermediate = -0.50 * einsum('kacd,cdkj,bi->abij', g_abab[oa, vb, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)*P(a,b)<k,a||d,c>_abab*t2_abab(d,c,k,j)*r1_bb(b,i)
        contracted_intermediate = -0.50 * einsum('kadc,dckj,bi->abij', g_abab[oa, vb, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<k,a||c,d>_bbbb*t2_bbbb(c,d,j,k)*r1_bb(b,i)
        contracted_intermediate =  0.50 * einsum('kacd,cdjk,bi->abij', g_bbbb[ob, vb, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(a,b)<k,a||c,d>_bbbb*t2_bbbb(c,d,i,j)*r1_bb(b,k)
        contracted_intermediate =  0.50 * einsum('kacd,cdij,bk->abij', g_bbbb[ob, vb, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,a||c,d>_abab*t2_abab(c,b,k,j)*r1_bb(d,i)
        contracted_intermediate =  1.00 * einsum('kacd,cbkj,di->abij', g_abab[oa, vb, va, vb], t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>_bbbb*t2_bbbb(c,b,j,k)*r1_bb(d,i)
        contracted_intermediate = -1.00 * einsum('kacd,cbjk,di->abij', g_bbbb[ob, vb, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(a,b)<k,a||d,c>_abab*t2_bbbb(c,b,i,j)*r1_aa(d,k)
        contracted_intermediate =  1.00 * einsum('kadc,cbij,dk->abij', g_abab[oa, vb, va, vb], t2_bbbb, r1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<k,a||c,d>_bbbb*t2_bbbb(c,b,i,j)*r1_bb(d,k)
        contracted_intermediate = -1.00 * einsum('kacd,cbij,dk->abij', g_bbbb[ob, vb, vb, vb], t2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  0.250 <l,k||c,d>_aaaa*t2_aaaa(c,d,l,k)*r2_bbbb(a,b,i,j)
        sigma2_bbbb +=  0.250 * einsum('lkcd,cdlk,abij->abij', g_aaaa[oa, oa, va, va], t2_aaaa, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||c,d>_abab*t2_abab(c,d,l,k)*r2_bbbb(a,b,i,j)
        sigma2_bbbb +=  0.250 * einsum('lkcd,cdlk,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,l||c,d>_abab*t2_abab(c,d,k,l)*r2_bbbb(a,b,i,j)
        sigma2_bbbb +=  0.250 * einsum('klcd,cdkl,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||d,c>_abab*t2_abab(d,c,l,k)*r2_bbbb(a,b,i,j)
        sigma2_bbbb +=  0.250 * einsum('lkdc,dclk,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <k,l||d,c>_abab*t2_abab(d,c,k,l)*r2_bbbb(a,b,i,j)
        sigma2_bbbb +=  0.250 * einsum('kldc,dckl,abij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	  0.250 <l,k||c,d>_bbbb*t2_bbbb(c,d,l,k)*r2_bbbb(a,b,i,j)
        sigma2_bbbb +=  0.250 * einsum('lkcd,cdlk,abij->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 P(i,j)<k,l||c,d>_abab*t2_abab(c,d,k,j)*r2_bbbb(a,b,i,l)
        contracted_intermediate = -0.50 * einsum('klcd,cdkj,abil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)<k,l||d,c>_abab*t2_abab(d,c,k,j)*r2_bbbb(a,b,i,l)
        contracted_intermediate = -0.50 * einsum('kldc,dckj,abil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(c,d,j,k)*r2_bbbb(a,b,i,l)
        contracted_intermediate = -0.50 * einsum('lkcd,cdjk,abil->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.250 <l,k||c,d>_bbbb*t2_bbbb(c,d,i,j)*r2_bbbb(a,b,l,k)
        sigma2_bbbb +=  0.250 * einsum('lkcd,cdij,ablk->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        
        #	 -0.50 P(a,b)<l,k||c,d>_abab*t2_abab(c,a,l,k)*r2_bbbb(d,b,i,j)
        contracted_intermediate = -0.50 * einsum('lkcd,calk,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -0.50 P(a,b)<k,l||c,d>_abab*t2_abab(c,a,k,l)*r2_bbbb(d,b,i,j)
        contracted_intermediate = -0.50 * einsum('klcd,cakl,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -0.50 P(a,b)<l,k||c,d>_bbbb*t2_bbbb(c,a,l,k)*r2_bbbb(d,b,i,j)
        contracted_intermediate = -0.50 * einsum('lkcd,calk,dbij->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t2_abab(c,a,k,j)*r2_abab(d,b,l,i)
        contracted_intermediate =  1.00 * einsum('lkcd,cakj,dbli->abij', g_aaaa[oa, oa, va, va], t2_abab, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,l||c,d>_abab*t2_abab(c,a,k,j)*r2_bbbb(d,b,i,l)
        contracted_intermediate =  1.00 * einsum('klcd,cakj,dbil->abij', g_abab[oa, ob, va, vb], t2_abab, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||d,c>_abab*t2_bbbb(c,a,j,k)*r2_abab(d,b,l,i)
        contracted_intermediate =  1.00 * einsum('lkdc,cajk,dbli->abij', g_abab[oa, ob, va, vb], t2_bbbb, r2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t2_bbbb(c,a,j,k)*r2_bbbb(d,b,i,l)
        contracted_intermediate =  1.00 * einsum('lkcd,cajk,dbil->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(a,b)<l,k||d,c>_abab*t2_bbbb(c,a,i,j)*r2_abab(d,b,l,k)
        contracted_intermediate =  0.50 * einsum('lkdc,caij,dblk->abij', g_abab[oa, ob, va, vb], t2_bbbb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  0.50 P(a,b)<k,l||d,c>_abab*t2_bbbb(c,a,i,j)*r2_abab(d,b,k,l)
        contracted_intermediate =  0.50 * einsum('kldc,caij,dbkl->abij', g_abab[oa, ob, va, vb], t2_bbbb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -0.50 P(a,b)<l,k||c,d>_bbbb*t2_bbbb(c,a,i,j)*r2_bbbb(d,b,l,k)
        contracted_intermediate = -0.50 * einsum('lkcd,caij,dblk->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  0.250 <l,k||c,d>_bbbb*t2_bbbb(a,b,l,k)*r2_bbbb(c,d,i,j)
        sigma2_bbbb +=  0.250 * einsum('lkcd,ablk,cdij->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        
        #	  0.50 P(i,j)<l,k||c,d>_abab*t2_bbbb(a,b,j,k)*r2_abab(c,d,l,i)
        contracted_intermediate =  0.50 * einsum('lkcd,abjk,cdli->abij', g_abab[oa, ob, va, vb], t2_bbbb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 P(i,j)<l,k||d,c>_abab*t2_bbbb(a,b,j,k)*r2_abab(d,c,l,i)
        contracted_intermediate =  0.50 * einsum('lkdc,abjk,dcli->abij', g_abab[oa, ob, va, vb], t2_bbbb, r2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(a,b,j,k)*r2_bbbb(c,d,i,l)
        contracted_intermediate = -0.50 * einsum('lkcd,abjk,cdil->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, r2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<k,l||c,j>_abab*t2_bbbb(a,b,i,l)*t1_aa(c,k)*r0
        contracted_intermediate = -1.00 * einsum('klcj,abil,ck,->abij', g_abab[oa, ob, va, ob], t2_bbbb, t1_aa, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,j>_bbbb*t2_bbbb(a,b,i,l)*t1_bb(c,k)*r0
        contracted_intermediate =  1.00 * einsum('lkcj,abil,ck,->abij', g_bbbb[ob, ob, vb, ob], t2_bbbb, t1_bb, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 P(i,j)<l,k||c,j>_bbbb*t2_bbbb(a,b,l,k)*t1_bb(c,i)*r0
        contracted_intermediate =  0.50 * einsum('lkcj,ablk,ci,->abij', g_bbbb[ob, ob, vb, ob], t2_bbbb, t1_bb, r0, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,j>_abab*t1_bb(a,k)*t2_abab(c,b,l,i)*r0
        contracted_intermediate =  1.00 * einsum('lkcj,ak,cbli,->abij', g_abab[oa, ob, va, ob], t1_bb, t2_abab, r0, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>_bbbb*t1_bb(a,k)*t2_bbbb(c,b,i,l)*r0
        contracted_intermediate = -1.00 * einsum('lkcj,ak,cbil,->abij', g_bbbb[ob, ob, vb, ob], t1_bb, t2_bbbb, r0, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(a,b)<k,a||c,d>_abab*t2_bbbb(d,b,i,j)*t1_aa(c,k)*r0
        contracted_intermediate =  1.00 * einsum('kacd,dbij,ck,->abij', g_abab[oa, vb, va, vb], t2_bbbb, t1_aa, r0, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(a,b)<k,a||c,d>_bbbb*t2_bbbb(d,b,i,j)*t1_bb(c,k)*r0
        contracted_intermediate =  1.00 * einsum('kacd,dbij,ck,->abij', g_bbbb[ob, vb, vb, vb], t2_bbbb, t1_bb, r0, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||d,c>_abab*t2_abab(d,b,k,i)*t1_bb(c,j)*r0
        contracted_intermediate = -1.00 * einsum('kadc,dbki,cj,->abij', g_abab[oa, vb, va, vb], t2_abab, t1_bb, r0, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>_bbbb*t2_bbbb(d,b,i,k)*t1_bb(c,j)*r0
        contracted_intermediate = -1.00 * einsum('kacd,dbik,cj,->abij', g_bbbb[ob, vb, vb, vb], t2_bbbb, t1_bb, r0, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(a,b)<k,a||c,d>_bbbb*t1_bb(b,k)*t2_bbbb(c,d,i,j)*r0
        contracted_intermediate =  0.50 * einsum('kacd,bk,cdij,->abij', g_bbbb[ob, vb, vb, vb], t1_bb, t2_bbbb, r0, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t2_abab(d,a,l,j)*t1_aa(c,k)*r1_bb(b,i)
        contracted_intermediate =  1.00 * einsum('lkcd,dalj,ck,bi->abij', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||d,c>_abab*t2_abab(d,a,l,j)*t1_bb(c,k)*r1_bb(b,i)
        contracted_intermediate = -1.00 * einsum('lkdc,dalj,ck,bi->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,l||c,d>_abab*t2_bbbb(d,a,j,l)*t1_aa(c,k)*r1_bb(b,i)
        contracted_intermediate =  1.00 * einsum('klcd,dajl,ck,bi->abij', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t2_bbbb(d,a,j,l)*t1_bb(c,k)*r1_bb(b,i)
        contracted_intermediate = -1.00 * einsum('lkcd,dajl,ck,bi->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(a,b)<k,l||c,d>_abab*t2_bbbb(d,a,i,j)*t1_aa(c,k)*r1_bb(b,l)
        contracted_intermediate =  1.00 * einsum('klcd,daij,ck,bl->abij', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<l,k||c,d>_bbbb*t2_bbbb(d,a,i,j)*t1_bb(c,k)*r1_bb(b,l)
        contracted_intermediate = -1.00 * einsum('lkcd,daij,ck,bl->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)<k,l||c,d>_abab*t2_bbbb(a,b,j,l)*t1_aa(c,k)*r1_bb(d,i)
        contracted_intermediate =  1.00 * einsum('klcd,abjl,ck,di->abij', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(a,b,j,l)*t1_bb(c,k)*r1_bb(d,i)
        contracted_intermediate = -1.00 * einsum('lkcd,abjl,ck,di->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<l,k||d,c>_abab*t2_abab(d,a,l,k)*t1_bb(c,j)*r1_bb(b,i)
        contracted_intermediate =  0.50 * einsum('lkdc,dalk,cj,bi->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<k,l||d,c>_abab*t2_abab(d,a,k,l)*t1_bb(c,j)*r1_bb(b,i)
        contracted_intermediate =  0.50 * einsum('kldc,dakl,cj,bi->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t2_bbbb(d,a,l,k)*t1_bb(c,j)*r1_bb(b,i)
        contracted_intermediate = -0.50 * einsum('lkcd,dalk,cj,bi->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,l||d,c>_abab*t2_abab(d,a,k,i)*t1_bb(c,j)*r1_bb(b,l)
        contracted_intermediate = -1.00 * einsum('kldc,daki,cj,bl->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t2_bbbb(d,a,i,k)*t1_bb(c,j)*r1_bb(b,l)
        contracted_intermediate =  1.00 * einsum('lkcd,daik,cj,bl->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(a,b,l,k)*t1_bb(c,j)*r1_bb(d,i)
        contracted_intermediate = -0.50 * einsum('lkcd,ablk,cj,di->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||d,c>_abab*t2_bbbb(a,b,i,k)*t1_bb(c,j)*r1_aa(d,l)
        contracted_intermediate = -1.00 * einsum('lkdc,abik,cj,dl->abij', g_abab[oa, ob, va, vb], t2_bbbb, t1_bb, r1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(a,b,i,k)*t1_bb(c,j)*r1_bb(d,l)
        contracted_intermediate =  1.00 * einsum('lkcd,abik,cj,dl->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, r1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<l,k||c,d>_abab*t1_bb(a,k)*t2_abab(c,d,l,j)*r1_bb(b,i)
        contracted_intermediate =  0.50 * einsum('lkcd,ak,cdlj,bi->abij', g_abab[oa, ob, va, vb], t1_bb, t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  0.50 P(i,j)*P(a,b)<l,k||d,c>_abab*t1_bb(a,k)*t2_abab(d,c,l,j)*r1_bb(b,i)
        contracted_intermediate =  0.50 * einsum('lkdc,ak,dclj,bi->abij', g_abab[oa, ob, va, vb], t1_bb, t2_abab, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t1_bb(a,k)*t2_bbbb(c,d,j,l)*r1_bb(b,i)
        contracted_intermediate = -0.50 * einsum('lkcd,ak,cdjl,bi->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 P(a,b)<l,k||c,d>_bbbb*t1_bb(a,k)*t2_bbbb(c,d,i,j)*r1_bb(b,l)
        contracted_intermediate = -0.50 * einsum('lkcd,ak,cdij,bl->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,d>_abab*t1_bb(a,k)*t2_abab(c,b,l,j)*r1_bb(d,i)
        contracted_intermediate = -1.00 * einsum('lkcd,ak,cblj,di->abij', g_abab[oa, ob, va, vb], t1_bb, t2_abab, r1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t1_bb(a,k)*t2_bbbb(c,b,j,l)*r1_bb(d,i)
        contracted_intermediate =  1.00 * einsum('lkcd,ak,cbjl,di->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, r1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<l,k||d,c>_abab*t1_bb(a,k)*t2_bbbb(c,b,i,j)*r1_aa(d,l)
        contracted_intermediate = -1.00 * einsum('lkdc,ak,cbij,dl->abij', g_abab[oa, ob, va, vb], t1_bb, t2_bbbb, r1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(a,b)<l,k||c,d>_bbbb*t1_bb(a,k)*t2_bbbb(c,b,i,j)*r1_bb(d,l)
        contracted_intermediate =  1.00 * einsum('lkcd,ak,cbij,dl->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, r1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 <l,k||i,j>_bbbb*t1_bb(a,k)*t1_bb(b,l)*r0
        sigma2_bbbb += -1.00 * einsum('lkij,ak,bl,->abij', g_bbbb[ob, ob, ob, ob], t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	  1.00 P(i,j)*P(a,b)<k,a||c,j>_bbbb*t1_bb(b,k)*t1_bb(c,i)*r0
        contracted_intermediate =  1.00 * einsum('kacj,bk,ci,->abij', g_bbbb[ob, vb, vb, ob], t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 <a,b||c,d>_bbbb*t1_bb(c,j)*t1_bb(d,i)*r0
        sigma2_bbbb += -1.00 * einsum('abcd,cj,di,->abij', g_bbbb[vb, vb, vb, vb], t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        
        #	  1.00 P(i,j)*P(a,b)<k,l||c,j>_abab*t1_bb(a,l)*t1_aa(c,k)*r1_bb(b,i)
        contracted_intermediate =  1.00 * einsum('klcj,al,ck,bi->abij', g_abab[oa, ob, va, ob], t1_bb, t1_aa, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>_bbbb*t1_bb(a,l)*t1_bb(c,k)*r1_bb(b,i)
        contracted_intermediate = -1.00 * einsum('lkcj,al,ck,bi->abij', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>_bbbb*t1_bb(a,k)*t1_bb(c,i)*r1_bb(b,l)
        contracted_intermediate = -1.00 * einsum('lkcj,ak,ci,bl->abij', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)<l,k||c,j>_bbbb*t1_bb(a,k)*t1_bb(b,l)*r1_bb(c,i)
        contracted_intermediate = -1.00 * einsum('lkcj,ak,bl,ci->abij', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>_abab*t1_aa(c,k)*t1_bb(d,j)*r1_bb(b,i)
        contracted_intermediate = -1.00 * einsum('kacd,ck,dj,bi->abij', g_abab[oa, vb, va, vb], t1_aa, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>_bbbb*t1_bb(c,k)*t1_bb(d,j)*r1_bb(b,i)
        contracted_intermediate = -1.00 * einsum('kacd,ck,dj,bi->abij', g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<k,a||c,d>_bbbb*t1_bb(c,j)*t1_bb(d,i)*r1_bb(b,k)
        contracted_intermediate = -1.00 * einsum('kacd,cj,di,bk->abij', g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>_bbbb*t1_bb(b,k)*t1_bb(c,j)*r1_bb(d,i)
        contracted_intermediate = -1.00 * einsum('kacd,bk,cj,di->abij', g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 <l,k||c,d>_aaaa*t1_aa(c,k)*t1_aa(d,l)*r2_bbbb(a,b,i,j)
        sigma2_bbbb += -0.50 * einsum('lkcd,ck,dl,abij->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <k,l||c,d>_abab*t1_aa(c,k)*t1_bb(d,l)*r2_bbbb(a,b,i,j)
        sigma2_bbbb +=  0.50 * einsum('klcd,ck,dl,abij->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  0.50 <l,k||d,c>_abab*t1_bb(c,k)*t1_aa(d,l)*r2_bbbb(a,b,i,j)
        sigma2_bbbb +=  0.50 * einsum('lkdc,ck,dl,abij->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_bbbb*t1_bb(c,k)*t1_bb(d,l)*r2_bbbb(a,b,i,j)
        sigma2_bbbb += -0.50 * einsum('lkcd,ck,dl,abij->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	 -1.00 P(i,j)<k,l||c,d>_abab*t1_aa(c,k)*t1_bb(d,j)*r2_bbbb(a,b,i,l)
        contracted_intermediate = -1.00 * einsum('klcd,ck,dj,abil->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,d>_bbbb*t1_bb(c,k)*t1_bb(d,j)*r2_bbbb(a,b,i,l)
        contracted_intermediate =  1.00 * einsum('lkcd,ck,dj,abil->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<k,l||c,d>_abab*t1_bb(a,l)*t1_aa(c,k)*r2_bbbb(d,b,i,j)
        contracted_intermediate = -1.00 * einsum('klcd,al,ck,dbij->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, r2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(a,b)<l,k||c,d>_bbbb*t1_bb(a,l)*t1_bb(c,k)*r2_bbbb(d,b,i,j)
        contracted_intermediate =  1.00 * einsum('lkcd,al,ck,dbij->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -0.50 <l,k||c,d>_bbbb*t1_bb(c,j)*t1_bb(d,i)*r2_bbbb(a,b,l,k)
        sigma2_bbbb += -0.50 * einsum('lkcd,cj,di,ablk->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
        
        #	  1.00 P(i,j)*P(a,b)<l,k||d,c>_abab*t1_bb(a,k)*t1_bb(c,j)*r2_abab(d,b,l,i)
        contracted_intermediate =  1.00 * einsum('lkdc,ak,cj,dbli->abij', g_abab[oa, ob, va, vb], t1_bb, t1_bb, r2_abab, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t1_bb(a,k)*t1_bb(c,j)*r2_bbbb(d,b,i,l)
        contracted_intermediate =  1.00 * einsum('lkcd,ak,cj,dbil->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 <l,k||c,d>_bbbb*t1_bb(a,k)*t1_bb(b,l)*r2_bbbb(c,d,i,j)
        sigma2_bbbb += -0.50 * einsum('lkcd,ak,bl,cdij->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, r2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
        
        #	 -0.50 P(i,j)<k,l||c,d>_abab*t2_bbbb(a,b,i,l)*t2_abab(c,d,k,j)*r0
        contracted_intermediate = -0.50 * einsum('klcd,abil,cdkj,->abij', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)<k,l||d,c>_abab*t2_bbbb(a,b,i,l)*t2_abab(d,c,k,j)*r0
        contracted_intermediate = -0.50 * einsum('kldc,abil,dckj,->abij', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -0.50 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(a,b,i,l)*t2_bbbb(c,d,j,k)*r0
        contracted_intermediate = -0.50 * einsum('lkcd,abil,cdjk,->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.250 <l,k||c,d>_bbbb*t2_bbbb(a,b,l,k)*t2_bbbb(c,d,i,j)*r0
        sigma2_bbbb +=  0.250 * einsum('lkcd,ablk,cdij,->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_abab*t2_abab(c,a,l,k)*t2_bbbb(d,b,i,j)*r0
        sigma2_bbbb += -0.50 * einsum('lkcd,calk,dbij,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <k,l||c,d>_abab*t2_abab(c,a,k,l)*t2_bbbb(d,b,i,j)*r0
        sigma2_bbbb += -0.50 * einsum('klcd,cakl,dbij,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_bbbb*t2_bbbb(c,a,l,k)*t2_bbbb(d,b,i,j)*r0
        sigma2_bbbb += -0.50 * einsum('lkcd,calk,dbij,->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, r0, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
        
        #	  1.00 P(i,j)<l,k||c,d>_aaaa*t2_abab(c,a,k,j)*t2_abab(d,b,l,i)*r0
        contracted_intermediate =  1.00 * einsum('lkcd,cakj,dbli,->abij', g_aaaa[oa, oa, va, va], t2_abab, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<k,l||c,d>_abab*t2_abab(c,a,k,j)*t2_bbbb(d,b,i,l)*r0
        contracted_intermediate =  1.00 * einsum('klcd,cakj,dbil,->abij', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||d,c>_abab*t2_bbbb(c,a,j,k)*t2_abab(d,b,l,i)*r0
        contracted_intermediate =  1.00 * einsum('lkdc,cajk,dbli,->abij', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(c,a,j,k)*t2_bbbb(d,b,i,l)*r0
        contracted_intermediate =  1.00 * einsum('lkcd,cajk,dbil,->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, r0, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  0.50 <l,k||d,c>_abab*t2_bbbb(c,a,i,j)*t2_abab(d,b,l,k)*r0
        sigma2_bbbb +=  0.50 * einsum('lkdc,caij,dblk,->abij', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	  0.50 <k,l||d,c>_abab*t2_bbbb(c,a,i,j)*t2_abab(d,b,k,l)*r0
        sigma2_bbbb +=  0.50 * einsum('kldc,caij,dbkl,->abij', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -0.50 <l,k||c,d>_bbbb*t2_bbbb(c,a,i,j)*t2_bbbb(d,b,l,k)*r0
        sigma2_bbbb += -0.50 * einsum('lkcd,caij,dblk,->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, r0, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
        
        #	 -1.00 P(i,j)<k,l||c,d>_abab*t2_bbbb(a,b,i,l)*t1_aa(c,k)*t1_bb(d,j)*r0
        contracted_intermediate = -1.00 * einsum('klcd,abil,ck,dj,->abij', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, t1_bb, r0, optimize=['einsum_path', (0, 2), (1, 3), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(a,b,i,l)*t1_bb(c,k)*t1_bb(d,j)*r0
        contracted_intermediate =  1.00 * einsum('lkcd,abil,ck,dj,->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 2), (1, 3), (1, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<k,l||c,d>_abab*t1_bb(a,l)*t2_bbbb(d,b,i,j)*t1_aa(c,k)*r0
        contracted_intermediate = -1.00 * einsum('klcd,al,dbij,ck,->abij', g_abab[oa, ob, va, vb], t1_bb, t2_bbbb, t1_aa, r0, optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(a,b)<l,k||c,d>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,i,j)*t1_bb(c,k)*r0
        contracted_intermediate =  1.00 * einsum('lkcd,al,dbij,ck,->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t1_bb, r0, optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	 -0.50 <l,k||c,d>_bbbb*t2_bbbb(a,b,l,k)*t1_bb(c,j)*t1_bb(d,i)*r0
        sigma2_bbbb += -0.50 * einsum('lkcd,ablk,cj,di,->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 2), (1, 2), (1, 2), (0, 1)])
        
        #	  1.00 P(i,j)*P(a,b)<l,k||d,c>_abab*t1_bb(a,k)*t2_abab(d,b,l,i)*t1_bb(c,j)*r0
        contracted_intermediate =  1.00 * einsum('lkdc,ak,dbli,cj,->abij', g_abab[oa, ob, va, vb], t1_bb, t2_abab, t1_bb, r0, optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t1_bb(a,k)*t2_bbbb(d,b,i,l)*t1_bb(c,j)*r0
        contracted_intermediate =  1.00 * einsum('lkcd,ak,dbil,cj,->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t1_bb, r0, optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -0.50 <l,k||c,d>_bbbb*t1_bb(a,k)*t1_bb(b,l)*t2_bbbb(c,d,i,j)*r0
        sigma2_bbbb += -0.50 * einsum('lkcd,ak,bl,cdij,->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t2_bbbb, r0, optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])
        
        #	 -1.00 P(i,j)<l,k||c,j>_bbbb*t1_bb(a,k)*t1_bb(b,l)*t1_bb(c,i)*r0
        contracted_intermediate = -1.00 * einsum('lkcj,ak,bl,ci,->abij', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 3), (0, 2), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	 -1.00 P(a,b)<k,a||c,d>_bbbb*t1_bb(b,k)*t1_bb(c,j)*t1_bb(d,i)*r0
        contracted_intermediate = -1.00 * einsum('kacd,bk,cj,di,->abij', g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 2), (0, 2), (0, 1), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)*P(a,b)<k,l||c,d>_abab*t1_bb(a,l)*t1_aa(c,k)*t1_bb(d,j)*r1_bb(b,i)
        contracted_intermediate =  1.00 * einsum('klcd,al,ck,dj,bi->abij', g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (1, 3), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	 -1.00 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t1_bb(a,l)*t1_bb(c,k)*t1_bb(d,j)*r1_bb(b,i)
        contracted_intermediate = -1.00 * einsum('lkcd,al,ck,dj,bi->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (1, 3), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
        
        #	  1.00 P(a,b)<l,k||c,d>_bbbb*t1_bb(a,k)*t1_bb(c,j)*t1_bb(d,i)*r1_bb(b,l)
        contracted_intermediate =  1.00 * einsum('lkcd,ak,cj,di,bl->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 2), (1, 3), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
        
        #	  1.00 P(i,j)<l,k||c,d>_bbbb*t1_bb(a,k)*t1_bb(b,l)*t1_bb(c,j)*r1_bb(d,i)
        contracted_intermediate =  1.00 * einsum('lkcd,ak,bl,cj,di->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, r1_bb, optimize=['einsum_path', (0, 3), (2, 3), (0, 2), (0, 1)])
        sigma2_bbbb +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
        
        #	  1.00 <l,k||c,d>_bbbb*t1_bb(a,k)*t1_bb(b,l)*t1_bb(c,j)*t1_bb(d,i)*r0
        sigma2_bbbb +=  1.00 * einsum('lkcd,ak,bl,cj,di,->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, t1_bb, r0, optimize=['einsum_path', (0, 3), (0, 3), (1, 2), (0, 2), (0, 1)])
        
        return sigma2_bbbb
    
