<img src="https://render.githubusercontent.com/render/math?math=p^{\dagger}q">

pdaggerq is a fermionic computer algebra package for bringing strings of creation / annihilation operators to normal
order with respect to a true vacuum or Fermi vacuum. The code also evalutes projections used in coupled cluster theory
and can be used to generate full working coupled cluster codes. In the examples section we provide worked examples that
generate  CCSD, lambda-CCSD, CC3, CCSDT, and CCSDTQ equations.

## Installation
Installing pdaggerq requires cmake is installed on your system.  To install first clone the package

```
git clone git@github.com:edeprince3/pdaggerq.git
```

Then install like a normal python package.  From the package top level directory run either

```
python setup.py install
```
or 
```
pip install .
```
which should compile pdaggerq.  These commands will produce `build` and`dist` folders which contain
the compiled c++ shared library. If you are a developer and make changes to the c++ code the package can 
be rebuilt by running 
```
python setup.py clean; python setup.py install
```

## Quickstart

The following is an example that generates the energy expression for CCSD. For the sake of
readability, commutators that evaluate to zero are excluded.

Python:

    import pdaggerq

    pq = pdaggerq.pq_helper("fermi")

    print('')
    print('    < 0 | e(-T) H e(T) | 0> :')
    print('')

    # one-electron part: 
    
    # f
    pq.add_operator_product(1.0,['f'])

    # [f, T1]
    pq.add_commutator(1.0,['f'],['t1'])

    # [f, T2]
    pq.add_commutator(1.0,['f'],['t2'])

    # two-electron part: 

    # v
    pq.add_operator_product(1.0,['v'])

    # [v, T1]
    pq.add_commutator(1.0,['v'],['t1'])

    # [v, T2]
    pq.add_commutator(1.0,['v'],['t2(a,b,i,j)'])

    # [[v, T1], T1]]
    pq.add_double_commutator(0.5, ['v'],['t1'],['t1'])

    pq.simplify()
    pq.print_fully_contracted()
    pq.clear()

Output:

    < 0 | e(-T) H e(T) | 0> :


    // fully-contracted strings:
    //     + 1.00000 f(i,i) 
    //     + 1.00000 f(i,a) t1(a,i) 
    //     - 0.50000 <i,j||i,j> 
    //     - 0.25000 <i,j||a,b> t2(a,b,j,i) 
    //     + 0.50000 <i,j||a,b> t1(a,i) t1(b,j)

The same result can be generated using the add_st_operator command. Here a different mechanism is used to print
the strings:

    import pdaggerq

    pq = pdaggerq.pq_helper("fermi")

    print('')
    print('    < 0 | e(-T) H e(T) | 0> :')
    print('')
    
    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])
    
    pq.simplify()
    energy_terms = pq.fully_contracted_strings()
    for my_term in energy_terms:
        print(my_term)
        
    pq.clear()

Output:

    < 0 | e(-T) H e(T) | 0> :

    ['+1.000000', 'f(i,i)']
    ['+1.000000', 'f(i,a)', 't1(a,i)']
    ['-0.500000', '<i,j||i,j>']
    ['-0.250000', '<i,j||a,b>', 't2(a,b,j,i)']
    ['+0.500000', '<i,j||a,b>', 't1(a,i)', 't1(b,j)']

## How to contribute
We'd love to accept your contributions and patches. All submissions, including submissions by project members, require review. 
We use GitHub pull requests for this purpose. Consult GitHub Help for more information on using pull requests. 
Furthermore, please make sure your new code comes with extensive tests!
Make sure you adhere to our style guide. Just have a look at our code for clues. 
We mostly follow PEP 8 and use the corresponding pylint to check.
Code should always come with documentation.

We use Github issues for tracking requests and bugs. 

## How to cite

When using pdaggerq for research projects, please cite:

```
@misc{pdaggerq_2021,
    author       = {A. Eugene DePrince and Nicholas C. Rubin},
    title        = {{pdaggerq}: https://github.com/edeprince3/pdaggerq},
    month        = {June},
    year         = {2021},
    url          = {https://github.com/edeprince3/pdaggerq} 
}
```

## Methods and Functionality

### Normal Order

Normal order may be defined relative to the true vacuum or the fermi vacuum. This selection is made when creating the
pq_helper class:

```
# true vacuum
pq = pdaggerq.pq_helper("")
```
or
```
# true vacuum
pq = pdaggerq.pq_helper("true")
```
or    
```
# fermi vacuum
pq = pdaggerq.pq_helper("fermi")
```

Note than full functionality is not yet available for manual string specification when normal order is defined relative
to the fermi vacuum. In this case, it is better to use the functions defined below that add complete strings in a single
command.

### Label convention

We follow the usual convention for labeling orbitals: i, j, k, l, m, and n represent occupied orbitals and a, b, c, d,
e, and f represent virtual orbitals. Additionally, any label starting with i or a will be considered occupied or
virtual, respectively (e.g., i_1 or a2). All other labels are considered general labels. Delta functions involving
occupied / virtual combinations will be set to zero. When normal order is defined relative to the fermi vacuum, sums
involving general labels are split into sums involving occupied and virtual orbitals using internal labels o1, o2, o3,
and o4 (occupied) or v1, v2, v3, and v4 (virtual). So, we recommend avoiding using these labels when specifying any
other components of your strings.

Orbital labels refer to spin orbitals. There is functionality to integrate out spin in final expressions, [see below](#spin).

### Methods
Strings are defined in Python using the pq_helper class, which has the following functions:

#### add_operator_product: 

set strings corresponding to a product of operators. 
```
add_operator_product( 0.5, ['h','t1','t2'])
 ```
Currently supported operators include 

the unit operator
```
'1'
```
a general one-body operator
```
'h' 
```
an general two-body operator

```
'g' 
```    
the fock operator
```
'f' 
```    
the fluctuation potental operator 
```
'v' 
```
creation / annihilation operators. e.g., <img src="https://render.githubusercontent.com/render/math?math=a^{\dagger}_i">, <img src="https://render.githubusercontent.com/render/math?math=a_j">, etc.
```
a*(i)
a(j)
```
up to four-body transition operators, e.g., p\*q, p\*q\*rs, etc.
```
'e1(p,q)' 
'e2(p,q,r,s)' 
```    
singles, doubles, triples, and quadruples t-amplitudes 
```
't1'
't2' 
't3'
't4'
```
reference, singles, doubles, triples, and quadruples left-hand amplitudes 

```
'l0'
'l1'  
'l2'  
'l3'
'l4'
```    
reference, singles, doubles, triples, and quadruples right-hand amplitudes 
```
'r0'
'r1'  
'r2'
'r3'
'r4'
```    
Note that all factors such as the 1/4 associated with t2, l2, and r2 are handled internally.
    
#### add_commutator: 

set strings corresponding to a commutator of operators. Note that the arguments are lists to allow for commutators of
products of operators.
```
add_commutator(1.0, ['f'], ['t2'])
```
#### add_double_commutator: 

set strings corresponding to a double commutator involving three operators. Note that the arguments are lists to allow
for commutators of products of operators.
```
add_double_commutator(1.0/2.0, ['f'], ['t2'], ['t1'])
```    
#### add_triple_commutator: 

set strings corresponding to a triple commutator involving four operators. Note that the arguments are lists to allow
for commutators of products of operators.

```
add_triple_commutator(1.0/6.0, ['f'], ['t2'], ['t1'], ['t1'])
```    
#### add_quadruple_commutator: 

set strings corresponding to a quadruple commutator involving five operators. Note that the arguments are lists to allow
for commutators of products of operators.
    
```
add_quadruple_commutator(1.0/24.0, ['f'], ['t2'], ['t1'], ['t1'], ['t1'])
```

#### add_st_operator: 

set strings corresponding to a similarity transformed operator commutator involving five operators. The first argument
after the numerical value is a list of operators; the product of these operators will be similarity transformed. The
next argument is a list of operators appearing as a sum the exponential function. The similarity transformation is
performed by applying the BCH expansion four nested commutators.

```
add_st_operator(1.0, ['v'],['t1','t2'])
```    
#### set_print_level: 

Control the amount of output. Any value greater than the default value of 0 will cause the code to print starting
strings.
```
set_print_level(0)
```    
#### set_left_operators:
    
set a sum (outer list) of products (inner lists) of operators that define the bra state
   
```
set_left_operators([['1'],['l1'],['l2']])
```
#### set_right_operators:
    
set a sum (outer list) of products (inner lists) of operators that define the ket state
```
set_right_operators([['r0'],['r1'],['r2']])
```    

#### simplify: 
    
consolidate/cancel terms and zero any delta functions that involve occupied / virtual combinations.
```
simplify()
```
        
#### print: 
    
print current list of strings. which strings is dictated by the string_type flag. the default value is string_type = 'fully-contracted'

```
print(string_type = 'all/fully-contracted/one-body/two-body')
```        

#### fully_contracted_strings: 
    
returns strings involving no creation / annihilation operators

```
fully_contracted_strings()
```    

<a name="spin"></a>

#### fully_contracted_with_spin: 
    
returns strings involving no creation / annihilation operators. integrate spin, eliminating non-spin-conserving terms, given a dictionary of spins for non-summed labels.

```
spin_labels = {
    'e' : 'a',
    'f' : 'b',
    'm' : 'a',
    'n' : 'b'
}
fully_contracted_strings_with_spin(spin_labels)
```


#### clear: 
clear the current set of strings

```
clear()
```    
                
        
