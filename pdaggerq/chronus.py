import re
import sys

replacement_mapping = {
    'v0': 'r',
    'v1': 'l',
    'o1': 'h',
    'o0': 'c',
}

# anything["0011_Loovv"](anything) => anything(anything).block(TAManager.toBlockRange("ccll"))
def replace_block_strings_active(input_string):

    pattern = re.compile(r'([^ ]+)\["([01]{1,4}_L?([ov]{1,4}))"\](\([^)]+\))')

    def replace_match(match):
        to_be_copied_1, key2, key1, to_be_copied_2 = match.groups()

        replaced_value = ''.join(replacement_mapping.get(ch + bit, ch) for ch, bit in zip(key1, key2))
        return f'{to_be_copied_1}_{key1}{to_be_copied_2}.block(b.{replaced_value})'

    result = pattern.sub(replace_match, input_string)

    return result


# eri["oovv"](ijab) => conj(eri["vvoo"](abij))
def replace_conj_strings_option1(input_string):

    pattern = re.compile(r'eri\["(oovv|oovo|vovv)"\]\("([a-o]),([a-o]),([a-o]),([a-o])"\)')

    def replace_match(match):
        vo, idx0, idx1, idx2, idx3 = match.groups()
        return f'conj(eri["{vo[2]}{vo[3]}{vo[0]}{vo[1]}"]("{idx2},{idx3},{idx0},{idx1}"))'
    
    result = pattern.sub(replace_match, input_string)
    return result

# eri["oovv_0011"](ijab) => conj(eri["vvoo_1100"](abij))
def replace_conj_strings_option1_active(input_string):

    pattern = re.compile(r'eri\["([01]{4})_(oovv|oovo|vovv)"\]\("([a-o]),([a-o]),([a-o]),([a-o])"\)')

    def replace_match(match):
        ae, vo, idx0, idx1, idx2, idx3 = match.groups()
        return f'conj(eri["{ae[2]}{ae[3]}{ae[0]}{ae[1]}_{vo[2]}{vo[3]}{vo[0]}{vo[1]}"]("{idx2},{idx3},{idx0},{idx1}"))'
    
    result = pattern.sub(replace_match, input_string)
    return result

# tmps_["123_Loovv"].~TArrayD => TAmanager.free("oovv", std::move(tmps_["123_Loovv"]))
def replace_free_strings(input_string):

    pattern = re.compile(r'(tmps_?\["[0-9perm]+_([ovL]+)"\]).~TArrayD\(\);')

    def replace_match(match):
        name, vo = match.groups()
        vo = re.sub(r'L','',vo) # remove the L
        replaced_value = vo # ''.join(replacement_mapping.get(ch + bit, ch) for ch, bit in zip(vo, ae))
        return f'TAmanager.free("{replaced_value}", std::move({name}));'
    
    result = pattern.sub(replace_match, input_string)
    return result

# tmps_["Loovv_0011_123"].~TArrayD => TAmanager.free("ccll", std::move(tmps_["Loovv_0011_123"]))
def replace_free_strings_active(input_string):

    pattern = re.compile(r'(tmps_\["[0-9]+_([01]+)_([ovL]+)"\]).~TArrayD\(\);')

    def replace_match(match):
        name, ae, vo = match.groups()
        vo = re.sub(r'L','',vo) # remove the L
        replaced_value = ''.join(replacement_mapping.get(ch + bit, ch) for ch, bit in zip(vo, ae))
        return f'TAmanager.free("{replaced_value}", std::move({name}));'
    
    result = pattern.sub(replace_match, input_string)
    return result

# (reused_)tmps_["123_Loovv"] = anything 
# => 
# (reused_)tmps_.emplace(std::make_pair("123_Loovv"), TAmamager.malloc<MatsT>("oovv"); original line
def add_malloc_strings(input_string):
    tmp_pattern = re.compile(r'((\s*reused_|\s*tmps_)\["([0-9]+)_([ovL]+)"\]\(".*"\) *= [^;]+;)')

    def replace_tmp_match(match):
        line, tmps, index, vo = match.groups()
        vo = re.sub(r'L','',vo) # remove the L
        replaced_value = vo #''.join(replacement_mapping.get(ch + bit, ch) for ch, bit in zip(vo, ae))
        return f'{tmps}.emplace(std::make_pair("{index}_{replaced_value}", TAmanager.malloc<MatsT>("{replaced_value}")));{line}'
    
    result = tmp_pattern.sub(replace_tmp_match, input_string)
    return result

# (reuse)tmps_["Loovv_0011_123"] = anything 
# => 
# (reuse)tmps_.emplace(std::make_pair("Loovv_0011_123"), TAmamager.malloc<MatsT>("ccll"); original line
def add_malloc_strings_active(input_string):
    tmp_pattern = re.compile(r'((\s*reused_|\s*tmps_)\["([0-9]+)_([01]+)_([ovL]+)"\]\(".*"\) *= [^;]+;)')

    def replace_tmp_match(match):
        line, tmps, index, ae, vo = match.groups()
        vo = re.sub(r'L','',vo) # remove the L
        replaced_value = ''.join(replacement_mapping.get(ch + bit, ch) for ch, bit in zip(vo, ae))
        return f'{tmps}.emplace(std::make_pair("{index}_{replaced_value}", TAmanager.malloc<MatsT>("{replaced_value}")));{line}'
    
    result = tmp_pattern.sub(replace_tmp_match, input_string)
    return result

# anything dot anything => anything dot anything; TA::get_default_world().gop.fence()
def add_fence_lines(input_string):
    result = re.sub(r'(\s*.*dot.*)(;\s*)',r'\1.get()\2TA::get_default_world().gop.fence();\n    ', input_string)
    return result

# (reuse_)tmps_["123_Loovv"] => (reuse_)tmps_["123_ccll"]
def replace_tmp_spaces(input_string):
    pattern = re.compile(r'(reused_|tmps_)\["([0-9]+)_([ovL]+)"\]')

    def replace_match(match):
        tmps, index, vo = match.groups()
        vo = re.sub(r'L','',vo) # remove the L
        replaced_value = vo #''.join(replacement_mapping.get(ch + bit, ch) for ch, bit in zip(vo, ae))
        return f'{tmps}["{index}_{replaced_value}"]'
    
    result = pattern.sub(replace_match, input_string)
    return result

# (reused_)tmps_["Loovv_0011_123"] => (reused_)tmps_["ccll_123"]
def replace_tmp_spaces_active(input_string):
    pattern = re.compile(r'(reused_|tmps_)\["([0-9]+)_([01]+)_([ovL]+)"\]')

    def replace_match(match):
        tmps, index, ae, vo = match.groups()
        vo = re.sub(r'L','',vo) # remove the L
        replaced_value = ''.join(replacement_mapping.get(ch + bit, ch) for ch, bit in zip(vo, ae))
        return f'{tmps}["{index}_{replaced_value}"]'
    
    result = pattern.sub(replace_match, input_string)
    return result

# append destructor
def add_destructor(input_string, class_name):
    text  = '  \n\n'
    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  '+class_name+'<MatsT,IntsT>::~'+class_name+'() {\n\n'
    text += '    TAManager &TAmanager = TAManager::get();\n\n'
    pattern = re.compile(r'reused_\.emplace\(std::make_pair\(("[0-9]+_[chlrvo]+"), TAmanager.malloc<MatsT>\(("[chlrvo]+")\)\)\);')
    for line in input_string.split('\n'):
        match = pattern.search(line)
        if match:
            name, size = match.groups()
            text += f'    TAmanager.free({size},std::move(reused_[{name}]), true); \n'
    text += '  }\n\n'
    return input_string + text

# append destructor
def add_destructor_active(input_string, class_name):
    text  = '  \n\n'
    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  '+class_name+'<MatsT,IntsT>::~'+class_name+'() {\n\n'
    text += '    TAManager &TAmanager = TAManager::get();\n\n'
    pattern = re.compile(r'reused_\.emplace\(std::make_pair\(("[0-9]+_[chlr]+"), TAmanager.malloc<MatsT>\(("[chlr]+")\)\)\);')
    for line in input_string.split('\n'):
        match = pattern.search(line)
        if match:
            name, size = match.groups()
            text += f'    TAmanager.free({size},std::move(reused_[{name}]), true); \n'
    text += '  }\n\n'
    return input_string + text

# remove lines defining scalars, i.e. containing 'scalar =', 'scalar +=' etc
def remove_scalar_lines(input_string):
    output = ""
    pattern = "scalar.*="
    for line in input_string.split('\n'):
        if not re.search(pattern, line):
            output += line + '\n'
    return output

# make sure the first equation in each LHS object uses =, not +=
# this assumes sigmaR variable name looks like sigmaR2 sigmaR3 sigmaR4
def first_LHS_direct_equal(input_string):
    LHSs  = re.findall(r'^(?!.*//).*sigmaR1[^ \(]+',input_string, re.M) # not commented out, sigmaR1 until eg.("a,i") 
    LHSs += re.findall(r'^(?!.*//).*sigmaR2[^ \(]+',input_string, re.M)
    LHSs += re.findall(r'^(?!.*//).*sigmaR3[^ \(]+',input_string, re.M)
    LHSs += re.findall(r'^(?!.*//).*sigmaR4[^ \(]+',input_string, re.M)
    LHSs += re.findall(r'tmps_\["[^\]]*?_[^\]]*?"\]',input_string) # each tmp should start from an equal sign
    LHSs = list(set(LHSs))
    lines = input_string.split('\n')
    for LHS in LHSs:
        LHS = LHS.replace(r'[','\[')
        LHS = LHS.replace(r']','\]')
        for i, line in enumerate(lines):
            if re.search(LHS, line):
            #if re.search(LHS, line) and not re.search('\/\/', line):
                line = line.replace(r'+=','=')
                lines[i] = line.replace(r'-= ','= -')
                break

    return '\n'.join(lines)


# gather all lines containing reuse_tmps_.emplace and put in the function initReuseTmps
def extract_malloc_reusetmps(input_string, class_name):
    pattern = r'\s*reused_.emplace[^;]+;'
    match = re.findall(pattern, input_string)
    if match:
        block_of_lines = ''.join(match)

        content_before = '  void '+class_name+'<MatsT,IntsT>::initializeEOMCC() {\n\n    TAManager &TAmanager = TAManager::get();\n'
        content_after = '\n  }\n\n'
        new_content = f"{content_before}\n{block_of_lines}\n{content_after}\n"

        new_content += re.sub(pattern, '', input_string)
        return new_content

# at the beginning of the buildSigmaRight function, define R names
def add_tenser_definition(output_content, class_name):
    cc_vec_def = ''

    cc_vec_def += '//TODO: add tensor definitions\n    '
    cc_vec_def += '//eg: const TArray &V2 = V.get_tensor("TwoBody");\n    '
    cc_vec_def += '//    TArray &HV2 = HV.get_tensor("TwoBody");\n    '
    cc_vec_def += '\n    '
    cc_vec_def += 'switch (vecType) {\n    '
    cc_vec_def += '  case EOMCCEigenVecType::RIGHT:\n    '
    cc_vec_def += '    //TODO: modify form of formR_tilde\n    '
    cc_vec_def += '    formR_tilde(V2, V3, HV2, HV3);\n    '
    cc_vec_def += '    break;\n    '
    cc_vec_def += '  case EOMCCEigenVecType::LEFT:\n    '
    cc_vec_def += '    //TODO: modify form of formL_tilde\n    '
    cc_vec_def += '    formL_tilde(V2, V3, HV2, HV3);\n    '
    cc_vec_def += '    break;\n    '
    cc_vec_def += '}\n  '
    cc_vec_def += '}\n\n'
    cc_vec_def += '  //TODO: modify form of formR_tilde\n'
    cc_vec_def += '  void '+class_name+'<MatsT,IntsT>::formR_tilde(const TArray &R2, const TArray &R3, TArray &sigmaR2, TArray &sigmaR3) const {\n    '
    match = re.compile("buildSigma.+{\s+").search(output_content)
    return output_content[:match.end()] + cc_vec_def + output_content[match.end():]

# add buildDiag function
def add_build_diag(output_content, class_name):
    # add definition of buildDiag
    build_diag_text = '  }\n\n  void '+class_name+'<MatsT,IntsT>::buildDiag(MatsT * diag, std::vector<double> eps) const {\n'

    build_diag_text += '  // TODO: buildDiagonals of the EOM Hamiltonian\n  '

    return output_content + build_diag_text + '  }\n\n'


def add_build_pc(output_content, class_name):
    # add definition of buildDiag
    build_pc_text  = '  template <typename MatsT, typename IntsT>\n'
    build_pc_text += '  typename Davidson<dcomplex>::LinearTrans_t '+class_name+'<MatsT,IntsT>::DavidsonPreconditionerBuilder(dcomplex * curEig, dcomplex * eomDiag){\n\n'
    build_pc_text += '  //TODO: build preconditioner\n  ' 

    build_pc_text += '  }\n\n'

    text = '\n'
    text += '\n'

    text += '  void '+class_name+'<MatsT,IntsT>::runLambda(){} \n'
    text += '\n'
    text += '\n'
    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  typename Davidson<dcomplex>::VecsGen_t '+class_name+'<MatsT,IntsT>::EmptyDavidsonVectorBuilder(){\n'
    text += '  //TODO: build Empty DavidsonVector\n  ' 
    text += '  }\n'
    text += '\n'
    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  typename Davidson<dcomplex>::LinearTrans_t '+class_name+'<MatsT,IntsT>::DavidsonResidualBuilder(EOMCCEigenVecType &eigenVecType){\n'
    text += '  //TODO: build Davidson residual\n  ' 
    text += '\n'
    text += '  }\n'
    return output_content + text + build_pc_text


# namespace at the beginning and the end
def add_namespace(output_content):
    text  = '#pragma once \n'
    text += '#include <chronusq_sys.hpp>\n'
    text += '#include <coupledcluster.hpp>\n'
    text += '\n'
    text += 'namespace ChronusQ{\n'
    return text + output_content + '\n}\n'

# empty constructor 
def add_constructor(output_content, class_name):
    text  = ''

    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  '+class_name+'<MatsT,IntsT>::'+class_name+'(const SafeFile &savFile,\n'
    text += '    CCIntermediates<MatsT> &intermediates,\n'
    text += '    const EOMSettings &eomSettings,\n'
    text += '    const CoupledClusterSettings &ccSettings):\n'
    text += '    EOMCCBase<MatsT,IntsT>(savFile, intermediates,eomSettings,ccSettings),\n'
    text += '    vLabel_(intermediates.vLabel), oLabel_(intermediates.oLabel),\n'
    text += '    reused_(intermediates.sigmaOps),\n'
    text += '    tmps_(intermediates.tempOps),\n'
    text += '    //TODO: construct T1, T2, T3 as needed \n'
    text += '    {\n'

    text += '  }\n\n'


    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  inline size_t '+class_name+'<MatsT,IntsT>::toCompoundS(size_t a, size_t i) const {\n'
    text += '  // TODO: you may need this function to build the diagonal elements of the EOM Hamiltonian\n'
    text += '  }\n'
    text += '\n'
    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  inline size_t '+class_name+'<MatsT,IntsT>::toCompoundD(size_t a, size_t b, size_t i, size_t j) const {\n'
    text += '  // TODO: you may need this function to build the diagonal elements of the EOM Hamiltonian\n'
    text += '  }\n\n'


    return text + output_content

def to_chronus_string(input_content, class_name="REPLACEME", is_active=False):

    # code
    output_content = re.sub(r'}',r'  }', input_content)

    # chop off area not needed,  function definitions
    match = re.compile("/+ Scalars /+\s*\n").search(output_content)
    if match:
        output_content = match.group(0) + output_content[match.end():]
        output_content = re.sub(r'/+ Scalars /+\s*\n','\n\n', output_content)
        output_content = re.sub(r'/+ End of Scalars /+\s*\n','\n\n', output_content)

    match = re.compile("/+ Shared  Operators /+\s*\n").search(output_content)
    if match:
        output_content = re.sub(r'/+ Shared  Operators /+\s*\n','  void '+class_name+'<MatsT,IntsT>::formEOMIntermediates() {\n\n    TAManager &TAmanager = TAManager::get();\n\n', output_content)
        output_content = re.sub(r'\n\s*/+ End of Shared Operators /+\s*\n','\n\n  }\n\n', output_content)

    match = re.compile("/+ Evaluate Equations /+\s*\n").search(output_content)
    if match:
        output_content = re.sub(r'/+ Evaluate Equations /+\s*\n','  void '+class_name+'<MatsT,IntsT>::buildSigma(const EOMCCSDVector<MatsT> &V, EOMCCSDVector<MatsT> &HV, EOMCCEigenVecType vecType) const {\n\n    TAManager &TAmanager = TAManager::get();\n\n', output_content)

    if is_active:
        output_content = re.sub(r't1\["(..)"\]',r't1["\1_vo"]' ,output_content) # must happen before block replacement
        output_content = re.sub(r't2\["(....)"\]',r't2["\1_vvoo"]' ,output_content) # must happen before block replacement
        output_content = replace_conj_strings_option1_active(output_content) # must happen before the block replacement
        output_content = replace_block_strings_active(output_content)
        output_content = re.sub(r't1_vo',r't1' ,output_content) # must happen before block replacement
        output_content = re.sub(r't2_vvoo',r't2' ,output_content) # must happen before block replacement
        output_content = replace_free_strings_active(output_content)
        output_content = first_LHS_direct_equal(output_content) #must happen after remove_scalar_lines, before add_tenser_definition and add_malloc
        output_content = add_malloc_strings_active(output_content)
        output_content = replace_tmp_spaces_active(output_content) # must happen after replace free and add_malloc
    else:
        output_content = replace_conj_strings_option1(output_content) # must happen before the block replacement
        output_content = replace_free_strings(output_content)
        output_content = first_LHS_direct_equal(output_content) #must happen after remove_scalar_lines, before add_tenser_definition and add_malloc
        output_content = add_malloc_strings(output_content)
        output_content = replace_tmp_spaces(output_content) # must happen after replace free and add_malloc
    output_content = remove_scalar_lines(output_content)
    output_content = add_fence_lines(output_content)
    output_content = re.sub(r'f\["oo"\]','this->fockMatrix_ta["oo"]', output_content)
    output_content = re.sub(r'f\["ov"\]','this->fockMatrix_ta["ov"]', output_content)
    output_content = re.sub(r'f\["vo"\]','this->fockMatrix_ta["vo"]', output_content)
    output_content = re.sub(r'f\["vv"\]','this->fockMatrix_ta["vv"]', output_content)

    output_content = re.sub(r't1','this->T1_', output_content)
    output_content = re.sub(r't2','this->T2_', output_content)
    output_content = re.sub(r't3','this->T3_', output_content)
    output_content = re.sub(r'r1','R1', output_content)
    output_content = re.sub(r'r2','R2', output_content)
    output_content = re.sub(r'r3','R3', output_content)
    output_content = re.sub(r'r4','R4', output_content)
    output_content = re.sub(r'eri\["oooo"\]','this->antiSymMoints["oooo"]', output_content)
    output_content = re.sub(r'eri\["vooo"\]','this->antiSymMoints["vooo"]', output_content)
    output_content = re.sub(r'eri\["vvoo"\]','this->antiSymMoints["vvoo"]', output_content)
    output_content = re.sub(r'eri\["vovo"\]','this->antiSymMoints["vovo"]', output_content)
    output_content = re.sub(r'eri\["vovv"\]','this->antiSymMoints["vovv"]', output_content)
    output_content = re.sub(r'eri\["vvvo"\]','this->antiSymMoints["vvvo"]', output_content)
    output_content = re.sub(r'eri\["vvvv"\]','this->antiSymMoints["vvvv"]', output_content)
    output_content = re.sub(r'eri_oooo','this->antiSymMoints["oooo"]', output_content)
    output_content = re.sub(r'eri_vooo','this->antiSymMoints["vooo"]', output_content)
    output_content = re.sub(r'eri_vvoo','this->antiSymMoints["vvoo"]', output_content)
    output_content = re.sub(r'eri_vovo','this->antiSymMoints["vovo"]', output_content)
    output_content = re.sub(r'eri_vovv','this->antiSymMoints["vovv"]', output_content)
    output_content = re.sub(r'eri_vvvo','this->antiSymMoints["vvvo"]', output_content)
    output_content = re.sub(r'eri_vvvv','this->antiSymMoints["vvvv"]', output_content)
    output_content = re.sub(r'Id','this->Id', output_content)

    # re-organize reusetmps mallocs into a new function
    match = re.compile("r'\s*reused_.emplace[^;]+;'").search(output_content)
    if match:
        output_content = extract_malloc_reusetmps(output_content, class_name) # must happen after 1. add_malloc 2. removing everything until ##### Scalars #####

    output_content = add_tenser_definition(output_content, class_name)
    output_content = add_constructor(output_content, class_name)
    output_content = add_build_diag(output_content, class_name)
    output_content = add_build_pc(output_content, class_name)
    output_content = add_destructor(output_content, class_name)

    # names of derived classes
    output_content = re.sub(r'void','template <typename MatsT, typename IntsT>\n  void', output_content)
    output_content = re.sub(r'##+','', output_content)

    # namespace
    output_content = add_namespace(output_content) #must happen after replace EOMDIP_active to individual versions

    # print the result
    output_content = "////// Start of ChronusQ generated code //////\n\n" + output_content
    print(output_content)
    output_content = output_content + "\n\n////// End of ChronusQ generated code //////\n"
    return output_content
    
