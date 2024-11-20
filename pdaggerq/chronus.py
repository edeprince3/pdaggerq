import re
import sys


# eri["oovv_0011"](anything) => conj(eri["oovv_0011"])
def replace_conj_strings_option1(input_string):

    pattern = re.compile(r'eri\["(oovv|oovo|vovv)"\]\("([a-o]),([a-o]),([a-o]),([a-o])"\)')

    def replace_match(match):
        vo, idx0, idx1, idx2, idx3 = match.groups()
        return f'conj(eri["{vo[2]}{vo[3]}{vo[0]}{vo[1]}"]("{idx2},{idx3},{idx0},{idx1}"))'
    
    result = pattern.sub(replace_match, input_string)
    return result

# tmps_["123_Loovv"].~TArrayD => TAmanager.free("oovv", std::move(tmps_["123_Loovv"]))
def replace_free_strings(input_string):

    pattern = re.compile(r'((perm)?tmps_?\["[0-9perm]+_([ovL]+)"\]).~TArrayD\(\);')

    def replace_match(match):
        name, junk, vo = match.groups()
        vo = re.sub(r'L','',vo) # remove the L
        replaced_value = vo # ''.join(replacement_mapping.get(ch + bit, ch) for ch, bit in zip(vo, ae))
        return f'TAmanager.free("{replaced_value}", std::move({name}));'
    
    result = pattern.sub(replace_match, input_string)
    return result

# (reused_)tmps_["123_Loovv"] = anything 
# => 
# (reused_)tmps_.emplace(std::make_pair("123_Loovv"), TAmamager.malloc<MatsT>("oovv"); original line
def add_malloc_strings(input_string):
    tmp_pattern = re.compile(r'((\s*reused_|\s*tmps_)\["([0-9]+)_([ovL]+)"\]\(".*"\) *= [^;]+;)')
    perm_pattern = re.compile(r'((\s*reused_|\s*tmps_)\["(perm)_([ovL]+)"\]\(".*"\) *= [^;]+;)')

    def replace_tmp_match(match):
        line, tmps, index, vo = match.groups()
        vo = re.sub(r'L','',vo) # remove the L
        replaced_value = vo #''.join(replacement_mapping.get(ch + bit, ch) for ch, bit in zip(vo, ae))
        return f'{tmps}.emplace(std::make_pair("{index}_{replaced_value}", TAmanager.malloc<MatsT>("{replaced_value}")));{line}'
    
    def replace_perm_match(match):
        line, tmps, index, vo = match.groups()
        replaced_value = re.sub(r'L','',vo) # remove the L
        return f'{tmps}["{index}_{vo}"] = TAmanager.malloc<MatsT>("{replaced_value}");{line}'
    
    result_tmp = tmp_pattern.sub(replace_tmp_match, input_string)
    result = perm_pattern.sub(replace_perm_match, result_tmp)
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

# append destructor
def add_destructor(input_string):
    text  = '  \n\n'
    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  EOMDIP_4h2pCCSDT<MatsT,IntsT>::~EOMDIP_4h2pCCSDT() {\n\n'
    text += '    TAManager &TAmanager = TAManager::get();\n\n'
    pattern = re.compile(r'reused_\.emplace\(std::make_pair\(("[0-9]+_[chlrvo]+"), TAmanager.malloc<MatsT>\(("[chlrvo]+")\)\)\);')
    for line in input_string.split('\n'):
        match = pattern.search(line)
        if match:
            name, size = match.groups()
            text += f'    TAmanager.free({size},std::move(reused_[{name}]), true); \n'
    text += '  }\n\n'
    return input_string + text

## look at the first reserve element, eg scalar_["23"], add scalars_.reserve(24) 
#def add_scalar_reserve(input_string):
#    pattern = re.compile(r'(\s*)scalars_\["([0-9]+)"\]\s*=')
#    #match = pattern.search(input_string)
#    matches = pattern.findall(input_string)
#
#    if matches:
#        indentation = matches[-1][0]
#        size = int(matches[-1][1]) + 1
#        replacement_line = f"{indentation}scalars_.reserve({size});\n"
#       
#        result = pattern.sub(replacement_line + r"\g<0>", input_string, count=1)
#        
#        return result

# remove all lines containing 'scalar'
def remove_scalar_lines(input_string):
    output = ""
    pattern = "scalar"
    for line in input_string.split('\n'):
        if not re.search(re.escape(pattern), line):
            output += line + '\n'
    return output

# make sure the first equation in each LHS object uses =, not +=
def first_LHS_direct_equal(input_string):
    LHSs = re.findall(r'sigmaR2_..',input_string)
    LHSs += re.findall(r'sigmaR3_....',input_string)
    LHSs += re.findall(r'sigmaR4_......',input_string)
    LHSs += re.findall(r'tmps_\[".*?_.*?"\]',input_string)
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
def extract_malloc_reusetmps(input_string):
    pattern = r'\s*reused_.emplace[^;]+;'
    match = re.findall(pattern, input_string)
    if match:
        block_of_lines = ''.join(match)

        content_before = '  void EOMDIP_4h2pCCSDT<MatsT,IntsT>::initializeEOMCC() {\n\n    TAManager &TAmanager = TAManager::get();\n'
        content_after = '\n  }\n\n'
        new_content = f"{content_before}\n{block_of_lines}\n{content_after}\n"

        new_content += re.sub(pattern, '', input_string)
        return new_content

# at the beginning of the buildSigmaRight function, define R names
def add_tenser_definition(output_content):
    cc_vec_def = ''

    cc_vec_def += 'const TArray &V4 = V.get_tensor("FourBody");\n    '
    cc_vec_def += 'TArray &HV4 = HV.get_tensor("FourBody");\n    '
    cc_vec_def += 'const TArray &V3 = V.get_tensor("ThreeBody");\n    '
    cc_vec_def += 'TArray &HV3 = HV.get_tensor("ThreeBody");\n    '
    cc_vec_def += 'const TArray &V2 = V.get_tensor("TwoBody");\n    '
    cc_vec_def += 'TArray &HV2 = HV.get_tensor("TwoBody");\n\n    '
    cc_vec_def += 'switch (vecType) {\n    '
    cc_vec_def += '  case EOMCCEigenVecType::RIGHT:\n    '
    cc_vec_def += '    formR_tilde(V2, V3, V4, HV2, HV3, HV4);\n    '
    cc_vec_def += '    break;\n    '
    cc_vec_def += '  case EOMCCEigenVecType::LEFT:\n    '
    cc_vec_def += '    formL_tilde(V2, V3, V4, HV2, HV3, HV4);\n    '
    cc_vec_def += '    break;\n    '
    cc_vec_def += '}\n  '
    cc_vec_def += '}\n\n'
    cc_vec_def += '  void EOMDIP_4h2pCCSDT<MatsT,IntsT>::formR_tilde(const TArray &R2, const TArray &R3, const TArray &R4, TArray &sigmaR2, TArray &sigmaR3, TArray &sigmaR4) const {\n    '
    match = re.compile("buildSigma.+{\s+").search(output_content)
    return output_content[:match.end()] + cc_vec_def + output_content[match.end():]

# add buildDiag function
def add_build_diag(output_content):
    # add definition of buildDiag
    build_diag_text = '  }\n\n  void EOMDIP_4h2pCCSDT<MatsT,IntsT>::buildDiag(MatsT * diag, std::vector<double> eps) const {\n\n    TAManager &TAmanager = TAManager::get();\n'

    build_diag_text += '    size_t n_v = TAmanager.getRange(vLabel_).extent();\n'
    build_diag_text += '    size_t n_o = TAmanager.getRange(oLabel_).extent();\n'

    build_diag_text += '    std::fill_n(diag, this->Hbar_dim, MatsT(0.0));\n\n'

    build_diag_text += '    for (auto j = 0; j < n_o; j++){\n'
    build_diag_text += '      for (auto i = 0; i < j; i++){\n'
    build_diag_text += '        diag[this->toCompoundS(i,j)] = - eps[i] - eps[j];\n'
    build_diag_text += '      }\n'
    build_diag_text += '    }\n'
    build_diag_text += '    MatsT * diag3 = diag + this->Hbar_dimension_offsets.at("ThreeBody");\n'
    build_diag_text += '    for (auto a = 0; a < n_v; a++){\n'
    build_diag_text += '      for (auto k = 0; k < n_o; k++){\n'
    build_diag_text += '        for (auto j = 0; j < k; j++){\n'
    build_diag_text += '          for (auto i = 0; i < j; i++){\n'
    build_diag_text += '            diag3[this->toCompoundD(a,i,j,k)] = eps[a+n_o] - eps[i] - eps[j] - eps[k];\n'
    build_diag_text += '          }\n'
    build_diag_text += '        }\n'
    build_diag_text += '      }\n'
    build_diag_text += '    }\n'
    build_diag_text += '    MatsT * diag4 = diag + this->Hbar_dimension_offsets.at("FourBody");\n'
    build_diag_text += '    for (auto b = 0; b < n_v; b++){\n'
    build_diag_text += '      for (auto a = 0; a < b; a++){\n'
    build_diag_text += '        for (auto l = 0; l < n_o; l++){\n'
    build_diag_text += '          for (auto k = 0; k < l; k++){\n'
    build_diag_text += '            for (auto j = 0; j < k; j++){\n'
    build_diag_text += '              for (auto i = 0; i < j; i++){\n'
    build_diag_text += '                diag4[this->toCompoundT(a,b,i,j,k,l)] = eps[a+n_o] + eps[b+n_o] - eps[i] - eps[j] - eps[k] - eps[l];\n'
    build_diag_text += '              }\n'
    build_diag_text += '            }\n'
    build_diag_text += '          }\n'
    build_diag_text += '        }\n'
    build_diag_text += '      }\n'
    build_diag_text += '    }\n'

    return output_content + build_diag_text + '  }\n\n'

def build_pc_2body():

    text = ''
    text += '          MatsT * Diag_IJ = eomDiag;\n'
    text += '          TA::foreach_inplace(curB.get_tensor("TwoBody"), [iVec, curEig, Diag_IJ, this, PCsmall](TA::TensorZ &tile){\n'
    text += '            const auto& lobound = tile.range().lobound();\n'
    text += '            const auto& upbound = tile.range().upbound();\n'
    text += '\n'
    text += '            dcomplex denom = 0.0;\n'
    text += '            std::vector<std::size_t> x{0, 0};\n'
    text += '            for(x[0] = lobound[0]; x[0] < upbound[0]; ++x[0]) {\n'
    text += '              for(x[1] = lobound[1]; x[1] < upbound[1]; ++x[1]) {\n'
    text += '                if (x[0] == x[1]) continue;\n'
    text += '                denom = curEig[iVec] - Diag_IJ[this->toCompoundS(x[0], x[1])];\n'
    text += '                if (std::abs(denom) >= PCsmall) tile[x] /= denom;\n'
    text += '              }\n'
    text += '            }\n'
    text += '          });\n'
    text += '          TA::get_default_world().gop.fence();\n'
    return text

def build_pc_3body():
    text = ''
    text += '          MatsT * Diag_AIJK = eomDiag + this->Hbar_dimension_offsets.at("ThreeBody");\n'
    text += '          TA::foreach_inplace(curB.get_tensor("ThreeBody"), [iVec, curEig, Diag_AIJK, this, PCsmall](TA::TensorZ &tile){\n'
    text += '            const auto& lobound = tile.range().lobound();\n'
    text += '            const auto& upbound = tile.range().upbound();\n'
    text += '\n'
    text += '            dcomplex denom = 0.0;\n'
    text += '            std::vector<std::size_t> x{0, 0, 0, 0};\n'
    text += '            for(x[0] = lobound[0]; x[0] < upbound[0]; ++x[0]){\n'
    text += '              size_t a = x[0];\n'
    text += '              for(x[1] = lobound[1]; x[1] < upbound[1]; ++x[1]){\n'
    text += '                for(x[2] = lobound[2]; x[2] < upbound[2]; ++x[2]){\n'
    text += '                  for(x[3] = lobound[3]; x[3] < upbound[3]; ++x[3]){\n'
    text += '                    if (x[1] == x[2] or x[2] == x[3] or x[3] == x[1])\n'
    text += '                      continue;\n'
    text += '                    size_t i = x[1];\n'
    text += '                    size_t j = x[2];\n'
    text += '                    size_t k = x[3];\n'
    text += '                    denom = curEig[iVec] - Diag_AIJK[this->toCompoundD(a, i, j, k)];\n'
    text += '                    if (std::abs(denom) >= PCsmall) tile[x] /= denom;\n'
    text += '                  }\n'
    text += '                }\n'
    text += '              }\n'
    text += '            }\n'
    text += '          });\n'
    text += '          TA::get_default_world().gop.fence();\n'

    return text

def build_pc_4body():
    text = ''
    text += '          MatsT * Diag_ABIJKL = eomDiag + this->Hbar_dimension_offsets.at("FourBody");\n'
    text += '          TA::foreach_inplace(curB.get_tensor("ThreeBody"), [iVec, curEig, Diag_ABIJKL, this, PCsmall](TA::TensorZ &tile){\n'
    text += '            const auto& lobound = tile.range().lobound();\n'
    text += '            const auto& upbound = tile.range().upbound();\n'
    text += '\n'
    text += '            dcomplex denom = 0.0;\n'
    text += '            std::vector<std::size_t> x{0, 0, 0, 0, 0, 0};\n'
    text += '            for(x[0] = lobound[0]; x[0] < upbound[0]; ++x[0]){\n'
    text += '              size_t a = x[0];\n'
    text += '              for(x[1] = lobound[1]; x[1] < upbound[1]; ++x[1]){\n'
    text += '                if (x[0] == x[1])\n'
    text += '                  continue;\n'
    text += '                size_t b = x[1];\n'
    text += '                for(x[2] = lobound[2]; x[2] < upbound[2]; ++x[2]){\n'
    text += '                  size_t i = x[2];\n'
    text += '                  for(x[3] = lobound[3]; x[3] < upbound[3]; ++x[3]){\n'
    text += '                    if (x[2] == x[3]) continue;\n'
    text += '                    size_t j = x[3];\n'
    text += '                    for(x[4] = lobound[4]; x[4] < upbound[4]; ++x[4]){\n'
    text += '                       if (x[4] == x[2] or x[4] == x[2]) continue; \n'
    text += '                      size_t k = x[4];\n'
    text += '                      for(x[5] = lobound[5]; x[5] < upbound[5]; ++x[5]){\n'
    text += '                        if (x[2] == x[5] or x[3] == x[5] or x[4] == x[5]) continue;\n'
    text += '                        size_t l = x[5];\n'
    text += '                        denom = curEig[iVec] - Diag_ABIJKL[this->toCompoundT(a, b, i, j, k, l)];\n'
    text += '                        if (std::abs(denom) >= PCsmall) tile[x] /= denom;\n'
    text += '                      }\n'
    text += '                    }\n'
    text += '                  }\n'
    text += '                }\n'
    text += '              }\n'
    text += '            }\n'
    text += '          });\n'
    text += '          TA::get_default_world().gop.fence();\n'

    return text

def add_build_pc(output_content):
    # add definition of buildDiag
    build_pc_text  = '  template <typename MatsT, typename IntsT>\n'
    build_pc_text += '  typename Davidson<dcomplex>::LinearTrans_t EOMDIP_4h2pCCSDT<MatsT,IntsT>::DavidsonPreconditionerBuilder(dcomplex * curEig, dcomplex * eomDiag){\n\n'

    build_pc_text += '      double PCsmall = this->eomSettings.davidson_preCond_small;\n'
    build_pc_text += '\n'
    build_pc_text += '      this->PCEOM = [this, eomDiag, curEig, PCsmall]( size_t nVec, SolverVectors<dcomplex> &V,\n'
    build_pc_text += '          SolverVectors<dcomplex> &AV) {\n'
    build_pc_text += '\n'
    build_pc_text += '        AV.set_data(0, nVec, V, 0);\n'
    build_pc_text += '\n'
    build_pc_text += '        EOMCCSDVectorSet<dcomplex> *AV_ptr = nullptr;\n'
    build_pc_text += '        size_t AVshift = 0;\n'
    build_pc_text += '\n'
    build_pc_text += '        try {\n'
    build_pc_text += '          AV_ptr = &dynamic_cast<EOMCCSDVectorSet<dcomplex>&>(AV);\n'
    build_pc_text += '        } catch(const std::bad_cast& e) {\n'
    build_pc_text += '          SolverVectorsView<dcomplex>& AV_view = dynamic_cast<SolverVectorsView<dcomplex>&>(AV);\n'
    build_pc_text += '          AV_ptr = &dynamic_cast<EOMCCSDVectorSet<dcomplex>&>(AV_view.getVecs());\n'
    build_pc_text += '          AVshift = AV_view.shift();\n'
    build_pc_text += '        }\n'
    build_pc_text += '\n'
    build_pc_text += '        for (size_t iVec = 0; iVec < nVec; iVec++) {\n'
    build_pc_text += '\n'
    build_pc_text += '          EOMCCSDVector<dcomplex> &curB = AV_ptr->get(iVec + AVshift);\n'
    build_pc_text += '\n'
    
    build_pc_text += build_pc_2body()

    build_pc_text += '\n'
    build_pc_text += '          dcomplex *diagD = eomDiag + nV_;\n'
    build_pc_text += '\n'

    build_pc_text += build_pc_3body() 

    build_pc_text += '\n'
    build_pc_text += build_pc_4body() 

    build_pc_text += '          curB.enforceSymmetry();\n'
    build_pc_text += '        }\n'
    build_pc_text += '      }; // implicit preConditioner\n'
    build_pc_text += '      return this->PCEOM;\n'
    build_pc_text += '  }\n\n'

    text = '\n'
    text += '\n'

    text += '  void EOMDIP_4h2pCCSDT<MatsT,IntsT>::runLambda(){} \n'
    text += '\n'
    text += '\n'
    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  typename Davidson<dcomplex>::VecsGen_t EOMDIP_4h2pCCSDT<MatsT,IntsT>::EmptyDavidsonVectorBuilder(){\n'
    text += '      // Algorithm with implicit Hbar matrix\n'
    text += '      typename Davidson<dcomplex>::VecsGen_t vecsGenEOM;\n'
    text += '      if (this->eomSettings.hbar_type == EOM_HBAR_TYPE::IMPLICIT) {\n'
    text += '        vecsGenEOM = [this](size_t nVec)->std::shared_ptr<SolverVectors<dcomplex>> {\n'
    text += '          return std::make_shared<EOMCCSDVectorSet<dcomplex>>(this->tensor_builder_, nVec);\n'
    text += '        }; // implicit vecsGenerator\n'
    text += '\n'
    text += '        return vecsGenEOM;\n'
    text += '      }\n'
    text += '  }\n'
    text += '\n'
    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  typename Davidson<dcomplex>::LinearTrans_t EOMDIP_4h2pCCSDT<MatsT,IntsT>::DavidsonResidualBuilder(EOMCCEigenVecType &eigenVecType){\n'
    text += '      if (this->eomSettings.hbar_type == EOM_HBAR_TYPE::IMPLICIT) {\n'
    text += '        this->funcEOM = [this, &eigenVecType]( size_t nVec, SolverVectors<dcomplex> &V,\n'
    text += '            SolverVectors<dcomplex> &AV) {\n'
    text += '\n'
    text += '          EOMCCSDVectorSet<dcomplex> *V_ptr = nullptr, *AV_ptr = nullptr;\n'
    text += '          size_t Vshift = 0, AVshift = 0;\n'
    text += '          try {\n'
    text += '            V_ptr = &dynamic_cast<EOMCCSDVectorSet<dcomplex>&>(V);\n'
    text += '          } catch(const std::bad_cast& e) {\n'
    text += '            SolverVectorsView<dcomplex>& V_view = dynamic_cast<SolverVectorsView<dcomplex>&>(V);\n'
    text += '            V_ptr = &dynamic_cast<EOMCCSDVectorSet<dcomplex>&>(V_view.getVecs());\n'
    text += '            Vshift = V_view.shift();\n'
    text += '          }\n'
    text += '\n'
    text += '          try {\n'
    text += '            AV_ptr = &dynamic_cast<EOMCCSDVectorSet<dcomplex>&>(AV);\n'
    text += '          } catch(const std::bad_cast& e) {\n'
    text += '            SolverVectorsView<dcomplex>& AV_view = dynamic_cast<SolverVectorsView<dcomplex>&>(AV);\n'
    text += '            AV_ptr = &dynamic_cast<EOMCCSDVectorSet<dcomplex>&>(AV_view.getVecs());\n'
    text += '            AVshift = AV_view.shift();\n'
    text += '          }\n'
    text += '\n'
    text += '          for (size_t i = 0; i < nVec; i++) {\n'
    text += '            const EOMCCSDVector<dcomplex> &Vi = V_ptr->get(i + Vshift);\n'
    text += '            EOMCCSDVector<dcomplex> &AVi = AV_ptr->get(i + AVshift);\n'
    text += '            buildSigma(Vi, AVi, eigenVecType);\n'
    text += '            TA::get_default_world().gop.fence();\n'
    text += '            AVi.enforceSymmetry();\n'
    text += '          }\n'
    text += '\n'
    text += '        }; // implicit sigmaBuilder\n'
    text += '        return this->funcEOM;\n'
    text += '      }\n'
    text += '\n'
    text += '  }\n'
    return output_content + text + build_pc_text


# namespace at the beginning and the end
def add_namespace(output_content):
    text  = '#pragma once \n'
    text += '#include <chronusq_sys.hpp>\n'
    text += '#include <coupledcluster.hpp>\n'
    #text += '#include <coupledcluster/EOMDIP_active.hpp>\n'
    #text += '#include <util/math.hpp>\n'
    #text += '#include <cqlinalg.hpp>\n'
    #text += '#include <util/matout.hpp>\n'
    #text += '#include <functional>\n'
    #text += '#include <util/timer.hpp>\n'
    #text += '#include <coupledcluster/EOMCCSDVector.hpp>\n'
    #text += '#include <itersolver/davidson.hpp>\n'
    #text += '#include <itersolver.hpp>\n'
    #text += '#include <coupledcluster/EOMDIP_3h1p_full_Hbar.hpp>\n'
    text += '\n'
    text += 'namespace ChronusQ{\n'
    return text + output_content + '\n}\n'

# empty constructor 
def add_constructor(output_content):
    text  = ''

    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  EOMDIP_4h2pCCSDT<MatsT,IntsT>::EOMDIP_4h2pCCSDT(const SafeFile &savFile,\n'
    text += '    CCIntermediates<MatsT> &intermediates,\n'
    text += '    const EOMSettings &eomSettings,\n'
    text += '    const CoupledClusterSettings &ccSettings):\n'
    text += '    EOMCCBase<MatsT,IntsT>(savFile, intermediates,eomSettings,ccSettings),\n'
    text += '    vLabel_(intermediates.vLabel), oLabel_(intermediates.oLabel),\n'
    text += '    reused_(intermediates.sigmaOps),\n'
    text += '    tmps_(intermediates.tempOps),\n'
    text += '    T1_(intermediates.T->get_tensor("OneBody")), T2_(intermediates.T->get_tensor("TwoBody")), T3_(intermediates.T->get_tensor("ThreeBody")) {\n'

    text += '    TAManager &TAmanager = TAManager::get();\n\n'
 
    text += '    // without L, we do not need D\n'
    text += '    if (eomSettings.oscillator_strength == false) {\n'
    text += '      if(intermediates.D_ai)   TAmanager.free("vo", std::move(intermediates.D_ai), true);\n'
    text += '      if(intermediates.D_abij) TAmanager.free("vvoo", std::move(intermediates.D_abij), true);\n'
    text += '    }\n\n'
 
    text += '    nV_ = TAmanager.getRange(vLabel_).extent();\n'
    text += '    nO_ = TAmanager.getRange(oLabel_).extent();\n'
    text += '    nV2shift_ = nV_ * (nV_ - 1) / 2;\n'
    text += '    nO2shift_ = nO_ * (nO_ - 1) / 2;\n'
    text += '    nO3shift_ = nO_ * (nO_ - 1) * (nO_ - 2) / 6;\n'
    text += '    nO4shift_ = nO_ * (nO_ - 1) * (nO_ - 2) * (nO_ - 3) / 24;\n'
 
    text += '    this->Hbar_dimension_offsets.emplace("TwoBody", 0);\n'
    text += '    this->Hbar_dimension_offsets.emplace("ThreeBody", nO2shift_);\n'
    text += '    this->Hbar_dimension_offsets.emplace("FourBody", nO2shift_ + nO3shift_*nV_);\n'
    text += '    this->Hbar_dim = nO2shift_ + nO3shift_*nV_ + nO4shift_*nV2shift_;\n'
    text += '    this->outOfBound_ = (this->Hbar_dim + 1) * (this->Hbar_dim + 1);\n'
 
    text += '    abIndices_.clear();\n'
    text += '    abIndices_.resize(nV_, std::vector<size_t>(nV_, outOfBound_));\n'
    text += '    ijIndices_.clear();\n'
    text += '    ijIndices_.resize(nO_, std::vector<size_t>(nO_, outOfBound_));\n'
    text += '    ijkIndices_.clear();\n'
    text += '    ijkIndices_.resize(nO_, std::vector<std::vector<size_t>>(nO_, std::vector<size_t>(nO_, outOfBound_)));\n'
    text += '    ijklIndices_.clear();\n'
    text += '    ijklIndices_.resize(nO_, std::vector<std::vector<std::vector<size_t>>>(nO_, std::vector<std::vector<size_t>>(nO_, std::vector<size_t>(nO_, outOfBound_))));\n'
 
    text += '    size_t idx = 0;\n'
    text += '    for (size_t b = 0; b < nV_; b++) {\n'
    text += '      for (size_t a = 0; a < b; a++) {\n'
    text += '        abIndices_[a][b] = idx;\n'
    text += '        abIndices_[b][a] = idx++;\n'
    text += '      }\n'
    text += '    }\n\n'

    text += '    idx = 0;\n'
    text += '    for (size_t j = 0; j < nO_; j++) {\n'
    text += '      for (size_t i = 0; i < j; i++) {\n'
    text += '        ijIndices_[i][j] = idx;\n'
    text += '        ijIndices_[j][i] = idx++;\n'
    text += '      }\n'
    text += '    }\n\n'

    text += '    idx = 0;\n'
    text += '    for (size_t k = 0; k < nO_; k++) {\n'
    text += '      for (size_t j = 0; j < k; j++) {\n'
    text += '        for (size_t i = 0; i < j; i++) {\n'
    text += '          ijkIndices_[i][j][k] = idx;\n'
    text += '          ijkIndices_[j][i][k] = idx;\n'
    text += '          ijkIndices_[i][k][j] = idx;\n'
    text += '          ijkIndices_[j][k][i] = idx;\n'
    text += '          ijkIndices_[k][i][j] = idx;\n'
    text += '          ijkIndices_[k][j][i] = idx++;\n'
    text += '        }\n'
    text += '      }\n'
    text += '    }\n\n'

    text += '    idx = 0;\n'
    text += '    for (size_t l = 0; l < nO_; l++) {\n'
    text += '      for (size_t k = 0; k < l; k++) {\n'
    text += '        for (size_t j = 0; j < k; j++) {\n'
    text += '          for (size_t i = 0; i < j; i++) {\n'
    text += '            ijklIndices_[i][j][k][l] = idx;\n'
    text += '            ijklIndices_[j][i][k][l] = idx;\n'
    text += '            ijklIndices_[i][k][j][l] = idx;\n'
    text += '            ijklIndices_[j][k][i][l] = idx;\n'
    text += '            ijklIndices_[k][i][j][l] = idx;\n'
    text += '            ijklIndices_[k][j][i][l] = idx;\n'
    text += '            ijklIndices_[i][j][l][k] = idx;\n'
    text += '            ijklIndices_[j][i][l][k] = idx;\n'
    text += '            ijklIndices_[i][k][l][j] = idx;\n'
    text += '            ijklIndices_[j][k][l][i] = idx;\n'
    text += '            ijklIndices_[k][i][l][j] = idx;\n'
    text += '            ijklIndices_[k][j][l][i] = idx;\n'
    text += '            ijklIndices_[i][l][j][k] = idx;\n'
    text += '            ijklIndices_[j][l][i][k] = idx;\n'
    text += '            ijklIndices_[i][l][k][j] = idx;\n'
    text += '            ijklIndices_[j][l][k][i] = idx;\n'
    text += '            ijklIndices_[k][l][i][j] = idx;\n'
    text += '            ijklIndices_[k][l][j][i] = idx;\n'
    text += '            ijklIndices_[l][i][j][k] = idx;\n'
    text += '            ijklIndices_[l][j][i][k] = idx;\n'
    text += '            ijklIndices_[l][i][k][j] = idx;\n'
    text += '            ijklIndices_[l][j][k][i] = idx;\n'
    text += '            ijklIndices_[l][k][i][j] = idx;\n'
    text += '            ijklIndices_[l][k][j][i] = idx++;\n'
    text += '          }\n'
    text += '        }\n'
    text += '      }\n'
    text += '    }\n\n'

    text += '    this->tensor_builder_.push_back(std::string(""));\n'
    text += '    this->tensor_builder_.push_back(std::string({intermediates.oLabel,intermediates.oLabel}));\n'
    text += '    this->tensor_builder_.push_back(std::string("TwoBody"));\n'
    text += '    this->tensor_builder_.push_back(std::string({intermediates.vLabel}));\n'
    text += '    this->tensor_builder_.push_back(std::string({intermediates.oLabel,intermediates.oLabel,intermediates.oLabel}));\n'
    text += '    this->tensor_builder_.push_back(std::string("ThreeBody"));\n\n'
    text += '    this->tensor_builder_.push_back(std::string({intermediates.vLabel,intermediates.vLabel}));\n'
    text += '    this->tensor_builder_.push_back(std::string({intermediates.oLabel,intermediates.oLabel,intermediates.oLabel,intermediates.oLabel}));\n'
    text += '    this->tensor_builder_.push_back(std::string("FourBody"));\n\n'
    text += '  }\n\n'


    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  inline size_t EOMDIP_4h2pCCSDT<MatsT,IntsT>::toCompoundS(size_t i, size_t j) const {\n'
    text += '    if (i >= nO_ or j >= nO_)\n'
    text += '      return outOfBound_;\n'
    text += '    return ijIndices_[i][j];\n'
    text += '  }\n'
    text += '\n'
    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  inline size_t EOMDIP_4h2pCCSDT<MatsT,IntsT>::toCompoundD(size_t a, size_t i, size_t j, size_t k) const {\n'
    text += '  size_t ijk = ijkIndices_[i][j][k];\n'
    text += '  if (ijk == outOfBound_)\n'
    text += '    return outOfBound_;\n'
    text += '  return a + ijk * nV_;\n'
    text += '  }\n\n'
    text += '  template <typename MatsT, typename IntsT>\n'
    text += '  inline size_t EOMDIP_4h2pCCSDT<MatsT,IntsT>::toCompoundT(size_t a, size_t b, size_t i, size_t j, size_t k, size_t l) const {\n'
    text += '  size_t ijkl = ijklIndices_[i][j][k][l];\n'
    text += '  size_t ab = abIndices_[a][b];\n'
    text += '  if (ijkl == outOfBound_ or ab == outOfBound_)\n'
    text += '      return outOfBound_;\n'
    text += '    return ab + ijkl * nV2shift_;\n'
    text += '  }\n\n'


    return text + output_content

def to_chronus_string(graph, class_name="REPLACEME", is_active=False):

    # Read the content of the input file
    input_content = graph.str("c++")

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
        output_content = re.sub(r'/+ Shared  Operators /+\s*\n','  void EOMDIP_4h2pCCSDT<MatsT,IntsT>::formEOMIntermediates() {\n\n    TAManager &TAmanager = TAManager::get();\n\n', output_content)
        output_content = re.sub(r'\n\s*/+ End of Shared Operators /+\s*\n','\n\n  }\n\n', output_content)

    match = re.compile("/+ Evaluate Equations /+\s*\n").search(output_content)
    if match:
        output_content = re.sub(r'/+ Evaluate Equations /+\s*\n','  void EOMDIP_4h2pCCSDT<MatsT,IntsT>::buildSigma(const EOMCCSDVector<MatsT> &V, EOMCCSDVector<MatsT> &HV, EOMCCEigenVecType vecType) const {\n\n    TAManager &TAmanager = TAManager::get();\n\n', output_content)

    # print(output_content)
    # exit()
    output_content = replace_conj_strings_option1(output_content) # must happen before the block replacement
    #output_content = replace_r2r3_block_strings(output_content) # must happen before the general block replacement
    #output_content = replace_block_strings(output_content)
    output_content = replace_free_strings(output_content)
    #output_content = add_scalar_reserve(output_content) # must happen before the scalars substitution
    #output_content = remove_scalar_lines(output_content)
    output_content = first_LHS_direct_equal(output_content) #must happen after remove_scalar_lines, before add_tenser_definition and add_malloc
    output_content = add_malloc_strings(output_content)
    output_content = replace_tmp_spaces(output_content) # must happen after replace free and add_malloc
    output_content = add_fence_lines(output_content)
    #output_content = re.sub(r'scalars_\["([0-9]+)"\]',r'scalars_[\1]', output_content)
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
    output_content = re.sub(r'sigmaR2_ij','sigmaR2', output_content)
    output_content = re.sub(r'sigmaR3_ijka','sigmaR3', output_content)
    output_content = re.sub(r'sigmaR4_ijklba','sigmaR4', output_content)
    output_content = re.sub(r'eri\["oooo"\]','this->antiSymMoints["oooo"]', output_content)
    output_content = re.sub(r'eri\["vooo"\]','this->antiSymMoints["vooo"]', output_content)
    output_content = re.sub(r'eri\["vvoo"\]','this->antiSymMoints["vvoo"]', output_content)
    output_content = re.sub(r'eri\["vovo"\]','this->antiSymMoints["vovo"]', output_content)
    output_content = re.sub(r'eri\["vovv"\]','this->antiSymMoints["vovv"]', output_content)
    output_content = re.sub(r'eri\["vvvo"\]','this->antiSymMoints["vvvo"]', output_content)
    output_content = re.sub(r'eri\["vvvv"\]','this->antiSymMoints["vvvv"]', output_content)
    output_content = re.sub(r'Id','this->Id', output_content)

    # re-organize reusetmps mallocs into a new function
    match = re.compile("r'\s*reused_.emplace[^;]+;'").search(output_content)
    if match:
        output_content = extract_malloc_reusetmps(output_content) # must happen after 1. add_malloc 2. removing everything until ##### Scalars #####

    ## add definition of R vector and sigmaR vector
    #pattern = re.compile(r'([0-3])h([01])p\.')
    #n_hole_str,n_particle_str = pattern.search(input_file_path).groups()
    #n_hole     = int(n_hole_str)
    #n_particle = int(n_particle_str)


    output_content = add_tenser_definition(output_content)
    output_content = add_constructor(output_content)
    output_content = add_build_diag(output_content)
    output_content = add_build_pc(output_content)
    output_content = add_destructor(output_content)

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
    
