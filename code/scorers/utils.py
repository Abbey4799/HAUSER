import re
import benepar, spacy
import nltk

import json
import os
import re
import copy

from stanfordcorenlp import StanfordCoreNLP

nlp = spacy.load('en_core_web_sm')
if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("./dependency/benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "./dependency/benepar_en3"})

nlp_rep = StanfordCoreNLP('./dependency/stanford-corenlp-4.2.0')


def get_parent_list(span):
    parent_list = []
    while True:
        pp = span._.parent
        if pp == None:
            break
        parent_list.append((pp._.labels,str(pp)))
        span = pp
    return parent_list
        
def rep(rr,txt):
    coref_list = nlp_rep.coref(txt)
    for cc in coref_list:
        lk_list = [i[-1] for i in cc]
        if rr in lk_list:
            lk_list.remove(rr)
            if len(lk_list) == 0:
                continue
            return lk_list[0]
    return '-'
    
def get_comp_lefttreeall(string, old_info,s2i,span_list):
    for _ in s2i[string]:
        ss = span_list[_]
        cand = '-'
        cand_child = ss
        # 先从子树锁定NP部分，不然会被修饰词干扰
        for aa in ss._.children:
            if len(aa._.labels) > 0 and aa._.labels[0] == 'NP':
                cand_child = aa
                break

        cand_child = ss
        while True:
            np_flag = 0
            np_pp_flag = 0
            svp_flag = 0
            null_flag = 0
            if len(list(ss._.children)) == 0:
                break
            for aa in ss._.children:
                if len(aa._.labels) > 0:
                    null_flag = 1
                if len(aa._.labels) > 0 and aa._.labels[0] not in ['S','VP','SBAR']:
                    svp_flag = 1
                if len(aa._.labels) > 0:
                    if aa._.labels[0] not in ['NP','PP'] or (aa._.labels[0] == 'PP' and str(aa)[:4] == 'like'):
                        np_pp_flag = 1
                if np_flag == 0 and len(aa._.labels) > 0 and aa._.labels[0] == 'NP':
                    s_list = ['every time','each time','one day']
                    ___ = 0
                    for _ in s_list:
                        if str(aa).lower().startswith(_):
                            ___ = 1
                            break
                    if ___ == 0 :
                        cand_child = aa
                        np_flag = 1

            if null_flag == 1 and svp_flag == 0:
                cand = '-'
                break

            if np_flag == 0:
                cand = str(ss)
                break

              # 有除NP和PP之外的
            if np_pp_flag == 1:
                # 一些特殊的NP
                if str(cand_child).endswith('\'s') or str(cand_child).endswith('’s'):
                    if not str(cand_child).startswith('like '):
                        cand = str(ss)
                        break
                cand = str(cand_child)
            else:
                if str(cand_child).endswith('\'s') or str(cand_child).endswith('’s'):
                    if not str(cand_child).startswith('like '):
                        cand = str(ss)
                        break
                start_list = ['of', 'and' ]
                test_str = str(ss)[str(ss).index(str(cand_child)) + len(str(cand_child)):].strip()
                test_flag = 0
                for startt in start_list:
                    if test_str.startswith(startt):
                        cand = str(ss)
                        test_flag = 1
                        break
                    
                if test_flag == 1:
                    break

                # 且没有打比方的句子
                if str(ss)[:5] == 'like ':
                    cand = str(ss)[5:]
                else:
                    cand = str(ss)
            
            ss = cand_child

        if cand in old_info and cand not in string[:string.index(old_info)]:
            cand = '-'
    return cand



def get_component_tree(text):
    '''
        component parsing based on constituency tree
    '''
    
    comps_list = []
    comps = []
    #排除掉哪些可能导致识别不准的因素
    text = text.replace('.,','\.,')
    text = re.sub(' +', ' ', text)
    text = re.sub(u"\\[\d*\\]","",text)
    text = re.sub(u"\\{\d*\\}","",text)
    text = re.sub(u"\\<.*?\\>","",text)
    text = re.sub(u"[`#_=�▷“”\"♥ℂ•â˜~|^*]+","",text)
    text = re.sub(u"\d*:\d*","",text)
    text = re.sub(u"\d*:\d*","",text)

    text = re.sub(u"[-–—]\s*",",",text)
    text = re.sub(u",+",",",text)
    text = re.sub(u"….",".",text)


    text = text.replace('[]','').replace(' AM ','').replace(' PM ', '')
    text = text.strip()
    

    
    doc = nlp(text).sents
    for sent in doc:
        if 'like' not in str(sent):
            continue

        # 生成like_list
        sent_ = nltk.word_tokenize(str(sent))
        like_sent_list = []
        for idx,ww in enumerate(sent_):
            if ww == 'like':
                new_sent = copy.copy(sent_)
                new_sent[idx] = '[MASK]'
                new_sent.insert(idx,'as')
                new_sent.insert(idx + 2,'as')
                like_sent_list.append(' '.join(new_sent))

        s2i = {}
        i2s = {}
        span_list = []
        for idx,cc in enumerate(sent._.constituents):
            span_list.append(cc)
            i2s[idx] = str(cc)
            if str(cc) not in s2i:
                s2i[str(cc)] = []
            s2i[str(cc)].append(idx)       

        if 'like' not in s2i:
            continue

        for like_idx,ii in enumerate(s2i['like']):
            topic = '-'
            vehicle = '-'
            pp_list = get_parent_list(span_list[ii])

            # 判断有没有特殊情况
            S_SBAR_NP_flag = 0
            S_SBAR_NP_idx = 0
            VP_VPorS_VP_flag = 0
            VP_VPorS_VP_idx = 0
            test_list = [i[0][0] for i in pp_list]
            tmp = ' '.join(test_list)
            if ' S SBAR NP ' in tmp:
                _ = len(tmp[:tmp.find(' S SBAR NP ')].split(' '))
                if not pp_list[_ + 1][1].startswith('where') and \
                    not pp_list[_ + 1][1].startswith('when'):
                    S_SBAR_NP_flag = 1
                    S_SBAR_NP_idx = _ + 2

            VP_VPorS_VP_list = [' VP VP VP ', ' VP S VP ', ' VP S S ',' VP VP S ']
            for vv in VP_VPorS_VP_list:
                if vv in tmp and tmp.find(vv) == tmp.find(' VP '):
                    _ = len(tmp[:tmp.find(vv)].split(' '))
                    if pp_list[_ + 1][1].startswith('to ') and \
                        pp_list[_ + 2][1].split(' ').index('to') > 1 and \
                            'by' not in pp_list[_ + 2][1].split(' ')[:pp_list[_ + 2][1].split(' ').index('to')]:
                        VP_VPorS_VP_flag = 1
                        VP_VPorS_VP_idx = len(tmp[:tmp.find(vv)].split(' ')) + 2
            
            flag = 0
            if pp_list[0][0][0] == 'PP':
                event = '-'
                for idx,ppp in enumerate(pp_list):
                    if flag == 0 and ppp[0][0] == 'PP':
                        # 左子树上第一个
                        vehicle = get_comp_lefttreeall(ppp[1],'',s2i,span_list)
                        flag = 1


                    if 'VP' in ppp[0] and event == '-':
                            event = nltk.word_tokenize(ppp[1])[0]

                    if flag == 1:
                        if ppp[0][0] == 'SBAR' and \
                            (ppp[1].startswith('what') or ppp[1].startswith('how')):
                            topic = '-'
                            flag = 2
                
                        if S_SBAR_NP_flag == 1:
                            topic = get_comp_lefttreeall(pp_list[S_SBAR_NP_idx][1],pp_list[S_SBAR_NP_idx-1][1],s2i,span_list)
                            flag = 2
                        elif VP_VPorS_VP_flag == 1:
                            topic = get_comp_lefttreeall(pp_list[VP_VPorS_VP_idx][1],pp_list[VP_VPorS_VP_idx-1][1],s2i,span_list)
                            if ' like ' in topic or topic == vehicle:
                                topic = '-'
                                VP_VPorS_VP_flag = 0
                                continue
                            else:
                                flag = 2
                        else:
                            # 先碰到S
                            if (len(ppp[0]) == 1 and ppp[0][0] == 'S') or \
                                (len(ppp[0]) == 2 and 'S' in ppp[0] and 'SBAR' in ppp[0]):
                                topic = get_comp_lefttreeall(ppp[1],pp_list[idx-1][1],s2i,span_list)
                                flag = 2
                            # 先碰到NP
                            if ppp[0][0] == 'NP':
                                topic = get_comp_lefttreeall(ppp[1],pp_list[idx-1][1],s2i,span_list)
                                flag = 2

                    if re.sub(u"[./]+","",topic).isdigit():
                        topic = '-'
                        S_SBAR_NP_flag = 0
                        VP_VPorS_VP_flag = 0
                        flag = 1
                        continue

                    tmp = topic.strip()
                    _flag = 1
                    for ___ in ['‘','’','\'',',','，']:
                        if tmp.startswith(___):
                            _flag = 0
                    if _flag == 0:
                        topic = '-'
                        S_SBAR_NP_flag = 0
                        VP_VPorS_VP_flag = 0
                        flag = 1
                        continue

            if topic in ['it','It']:
                tmp = str(sent).replace(' "',' ""')
                tmp = rep(topic, tmp)
                if tmp != '-':
                    topic = tmp
        
            comps_list.append((topic, vehicle, '-'))
    return comps_list



def clean(word,mode = 'vehicle'):
    # 合并多个空格
    word = word.lower()
    new_word = re.sub(' +', ' ', word)

    # 去除没意义的特殊字符
    stp_words = u"[\"'”“’‘[]_*-•…/><=]+"
    new_word = re.sub(stp_words,"",new_word)

    # 去除首尾的修饰词
    new_word = new_word.strip()
    if mode == 'vehicle':
        new_word = re.sub(u"^(like) ","",new_word)
    new_word = re.sub(u"^(like|your|my|her|his|its|their|our|the|that|this|these|thy|those|all|other|another|what|every|a|an|even|just|only|each) ","",new_word)

    return new_word


def readjsonl(datapath):
    res = []
    with open(datapath, "r", encoding='utf-8') as f:
        for line in f.readlines():
            res.append(json.loads(line))
    return res


def writejsonl(data, datapath):
    with open(datapath, "w", encoding='utf-8') as f:
        for item in data:
            json_item = json.dumps(item, ensure_ascii=False)
            f.write(json_item + "\n")

def writejson(data, json_path):
    json_str = json.dumps(data, indent=4, ensure_ascii=False)
    with open(json_path, "w", encoding='utf-8') as json_file:
        json_file.write(json_str)

def readjson(datapath):
    with open(datapath, "r", encoding='utf-8') as f:
        res = json.load(f)
    return res


def check_folder(path):
    if not os.path.exists(path):
        print(f"{path} not exists, create it")
        os.makedirs(path)

def get_name(name, pattern, mode = 0):
    match = re.search(pattern, name)

    if match:
        extracted_content = match.group(mode)
        return extracted_content
    else:
        print("未找到匹配的内容")