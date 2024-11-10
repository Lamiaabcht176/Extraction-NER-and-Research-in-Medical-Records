import os
import pickle
import pprint
from nltk import RegexpTokenizer
from PyPDF2 import PdfReader
import cv2
import numpy as np
import magic

# --- File Reading Functions ---
def read_file(path, file, ext):    
    """
    Reads a file and returns its content as a list of lines.

    The function reads a file from the specified path with the given extension 
    and file name, and returns the content as a list of lines.

    Args:
        path (str): The directory path where the file is located.
        ext (str): The file extension (e.g., '.txt').
        file (str): The name of the file to be read.

    Returns:
        list: A list of lines from the file.
    """
    f = os.path.join(path, file + ext)
    with open(f, 'rt', encoding='utf-8') as myfile:
        data = myfile.readlines()
    return data

# --- Data Handling Functions ---
def save_pickle(data, file):
    with open(file + ".pkl", "wb") as pick_file:
        pickle.dump(data, pick_file)

def load_pickle(file):
    with open(file + ".pkl", "rb") as pick_file:
        data = pickle.load(pick_file)
    return data

# --- Data Processing Functions ---
def ann_text2dict(lines):
    """
    Converts .ANN file content into a dictionary.

    Ignores comment lines, then splits each non-comment line into three parts: 
    line ID (dictionary key), label (with possible semicolon formatting), 
    and text. Returns a dictionary with the processed information.

    Args:
        lines (list): List of lines from the .ANN file.

    Returns:
        dict: Processed dictionary with line IDs as keys and label-text pairs.
    """
    d = {}
    for l in lines:
        if not l.startswith('#'):
            t = l.split('\t')
            if ';' in t[1]:
                t[1] = t[1].replace(';',' ; ')
            d[t[0]] = {
                'label': t[1].split(' '),
                'text': t[2].replace('\n','')
            }
    return d



def is_subrange(r1, r2):
    """
    Checks if range r1 is contained within range r2.
    
    Args:
        r1 (tuple): A tuple representing the first range (start, end).
        r2 (tuple): A tuple representing the second range (start, end).
    
    Returns:
        bool: True if r1 is a subrange of r2, False otherwise.
    """
    [a1, b1] = r1
    [a2, b2] = r2
    if int(a2) <= int(a1) and int(b1) <= int(b2): # [a1,b1] subrange of [a2,b2]
        return True
    else:
        return False

def contained(key, ann_dic):
    """
    Verifies if the segment identified by key1 is contained within the segment identified by key2.
    
    Args:
        segments (dict): A dictionary where the keys are segment identifiers and the values are tuples 
                          representing the (start, end) index ranges of each segment.
        key1 (str): The key of the segment that we want to check if it's contained.
        key2 (str): The key of the segment that we want to check if it contains the first segment.
    
    Returns:
        bool: True if the segment with key1 is contained within the segment with key2, False otherwise.
    """
    piv = ann_dic[key]['label']
    for k in ann_dic.keys():
        if not k == key:
            lab_field = ann_dic[k]['label']
            if len(lab_field) == 3:
                if len(piv) == 3:
                    if is_subrange([piv[1], piv[2]], [lab_field[1], lab_field[2]]):
                        return True
    return False

def remove_contained(ann_dic):
    """
    Removes segments whose ranges are fully contained within the ranges of other segments.
    
    Args:
        segments (dict): A dictionary where the keys are segment identifiers and the values are tuples 
                          representing the (start, end) index ranges of each segment.
    
    Returns:
        dict: A dictionary with the contained segments removed.
    """
    lrem = []
    for k in ann_dic.keys():
        if contained(k, ann_dic):
            lrem.append(k)
    for i in lrem:
        del ann_dic[i]
    return ann_dic

def cont_ncont(ann_dic):
    nnc = 0 # number of non continuous segments
    ntot = 0 # total number of segments
    for k in ann_dic.keys():
        piv = ann_dic[k]['label']
        ntot += 1
        if not len(piv) == 3:
            nnc += 1
    return [nnc, ntot]

def count_non_continuous(set):
    """
    Counts the number of non-continuous segments in a single annotation dictionary.
    
    Args:
        ann_dic (dict): A dictionary where keys are segment identifiers and values are tuples (start, end) representing the range of each segment.
    
    Returns:
        tuple: A tuple containing:
               - The total number of segments
               - The number of non-continuous segments
    """
    ldics = load_pickle(set + '_ann_dics')
    cnc = 0 # cont non continuous segments
    ctot = 0 # cont total segments
    for i in range(len(ldics)):
        [nnc,ntot] = cont_ncont(ldics[i])
        cnc+=nnc
        ctot+=ntot
    print("set",set)
    print("Number of non continuous segments",cnc,'%',(cnc/ctot)*100)
    print("Total number of segments",ctot)

def simple_dic(ann_dic):
    list = []
    for t in ann_dic.keys():
        pt = ann_dic[t]
        dic = {
            'label' : pt['label'][0],
            'range' : pt['label'][1:],
            'text' : pt['text']
        }
        list.append(dic)
    return list

def ldic2ltup(i, listai):
    """
    Converts a list of dictionaries into a list of tuples.
    
    Args:
        ldic (list): A list of dictionaries, where each dictionary contains key-value pairs.
        
    Returns:
        list: A list of tuples, where each tuple contains (key, value) from the dictionaries.
    """
    ann_dic = listai['ann_dic']
    txt = listai['txt'][0]
    ltup = []
    for dic in ann_dic:
        etiq = dic['label']
        rango = dic['range']
        if len(rango) < 3:
            a = int(rango[0])
            b = int(rango[1])
            ltup.append((a, b, etiq, txt[a:b]))
    ltup.sort()
    return ltup

# --- File Collection Functions ---
def collect_files(path, set_name):
    """
    Collects names of .TXT and .ANN files from the specified directory.

    Extracts filenames (without extensions) into two separate lists: one for .TXT files
    and another for .ANN files. These lists are saved as pickle files for efficient loading
    in future operations. Pickle is used for serializing the lists into byte streams that can
    be stored and deserialized later.

    Args:
        path (str): Directory to search for .TXT and .ANN files.
    """
    dirs = os.listdir(path)
    ltxt = []
    lann = []
    for f in dirs:
        if f.endswith('.txt'):
            f = f.replace('.txt','')
            ltxt.append(f)
        elif f.endswith('.ann'):
            f = f.replace('.ann','')
            lann.append(f)
    save_pickle(ltxt, set_name + '_txt')
    save_pickle(lann, set_name + '_ann')

def ann_files2dict(pic_file, path, set_name):
    """
    Processes all .ANN files in the specified path and converts their content into dictionaries.

    Each dictionary represents the processed content of a file and is collected into a list,
    which is then saved as a pickle file for later use.

    Args:
        path (str): Directory containing the .ANN files.
    """
    lann = load_pickle(pic_file)
    lnew = []
    c = 0
    for ann in lann:
        data = read_file(path, ann, ".ann")
        dic = ann_text2dict(data)
        c += 1
        lnew.append(dic)
    save_pickle(lnew, set_name + '_ann_dics')
    return lnew

def mix_txt_ann(pic_file, path, set_name):
    """
    Combines the text from the .TXT file and the annotation dictionary from the .ANN file into a single dictionary.
    
    Args:
        txt_path (str): Path to the directory containing the .TXT files.
        ann_path (str): Path to the directory containing the .ANN files.
    
    Returns:
        list: A list of dictionaries where each dictionary contains:
              - 'txt': Text from the corresponding .TXT file.
              - 'ann_dic': Annotation dictionary processed from the .ANN file.
    """
    ltxt = load_pickle(pic_file)
    lnew = []
    for i in range(len(ltxt)):
        data = read_file(path, ltxt[i], ".txt")
        ldics = load_pickle(set_name + '_ann_dics')
        ann_dic = remove_contained(ldics[i])
        ndic = {
            'txt': data,
            'ann_dic': simple_dic(ann_dic)
        }
        lnew.append(ndic)
    save_pickle(lnew, set_name + '_txt_ann')

def complete_segments(set_name):
    """
    Tags the segments that are not already tagged with 'NONE'.
    
    Args:
        annotation (dict): A dictionary where keys represent segment identifiers and values are lists of tags.
        
    Returns:
        dict: A dictionary where untagged segments are tagged with 'NONE'.
    """
    lista = load_pickle(set_name + '_txt_ann')
    newl = []
    for i in range(len(lista)):
        txt = lista[i]['txt'][0]
        ltup = ldic2ltup(i, lista[i])
        ltup.sort()
        lt1 = []
        if len(ltup) == 0:
            tup = (0, len(txt), 'NONE', txt)
            lt1.append(tup)
            continue
        if ltup[0][0] > 0:
            a = 0
            b = ltup[0][0] - 1
            tup = (a, b, 'NONE', txt[a:b])
            lt1.insert(0, tup)
        for j in range(len(ltup) - 1):
            if ltup[j][1] + 1 == ltup[j+1][0]: # consecutive
                lt1.append(ltup[j])
            else: # non consecutive
                a = ltup[j][1] + 1
                lt1.append(ltup[j])
                b = ltup[j+1][0] - 1
                tup = (a, b, 'NONE', txt[a:b])
                lt1.append(tup)
        lt1.append(ltup[-1])
        if ltup[-1][1] < len(txt):
            a = ltup[-1][1] + 1
            b = len(txt) - 1
            tup = (a, b, 'NONE', txt[a:b])
            lt1.append(tup)
        lt1.sort()
        newl.append({
            'ann_dic': lt1,
            'txt': txt
        })
    save_pickle(newl, set_name + '_txt_ann2')

# --- Tokenization Functions ---
def ldic2ltok_lab(lsent):
    ls_tok_lab = []
    toknizer = RegexpTokenizer(r'''\w'|\w+|[^\w\s]''')
    for sent in lsent:
        ltup = sent['ann_dic']
        lt_tok_lab = []
        for (a, b, lab, txt) in ltup:
            lts = toknizer.tokenize(txt)
            ltoks = [(t, lab) for t in lts]
            lt_tok_lab.extend(ltoks)
        ls_tok_lab.append(lt_tok_lab)
    return ls_tok_lab

# --- PDF File Checking Functions ---
def is_pdf(file_path):
    mime = magic.Magic(magic_file="C:/Users/lamia/Downloads/magic.mgc")
    file_type = mime.from_file(file_path)
    return "PDF document" in file_type

def is_scanned_pdf(pdf_path, threshold=0.2):
    pdf_reader = PdfReader(pdf_path)
    for page_num in range(len(pdf_reader.pages)):
        text_content = pdf_reader.pages[page_num].extract_text()
        img = np.array(list(map(ord, text_content)), dtype=np.uint8).reshape(-1, 1)

        try:
            gradient_mean = cv2.Laplacian(img, cv2.CV_64F).var()
        except cv2.error:
            return True

        if gradient_mean < threshold:
            return True

    return False

# --- Main Execution --- 
if __name__ == "__main__":
    path_train = "C:\\Users\\lamia\\Desktop\\Extraction-NER-Recherche\\Analyses\\train"
    set_name = 'train'
    
    # Collect Files
    collect_files(path_train, set_name)
    
    # Process .ann files
    lnew = ann_files2dict(set_name + '_ann', path_train, set_name)
    pprint.pprint(lnew)
    
    # Create text and annotations mixed data
    mix_txt_ann(set_name + '_txt', path_train, set_name)
    
    # Complete segments in the data
    complete_segments(set_name)
    
    # Tokenize and save labeled tokens
    ls_tok_lab = ldic2ltok_lab(load_pickle(set_name + '_txt_ann2'))
    save_pickle(ls_tok_lab, set_name + '_txt_ann3')
