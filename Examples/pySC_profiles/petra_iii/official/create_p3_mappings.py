import at
import yaml

#ring = at.load_mat('data/p3_low_beta.mat')
ring = at.load_mat('data/p3_v24.mat')


def find_index_by_commonname(name, ring):
    for ii, el in enumerate(ring):
        if el.CommonName == name:
            return ii
    raise Exception(f"{name=} not found in at ring.")

BPM_names = [name.strip() for name in open('data/BPM_names.txt').readlines()]
BPM_mapping = {name: find_index_by_commonname(name, ring) for name in BPM_names}
yaml.safe_dump(BPM_mapping, sort_keys=False, stream=open('data/p3_BPM_mapping.yaml', 'w'))

HCM_names = [name.strip() for name in open('data/HCM_names.txt').readlines()]
HCM_mapping = {name: find_index_by_commonname(name, ring) for name in HCM_names}
yaml.safe_dump(HCM_mapping, sort_keys=False, stream=open('data/p3_HCM_mapping.yaml', 'w'))

VCM_names = [name.strip() for name in open('data/VCM_names.txt').readlines()]
VCM_mapping = {name: find_index_by_commonname(name, ring) for name in VCM_names}
yaml.safe_dump(VCM_mapping, sort_keys=False, stream=open('data/p3_VCM_mapping.yaml', 'w'))
