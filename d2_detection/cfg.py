#coding=utf-8


def parse_model_cfg(cfgfile):
    blocks = []
    with open(cfgfile, 'r') as fp:
        for line in fp.readlines():
            line = line.strip()
            if line == '' or line[0] == '#':
                continue
            if line[0] == '[':
                blocks.append({})
                blocks[-1]['type'] = line.lstrip('[').rstrip(']')
                if blocks[-1]['type'] == 'convolutional':
                    blocks[-1]['batch_normalize'] = 0
            else:
                key, value = line.split('=')
                key, value = key.strip(), value.strip()
                blocks[-1][key] = value
    return blocks

def parse_data_cfg(cfgfile):
    options = dict()
    with open(cfgfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()

    return options

