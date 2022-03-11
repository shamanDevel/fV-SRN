import sys
import os
import inspect

import common.utils as utils
import pyrenderer

_NAME_REPLACEMENT = {
    # Fixes for incorrect registrations in the docstring
    # Often caused by circular dependencies
    "at::Tensor": "torch.Tensor",
    "self: handle": "self",
    "pyrenderer.": "",
    "renderer::IModule": "Any",
    "renderer::IImageEvaluator": "IImageEvaluator",
    "renderer::Volume": "Volume",
    ": buffer": ": Any",
    "<ChannelMode.Color: 3>": "IImageEvaluator.ChannelMode.Color", # Enum as default value
}

def _replace_known_tags(line: str):
    for k,v in _NAME_REPLACEMENT.items():
        line = line.replace(k, v)
    return line

def _print_indented_docs(lines, prefix:str, include_stringtags, out):
    # remove empty lines at the start
    num_empty_lines = 0
    for i in range(len(lines)):
        if len(lines[i].strip())!=0:
            break
        num_empty_lines += 1
    lines = lines[num_empty_lines:]
    # remove empty lines at the end
    num_empty_lines = 0
    for i in range(len(lines)-1, 0, -1):
        if len(lines[i].strip()) != 0:
            break
        num_empty_lines += 1
    if num_empty_lines>0:
        lines = lines[:-num_empty_lines]
    if len(lines)==0: return
    # collect leading whitespace
    leading_spaces = len(lines[0]) - len(lines[0].lstrip())
    # write out
    if include_stringtags:
        print(prefix+'"""', file=out)
    for line in lines:
        print(prefix+line[leading_spaces:], file=out)
    if include_stringtags:
        print(prefix + '"""', file=out)

def print_docs(root, prefix='', out=sys.stdout):
    for name, obj in inspect.getmembers(root):
        if name.startswith('__') and not name.startswith("__init__"): continue
        if inspect.isclass(obj):
            # class name + documentation
            bases = [_replace_known_tags(base.__name__) for base in obj.__bases__]
            bases = filter(lambda x: x!='pybind11_object', bases)
            print(prefix+"class "+name+"("+",".join(bases)+'):', file=out)
            if obj.__doc__ is not None:
                _print_indented_docs(obj.__doc__.split('\n'), prefix+'    ', True, out)
                print(prefix, file=out)
            # member
            print_docs(obj, prefix+'    ', out)
            print("", file=out)
        elif inspect.ismodule(obj):
            print(prefix+"class "+name+": # NAMESPACE", file=out)
            if obj.__doc__ is not None:
                _print_indented_docs(obj.__doc__.split('\n'), prefix+'    ', True, out)
                print(prefix, file=out)
            # member
            print_docs(obj, prefix+'    ', out)
            print("", file=out)
        else:
            # method name + documentation
            if obj.__doc__ is None:
                print(prefix, "#", name, "= ??", file=out) # unknown variable
            else:
                # a method or function
                lines = obj.__doc__.split('\n')
                # Check if it is a function or method
                is_method = True
                if len(lines)==0: is_method = False
                elif len(lines[0])==0: is_method = False
                elif not lines[0].startswith(name): is_method = False
                if is_method:
                    signature = _replace_known_tags(lines[0])
                    if not "self" in signature and len(prefix)>0:
                        print(prefix + "@staticmethod", file=out)
                    print(prefix + "def", signature + ":", file=out)
                    _print_indented_docs(lines[1:], prefix + '    ', True, out)
                    print(prefix + "    ...", file=out)
                else:
                    is_enum = lines[0].startswith("Members:")
                    if is_enum:
                        print(prefix + name + " = ... # Enum", file=out)
                        #_print_indented_docs(lines[1:], prefix+"#", False, out)
                    else:
                        print(prefix + name + " = ... # Property", file=out)
                        _print_indented_docs(lines, prefix+"#", False, out)
                print("", file=out)

if __name__ == '__main__':
    outfile = os.path.abspath(os.path.join(__file__, "..", "..", "pyrenderer.pyi"))
    print("Save to", outfile)
    with open(outfile, "w") as f:
        print("import torch", file=f)
        print("import numpy", file=f)
        print("from typing import List, Any, Optional, Tuple, Dict", file=f)
        print("", file=f)
        print_docs(pyrenderer, out=f)
