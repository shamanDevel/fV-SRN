"""
Mathematical expression parser.
Copied and modified from https://stackoverflow.com/a/9558001
"""

import ast
import operator as op


class MathParser:
    def __init__(self, max_value:float = None):
        """
        Initializes the math parser.
        :param max_value: if not None, a maximal value for intermediate results.
        """
        self.__operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
             ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
             ast.USub: op.neg}
        self.__max_value = max_value


    def __eval0(self, node):
        if isinstance(node, ast.Num): # <number>
            return node.n
        elif isinstance(node, ast.BinOp): # <left> <operator> <right>
            return self.__operators[type(node.op)](self.__eval(node.left), self.__eval(node.right))
        elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
            return self.__operators[type(node.op)](self.__eval(node.operand))
        else:
            raise TypeError(node)

    def __eval(self, node):
        ret = self.__eval0(node)
        if self.__max_value is not None:
            try:
                mag = abs(ret)
            except TypeError:
                pass  # not applicable
            else:
                if mag > self.__max_value:
                    raise ValueError(ret)
        return ret

    def eval_expr(self, expr:str):
        """
        >>> p = MathParser()
        >>> p.eval_expr('2^6')
        4
        >>> p.eval_expr('2**6')
        64
        >>> p.eval_expr('1 + 2*3**(4^5) / (6 + -7)')
        -5.0
        """
        return self.__eval(ast.parse(expr, mode='eval').body)

"""
Custom type for the argument parser that accepts a mathematical expression
and converts it to a float.
"""
def BigFloat(s:str):
    return MathParser().eval_expr(s)

"""
Custom type for the argument parser that accepts a mathematical expression
and converts it to a float.
"""
def BigInteger(s:str):
    return int(round(BigFloat(s)))