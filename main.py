#!/usr/bin/env python3
"""
SimpleLang Compiler Implementation
A complete compiler for a simple C-like programming language
Author: Soutrik Mukherjee
Date: May 2025
"""

import json
import sys
from typing import List, Dict, Any, Optional, Tuple

class Token:
    """Represents a token in the source code"""
    def __init__(self, type: str, value: Any, line: int, column: int):
        self.type = type
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self):
        return f"Token({self.type}, {self.value}, {self.line}, {self.column})"

class Lexer:
    """Lexical analyzer for SimpleLang"""
    def __init__(self, source_code: str):
        self.source = source_code
        self.position = 0
        self.line = 1
        self.column = 1
        self.current_char = self.source[0] if self.source else None
        
        # Define keywords
        self.KEYWORDS = {
            'if': 'IF',
            'else': 'ELSE',
            'while': 'WHILE',
            'for': 'FOR',
            'return': 'RETURN',
            'int': 'INT_TYPE',
            'float': 'FLOAT_TYPE',
            'string': 'STRING_TYPE',
            'bool': 'BOOL_TYPE',
            'void': 'VOID_TYPE',
            'true': 'TRUE',
            'false': 'FALSE'
        }
    
    def error(self, message: str):
        raise Exception(f"Lexical error at line {self.line}, column {self.column}: {message}")
    
    def advance(self):
        """Move to the next character"""
        if self.position < len(self.source) - 1:
            self.position += 1
            self.column += 1
            self.current_char = self.source[self.position]
        else:
            self.current_char = None
        
        if self.current_char == '\n':
            self.line += 1
            self.column = 1
    
    def peek(self):
        """Look at the next character without advancing"""
        peek_pos = self.position + 1
        if peek_pos < len(self.source):
            return self.source[peek_pos]
        return None
    
    def skip_whitespace(self):
        """Skip whitespace characters"""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def skip_comment(self):
        """Skip single-line and multi-line comments"""
        if self.current_char == '/' and self.peek() == '/':
            # Single-line comment
            while self.current_char is not None and self.current_char != '\n':
                self.advance()
        elif self.current_char == '/' and self.peek() == '*':
            # Multi-line comment
            self.advance()  # Skip '/'
            self.advance()  # Skip '*'
            while self.current_char is not None:
                if self.current_char == '*' and self.peek() == '/':
                    self.advance()  # Skip '*'
                    self.advance()  # Skip '/'
                    break
                self.advance()
    
    def read_number(self):
        """Read a numeric literal"""
        num_str = ''
        has_dot = False
        
        while self.current_char is not None and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                if has_dot:
                    self.error("Invalid number format")
                has_dot = True
            num_str += self.current_char
            self.advance()
        
        if has_dot:
            return Token('FLOAT_LITERAL', float(num_str), self.line, self.column)
        else:
            return Token('INTEGER_LITERAL', int(num_str), self.line, self.column)
    
    def read_string(self):
        """Read a string literal"""
        string_val = ''
        self.advance()  # Skip opening quote
        
        while self.current_char is not None and self.current_char != '"':
            if self.current_char == '\\':
                self.advance()
                if self.current_char == 'n':
                    string_val += '\n'
                elif self.current_char == 't':
                    string_val += '\t'
                elif self.current_char == '\\':
                    string_val += '\\'
                elif self.current_char == '"':
                    string_val += '"'
                else:
                    self.error(f"Invalid escape sequence: \\{self.current_char}")
            else:
                string_val += self.current_char
            self.advance()
        
        if self.current_char != '"':
            self.error("Unterminated string literal")
        
        self.advance()  # Skip closing quote
        return Token('STRING_LITERAL', string_val, self.line, self.column)
    
    def read_identifier(self):
        """Read an identifier or keyword"""
        id_str = ''
        
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            id_str += self.current_char
            self.advance()
        
        token_type = self.KEYWORDS.get(id_str, 'IDENTIFIER')
        return Token(token_type, id_str, self.line, self.column)
    
    def get_next_token(self):
        """Get the next token from the source code"""
        while self.current_char is not None:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char == '/' and (self.peek() == '/' or self.peek() == '*'):
                self.skip_comment()
                continue
            
            if self.current_char.isdigit():
                return self.read_number()
            
            if self.current_char == '"':
                return self.read_string()
            
            if self.current_char.isalpha() or self.current_char == '_':
                return self.read_identifier()
            
            # Single-character tokens
            if self.current_char == '+':
                self.advance()
                return Token('PLUS', '+', self.line, self.column)
            
            if self.current_char == '-':
                self.advance()
                return Token('MINUS', '-', self.line, self.column)
            
            if self.current_char == '*':
                self.advance()
                return Token('MULTIPLY', '*', self.line, self.column)
            
            if self.current_char == '/':
                self.advance()
                return Token('DIVIDE', '/', self.line, self.column)
            
            if self.current_char == '%':
                self.advance()
                return Token('MODULO', '%', self.line, self.column)
            
            if self.current_char == '=':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token('EQUAL', '==', self.line, self.column)
                return Token('ASSIGN', '=', self.line, self.column)
            
            if self.current_char == '!':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token('NOT_EQUAL', '!=', self.line, self.column)
                return Token('NOT', '!', self.line, self.column)
            
            if self.current_char == '<':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token('LESS_EQUAL', '<=', self.line, self.column)
                return Token('LESS', '<', self.line, self.column)
            
            if self.current_char == '>':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token('GREATER_EQUAL', '>=', self.line, self.column)
                return Token('GREATER', '>', self.line, self.column)
            
            if self.current_char == '&':
                self.advance()
                if self.current_char == '&':
                    self.advance()
                    return Token('AND', '&&', self.line, self.column)
                self.error("Expected '&' after '&'")
            
            if self.current_char == '|':
                self.advance()
                if self.current_char == '|':
                    self.advance()
                    return Token('OR', '||', self.line, self.column)
                self.error("Expected '|' after '|'")
            
            if self.current_char == '(':
                self.advance()
                return Token('LPAREN', '(', self.line, self.column)
            
            if self.current_char == ')':
                self.advance()
                return Token('RPAREN', ')', self.line, self.column)
            
            if self.current_char == '{':
                self.advance()
                return Token('LBRACE', '{', self.line, self.column)
            
            if self.current_char == '}':
                self.advance()
                return Token('RBRACE', '}', self.line, self.column)
            
            if self.current_char == ';':
                self.advance()
                return Token('SEMICOLON', ';', self.line, self.column)
            
            if self.current_char == ',':
                self.advance()
                return Token('COMMA', ',', self.line, self.column)
            
            self.error(f"Unexpected character: '{self.current_char}'")
        
        return Token('EOF', None, self.line, self.column)
    
    def tokenize(self):
        """Tokenize the entire source code"""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == 'EOF':
                break
        return tokens


class Parser:
    """Syntax analyzer for SimpleLang"""
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.current_token = self.tokens[0] if self.tokens else None
        self.ast = None
    
    def error(self, message: str):
        if self.current_token:
            raise Exception(f"Parse error at line {self.current_token.line}, column {self.current_token.column}: {message}")
        else:
            raise Exception(f"Parse error: {message}")
    
    def advance(self):
        """Move to the next token"""
        if self.position < len(self.tokens) - 1:
            self.position += 1
            self.current_token = self.tokens[self.position]
    
    def expect(self, token_type: str):
        """Expect a specific token type and advance"""
        if self.current_token.type != token_type:
            self.error(f"Expected {token_type}, got {self.current_token.type}")
        self.advance()
    
    def parse(self):
        """Parse the input tokens into an AST"""
        self.ast = self.program()
        return self.ast
    
    def program(self):
        """Parse a program"""
        statements = []
        while self.current_token.type != 'EOF':
            stmt = self.statement()
            if stmt:
                statements.append(stmt)
        return {'type': 'program', 'statements': statements}
    
    def statement(self):
        """Parse a statement"""
        if self.current_token.type in ['INT_TYPE', 'FLOAT_TYPE', 'STRING_TYPE', 'BOOL_TYPE', 'VOID_TYPE']:
            # Could be a declaration or function declaration
            type_token = self.current_token
            self.advance()
            
            if self.current_token.type != 'IDENTIFIER':
                self.error("Expected identifier after type")
            
            identifier = self.current_token.value
            self.advance()
            
            if self.current_token.type == 'LPAREN':
                # Function declaration
                return self.function_declaration(type_token.value, identifier)
            else:
                # Variable declaration
                return self.declaration(type_token.value, identifier)
        
        elif self.current_token.type == 'IDENTIFIER':
            # Assignment or function call
            identifier = self.current_token.value
            self.advance()
            
            if self.current_token.type == 'ASSIGN':
                return self.assignment(identifier)
            elif self.current_token.type == 'LPAREN':
                return self.function_call_statement(identifier)
            else:
                self.error(f"Unexpected token after identifier: {self.current_token.type}")
        
        elif self.current_token.type == 'IF':
            return self.if_statement()
        
        elif self.current_token.type == 'WHILE':
            return self.while_statement()
        
        elif self.current_token.type == 'FOR':
            return self.for_statement()
        
        elif self.current_token.type == 'RETURN':
            return self.return_statement()
        
        elif self.current_token.type == 'LBRACE':
            return self.block()
        
        else:
            self.error(f"Unexpected token: {self.current_token.type}")
    
    def declaration(self, var_type: str, identifier: str):
        """Parse a variable declaration"""
        node = {'type': 'declaration', 'var_type': var_type, 'identifier': identifier}
        
        if self.current_token.type == 'ASSIGN':
            self.advance()
            node['initializer'] = self.expression()
        
        self.expect('SEMICOLON')
        return node
    
    def assignment(self, identifier: str):
        """Parse an assignment statement"""
        self.expect('ASSIGN')
        expression = self.expression()
        self.expect('SEMICOLON')
        return {'type': 'assignment', 'identifier': identifier, 'expression': expression}
    
    def if_statement(self):
        """Parse an if statement"""
        self.expect('IF')
        self.expect('LPAREN')
        condition = self.condition()
        self.expect('RPAREN')
        if_body = self.statement()
        
        else_body = None
        if self.current_token.type == 'ELSE':
            self.advance()
            else_body = self.statement()
        
        return {'type': 'if_statement', 'condition': condition, 'if_body': if_body, 'else_body': else_body}
    
    def while_statement(self):
        """Parse a while statement"""
        self.expect('WHILE')
        self.expect('LPAREN')
        condition = self.condition()
        self.expect('RPAREN')
        body = self.statement()
        
        return {'type': 'while_statement', 'condition': condition, 'body': body}
    
    def for_statement(self):
        """Parse a for statement"""
        self.expect('FOR')
        self.expect('LPAREN')
        
        # For init
        init = None
        if self.current_token.type != 'SEMICOLON':
            if self.current_token.type in ['INT_TYPE', 'FLOAT_TYPE', 'STRING_TYPE', 'BOOL_TYPE']:
                type_token = self.current_token
                self.advance()
                identifier = self.current_token.value
                self.advance()
                init = self.declaration(type_token.value, identifier)
                # Declaration already consumed semicolon
            elif self.current_token.type == 'IDENTIFIER':
                identifier = self.current_token.value
                self.advance()
                init = self.assignment(identifier)
                # Assignment already consumed semicolon
        else:
            self.advance()  # Skip semicolon
        
        # For condition
        condition = None
        if self.current_token.type != 'SEMICOLON':
            condition = self.condition()
        self.expect('SEMICOLON')
        
        # For update
        update = None
        if self.current_token.type != 'RPAREN':
            if self.current_token.type == 'IDENTIFIER':
                identifier = self.current_token.value
                self.advance()
                if self.current_token.type == 'ASSIGN':
                    self.advance()
                    expression = self.expression()
                    update = {'type': 'assignment', 'identifier': identifier, 'expression': expression}
        
        self.expect('RPAREN')
        body = self.statement()
        
        return {'type': 'for_statement', 'init': init, 'condition': condition, 'update': update, 'body': body}
    
    def function_declaration(self, return_type: str, identifier: str):
        """Parse a function declaration"""
        self.expect('LPAREN')
        parameters = []
        
        while self.current_token.type != 'RPAREN':
            if self.current_token.type in ['INT_TYPE', 'FLOAT_TYPE', 'STRING_TYPE', 'BOOL_TYPE']:
                param_type = self.current_token.value
                self.advance()
                
                if self.current_token.type != 'IDENTIFIER':
                    self.error("Expected parameter name")
                
                param_name = self.current_token.value
                self.advance()
                
                parameters.append({'type': param_type, 'name': param_name})
                
                if self.current_token.type == 'COMMA':
                    self.advance()
                elif self.current_token.type != 'RPAREN':
                    self.error("Expected ',' or ')' in parameter list")
            else:
                self.error("Expected parameter type")
        
        self.expect('RPAREN')
        body = self.block()
        
        return {
            'type': 'function_declaration',
            'return_type': return_type,
            'identifier': identifier,
            'parameters': parameters,
            'body': body
        }
    
    def function_call_statement(self, identifier: str):
        """Parse a function call statement"""
        call = self.function_call(identifier)
        self.expect('SEMICOLON')
        return {'type': 'function_call_statement', 'call': call}
    
    def function_call(self, identifier: str):
        """Parse a function call expression"""
        self.expect('LPAREN')
        arguments = []
        
        while self.current_token.type != 'RPAREN':
            arguments.append(self.expression())
            if self.current_token.type == 'COMMA':
                self.advance()
            elif self.current_token.type != 'RPAREN':
                self.error("Expected ',' or ')' in argument list")
        
        self.expect('RPAREN')
        return {'type': 'function_call_expression', 'function_name': identifier, 'arguments': arguments}
    
    def return_statement(self):
        """Parse a return statement"""
        self.expect('RETURN')
        expression = None
        
        if self.current_token.type != 'SEMICOLON':
            expression = self.expression()
        
        self.expect('SEMICOLON')
        return {'type': 'return_statement', 'expression': expression}
    
    def block(self):
        """Parse a block statement"""
        self.expect('LBRACE')
        statements = []
        
        while self.current_token.type != 'RBRACE':
            stmt = self.statement()
            if stmt:
                statements.append(stmt)
        
        self.expect('RBRACE')
        return {'type': 'block', 'statements': statements}
    
    def condition(self):
        """Parse a condition"""
        left = self.expression()
        
        if self.current_token.type in ['EQUAL', 'NOT_EQUAL', 'LESS', 'GREATER', 'LESS_EQUAL', 'GREATER_EQUAL']:
            operator = self.current_token.value
            self.advance()
            right = self.expression()
            left = {'type': 'binary_condition', 'left': left, 'operator': operator, 'right': right}
        
        while self.current_token.type in ['AND', 'OR']:
            operator = self.current_token.value
            self.advance()
            right = self.condition()
            left = {'type': 'logical_condition', 'left': left, 'operator': operator, 'right': right}
        
        return left
    
    def expression(self):
        """Parse an expression"""
        node = self.term()
        
        while self.current_token.type in ['PLUS', 'MINUS']:
            operator = self.current_token.value
            self.advance()
            right = self.term()
            node = {'type': 'binary_expression', 'left': node, 'operator': operator, 'right': right}
        
        return node
    
    def term(self):
        """Parse a term"""
        node = self.factor()
        
        while self.current_token.type in ['MULTIPLY', 'DIVIDE', 'MODULO']:
            operator = self.current_token.value
            self.advance()
            right = self.factor()
            node = {'type': 'binary_expression', 'left': node, 'operator': operator, 'right': right}
        
        return node
    
    def factor(self):
        """Parse a factor"""
        if self.current_token.type == 'INTEGER_LITERAL':
            value = self.current_token.value
            self.advance()
            return {'type': 'literal', 'value': value, 'literal_type': 'INTEGER_LITERAL'}
        
        elif self.current_token.type == 'FLOAT_LITERAL':
            value = self.current_token.value
            self.advance()
            return {'type': 'literal', 'value': value, 'literal_type': 'FLOAT_LITERAL'}
        
        elif self.current_token.type == 'STRING_LITERAL':
            value = self.current_token.value
            self.advance()
            return {'type': 'literal', 'value': value, 'literal_type': 'STRING_LITERAL'}
        
        elif self.current_token.type in ['TRUE', 'FALSE']:
            value = self.current_token.type == 'TRUE'
            self.advance()
            return {'type': 'literal', 'value': value, 'literal_type': 'BOOL_LITERAL'}
        
        elif self.current_token.type == 'IDENTIFIER':
            identifier = self.current_token.value
            self.advance()
            
            if self.current_token.type == 'LPAREN':
                return self.function_call(identifier)
            else:
                return {'type': 'variable', 'name': identifier}
        
        elif self.current_token.type == 'LPAREN':
            self.advance()
            node = self.expression()
            self.expect('RPAREN')
            return node
        
        elif self.current_token.type in ['MINUS', 'NOT']:
            operator = self.current_token.value
            self.advance()
            operand = self.factor()
            return {'type': 'unary_expression', 'operator': operator, 'operand': operand}
        
        else:
            self.error(f"Unexpected token in factor: {self.current_token.type}")


class Symbol:
    """Represents a symbol in the symbol table"""
    def __init__(self, name: str, symbol_type: str, data_type: str, scope_level: int):
        self.name = name
        self.symbol_type = symbol_type  # 'variable' or 'function'
        self.data_type = data_type
        self.scope_level = scope_level
        self.parameters = []  # For functions
        self.memory_address = None  # For variables


class SymbolTable:
    """Symbol table for tracking variables and functions"""
    def __init__(self, parent_scope=None):
        self.symbols = {}
        self.parent_scope = parent_scope
        self.scope_level = 0 if parent_scope is None else parent_scope.scope_level + 1
    
    def define(self, symbol: Symbol):
        """Add a symbol to the current scope"""
        if symbol.name in self.symbols:
            raise Exception(f"Symbol '{symbol.name}' already defined in current scope")
        self.symbols[symbol.name] = symbol
        return symbol
    
    def lookup(self, name: str, current_scope_only: bool = False):
        """Look up a symbol by name"""
        if name in self.symbols:
            return self.symbols[name]
        
        if not current_scope_only and self.parent_scope is not None:
            return self.parent_scope.lookup(name)
        
        return None


class SemanticAnalyzer:
    """Semantic analyzer for SimpleLang"""
    def __init__(self):
        self.current_scope = SymbolTable()  # Global scope
        self.errors = []
        self.current_function = None
    
    def error(self, message: str):
        self.errors.append(message)
    
    def analyze(self, ast):
        """Perform semantic analysis on the AST"""
        self.visit(ast)
        return self.errors
    
    def visit(self, node):
        """Visit a node in the AST"""
        if node is None:
            return None
        
        method_name = f"visit_{node['type']}"
        method = getattr(self, method_name, self.generic_visit)
        return method(node)
    
    def generic_visit(self, node):
        """Default visitor method"""
        self.error(f"No visitor method for node type: {node['type']}")
    
    def visit_program(self, node):
        """Visit a program node"""
        for statement in node['statements']:
            self.visit(statement)
    
    def visit_declaration(self, node):
        """Visit a declaration node"""
        var_name = node['identifier']
        var_type = node['var_type']
        
        # Check if variable already exists in current scope
        if self.current_scope.lookup(var_name, current_scope_only=True):
            self.error(f"Variable '{var_name}' already declared in current scope")
            return
        
        # Create symbol
        symbol = Symbol(var_name, 'variable', var_type, self.current_scope.scope_level)
        self.current_scope.define(symbol)
        
        # Check initializer if present
        if 'initializer' in node and node['initializer']:
            init_type = self.get_expression_type(node['initializer'])
            if not self.is_type_compatible(var_type, init_type):
                self.error(f"Type mismatch: cannot assign {init_type} to {var_type}")
    
    def visit_assignment(self, node):
        """Visit an assignment node"""
        var_name = node['identifier']
        
        # Look up variable
        symbol = self.current_scope.lookup(var_name)
        if not symbol:
            self.error(f"Undefined variable: '{var_name}'")
            return
        
        # Check expression type
        expr_type = self.get_expression_type(node['expression'])
        if not self.is_type_compatible(symbol.data_type, expr_type):
            self.error(f"Type mismatch: cannot assign {expr_type} to {symbol.data_type}")
    
    def visit_if_statement(self, node):
        """Visit an if statement node"""
        # Check condition is boolean
        cond_type = self.get_expression_type(node['condition'])
        if cond_type != 'bool':
            self.error(f"Condition must be boolean, got {cond_type}")
        
        # Visit bodies
        self.visit(node['if_body'])
        if node['else_body']:
            self.visit(node['else_body'])
    
    def visit_while_statement(self, node):
        """Visit a while statement node"""
        # Check condition is boolean
        cond_type = self.get_expression_type(node['condition'])
        if cond_type != 'bool':
            self.error(f"Condition must be boolean, got {cond_type}")
        
        # Visit body
        self.visit(node['body'])
    
    def visit_for_statement(self, node):
        """Visit a for statement node"""
        # Create new scope for loop
        self.current_scope = SymbolTable(self.current_scope)
        
        # Visit init
        if node['init']:
            self.visit(node['init'])
        
        # Check condition
        if node['condition']:
            cond_type = self.get_expression_type(node['condition'])
            if cond_type != 'bool':
                self.error(f"Condition must be boolean, got {cond_type}")
        
        # Visit update
        if node['update']:
            self.visit(node['update'])
        
        # Visit body
        self.visit(node['body'])
        
        # Restore scope
        self.current_scope = self.current_scope.parent_scope
    
    def visit_function_declaration(self, node):
        """Visit a function declaration node"""
        func_name = node['identifier']
        return_type = node['return_type']
        
        # Check if function already exists
        if self.current_scope.lookup(func_name, current_scope_only=True):
            self.error(f"Function '{func_name}' already declared")
            return
        
        # Create function symbol
        symbol = Symbol(func_name, 'function', return_type, self.current_scope.scope_level)
        symbol.parameters = node['parameters']
        self.current_scope.define(symbol)
        
        # Create new scope for function body
        self.current_scope = SymbolTable(self.current_scope)
        self.current_function = symbol
        
        # Add parameters to scope
        for param in node['parameters']:
            param_symbol = Symbol(param['name'], 'variable', param['type'], self.current_scope.scope_level)
            self.current_scope.define(param_symbol)
        
        # Visit body
        self.visit(node['body'])
        
        # Restore scope
        self.current_scope = self.current_scope.parent_scope
        self.current_function = None
    
    def visit_function_call_statement(self, node):
        """Visit a function call statement node"""
        self.visit(node['call'])
    
    def visit_return_statement(self, node):
        """Visit a return statement node"""
        if not self.current_function:
            self.error("Return statement outside function")
            return
        
        if node['expression']:
            expr_type = self.get_expression_type(node['expression'])
            if not self.is_type_compatible(self.current_function.data_type, expr_type):
                self.error(f"Return type mismatch: expected {self.current_function.data_type}, got {expr_type}")
        else:
            if self.current_function.data_type != 'void':
                self.error(f"Function '{self.current_function.name}' must return a value")
    
    def visit_block(self, node):
        """Visit a block node"""
        # Create new scope
        self.current_scope = SymbolTable(self.current_scope)
        
        # Visit statements
        for statement in node['statements']:
            self.visit(statement)
        
        # Restore scope
        self.current_scope = self.current_scope.parent_scope
    
    def get_expression_type(self, node):
        """Get the type of an expression"""
        if node['type'] == 'literal':
            if node['literal_type'] == 'INTEGER_LITERAL':
                return 'int'
            elif node['literal_type'] == 'FLOAT_LITERAL':
                return 'float'
            elif node['literal_type'] == 'STRING_LITERAL':
                return 'string'
            elif node['literal_type'] == 'BOOL_LITERAL':
                return 'bool'
        
        elif node['type'] == 'variable':
            symbol = self.current_scope.lookup(node['name'])
            if not symbol:
                self.error(f"Undefined variable: '{node['name']}'")
                return 'error'
            return symbol.data_type
        
        elif node['type'] == 'binary_expression':
            left_type = self.get_expression_type(node['left'])
            right_type = self.get_expression_type(node['right'])
            
            if node['operator'] in ['+', '-', '*', '/', '%']:
                if left_type in ['int', 'float'] and right_type in ['int', 'float']:
                    return 'float' if 'float' in [left_type, right_type] else 'int'
                else:
                    self.error(f"Invalid operands for {node['operator']}: {left_type} and {right_type}")
                    return 'error'
        
        elif node['type'] == 'binary_condition':
            left_type = self.get_expression_type(node['left'])
            right_type = self.get_expression_type(node['right'])
            
            if not self.is_type_compatible(left_type, right_type) and not self.is_type_compatible(right_type, left_type):
                self.error(f"Cannot compare {left_type} with {right_type}")
            
            return 'bool'
        
        elif node['type'] == 'logical_condition':
            return 'bool'
        
        elif node['type'] == 'unary_expression':
            operand_type = self.get_expression_type(node['operand'])
            
            if node['operator'] == '-':
                if operand_type in ['int', 'float']:
                    return operand_type
                else:
                    self.error(f"Cannot apply unary minus to {operand_type}")
                    return 'error'
            elif node['operator'] == '!':
                if operand_type == 'bool':
                    return 'bool'
                else:
                    self.error(f"Cannot apply logical NOT to {operand_type}")
                    return 'error'
        
        elif node['type'] == 'function_call_expression':
            func_symbol = self.current_scope.lookup(node['function_name'])
            if not func_symbol:
                self.error(f"Undefined function: '{node['function_name']}'")
                return 'error'
            
            if func_symbol.symbol_type != 'function':
                self.error(f"'{node['function_name']}' is not a function")
                return 'error'
            
            # Check argument count
            if len(node['arguments']) != len(func_symbol.parameters):
                self.error(f"Function '{node['function_name']}' expects {len(func_symbol.parameters)} arguments, got {len(node['arguments'])}")
            else:
                # Check argument types
                for i, (arg, param) in enumerate(zip(node['arguments'], func_symbol.parameters)):
                    arg_type = self.get_expression_type(arg)
                    if not self.is_type_compatible(param['type'], arg_type):
                        self.error(f"Argument {i+1} type mismatch: expected {param['type']}, got {arg_type}")
            
            return func_symbol.data_type
        
        return 'error'
    
    def is_type_compatible(self, target_type, source_type):
        """Check if source_type is compatible with target_type"""
        if target_type == source_type:
            return True
        
        # Allow int to float conversion
        if target_type == 'float' and source_type == 'int':
            return True
        
        return False


class CodeGenerator:
    """Code generator for SimpleLang"""
    def __init__(self):
        self.instructions = []
        self.symbol_table = {}  # Maps variable names to memory addresses
        self.current_memory_address = 0
        self.function_table = {}  # Maps function names to instruction addresses
        self.label_counter = 0
        self.current_scope_offset = 0
        self.scope_stack = []
    
    def generate(self, ast):
        """Generate code from the AST"""
        self.visit(ast)
        return self.instructions, self.function_table
    
    def emit(self, opcode, operand=None):
        """Emit an instruction"""
        if operand is not None:
            self.instructions.append(f"{opcode} {operand}")
        else:
            self.instructions.append(opcode)
    
    def generate_label(self, prefix):
        """Generate a unique label"""
        label = f"{prefix}_{self.label_counter}"
        self.label_counter += 1
        return label
    
    def get_memory_address(self, var_name):
        """Get or allocate memory address for a variable"""
        if var_name not in self.symbol_table:
            self.symbol_table[var_name] = self.current_memory_address
            self.current_memory_address += 1
        return self.symbol_table[var_name]
    
    def visit(self, node):
        """Visit a node in the AST"""
        if node is None:
            return
        
        method_name = f"visit_{node['type']}"
        method = getattr(self, method_name, self.generic_visit)
        return method(node)
    
    def generic_visit(self, node):
        """Default visitor method"""
        raise Exception(f"No code generator for node type: {node['type']}")
    
    def visit_program(self, node):
        """Visit a program node"""
        # Generate code for all statements
        for statement in node['statements']:
            self.visit(statement)
        
        # Add HALT instruction at the end
        self.emit("HALT")
    
    def visit_declaration(self, node):
        """Visit a declaration node"""
        var_name = node['identifier']
        addr = self.get_memory_address(var_name)
        
        if 'initializer' in node and node['initializer']:
            self.visit(node['initializer'])
            self.emit("STORE", addr)
    
    def visit_assignment(self, node):
        """Visit an assignment node"""
        var_name = node['identifier']
        addr = self.get_memory_address(var_name)
        
        self.visit(node['expression'])
        self.emit("STORE", addr)
    
    def visit_if_statement(self, node):
        """Visit an if statement node"""
        else_label = self.generate_label("else")
        end_label = self.generate_label("endif")
        
        # Generate code for condition
        self.visit(node['condition'])
        
        # Jump to else if condition is false
        self.emit("JZ", else_label)
        
        # Generate code for if body
        self.visit(node['if_body'])
        
        # Jump to end
        self.emit("JMP", end_label)
        
        # Else label
        self.emit("LABEL", else_label)
        
        # Generate code for else body if present
        if node['else_body']:
            self.visit(node['else_body'])
        
        # End label
        self.emit("LABEL", end_label)
    
    def visit_while_statement(self, node):
        """Visit a while statement node"""
        start_label = self.generate_label("while_start")
        end_label = self.generate_label("while_end")
        
        # Start label
        self.emit("LABEL", start_label)
        
        # Generate code for condition
        self.visit(node['condition'])
        
        # Jump to end if condition is false
        self.emit("JZ", end_label)
        
        # Generate code for body
        self.visit(node['body'])
        
        # Jump back to start
        self.emit("JMP", start_label)
        
        # End label
        self.emit("LABEL", end_label)
    
    def visit_for_statement(self, node):
        """Visit a for statement node"""
        # Push scope for loop variables
        self.scope_stack.append(self.current_memory_address)
        
        # Generate code for init
        if node['init']:
            self.visit(node['init'])
        
        start_label = self.generate_label("for_start")
        end_label = self.generate_label("for_end")
        
        # Start label
        self.emit("LABEL", start_label)
        
        # Generate code for condition
        if node['condition']:
            self.visit(node['condition'])
            self.emit("JZ", end_label)
        
        # Generate code for body
        self.visit(node['body'])
        
        # Generate code for update
        if node['update']:
            self.visit(node['update'])
        
        # Jump back to start
        self.emit("JMP", start_label)
        
        # End label
        self.emit("LABEL", end_label)
        
        # Pop scope
        if self.scope_stack:
            self.current_memory_address = self.scope_stack.pop()
    
    def visit_function_declaration(self, node):
        """Visit a function declaration node"""
        func_name = node['identifier']
        
        # Skip to end of function
        skip_label = self.generate_label("skip_func")
        self.emit("JMP", skip_label)
        
        # Function label
        func_label = func_name
        self.emit("LABEL", func_label)
        
        # Record function address
        self.function_table[func_name] = len(self.instructions) - 1
        
        # Save old symbol table and create new one for function
        old_symbol_table = self.symbol_table.copy()
        old_memory_address = self.current_memory_address
        self.symbol_table = {}
        self.current_memory_address = 0
        
        # Allocate memory for parameters
        for i, param in enumerate(node['parameters']):
            self.get_memory_address(param['name'])
        
        # Generate code for body
        self.visit(node['body'])
        
        # Add default return for void functions
        if node['return_type'] == 'void':
            self.emit("PUSH", 0)
            self.emit("RET")
        
        # Skip label
        self.emit("LABEL", skip_label)
        
        # Restore symbol table
        self.symbol_table = old_symbol_table
        self.current_memory_address = old_memory_address
    
    def visit_function_call_statement(self, node):
        """Visit a function call statement node"""
        self.visit(node['call'])
        # Pop the return value since it's not used
        self.emit("POP")
    
    def visit_function_call_expression(self, node):
        """Visit a function call expression node"""
        # Push arguments in reverse order
        for arg in reversed(node['arguments']):
            self.visit(arg)
        
        # Call function
        self.emit("CALL", node['function_name'])
    
    def visit_return_statement(self, node):
        """Visit a return statement node"""
        if node['expression']:
            self.visit(node['expression'])
        else:
            self.emit("PUSH", 0)  # Default return value
        
        self.emit("RET")
    
    def visit_block(self, node):
        """Visit a block node"""
        # Push scope
        self.scope_stack.append(self.current_memory_address)
        
        # Generate code for statements
        for statement in node['statements']:
            self.visit(statement)
        
        # Pop scope
        if self.scope_stack:
            self.current_memory_address = self.scope_stack.pop()
    
    def visit_literal(self, node):
        """Visit a literal node"""
        self.emit("PUSH", node['value'])
    
    def visit_variable(self, node):
        """Visit a variable node"""
        addr = self.get_memory_address(node['name'])
        self.emit("LOAD", addr)
    
    def visit_binary_expression(self, node):
        """Visit a binary expression node"""
        self.visit(node['left'])
        self.visit(node['right'])
        
        if node['operator'] == '+':
            self.emit("ADD")
        elif node['operator'] == '-':
            self.emit("SUB")
        elif node['operator'] == '*':
            self.emit("MUL")
        elif node['operator'] == '/':
            self.emit("DIV")
        elif node['operator'] == '%':
            self.emit("MOD")
    
    def visit_binary_condition(self, node):
        """Visit a binary condition node"""
        self.visit(node['left'])
        self.visit(node['right'])
        
        if node['operator'] == '==':
            self.emit("EQ")
        elif node['operator'] == '!=':
            self.emit("NEQ")
        elif node['operator'] == '<':
            self.emit("LT")
        elif node['operator'] == '>':
            self.emit("GT")
        elif node['operator'] == '<=':
            self.emit("LE")
        elif node['operator'] == '>=':
            self.emit("GE")
    
    def visit_logical_condition(self, node):
        """Visit a logical condition node"""
        if node['operator'] == '&&':
            # Short-circuit AND
            false_label = self.generate_label("and_false")
            end_label = self.generate_label("and_end")
            
            self.visit(node['left'])
            self.emit("DUP")
            self.emit("JZ", false_label)
            self.emit("POP")
            self.visit(node['right'])
            self.emit("JMP", end_label)
            
            self.emit("LABEL", false_label)
            self.emit("POP")
            self.emit("PUSH", 0)
            
            self.emit("LABEL", end_label)
        
        elif node['operator'] == '||':
            # Short-circuit OR
            true_label = self.generate_label("or_true")
            end_label = self.generate_label("or_end")
            
            self.visit(node['left'])
            self.emit("DUP")
            self.emit("JNZ", true_label)
            self.emit("POP")
            self.visit(node['right'])
            self.emit("JMP", end_label)
            
            self.emit("LABEL", true_label)
            self.emit("POP")
            self.emit("PUSH", 1)
            
            self.emit("LABEL", end_label)
    
    def visit_unary_expression(self, node):
        """Visit a unary expression node"""
        self.visit(node['operand'])
        
        if node['operator'] == '-':
            self.emit("NEG")
        elif node['operator'] == '!':
            self.emit("NOT")


class VirtualMachine:
    """Virtual machine for executing SimpleLang bytecode"""
    def __init__(self, instructions, function_table):
        self.instructions = instructions
        self.function_table = function_table
        self.pc = 0  # Program counter
        self.stack = []  # Operand stack
        self.memory = {}  # Variable memory
        self.call_stack = []  # Call stack for function returns
        self.labels = {}  # Label to instruction address mapping
        
        # Preprocess labels
        self.preprocess_labels()
    
    def preprocess_labels(self):
        """Build label to address mapping"""
        for i, instruction in enumerate(self.instructions):
            parts = instruction.split()
            if parts[0] == "LABEL":
                self.labels[parts[1]] = i
    
    def push(self, value):
        """Push a value onto the stack"""
        self.stack.append(value)
    
    def pop(self):
        """Pop a value from the stack"""
        if not self.stack:
            raise Exception("Stack underflow")
        return self.stack.pop()
    
    def execute(self):
        """Execute the bytecode"""
        while self.pc < len(self.instructions):
            instruction = self.instructions[self.pc]
            parts = instruction.split(None, 1)
            opcode = parts[0]
            operand = parts[1] if len(parts) > 1 else None
            
            if opcode == "PUSH":
                value = self.parse_value(operand)
                self.push(value)
            
            elif opcode == "POP":
                self.pop()
            
            elif opcode == "DUP":
                if self.stack:
                    self.push(self.stack[-1])
            
            elif opcode == "LOAD":
                addr = int(operand)
                value = self.memory.get(addr, 0)
                self.push(value)
            
            elif opcode == "STORE":
                addr = int(operand)
                value = self.pop()
                self.memory[addr] = value
            
            elif opcode == "ADD":
                right = self.pop()
                left = self.pop()
                self.push(left + right)
            
            elif opcode == "SUB":
                right = self.pop()
                left = self.pop()
                self.push(left - right)
            
            elif opcode == "MUL":
                right = self.pop()
                left = self.pop()
                self.push(left * right)
            
            elif opcode == "DIV":
                right = self.pop()
                left = self.pop()
                if right == 0:
                    raise Exception("Division by zero")
                self.push(left / right if isinstance(left, float) or isinstance(right, float) else left // right)
            
            elif opcode == "MOD":
                right = self.pop()
                left = self.pop()
                if right == 0:
                    raise Exception("Division by zero")
                self.push(left % right)
            
            elif opcode == "NEG":
                value = self.pop()
                self.push(-value)
            
            elif opcode == "EQ":
                right = self.pop()
                left = self.pop()
                self.push(1 if left == right else 0)
            
            elif opcode == "NEQ":
                right = self.pop()
                left = self.pop()
                self.push(1 if left != right else 0)
            
            elif opcode == "LT":
                right = self.pop()
                left = self.pop()
                self.push(1 if left < right else 0)
            
            elif opcode == "GT":
                right = self.pop()
                left = self.pop()
                self.push(1 if left > right else 0)
            
            elif opcode == "LE":
                right = self.pop()
                left = self.pop()
                self.push(1 if left <= right else 0)
            
            elif opcode == "GE":
                right = self.pop()
                left = self.pop()
                self.push(1 if left >= right else 0)
            
            elif opcode == "NOT":
                value = self.pop()
                self.push(1 if value == 0 else 0)
            
            elif opcode == "JMP":
                self.pc = self.labels[operand]
                continue
            
            elif opcode == "JZ":
                condition = self.pop()
                if condition == 0:
                    self.pc = self.labels[operand]
                    continue
            
            elif opcode == "JNZ":
                condition = self.pop()
                if condition != 0:
                    self.pc = self.labels[operand]
                    continue
            
            elif opcode == "CALL":
                # Save return address
                self.call_stack.append(self.pc + 1)
                # Jump to function
                self.pc = self.labels[operand]
                continue
            
            elif opcode == "RET":
                # Get return value
                return_value = self.pop() if self.stack else 0
                # Restore program counter
                if self.call_stack:
                    self.pc = self.call_stack.pop()
                    # Push return value
                    self.push(return_value)
                    continue
                else:
                    # No more calls, end execution
                    return return_value
            
            elif opcode == "LABEL":
                # Labels are no-ops during execution
                pass
            
            elif opcode == "HALT":
                break
            
            else:
                raise Exception(f"Unknown opcode: {opcode}")
            
            self.pc += 1
        
        # Return the top of stack if any
        return self.stack[-1] if self.stack else 0
    
    def parse_value(self, operand):
        """Parse an operand value"""
        try:
            # Try integer
            return int(operand)
        except ValueError:
            try:
                # Try float
                return float(operand)
            except ValueError:
                # Return as string
                return operand


class SimpleLangCompiler:
    """Main compiler class that orchestrates the compilation process"""
    def __init__(self, source_code):
        self.source_code = source_code
        self.tokens = None
        self.ast = None
        self.errors = []
        self.bytecode = None
        self.function_table = None
    
    def compile(self):
        """Compile the source code"""
        try:
            # Lexical analysis
            print("=== Lexical Analysis ===")
            lexer = Lexer(self.source_code)
            self.tokens = lexer.tokenize()
            print(f"Generated {len(self.tokens)} tokens")
            
            # Syntax analysis
            print("\n=== Syntax Analysis ===")
            parser = Parser(self.tokens)
            self.ast = parser.parse()
            print("AST generated successfully")
            
            # Semantic analysis
            print("\n=== Semantic Analysis ===")
            analyzer = SemanticAnalyzer()
            self.errors = analyzer.analyze(self.ast)
            
            if self.errors:
                print("Semantic errors found:")
                for error in self.errors:
                    print(f"  - {error}")
                return None
            else:
                print("No semantic errors found")
            
            # Code generation
            print("\n=== Code Generation ===")
            generator = CodeGenerator()
            self.bytecode, self.function_table = generator.generate(self.ast)
            print(f"Generated {len(self.bytecode)} instructions")
            
            return self.ast
            
        except Exception as e:
            print(f"Compilation error: {e}")
            return None
    
    def execute(self):
        """Execute the compiled bytecode"""
        if not self.bytecode:
            print("No bytecode to execute")
            return None
        
        print("\n=== Execution ===")
        vm = VirtualMachine(self.bytecode, self.function_table)
        result = vm.execute()
        return result
    
    def print_tokens(self):
        """Print the token list"""
        if self.tokens:
            print("\n=== Token List ===")
            for token in self.tokens:
                print(token)
    
    def print_ast(self):
        """Print the AST"""
        if self.ast:
            print("\n=== Abstract Syntax Tree ===")
            print(json.dumps(self.ast, indent=2))
    
    def print_bytecode(self):
        """Print the generated bytecode"""
        if self.bytecode:
            print("\n=== Bytecode ===")
            for i, instruction in enumerate(self.bytecode):
                print(f"{i:4d}: {instruction}")


def main():
    """Main function for testing the compiler"""
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r') as f:
            source_code = f.read()
    else:
        # Default test program
        source_code = """
        int factorial(int n) {
            if (n <= 1) {
                return 1;
            } else {
                return n * factorial(n - 1);
            }
        }
        
        int main() {
            int num = 5;
            int result = factorial(num);
            return result;
        }
        """
    
    # Compile and execute
    compiler = SimpleLangCompiler(source_code)
    
    if compiler.compile():
        compiler.print_bytecode()
        result = compiler.execute()
        print(f"\nProgram returned: {result}")
    else:
        print("\nCompilation failed")


if __name__ == "__main__":
    main()